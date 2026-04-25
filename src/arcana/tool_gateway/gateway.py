"""ToolGateway - central orchestrator for tool execution."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from arcana.contracts.tool import (
    ErrorType,
    SideEffect,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)
from arcana.contracts.trace import (
    EventType,
    ToolCallRecord,
    TraceEvent,
)
from arcana.tool_gateway.base import ToolExecutionError, ToolProvider
from arcana.tool_gateway.execution_backend import ExecutionBackend, InProcessBackend
from arcana.tool_gateway.validators import validate_arguments
from arcana.utils.hashing import canonical_hash

if TYPE_CHECKING:
    from arcana.contracts.trace import TraceContext
    from arcana.tool_gateway.registry import ToolRegistry
    from arcana.trace.writer import TraceWriter


class AuthorizationError(Exception):
    """Raised when a tool call is not authorized."""

    def __init__(self, tool_name: str, missing_capabilities: list[str]) -> None:
        self.tool_name = tool_name
        self.missing_capabilities = missing_capabilities
        super().__init__(
            f"Tool '{tool_name}' requires capabilities "
            f"{missing_capabilities} which are not granted"
        )


class ConfirmationRequired(Exception):
    """Raised when a tool call requires human confirmation."""

    def __init__(self, tool_call: ToolCall, spec: ToolSpec) -> None:
        self.tool_call = tool_call
        self.spec = spec
        super().__init__(
            f"Tool '{tool_call.name}' (side_effect={spec.side_effect.value}) "
            f"requires confirmation before execution"
        )


class ToolGateway:
    """
    Central orchestrator for tool execution.

    Pipeline: Resolve → Authorize → Validate → Idempotency → Confirm → Execute+Retry → Audit

    Responsibilities:
    1. Authorization - checks agent capabilities against tool requirements
    2. Validation - validates arguments against input_schema
    3. Idempotency - caches results for duplicate idempotency_keys
    4. Confirmation - gates write tools behind confirmation callback
    5. Retry - retries retryable errors with exponential backoff
    6. Audit - logs every call to trace as ToolCallRecord
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        trace_writer: TraceWriter | None = None,
        granted_capabilities: set[str] | None = None,
        confirmation_callback: Callable[[ToolCall, ToolSpec], Awaitable[bool]] | None = None,
        backend: ExecutionBackend | None = None,
        idempotency_cache_limit: int | None = 1024,
    ) -> None:
        """
        Initialize the ToolGateway.

        Args:
            registry: Tool registry with registered providers
            trace_writer: Optional trace writer for audit logging
            granted_capabilities: Set of capabilities granted to this agent
            confirmation_callback: Async callback for write confirmation.
                If None and a write tool is called, ConfirmationRequired is raised.
            backend: Execution backend for tool isolation.
                Defaults to InProcessBackend (current behavior, zero overhead).
            idempotency_cache_limit: Maximum number of ``ToolResult`` entries
                retained in the idempotency cache. Defaults to ``1024``.
                ``None`` keeps unbounded retention -- matches pre-v0.8.2
                behaviour and is an explicit opt-in for callers that
                genuinely need it. ``int >= 0`` caps the cache at that size
                via LRU eviction; ``0`` disables dedup entirely (every
                insert is immediately evicted). Negative values raise
                ``ValueError``.
        """
        if idempotency_cache_limit is not None and idempotency_cache_limit < 0:
            raise ValueError(
                f"idempotency_cache_limit must be None or >= 0, got {idempotency_cache_limit}"
            )

        self.registry = registry
        self.trace_writer = trace_writer
        self.granted_capabilities = granted_capabilities or set()
        self.confirmation_callback = confirmation_callback
        self.backend = backend or InProcessBackend()

        self._idempotency_cache_limit = idempotency_cache_limit
        self._idempotency_cache: OrderedDict[str, ToolResult] = OrderedDict()
        self._cache_lock = asyncio.Lock()

    async def close(self) -> None:
        """Release backend resources."""
        await self.backend.cleanup()
        async with self._cache_lock:
            self._idempotency_cache.clear()

    async def call(
        self,
        tool_call: ToolCall,
        *,
        trace_ctx: TraceContext | None = None,
    ) -> ToolResult:
        """
        Execute a tool call through the full gateway pipeline.

        Args:
            tool_call: The tool call to execute
            trace_ctx: Optional trace context for audit logging

        Returns:
            ToolResult with output or error
        """
        # 1. Resolve provider
        provider = self.registry.get(tool_call.name)
        if provider is None:
            available = self.registry.list_tools()
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                error=ToolError(
                    error_type=ErrorType.NON_RETRYABLE,
                    message=(
                        f"Tool '{tool_call.name}' not found in registry. "
                        f"Available tools: {available}. "
                        f"Check the tool name for typos or register it with Runtime(tools=[...])."
                    ),
                    code="TOOL_NOT_FOUND",
                ),
            )

        spec = provider.spec

        # 2. Authorize
        missing = self._check_capabilities(spec)
        if missing:
            self._log_authorization_failure(tool_call, missing, trace_ctx)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                error=ToolError(
                    error_type=ErrorType.NON_RETRYABLE,
                    message=f"Missing capabilities: {', '.join(missing)}",
                    code="UNAUTHORIZED",
                ),
            )

        # 3. Validate arguments
        validation_error = validate_arguments(spec, tool_call.arguments)
        if validation_error is not None:
            result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                error=validation_error,
            )
            if trace_ctx:
                self._log_tool_call(tool_call, result, spec, trace_ctx)
            return result

        # 4. Check idempotency cache (lock held through execute to prevent duplicate runs)
        if tool_call.idempotency_key is not None:
            async with self._cache_lock:
                cached = self._idempotency_cache.get(tool_call.idempotency_key)
                if cached is not None:
                    self._idempotency_cache.move_to_end(tool_call.idempotency_key)
                    return cached

                # 5. Confirm (write tools)
                confirmation_result = await self._confirm_execution(tool_call, spec)
                if confirmation_result is not None:
                    return confirmation_result

                # 6. Execute with retry
                result = await self._execute_with_retry(provider, tool_call, spec)

                # 7. Cache result
                self._idempotency_cache[tool_call.idempotency_key] = result
                if self._idempotency_cache_limit is not None:
                    while len(self._idempotency_cache) > self._idempotency_cache_limit:
                        self._idempotency_cache.popitem(last=False)
        else:
            # No idempotency key — skip cache entirely
            confirmation_result = await self._confirm_execution(tool_call, spec)
            if confirmation_result is not None:
                return confirmation_result
            result = await self._execute_with_retry(provider, tool_call, spec)

        # 8. Audit
        if trace_ctx:
            self._log_tool_call(tool_call, result, spec, trace_ctx)

        return result

    async def call_many_concurrent(
        self,
        tool_calls: list[ToolCall],
        *,
        trace_ctx: TraceContext | None = None,
    ) -> list[ToolResult]:
        """
        Execute ALL tool calls concurrently via asyncio.gather.

        Every call runs in parallel regardless of side effect type.
        Individual failures are caught gracefully — one failing tool
        does not block or cancel the others.  Result order matches
        the input ``tool_calls`` order.
        """
        if not tool_calls:
            return []

        tasks = [self.call(tc, trace_ctx=trace_ctx) for tc in tool_calls]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[ToolResult] = []
        for tc, raw in zip(tool_calls, raw_results, strict=True):
            if isinstance(raw, BaseException):
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        success=False,
                        error=ToolError(
                            error_type=ErrorType.NON_RETRYABLE,
                            message=f"Unexpected error executing '{tc.name}': {raw}",
                            code="GATHER_EXCEPTION",
                        ),
                    )
                )
            else:
                results.append(raw)

        return results

    async def call_many(
        self,
        tool_calls: list[ToolCall],
        *,
        trace_ctx: TraceContext | None = None,
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls.

        Read-only tools run concurrently; write tools run sequentially.
        """
        read_calls: list[tuple[int, ToolCall]] = []
        write_calls: list[tuple[int, ToolCall]] = []

        for i, tc in enumerate(tool_calls):
            provider = self.registry.get(tc.name)
            if provider and provider.spec.side_effect == SideEffect.WRITE:
                write_calls.append((i, tc))
            else:
                read_calls.append((i, tc))

        results: list[tuple[int, ToolResult]] = []

        # Run read tools concurrently
        if read_calls:
            read_tasks = [
                self.call(tc, trace_ctx=trace_ctx) for _, tc in read_calls
            ]
            read_results = await asyncio.gather(*read_tasks)
            for (idx, _), result in zip(read_calls, read_results, strict=True):
                results.append((idx, result))

        # Run write tools sequentially
        for idx, tc in write_calls:
            result = await self.call(tc, trace_ctx=trace_ctx)
            results.append((idx, result))

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    def _check_capabilities(self, spec: ToolSpec) -> list[str]:
        """Return list of missing capabilities, empty if authorized."""
        required = set(spec.capabilities)
        missing = required - self.granted_capabilities
        return sorted(missing)

    async def _check_idempotency(self, key: str | None) -> ToolResult | None:
        """Return cached result if idempotency key was seen before."""
        if key is None:
            return None
        async with self._cache_lock:
            return self._idempotency_cache.get(key)

    async def _cache_result(self, key: str | None, result: ToolResult) -> None:
        """Store result in idempotency cache."""
        if key is None:
            return
        async with self._cache_lock:
            self._idempotency_cache[key] = result

    async def _confirm_execution(
        self, tool_call: ToolCall, spec: ToolSpec
    ) -> ToolResult | None:
        """
        Gate execution behind confirmation for write tools.

        Returns None if confirmed (proceed with execution).
        Returns ToolResult if rejected or confirmation required.
        """
        if spec.side_effect != SideEffect.WRITE and not spec.requires_confirmation:
            return None

        if self.confirmation_callback is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                error=ToolError(
                    error_type=ErrorType.REQUIRES_HUMAN,
                    message=f"Tool '{tool_call.name}' requires human confirmation",
                    code="CONFIRMATION_REQUIRED",
                ),
            )

        confirmed = await self.confirmation_callback(tool_call, spec)
        if not confirmed:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                error=ToolError(
                    error_type=ErrorType.REQUIRES_HUMAN,
                    message=f"Execution of '{tool_call.name}' was rejected",
                    code="CONFIRMATION_REJECTED",
                ),
            )

        return None  # Confirmed, proceed

    async def _execute_with_retry(
        self,
        provider: ToolProvider,
        call: ToolCall,
        spec: ToolSpec,
    ) -> ToolResult:
        """
        Execute tool with retry for retryable errors.

        Uses exponential backoff: retry_delay_ms * 2^attempt.
        """
        last_error: ToolError | None = None

        for attempt in range(1 + spec.max_retries):
            start_ms = int(time.monotonic() * 1000)

            try:
                result = await asyncio.wait_for(
                    self.backend.execute(provider, call),
                    timeout=spec.timeout_ms / 1000.0,
                )
                result.duration_ms = int(time.monotonic() * 1000) - start_ms
                result.retry_count = attempt

                if result.success:
                    return result

                # Check if error is retryable
                if result.error and result.error.is_retryable and attempt < spec.max_retries:
                    delay = spec.retry_delay_ms * (2**attempt) / 1000.0
                    await asyncio.sleep(delay)
                    last_error = result.error
                    continue

                return result

            except TimeoutError:
                last_error = ToolError(
                    error_type=ErrorType.RETRYABLE,
                    message=f"Tool '{call.name}' timed out after {spec.timeout_ms}ms",
                    code="TIMEOUT",
                )
                if attempt < spec.max_retries:
                    delay = spec.retry_delay_ms * (2**attempt) / 1000.0
                    await asyncio.sleep(delay)
                    continue

            except ToolExecutionError as e:
                error_type = ErrorType.RETRYABLE if e.retryable else ErrorType.NON_RETRYABLE
                last_error = ToolError(
                    error_type=error_type,
                    message=str(e),
                    code=e.error_code,
                )
                if e.retryable and attempt < spec.max_retries:
                    delay = spec.retry_delay_ms * (2**attempt) / 1000.0
                    await asyncio.sleep(delay)
                    continue

            except Exception as e:
                last_error = ToolError(
                    error_type=ErrorType.NON_RETRYABLE,
                    message=str(e),
                    code="UNEXPECTED",
                )
                break  # Don't retry unexpected errors

        # All attempts exhausted
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=False,
            error=last_error,
            duration_ms=int(time.monotonic() * 1000) - start_ms,
            retry_count=min(attempt, spec.max_retries),
        )

    def _log_tool_call(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        spec: ToolSpec,
        trace_ctx: TraceContext,
    ) -> None:
        """Write a TraceEvent with ToolCallRecord for audit."""
        if not self.trace_writer:
            return

        result_data: Any = result.output
        record = ToolCallRecord(
            name=tool_call.name,
            args_digest=canonical_hash(tool_call.arguments),
            idempotency_key=tool_call.idempotency_key,
            result_digest=canonical_hash(result_data) if result_data is not None else None,
            error=result.error.message if result.error else None,
            duration_ms=result.duration_ms,
            side_effect=spec.side_effect.value,
        )

        event = TraceEvent(
            run_id=trace_ctx.run_id,
            task_id=trace_ctx.task_id,
            step_id=tool_call.step_id or trace_ctx.new_step_id(),
            parent_step_id=tool_call.parent_step_id,
            event_type=EventType.TOOL_CALL,
            tool_call=record,
        )
        self.trace_writer.write(event)

    def _log_authorization_failure(
        self,
        tool_call: ToolCall,
        missing: list[str],
        trace_ctx: TraceContext | None,
    ) -> None:
        """Audit unauthorized tool call attempts."""
        if not self.trace_writer or not trace_ctx:
            return

        record = ToolCallRecord(
            name=tool_call.name,
            args_digest=canonical_hash(tool_call.arguments),
            error=f"UNAUTHORIZED: missing capabilities {missing}",
        )

        event = TraceEvent(
            run_id=trace_ctx.run_id,
            task_id=trace_ctx.task_id,
            step_id=tool_call.step_id or trace_ctx.new_step_id(),
            parent_step_id=tool_call.parent_step_id,
            event_type=EventType.TOOL_CALL,
            tool_call=record,
        )
        self.trace_writer.write(event)
