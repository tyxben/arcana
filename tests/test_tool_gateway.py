"""Unit tests for Tool Gateway."""

import asyncio

import pytest

from arcana.contracts.tool import (
    ErrorType,
    SideEffect,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)
from arcana.contracts.trace import TraceContext
from arcana.tool_gateway.base import ToolExecutionError, ToolProvider
from arcana.tool_gateway.gateway import ToolGateway
from arcana.tool_gateway.registry import ToolRegistry
from arcana.tool_gateway.validators import validate_arguments
from arcana.trace.writer import TraceWriter


# ── Test Tool Providers ──────────────────────────────────────────


class EchoTool(ToolProvider):
    """Tool that echoes its input."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echo input back",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            side_effect=SideEffect.NONE,
            capabilities=[],
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output=call.arguments.get("message", ""),
        )


class WriteTool(ToolProvider):
    """Tool with WRITE side effect requiring capabilities."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_write",
            description="Write a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            side_effect=SideEffect.WRITE,
            requires_confirmation=True,
            capabilities=["fs:write", "fs:read"],
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output="written",
        )


class FailingTool(ToolProvider):
    """Tool that fails with retryable errors, then succeeds."""

    def __init__(self, fail_count: int = 1) -> None:
        self._fail_count = fail_count
        self._attempt = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="flaky",
            description="Flaky tool",
            input_schema={"type": "object", "properties": {}},
            side_effect=SideEffect.READ,
            capabilities=[],
            max_retries=3,
            retry_delay_ms=1,  # Fast retries for tests
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        self._attempt += 1
        if self._attempt <= self._fail_count:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=ToolError(
                    error_type=ErrorType.RETRYABLE,
                    message="Temporary failure",
                ),
            )
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output="ok",
        )


class NonRetryableFailTool(ToolProvider):
    """Tool that fails with non-retryable error."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="hard_fail",
            description="Always fails permanently",
            input_schema={"type": "object", "properties": {}},
            side_effect=SideEffect.READ,
            capabilities=[],
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=False,
            error=ToolError(
                error_type=ErrorType.NON_RETRYABLE,
                message="Permanent failure",
                code="FATAL",
            ),
        )


class SlowTool(ToolProvider):
    """Tool that takes too long (for timeout tests)."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="slow",
            description="Slow tool",
            input_schema={"type": "object", "properties": {}},
            side_effect=SideEffect.READ,
            capabilities=[],
            timeout_ms=50,
            max_retries=0,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        await asyncio.sleep(1)  # Sleep longer than timeout
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output="done",
        )


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def registry():
    """Create a registry with test tools."""
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(WriteTool())
    return reg


@pytest.fixture
def trace_ctx():
    return TraceContext(run_id="test-run")


def _make_call(name: str = "echo", **kwargs) -> ToolCall:
    """Helper to create a ToolCall."""
    defaults = {
        "id": "call-1",
        "name": name,
        "arguments": {"message": "hello"},
    }
    defaults.update(kwargs)
    return ToolCall(**defaults)


# ── TestToolRegistry ─────────────────────────────────────────────


class TestToolRegistry:
    """Test the tool registry."""

    def test_register_and_get(self, registry):
        provider = registry.get("echo")
        assert provider is not None
        assert provider.spec.name == "echo"

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    def test_unregister(self, registry):
        assert registry.unregister("echo")
        assert registry.get("echo") is None

    def test_unregister_nonexistent(self, registry):
        assert not registry.unregister("nonexistent")

    def test_list_tools(self, registry):
        tools = registry.list_tools()
        assert "echo" in tools
        assert "file_write" in tools

    def test_get_specs(self, registry):
        specs = registry.get_specs()
        assert len(specs) == 2
        names = [s.name for s in specs]
        assert "echo" in names

    def test_to_openai_tools(self, registry):
        tools = registry.to_openai_tools()
        assert len(tools) == 2
        echo_tool = next(t for t in tools if t["function"]["name"] == "echo")
        assert echo_tool["type"] == "function"
        assert "description" in echo_tool["function"]
        assert "parameters" in echo_tool["function"]


# ── TestAuthorization ────────────────────────────────────────────


class TestAuthorization:
    """Test capability-based authorization."""

    async def test_no_capabilities_required(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry, granted_capabilities=set())
        result = await gw.call(_make_call("echo"), trace_ctx=trace_ctx)
        assert result.success

    async def test_authorized_call(self, registry, trace_ctx):
        gw = ToolGateway(
            registry=registry,
            granted_capabilities={"fs:write", "fs:read"},
            confirmation_callback=_auto_confirm,
        )
        call = _make_call("file_write", arguments={"path": "a.txt", "content": "hi"})
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert result.success

    async def test_missing_capability_rejected(self, registry, trace_ctx):
        gw = ToolGateway(
            registry=registry,
            granted_capabilities={"fs:read"},  # Missing fs:write
        )
        call = _make_call("file_write", arguments={"path": "a.txt", "content": "hi"})
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert not result.success
        assert result.error is not None
        assert result.error.code == "UNAUTHORIZED"

    async def test_unauthorized_write_is_audited(self, registry, tmp_path, trace_ctx):
        trace_writer = TraceWriter(trace_dir=tmp_path)
        gw = ToolGateway(
            registry=registry,
            trace_writer=trace_writer,
            granted_capabilities=set(),  # No capabilities
        )
        call = _make_call("file_write", arguments={"path": "x", "content": "y"})
        result = await gw.call(call, trace_ctx=trace_ctx)

        assert not result.success
        # Check that trace event was written
        trace_files = list(tmp_path.glob("*.jsonl"))
        assert len(trace_files) > 0


# ── TestValidation ───────────────────────────────────────────────


class TestValidation:
    """Test argument validation."""

    def test_valid_arguments(self):
        spec = ToolSpec(
            name="t",
            description="t",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            },
        )
        assert validate_arguments(spec, {"name": "Alice", "age": 30}) is None

    def test_missing_required_field(self):
        spec = ToolSpec(
            name="t",
            description="t",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        error = validate_arguments(spec, {})
        assert error is not None
        assert error.code == "VALIDATION_ERROR"

    def test_wrong_type(self):
        spec = ToolSpec(
            name="t",
            description="t",
            input_schema={
                "type": "object",
                "properties": {"age": {"type": "integer"}},
            },
        )
        error = validate_arguments(spec, {"age": "not_a_number"})
        assert error is not None
        assert "integer" in error.message

    def test_empty_schema_always_valid(self):
        spec = ToolSpec(name="t", description="t", input_schema={})
        assert validate_arguments(spec, {"anything": "goes"}) is None

    async def test_validation_failure_in_gateway(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        call = _make_call("echo", arguments={})  # Missing required 'message'
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert not result.success
        assert result.error is not None
        assert result.error.code == "VALIDATION_ERROR"


# ── TestIdempotency ──────────────────────────────────────────────


class TestIdempotency:
    """Test idempotency cache."""

    async def test_same_key_returns_cached(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        call1 = _make_call("echo", idempotency_key="key-1")
        call2 = _make_call("echo", id="call-2", idempotency_key="key-1")

        result1 = await gw.call(call1, trace_ctx=trace_ctx)
        result2 = await gw.call(call2, trace_ctx=trace_ctx)

        assert result1.success
        assert result2.success
        # Should be the same cached result
        assert result2.tool_call_id == result1.tool_call_id

    async def test_no_key_not_cached(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        call1 = _make_call("echo")
        call2 = _make_call("echo", id="call-2")

        result1 = await gw.call(call1, trace_ctx=trace_ctx)
        result2 = await gw.call(call2, trace_ctx=trace_ctx)

        assert result1.success
        assert result2.success
        # Different executions
        assert result2.tool_call_id == "call-2"

    async def test_different_keys_execute_separately(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        call1 = _make_call("echo", idempotency_key="key-1")
        call2 = _make_call("echo", id="call-2", idempotency_key="key-2")

        result1 = await gw.call(call1, trace_ctx=trace_ctx)
        result2 = await gw.call(call2, trace_ctx=trace_ctx)

        assert result2.tool_call_id == "call-2"


# ── TestConfirmation ─────────────────────────────────────────────


async def _auto_confirm(call: ToolCall, spec: ToolSpec) -> bool:
    return True


async def _auto_reject(call: ToolCall, spec: ToolSpec) -> bool:
    return False


class TestConfirmation:
    """Test write confirmation gate."""

    async def test_read_tool_auto_confirmed(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)  # No callback needed for reads
        result = await gw.call(_make_call("echo"), trace_ctx=trace_ctx)
        assert result.success

    async def test_write_tool_no_callback_returns_error(self, registry, trace_ctx):
        gw = ToolGateway(
            registry=registry,
            granted_capabilities={"fs:write", "fs:read"},
            # No confirmation_callback
        )
        call = _make_call("file_write", arguments={"path": "a", "content": "b"})
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert not result.success
        assert result.error is not None
        assert result.error.code == "CONFIRMATION_REQUIRED"

    async def test_write_tool_with_confirm_callback(self, registry, trace_ctx):
        gw = ToolGateway(
            registry=registry,
            granted_capabilities={"fs:write", "fs:read"},
            confirmation_callback=_auto_confirm,
        )
        call = _make_call("file_write", arguments={"path": "a", "content": "b"})
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert result.success

    async def test_write_tool_rejected_by_callback(self, registry, trace_ctx):
        gw = ToolGateway(
            registry=registry,
            granted_capabilities={"fs:write", "fs:read"},
            confirmation_callback=_auto_reject,
        )
        call = _make_call("file_write", arguments={"path": "a", "content": "b"})
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert not result.success
        assert result.error is not None
        assert result.error.code == "CONFIRMATION_REJECTED"


# ── TestRetry ────────────────────────────────────────────────────


class TestRetry:
    """Test retry behavior."""

    async def test_retryable_error_retries_and_succeeds(self, trace_ctx):
        reg = ToolRegistry()
        reg.register(FailingTool(fail_count=2))

        gw = ToolGateway(registry=reg)
        call = _make_call("flaky", arguments={})
        result = await gw.call(call, trace_ctx=trace_ctx)

        assert result.success
        assert result.retry_count == 2

    async def test_non_retryable_error_fails_immediately(self, trace_ctx):
        reg = ToolRegistry()
        reg.register(NonRetryableFailTool())

        gw = ToolGateway(registry=reg)
        call = _make_call("hard_fail", arguments={})
        result = await gw.call(call, trace_ctx=trace_ctx)

        assert not result.success
        assert result.error is not None
        assert result.error.code == "FATAL"
        assert result.retry_count == 0

    async def test_max_retries_exhausted(self, trace_ctx):
        reg = ToolRegistry()
        reg.register(FailingTool(fail_count=100))  # Always fails

        gw = ToolGateway(registry=reg)
        call = _make_call("flaky", arguments={})
        result = await gw.call(call, trace_ctx=trace_ctx)

        assert not result.success
        assert result.retry_count == 3  # max_retries from spec

    async def test_timeout_returns_error(self, trace_ctx):
        reg = ToolRegistry()
        reg.register(SlowTool())

        gw = ToolGateway(registry=reg)
        call = _make_call("slow", arguments={})
        result = await gw.call(call, trace_ctx=trace_ctx)

        assert not result.success
        assert result.error is not None
        assert result.error.code == "TIMEOUT"


# ── TestTracing ──────────────────────────────────────────────────


class TestTracing:
    """Test audit/trace logging."""

    async def test_successful_call_logged(self, registry, tmp_path, trace_ctx):
        trace_writer = TraceWriter(trace_dir=tmp_path)
        gw = ToolGateway(registry=registry, trace_writer=trace_writer)

        await gw.call(_make_call("echo"), trace_ctx=trace_ctx)

        trace_files = list(tmp_path.glob("*.jsonl"))
        assert len(trace_files) > 0

    async def test_failed_call_logged(self, tmp_path, trace_ctx):
        reg = ToolRegistry()
        reg.register(NonRetryableFailTool())
        trace_writer = TraceWriter(trace_dir=tmp_path)

        gw = ToolGateway(registry=reg, trace_writer=trace_writer)
        await gw.call(_make_call("hard_fail", arguments={}), trace_ctx=trace_ctx)

        trace_files = list(tmp_path.glob("*.jsonl"))
        assert len(trace_files) > 0

    async def test_tool_not_found(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        result = await gw.call(_make_call("nonexistent"), trace_ctx=trace_ctx)
        assert not result.success
        assert result.error is not None
        assert result.error.code == "TOOL_NOT_FOUND"


# ── TestCallMany ─────────────────────────────────────────────────


class TestCallMany:
    """Test batch execution."""

    async def test_multiple_read_tools(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        calls = [
            _make_call("echo", id="c1", arguments={"message": "a"}),
            _make_call("echo", id="c2", arguments={"message": "b"}),
        ]
        results = await gw.call_many(calls, trace_ctx=trace_ctx)
        assert len(results) == 2
        assert all(r.success for r in results)
        # Results should be in original order
        assert results[0].tool_call_id == "c1"
        assert results[1].tool_call_id == "c2"
