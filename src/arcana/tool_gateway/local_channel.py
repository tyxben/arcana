"""Local in-process execution channel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.tool import ToolCall, ToolResult

if TYPE_CHECKING:
    from arcana.contracts.trace import TraceContext
    from arcana.tool_gateway.gateway import ToolGateway


class LocalChannel:
    """Direct in-process channel to ToolGateway. Zero overhead wrapper.

    This is the default channel -- same behavior as calling ToolGateway directly.
    """

    def __init__(
        self,
        gateway: ToolGateway,
        trace_ctx: TraceContext | None = None,
    ) -> None:
        self._gateway = gateway
        self._trace_ctx = trace_ctx

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call via the gateway."""
        return await self._gateway.call(call, trace_ctx=self._trace_ctx)

    async def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls via the gateway.

        Read tools run concurrently; write tools run sequentially. The
        runtime, not the LLM, owns this safety boundary (Constitution
        Principle 6, Principle 3).
        """
        return await self._gateway.call_many(calls, trace_ctx=self._trace_ctx)

    async def close(self) -> None:
        """Release underlying gateway resources."""
        await self._gateway.close()
