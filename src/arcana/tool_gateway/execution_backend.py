"""Execution backends for tool isolation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from arcana.contracts.tool import ToolCall, ToolResult

if TYPE_CHECKING:
    from arcana.tool_gateway.base import ToolProvider


@runtime_checkable
class ExecutionBackend(Protocol):
    """Protocol for tool execution environments.

    Decouples WHERE a tool runs from WHAT it does.
    Default: InProcessBackend (current behavior, zero overhead).
    """

    async def execute(self, provider: ToolProvider, call: ToolCall) -> ToolResult:
        """Execute a tool call in this backend's environment."""
        ...

    async def cleanup(self) -> None:
        """Release backend resources."""
        ...


class InProcessBackend:
    """Execute tools directly in the current process. Default, zero overhead."""

    async def execute(self, provider: ToolProvider, call: ToolCall) -> ToolResult:
        """Delegate execution to the provider in-process."""
        return await provider.execute(call)

    async def cleanup(self) -> None:
        """No-op for in-process execution."""
