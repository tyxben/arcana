"""Execution channel contracts for Brain/Hands separation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from arcana.contracts.tool import ToolCall, ToolResult


@runtime_checkable
class ExecutionChannel(Protocol):
    """Protocol for Brain <-> Hands communication.

    Decouples HOW the agent loop communicates with tool execution.
    Default: LocalChannel (direct call, zero overhead).
    Future: SocketChannel, HTTPChannel for physical separation.
    """

    async def execute(self, call: ToolCall) -> ToolResult:
        """Send a tool call and receive the result."""
        ...

    async def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Send multiple tool calls and receive results."""
        ...

    async def close(self) -> None:
        """Close the channel."""
        ...
