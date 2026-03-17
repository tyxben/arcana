"""Protocol for tool providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from arcana.contracts.tool import ToolCall, ToolResult, ToolSpec


@runtime_checkable
class ToolProvider(Protocol):
    """
    Protocol for tool implementations.

    Each ToolProvider encapsulates a single tool. Authorization,
    validation, retry, and tracing are handled by the ToolGateway
    layer above.
    """

    @property
    def spec(self) -> ToolSpec:
        """Return the tool specification."""
        ...

    async def execute(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            call: The tool call with arguments

        Returns:
            ToolResult with output or error
        """
        ...

    async def health_check(self) -> bool:
        """Check if the tool is operational."""
        return True


class ToolExecutionError(Exception):
    """Exception raised during tool execution."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        retryable: bool = False,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.retryable = retryable
        self.error_code = error_code
