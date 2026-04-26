"""MCPToolProvider -- bridges a single MCP tool into Arcana's ToolGateway."""

from __future__ import annotations

from typing import Any

from arcana.contracts.tool import ToolCall, ToolError, ToolErrorCategory, ToolResult, ToolSpec


class MCPToolProvider:
    """
    Bridges one MCP tool into Arcana's ToolProvider protocol.

    After registration, this MCP tool is indistinguishable from
    native tools in the ToolGateway pipeline (auth, validation,
    retry, audit all apply).
    """

    def __init__(
        self,
        client: Any,  # MCPClient (avoid circular import)
        server_name: str,
        mcp_tool_name: str,
        arcana_spec: ToolSpec,
    ) -> None:
        self._client = client
        self._server_name = server_name
        self._mcp_tool_name = mcp_tool_name
        self._arcana_spec = arcana_spec

    @property
    def spec(self) -> ToolSpec:
        return self._arcana_spec

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute via MCP client, return Arcana ToolResult."""
        try:
            result = await self._client.call_tool(
                self._server_name,
                self._mcp_tool_name,
                call.arguments,
            )
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=True,
                output=result,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=ToolError(
                    # Treat client-side MCP exceptions as transport-level —
                    # connection drops, server restart, etc. Logical MCP
                    # errors with a JSON-RPC error code go through
                    # ``mcp_error_to_tool_error`` and get categorized there.
                    category=ToolErrorCategory.TRANSPORT,
                    message=str(e),
                    code="MCP_ERROR",
                ),
            )

    async def health_check(self) -> bool:
        return self._server_name in self._client.connected_servers
