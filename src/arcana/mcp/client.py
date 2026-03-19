"""MCP Client -- manages connections to MCP servers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from arcana.contracts.mcp import MCPServerConfig, MCPToolSpec, MCPTransportType
from arcana.mcp.protocol import make_request
from arcana.mcp.transport.base import MCPTransport

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Manages connections to one or more MCP servers.

    Lifecycle: connect() -> list_tools() -> call_tool() -> disconnect()
    """

    def __init__(self) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._tools: dict[str, tuple[str, MCPToolSpec]] = {}  # qualified_name -> (server_name, spec)

    async def connect(self, config: MCPServerConfig) -> list[MCPToolSpec]:
        """Connect to an MCP server and discover its tools."""
        transport = _create_transport(config)
        connection = MCPConnection(config=config, transport=transport)
        try:
            await connection.connect()
        except Exception as exc:
            cmd_str = f"{config.command} {' '.join(config.args)}" if config.command else "(no command)"
            raise ConnectionError(
                f"Failed to connect to MCP server '{config.name}' (command: {cmd_str}). "
                f"Error: {exc}. "
                f"Check that '{config.command}' is installed and available on PATH."
            ) from exc

        # Discover tools
        tools = await connection.list_tools()
        for tool in tools:
            qualified_name = f"{config.name}.{tool.name}"
            self._tools[qualified_name] = (config.name, tool)

        self._connections[config.name] = connection
        logger.info(
            "Connected to MCP server '%s', discovered %d tools",
            config.name,
            len(tools),
        )
        return tools

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server."""
        conn = self._connections.pop(name, None)
        if conn:
            await conn.disconnect()
            # Remove tools from this server
            self._tools = {k: v for k, v in self._tools.items() if v[0] != name}

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._connections.keys()):
            await self.disconnect(name)

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        """Call a tool on a specific server."""
        conn = self._connections.get(server_name)
        if not conn:
            connected = self.connected_servers
            raise ConnectionError(
                f"Not connected to MCP server '{server_name}'. "
                f"Connected servers: {connected}. "
                f"Call connect() first or check the server name."
            )
        return await conn.call_tool(tool_name, arguments)

    def get_all_tools(self) -> list[tuple[str, MCPToolSpec]]:
        """Return all discovered tools: [(qualified_name, spec), ...]"""
        return [(name, spec) for name, (_, spec) in self._tools.items()]

    @property
    def connected_servers(self) -> list[str]:
        return list(self._connections.keys())


class MCPConnection:
    """Single MCP server connection."""

    def __init__(self, config: MCPServerConfig, transport: MCPTransport) -> None:
        self._config = config
        self._transport = transport
        self._msg_counter = 0

    async def connect(self) -> None:
        """Connect and initialize."""
        await self._transport.connect()
        # MCP handshake: initialize
        await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "arcana", "version": "0.1.0"},
            },
        )

    async def disconnect(self) -> None:
        await self._transport.close()

    async def list_tools(self) -> list[MCPToolSpec]:
        """Get available tools from server."""
        result = await self._send_request("tools/list", {})
        tools_data = result.get("tools", []) if isinstance(result, dict) else []
        return [MCPToolSpec.model_validate(t) for t in tools_data]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool and return the result."""
        result = await self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )
        # MCP tool results have content array
        if isinstance(result, dict):
            content = result.get("content", [])
            if content and isinstance(content, list):
                texts = [
                    c.get("text", "") for c in content if c.get("type") == "text"
                ]
                return "\n".join(texts)
        return str(result)

    async def _send_request(self, method: str, params: dict[str, Any]) -> Any:
        """Send JSON-RPC request and wait for response."""
        self._msg_counter += 1
        msg_id = self._msg_counter

        request = make_request(method, params, msg_id)
        await self._transport.send(request)

        # Read response (skip notifications)
        while True:
            response = await self._transport.receive()
            if response.id == msg_id:
                if response.error:
                    raise MCPCallError(response.error.code, response.error.message)
                return response.result
            # else: notification, skip

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        for attempt in range(self._config.reconnect_attempts):
            try:
                await self._transport.close()
                await self._transport.connect()
                await self.connect()
                return
            except Exception:
                delay = self._config.reconnect_delay_ms * (2**attempt) / 1000
                await asyncio.sleep(delay)
        cmd_str = f"{self._config.command} {' '.join(self._config.args)}" if self._config.command else "(no command)"
        raise ConnectionError(
            f"Failed to reconnect to MCP server '{self._config.name}' "
            f"after {self._config.reconnect_attempts} attempts (command: {cmd_str}). "
            f"Check that the server process is running and the command is correct."
        )


class MCPCallError(Exception):
    """Error from MCP tool call."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(
            f"MCP server returned error (code={code}): {message}. "
            f"This is an error from the MCP server, not from Arcana."
        )
        self.code = code


def _create_transport(config: MCPServerConfig) -> MCPTransport:
    """Factory: create transport from config."""
    if config.transport == MCPTransportType.STDIO:
        from arcana.mcp.transport.stdio import StdioTransport

        return StdioTransport(config)
    elif config.transport == MCPTransportType.SSE:
        raise NotImplementedError(
            f"SSE transport not yet implemented for MCP server '{config.name}'. "
            f"Use transport='stdio' instead."
        )
    elif config.transport == MCPTransportType.STREAMABLE_HTTP:
        from arcana.mcp.transport.streamable_http import StreamableHTTPTransport

        return StreamableHTTPTransport(config)
    else:
        raise ValueError(
            f"Unknown transport type '{config.transport}' for MCP server '{config.name}'. "
            f"Supported transports: stdio, streamable_http."
        )
