"""MCP Client -- manages connections to MCP servers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from arcana.contracts.mcp import MCPServerConfig, MCPToolSpec, MCPTransportType
from arcana.mcp.protocol import is_notification, make_request
from arcana.mcp.transport.base import MCPTransport

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Manages connections to one or more MCP servers.

    Lifecycle: connect() -> list_tools() -> call_tool() -> disconnect()

    Parameters
    ----------
    on_tools_changed:
        Optional async callback invoked when a server sends a
        ``notifications/tools/list_changed`` notification.  The callback
        receives ``(server_name, new_tool_list)`` *after* this client's
        ``_tools`` dict has already been updated.
    """

    def __init__(
        self,
        on_tools_changed: Callable[[str, list[MCPToolSpec]], Awaitable[None]] | None = None,
    ) -> None:
        self._connections: dict[str, MCPConnection] = {}
        self._tools: dict[str, tuple[str, MCPToolSpec]] = {}  # qualified_name -> (server_name, spec)
        self._on_tools_changed = on_tools_changed

    async def connect(self, config: MCPServerConfig) -> list[MCPToolSpec]:
        """Connect to an MCP server and discover its tools."""
        transport = _create_transport(config)
        connection = MCPConnection(config=config, transport=transport)

        # Wire up notification callback if the caller wants tool-change events.
        if self._on_tools_changed is not None:
            connection.set_notification_callback(
                self._make_tools_changed_handler(config.name)
            )

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
        self._update_tools_for_server(config.name, tools)

        self._connections[config.name] = connection
        logger.info(
            "Connected to MCP server '%s', discovered %d tools",
            config.name,
            len(tools),
        )
        return tools

    # ------------------------------------------------------------------
    # Internal helpers for dynamic tool updates
    # ------------------------------------------------------------------

    def _update_tools_for_server(
        self, server_name: str, tools: list[MCPToolSpec]
    ) -> None:
        """Replace all tool entries for *server_name* with *tools*."""
        # Remove old entries for this server
        self._tools = {
            k: v for k, v in self._tools.items() if v[0] != server_name
        }
        # Add new entries
        for tool in tools:
            qualified_name = f"{server_name}.{tool.name}"
            self._tools[qualified_name] = (server_name, tool)

    def _make_tools_changed_handler(
        self, server_name: str
    ) -> Callable[[str, list[MCPToolSpec]], Awaitable[None]]:
        """Return an async handler bound to *server_name*."""

        async def _handler(
            _server_name: str, new_tools: list[MCPToolSpec]
        ) -> None:
            self._update_tools_for_server(server_name, new_tools)
            assert self._on_tools_changed is not None
            await self._on_tools_changed(server_name, new_tools)

        return _handler

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server."""
        conn = self._connections.pop(name, None)
        if conn:
            await conn.disconnect()
            # Remove tools from this server
            self._tools = {k: v for k, v in self._tools.items() if v[0] != name}

    async def disconnect_all(self) -> None:
        """Disconnect from all servers (continues on individual failures)."""
        for name in list(self._connections.keys()):
            try:
                await self.disconnect(name)
            except Exception:
                logger.warning("Failed to disconnect MCP server '%s'", name, exc_info=True)

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
        self._connected = False
        self._reconnect_lock = asyncio.Lock()
        self._notification_callback: (
            Callable[[str, list[MCPToolSpec]], Awaitable[None]] | None
        ) = None

    def set_notification_callback(
        self,
        callback: Callable[[str, list[MCPToolSpec]], Awaitable[None]],
    ) -> None:
        """Register a callback for ``notifications/tools/list_changed``.

        The callback receives ``(server_name, new_tool_list)`` and is
        invoked during normal request-response processing whenever a
        notification is encountered inline.
        """
        self._notification_callback = callback

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
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        await self._transport.close()

    async def list_tools(self) -> list[MCPToolSpec]:
        """Get available tools from server."""
        try:
            result = await self._send_request("tools/list", {})
        except ConnectionError:
            await self._reconnect()
            result = await self._send_request("tools/list", {})
        tools_data = result.get("tools", []) if isinstance(result, dict) else []
        return [MCPToolSpec.model_validate(t) for t in tools_data]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool and return the result."""
        try:
            result = await self._send_request(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
            )
        except ConnectionError:
            await self._reconnect()
            result = await self._send_request(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
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

        # Read response (dispatch notifications), bounded by server timeout
        deadline = asyncio.get_event_loop().time() + (self._config.timeout_ms / 1000)
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"MCP server '{self._config.name}' did not respond to '{method}' "
                    f"within {self._config.timeout_ms}ms."
                )
            response = await asyncio.wait_for(
                self._transport.receive(), timeout=remaining,
            )
            # Handle server-initiated notifications inline
            if is_notification(response):
                await self._handle_notification(response)
                continue
            if response.id == msg_id:
                if response.error:
                    raise MCPCallError(response.error.code, response.error.message)
                return response.result
            # else: response for a different id, skip

    async def _handle_notification(self, message: Any) -> None:
        """Route a server notification to the appropriate handler."""
        method = message.method
        if method == "notifications/tools/list_changed":
            logger.info(
                "MCP server '%s' signalled tools/list_changed",
                self._config.name,
            )
            if self._notification_callback is not None:
                new_tools = await self.list_tools()
                await self._notification_callback(self._config.name, new_tools)
        else:
            logger.debug(
                "Ignoring unknown MCP notification '%s' from '%s'",
                method,
                self._config.name,
            )

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff (serialized via lock)."""
        async with self._reconnect_lock:
            # Mark disconnected; if another coroutine already reconnected, skip
            self._connected = False
            for attempt in range(self._config.reconnect_attempts):
                try:
                    await self._transport.close()
                    await self._transport.connect()
                    # Re-initialize (handshake) — sets self._connected = True
                    self._msg_counter = 0
                    await self._send_request(
                        "initialize",
                        {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "arcana", "version": "0.1.0"},
                        },
                    )
                    self._connected = True
                    logger.info("Reconnected to MCP server '%s'", self._config.name)
                    return
                except Exception:
                    delay = self._config.reconnect_delay_ms * (2**attempt) / 1000
                    logger.warning(
                        "Reconnect attempt %d/%d for '%s' failed, retrying in %.1fs",
                        attempt + 1, self._config.reconnect_attempts,
                        self._config.name, delay,
                    )
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
