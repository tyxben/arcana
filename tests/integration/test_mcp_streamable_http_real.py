"""Integration test: MCP Streamable HTTP against a local test server (Task 1.5).

Starts a minimal FastAPI-based MCP server that speaks Streamable HTTP,
then connects with MCPClient and verifies tool discovery + invocation.
"""

from __future__ import annotations

import socket

import pytest

# ---------------------------------------------------------------------------
# Minimal MCP test server
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _handle_jsonrpc(body: dict) -> dict:
    """Route a JSON-RPC request to a handler."""
    method = body.get("method", "")
    msg_id = body.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-mcp-server", "version": "0.1"},
            },
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Returns its input",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ]
            },
        }
    elif method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "echo":
            text = arguments.get("text", "")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": text}]
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }


async def _start_test_server(port: int):
    """Start a simple HTTP server that handles MCP JSON-RPC POST requests."""
    from aiohttp import web

    session_counter = 0

    async def mcp_handler(request: web.Request) -> web.Response:
        nonlocal session_counter

        body = await request.json()
        result = _handle_jsonrpc(body)

        session_counter += 1
        return web.json_response(
            result,
            headers={
                "Mcp-Session-Id": f"test-session-{session_counter}",
            },
        )

    app = web.Application()
    app.router.add_post("/mcp", mcp_handler)
    # Also handle DELETE for session cleanup
    app.router.add_delete("/mcp", lambda r: web.Response(status=200))

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMCPStreamableHTTPReal:
    """Test MCP Streamable HTTP transport against a real HTTP server."""

    async def test_connect_discover_call(self):
        """Full lifecycle: connect -> discover tools -> call echo -> verify."""
        try:
            from aiohttp import web  # noqa: F401
        except ImportError:
            pytest.skip("aiohttp not installed")

        from arcana.contracts.mcp import MCPServerConfig, MCPTransportType
        from arcana.mcp.client import MCPClient

        port = _find_free_port()
        runner = await _start_test_server(port)

        try:
            client = MCPClient()
            config = MCPServerConfig(
                name="test-http",
                transport=MCPTransportType.STREAMABLE_HTTP,
                url=f"http://127.0.0.1:{port}/mcp",
                timeout_ms=5000,
            )

            # Connect and discover tools
            tools = await client.connect(config)
            assert len(tools) == 1
            assert tools[0].name == "echo"

            # Call the echo tool
            result = await client.call_tool("test-http", "echo", {"text": "hello arcana"})
            assert result == "hello arcana"

            await client.disconnect_all()
        finally:
            await runner.cleanup()

    async def test_server_error_handling(self):
        """Calling unknown tool should return error."""
        try:
            from aiohttp import web  # noqa: F401
        except ImportError:
            pytest.skip("aiohttp not installed")

        from arcana.contracts.mcp import MCPServerConfig, MCPTransportType
        from arcana.mcp.client import MCPCallError, MCPClient

        port = _find_free_port()
        runner = await _start_test_server(port)

        try:
            client = MCPClient()
            config = MCPServerConfig(
                name="test-http",
                transport=MCPTransportType.STREAMABLE_HTTP,
                url=f"http://127.0.0.1:{port}/mcp",
                timeout_ms=5000,
            )

            await client.connect(config)

            with pytest.raises(MCPCallError):
                await client.call_tool("test-http", "nonexistent_tool", {})

            await client.disconnect_all()
        finally:
            await runner.cleanup()
