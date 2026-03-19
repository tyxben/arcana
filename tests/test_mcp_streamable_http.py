"""Tests for MCP Streamable HTTP transport."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.contracts.mcp import MCPServerConfig, MCPTransportType


class TestStreamableHTTPTransport:
    """Unit tests for StreamableHTTPTransport."""

    def _make_config(self, **kwargs) -> MCPServerConfig:
        return MCPServerConfig(
            name="test-server",
            transport=MCPTransportType.STREAMABLE_HTTP,
            url="https://example.com/mcp",
            **kwargs,
        )

    async def test_connect_sets_connected(self) -> None:
        from arcana.mcp.transport.streamable_http import StreamableHTTPTransport

        transport = StreamableHTTPTransport(self._make_config())
        await transport.connect()
        assert transport.is_connected
        await transport.close()
        assert not transport.is_connected

    async def test_connect_requires_url(self) -> None:
        from arcana.mcp.transport.streamable_http import StreamableHTTPTransport

        config = MCPServerConfig(
            name="no-url",
            transport=MCPTransportType.STREAMABLE_HTTP,
        )
        transport = StreamableHTTPTransport(config)
        with pytest.raises(ValueError, match="no URL"):
            await transport.connect()

    async def test_send_posts_json(self) -> None:
        from arcana.mcp.protocol import make_request
        from arcana.mcp.transport.streamable_http import StreamableHTTPTransport

        transport = StreamableHTTPTransport(self._make_config())
        await transport.connect()

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"protocolVersion": "2025-03-26"},
        }).encode()
        transport._client.post = AsyncMock(return_value=mock_response)

        msg = make_request("initialize", {"protocolVersion": "2025-03-26"}, 1)
        await transport.send(msg)

        # Should have posted to the URL
        transport._client.post.assert_called_once()
        call_kwargs = transport._client.post.call_args
        assert call_kwargs[0][0] == "https://example.com/mcp"

        # Response should be in inbox
        received = await transport.receive()
        assert received.id == 1
        assert received.result is not None

        await transport.close()

    async def test_send_handles_sse_response(self) -> None:
        from arcana.mcp.protocol import make_request
        from arcana.mcp.transport.streamable_http import StreamableHTTPTransport

        transport = StreamableHTTPTransport(self._make_config())
        await transport.connect()

        # SSE response with multiple events
        sse_body = (
            'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progress":50}}\n\n'
            'data: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n\n'
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.text = sse_body
        transport._client.post = AsyncMock(return_value=mock_response)

        msg = make_request("tools/list", {}, 1)
        await transport.send(msg)

        # Should have 2 messages in inbox
        msg1 = await transport.receive()
        assert msg1.method == "notifications/progress"

        msg2 = await transport.receive()
        assert msg2.id == 1

        await transport.close()

    async def test_session_id_captured(self) -> None:
        from arcana.mcp.protocol import make_request
        from arcana.mcp.transport.streamable_http import StreamableHTTPTransport

        transport = StreamableHTTPTransport(self._make_config())
        await transport.connect()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "application/json",
            "mcp-session-id": "abc-123",
        }
        mock_response.content = json.dumps({
            "jsonrpc": "2.0", "id": 1, "result": {},
        }).encode()
        transport._client.post = AsyncMock(return_value=mock_response)

        msg = make_request("initialize", {}, 1)
        await transport.send(msg)

        assert transport._session_id == "abc-123"

        # Next send should include session ID
        mock_response2 = MagicMock()
        mock_response2.status_code = 202
        mock_response2.headers = {}
        transport._client.post = AsyncMock(return_value=mock_response2)

        from arcana.mcp.protocol import make_request
        notif = make_request("initialized", {})
        notif.id = None  # notification
        await transport.send(notif)

        call_headers = transport._client.post.call_args[1]["headers"]
        assert call_headers.get("Mcp-Session-Id") == "abc-123"

        await transport.close()

    async def test_notification_returns_no_body(self) -> None:
        from arcana.contracts.mcp import MCPMessage
        from arcana.mcp.transport.streamable_http import StreamableHTTPTransport

        transport = StreamableHTTPTransport(self._make_config())
        await transport.connect()

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {}
        transport._client.post = AsyncMock(return_value=mock_response)

        notif = MCPMessage(method="notifications/initialized")
        await transport.send(notif)

        # Inbox should be empty — 202 doesn't produce a message
        assert transport._inbox.empty()

        await transport.close()

    async def test_http_error_raises(self) -> None:
        from arcana.mcp.protocol import make_request
        from arcana.mcp.transport.streamable_http import StreamableHTTPTransport

        transport = StreamableHTTPTransport(self._make_config())
        await transport.connect()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.text = "Internal Server Error"
        transport._client.post = AsyncMock(return_value=mock_response)

        msg = make_request("test", {}, 1)
        with pytest.raises(ConnectionError, match="500"):
            await transport.send(msg)

        await transport.close()


class TestTransportFactory:
    """Test that the factory creates StreamableHTTPTransport."""

    def test_factory_creates_streamable_http(self) -> None:
        from arcana.mcp.client import _create_transport
        from arcana.mcp.transport.streamable_http import StreamableHTTPTransport

        config = MCPServerConfig(
            name="test",
            transport=MCPTransportType.STREAMABLE_HTTP,
            url="https://example.com/mcp",
        )
        transport = _create_transport(config)
        assert isinstance(transport, StreamableHTTPTransport)
