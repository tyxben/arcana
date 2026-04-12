"""Tests for MCP dynamic tool discovery via notifications/tools/list_changed."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from arcana.contracts.mcp import MCPMessage, MCPServerConfig, MCPToolSpec
from arcana.mcp.protocol import is_notification

# ---------------------------------------------------------------------------
# is_notification() pure function tests
# ---------------------------------------------------------------------------


class TestIsNotification:
    """is_notification correctly distinguishes notifications from responses."""

    def test_notification_has_method_no_id(self):
        msg = MCPMessage(method="notifications/tools/list_changed")
        assert is_notification(msg) is True

    def test_notification_with_params(self):
        msg = MCPMessage(method="notifications/progress", params={"progress": 50})
        assert is_notification(msg) is True

    def test_response_has_id_and_result(self):
        msg = MCPMessage(id=1, result={"tools": []})
        assert is_notification(msg) is False

    def test_request_has_id_and_method(self):
        """A request (client-initiated) has both id and method -- not a notification."""
        msg = MCPMessage(id=42, method="tools/list", params={})
        assert is_notification(msg) is False

    def test_error_response(self):
        from arcana.contracts.mcp import MCPError

        msg = MCPMessage(id=1, error=MCPError(code=-32601, message="Not found"))
        assert is_notification(msg) is False

    def test_bare_message_no_method_no_id(self):
        """Edge case: neither method nor id -- not a notification."""
        msg = MCPMessage(result={"data": "x"})
        assert is_notification(msg) is False


# ---------------------------------------------------------------------------
# Helper to build mock transports
# ---------------------------------------------------------------------------


def _make_mock_transport(responses: list[MCPMessage]) -> AsyncMock:
    """Create a mock transport that returns responses in sequence."""
    mock = AsyncMock()
    mock.is_connected = True
    mock.connect = AsyncMock()
    mock.send = AsyncMock()
    mock.receive = AsyncMock(side_effect=responses)
    mock.close = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# MCPConnection notification handling
# ---------------------------------------------------------------------------


class TestMCPConnectionNotifications:
    @pytest.mark.asyncio
    async def test_tools_list_changed_triggers_relist_and_callback(self):
        """When a tools/list_changed notification arrives during _send_request,
        the connection should re-list tools and invoke the callback."""
        from arcana.mcp.client import MCPConnection

        config = MCPServerConfig(name="test-server", command="echo")

        callback_calls: list[tuple[str, list[MCPToolSpec]]] = []

        async def on_change(server_name: str, tools: list[MCPToolSpec]) -> None:
            callback_calls.append((server_name, tools))

        call_count = 0

        async def receive_side_effect():
            nonlocal call_count
            call_count += 1
            # 1: init response
            if call_count == 1:
                return MCPMessage(id=1, result={"capabilities": {}})
            # 2: notification arrives before the actual response
            if call_count == 2:
                return MCPMessage(method="notifications/tools/list_changed")
            # 3: response to the list_tools() triggered by the notification (id=3)
            if call_count == 3:
                return MCPMessage(
                    id=3,
                    result={
                        "tools": [
                            {"name": "new_tool", "inputSchema": {"type": "object"}}
                        ]
                    },
                )
            # 4: the original request response (id=2)
            if call_count == 4:
                return MCPMessage(
                    id=2,
                    result={"content": [{"type": "text", "text": "hello"}]},
                )
            return MCPMessage(id=999, result={})

        transport = AsyncMock()
        transport.connect = AsyncMock()
        transport.close = AsyncMock()
        transport.send = AsyncMock()
        transport.receive = AsyncMock(side_effect=receive_side_effect)

        conn = MCPConnection(config=config, transport=transport)
        conn.set_notification_callback(on_change)
        await conn.connect()

        # Now make a call_tool that will encounter the notification inline
        result = await conn.call_tool("some_tool", {"arg": "val"})

        assert result == "hello"
        # The callback should have been invoked once
        assert len(callback_calls) == 1
        server_name, new_tools = callback_calls[0]
        assert server_name == "test-server"
        assert len(new_tools) == 1
        assert new_tools[0].name == "new_tool"

    @pytest.mark.asyncio
    async def test_unknown_notification_is_ignored(self, caplog):
        """Unknown notifications should be logged at DEBUG and skipped."""
        from arcana.mcp.client import MCPConnection

        config = MCPServerConfig(name="test-server", command="echo")

        call_count = 0

        async def receive_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MCPMessage(id=1, result={"capabilities": {}})
            # Unknown notification
            if call_count == 2:
                return MCPMessage(method="notifications/something/unknown")
            # Actual response
            if call_count == 3:
                return MCPMessage(id=2, result={"tools": []})
            return MCPMessage(id=999, result={})

        transport = AsyncMock()
        transport.connect = AsyncMock()
        transport.close = AsyncMock()
        transport.send = AsyncMock()
        transport.receive = AsyncMock(side_effect=receive_side_effect)

        conn = MCPConnection(config=config, transport=transport)
        await conn.connect()

        with caplog.at_level(logging.DEBUG, logger="arcana.mcp.client"):
            tools = await conn.list_tools()

        assert tools == []
        # Verify the unknown notification was logged
        assert any(
            "Ignoring unknown MCP notification" in rec.message
            and "notifications/something/unknown" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_no_callback_means_notification_logged_only(self, caplog):
        """Without a callback, tools/list_changed is logged but no re-list happens."""
        from arcana.mcp.client import MCPConnection

        config = MCPServerConfig(name="test-server", command="echo")

        call_count = 0

        async def receive_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MCPMessage(id=1, result={"capabilities": {}})
            # tools/list_changed notification, but no callback set
            if call_count == 2:
                return MCPMessage(method="notifications/tools/list_changed")
            # Actual response
            if call_count == 3:
                return MCPMessage(
                    id=2,
                    result={"content": [{"type": "text", "text": "ok"}]},
                )
            return MCPMessage(id=999, result={})

        transport = AsyncMock()
        transport.connect = AsyncMock()
        transport.close = AsyncMock()
        transport.send = AsyncMock()
        transport.receive = AsyncMock(side_effect=receive_side_effect)

        conn = MCPConnection(config=config, transport=transport)
        # Deliberately NOT setting a notification callback
        await conn.connect()

        with caplog.at_level(logging.INFO, logger="arcana.mcp.client"):
            result = await conn.call_tool("some_tool", {})

        assert result == "ok"
        # Should be logged but no list_tools re-call (only 3 receive calls total)
        assert call_count == 3
        assert any(
            "tools/list_changed" in rec.message for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# MCPClient integration: on_tools_changed + _tools update
# ---------------------------------------------------------------------------


class TestMCPClientDynamicDiscovery:
    @pytest.mark.asyncio
    async def test_on_tools_changed_updates_tools_dict(self):
        """MCPClient._tools should be updated when tools change."""
        from arcana.mcp.client import MCPClient

        callback_calls: list[tuple[str, list[MCPToolSpec]]] = []

        async def on_change(server_name: str, tools: list[MCPToolSpec]) -> None:
            callback_calls.append((server_name, tools))

        # Initial connect: init + tools/list with 1 tool
        init_response = MCPMessage(id=1, result={"capabilities": {}})
        tools_response = MCPMessage(
            id=2,
            result={
                "tools": [
                    {"name": "old_tool", "description": "Old", "inputSchema": {}}
                ]
            },
        )

        call_count = 0

        async def receive_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return init_response
            if call_count == 2:
                return tools_response
            # After connect, simulate a call_tool that encounters a notification:
            # 3: notification
            if call_count == 3:
                return MCPMessage(method="notifications/tools/list_changed")
            # 4: response to re-list triggered by notification (id=4)
            if call_count == 4:
                return MCPMessage(
                    id=4,
                    result={
                        "tools": [
                            {"name": "new_tool_a", "description": "A", "inputSchema": {}},
                            {"name": "new_tool_b", "description": "B", "inputSchema": {}},
                        ]
                    },
                )
            # 5: the original call_tool response (id=3)
            if call_count == 5:
                return MCPMessage(
                    id=3,
                    result={"content": [{"type": "text", "text": "done"}]},
                )
            return MCPMessage(id=999, result={})

        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        mock_transport.connect = AsyncMock()
        mock_transport.send = AsyncMock()
        mock_transport.receive = AsyncMock(side_effect=receive_side_effect)
        mock_transport.close = AsyncMock()

        with patch(
            "arcana.mcp.client._create_transport", return_value=mock_transport
        ):
            client = MCPClient(on_tools_changed=on_change)
            config = MCPServerConfig(name="srv", command="echo")
            await client.connect(config)

            # Verify initial tools
            assert "srv.old_tool" in client._tools

            # Now call a tool -- the notification will fire inline
            result = await client.call_tool("srv", "some_tool", {})
            assert result == "done"

            # Callback should have been invoked
            assert len(callback_calls) == 1
            assert callback_calls[0][0] == "srv"
            assert len(callback_calls[0][1]) == 2

            # _tools should be updated: old_tool removed, new_tool_a + new_tool_b added
            assert "srv.old_tool" not in client._tools
            assert "srv.new_tool_a" in client._tools
            assert "srv.new_tool_b" in client._tools

            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_no_callback_backward_compatible(self):
        """MCPClient() with no on_tools_changed should work exactly as before."""
        from arcana.mcp.client import MCPClient

        init_response = MCPMessage(id=1, result={"capabilities": {}})
        tools_response = MCPMessage(
            id=2,
            result={
                "tools": [
                    {"name": "read_file", "description": "Read", "inputSchema": {}}
                ]
            },
        )
        mock_transport = _make_mock_transport([init_response, tools_response])

        with patch(
            "arcana.mcp.client._create_transport", return_value=mock_transport
        ):
            client = MCPClient()  # No on_tools_changed
            config = MCPServerConfig(name="fs", command="echo")
            tools = await client.connect(config)

            assert len(tools) == 1
            assert "fs.read_file" in client._tools
            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_get_all_tools_reflects_updates(self):
        """get_all_tools() should reflect dynamic tool updates."""
        from arcana.mcp.client import MCPClient

        async def noop_callback(server_name: str, tools: list[MCPToolSpec]) -> None:
            pass

        client = MCPClient(on_tools_changed=noop_callback)

        # Manually populate _tools to test _update_tools_for_server
        client._tools = {
            "srv.a": ("srv", MCPToolSpec(name="a", input_schema={})),
            "srv.b": ("srv", MCPToolSpec(name="b", input_schema={})),
            "other.x": ("other", MCPToolSpec(name="x", input_schema={})),
        }

        # Update tools for "srv"
        client._update_tools_for_server(
            "srv",
            [MCPToolSpec(name="c", input_schema={})],
        )

        all_tools = client.get_all_tools()
        names = [name for name, _ in all_tools]
        assert "srv.a" not in names
        assert "srv.b" not in names
        assert "srv.c" in names
        # Other server's tools are untouched
        assert "other.x" in names
