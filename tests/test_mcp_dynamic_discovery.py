"""Tests for MCP dynamic tool discovery via notifications/tools/list_changed."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

from arcana.contracts.mcp import MCPMessage, MCPServerConfig, MCPToolSpec
from arcana.contracts.tool import (
    SideEffect,
    ToolCall,
    ToolProvenance,
    ToolResult,
    ToolSpec,
)
from arcana.contracts.trace import EventType
from arcana.mcp.protocol import is_notification
from arcana.mcp.setup import setup_mcp_tools, unregister_mcp_tools_for_server
from arcana.tool_gateway.registry import ToolRegistry
from arcana.trace.writer import TraceWriter

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


class _StaticProvider:
    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output="ok",
        )

    async def health_check(self) -> bool:
        return True


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


# ---------------------------------------------------------------------------
# setup_mcp_tools bridge: dynamic discovery reaches ToolRegistry
# ---------------------------------------------------------------------------


class TestMCPSetupDynamicRegistryBridge:
    def test_unregister_mcp_tools_for_server_preserves_non_matching_tools(self):
        registry = ToolRegistry()
        registry.register(
            _StaticProvider(
                ToolSpec(
                    name="srv.remote",
                    description="Remote",
                    input_schema={},
                    provenance=ToolProvenance(origin="mcp", server_name="srv"),
                )
            )
        )
        registry.register(
            _StaticProvider(
                ToolSpec(
                    name="other.remote",
                    description="Other remote",
                    input_schema={},
                    provenance=ToolProvenance(origin="mcp", server_name="other"),
                )
            )
        )
        registry.register(
            _StaticProvider(
                ToolSpec(
                    name="srv.local_shadow",
                    description="Local",
                    input_schema={},
                )
            )
        )

        removed = unregister_mcp_tools_for_server(
            registry=registry,
            server_name="srv",
        )

        assert removed == ["srv.remote"]
        assert "srv.remote" not in registry.list_tools()
        assert "other.remote" in registry.list_tools()
        assert "srv.local_shadow" in registry.list_tools()

    @pytest.mark.asyncio
    async def test_setup_bridge_refreshes_registry_and_traces(self, tmp_path):
        registry = ToolRegistry()
        registry.register(
            _StaticProvider(
                ToolSpec(
                    name="srv.local_shadow",
                    description="Local",
                    input_schema={},
                )
            )
        )
        trace_writer = TraceWriter(trace_dir=tmp_path)

        call_count = 0

        async def receive_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MCPMessage(id=1, result={"capabilities": {}})
            if call_count == 2:
                return MCPMessage(
                    id=2,
                    result={
                        "tools": [
                            {
                                "name": "old_tool",
                                "description": "Old",
                                "inputSchema": {},
                                "annotations": {"readOnlyHint": True},
                            }
                        ]
                    },
                )
            if call_count == 3:
                return MCPMessage(method="notifications/tools/list_changed")
            if call_count == 4:
                return MCPMessage(
                    id=4,
                    result={
                        "tools": [
                            {
                                "name": "new_read",
                                "description": "New read",
                                "inputSchema": {},
                                "annotations": {"readOnlyHint": True},
                            },
                            {
                                "name": "new_unknown",
                                "description": "No authoritative hint",
                                "inputSchema": {},
                            },
                        ]
                    },
                )
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
            "arcana.mcp.client._create_transport",
            return_value=mock_transport,
        ):
            config = MCPServerConfig(name="srv", command="echo")
            client = await setup_mcp_tools(
                [config],
                registry,
                trace_writer=trace_writer,
            )

            assert "srv.old_tool" in registry.list_tools()
            assert "srv.local_shadow" in registry.list_tools()

            result = await client.call_tool("srv", "old_tool", {})
            assert result == "done"

            names = registry.list_tools()
            assert "srv.old_tool" not in names
            assert "srv.new_read" in names
            assert "srv.new_unknown" in names
            assert "srv.local_shadow" in names

            read_spec = registry.get("srv.new_read").spec
            assert read_spec.side_effect == SideEffect.READ
            assert read_spec.requires_confirmation is False
            assert read_spec.provenance is not None
            assert read_spec.provenance.server_name == "srv"

            unknown_spec = registry.get("srv.new_unknown").spec
            assert unknown_spec.side_effect == SideEffect.WRITE
            assert unknown_spec.requires_confirmation is True

            run_ids = trace_writer.list_runs()
            assert len(run_ids) == 1
            events = [
                json.loads(line)
                for line in (tmp_path / f"{run_ids[0]}.jsonl").read_text().splitlines()
            ]
            admissions = [
                e
                for e in events
                if e["event_type"] == EventType.CAPABILITY_ADMISSION.value
            ]
            metadata_by_tool = {
                e["metadata"]["tool_name"]: e["metadata"] for e in admissions
            }

            assert set(metadata_by_tool) == {
                "srv.old_tool",
                "srv.new_read",
                "srv.new_unknown",
            }
            assert metadata_by_tool["srv.new_read"]["decision"] == "admitted"
            assert metadata_by_tool["srv.new_unknown"]["decision"] == "downgraded"
            assert (
                metadata_by_tool["srv.new_unknown"]["side_effect_basis"] == "inferred"
            )
            assert all(
                len(metadata["capability_digest"]) == 16
                for metadata in metadata_by_tool.values()
            )

            await client.disconnect_all()
