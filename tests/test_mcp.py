"""Tests for MCP module — protocol, client, tool provider, setup."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arcana.contracts.mcp import (
    MCPError,
    MCPMessage,
    MCPServerConfig,
    MCPToolSpec,
    MCPTransportType,
)
from arcana.contracts.tool import SideEffect, ToolCall, ToolErrorCategory, ToolSpec
from arcana.mcp.protocol import (
    _infer_side_effect,
    arcana_spec_to_mcp_tool,
    deserialize_message,
    make_error_response,
    make_request,
    make_response,
    mcp_error_to_tool_error,
    mcp_tool_to_arcana_spec,
    serialize_message,
)


class TestMCPContracts:
    def test_server_config_defaults(self):
        c = MCPServerConfig(name="test")
        assert c.transport == MCPTransportType.STDIO
        assert c.timeout_ms == 30000

    def test_server_config_stdio(self):
        c = MCPServerConfig(
            name="fs",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        assert c.command == "npx"

    def test_tool_spec(self):
        t = MCPToolSpec(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object"},
        )
        assert t.name == "read_file"

    def test_message(self):
        m = MCPMessage(id=1, method="tools/list", params={})
        assert m.jsonrpc == "2.0"

    def test_error(self):
        e = MCPError(code=-32600, message="Invalid request")
        assert e.code == -32600


class TestProtocol:
    def test_serialize_deserialize(self):
        msg = MCPMessage(id=1, method="test", params={"key": "value"})
        data = serialize_message(msg)
        assert isinstance(data, bytes)
        restored = deserialize_message(data)
        assert restored.method == "test"
        assert restored.id == 1

    def test_make_request(self):
        msg = make_request("tools/list", {"cursor": None}, msg_id=42)
        assert msg.method == "tools/list"
        assert msg.id == 42

    def test_make_response(self):
        msg = make_response(42, {"tools": []})
        assert msg.id == 42
        assert msg.result == {"tools": []}

    def test_make_error_response(self):
        msg = make_error_response(42, -32601, "Method not found")
        assert msg.error.code == -32601

    def test_mcp_to_arcana_spec(self):
        mcp = MCPToolSpec(
            name="read_file",
            description="Read",
            input_schema={"type": "object"},
        )
        spec = mcp_tool_to_arcana_spec(mcp, server_name="fs")
        assert spec.name == "fs.read_file"
        assert "Read" in spec.description

    def test_mcp_to_arcana_with_capability(self):
        mcp = MCPToolSpec(name="write", input_schema={})
        spec = mcp_tool_to_arcana_spec(mcp, "fs", capability_prefix="mcp.fs")
        assert "mcp.fs.write" in spec.capabilities

    def test_arcana_to_mcp(self):
        spec = ToolSpec(
            name="search", description="Search", input_schema={"type": "object"}
        )
        mcp = arcana_spec_to_mcp_tool(spec)
        assert mcp.name == "search"

    def test_infer_side_effect_read(self):
        mcp = MCPToolSpec(
            name="read_file", description="Read a file", input_schema={}
        )
        assert _infer_side_effect(mcp) == SideEffect.READ

    def test_infer_side_effect_write(self):
        mcp = MCPToolSpec(
            name="write_file",
            description="Write content to a file",
            input_schema={},
        )
        assert _infer_side_effect(mcp) == SideEffect.WRITE

    def test_error_mapping_retryable(self):
        err = MCPError(code=-32603, message="Internal error")
        tool_err = mcp_error_to_tool_error(err)
        assert tool_err.category == ToolErrorCategory.TRANSPORT
        assert tool_err.is_retryable

    def test_error_mapping_non_retryable(self):
        err = MCPError(code=-32600, message="Invalid request")
        tool_err = mcp_error_to_tool_error(err)
        assert tool_err.category == ToolErrorCategory.VALIDATION
        assert not tool_err.is_retryable


class TestMCPToolProvider:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        from arcana.mcp.tool_provider import MCPToolProvider

        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(return_value="file contents here")
        mock_client.connected_servers = ["fs"]

        spec = ToolSpec(name="fs.read_file", description="Read", input_schema={})
        provider = MCPToolProvider(
            client=mock_client,
            server_name="fs",
            mcp_tool_name="read_file",
            arcana_spec=spec,
        )

        call = ToolCall(id="1", name="fs.read_file", arguments={"path": "/tmp/test"})
        result = await provider.execute(call)

        assert result.success
        assert result.output == "file contents here"
        mock_client.call_tool.assert_called_once_with(
            "fs", "read_file", {"path": "/tmp/test"}
        )

    @pytest.mark.asyncio
    async def test_execute_error(self):
        from arcana.mcp.tool_provider import MCPToolProvider

        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(side_effect=Exception("Connection lost"))
        mock_client.connected_servers = ["fs"]

        spec = ToolSpec(name="fs.read", description="Read", input_schema={})
        provider = MCPToolProvider(
            client=mock_client,
            server_name="fs",
            mcp_tool_name="read",
            arcana_spec=spec,
        )

        call = ToolCall(id="1", name="fs.read", arguments={})
        result = await provider.execute(call)
        assert not result.success
        assert "Connection lost" in result.error.message

    @pytest.mark.asyncio
    async def test_health_check(self):
        from arcana.mcp.tool_provider import MCPToolProvider

        mock_client = MagicMock()
        mock_client.connected_servers = ["fs"]

        spec = ToolSpec(name="fs.x", description="X", input_schema={})
        provider = MCPToolProvider(mock_client, "fs", "x", spec)
        assert await provider.health_check()

        mock_client.connected_servers = []
        assert not await provider.health_check()


def _make_mock_transport(responses: list[MCPMessage]) -> AsyncMock:
    """Create a mock transport that returns the given responses in sequence."""
    mock_transport = AsyncMock()
    mock_transport.is_connected = True
    mock_transport.connect = AsyncMock()
    mock_transport.send = AsyncMock()
    mock_transport.receive = AsyncMock(side_effect=responses)
    mock_transport.close = AsyncMock()
    return mock_transport


class TestMCPClient:
    @pytest.mark.asyncio
    async def test_connect_and_discover(self):
        from arcana.mcp.client import MCPClient

        # Response to initialize request (id=1)
        init_response = MCPMessage(id=1, result={"capabilities": {}})
        # Response to tools/list request (id=2)
        tools_response = MCPMessage(
            id=2,
            result={
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "write_file",
                        "description": "Write a file",
                        "inputSchema": {"type": "object"},
                    },
                ]
            },
        )
        mock_transport = _make_mock_transport([init_response, tools_response])

        with patch(
            "arcana.mcp.client._create_transport", return_value=mock_transport
        ):
            client = MCPClient()
            config = MCPServerConfig(name="test", command="echo")
            tools = await client.connect(config)

            assert len(tools) == 2
            assert tools[0].name == "read_file"
            assert "test" in client.connected_servers

            await client.disconnect_all()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        from arcana.mcp.client import MCPClient

        client = MCPClient()
        # No connection -- disconnect should not raise
        await client.disconnect("nonexistent")
        await client.disconnect_all()


class TestMCPSetup:
    @pytest.mark.asyncio
    async def test_setup_registers_tools(self):
        from arcana.mcp.setup import setup_mcp_tools
        from arcana.tool_gateway.registry import ToolRegistry

        # Response to initialize request (id=1)
        init_response = MCPMessage(id=1, result={})
        # Response to tools/list request (id=2)
        tools_response = MCPMessage(
            id=2,
            result={
                "tools": [
                    {
                        "name": "search",
                        "description": "Search",
                        "inputSchema": {"type": "object"},
                    }
                ]
            },
        )
        mock_transport = _make_mock_transport([init_response, tools_response])

        registry = ToolRegistry()
        config = MCPServerConfig(name="test", command="echo")

        with patch(
            "arcana.mcp.client._create_transport", return_value=mock_transport
        ):
            client = await setup_mcp_tools([config], registry)

            assert "test.search" in registry.list_tools()
            assert "test" in client.connected_servers

            await client.disconnect_all()
