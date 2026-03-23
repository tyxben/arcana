"""Tests for MCP auto-reconnect on connection loss (Task 1.4)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from arcana.contracts.mcp import MCPMessage, MCPServerConfig


def _make_mock_transport(responses: list[MCPMessage]) -> AsyncMock:
    """Create a mock transport with given response sequence."""
    mock = AsyncMock()
    mock.is_connected = True
    mock.connect = AsyncMock()
    mock.send = AsyncMock()
    mock.receive = AsyncMock(side_effect=responses)
    mock.close = AsyncMock()
    return mock


class TestMCPReconnect:
    @pytest.mark.asyncio
    async def test_call_tool_reconnects_on_connection_error(self):
        """call_tool should reconnect and retry on ConnectionError."""
        from arcana.mcp.client import MCPConnection

        config = MCPServerConfig(name="test", command="echo", reconnect_attempts=2, reconnect_delay_ms=1)

        # Build responses: init succeeds, then call_tool fails, then reconnect succeeds
        init_response = MCPMessage(id=1, result={"capabilities": {}})

        transport = AsyncMock()
        transport.connect = AsyncMock()
        transport.close = AsyncMock()
        transport.send = AsyncMock()

        call_count = 0

        async def receive_side_effect():
            nonlocal call_count
            call_count += 1
            # First call: init response (id=1)
            if call_count == 1:
                return init_response
            # Second call: simulate connection loss
            if call_count == 2:
                raise ConnectionError("connection lost")
            # Third call: reconnect init response (id=1 again after counter reset)
            if call_count == 3:
                return MCPMessage(id=1, result={"capabilities": {}})
            # Fourth call: retried tool call response
            if call_count == 4:
                return MCPMessage(id=2, result={"content": [{"type": "text", "text": "reconnected"}]})
            return MCPMessage(id=999, result={})

        transport.receive = AsyncMock(side_effect=receive_side_effect)

        conn = MCPConnection(config=config, transport=transport)
        await conn.connect()
        assert conn._connected is True

        result = await conn.call_tool("test_tool", {"arg": "val"})
        assert result == "reconnected"
        assert conn._connected is True

    @pytest.mark.asyncio
    async def test_list_tools_reconnects_on_connection_error(self):
        """list_tools should reconnect and retry on ConnectionError."""
        from arcana.mcp.client import MCPConnection

        config = MCPServerConfig(name="test", command="echo", reconnect_attempts=2, reconnect_delay_ms=1)

        call_count = 0

        async def receive_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MCPMessage(id=1, result={"capabilities": {}})
            if call_count == 2:
                raise ConnectionError("lost")
            if call_count == 3:
                return MCPMessage(id=1, result={"capabilities": {}})
            if call_count == 4:
                return MCPMessage(id=2, result={"tools": [{"name": "t", "inputSchema": {}}]})
            return MCPMessage(id=999, result={})

        transport = AsyncMock()
        transport.connect = AsyncMock()
        transport.close = AsyncMock()
        transport.send = AsyncMock()
        transport.receive = AsyncMock(side_effect=receive_side_effect)

        conn = MCPConnection(config=config, transport=transport)
        await conn.connect()

        tools = await conn.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "t"

    @pytest.mark.asyncio
    async def test_reconnect_all_attempts_fail(self):
        """If all reconnect attempts fail, ConnectionError should propagate."""
        from arcana.mcp.client import MCPConnection

        config = MCPServerConfig(name="test", command="echo", reconnect_attempts=2, reconnect_delay_ms=1)

        call_count = 0

        async def receive_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MCPMessage(id=1, result={"capabilities": {}})
            # All subsequent calls fail
            raise ConnectionError("permanently down")

        transport = AsyncMock()
        transport.connect = AsyncMock(side_effect=[None, ConnectionError("fail"), ConnectionError("fail")])
        transport.close = AsyncMock()
        transport.send = AsyncMock()
        transport.receive = AsyncMock(side_effect=receive_side_effect)

        conn = MCPConnection(config=config, transport=transport)
        await conn.connect()

        with pytest.raises(ConnectionError, match="Failed to reconnect"):
            await conn.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_connected_state_tracking(self):
        """MCPConnection should track _connected state."""
        from arcana.mcp.client import MCPConnection

        config = MCPServerConfig(name="test", command="echo")
        transport = AsyncMock()
        transport.connect = AsyncMock()
        transport.close = AsyncMock()
        transport.send = AsyncMock()
        transport.receive = AsyncMock(
            return_value=MCPMessage(id=1, result={"capabilities": {}})
        )

        conn = MCPConnection(config=config, transport=transport)
        assert conn._connected is False

        await conn.connect()
        assert conn._connected is True

        await conn.disconnect()
        assert conn._connected is False
