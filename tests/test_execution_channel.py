"""Tests for ExecutionChannel protocol and LocalChannel implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from arcana.contracts.channel import ExecutionChannel
from arcana.contracts.tool import ToolCall, ToolResult
from arcana.tool_gateway.local_channel import LocalChannel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tool_call(name: str = "test_tool", call_id: str = "tc-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={"key": "value"})


def _make_tool_result(call_id: str = "tc-1", name: str = "test_tool") -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        name=name,
        success=True,
        output="ok",
    )


class FakeGateway:
    """Minimal fake that records calls to verify delegation."""

    def __init__(self) -> None:
        self.call_log: list[tuple[str, ToolCall]] = []
        self.call_many_log: list[tuple[str, list[ToolCall]]] = []

    async def call(
        self,
        tool_call: ToolCall,
        *,
        trace_ctx: object | None = None,
    ) -> ToolResult:
        self.call_log.append(("call", tool_call))
        return _make_tool_result(call_id=tool_call.id, name=tool_call.name)

    async def call_many_concurrent(
        self,
        tool_calls: list[ToolCall],
        *,
        trace_ctx: object | None = None,
    ) -> list[ToolResult]:
        self.call_many_log.append(("call_many_concurrent", tool_calls))
        return [
            _make_tool_result(call_id=tc.id, name=tc.name) for tc in tool_calls
        ]


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestExecutionChannelProtocol:
    """Verify ExecutionChannel is a runtime-checkable protocol."""

    def test_local_channel_is_execution_channel(self) -> None:
        gw = FakeGateway()
        ch = LocalChannel(gateway=gw)  # type: ignore[arg-type]
        assert isinstance(ch, ExecutionChannel)

    def test_custom_channel_satisfies_protocol(self) -> None:
        """A plain class with the right methods satisfies the protocol."""

        class CustomChannel:
            async def execute(self, call: ToolCall) -> ToolResult:
                return _make_tool_result()

            async def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
                return []

            async def close(self) -> None:
                pass

        assert isinstance(CustomChannel(), ExecutionChannel)

    def test_incomplete_class_does_not_satisfy_protocol(self) -> None:
        """A class missing methods does NOT satisfy the protocol."""

        class Incomplete:
            async def execute(self, call: ToolCall) -> ToolResult:
                return _make_tool_result()
            # Missing execute_many and close

        assert not isinstance(Incomplete(), ExecutionChannel)


# ---------------------------------------------------------------------------
# LocalChannel delegation
# ---------------------------------------------------------------------------

class TestLocalChannelExecute:
    """LocalChannel.execute delegates to ToolGateway.call."""

    @pytest.mark.asyncio
    async def test_execute_delegates_to_gateway_call(self) -> None:
        gw = FakeGateway()
        ch = LocalChannel(gateway=gw)  # type: ignore[arg-type]
        tc = _make_tool_call()

        result = await ch.execute(tc)

        assert result.success is True
        assert result.tool_call_id == tc.id
        assert len(gw.call_log) == 1
        assert gw.call_log[0] == ("call", tc)

    @pytest.mark.asyncio
    async def test_execute_passes_trace_ctx(self) -> None:
        """trace_ctx given at construction is forwarded to gateway.call."""

        class CtxCapturingGateway(FakeGateway):
            def __init__(self) -> None:
                super().__init__()
                self.captured_ctx: object | None = None

            async def call(
                self,
                tool_call: ToolCall,
                *,
                trace_ctx: object | None = None,
            ) -> ToolResult:
                self.captured_ctx = trace_ctx
                return await super().call(tool_call, trace_ctx=trace_ctx)

        sentinel = object()
        gw = CtxCapturingGateway()
        ch = LocalChannel(gateway=gw, trace_ctx=sentinel)  # type: ignore[arg-type]

        await ch.execute(_make_tool_call())
        assert gw.captured_ctx is sentinel


class TestLocalChannelExecuteMany:
    """LocalChannel.execute_many delegates to ToolGateway.call_many_concurrent."""

    @pytest.mark.asyncio
    async def test_execute_many_delegates(self) -> None:
        gw = FakeGateway()
        ch = LocalChannel(gateway=gw)  # type: ignore[arg-type]
        calls = [_make_tool_call(call_id="tc-1"), _make_tool_call(call_id="tc-2")]

        results = await ch.execute_many(calls)

        assert len(results) == 2
        assert results[0].tool_call_id == "tc-1"
        assert results[1].tool_call_id == "tc-2"
        assert len(gw.call_many_log) == 1
        assert gw.call_many_log[0] == ("call_many_concurrent", calls)

    @pytest.mark.asyncio
    async def test_execute_many_empty_list(self) -> None:
        gw = FakeGateway()
        ch = LocalChannel(gateway=gw)  # type: ignore[arg-type]

        results = await ch.execute_many([])

        assert results == []
        assert len(gw.call_many_log) == 1


class TestLocalChannelClose:
    """LocalChannel.close delegates to gateway."""

    @pytest.mark.asyncio
    async def test_close_delegates_to_gateway(self) -> None:
        class CloseTrackingGateway(FakeGateway):
            def __init__(self) -> None:
                super().__init__()
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        gw = CloseTrackingGateway()
        ch = LocalChannel(gateway=gw)  # type: ignore[arg-type]
        await ch.close()
        assert gw.closed


# ---------------------------------------------------------------------------
# Custom channel (mock that records calls)
# ---------------------------------------------------------------------------

class TestCustomRecordingChannel:
    """A custom channel implementation that records all calls."""

    @pytest.mark.asyncio
    async def test_recording_channel(self) -> None:
        class RecordingChannel:
            def __init__(self) -> None:
                self.executed: list[ToolCall] = []
                self.closed = False

            async def execute(self, call: ToolCall) -> ToolResult:
                self.executed.append(call)
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    success=True,
                    output="recorded",
                )

            async def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
                self.executed.extend(calls)
                return [
                    ToolResult(
                        tool_call_id=c.id,
                        name=c.name,
                        success=True,
                        output="recorded",
                    )
                    for c in calls
                ]

            async def close(self) -> None:
                self.closed = True

        ch = RecordingChannel()
        assert isinstance(ch, ExecutionChannel)

        # Single execute
        tc1 = _make_tool_call(call_id="r-1")
        r1 = await ch.execute(tc1)
        assert r1.output == "recorded"
        assert len(ch.executed) == 1

        # Batch execute
        tc2 = _make_tool_call(call_id="r-2")
        tc3 = _make_tool_call(call_id="r-3")
        results = await ch.execute_many([tc2, tc3])
        assert len(results) == 2
        assert len(ch.executed) == 3

        # Close
        await ch.close()
        assert ch.closed is True


# ---------------------------------------------------------------------------
# ConversationAgent routes through channel when provided
# ---------------------------------------------------------------------------

class TestConversationAgentChannelRouting:
    """Verify _execute_tools prefers channel over tool_gateway."""

    def _make_agent(self, *, channel=None, tool_gateway=None, input_handler=None):
        from arcana.runtime.conversation import ConversationAgent
        return ConversationAgent(
            gateway=AsyncMock(),
            model_config=AsyncMock(),
            tool_gateway=tool_gateway,
            budget_tracker=AsyncMock(),
            channel=channel,
            input_handler=input_handler,
        )

    @pytest.mark.asyncio
    async def test_execute_tools_uses_channel(self) -> None:
        """When a channel is provided, _execute_tools routes through it."""
        from arcana.contracts.llm import ToolCallRequest

        channel_result = ToolResult(
            tool_call_id="tc-1",
            name="my_tool",
            success=True,
            output="from-channel",
        )

        class StubChannel:
            def __init__(self) -> None:
                self.called = False

            async def execute(self, call: ToolCall) -> ToolResult:
                return channel_result

            async def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
                self.called = True
                return [channel_result] * len(calls)

            async def close(self) -> None:
                pass

        stub = StubChannel()
        agent = self._make_agent(channel=stub)

        tool_calls = [
            ToolCallRequest(id="tc-1", name="my_tool", arguments='{"x": 1}'),
        ]
        results, events = await agent._execute_tools(tool_calls)

        assert stub.called
        assert len(results) == 1
        assert results[0].output == "from-channel"

    @pytest.mark.asyncio
    async def test_execute_tools_falls_back_to_gateway(self) -> None:
        """Without a channel, _execute_tools uses tool_gateway as before."""
        from arcana.contracts.llm import ToolCallRequest

        gw_result = ToolResult(
            tool_call_id="tc-1",
            name="my_tool",
            success=True,
            output="from-gateway",
        )

        fake_gw = AsyncMock()
        fake_gw.call_many_concurrent = AsyncMock(return_value=[gw_result])
        agent = self._make_agent(tool_gateway=fake_gw)

        tool_calls = [
            ToolCallRequest(id="tc-1", name="my_tool", arguments='{"x": 1}'),
        ]
        results, events = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert results[0].output == "from-gateway"
        fake_gw.call_many_concurrent.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_channel_no_gateway_returns_errors(self) -> None:
        """With neither channel nor gateway, tools get synthetic errors."""
        from arcana.contracts.llm import ToolCallRequest

        agent = self._make_agent(channel=None, tool_gateway=None)

        tool_calls = [
            ToolCallRequest(id="tc-1", name="my_tool", arguments='{"x": 1}'),
        ]
        results, events = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert results[0].success is False
        assert "cannot be executed" in results[0].output

    @pytest.mark.asyncio
    async def test_channel_exception_propagates(self) -> None:
        """If channel.execute_many raises, the exception propagates."""
        from arcana.contracts.llm import ToolCallRequest

        class BrokenChannel:
            async def execute(self, call: ToolCall) -> ToolResult:
                raise ConnectionError("remote down")

            async def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
                raise ConnectionError("remote down")

            async def close(self) -> None:
                pass

        agent = self._make_agent(channel=BrokenChannel())

        tool_calls = [
            ToolCallRequest(id="tc-1", name="my_tool", arguments='{"x": 1}'),
        ]
        with pytest.raises(ConnectionError, match="remote down"):
            await agent._execute_tools(tool_calls)

    @pytest.mark.asyncio
    async def test_ask_user_bypasses_channel(self) -> None:
        """ask_user calls go through handler, not channel, even with channel set."""
        from arcana.contracts.llm import ToolCallRequest

        class SpyChannel:
            def __init__(self) -> None:
                self.called = False

            async def execute(self, call: ToolCall) -> ToolResult:
                self.called = True
                return ToolResult(tool_call_id=call.id, name=call.name, success=True, output="nope")

            async def execute_many(self, calls: list[ToolCall]) -> list[ToolResult]:
                self.called = True
                return []

            async def close(self) -> None:
                pass

        spy = SpyChannel()
        agent = self._make_agent(
            channel=spy,
            input_handler=lambda q: "user answer",
        )

        tool_calls = [
            ToolCallRequest(id="tc-1", name="ask_user", arguments='{"question": "hi?"}'),
        ]
        results, events = await agent._execute_tools(tool_calls)

        assert not spy.called  # channel was NOT used for ask_user
        assert len(results) == 1
        assert results[0].name == "ask_user"
        assert results[0].output == "user answer"
