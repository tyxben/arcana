"""Tests for V2 lifecycle (observer) hooks.

The V2 ConversationAgent notifies observer hooks registered via
``Runtime.on(name, callback)`` at turn/tool boundaries (plus run-level events
on the chat path). Observers cannot block or rewrite the run, and a raising
observer is swallowed (fail open).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

import arcana
from arcana.contracts.lifecycle import (
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from arcana.contracts.llm import LLMResponse, TokenUsage, ToolCallRequest
from arcana.runtime_core import Runtime, RuntimeConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(t: str, pt: int = 10, ct: int = 20) -> LLMResponse:
    return LLMResponse(
        content=t,
        tool_calls=None,
        usage=TokenUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
        model="m",
        finish_reason="stop",
    )


def _call(name: str, arguments: str, cid: str = "tc-1") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=cid, name=name, arguments=arguments)],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="m",
        finish_reason="tool_calls",
    )


def _runtime(tools=None) -> Runtime:
    return Runtime(
        providers={"ollama": ""},
        tools=tools,
        config=RuntimeConfig(default_provider="ollama"),
    )


def _record(rt: Runtime, names: list[str]) -> list[tuple]:
    seen: list[tuple] = []

    def mk(name: str):
        def handler(**k):
            ev = k.get("event")
            seen.append((name, ev))
        return handler

    for n in names:
        rt.on(n, mk(n))
    return seen


# ---------------------------------------------------------------------------
# Payload contract
# ---------------------------------------------------------------------------


class TestLifecyclePayloads:
    def test_events_are_frozen(self):
        ev = TurnStartEvent(run_id="r", turn=1)
        with pytest.raises(ValidationError):
            ev.turn = 2

    def test_payload_fields(self):
        te = TurnEndEvent(
            run_id="r", turn=2, turn_tokens=30, turn_cost_usd=0.01,
            tool_calls_made=1, completed=True,
        )
        assert te.turn == 2
        assert te.tool_calls_made == 1
        assert te.completed is True
        ts = ToolStartEvent(run_id="r", turn=1, tool_name="x", tool_call_id="c1")
        assert ts.tool_name == "x"
        end = ToolEndEvent(run_id="r", turn=1, tool_name="x", success=False, error="boom")
        assert end.success is False
        assert end.error == "boom"


# ---------------------------------------------------------------------------
# Emission at boundaries
# ---------------------------------------------------------------------------


class TestEmissionBoundaries:
    @pytest.mark.asyncio
    async def test_turn_and_tool_events_fire(self):
        @arcana.tool(side_effect="read")
        async def look(x: str) -> str:
            return "data"

        rt = _runtime(tools=[look])
        rt._gateway.generate = AsyncMock(
            side_effect=[_call("look", '{"x": "a"}'), _text("done")]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        seen = _record(rt, ["turn_start", "turn_end", "tool_start", "tool_end"])
        async with rt.chat() as c:
            await c.send("use the tool")

        names = [s[0] for s in seen]
        # First turn: start, tool_start, tool_end, end; second turn: start, end.
        assert names == [
            "turn_start", "tool_start", "tool_end", "turn_end",
            "turn_start", "turn_end",
        ]

    @pytest.mark.asyncio
    async def test_typed_payloads_delivered(self):
        @arcana.tool(side_effect="read")
        async def look(x: str) -> str:
            return "data"

        rt = _runtime(tools=[look])
        rt._gateway.generate = AsyncMock(
            side_effect=[_call("look", '{"x": "a"}'), _text("done")]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        seen = _record(rt, ["turn_start", "tool_start", "tool_end", "turn_end"])
        async with rt.chat() as c:
            await c.send("go")

        by_name: dict[str, object] = {}
        for name, ev in seen:  # keep first occurrence of each event
            by_name.setdefault(name, ev)
        assert isinstance(by_name["turn_start"], TurnStartEvent)
        assert by_name["turn_start"].turn == 1
        assert isinstance(by_name["tool_start"], ToolStartEvent)
        assert by_name["tool_start"].tool_name == "look"
        assert by_name["tool_start"].tool_call_id == "tc-1"
        assert isinstance(by_name["tool_end"], ToolEndEvent)
        assert by_name["tool_end"].success is True
        assert isinstance(by_name["turn_end"], TurnEndEvent)

    @pytest.mark.asyncio
    async def test_tool_end_reports_failure(self):
        @arcana.tool(side_effect="read")
        async def boom(x: str) -> str:
            raise RuntimeError("kaboom")

        rt = _runtime(tools=[boom])
        rt._gateway.generate = AsyncMock(
            side_effect=[_call("boom", '{"x": "a"}'), _text("handled")]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        seen = _record(rt, ["tool_end"])
        async with rt.chat() as c:
            await c.send("go")

        tool_end = seen[0][1]
        assert tool_end.success is False
        assert tool_end.error

    @pytest.mark.asyncio
    async def test_direct_answer_path_emits_one_turn(self):
        """The direct-answer fast path is one logical completed turn."""
        rt = _runtime()
        rt._gateway.generate = AsyncMock(return_value=_text("hi there"))
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        seen = _record(rt, ["turn_start", "turn_end"])
        # run() goes through the intent classifier -> direct answer for "hi".
        await rt.run("hi")

        names = [s[0] for s in seen]
        assert names == ["turn_start", "turn_end"]
        assert seen[1][1].completed is True
        assert seen[1][1].tool_calls_made == 0


# ---------------------------------------------------------------------------
# Run-level events on the chat path
# ---------------------------------------------------------------------------


class TestChatRunLevelEvents:
    @pytest.mark.asyncio
    async def test_run_start_end_fire_on_chat(self):
        rt = _runtime()
        rt._gateway.generate = AsyncMock(return_value=_text("ok"))
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        order: list[str] = []
        rt.on("run_start", lambda **k: order.append("run_start"))
        rt.on("run_end", lambda **k: order.append("run_end"))

        async with rt.chat() as c:
            await c.send("hello")

        assert order == ["run_start", "run_end"]

    @pytest.mark.asyncio
    async def test_error_event_fires_on_chat_failure(self):
        rt = _runtime()
        rt._gateway.generate = AsyncMock(side_effect=RuntimeError("provider down"))
        rt._gateway.stream = MagicMock(side_effect=RuntimeError("provider down"))

        errors: list[Exception] = []
        rt.on("error", lambda **k: errors.append(k.get("error")))

        with pytest.raises(RuntimeError):
            async with rt.chat() as c:
                await c.send("hello")

        assert errors  # the error event fired before the exception propagated


# ---------------------------------------------------------------------------
# Fail-open + observer-only
# ---------------------------------------------------------------------------


class TestObserverFailOpen:
    @pytest.mark.asyncio
    async def test_raising_observer_does_not_crash_run(self):
        rt = _runtime()
        rt._gateway.generate = AsyncMock(return_value=_text("answer"))
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        def boom(**k):
            raise RuntimeError("observer blew up")

        rt.on("turn_start", boom)
        rt.on("turn_end", boom)
        rt.on("run_start", boom)
        rt.on("run_end", boom)

        # The run completes despite every observer raising.
        async with rt.chat() as c:
            r = await c.send("hello")
        assert r.content == "answer"

    @pytest.mark.asyncio
    async def test_observer_cannot_alter_control_flow(self):
        """An observer's return value is ignored — no control channel."""
        ran = {"look": False}

        @arcana.tool(side_effect="read")
        async def look(x: str) -> str:
            ran["look"] = True
            return "data"

        rt = _runtime(tools=[look])
        rt._gateway.generate = AsyncMock(
            side_effect=[_call("look", '{"x": "a"}'), _text("done")]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        # A tool_start observer that "tries" to block by returning False —
        # the return value has no channel to influence execution.
        rt.on("tool_start", lambda **k: False)
        seen = _record(rt, ["tool_end"])

        async with rt.chat() as c:
            await c.send("go")

        # The tool still executed and tool_end reports success: the observer's
        # return value changed nothing.
        assert ran["look"] is True
        assert seen[0][1].success is True
