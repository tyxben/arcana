"""Constitutional invariant tests.

These tests pin architectural promises from ``CONSTITUTION.md`` against the
implementation. Each test class targets one Principle or Inviolable Rule.
A failure here means the constitution is no longer being honored — fix the
code, not the test.

Invariants covered:

- **Side-effect aware tool dispatch** (Principle 3 + 6) — write tools must
  serialize when dispatched in batch; the runtime, not the LLM, owns this
  safety boundary.
- **Cognitive primitives are opt-in** (Principle 9 + Chapter IV Inviolable
  Rule) — default ``Runtime`` does not auto-expose ``recall``/``pin``/
  ``unpin`` to the LLM.
- **ask_user never blocks the user** (Principle 8 + Chapter IV) — without
  an ``input_handler``, the handler returns a fallback message rather than
  awaiting indefinitely.
- **Structured output coexists with tools** (Principle 6) — setting
  ``response_format_schema`` does not disable the tool surface.

Future invariants (not yet pinned, tracked as work):

- Pinned content survives ``WorkingSetBuilder`` compression at L0 fidelity.
- Final-answer detection does not depend on forced-output markers.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from arcana.contracts.tool import (
    ASK_USER_TOOL_NAME,
    SideEffect,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from arcana.runtime.ask_user import AskUserHandler
from arcana.runtime.cognitive import (
    PIN_TOOL_NAME,
    RECALL_TOOL_NAME,
    UNPIN_TOOL_NAME,
)
from arcana.runtime.conversation import ConversationAgent
from arcana.tool_gateway.base import ToolProvider
from arcana.tool_gateway.gateway import ToolGateway
from arcana.tool_gateway.local_channel import LocalChannel
from arcana.tool_gateway.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DelayedTool(ToolProvider):
    """Tool that records its execution start/end timestamps."""

    def __init__(self, name: str, *, side_effect: SideEffect, delay: float) -> None:
        self._name = name
        self._side_effect = side_effect
        self._delay = delay
        self.starts: list[float] = []
        self.ends: list[float] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=f"{self._name} tool ({self._side_effect.value})",
            input_schema={"type": "object", "properties": {}},
            side_effect=self._side_effect,
            max_retries=0,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        self.starts.append(time.monotonic())
        await asyncio.sleep(self._delay)
        self.ends.append(time.monotonic())
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output="ok",
        )


async def _auto_confirm(call: ToolCall, spec: ToolSpec) -> bool:
    """Auto-approve write tools in tests; the invariant under test is the
    *dispatch ordering*, not the confirmation gate itself."""
    return True


def _gateway_with(*providers: ToolProvider) -> ToolGateway:
    registry = ToolRegistry()
    for p in providers:
        registry.register(p)
    return ToolGateway(registry=registry, confirmation_callback=_auto_confirm)


def _agent(**overrides: object) -> ConversationAgent:
    """Build a ConversationAgent with the minimal arg set tests need.

    The agent is never run — these tests only inspect schema-shaped state.
    Passing ``gateway=None`` is fine because no LLM call is made.
    """
    return ConversationAgent(gateway=None, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Principle 3 + 6 — side-effect aware tool dispatch
# ---------------------------------------------------------------------------


class TestSideEffectDispatch:
    """Write tools must serialize; read tools may run concurrently."""

    @pytest.mark.asyncio
    async def test_write_tools_serialize_in_call_many(self) -> None:
        delay = 0.1
        write_tool = _DelayedTool("w", side_effect=SideEffect.WRITE, delay=delay)
        gw = _gateway_with(write_tool)

        calls = [ToolCall(id=f"w{i}", name="w", arguments={}) for i in range(3)]
        results = await gw.call_many(calls)

        assert all(r.success for r in results)
        # Three serial writes must finish AFTER the previous one started.
        # Prove non-overlap: start[i+1] >= end[i] (modulo a small jitter).
        for i in range(len(write_tool.starts) - 1):
            assert write_tool.starts[i + 1] >= write_tool.ends[i] - 0.01, (
                f"Write tools overlapped: start[{i + 1}]={write_tool.starts[i + 1]:.3f} "
                f"end[{i}]={write_tool.ends[i]:.3f}"
            )

    @pytest.mark.asyncio
    async def test_read_tools_run_concurrently_in_call_many(self) -> None:
        delay = 0.15
        read_tool = _DelayedTool("r", side_effect=SideEffect.READ, delay=delay)
        gw = _gateway_with(read_tool)

        calls = [ToolCall(id=f"r{i}", name="r", arguments={}) for i in range(3)]
        start = time.monotonic()
        results = await gw.call_many(calls)
        elapsed = time.monotonic() - start

        assert all(r.success for r in results)
        # Concurrent: ~delay total. Sequential would be ~3*delay.
        assert elapsed < delay * 2, (
            f"Read tools serialized: {elapsed:.2f}s for 3x{delay}s tools"
        )

    @pytest.mark.asyncio
    async def test_local_channel_uses_side_effect_aware_dispatch(self) -> None:
        """LocalChannel.execute_many must route through call_many, not call_many_concurrent.

        This is the regression guard for the runtime's default execution
        path — ConversationAgent dispatches batched tool calls through
        LocalChannel, so the channel's choice of dispatcher is the actual
        boundary the constitution rests on.
        """
        delay = 0.1
        write_tool = _DelayedTool("w", side_effect=SideEffect.WRITE, delay=delay)
        gw = _gateway_with(write_tool)
        channel = LocalChannel(gw)

        calls = [ToolCall(id=f"w{i}", name="w", arguments={}) for i in range(3)]
        await channel.execute_many(calls)

        for i in range(len(write_tool.starts) - 1):
            assert write_tool.starts[i + 1] >= write_tool.ends[i] - 0.01, (
                "LocalChannel.execute_many ran write tools concurrently"
            )


# ---------------------------------------------------------------------------
# Principle 9 — cognitive primitives are opt-in
# ---------------------------------------------------------------------------


class TestCognitivePrimitivesOptIn:
    """Default Runtime must not expose recall/pin/unpin to the LLM."""

    def test_default_agent_does_not_expose_cognitive_tools(self) -> None:
        agent = _agent()

        tool_names = {
            t["function"]["name"] for t in (agent._get_current_tools() or [])
        }

        assert RECALL_TOOL_NAME not in tool_names
        assert PIN_TOOL_NAME not in tool_names
        assert UNPIN_TOOL_NAME not in tool_names
        # ask_user is the only built-in always exposed (Principle 8).
        assert ASK_USER_TOOL_NAME in tool_names

    def test_opting_in_to_pin_exposes_pin_and_unpin(self) -> None:
        agent = _agent(cognitive_primitives=["pin"])

        tool_names = {
            t["function"]["name"] for t in (agent._get_current_tools() or [])
        }

        assert PIN_TOOL_NAME in tool_names
        # unpin rides with pin — they are a symmetric pair.
        assert UNPIN_TOOL_NAME in tool_names
        # recall is independent: opting into pin does not enable recall.
        assert RECALL_TOOL_NAME not in tool_names


# ---------------------------------------------------------------------------
# Principle 8 + Chapter IV — ask_user never blocks the user
# ---------------------------------------------------------------------------


class TestAskUserNeverBlocks:
    """Without an input_handler, ask_user falls back synchronously."""

    @pytest.mark.asyncio
    async def test_ask_user_without_handler_returns_fallback(self) -> None:
        handler = AskUserHandler(input_handler=None)

        # Wrap in wait_for to assert non-blocking even with no answer source.
        result = await asyncio.wait_for(handler.handle("anything?"), timeout=0.5)

        assert isinstance(result, str)
        assert result  # non-empty
        # The exact fallback wording is internal; the invariant is that the
        # LLM gets *some* string and is not stuck. Spot-check a stable token.
        assert "proceed" in result.lower()


# ---------------------------------------------------------------------------
# Principle 6 — structured output does not disable tools
# ---------------------------------------------------------------------------


class TestStructuredOutputCoexistsWithTools:
    """response_format must not strip the tool surface from the LLM."""

    def test_tools_remain_when_response_format_is_set(self) -> None:
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        agent = _agent(response_format_schema=schema)

        tools = agent._get_current_tools()
        assert tools is not None
        names = {t["function"]["name"] for t in tools}
        # The runtime never silently disables tools when structured output
        # is requested. ask_user is the always-on built-in; its presence
        # proves the tool list survived.
        assert ASK_USER_TOOL_NAME in names
