"""Tests for the cognitive `pin` / `unpin` primitives (v0.7.0).

Validates:
- Basic pin returns a pin_id and marks content as protected
- unpin removes a pin by pin_id
- Idempotent pin: same content twice returns the same pin_id
- Pinned content survives WorkingSetBuilder compression passes
- Pin budget cap rejects oversize pins via structured PinResult
- until_turn auto-expires the pin (but leaves it in the registry for trace)
- ContextDecision.decisions surfaces pinned blocks with reason="pinned"
- pin + unpin round-trip frees budget
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from arcana.context.builder import WorkingSetBuilder
from arcana.contracts.cognitive import (
    PinRequest,
    PinState,
    UnpinRequest,
    _content_hash,
)
from arcana.contracts.context import ContextStrategy, TokenBudget
from arcana.contracts.llm import (
    Message,
    MessageRole,
    ModelConfig,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from arcana.runtime.cognitive import (
    PIN_TOOL_NAME,
    UNPIN_TOOL_NAME,
    CognitiveHandler,
)
from arcana.runtime.conversation import ConversationAgent

# ---------------------------------------------------------------------------
# Unit tests — CognitiveHandler
# ---------------------------------------------------------------------------


class TestPinBasic:
    def test_pin_returns_pin_id(self) -> None:
        handler = CognitiveHandler(enabled={"pin"})
        result = handler.handle_pin(
            PinRequest(content="critical fact"), current_turn=1,
        )
        assert result.pinned is True
        assert result.pin_id is not None
        assert result.pin_id.startswith("p_")
        assert result.already_pinned is False

    def test_pin_registers_entry(self) -> None:
        handler = CognitiveHandler(enabled={"pin"})
        handler.handle_pin(
            PinRequest(content="critical fact", label="key"),
            current_turn=1,
        )
        assert len(handler.pin_state.entries) == 1
        entry = handler.pin_state.entries[0]
        assert entry.content == "critical fact"
        assert entry.label == "key"
        assert entry.created_turn == 1
        assert entry.content_hash == _content_hash("critical fact")

    def test_pin_empty_content_rejected(self) -> None:
        handler = CognitiveHandler(enabled={"pin"})
        result = handler.handle_pin(
            PinRequest(content=""), current_turn=1,
        )
        assert result.pinned is False
        assert result.reason == "empty_content"


class TestPinIdempotency:
    def test_same_content_returns_same_pin_id(self) -> None:
        handler = CognitiveHandler(enabled={"pin"})
        r1 = handler.handle_pin(
            PinRequest(content="X", label="first"), current_turn=1,
        )
        r2 = handler.handle_pin(
            PinRequest(content="X", label="second_label"), current_turn=2,
        )
        assert r1.pin_id == r2.pin_id
        assert r2.already_pinned is True
        # Still a single entry
        assert len(handler.pin_state.entries) == 1
        # Original label preserved (idempotent no-op)
        assert handler.pin_state.entries[0].label == "first"


class TestUnpin:
    def test_unpin_removes_pin(self) -> None:
        handler = CognitiveHandler(enabled={"pin"})
        pin = handler.handle_pin(
            PinRequest(content="X"), current_turn=1,
        )
        assert pin.pin_id is not None
        unpin = handler.handle_unpin(UnpinRequest(pin_id=pin.pin_id))
        assert unpin.unpinned is True
        assert len(handler.pin_state.entries) == 0

    def test_unpin_unknown_id(self) -> None:
        handler = CognitiveHandler(enabled={"pin"})
        result = handler.handle_unpin(UnpinRequest(pin_id="p_nope"))
        assert result.unpinned is False
        assert result.note is not None

    def test_pin_then_unpin_round_trip_frees_budget(self) -> None:
        # Tight cap so a second distinct pin of the same size would
        # exceed it — until the first is unpinned.
        handler = CognitiveHandler(
            enabled={"pin"},
            pin_budget_fraction=0.5,
            total_token_window=1200,  # cap = 600 tokens
        )
        big = "x" * 2000  # ~501 tokens — within cap alone
        r1 = handler.handle_pin(
            PinRequest(content=big), current_turn=1,
        )
        assert r1.pinned is True
        # A second distinct pin would exceed the cap
        r2_rejected = handler.handle_pin(
            PinRequest(content=big + "y"), current_turn=1,
        )
        assert r2_rejected.pinned is False
        # Unpin the first — budget should free up
        handler.handle_unpin(UnpinRequest(pin_id=r1.pin_id or ""))
        assert handler.pin_state.total_tokens() == 0
        # Now the second pin should fit
        r2 = handler.handle_pin(
            PinRequest(content=big + "y"), current_turn=2,
        )
        assert r2.pinned is True


class TestPinBudgetCap:
    def test_oversize_pin_rejected(self) -> None:
        handler = CognitiveHandler(
            enabled={"pin"},
            pin_budget_fraction=0.5,
            total_token_window=100,   # cap = 50 tokens
        )
        big_content = "x" * 4000  # way over 50 tokens
        result = handler.handle_pin(
            PinRequest(content=big_content), current_turn=1,
        )
        assert result.pinned is False
        assert result.reason == "pin_budget_exceeded"
        assert result.cap == 50
        assert result.requested_tokens is not None and result.requested_tokens > 50
        assert result.suggestion is not None
        # Registry unchanged
        assert handler.pin_state.entries == []

    def test_existing_pin_never_auto_removed(self) -> None:
        """Second pin that would exceed cap must not evict first pin."""
        handler = CognitiveHandler(
            enabled={"pin"},
            pin_budget_fraction=0.5,
            total_token_window=200,   # cap = 100 tokens
        )
        r1 = handler.handle_pin(
            PinRequest(content="y" * 320),  # ~80 tokens
            current_turn=1,
        )
        assert r1.pinned is True
        # Second pin would blow budget
        r2 = handler.handle_pin(
            PinRequest(content="z" * 320),
            current_turn=2,
        )
        assert r2.pinned is False
        # First pin still present
        assert len(handler.pin_state.entries) == 1


class TestPinUntilTurn:
    def test_pin_active_before_expiry(self) -> None:
        state = PinState()
        handler = CognitiveHandler(enabled={"pin"})
        handler.handle_pin(
            PinRequest(content="A", until_turn=5), current_turn=1,
        )
        state = handler.pin_state
        assert len(state.active_at(3)) == 1

    def test_pin_inactive_after_until_turn(self) -> None:
        handler = CognitiveHandler(enabled={"pin"})
        handler.handle_pin(
            PinRequest(content="A", until_turn=5), current_turn=1,
        )
        # After turn 5, pin should no longer be active
        assert len(handler.pin_state.active_at(6)) == 0
        # But the entry is still in the registry for trace continuity
        assert len(handler.pin_state.entries) == 1

    def test_pin_without_until_turn_is_always_active(self) -> None:
        handler = CognitiveHandler(enabled={"pin"})
        handler.handle_pin(
            PinRequest(content="A"), current_turn=1,
        )
        assert len(handler.pin_state.active_at(1)) == 1
        assert len(handler.pin_state.active_at(100)) == 1


# ---------------------------------------------------------------------------
# Builder integration — pinned blocks survive compression
# ---------------------------------------------------------------------------


class TestPinInContextBuilder:
    def test_pin_surfaced_in_decisions(self) -> None:
        builder = WorkingSetBuilder(
            identity="id",
            token_budget=TokenBudget(total_window=128_000),
        )
        state = PinState()
        from arcana.contracts.cognitive import PinEntry

        state.add(
            PinEntry(
                pin_id="p_test",
                content="pinned content",
                content_hash=_content_hash("pinned content"),
                label="key",
                token_count=5,
            )
        )
        builder.set_pin_state(state)
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="hello"),
        ]
        curated = builder.build_conversation_context(messages, turn=1)
        assert builder.last_decision is not None
        # Should have at least one decision with reason="pinned"
        pinned = [
            d for d in builder.last_decision.decisions if d.reason == "pinned"
        ]
        assert len(pinned) == 1
        assert pinned[0].outcome == "kept"
        assert pinned[0].fidelity == "L0"
        # Curated output should contain the pinned content
        all_text = " ".join(
            m.content if isinstance(m.content, str) else "" for m in curated
        )
        assert "pinned content" in all_text

    def test_pin_survives_aggressive_compression(self) -> None:
        """Pinned block must remain even when the compressor drops history."""
        strategy = ContextStrategy(mode="always_compress", aggressive_keep_turns=1)
        builder = WorkingSetBuilder(
            identity="id",
            token_budget=TokenBudget(total_window=2000, response_reserve=100),
            strategy=strategy,
        )
        from arcana.contracts.cognitive import PinEntry

        state = PinState()
        state.add(
            PinEntry(
                pin_id="p_keep",
                content="LOAD_BEARING_CONCLUSION",
                content_hash=_content_hash("LOAD_BEARING_CONCLUSION"),
                label="load bearing",
                token_count=8,
            )
        )
        builder.set_pin_state(state)
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(
                role=MessageRole.USER,
                content="first message " + ("filler " * 100),
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="ok " + ("pad " * 100),
            ),
            Message(role=MessageRole.USER, content="latest"),
        ]
        curated = builder.build_conversation_context(messages, turn=2)
        joined = " ".join(
            m.content if isinstance(m.content, str) else "" for m in curated
        )
        assert "LOAD_BEARING_CONCLUSION" in joined


# ---------------------------------------------------------------------------
# ConversationAgent interception of pin / unpin
# ---------------------------------------------------------------------------


def _make_agent(cognitive_primitives: list[str]) -> ConversationAgent:
    gateway = MagicMock()
    gateway.default_provider = "mock"
    gateway.get = MagicMock(return_value=None)

    async def _stream(request, config, trace_ctx=None):  # noqa: ANN001
        yield StreamChunk(
            type="done",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            metadata={"finish_reason": "stop", "model": "mock-model"},
        )

    gateway.stream = _stream

    return ConversationAgent(
        gateway=gateway,
        model_config=ModelConfig(provider="mock", model_id="mock-model"),
        max_turns=3,
        cognitive_primitives=cognitive_primitives,
    )


class TestPinInterceptionInAgent:
    def test_pin_schema_in_tool_list_when_enabled(self) -> None:
        agent = _make_agent(cognitive_primitives=["pin"])
        tools = agent._get_current_tools()
        assert tools is not None
        names = [t["function"]["name"] for t in tools]
        assert PIN_TOOL_NAME in names
        # unpin rides with pin
        assert UNPIN_TOOL_NAME in names

    def test_pin_schema_absent_when_disabled(self) -> None:
        agent = _make_agent(cognitive_primitives=[])
        tools = agent._get_current_tools()
        assert tools is not None
        names = [t["function"]["name"] for t in tools]
        assert PIN_TOOL_NAME not in names
        assert UNPIN_TOOL_NAME not in names

    def test_pin_tool_call_intercepted_and_returns_json(self) -> None:
        agent = _make_agent(cognitive_primitives=["pin"])
        tc = ToolCallRequest(
            id="tc-pin",
            name=PIN_TOOL_NAME,
            arguments=json.dumps({"content": "critical", "label": "c"}),
        )
        result = agent._handle_cognitive_tool_call(tc, run_id="r1")
        assert result.success is True
        payload = json.loads(result.output_str)
        assert payload["pinned"] is True
        assert payload["pin_id"].startswith("p_")
        assert payload["label"] == "c"
        # Cognitive handler should have the pin
        assert len(agent._cognitive_handler.pin_state.entries) == 1

    def test_unpin_tool_call_intercepted(self) -> None:
        agent = _make_agent(cognitive_primitives=["pin"])
        pin_tc = ToolCallRequest(
            id="tc-pin",
            name=PIN_TOOL_NAME,
            arguments=json.dumps({"content": "X"}),
        )
        pin_result = agent._handle_cognitive_tool_call(pin_tc, run_id="r1")
        pin_id = json.loads(pin_result.output_str)["pin_id"]
        unpin_tc = ToolCallRequest(
            id="tc-unpin",
            name=UNPIN_TOOL_NAME,
            arguments=json.dumps({"pin_id": pin_id}),
        )
        unpin_result = agent._handle_cognitive_tool_call(
            unpin_tc, run_id="r1",
        )
        assert unpin_result.success is True
        payload = json.loads(unpin_result.output_str)
        assert payload["unpinned"] is True
        assert len(agent._cognitive_handler.pin_state.entries) == 0

    def test_cognitive_primitive_trace_event_emitted(self) -> None:
        """Pin invocation must emit a COGNITIVE_PRIMITIVE trace event."""
        from arcana.contracts.trace import EventType

        trace = MagicMock()
        agent = _make_agent(cognitive_primitives=["pin"])
        agent.trace_writer = trace

        tc = ToolCallRequest(
            id="tc-pin-trace",
            name=PIN_TOOL_NAME,
            arguments=json.dumps({"content": "trace me"}),
        )
        agent._handle_cognitive_tool_call(tc, run_id="r1")
        assert trace.write.called
        call_args = trace.write.call_args[0][0]
        assert call_args.event_type == EventType.COGNITIVE_PRIMITIVE
        assert call_args.metadata["primitive"] == PIN_TOOL_NAME
        assert call_args.metadata["args"]["content"] == "trace me"
        assert call_args.metadata["result"]["pinned"] is True
