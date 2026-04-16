"""Tests for structured MessageDecision evidence in ContextDecision.

Each strategy path in WorkingSetBuilder must emit one MessageDecision per
input message, with appropriate outcome / reason / fidelity / token counts.
"""

from __future__ import annotations

from arcana.context.builder import WorkingSetBuilder
from arcana.contracts.context import ContextStrategy, TokenBudget
from arcana.contracts.llm import Message, MessageRole


def _msg(role: MessageRole, content: str) -> Message:
    return Message(role=role, content=content)


def _builder(
    *,
    total_window: int = 128_000,
    response_reserve: int = 4096,
    strategy: ContextStrategy | None = None,
) -> WorkingSetBuilder:
    return WorkingSetBuilder(
        identity="You are a test assistant.",
        token_budget=TokenBudget(
            total_window=total_window,
            response_reserve=response_reserve,
        ),
        strategy=strategy,
    )


class TestPassthrough:
    def test_all_kept_when_under_budget(self):
        builder = _builder()
        messages = [
            _msg(MessageRole.SYSTEM, "sys"),
            _msg(MessageRole.USER, "hello"),
            _msg(MessageRole.ASSISTANT, "hi"),
        ]
        builder.build_conversation_context(messages, turn=0)

        decision = builder.last_decision
        assert decision is not None
        assert decision.strategy == "passthrough"
        assert len(decision.decisions) == len(messages)
        for i, d in enumerate(decision.decisions):
            assert d.index == i
            assert d.outcome == "kept"
            assert d.reason == "passthrough"
            assert d.token_count_before == messages[i].token_count
            assert d.token_count_after == messages[i].token_count
            assert d.fidelity is None
            assert d.relevance_score is None


class TestAggressiveTruncate:
    def test_dropped_messages_marked(self):
        strategy = ContextStrategy(mode="always_compress", aggressive_keep_turns=1)
        builder = _builder(total_window=2000, response_reserve=100, strategy=strategy)

        # Construct a conversation long enough that aggressive_truncate fires.
        big = "x" * 3000
        messages = [
            _msg(MessageRole.SYSTEM, "sys"),
            _msg(MessageRole.USER, big),
            _msg(MessageRole.ASSISTANT, big),
            _msg(MessageRole.USER, big),
            _msg(MessageRole.ASSISTANT, big),
            _msg(MessageRole.USER, "last"),
            _msg(MessageRole.ASSISTANT, "ok"),
        ]

        # Force aggressive_truncate by using a window that's too small
        strategy2 = ContextStrategy(
            mode="adaptive",
            passthrough_threshold=0.01,
            tail_preserve_threshold=0.02,
            llm_summarize_threshold=0.03,
            aggressive_keep_turns=1,
        )
        builder = _builder(total_window=500, response_reserve=50, strategy=strategy2)
        builder.build_conversation_context(messages, turn=0)

        decision = builder.last_decision
        assert decision is not None
        # Must have one decision per input message
        assert len(decision.decisions) == len(messages)
        outcomes = {d.outcome for d in decision.decisions}
        assert "dropped" in outcomes or "kept" in outcomes
        # The dropped entries must have token_count_after == 0
        for d in decision.decisions:
            if d.outcome == "dropped":
                assert d.token_count_after == 0
                assert d.reason == "aggressive_truncate_drop"


class TestTailPreserveFidelity:
    def test_middle_compressed_with_fidelity(self):
        # Force tail_preserve by creating a lot of middle messages
        strategy = ContextStrategy(
            mode="adaptive",
            passthrough_threshold=0.01,
            tail_preserve_threshold=0.99,
            llm_summarize_threshold=0.995,
            tail_preserve_keep_recent=2,
        )
        builder = _builder(total_window=2000, response_reserve=100, strategy=strategy)

        messages = [_msg(MessageRole.SYSTEM, "sys")]
        for i in range(8):
            messages.append(_msg(MessageRole.USER, f"middle user {i} " * 50))
            messages.append(_msg(MessageRole.ASSISTANT, f"middle reply {i} " * 50))
        messages.append(_msg(MessageRole.USER, "recent q"))
        messages.append(_msg(MessageRole.ASSISTANT, "recent a"))

        builder.build_conversation_context(messages, turn=0)

        decision = builder.last_decision
        assert decision is not None
        assert decision.strategy == "tail_preserve"
        assert len(decision.decisions) == len(messages)

        # Head should be kept
        assert decision.decisions[0].outcome == "kept"
        assert decision.decisions[0].reason == "tail_preserve_head"

        # Tail (last 2) should be kept
        assert decision.decisions[-1].reason == "tail_preserve_tail"
        assert decision.decisions[-2].reason == "tail_preserve_tail"

        # At least one middle should have fidelity set
        fidelity_levels = {
            d.fidelity for d in decision.decisions
            if d.fidelity is not None
        }
        assert len(fidelity_levels) > 0
        assert all(lv in {"L0", "L1", "L2", "L3"} for lv in fidelity_levels)


class TestNoSummaryBudget:
    def test_middle_dropped_when_no_budget(self):
        # Directly invoke _compress_with_relevance with a budget that's
        # smaller than head + tail + margin, forcing the "no summary budget"
        # branch.
        strategy = ContextStrategy(tail_preserve_keep_recent=2)
        builder = _builder(strategy=strategy)
        messages = [
            _msg(MessageRole.SYSTEM, "s" * 2000),  # ~500 tokens, big head
            _msg(MessageRole.USER, "mid1"),
            _msg(MessageRole.ASSISTANT, "mid2"),
            _msg(MessageRole.USER, "tail_u"),
            _msg(MessageRole.ASSISTANT, "tail_a"),
        ]
        builder._compress_with_relevance(
            messages,
            budget=50,   # deliberately smaller than head tokens
            turn=0,
            messages_in=len(messages),
            effective_tool_tokens=0,
            memory_injected=False,
        )

        decision = builder.last_decision
        assert decision is not None
        assert decision.strategy == "tail_preserve"
        assert len(decision.decisions) == len(messages)
        middle_outcomes = [decision.decisions[i].outcome for i in (1, 2)]
        assert all(o == "dropped" for o in middle_outcomes)
        assert decision.decisions[1].reason == "tail_preserve_no_budget_for_middle"


class TestStaleToolPruning:
    def test_pruned_indices_marked_in_decisions(self):
        strategy = ContextStrategy(tool_result_staleness_turns=1)
        builder = _builder(strategy=strategy)

        messages: list[Message] = [_msg(MessageRole.SYSTEM, "sys")]
        # Build 4 turns of user/assistant/tool -> 12 messages after system.
        # staleness * 3 = 3, so only the very last turn is "recent".
        for i in range(4):
            messages.append(_msg(MessageRole.USER, f"u{i}"))
            messages.append(_msg(MessageRole.ASSISTANT, f"a{i}"))
            messages.append(Message(
                role=MessageRole.TOOL,
                content="x" * 500,  # big enough to be worth pruning
                tool_call_id=f"call_{i}",
            ))

        builder.build_conversation_context(messages, turn=5)

        decision = builder.last_decision
        assert decision is not None
        # Phase 0 should have pruned the old tool messages; those indices
        # must be visible in the decisions.
        pruned_decisions = [
            d for d in decision.decisions
            if "stale_tool_result" in d.reason
        ]
        assert len(pruned_decisions) >= 1
        for d in pruned_decisions:
            assert d.outcome == "compressed"
            # before > after because pruning shortened the content
            assert d.token_count_before > d.token_count_after
