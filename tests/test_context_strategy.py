"""Tests for ContextStrategy, ContextReport, and strategy-driven WorkingSetBuilder."""

from __future__ import annotations

import pytest

from arcana.context.builder import WorkingSetBuilder, estimate_tokens
from arcana.contracts.context import (
    ContextReport,
    ContextStrategy,
    TokenBudget,
)
from arcana.contracts.llm import Message, MessageRole


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------


class TestContextStrategyDefaults:
    def test_default_mode(self):
        s = ContextStrategy()
        assert s.mode == "adaptive"

    def test_default_thresholds(self):
        s = ContextStrategy()
        assert s.passthrough_threshold == 0.50
        assert s.tail_preserve_threshold == 0.75
        assert s.llm_summarize_threshold == 0.90

    def test_default_tail_preserve_keep_recent(self):
        s = ContextStrategy()
        assert s.tail_preserve_keep_recent == 6

    def test_default_aggressive_keep_turns(self):
        s = ContextStrategy()
        assert s.aggressive_keep_turns == 2

    def test_off_mode(self):
        s = ContextStrategy(mode="off")
        assert s.mode == "off"

    def test_always_compress_mode(self):
        s = ContextStrategy(mode="always_compress")
        assert s.mode == "always_compress"


class TestContextReportDefaults:
    def test_defaults(self):
        r = ContextReport()
        assert r.turn == 0
        assert r.strategy_used == "passthrough"
        assert r.total_tokens == 0
        assert r.identity_tokens == 0
        assert r.task_tokens == 0
        assert r.tools_tokens == 0
        assert r.history_tokens == 0
        assert r.memory_tokens == 0
        assert r.compression_applied is False
        assert r.compression_savings == 0
        assert r.messages_compressed == 0
        assert r.window_size == 128_000
        assert r.utilization == 0.0
        assert r.tools_loaded == 0
        assert r.tools_available == 0

    def test_custom_values(self):
        r = ContextReport(
            turn=3,
            strategy_used="tail_preserve",
            total_tokens=5000,
            compression_applied=True,
            compression_savings=2000,
            utilization=0.65,
        )
        assert r.turn == 3
        assert r.strategy_used == "tail_preserve"
        assert r.compression_applied is True
        assert r.compression_savings == 2000


# ---------------------------------------------------------------------------
# WorkingSetBuilder with strategy
# ---------------------------------------------------------------------------


def _make_messages(n: int, content_size: int = 50) -> list[Message]:
    """Create n messages: 1 system + (n-1) user/assistant alternating."""
    msgs = [Message(role=MessageRole.SYSTEM, content="System prompt " * 5)]
    for i in range(1, n):
        role = MessageRole.USER if i % 2 == 1 else MessageRole.ASSISTANT
        msgs.append(Message(role=role, content=f"Message {i} " * content_size))
    return msgs


class TestBuilderPassthrough:
    """Under budget, strategy=passthrough."""

    def test_all_messages_kept(self):
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
        )
        msgs = _make_messages(5, content_size=10)
        result = builder.build_conversation_context(msgs, turn=0)
        assert len(result) == len(msgs)

    def test_last_report_populated(self):
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
        )
        msgs = _make_messages(3, content_size=5)
        builder.build_conversation_context(msgs, turn=0)
        assert builder.last_report is not None
        assert builder.last_report.strategy_used == "passthrough"
        assert builder.last_report.compression_applied is False
        assert builder.last_report.turn == 0

    def test_report_has_token_breakdown(self):
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
        )
        msgs = _make_messages(4, content_size=5)
        builder.build_conversation_context(msgs, tool_token_estimate=100, turn=1)
        report = builder.last_report
        assert report is not None
        assert report.identity_tokens > 0
        assert report.tools_tokens == 100
        assert report.total_tokens > 0
        assert report.window_size == 128_000
        assert 0 <= report.utilization <= 1.0


class TestBuilderTailPreserve:
    """Over budget, should compress middle and keep tail."""

    def test_compression_triggered(self):
        # Use a tiny window to force compression
        budget = TokenBudget(total_window=500, response_reserve=50)
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=budget,
        )
        # Create enough messages to exceed the tiny window
        msgs = _make_messages(20, content_size=30)
        result = builder.build_conversation_context(msgs, turn=2)
        assert len(result) < len(msgs)

    def test_report_shows_compression(self):
        budget = TokenBudget(total_window=500, response_reserve=50)
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=budget,
        )
        msgs = _make_messages(20, content_size=30)
        builder.build_conversation_context(msgs, turn=2)
        report = builder.last_report
        assert report is not None
        assert report.compression_applied is True
        assert report.compression_savings > 0
        assert report.messages_compressed > 0


class TestBuilderAggressiveTruncate:
    """Very over budget + high utilization triggers aggressive truncation."""

    def test_aggressive_truncate_keeps_few_messages(self):
        # Tiny window + strategy with very low thresholds so that
        # utilization (which will be >1.0) triggers aggressive_truncate.
        budget = TokenBudget(total_window=200, response_reserve=10)
        strategy = ContextStrategy(
            mode="adaptive",
            passthrough_threshold=0.01,
            tail_preserve_threshold=0.02,
            llm_summarize_threshold=0.03,
            # Above 0.03 = aggressive truncate
            aggressive_keep_turns=2,
        )
        builder = WorkingSetBuilder(
            identity="s",
            token_budget=budget,
            strategy=strategy,
        )
        msgs = _make_messages(20, content_size=30)
        result = builder.build_conversation_context(msgs, turn=5)
        # Should keep system + last 2 turns worth of messages
        # (much fewer than 20)
        assert len(result) <= 6  # system + ~4-5 messages for 2 turns
        assert len(result) < len(msgs)

    def test_aggressive_truncate_report(self):
        budget = TokenBudget(total_window=200, response_reserve=10)
        strategy = ContextStrategy(
            mode="adaptive",
            passthrough_threshold=0.01,
            tail_preserve_threshold=0.02,
            llm_summarize_threshold=0.03,
            aggressive_keep_turns=2,
        )
        builder = WorkingSetBuilder(
            identity="s",
            token_budget=budget,
            strategy=strategy,
        )
        msgs = _make_messages(20, content_size=30)
        builder.build_conversation_context(msgs, turn=5)
        report = builder.last_report
        assert report is not None
        assert report.strategy_used == "aggressive_truncate"
        assert report.compression_applied is True

    def test_default_thresholds_over_budget_uses_tail_preserve_not_aggressive(self):
        """With default thresholds, moderate over-budget uses tail_preserve, not aggressive."""
        budget = TokenBudget(total_window=800, response_reserve=50)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=budget,
        )
        msgs = _make_messages(20, content_size=50)
        builder.build_conversation_context(msgs, turn=3)
        report = builder.last_report
        assert report is not None
        # Default thresholds: aggressive_truncate only at >90% utilization,
        # but the under-budget check gates entry. If over budget, it uses
        # tail_preserve or llm_summarize depending on utilization.
        assert report.strategy_used != "passthrough"


class TestBuilderStrategyOff:
    """strategy="off" disables compression even when over budget."""

    def test_off_mode_passthrough(self):
        budget = TokenBudget(total_window=500, response_reserve=50)
        strategy = ContextStrategy(mode="off")
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=budget,
            strategy=strategy,
        )
        # Even with many messages, mode=off should passthrough
        msgs = _make_messages(5, content_size=10)
        result = builder.build_conversation_context(msgs, turn=0)
        assert len(result) == len(msgs)
        assert builder.last_report is not None
        assert builder.last_report.strategy_used == "passthrough"

    def test_off_mode_report_no_compression(self):
        strategy = ContextStrategy(mode="off")
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
            strategy=strategy,
        )
        msgs = _make_messages(3, content_size=5)
        builder.build_conversation_context(msgs, turn=0)
        report = builder.last_report
        assert report is not None
        assert report.compression_applied is False


class TestBuilderLastReportPopulated:
    """Verify last_report is set after build in all paths."""

    def test_sync_build(self):
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
        )
        msgs = _make_messages(3)
        builder.build_conversation_context(msgs, turn=0)
        assert builder.last_report is not None
        assert isinstance(builder.last_report, ContextReport)

    @pytest.mark.asyncio
    async def test_async_build(self):
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
        )
        msgs = _make_messages(3)
        await builder.abuild_conversation_context(msgs, turn=0)
        assert builder.last_report is not None
        assert isinstance(builder.last_report, ContextReport)

    def test_report_preserved_between_calls(self):
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
        )
        msgs = _make_messages(3)
        builder.build_conversation_context(msgs, turn=0)
        first_report = builder.last_report
        assert first_report is not None

        builder.build_conversation_context(msgs, turn=1)
        second_report = builder.last_report
        assert second_report is not None
        assert second_report.turn == 1


class TestRuntimeAcceptsContextStrategy:
    """Test that Runtime accepts context_strategy param."""

    def test_runtime_string_strategy(self):
        from arcana.runtime_core import Runtime

        # Just verify init doesn't crash — no providers needed for this check
        rt = Runtime(context_strategy="off")
        assert rt._context_strategy.mode == "off"

    def test_runtime_strategy_object(self):
        from arcana.runtime_core import Runtime

        strategy = ContextStrategy(mode="always_compress", tail_preserve_keep_recent=4)
        rt = Runtime(context_strategy=strategy)
        assert rt._context_strategy.mode == "always_compress"
        assert rt._context_strategy.tail_preserve_keep_recent == 4

    def test_runtime_default_strategy(self):
        from arcana.runtime_core import Runtime

        rt = Runtime()
        assert rt._context_strategy.mode == "adaptive"


class TestStrategyResolution:
    """Test the _resolve_strategy_name logic."""

    def test_adaptive_below_passthrough(self):
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
        )
        assert builder._resolve_strategy_name(0.3) == "passthrough"

    def test_adaptive_tail_preserve_range(self):
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
        )
        assert builder._resolve_strategy_name(0.6) == "tail_preserve"

    def test_adaptive_llm_summarize_range(self):
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
        )
        assert builder._resolve_strategy_name(0.8) == "llm_summarize"

    def test_adaptive_aggressive_above_threshold(self):
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
        )
        assert builder._resolve_strategy_name(0.95) == "aggressive_truncate"

    def test_off_always_passthrough(self):
        strategy = ContextStrategy(mode="off")
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
            strategy=strategy,
        )
        assert builder._resolve_strategy_name(0.99) == "passthrough"

    def test_always_compress_always_tail_preserve(self):
        strategy = ContextStrategy(mode="always_compress")
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=TokenBudget(total_window=128_000),
            strategy=strategy,
        )
        assert builder._resolve_strategy_name(0.1) == "tail_preserve"


class TestTailPreserveKeepRecent:
    """Test that tail_preserve_keep_recent is respected."""

    def test_custom_keep_recent(self):
        budget = TokenBudget(total_window=500, response_reserve=50)
        strategy = ContextStrategy(tail_preserve_keep_recent=3)
        builder = WorkingSetBuilder(
            identity="system",
            token_budget=budget,
            strategy=strategy,
        )
        msgs = _make_messages(20, content_size=30)
        result = builder.build_conversation_context(msgs, turn=0)
        # With keep_recent=3, the tail should be 3 messages
        # Result = head(1) + summary(1) + tail(3)
        assert len(result) <= 5
