"""Tests for Phase 0 tool result pruning in WorkingSetBuilder."""

from __future__ import annotations

import pytest

from arcana.context.builder import WorkingSetBuilder, _content_text
from arcana.contracts.context import ContextStrategy, TokenBudget
from arcana.contracts.llm import Message, MessageRole


def _make_tool_message(content: str, *, name: str = "my_tool", tool_call_id: str = "tc1") -> Message:
    """Create a tool result message."""
    return Message(role=MessageRole.TOOL, content=content, name=name, tool_call_id=tool_call_id)


def _make_conversation(
    *,
    old_tool_content: str = "a" * 1000,
    recent_tool_content: str = "recent tool output",
    staleness_turns: int = 4,
) -> list[Message]:
    """Build a conversation with stale and recent tool results.

    Structure:
    - 1 system message
    - Several old turns (user / assistant / tool) in the stale region
    - Recent turns within the staleness window

    The staleness boundary is: len(messages) - staleness_turns * 3.
    We need enough messages so the old tool results fall before this boundary.
    """
    msgs: list[Message] = [Message(role=MessageRole.SYSTEM, content="System prompt.")]

    # Old turns (6 turns = 18 messages, well before the boundary)
    for i in range(6):
        msgs.append(Message(role=MessageRole.USER, content=f"Old user message {i}"))
        msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Old assistant message {i}"))
        msgs.append(_make_tool_message(old_tool_content, tool_call_id=f"old_tc_{i}"))

    # Recent turns (staleness_turns * 3 messages = 12 by default)
    for i in range(staleness_turns):
        msgs.append(Message(role=MessageRole.USER, content=f"Recent user {i}"))
        msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Recent assistant {i}"))
        msgs.append(_make_tool_message(recent_tool_content, tool_call_id=f"recent_tc_{i}"))

    return msgs


class TestRecentToolResultsNotPruned:
    """Tool results within the staleness threshold must NOT be pruned."""

    def test_recent_tool_results_untouched(self):
        strategy = ContextStrategy(tool_result_staleness_turns=4)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        msgs = _make_conversation(staleness_turns=4)
        result = builder.build_conversation_context(msgs, turn=10)

        # Recent tool messages should be unchanged
        recent_tools = [
            m for m in result
            if (m.role == MessageRole.TOOL or m.role == "tool")
            and m.tool_call_id is not None
            and m.tool_call_id.startswith("recent_tc_")
        ]
        for m in recent_tools:
            content = _content_text(m.content)
            assert "[tool result pruned" not in content

    def test_all_messages_recent_no_pruning(self):
        """When all messages are within the staleness window, nothing is pruned."""
        strategy = ContextStrategy(tool_result_staleness_turns=10)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        # Only 3 turns = 9 + 1 system = 10 messages; staleness window = 10 * 3 = 30
        msgs = [Message(role=MessageRole.SYSTEM, content="System.")]
        for i in range(3):
            msgs.append(Message(role=MessageRole.USER, content=f"User {i}"))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Asst {i}"))
            msgs.append(_make_tool_message("tool output", tool_call_id=f"tc_{i}"))

        result = builder.build_conversation_context(msgs, turn=3)
        assert len(result) == len(msgs)  # No messages changed count
        for m in result:
            content = _content_text(m.content)
            assert "[tool result pruned" not in content


class TestOldToolResultsPruned:
    """Tool results beyond the staleness threshold must be pruned to summary."""

    def test_old_tool_results_pruned(self):
        strategy = ContextStrategy(tool_result_staleness_turns=4, tool_result_prune_max_chars=200)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        old_content = "x" * 1000
        msgs = _make_conversation(old_tool_content=old_content, staleness_turns=4)
        result = builder.build_conversation_context(msgs, turn=10)

        # Old tool messages should be pruned
        old_tools = [
            m for m in result
            if (m.role == MessageRole.TOOL or m.role == "tool")
            and m.tool_call_id is not None
            and m.tool_call_id.startswith("old_tc_")
        ]
        assert len(old_tools) == 6
        for m in old_tools:
            content = _content_text(m.content)
            assert "[tool result pruned" in content
            # Should contain token count info
            assert "tokens]" in content

    def test_pruned_message_preserves_metadata(self):
        """Pruned tool messages should preserve name and tool_call_id."""
        strategy = ContextStrategy(tool_result_staleness_turns=2)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        msgs = _make_conversation(old_tool_content="y" * 500, staleness_turns=2)
        result = builder.build_conversation_context(msgs, turn=10)

        old_tools = [
            m for m in result
            if (m.role == MessageRole.TOOL or m.role == "tool")
            and m.tool_call_id is not None
            and m.tool_call_id.startswith("old_tc_")
        ]
        for m in old_tools:
            assert m.name == "my_tool"
            assert m.tool_call_id is not None
            assert m.tool_call_id.startswith("old_tc_")


class TestErrorToolResultsNeverPruned:
    """Tool results containing 'error' or 'failed' must NEVER be pruned."""

    def test_error_result_not_pruned(self):
        strategy = ContextStrategy(tool_result_staleness_turns=2)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        msgs = [Message(role=MessageRole.SYSTEM, content="System.")]
        # Old turns with error content
        for i in range(6):
            msgs.append(Message(role=MessageRole.USER, content=f"User {i}"))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Asst {i}"))
            error_content = f"Error: something went wrong in operation {i}. " + "z" * 500
            msgs.append(_make_tool_message(error_content, tool_call_id=f"err_tc_{i}"))
        # Recent turns
        for i in range(2):
            msgs.append(Message(role=MessageRole.USER, content=f"Recent {i}"))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Recent asst {i}"))
            msgs.append(_make_tool_message("recent", tool_call_id=f"recent_tc_{i}"))

        result = builder.build_conversation_context(msgs, turn=8)

        # Error tool messages in the stale region should NOT be pruned
        err_tools = [
            m for m in result
            if (m.role == MessageRole.TOOL or m.role == "tool")
            and m.tool_call_id is not None
            and m.tool_call_id.startswith("err_tc_")
        ]
        for m in err_tools:
            content = _content_text(m.content)
            assert "[tool result pruned" not in content
            assert "Error:" in content

    def test_failed_result_not_pruned(self):
        strategy = ContextStrategy(tool_result_staleness_turns=2)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        msgs = [Message(role=MessageRole.SYSTEM, content="System.")]
        for i in range(6):
            msgs.append(Message(role=MessageRole.USER, content=f"User {i}"))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Asst {i}"))
            failed_content = f"Operation failed: could not complete step {i}. " + "w" * 500
            msgs.append(_make_tool_message(failed_content, tool_call_id=f"fail_tc_{i}"))
        for i in range(2):
            msgs.append(Message(role=MessageRole.USER, content=f"Recent {i}"))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Recent asst {i}"))
            msgs.append(_make_tool_message("recent", tool_call_id=f"recent_tc_{i}"))

        result = builder.build_conversation_context(msgs, turn=8)

        fail_tools = [
            m for m in result
            if (m.role == MessageRole.TOOL or m.role == "tool")
            and m.tool_call_id is not None
            and m.tool_call_id.startswith("fail_tc_")
        ]
        for m in fail_tools:
            content = _content_text(m.content)
            assert "[tool result pruned" not in content
            assert "failed" in content.lower()


class TestPrunedMessagePreservesFirstNChars:
    """Pruned message should preserve the first N chars of original content."""

    def test_preserves_max_chars(self):
        max_chars = 150
        strategy = ContextStrategy(
            tool_result_staleness_turns=2,
            tool_result_prune_max_chars=max_chars,
        )
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        original_content = "UNIQUE_PREFIX_" + "q" * 500
        msgs = [Message(role=MessageRole.SYSTEM, content="System.")]
        for i in range(6):
            msgs.append(Message(role=MessageRole.USER, content=f"User {i}"))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Asst {i}"))
            msgs.append(_make_tool_message(original_content, tool_call_id=f"tc_{i}"))
        for i in range(2):
            msgs.append(Message(role=MessageRole.USER, content=f"Recent {i}"))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Recent asst {i}"))
            msgs.append(_make_tool_message("recent", tool_call_id=f"recent_tc_{i}"))

        result = builder.build_conversation_context(msgs, turn=8)

        old_tools = [
            m for m in result
            if (m.role == MessageRole.TOOL or m.role == "tool")
            and m.tool_call_id is not None
            and m.tool_call_id.startswith("tc_")
        ]
        for m in old_tools:
            content = _content_text(m.content)
            # The preview must contain the first max_chars of original
            assert original_content[:max_chars] in content
            # But NOT the full original
            assert original_content not in content


class TestTokensSavedCalculation:
    """tokens_saved must be correctly calculated."""

    def test_tokens_saved_positive(self):
        strategy = ContextStrategy(tool_result_staleness_turns=2, tool_result_prune_max_chars=50)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        # Large tool content -> significant savings
        large_content = "d" * 2000
        msgs = _make_conversation(old_tool_content=large_content, staleness_turns=2)

        # Use the internal method directly to check tokens_saved
        _, count, saved = builder._prune_stale_tool_results(msgs, current_turn=10)
        assert count == 6  # 6 old tool messages
        assert saved > 0

        # Each pruned message saves roughly (2000/4) - (50 + header)/4 tokens
        # Just verify it's substantial
        assert saved > 100


class TestContextReportIncludesPruningStats:
    """ContextReport must include Phase 0 pruning statistics."""

    def test_report_has_pruning_stats(self):
        strategy = ContextStrategy(tool_result_staleness_turns=2, tool_result_prune_max_chars=100)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        msgs = _make_conversation(old_tool_content="r" * 800, staleness_turns=2)
        builder.build_conversation_context(msgs, turn=10)

        report = builder.last_report
        assert report is not None
        assert report.tool_results_pruned == 6
        assert report.tool_results_tokens_saved > 0

    def test_report_zero_when_no_pruning(self):
        """When nothing is pruned, report should have zeros."""
        strategy = ContextStrategy(tool_result_staleness_turns=100)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System."),
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
        ]
        builder.build_conversation_context(msgs, turn=1)

        report = builder.last_report
        assert report is not None
        assert report.tool_results_pruned == 0
        assert report.tool_results_tokens_saved == 0


class TestNoToolResults:
    """Messages with no tool results should pass through unchanged."""

    def test_no_tool_messages(self):
        strategy = ContextStrategy(tool_result_staleness_turns=2)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        msgs = [Message(role=MessageRole.SYSTEM, content="System.")]
        for i in range(20):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            msgs.append(Message(role=role, content=f"Message {i}: " + "x" * 100))

        result = builder.build_conversation_context(msgs, turn=10)
        # No tool messages -> no pruning happened
        assert len(result) == len(msgs)

        report = builder.last_report
        assert report is not None
        assert report.tool_results_pruned == 0
        assert report.tool_results_tokens_saved == 0


class TestPhase0IntegrationWithCompression:
    """Phase 0 runs first and reduces token count before compression pipeline."""

    def test_phase0_reduces_tokens_before_compression(self):
        """Phase 0 should prune stale tool results, reducing total tokens
        so that the compression pipeline works on already-reduced data."""
        strategy = ContextStrategy(
            tool_result_staleness_turns=2,
            tool_result_prune_max_chars=50,
        )
        # Small window to force compression after Phase 0
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=5000, response_reserve=500),
            strategy=strategy,
        )
        # Large tool content that Phase 0 will prune
        large_content = "z" * 2000
        msgs = _make_conversation(old_tool_content=large_content, staleness_turns=2)

        # Without Phase 0, total tokens would be much higher
        original_total = sum(m.token_count for m in msgs)

        result = builder.build_conversation_context(msgs, turn=10)
        result_total = sum(m.token_count for m in result)

        # Phase 0 + compression should significantly reduce tokens
        assert result_total < original_total

        # Report should show pruning happened
        report = builder.last_report
        assert report is not None
        assert report.tool_results_pruned > 0


class TestPhase0AsyncMethod:
    """Phase 0 must also work in the async build method."""

    @pytest.mark.asyncio
    async def test_async_build_prunes_stale_tools(self):
        strategy = ContextStrategy(tool_result_staleness_turns=2, tool_result_prune_max_chars=100)
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
            strategy=strategy,
        )
        msgs = _make_conversation(old_tool_content="a" * 800, staleness_turns=2)
        result = await builder.abuild_conversation_context(msgs, turn=10)

        # Old tool messages should be pruned
        old_tools = [
            m for m in result
            if (m.role == MessageRole.TOOL or m.role == "tool")
            and m.tool_call_id is not None
            and m.tool_call_id.startswith("old_tc_")
        ]
        for m in old_tools:
            content = _content_text(m.content)
            assert "[tool result pruned" in content

        report = builder.last_report
        assert report is not None
        assert report.tool_results_pruned == 6
        assert report.tool_results_tokens_saved > 0
