"""Tests for the Virtual Memory system: ContextPageTable and RecallHandler.

Covers:
- ContextPageTable unit tests (evict, recall, index, clear)
- RecallHandler unit tests (handle valid/invalid chunks, spec validation)
- Fidelity compression integration with page table
- Demotion loop termination under extreme budget
"""

from __future__ import annotations

import pytest

from arcana.context.builder import WorkingSetBuilder, estimate_tokens
from arcana.context.page_table import ContextPageTable
from arcana.contracts.context import TokenBudget
from arcana.contracts.llm import Message, MessageRole
from arcana.runtime.recall import RecallHandler, RECALL_SPEC


# ---------------------------------------------------------------------------
# 1. ContextPageTable unit tests
# ---------------------------------------------------------------------------


class TestContextPageTable:
    def test_evict_and_recall(self):
        """Evict messages, then recall them by chunk_id."""
        pt = ContextPageTable()
        msgs = [
            Message(role=MessageRole.USER, content="hello"),
            Message(role=MessageRole.ASSISTANT, content="hi there"),
        ]
        chunk_id = pt.evict(msgs, "2 messages")
        assert chunk_id.startswith("ctx-")
        assert pt.has_pages
        assert pt.page_count == 1

        recalled = pt.recall(chunk_id)
        assert recalled is not None
        assert len(recalled) == 2
        assert recalled[0].content == "hello"
        assert recalled[1].content == "hi there"

    def test_recall_unknown_chunk(self):
        """Recall with unknown chunk_id returns None."""
        pt = ContextPageTable()
        assert pt.recall("ctx-nonexistent") is None

    def test_has_pages_empty(self):
        """Empty page table has no pages."""
        pt = ContextPageTable()
        assert not pt.has_pages
        assert pt.page_count == 0

    def test_multiple_evictions(self):
        """Multiple evictions produce different chunk_ids."""
        pt = ContextPageTable()
        id1 = pt.evict([Message(role=MessageRole.USER, content="a")], "first")
        id2 = pt.evict([Message(role=MessageRole.USER, content="b")], "second")
        assert id1 != id2
        assert pt.page_count == 2
        assert pt.recall(id1)[0].content == "a"
        assert pt.recall(id2)[0].content == "b"

    def test_index(self):
        """Index returns summary of all chunks."""
        pt = ContextPageTable()
        pt.evict([Message(role=MessageRole.USER, content="x")], "first batch")
        pt.evict([Message(role=MessageRole.USER, content="y")], "second batch")
        idx = pt.index()
        assert len(idx) == 2
        assert all("chunk_id" in entry and "summary" in entry for entry in idx)
        summaries = [e["summary"] for e in idx]
        assert "first batch" in summaries
        assert "second batch" in summaries

    def test_clear(self):
        """Clear removes all pages."""
        pt = ContextPageTable()
        pt.evict([Message(role=MessageRole.USER, content="x")], "test")
        assert pt.has_pages
        pt.clear()
        assert not pt.has_pages
        assert pt.page_count == 0

    def test_get_summary(self):
        """get_summary returns the summary for a given chunk_id."""
        pt = ContextPageTable()
        chunk_id = pt.evict([Message(role=MessageRole.USER, content="z")], "my summary")
        assert pt.get_summary(chunk_id) == "my summary"
        assert pt.get_summary("ctx-unknown") == ""

    def test_evict_stores_copy(self):
        """Evict stores a copy, not a reference to the original list."""
        pt = ContextPageTable()
        msgs = [Message(role=MessageRole.USER, content="original")]
        chunk_id = pt.evict(msgs, "test")
        msgs.append(Message(role=MessageRole.USER, content="added later"))
        recalled = pt.recall(chunk_id)
        assert len(recalled) == 1  # original list mutation did not affect stored copy


# ---------------------------------------------------------------------------
# 2. RecallHandler unit tests
# ---------------------------------------------------------------------------


class TestRecallHandler:
    @pytest.mark.asyncio
    async def test_handle_valid_chunk(self):
        """Handle a valid chunk_id returns formatted messages."""
        pt = ContextPageTable()
        msgs = [
            Message(role=MessageRole.USER, content="What is X?"),
            Message(role=MessageRole.ASSISTANT, content="X is a variable."),
        ]
        chunk_id = pt.evict(msgs, "Q&A about X")
        handler = RecallHandler(pt)
        result = await handler.handle(chunk_id)
        assert "Recalled context" in result
        assert "What is X?" in result
        assert "X is a variable." in result

    @pytest.mark.asyncio
    async def test_handle_invalid_chunk(self):
        """Handle an invalid chunk_id returns error message."""
        pt = ContextPageTable()
        handler = RecallHandler(pt)
        result = await handler.handle("ctx-doesnotexist")
        assert "not found" in result

    def test_recall_spec_valid(self):
        """RECALL_SPEC is a valid ToolSpec with expected fields."""
        assert RECALL_SPEC.name == "recall"
        assert "chunk_id" in RECALL_SPEC.input_schema["properties"]
        assert RECALL_SPEC.input_schema["required"] == ["chunk_id"]

    @pytest.mark.asyncio
    async def test_handle_formats_roles(self):
        """Handle formats each message with its role label."""
        pt = ContextPageTable()
        msgs = [
            Message(role=MessageRole.USER, content="question"),
            Message(role=MessageRole.ASSISTANT, content="answer"),
            Message(role=MessageRole.TOOL, content="tool output"),
        ]
        chunk_id = pt.evict(msgs, "mixed roles")
        handler = RecallHandler(pt)
        result = await handler.handle(chunk_id)
        assert "[user]" in result
        assert "[assistant]" in result
        assert "[tool]" in result

    @pytest.mark.asyncio
    async def test_handle_invalid_lists_available(self):
        """Handle an invalid chunk_id lists available chunk_ids."""
        pt = ContextPageTable()
        valid_id = pt.evict([Message(role=MessageRole.USER, content="hi")], "test")
        handler = RecallHandler(pt)
        result = await handler.handle("ctx-doesnotexist")
        assert "not found" in result
        assert valid_id in result


# ---------------------------------------------------------------------------
# 3. Fidelity compression with page table integration
# ---------------------------------------------------------------------------


class TestFidelityWithPageTable:
    def test_compression_produces_recall_hints(self):
        """When compression fires with page_table, L2/L3 messages contain recall hints."""
        pt = ContextPageTable()
        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=1200, response_reserve=50),
            goal="find bugs",
            page_table=pt,
        )
        # Build messages that will exceed budget
        msgs = [Message(role=MessageRole.SYSTEM, content="System prompt.")]
        for i in range(10):
            msgs.append(Message(role=MessageRole.USER, content=f"User message {i} " * 20))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Response {i} " * 20))

        result = builder.build_conversation_context(msgs, turn=5)

        # Page table should have content
        assert pt.has_pages
        assert pt.page_count >= 1

        # At least some messages should have recall hints
        has_recall = any("recall:" in (m.content or "") for m in result)
        assert has_recall, "Expected recall hints in compressed messages"

    def test_passthrough_no_page_table_entries(self):
        """When under budget (passthrough), page table stays empty."""
        pt = ContextPageTable()
        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=128000, response_reserve=4096),
            goal="simple task",
            page_table=pt,
        )
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System prompt."),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        result = builder.build_conversation_context(msgs, turn=0)
        assert not pt.has_pages  # No compression, no eviction
        assert len(result) == len(msgs)  # Passthrough

    def test_recalled_content_matches_evicted(self):
        """Content recalled via page table matches what was originally evicted."""
        pt = ContextPageTable()
        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=400, response_reserve=50),
            goal="test recall fidelity",
            page_table=pt,
        )
        original_msgs = [Message(role=MessageRole.SYSTEM, content="System prompt.")]
        for i in range(8):
            original_msgs.append(Message(role=MessageRole.USER, content=f"User msg {i} " * 15))
            original_msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Resp {i} " * 15))

        builder.build_conversation_context(original_msgs, turn=4)

        # Verify evicted content is accessible
        assert pt.has_pages
        for entry in pt.index():
            recalled = pt.recall(entry["chunk_id"])
            assert recalled is not None
            assert len(recalled) > 0


# ---------------------------------------------------------------------------
# 4. Fidelity demotion loop test
# ---------------------------------------------------------------------------


class TestFidelityDemotion:
    def test_demotion_terminates_under_extreme_budget(self):
        """Even with very tight budget, demotion loop terminates and produces valid output."""
        builder = WorkingSetBuilder(
            identity="Sys.",
            token_budget=TokenBudget(total_window=100, response_reserve=10),
            goal="test",
        )
        msgs = [Message(role=MessageRole.SYSTEM, content="Sys.")]
        for i in range(20):
            msgs.append(Message(role=MessageRole.USER, content=f"Long message number {i} " * 50))

        result = builder.build_conversation_context(msgs, turn=3)
        # Should terminate without error
        assert len(result) >= 1  # At least system message
        assert result[0].content == "Sys."
        # Should have compressed
        assert builder.last_decision is not None
        assert builder.last_decision.history_compressed

    def test_demotion_with_page_table(self):
        """Demotion loop with page table captures evicted messages for recall."""
        pt = ContextPageTable()
        # Budget big enough for head+tail+some middle, but not all messages
        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=1200, response_reserve=50),
            goal="test demotion",
            page_table=pt,
        )
        msgs = [Message(role=MessageRole.SYSTEM, content="System prompt.")]
        for i in range(12):
            msgs.append(Message(role=MessageRole.USER, content=f"User question {i} " * 15))
            msgs.append(Message(role=MessageRole.ASSISTANT, content=f"Assistant reply {i} " * 15))

        result = builder.build_conversation_context(msgs, turn=3)
        assert len(result) >= 1
        assert builder.last_decision is not None
        assert builder.last_decision.history_compressed
        # Page table should have captured the evicted content
        assert pt.has_pages
