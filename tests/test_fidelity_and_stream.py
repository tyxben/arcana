"""Tests for Fidelity Spectrum context compression and StreamAccumulator thinking."""

from __future__ import annotations

from arcana.context.builder import WorkingSetBuilder
from arcana.contracts.context import TokenBudget
from arcana.contracts.llm import Message, MessageRole, StreamChunk, TokenUsage
from arcana.runtime.stream_accumulator import StreamAccumulator

# ---------------------------------------------------------------------------
# 1. Fidelity level assignment tests
# ---------------------------------------------------------------------------


class TestFidelityLevelAssignment:
    def _make_builder(self, window: int = 2000) -> WorkingSetBuilder:
        return WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=window, response_reserve=100),
            goal="find authentication bugs in the codebase",
        )

    def _force_compression(
        self, builder: WorkingSetBuilder, msg_count: int = 15, msg_length: int = 30
    ) -> list[Message]:
        """Build a message list that exceeds budget to trigger compression."""
        msgs = [Message(role=MessageRole.SYSTEM, content="System prompt.")]
        for i in range(msg_count):
            if i % 3 == 0:
                msgs.append(
                    Message(
                        role=MessageRole.USER,
                        content=f"User question about auth bugs number {i} "
                        * msg_length,
                    )
                )
            elif i % 3 == 1:
                msgs.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=f"The authentication issue is in module {i} "
                        * msg_length,
                    )
                )
            else:
                msgs.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=f"Error in auth handler at line {i}: authentication failed "
                        * msg_length,
                    )
                )
        return msgs

    def test_fidelity_preserves_message_roles(self) -> None:
        """Compressed messages preserve their original roles (not all USER)."""
        builder = self._make_builder(window=1500)
        msgs = self._force_compression(builder)
        result = builder.build_conversation_context(msgs, turn=5)

        # Should have messages with different roles (not just system + one USER summary)
        roles = set()
        for m in result:
            r = m.role.value if hasattr(m.role, "value") else str(m.role)
            roles.add(r)
        # At minimum system + user should be present, and ideally assistant/tool too
        assert "system" in roles
        assert len(roles) >= 2

    def test_fidelity_report_has_distribution(self) -> None:
        """ContextReport includes fidelity_distribution when compression fires."""
        # Window must be small enough that total > budget (triggers compression)
        # but large enough that fidelity-graded compression can fit within budget
        # (avoiding fallback to aggressive_truncate which loses fidelity metadata).
        builder = self._make_builder(window=4500)
        msgs = self._force_compression(builder)
        builder.build_conversation_context(msgs, turn=5)

        report = builder.last_report
        assert report is not None
        assert report.compression_applied
        # fidelity_distribution should have at least one entry
        assert len(report.fidelity_distribution) > 0
        # All keys should be L0-L3
        for key in report.fidelity_distribution:
            assert key in ("L0", "L1", "L2", "L3")

    def test_fidelity_no_distribution_on_passthrough(self) -> None:
        """ContextReport has empty fidelity_distribution when no compression needed."""
        builder = self._make_builder(window=128000)
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System prompt."),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        builder.build_conversation_context(msgs, turn=0)
        report = builder.last_report
        assert report is not None
        assert not report.compression_applied
        assert report.fidelity_distribution == {}

    def test_compressed_marker_in_l2(self) -> None:
        """L2 messages contain [compressed] marker."""
        # Use a window tight enough to force L2/L3 markers but not so tight
        # that aggressive_truncate takes over (which strips fidelity markers).
        builder = self._make_builder(window=4000)
        msgs = self._force_compression(builder, msg_count=20, msg_length=20)
        result = builder.build_conversation_context(msgs, turn=5)

        # Look for [compressed] marker in any message
        has_compressed = any("[compressed" in (m.content or "") for m in result)
        # With enough messages in a tight window, we should see L2 or L3
        has_earlier = any("(earlier message)" in (m.content or "") for m in result)
        assert has_compressed or has_earlier, (
            "Expected L2 or L3 markers in compressed output"
        )


# ---------------------------------------------------------------------------
# 2. StreamAccumulator thinking tests
# ---------------------------------------------------------------------------


class TestStreamAccumulatorThinking:
    def test_thinking_accumulation(self) -> None:
        """Thinking deltas are accumulated and available in response."""
        acc = StreamAccumulator(model="test-model")
        acc.feed(StreamChunk(type="thinking_delta", thinking="Let me think..."))
        acc.feed(StreamChunk(type="thinking_delta", thinking=" The answer is 42."))
        acc.feed(StreamChunk(type="text_delta", text="The answer is 42."))
        acc.feed(
            StreamChunk(
                type="done",
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )
        )

        assert acc.thinking == "Let me think... The answer is 42."
        response = acc.to_response()
        assert response.content == "The answer is 42."
        assert response.anthropic is not None
        assert response.anthropic.thinking_blocks is not None
        assert len(response.anthropic.thinking_blocks) == 1
        assert "Let me think" in response.anthropic.thinking_blocks[0].thinking

    def test_no_thinking_no_anthropic_ext(self) -> None:
        """When no thinking chunks received, anthropic extension is None."""
        acc = StreamAccumulator(model="test-model")
        acc.feed(StreamChunk(type="text_delta", text="Hello"))
        acc.feed(StreamChunk(type="done"))

        response = acc.to_response()
        assert response.content == "Hello"
        assert response.anthropic is None

    def test_interleaved_thinking_and_text(self) -> None:
        """Thinking and text chunks arriving in mixed order are accumulated separately."""
        acc = StreamAccumulator(model="test-model")
        acc.feed(StreamChunk(type="thinking_delta", thinking="Hmm..."))
        acc.feed(StreamChunk(type="text_delta", text="Part 1"))
        acc.feed(StreamChunk(type="thinking_delta", thinking=" more thought"))
        acc.feed(StreamChunk(type="text_delta", text=" Part 2"))
        acc.feed(StreamChunk(type="done"))

        assert acc.thinking == "Hmm... more thought"
        assert acc.text == "Part 1 Part 2"
        response = acc.to_response()
        assert response.anthropic is not None
        assert response.anthropic.thinking_blocks is not None
        assert response.anthropic.thinking_blocks[0].thinking == "Hmm... more thought"

    def test_tool_call_accumulation(self) -> None:
        """Tool call deltas are correctly assembled."""
        acc = StreamAccumulator(model="test-model")
        acc.feed(
            StreamChunk(
                type="tool_call_delta", tool_call_id="tc1", tool_name="search"
            )
        )
        acc.feed(
            StreamChunk(
                type="tool_call_delta", tool_call_id="tc1", arguments_delta='{"qu'
            )
        )
        acc.feed(
            StreamChunk(
                type="tool_call_delta",
                tool_call_id="tc1",
                arguments_delta='ery": "test"}',
            )
        )
        acc.feed(StreamChunk(type="done"))

        response = acc.to_response()
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments == '{"query": "test"}'

    def test_empty_accumulator(self) -> None:
        """Empty accumulator produces valid response with defaults."""
        acc = StreamAccumulator(model="test-model")
        response = acc.to_response()
        assert response.content is None
        assert response.tool_calls is None
        assert response.model == "test-model"
        assert response.finish_reason == "stop"


# ---------------------------------------------------------------------------
# 3. Memory injection test (cache topology)
# ---------------------------------------------------------------------------


class TestMemoryInjectionCacheTopology:
    def test_memory_does_not_modify_system_prompt(self) -> None:
        """Memory injection keeps system prompt untouched (for prompt cache stability)."""
        builder = WorkingSetBuilder(
            identity="You are a helpful assistant.",
            token_budget=TokenBudget(total_window=128000, response_reserve=4096),
        )
        msgs = [
            Message(
                role=MessageRole.SYSTEM, content="You are a helpful assistant."
            ),
            Message(role=MessageRole.USER, content="What is 2+2?"),
        ]
        result = builder.build_conversation_context(
            msgs, memory_context="User prefers concise answers.", turn=0
        )

        # System prompt should be UNCHANGED
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "You are a helpful assistant."
        # Memory should be merged into user message
        user_msgs = [m for m in result if m.role == MessageRole.USER]
        assert len(user_msgs) >= 1
        user_text = " ".join(m.content or "" for m in user_msgs)
        assert "User prefers concise answers" in user_text
        assert "What is 2+2?" in user_text

    def test_no_consecutive_user_messages(self) -> None:
        """Memory injection must not create back-to-back user messages."""
        builder = WorkingSetBuilder(
            identity="System.",
            token_budget=TokenBudget(total_window=128000, response_reserve=4096),
        )
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System."),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        result = builder.build_conversation_context(
            msgs, memory_context="Some context.", turn=0
        )

        # Check no consecutive same-role messages
        for i in range(1, len(result)):
            prev_role = (
                result[i - 1].role.value
                if hasattr(result[i - 1].role, "value")
                else str(result[i - 1].role)
            )
            curr_role = (
                result[i].role.value
                if hasattr(result[i].role, "value")
                else str(result[i].role)
            )
            if prev_role == "system":
                continue  # system -> user is fine
            assert not (prev_role == curr_role == "user"), (
                f"Consecutive user messages at index {i - 1} and {i}"
            )
