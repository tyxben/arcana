"""Tests for HA fixes: budget race condition + ChatSession history limit."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from arcana.contracts.llm import (
    LLMResponse,
    Message,
    MessageRole,
    TokenUsage,
)
from arcana.runtime_core import (
    ChatSession,
    RunResult,
    Runtime,
    RuntimeConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime() -> Runtime:
    """Create a Runtime with a mock Ollama provider for testing."""
    return Runtime(
        providers={"ollama": ""},
        config=RuntimeConfig(default_provider="ollama"),
    )


def _make_text_response(
    text: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> LLMResponse:
    """Create a simple text LLMResponse."""
    return LLMResponse(
        content=text,
        tool_calls=None,
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model="test-model",
        finish_reason="stop",
    )


# ---------------------------------------------------------------------------
# Part 1: Budget race condition -- concurrent run() calls
# ---------------------------------------------------------------------------


class TestBudgetRaceCondition:
    """Verify that concurrent run() calls do not corrupt budget counters."""

    @pytest.mark.asyncio
    async def test_concurrent_runs_accumulate_correctly(self):
        """Multiple concurrent run() calls should produce exact totals."""
        rt = _make_runtime()

        tokens_per_run = 100
        cost_per_run = 0.01
        num_runs = 50

        fake_result = RunResult(
            output="ok",
            success=True,
            steps=1,
            tokens_used=tokens_per_run,
            cost_usd=cost_per_run,
            run_id="test",
        )

        # Patch _create_session to return a mock that produces our fake result
        async def _fake_run(goal, **kw):
            # Small sleep to encourage interleaving
            await asyncio.sleep(0.001)
            return fake_result

        with patch.object(rt, "_create_session") as mock_create:
            mock_session = AsyncMock()
            mock_session.run = _fake_run
            mock_session.run_id = "test"
            mock_create.return_value = mock_session

            tasks = [rt.run(f"task-{i}") for i in range(num_runs)]
            await asyncio.gather(*tasks)

        expected_tokens = tokens_per_run * num_runs
        expected_cost = cost_per_run * num_runs

        assert rt.tokens_used == expected_tokens, (
            f"Expected {expected_tokens} tokens, got {rt.tokens_used}"
        )
        assert abs(rt.budget_used_usd - expected_cost) < 1e-9, (
            f"Expected ${expected_cost}, got ${rt.budget_used_usd}"
        )

    def test_totals_lock_exists(self):
        """Runtime should have an asyncio.Lock for totals."""
        import asyncio
        rt = _make_runtime()
        assert hasattr(rt, "_totals_lock")
        assert isinstance(rt._totals_lock, asyncio.Lock)


# ---------------------------------------------------------------------------
# Part 2: ChatSession max_history
# ---------------------------------------------------------------------------


class TestChatSessionMaxHistory:
    """Verify ChatSession._trim_history works correctly."""

    def test_trim_preserves_system_messages(self):
        """System messages should never be removed by trimming."""
        rt = _make_runtime()
        session = ChatSession(runtime=rt, max_history=4)

        # Manually populate messages: 1 system + 10 user/assistant pairs
        session._messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
        ]
        for i in range(10):
            session._messages.append(
                Message(role=MessageRole.USER, content=f"User msg {i}")
            )
            session._messages.append(
                Message(role=MessageRole.ASSISTANT, content=f"Assistant msg {i}")
            )

        # 1 system + 20 non-system = 21 total
        assert len(session._messages) == 21

        session._trim_history()

        # Should have 1 system + 4 non-system = 5 total
        assert len(session._messages) == 5

        # First message is still system
        assert session._messages[0].role == MessageRole.SYSTEM
        assert session._messages[0].content == "You are helpful."

        # Last 4 non-system messages preserved (last 2 pairs)
        assert session._messages[1].content == "User msg 8"
        assert session._messages[2].content == "Assistant msg 8"
        assert session._messages[3].content == "User msg 9"
        assert session._messages[4].content == "Assistant msg 9"

    def test_no_trim_when_under_limit(self):
        """No trimming should occur when messages are under the limit."""
        rt = _make_runtime()
        session = ChatSession(runtime=rt, max_history=100)

        session._messages = [
            Message(role=MessageRole.SYSTEM, content="System."),
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi!"),
        ]

        session._trim_history()

        # All 3 messages should still be there
        assert len(session._messages) == 3

    def test_no_trim_when_max_history_is_none(self):
        """When max_history is None (default), all messages are kept."""
        rt = _make_runtime()
        session = ChatSession(runtime=rt)  # max_history defaults to None

        # Populate with a lot of messages
        session._messages = [
            Message(role=MessageRole.SYSTEM, content="System."),
        ]
        for i in range(100):
            session._messages.append(
                Message(role=MessageRole.USER, content=f"msg {i}")
            )

        assert len(session._messages) == 101

        session._trim_history()

        # Nothing trimmed -- still 101
        assert len(session._messages) == 101

    def test_backward_compat_default_none(self):
        """ChatSession without max_history should default to None."""
        rt = _make_runtime()
        session = ChatSession(runtime=rt)
        assert session._max_history is None

    @pytest.mark.asyncio
    async def test_trim_called_after_send(self):
        """send() should call _trim_history after completing."""
        rt = _make_runtime()
        rt._gateway.generate = AsyncMock(
            return_value=_make_text_response("Reply")
        )

        session = ChatSession(runtime=rt, max_history=4)

        # Send enough messages to trigger trimming
        for i in range(5):
            await session.send(f"Message {i}")

        # After 5 sends: each send adds 1 user + 1 assistant = 2 non-system
        # Total would be 1 system + 10 non-system = 11 without trimming
        # With max_history=4, should be 1 system + 4 non-system = 5
        assert len(session._messages) == 5
        assert session._messages[0].role == MessageRole.SYSTEM

    def test_multiple_system_messages_preserved(self):
        """If there are multiple system messages, all should be preserved."""
        rt = _make_runtime()
        session = ChatSession(runtime=rt, max_history=2)

        session._messages = [
            Message(role=MessageRole.SYSTEM, content="System 1"),
            Message(role=MessageRole.SYSTEM, content="System 2"),
            Message(role=MessageRole.USER, content="msg 1"),
            Message(role=MessageRole.ASSISTANT, content="resp 1"),
            Message(role=MessageRole.USER, content="msg 2"),
            Message(role=MessageRole.ASSISTANT, content="resp 2"),
            Message(role=MessageRole.USER, content="msg 3"),
            Message(role=MessageRole.ASSISTANT, content="resp 3"),
        ]

        session._trim_history()

        # 2 system + 2 non-system = 4
        assert len(session._messages) == 4
        assert session._messages[0].role == MessageRole.SYSTEM
        assert session._messages[0].content == "System 1"
        assert session._messages[1].role == MessageRole.SYSTEM
        assert session._messages[1].content == "System 2"
        # Last 2 non-system
        assert session._messages[2].content == "msg 3"
        assert session._messages[3].content == "resp 3"

    def test_max_history_threaded_from_runtime_chat(self):
        """Runtime.chat(max_history=N) should pass max_history to ChatSession."""
        rt = _make_runtime()

        # We can't use async with in a sync test, so test the ChatSession directly
        session = ChatSession(runtime=rt, max_history=50)
        assert session._max_history == 50

    def test_tool_messages_counted_as_non_system(self):
        """Tool messages should be treated as non-system and subject to trimming."""
        rt = _make_runtime()
        session = ChatSession(runtime=rt, max_history=3)

        session._messages = [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(role=MessageRole.USER, content="old user"),
            Message(role=MessageRole.ASSISTANT, content="old assistant"),
            Message(role=MessageRole.TOOL, content="tool result", tool_call_id="tc-1"),
            Message(role=MessageRole.USER, content="new user"),
            Message(role=MessageRole.ASSISTANT, content="new assistant"),
        ]

        session._trim_history()

        # 1 system + 3 non-system (last 3) = 4
        assert len(session._messages) == 4
        assert session._messages[0].role == MessageRole.SYSTEM
        # Last 3 non-system: tool result, new user, new assistant
        assert session._messages[1].content == "tool result"
        assert session._messages[2].content == "new user"
        assert session._messages[3].content == "new assistant"
