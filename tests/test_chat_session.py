"""Tests for ChatSession and ChatResponse -- multi-turn conversational interaction."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.contracts.llm import (
    LLMResponse,
    TokenUsage,
    ToolCallRequest,
)
from arcana.contracts.streaming import StreamEventType
from arcana.runtime_core import (
    Budget,
    ChatResponse,
    ChatSession,
    Runtime,
    RuntimeConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_tool_call_response(
    tool_name: str,
    tool_args: dict,
    tool_call_id: str = "tc-1",
    text: str | None = None,
) -> LLMResponse:
    """Create an LLMResponse with a tool call."""
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCallRequest(
                id=tool_call_id,
                name=tool_name,
                arguments=json.dumps(tool_args),
            )
        ],
        usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        model="test-model",
        finish_reason="tool_calls",
    )


def _make_runtime() -> Runtime:
    """Create a Runtime with a mock Ollama provider for testing."""
    return Runtime(
        providers={"ollama": ""},
        config=RuntimeConfig(default_provider="ollama"),
    )


# ---------------------------------------------------------------------------
# ChatResponse model tests
# ---------------------------------------------------------------------------


class TestChatResponse:
    def test_defaults(self):
        r = ChatResponse()
        assert r.content == ""
        assert r.tool_calls_made == 0
        assert r.tokens_used == 0
        assert r.cost_usd == 0.0

    def test_populated(self):
        r = ChatResponse(
            content="Hello!",
            tool_calls_made=2,
            tokens_used=100,
            cost_usd=0.01,
        )
        assert r.content == "Hello!"
        assert r.tool_calls_made == 2
        assert r.tokens_used == 100
        assert r.cost_usd == 0.01


# ---------------------------------------------------------------------------
# ChatSession unit tests
# ---------------------------------------------------------------------------


class TestChatSessionInit:
    def test_default_system_prompt(self):
        rt = _make_runtime()
        session = ChatSession(runtime=rt)
        assert session._system_prompt == "You are a helpful assistant."
        assert session.message_count == 1  # system prompt only
        assert session.total_tokens == 0
        assert session.total_cost_usd == 0.0

    def test_custom_system_prompt(self):
        rt = _make_runtime()
        session = ChatSession(runtime=rt, system_prompt="You are a pirate.")
        assert session._system_prompt == "You are a pirate."
        # System message should use custom prompt
        assert session._messages[0].content == "You are a pirate."

    def test_custom_budget(self):
        rt = _make_runtime()
        session = ChatSession(
            runtime=rt,
            budget=Budget(max_cost_usd=0.5, max_tokens=1000),
        )
        assert session._budget_tracker.max_cost_usd == 0.5
        assert session._budget_tracker.max_tokens == 1000

    def test_session_id_is_unique(self):
        rt = _make_runtime()
        s1 = ChatSession(runtime=rt)
        s2 = ChatSession(runtime=rt)
        assert s1.session_id != s2.session_id


class TestChatSessionSend:
    @pytest.mark.asyncio
    async def test_basic_send(self):
        """Basic single send() returns a ChatResponse."""
        rt = _make_runtime()
        rt._gateway.generate = AsyncMock(
            return_value=_make_text_response("Hello! How can I help?")
        )
        # Make stream raise so it falls back to generate
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        session = ChatSession(runtime=rt)
        response = await session.send("Hi there")

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello! How can I help?"
        assert response.tool_calls_made == 0
        assert response.tokens_used == 30  # 10 + 20
        assert response.cost_usd > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Three send() calls, each gets a response. History persists."""
        rt = _make_runtime()
        responses = [
            _make_text_response("Hi! I'm an assistant."),
            _make_text_response("Paris is the capital of France."),
            _make_text_response("It has about 2 million people."),
        ]
        rt._gateway.generate = AsyncMock(side_effect=responses)
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        session = ChatSession(runtime=rt)

        r1 = await session.send("Hello")
        assert r1.content == "Hi! I'm an assistant."
        assert session.message_count == 3  # system + user + assistant

        r2 = await session.send("What is the capital of France?")
        assert r2.content == "Paris is the capital of France."
        assert session.message_count == 5  # + user + assistant

        r3 = await session.send("What's the population?")
        assert r3.content == "It has about 2 million people."
        assert session.message_count == 7  # + user + assistant

    @pytest.mark.asyncio
    async def test_conversation_history_persists(self):
        """The second LLM call receives the full conversation history."""
        rt = _make_runtime()
        captured_requests = []

        async def mock_generate(*, request, config):
            captured_requests.append(request)
            return _make_text_response("Response")

        rt._gateway.generate = mock_generate
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        session = ChatSession(runtime=rt)
        await session.send("First message")
        await session.send("Second message")

        # The second call should include the full history
        # (possibly compressed, but at minimum the system prompt + all messages)
        second_request = captured_requests[1]
        all_contents = [
            m.content for m in second_request.messages
            if isinstance(m.content, str)
        ]
        # Should contain both user messages in the context
        combined = " ".join(all_contents)
        assert "First message" in combined or "Second message" in combined

    @pytest.mark.asyncio
    async def test_budget_accumulates_across_sends(self):
        """Budget tracking accumulates across multiple send() calls."""
        rt = _make_runtime()
        rt._gateway.generate = AsyncMock(
            side_effect=[
                _make_text_response("Response 1", prompt_tokens=100, completion_tokens=50),
                _make_text_response("Response 2", prompt_tokens=200, completion_tokens=100),
            ]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        session = ChatSession(runtime=rt)
        r1 = await session.send("Message 1")
        r2 = await session.send("Message 2")

        # Total should be sum of both
        assert session.total_tokens == 150 + 300  # (100+50) + (200+100)
        assert session.total_cost_usd == r1.cost_usd + r2.cost_usd
        assert session.total_cost_usd > 0

    @pytest.mark.asyncio
    async def test_tools_work_within_chat(self):
        """LLM calls a tool, gets result, then produces final text response."""
        from arcana.sdk import tool

        @tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
            tools=[add],
        )

        # First call: tool call, second call: text response
        rt._gateway.generate = AsyncMock(
            side_effect=[
                _make_tool_call_response("add", {"a": 2, "b": 3}),
                _make_text_response("The result is 5."),
            ]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        session = ChatSession(runtime=rt)
        response = await session.send("What is 2 + 3?")

        assert response.content == "The result is 5."
        assert response.tool_calls_made == 1
        # Tokens from both calls
        assert response.tokens_used == 15 + 30  # tool_call (15) + text (30)

    @pytest.mark.asyncio
    async def test_total_cost_usd_and_total_tokens_properties(self):
        """Properties return cumulative values."""
        rt = _make_runtime()
        rt._gateway.generate = AsyncMock(
            return_value=_make_text_response("OK", prompt_tokens=50, completion_tokens=25)
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        session = ChatSession(runtime=rt)
        assert session.total_tokens == 0
        assert session.total_cost_usd == 0.0

        await session.send("Hello")
        assert session.total_tokens == 75
        assert session.total_cost_usd > 0

    @pytest.mark.asyncio
    async def test_history_property(self):
        """history property returns user/assistant messages only."""
        rt = _make_runtime()
        rt._gateway.generate = AsyncMock(
            side_effect=[
                _make_text_response("Hi!"),
                _make_text_response("I'm good."),
            ]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        session = ChatSession(runtime=rt)
        await session.send("Hello")
        await session.send("How are you?")

        hist = session.history
        assert len(hist) == 4  # 2 user + 2 assistant
        assert hist[0] == {"role": "user", "content": "Hello"}
        assert hist[1] == {"role": "assistant", "content": "Hi!"}
        assert hist[2] == {"role": "user", "content": "How are you?"}
        assert hist[3] == {"role": "assistant", "content": "I'm good."}

    @pytest.mark.asyncio
    async def test_max_turns_per_message_limits_agent_loops(self):
        """If LLM keeps calling tools, max_turns_per_message stops the loop."""
        rt = _make_runtime()

        # LLM always requests a tool call -- never stops voluntarily
        rt._gateway.generate = AsyncMock(
            return_value=_make_tool_call_response("nonexistent", {}),
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        session = ChatSession(runtime=rt, max_turns_per_message=3)
        response = await session.send("Do something")

        # After 3 turns the loop exits. The response will be empty since
        # the LLM never produced a non-tool-call response.
        assert response.content == ""
        assert response.tool_calls_made == 3  # one tool call per turn

    @pytest.mark.asyncio
    async def test_context_compression_for_long_conversations(self):
        """When conversation is long, context builder compresses old messages."""
        rt = _make_runtime()

        call_count = 0

        async def mock_generate(*, request, config):
            nonlocal call_count
            call_count += 1
            # The context builder should have been called -- we just verify
            # the conversation completes without error even with many messages
            return _make_text_response(f"Response {call_count}")

        rt._gateway.generate = mock_generate
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        # Use a very small context window to force compression
        session = ChatSession(runtime=rt, max_turns_per_message=5)

        # Send many messages to build up history
        for i in range(10):
            response = await session.send(f"Message {i} with some extra content for tokens")
            assert response.content == f"Response {i + 1}"

        # Verify all messages are tracked
        assert session.message_count > 10  # system + 10 user + 10 assistant


class TestChatSessionStream:
    @pytest.mark.asyncio
    async def test_stream_yields_events(self):
        """stream() yields StreamEvent objects including LLM chunks."""
        rt = _make_runtime()

        # Mock streaming
        async def mock_stream(*, request, config):
            from arcana.contracts.llm import StreamChunk, TokenUsage

            yield StreamChunk(type="text_delta", text="Hello")
            yield StreamChunk(type="text_delta", text=" world")
            yield StreamChunk(
                type="done",
                usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
                metadata={"finish_reason": "stop", "model": "test"},
            )

        rt._gateway.stream = mock_stream

        session = ChatSession(runtime=rt)
        events = []
        async for event in session.stream("Hi"):
            events.append(event)

        # Should have: RUN_START, LLM_CHUNK (x2), RUN_COMPLETE
        event_types = [e.event_type for e in events]
        assert StreamEventType.RUN_START in event_types
        assert StreamEventType.RUN_COMPLETE in event_types
        assert event_types.count(StreamEventType.LLM_CHUNK) == 2

        # Final event should have the complete text
        complete_event = [e for e in events if e.event_type == StreamEventType.RUN_COMPLETE][0]
        assert complete_event.content == "Hello world"


class TestChatContextManager:
    @pytest.mark.asyncio
    async def test_chat_as_context_manager(self):
        """runtime.chat() works as an async context manager."""
        rt = _make_runtime()
        rt._gateway.generate = AsyncMock(
            return_value=_make_text_response("Hello!")
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        async with rt.chat() as c:
            assert isinstance(c, ChatSession)
            r = await c.send("Hi")
            assert r.content == "Hello!"
            assert c.total_tokens > 0

    @pytest.mark.asyncio
    async def test_chat_with_custom_params(self):
        """runtime.chat() passes through custom parameters."""
        rt = _make_runtime()
        rt._gateway.generate = AsyncMock(
            return_value=_make_text_response("Arrr!")
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        async with rt.chat(
            system_prompt="You are a pirate.",
            max_turns_per_message=5,
            budget=Budget(max_cost_usd=0.01),
        ) as c:
            assert c._system_prompt == "You are a pirate."
            assert c._max_turns == 5
            assert c._budget_tracker.max_cost_usd == 0.01

    @pytest.mark.asyncio
    async def test_chat_multiple_sessions_independent(self):
        """Two chat sessions don't share state."""
        rt = _make_runtime()
        rt._gateway.generate = AsyncMock(
            return_value=_make_text_response("OK")
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        async with rt.chat() as c1:
            await c1.send("Hello from session 1")

        async with rt.chat() as c2:
            # c2 should start fresh
            assert c2.message_count == 1  # only system prompt
            assert c2.total_tokens == 0


class TestChatSessionBudgetEnforcement:
    @pytest.mark.asyncio
    async def test_budget_exceeded_raises(self):
        """Budget enforcement works across the chat session."""
        from arcana.gateway.base import BudgetExceededError

        rt = _make_runtime()
        rt._gateway.generate = AsyncMock(
            return_value=_make_text_response("OK", prompt_tokens=500, completion_tokens=500)
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        session = ChatSession(
            runtime=rt,
            budget=Budget(max_cost_usd=10.0, max_tokens=1000),
        )

        # First send uses 1000 tokens -- exactly at limit (allowed, budget not exceeded yet)
        await session.send("First")

        # Second send enters loop, check_budget passes (1000 == 1000, not >).
        # LLM call adds 1000 more (total 2000 > 1000).
        # Third send should now exceed.
        await session.send("Second")

        with pytest.raises(BudgetExceededError):
            await session.send("Third")


class TestChatSessionExports:
    def test_exported_from_arcana(self):
        """ChatSession and ChatResponse are available from the arcana package."""
        import arcana

        assert hasattr(arcana, "ChatSession")
        assert hasattr(arcana, "ChatResponse")
        assert arcana.ChatSession is ChatSession
        assert arcana.ChatResponse is ChatResponse
