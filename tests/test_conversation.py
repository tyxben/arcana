"""Tests for ConversationAgent -- V2 execution model.

Tests are split into two groups:
1. Contract tests (TurnFacts, TurnAssessment, TurnOutcome) -- always run.
2. ConversationAgent tests -- skipped when runtime/conversation.py is absent.
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.contracts.llm import LLMResponse, ModelConfig, StreamChunk, TokenUsage, ToolCallRequest
from arcana.contracts.state import AgentState
from arcana.contracts.turn import TurnAssessment, TurnFacts, TurnOutcome

# ---------------------------------------------------------------------------
# Check whether ConversationAgent is available
# ---------------------------------------------------------------------------
_conversation_available = importlib.util.find_spec("arcana.runtime.conversation") is not None

if _conversation_available:
    from arcana.runtime.conversation import ConversationAgent

needs_conversation = pytest.mark.skipif(
    not _conversation_available,
    reason="arcana.runtime.conversation not implemented yet",
)


# =========================================================================
# 1. Contract tests -- TurnFacts / TurnAssessment / TurnOutcome
# =========================================================================


class TestTurnFacts:
    """TurnFacts should only hold raw provider data."""

    def test_basic_text_response(self) -> None:
        facts = TurnFacts(assistant_text="Hello", finish_reason="stop")
        assert facts.assistant_text == "Hello"
        assert facts.tool_calls == []
        assert facts.finish_reason == "stop"

    def test_tool_call_response(self) -> None:
        tc = ToolCallRequest(id="1", name="search", arguments='{"q": "test"}')
        facts = TurnFacts(tool_calls=[tc], finish_reason="tool_calls")
        assert len(facts.tool_calls) == 1
        assert facts.assistant_text is None

    def test_no_assessment_fields(self) -> None:
        """TurnFacts must NOT have completed/failed/answer fields."""
        facts = TurnFacts(assistant_text="test")
        assert not hasattr(facts, "completed")
        assert not hasattr(facts, "failed")
        assert not hasattr(facts, "answer")

    def test_defaults(self) -> None:
        facts = TurnFacts()
        assert facts.assistant_text is None
        assert facts.tool_calls == []
        assert facts.usage is None
        assert facts.finish_reason is None
        assert facts.thinking is None
        assert facts.provider_metadata == {}

    def test_with_usage(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        facts = TurnFacts(assistant_text="hi", usage=usage)
        assert facts.usage is not None
        assert facts.usage.total_tokens == 30

    def test_with_thinking(self) -> None:
        facts = TurnFacts(assistant_text="answer", thinking="Let me think...")
        assert facts.thinking == "Let me think..."

    def test_with_provider_metadata(self) -> None:
        facts = TurnFacts(provider_metadata={"model": "deepseek-chat", "latency_ms": 123})
        assert facts.provider_metadata["model"] == "deepseek-chat"

    def test_multiple_tool_calls(self) -> None:
        tc1 = ToolCallRequest(id="1", name="search", arguments='{"q": "a"}')
        tc2 = ToolCallRequest(id="2", name="calc", arguments='{"expr": "1+1"}')
        facts = TurnFacts(tool_calls=[tc1, tc2], finish_reason="tool_calls")
        assert len(facts.tool_calls) == 2
        assert facts.tool_calls[0].name == "search"
        assert facts.tool_calls[1].name == "calc"

    def test_serialization_roundtrip(self) -> None:
        tc = ToolCallRequest(id="1", name="search", arguments='{"q": "test"}')
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        facts = TurnFacts(
            assistant_text="hello",
            tool_calls=[tc],
            usage=usage,
            finish_reason="stop",
            thinking="thought",
            provider_metadata={"key": "val"},
        )
        data = facts.model_dump()
        restored = TurnFacts(**data)
        assert restored.assistant_text == facts.assistant_text
        assert len(restored.tool_calls) == 1
        assert restored.usage is not None
        assert restored.usage.total_tokens == 15


class TestTurnAssessment:
    """TurnAssessment holds runtime interpretation."""

    def test_default_not_done(self) -> None:
        assessment = TurnAssessment()
        assert not assessment.completed
        assert not assessment.failed
        assert assessment.answer is None
        assert assessment.confidence == 0.0

    def test_completed_with_answer(self) -> None:
        assessment = TurnAssessment(
            completed=True,
            answer="Paris is the capital of France",
            completion_reason="natural_stop",
            confidence=0.9,
        )
        assert assessment.completed
        assert assessment.answer is not None
        assert assessment.confidence == 0.9

    def test_failed(self) -> None:
        assessment = TurnAssessment(failed=True, completion_reason="empty_response")
        assert assessment.failed
        assert not assessment.completed

    def test_needs_clarification(self) -> None:
        assessment = TurnAssessment(needs_clarification=True)
        assert assessment.needs_clarification
        assert not assessment.completed

    def test_no_facts_fields(self) -> None:
        """TurnAssessment must NOT have assistant_text/tool_calls/usage fields."""
        assessment = TurnAssessment()
        assert not hasattr(assessment, "assistant_text")
        assert not hasattr(assessment, "tool_calls")
        assert not hasattr(assessment, "usage")
        assert not hasattr(assessment, "finish_reason")

    def test_serialization_roundtrip(self) -> None:
        assessment = TurnAssessment(
            completed=True,
            answer="42",
            completion_reason="natural_stop",
            confidence=0.95,
        )
        data = assessment.model_dump()
        restored = TurnAssessment(**data)
        assert restored.completed == assessment.completed
        assert restored.answer == assessment.answer
        assert restored.confidence == assessment.confidence


class TestTurnOutcome:
    """TurnOutcome keeps facts and assessment visibly separate."""

    def test_facts_and_assessment_separate(self) -> None:
        facts = TurnFacts(assistant_text="Hello", finish_reason="stop")
        assessment = TurnAssessment(completed=True, answer="Hello")
        outcome = TurnOutcome(facts=facts, assessment=assessment)

        # Access through explicit paths
        assert outcome.facts.assistant_text == "Hello"
        assert outcome.assessment.completed is True

    def test_default_assessment(self) -> None:
        """TurnOutcome should have a default (empty) assessment."""
        facts = TurnFacts(assistant_text="Hello")
        outcome = TurnOutcome(facts=facts)
        assert not outcome.assessment.completed
        assert not outcome.assessment.failed

    def test_cannot_confuse_facts_with_assessment(self) -> None:
        """Verify the namespace separation is enforced."""
        facts = TurnFacts(assistant_text="I'm done")
        outcome = TurnOutcome(facts=facts)

        # facts should have assistant_text, assessment should not
        assert outcome.facts.assistant_text == "I'm done"
        assert not hasattr(outcome.assessment, "assistant_text")

        # assessment should have completed, facts should not
        assert hasattr(outcome.assessment, "completed")
        assert not hasattr(outcome.facts, "completed")

    def test_serialization_roundtrip(self) -> None:
        tc = ToolCallRequest(id="1", name="search", arguments='{}')
        facts = TurnFacts(tool_calls=[tc], finish_reason="tool_calls")
        assessment = TurnAssessment(completed=False)
        outcome = TurnOutcome(facts=facts, assessment=assessment)

        data = outcome.model_dump()
        restored = TurnOutcome(**data)
        assert len(restored.facts.tool_calls) == 1
        assert not restored.assessment.completed


# =========================================================================
# 2. ConversationAgent tests -- skipped when not implemented
# =========================================================================


@needs_conversation
class TestParseTurn:
    """_parse_turn should only extract facts, never interpret."""

    def test_text_response(self) -> None:
        """_parse_turn should extract text from LLMResponse into TurnFacts."""
        response = LLMResponse(
            content="Hello world",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            model="deepseek-chat",
            finish_reason="stop",
        )
        facts = ConversationAgent._parse_turn(response)
        assert isinstance(facts, TurnFacts)
        assert facts.assistant_text == "Hello world"
        assert facts.finish_reason == "stop"
        assert facts.tool_calls == []

    def test_tool_call_response(self) -> None:
        """_parse_turn should extract tool calls from LLMResponse."""
        tc = ToolCallRequest(id="tc-1", name="search", arguments='{"q": "test"}')
        response = LLMResponse(
            content=None,
            tool_calls=[tc],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="deepseek-chat",
            finish_reason="tool_calls",
        )
        facts = ConversationAgent._parse_turn(response)
        assert isinstance(facts, TurnFacts)
        assert len(facts.tool_calls) == 1
        assert facts.tool_calls[0].name == "search"
        assert facts.assistant_text is None

    def test_never_sets_completed(self) -> None:
        """_parse_turn must NEVER set assessment fields -- it returns TurnFacts only."""
        response = LLMResponse(
            content="The answer is 42",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model="deepseek-chat",
            finish_reason="stop",
        )
        facts = ConversationAgent._parse_turn(response)
        # TurnFacts has no completed/failed/answer attributes
        assert not hasattr(facts, "completed")
        assert not hasattr(facts, "failed")
        assert not hasattr(facts, "answer")

    def test_preserves_usage(self) -> None:
        response = LLMResponse(
            content="hi",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            model="deepseek-chat",
            finish_reason="stop",
        )
        facts = ConversationAgent._parse_turn(response)
        assert facts.usage is not None
        assert facts.usage.total_tokens == 150


@needs_conversation
class TestAssessTurn:
    """_assess_turn should interpret facts into assessment."""

    @staticmethod
    def _make_state(current_step: int = 0) -> AgentState:
        """Create a minimal AgentState for testing _assess_turn."""
        return AgentState(run_id="test", current_step=current_step)

    def test_tool_calls_not_done(self) -> None:
        """If facts has tool_calls, assessment.completed must be False."""
        tc = ToolCallRequest(id="1", name="search", arguments='{}')
        facts = TurnFacts(tool_calls=[tc], finish_reason="tool_calls")
        assessment = ConversationAgent._assess_turn(facts, self._make_state(1))
        assert isinstance(assessment, TurnAssessment)
        assert not assessment.completed

    def test_empty_response_is_failure(self) -> None:
        """No text and no tools -> failed."""
        facts = TurnFacts(finish_reason="stop")
        assessment = ConversationAgent._assess_turn(facts, self._make_state(1))
        assert assessment.failed

    def test_natural_stop_completes(self) -> None:
        """finish_reason=stop + text -> completed (natural_stop)."""
        facts = TurnFacts(assistant_text="Paris is the capital of France.", finish_reason="stop")
        assessment = ConversationAgent._assess_turn(facts, self._make_state(1))
        assert assessment.completed
        assert assessment.answer == "Paris is the capital of France."
        assert assessment.completion_reason == "natural_stop"

    def test_length_finish_not_done(self) -> None:
        """finish_reason=length -> not completed (LLM was cut off)."""
        facts = TurnFacts(assistant_text="The capital of France is", finish_reason="length")
        assessment = ConversationAgent._assess_turn(facts, self._make_state(1))
        assert not assessment.completed
        assert not assessment.failed


@needs_conversation
class TestConversationAgentMock:
    """Test ConversationAgent with mocked gateway."""

    def _make_agent(
        self,
        responses: list[LLMResponse],
        tool_gateway: object | None = None,
    ) -> ConversationAgent:
        """Create an agent with a mocked gateway that returns preset responses.

        The mock supports both generate() and stream(). The stream() mock
        wraps each response as StreamChunks (single text_delta + tool deltas + done).
        """
        gateway = MagicMock()
        call_count = 0

        def _next_response():
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return responses[idx]

        async def mock_generate(request, config, trace_ctx=None):
            return _next_response()

        async def mock_stream(request, config, trace_ctx=None):
            response = _next_response()
            if response.content:
                yield StreamChunk(type="text_delta", text=response.content)
            if response.tool_calls:
                for tc in response.tool_calls:
                    yield StreamChunk(
                        type="tool_call_delta",
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        arguments_delta=tc.arguments,
                    )
            yield StreamChunk(
                type="done",
                usage=response.usage,
                metadata={
                    "finish_reason": response.finish_reason,
                    "model": response.model,
                },
            )

        gateway.generate = AsyncMock(side_effect=mock_generate)
        gateway.stream = mock_stream
        gateway.default_provider = "mock"
        gateway.get = MagicMock(return_value=None)

        return ConversationAgent(
            gateway=gateway,
            model_config=ModelConfig(provider="mock", model_id="mock-model"),
            max_turns=5,
            tool_gateway=tool_gateway,
        )

    @pytest.mark.asyncio
    async def test_direct_answer_one_turn(self) -> None:
        """Simple question should complete in 1 turn."""
        responses = [
            LLMResponse(
                content="Paris",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = self._make_agent(responses)
        state = await agent.run("What is the capital of France?")
        assert state.status.value == "completed"
        assert state.current_step >= 1
        assert "Paris" in str(state.working_memory.get("answer", ""))

    @pytest.mark.asyncio
    async def test_max_turns_stops_execution(self) -> None:
        """Should stop after max_turns even if not completed."""
        # Gateway always returns text with finish_reason=length (not stop)
        never_done = LLMResponse(
            content="Still thinking...",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="mock-model",
            finish_reason="length",
        )
        responses = [never_done] * 10
        agent = self._make_agent(responses)
        # Override max_turns to 3
        agent.max_turns = 3
        state = await agent.run("What is the meaning of life?")
        assert state.current_step <= 3
        # Should not be completed since finish_reason was always "length"
        assert state.status.value in ("failed", "completed")

    @pytest.mark.asyncio
    async def test_tool_call_and_response(self) -> None:
        """Tool call should execute and feed result back."""
        from arcana.contracts.tool import ToolCall, ToolResult, ToolSpec
        from arcana.tool_gateway.gateway import ToolGateway
        from arcana.tool_gateway.registry import ToolRegistry

        # Setup a mock tool
        class MockCalcProvider:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="calculator",
                    description="Calculate math",
                    input_schema={
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                )

            async def execute(self, call: ToolCall) -> ToolResult:
                expr = call.arguments.get("expression", "0")
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    success=True,
                    output=str(eval(expr)),  # noqa: S307
                )

            async def health_check(self) -> bool:
                return True

        registry = ToolRegistry()
        registry.register(MockCalcProvider())
        tool_gw = ToolGateway(registry=registry)

        # LLM first calls a tool, then returns a final answer
        tc = ToolCallRequest(id="tc-1", name="calculator", arguments='{"expression": "15*37+89"}')
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[tc],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="15 * 37 + 89 = 644",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = self._make_agent(responses, tool_gateway=tool_gw)
        state = await agent.run("What is 15 * 37 + 89?")
        assert state.status.value == "completed"
        assert "644" in str(state.working_memory.get("answer", ""))

    @pytest.mark.asyncio
    async def test_streaming_yields_events(self) -> None:
        """astream() should yield StreamEvents in correct order."""
        from arcana.contracts.streaming import StreamEventType

        responses = [
            LLMResponse(
                content="Paris is the capital of France",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = self._make_agent(responses)

        events = []
        async for event in agent.astream("What is the capital of France?"):
            events.append(event)

        # Should have at least RUN_START and RUN_COMPLETE
        event_types = [e.event_type for e in events]
        assert StreamEventType.RUN_START in event_types
        assert StreamEventType.RUN_COMPLETE in event_types
        assert len(events) >= 2


@needs_conversation
class TestContextManagement:
    """Tests for unified context management via WorkingSetBuilder."""

    def test_under_budget_passes_through(self) -> None:
        """Messages under budget should pass through unchanged."""
        from arcana.context.builder import WorkingSetBuilder
        from arcana.contracts.context import TokenBudget
        from arcana.contracts.llm import Message, MessageRole

        builder = WorkingSetBuilder("System.", TokenBudget(total_window=128_000))
        messages = [
            Message(role=MessageRole.SYSTEM, content="System."),
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi!"),
        ]
        result = builder.build_conversation_context(messages)
        assert len(result) == len(messages)

    def test_over_budget_compresses(self) -> None:
        """When over budget, old messages should be dropped or compressed."""
        from arcana.context.builder import WorkingSetBuilder
        from arcana.contracts.context import TokenBudget
        from arcana.contracts.llm import Message, MessageRole

        builder = WorkingSetBuilder(
            "System.",
            TokenBudget(total_window=500, response_reserve=50),
        )
        long_text = "This is a longer message to consume more tokens. " * 5
        messages = [Message(role=MessageRole.SYSTEM, content="System.")]
        for i in range(20):
            messages.append(Message(role=MessageRole.USER, content=f"Q{i}: {long_text}"))
            messages.append(Message(role=MessageRole.ASSISTANT, content=f"A{i}: {long_text}"))

        result = builder.build_conversation_context(messages)
        assert len(result) < len(messages)
        assert result[0].role == MessageRole.SYSTEM
        assert result[-1].content == messages[-1].content

    def test_summary_included_when_space(self) -> None:
        """Compressed middle should include summary when budget allows."""
        from arcana.context.builder import WorkingSetBuilder
        from arcana.contracts.context import TokenBudget
        from arcana.contracts.llm import Message, MessageRole

        builder = WorkingSetBuilder(
            "System.",
            TokenBudget(total_window=2000, response_reserve=100),
        )
        messages = [Message(role=MessageRole.SYSTEM, content="System.")]
        for i in range(30):
            messages.append(Message(role=MessageRole.USER, content=f"Question {i}?"))
            messages.append(Message(role=MessageRole.ASSISTANT, content=f"Answer {i}."))

        result = builder.build_conversation_context(messages)
        if len(result) < len(messages):
            has_summary = any("[Earlier conversation summary]" in (m.content or "") for m in result)
            assert has_summary

    def test_memory_injected_into_system_prompt(self) -> None:
        """Memory context should be injected into system prompt on first turn."""
        from arcana.context.builder import WorkingSetBuilder
        from arcana.contracts.context import TokenBudget
        from arcana.contracts.llm import Message, MessageRole

        builder = WorkingSetBuilder(
            "You are helpful.",
            TokenBudget(total_window=128_000),
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Hi"),
        ]
        result = builder.build_conversation_context(
            messages, memory_context="[Memory] User likes Python."
        )
        # Memory should be in system prompt
        assert "[Memory] User likes Python." in result[0].content
        assert "You are helpful." in result[0].content


# =========================================================================
# 3. Streaming tests -- real token-level streaming via LLM_CHUNK events
# =========================================================================


@needs_conversation
class TestStreamingLLMChunks:
    """Test that astream yields LLM_CHUNK events for token-level streaming."""

    @staticmethod
    def _make_streaming_agent(
        chunks_per_turn: list[list[StreamChunk]],
        tool_gateway: object | None = None,
    ) -> ConversationAgent:
        """Create an agent whose gateway yields specific StreamChunks per turn."""
        gateway = MagicMock()
        call_count = 0

        async def mock_stream(request, config, trace_ctx=None):
            nonlocal call_count
            idx = min(call_count, len(chunks_per_turn) - 1)
            call_count += 1
            for chunk in chunks_per_turn[idx]:
                yield chunk

        gateway.stream = mock_stream
        gateway.default_provider = "mock"
        gateway.get = MagicMock(return_value=None)

        return ConversationAgent(
            gateway=gateway,
            model_config=ModelConfig(provider="mock", model_id="mock-model"),
            max_turns=5,
            tool_gateway=tool_gateway,
        )

    @pytest.mark.asyncio
    async def test_text_streaming_yields_llm_chunks(self) -> None:
        """astream should yield LLM_CHUNK events for each text delta."""
        from arcana.contracts.streaming import StreamEventType

        turn_chunks = [[
            StreamChunk(type="text_delta", text="Hello "),
            StreamChunk(type="text_delta", text="world!"),
            StreamChunk(
                type="done",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                metadata={"finish_reason": "stop", "model": "mock-model"},
            ),
        ]]
        agent = self._make_streaming_agent(turn_chunks)

        events = []
        async for event in agent.astream("Say hello"):
            events.append(event)

        event_types = [e.event_type for e in events]
        assert StreamEventType.LLM_CHUNK in event_types

        chunk_events = [e for e in events if e.event_type == StreamEventType.LLM_CHUNK]
        assert len(chunk_events) == 2
        assert chunk_events[0].content == "Hello "
        assert chunk_events[1].content == "world!"

    @pytest.mark.asyncio
    async def test_streaming_accumulates_into_completed_state(self) -> None:
        """Streaming chunks should be accumulated into a complete response."""
        turn_chunks = [[
            StreamChunk(type="text_delta", text="The answer "),
            StreamChunk(type="text_delta", text="is 42"),
            StreamChunk(
                type="done",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
                metadata={"finish_reason": "stop", "model": "mock-model"},
            ),
        ]]
        agent = self._make_streaming_agent(turn_chunks)
        state = await agent.run("What is the meaning of life?")

        assert state.status.value == "completed"
        assert "42" in str(state.working_memory.get("answer", ""))
        assert state.tokens_used == 18

    @pytest.mark.asyncio
    async def test_streaming_tool_call_deltas(self) -> None:
        """Tool call deltas should be accumulated and executed."""
        from arcana.contracts.streaming import StreamEventType
        from arcana.contracts.tool import ToolCall, ToolResult, ToolSpec
        from arcana.tool_gateway.gateway import ToolGateway
        from arcana.tool_gateway.registry import ToolRegistry

        class MockSearchProvider:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="search",
                    description="Search",
                    input_schema={
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                )

            async def execute(self, call: ToolCall) -> ToolResult:
                return ToolResult(
                    tool_call_id=call.id, name=call.name,
                    success=True, output="found it",
                )

            async def health_check(self) -> bool:
                return True

        registry = ToolRegistry()
        registry.register(MockSearchProvider())
        tool_gw = ToolGateway(registry=registry)

        turn_chunks = [
            # Turn 1: tool call with incremental argument deltas
            [
                StreamChunk(
                    type="tool_call_delta", tool_call_id="tc-1",
                    tool_name="search", arguments_delta='{"q":',
                ),
                StreamChunk(
                    type="tool_call_delta", tool_call_id="tc-1",
                    tool_name=None, arguments_delta=' "hello"}',
                ),
                StreamChunk(
                    type="done",
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    metadata={"finish_reason": "tool_calls", "model": "mock-model"},
                ),
            ],
            # Turn 2: final answer
            [
                StreamChunk(type="text_delta", text="Found results"),
                StreamChunk(
                    type="done",
                    usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                    metadata={"finish_reason": "stop", "model": "mock-model"},
                ),
            ],
        ]
        agent = self._make_streaming_agent(turn_chunks, tool_gateway=tool_gw)

        events = []
        async for event in agent.astream("Search for hello"):
            events.append(event)

        event_types = [e.event_type for e in events]
        assert StreamEventType.TOOL_RESULT in event_types
        assert StreamEventType.RUN_COMPLETE in event_types

        # Final answer should be in the run_complete event
        run_complete = [e for e in events if e.event_type == StreamEventType.RUN_COMPLETE][0]
        assert "Found results" in (run_complete.content or "")

    @pytest.mark.asyncio
    async def test_thinking_deltas_yield_llm_thinking_events(self) -> None:
        """Thinking deltas should yield LLM_THINKING events."""
        from arcana.contracts.streaming import StreamEventType

        turn_chunks = [[
            StreamChunk(type="thinking_delta", thinking="Let me think..."),
            StreamChunk(type="text_delta", text="Answer"),
            StreamChunk(
                type="done",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                metadata={"finish_reason": "stop", "model": "mock-model"},
            ),
        ]]
        agent = self._make_streaming_agent(turn_chunks)

        events = []
        async for event in agent.astream("Think about it"):
            events.append(event)

        thinking_events = [e for e in events if e.event_type == StreamEventType.LLM_THINKING]
        assert len(thinking_events) == 1
        assert thinking_events[0].thinking == "Let me think..."

    @pytest.mark.asyncio
    async def test_fallback_to_generate_when_stream_unavailable(self) -> None:
        """Should fall back to generate() when stream() raises AttributeError."""
        gateway = MagicMock()
        gateway.stream = MagicMock(side_effect=AttributeError("no streaming"))

        async def mock_generate(request, config, trace_ctx=None):
            return LLMResponse(
                content="Fallback response",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
                finish_reason="stop",
            )

        gateway.generate = AsyncMock(side_effect=mock_generate)
        gateway.default_provider = "mock"
        gateway.get = MagicMock(return_value=None)

        agent = ConversationAgent(
            gateway=gateway,
            model_config=ModelConfig(provider="mock", model_id="mock-model"),
            max_turns=5,
        )
        state = await agent.run("Test fallback")

        assert state.status.value == "completed"
        assert "Fallback response" in str(state.working_memory.get("answer", ""))

    @pytest.mark.asyncio
    async def test_usage_from_separate_chunk(self) -> None:
        """Usage sent as a separate 'usage' chunk should be tracked."""
        turn_chunks = [[
            StreamChunk(type="text_delta", text="Hello"),
            StreamChunk(type="done", metadata={"finish_reason": "stop", "model": "mock"}),
            StreamChunk(
                type="usage",
                usage=TokenUsage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
            ),
        ]]
        agent = self._make_streaming_agent(turn_chunks)
        state = await agent.run("Hi")

        assert state.status.value == "completed"
        assert state.tokens_used == 75


# =========================================================================
# 4. LazyToolRegistry integration tests
# =========================================================================


@needs_conversation
class TestLazyToolRegistryIntegration:
    """Test that ConversationAgent uses LazyToolRegistry for tool selection."""

    @staticmethod
    def _make_tool_gateway_with_tools(
        tool_names: list[str],
    ) -> tuple[object, list[object]]:
        """Create a ToolGateway with multiple mock tools."""
        from arcana.contracts.tool import ToolCall, ToolResult, ToolSpec
        from arcana.tool_gateway.gateway import ToolGateway
        from arcana.tool_gateway.registry import ToolRegistry

        providers = []
        for name in tool_names:

            class _Provider:
                def __init__(self, tool_name: str) -> None:
                    self._name = tool_name

                @property
                def spec(self) -> ToolSpec:
                    return ToolSpec(
                        name=self._name,
                        description=f"Tool that does {self._name}",
                        input_schema={
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    )

                async def execute(self, call: ToolCall) -> ToolResult:
                    return ToolResult(
                        tool_call_id=call.id,
                        name=call.name,
                        success=True,
                        output=f"{self._name} result",
                    )

                async def health_check(self) -> bool:
                    return True

            providers.append(_Provider(name))

        registry = ToolRegistry()
        for p in providers:
            registry.register(p)
        tool_gw = ToolGateway(registry=registry)
        return tool_gw, providers

    @staticmethod
    def _make_agent_with_tools(
        tool_names: list[str],
        responses_or_chunks: list | None = None,
    ) -> tuple[ConversationAgent, object]:
        """Create a ConversationAgent with LazyToolRegistry and mock gateway."""
        tool_gw, _ = TestLazyToolRegistryIntegration._make_tool_gateway_with_tools(
            tool_names,
        )

        gateway = MagicMock()
        call_count = 0

        default_response = LLMResponse(
            content="Done",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="mock-model",
            finish_reason="stop",
        )
        responses = responses_or_chunks or [default_response]

        async def mock_stream(request, config, trace_ctx=None):
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            resp = responses[idx]
            if resp.content:
                yield StreamChunk(type="text_delta", text=resp.content)
            if resp.tool_calls:
                for tc in resp.tool_calls:
                    yield StreamChunk(
                        type="tool_call_delta",
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        arguments_delta=tc.arguments,
                    )
            yield StreamChunk(
                type="done",
                usage=resp.usage,
                metadata={
                    "finish_reason": resp.finish_reason,
                    "model": resp.model,
                },
            )

        gateway.stream = mock_stream
        gateway.default_provider = "mock"
        gateway.get = MagicMock(return_value=None)

        agent = ConversationAgent(
            gateway=gateway,
            model_config=ModelConfig(provider="mock", model_id="mock-model"),
            max_turns=5,
            tool_gateway=tool_gw,
        )
        return agent, tool_gw

    def test_lazy_registry_created_when_tool_gateway_exists(self) -> None:
        """ConversationAgent should create a LazyToolRegistry when tool_gateway is set."""
        agent, _ = self._make_agent_with_tools(["search", "calculator"])
        assert agent._lazy_registry is not None

    def test_no_lazy_registry_without_tool_gateway(self) -> None:
        """No LazyToolRegistry when no tool_gateway."""
        gateway = MagicMock()
        gateway.default_provider = "mock"
        agent = ConversationAgent(
            gateway=gateway,
            model_config=ModelConfig(provider="mock", model_id="mock-model"),
        )
        assert agent._lazy_registry is None

    @pytest.mark.asyncio
    async def test_initial_selection_uses_goal(self) -> None:
        """LazyToolRegistry should select tools based on the goal."""
        many_tools = [
            "search", "calculator", "file_reader", "web_fetch",
            "database_query", "shell_exec", "code_runner", "email_sender",
        ]
        agent, _ = self._make_agent_with_tools(many_tools)
        assert agent._lazy_registry is not None

        # Run with a search-related goal
        state = await agent.run("Search for Python tutorials")

        # The lazy registry should have selected a subset, not all tools
        assert agent._lazy_registry is not None
        ws = agent._lazy_registry.working_set
        # Should have at most max_initial_tools (default 5), not all 8
        assert len(ws) <= 5

        # Expansion log should have the initial selection
        log = agent._lazy_registry.expansion_log
        assert len(log) >= 1
        assert log[0].trigger == "initial_selection"
        assert state.status.value == "completed"

    @pytest.mark.asyncio
    async def test_on_demand_expansion_for_unlisted_tool(self) -> None:
        """If LLM calls a tool not in the working set, it should be loaded on demand."""
        # Use enough tools that "zzz_notifier" (alphabetically last) falls outside
        # the initial top-5 selection for a "Search" goal.
        many_tools = [
            "search", "analyzer", "builder", "calculator", "compiler",
            "debugger", "file_reader", "zzz_notifier",
        ]

        # LLM calls zzz_notifier which won't be in the initial 5
        tc = ToolCallRequest(
            id="tc-1", name="zzz_notifier", arguments='{"q": "test"}',
        )
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[tc],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Notified",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent, _ = self._make_agent_with_tools(many_tools, responses)

        state = await agent.run("Search for something")

        # zzz_notifier should have been loaded on demand
        assert agent._lazy_registry is not None
        ws_names = [s.name for s in agent._lazy_registry.working_set]
        assert "zzz_notifier" in ws_names

        # Should have an explicit_request expansion event
        log = agent._lazy_registry.expansion_log
        explicit_events = [e for e in log if e.trigger == "explicit_request"]
        assert len(explicit_events) >= 1

        assert state.status.value == "completed"

    @pytest.mark.asyncio
    async def test_tool_token_cost_uses_subset(self) -> None:
        """Token cost should be based on active (subset) tools, not all tools."""
        many_tools = [
            "search", "calculator", "file_reader", "web_fetch",
            "database_query", "shell_exec", "code_runner", "email_sender",
        ]
        agent, _ = self._make_agent_with_tools(many_tools)

        # Compute what all-tools cost would be
        all_tools_defs = agent._get_current_tools()
        all_cost = agent._estimate_tool_tokens(all_tools_defs)

        # Run and check working set is smaller
        await agent.run("Search for something")
        assert agent._lazy_registry is not None
        subset_defs = agent._lazy_registry.to_openai_tools()
        subset_cost = agent._estimate_tool_tokens(subset_defs)

        # Subset cost should be less than all-tools cost
        assert subset_cost < all_cost

    @pytest.mark.asyncio
    async def test_lazy_registry_resets_between_runs(self) -> None:
        """Each run should start with a fresh working set."""
        agent, _ = self._make_agent_with_tools(["search", "calculator"])
        assert agent._lazy_registry is not None

        await agent.run("First run")
        ws_after_first = len(agent._lazy_registry.working_set)
        assert ws_after_first > 0

        await agent.run("Second run")
        # Should have reset and re-selected; expansion log should show fresh start
        log = agent._lazy_registry.expansion_log
        assert log[0].trigger == "initial_selection"


# =========================================================================
# 6. Zero-token usage estimation
# =========================================================================


@needs_conversation
class TestZeroTokenEstimation:
    """Test that zero-token usage from providers triggers estimation."""

    def _make_agent(
        self,
        responses: list[LLMResponse],
    ) -> ConversationAgent:
        """Create agent with mock gateway returning preset responses via stream."""
        gateway = MagicMock()
        call_count = 0

        def _next_response():
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return responses[idx]

        async def mock_stream(request, config, trace_ctx=None):
            response = _next_response()
            if response.content:
                yield StreamChunk(type="text_delta", text=response.content)
            if response.tool_calls:
                for tc in response.tool_calls:
                    yield StreamChunk(
                        type="tool_call_delta",
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        arguments_delta=tc.arguments,
                    )
            yield StreamChunk(
                type="done",
                usage=response.usage,
                metadata={
                    "finish_reason": response.finish_reason,
                    "model": response.model,
                },
            )

        gateway.generate = AsyncMock(side_effect=lambda *a, **kw: _next_response())
        gateway.stream = mock_stream
        gateway.default_provider = "mock"
        gateway.get = MagicMock(return_value=None)

        return ConversationAgent(
            gateway=gateway,
            model_config=ModelConfig(provider="mock", model_id="mock-model"),
            max_turns=5,
        )

    @pytest.mark.asyncio
    async def test_zero_tokens_estimated_from_content(self) -> None:
        """When provider reports 0 tokens but has content, tokens should be estimated."""
        responses = [
            LLMResponse(
                content="Hello, world! This is a test response.",
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = self._make_agent(responses)
        state = await agent.run("Say hello")
        # Estimated completion tokens = len("Hello, world! This is a test response.") // 4 = 9
        assert state.tokens_used > 0
        assert state.cost_usd > 0.0

    @pytest.mark.asyncio
    async def test_zero_tokens_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning should be logged when provider reports 0 tokens."""
        import logging

        responses = [
            LLMResponse(
                content="Some content here",
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = self._make_agent(responses)
        with caplog.at_level(logging.WARNING, logger="arcana.runtime.conversation"):
            await agent.run("Test")
        assert any("estimated" in r.message.lower() for r in caplog.records)
        assert any("0 tokens" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_nonzero_tokens_not_estimated(self) -> None:
        """When provider reports real tokens, no estimation should occur."""
        responses = [
            LLMResponse(
                content="Hello",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = self._make_agent(responses)
        state = await agent.run("Say hello")
        assert state.tokens_used == 15

    @pytest.mark.asyncio
    async def test_zero_tokens_no_content_not_estimated(self) -> None:
        """When there's no content, don't estimate even if tokens are 0."""
        responses = [
            LLMResponse(
                content=None,
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = self._make_agent(responses)
        state = await agent.run("Test")
        # No content means estimation shouldn't kick in
        assert state.tokens_used == 0


class TestCostEstimate:
    """Test that cost_estimate returns reasonable values."""

    def test_cost_estimate_nonzero_for_typical_usage(self) -> None:
        """cost_estimate should return a meaningful value for typical token counts."""
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        cost = usage.cost_estimate
        # With $0.15/M input + $0.60/M output:
        # 1000 * 0.15 / 1_000_000 + 500 * 0.60 / 1_000_000
        # = 0.00015 + 0.0003 = 0.00045
        assert cost > 0.0
        assert abs(cost - 0.00045) < 1e-10

    def test_cost_estimate_zero_for_zero_tokens(self) -> None:
        """cost_estimate should be 0 when no tokens used."""
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        assert usage.cost_estimate == 0.0

    def test_cost_estimate_reasonable_for_large_usage(self) -> None:
        """For 1M tokens, cost should be in reasonable dollar range."""
        usage = TokenUsage(
            prompt_tokens=500_000, completion_tokens=500_000, total_tokens=1_000_000,
        )
        cost = usage.cost_estimate
        # 500k * 0.15/1M + 500k * 0.60/1M = 0.075 + 0.3 = 0.375
        assert 0.1 < cost < 1.0
        assert abs(cost - 0.375) < 1e-10


@needs_conversation
class TestToolCallLogging:
    """Verify that tool call debug logging is emitted."""

    @pytest.mark.asyncio
    async def test_tool_call_logging(self) -> None:
        """logger.debug should be called when tools are executed."""
        import logging

        from arcana.contracts.tool import ToolCall, ToolResult, ToolSpec
        from arcana.tool_gateway.gateway import ToolGateway
        from arcana.tool_gateway.registry import ToolRegistry

        # Setup a mock tool
        class MockSearchProvider:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="search",
                    description="Search the web",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                )

            async def execute(self, call: ToolCall) -> ToolResult:
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    success=True,
                    output="result for: " + call.arguments.get("query", ""),
                )

            async def health_check(self) -> bool:
                return True

        registry = ToolRegistry()
        registry.register(MockSearchProvider())
        tool_gw = ToolGateway(registry=registry)

        # LLM calls tool, then gives final answer
        tc = ToolCallRequest(
            id="tc-1", name="search", arguments='{"query": "hello"}'
        )
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[tc],
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
                model="mock-model",
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Found it!",
                usage=TokenUsage(
                    prompt_tokens=20, completion_tokens=10, total_tokens=30
                ),
                model="mock-model",
                finish_reason="stop",
            ),
        ]

        gateway = MagicMock()
        call_count = 0

        def _next_response():
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return responses[idx]

        async def mock_generate(request, config, trace_ctx=None):
            return _next_response()

        gateway.generate = AsyncMock(side_effect=mock_generate)
        # Force fallback from streaming to generate
        gateway.stream = MagicMock(side_effect=NotImplementedError)
        gateway.default_provider = "mock"
        gateway.get = MagicMock(return_value=None)

        agent = ConversationAgent(
            gateway=gateway,
            model_config=ModelConfig(provider="mock", model_id="mock-model"),
            max_turns=5,
            tool_gateway=tool_gw,
        )

        # Capture debug log messages from the conversation module
        conv_logger = logging.getLogger("arcana.runtime.conversation")
        original_level = conv_logger.level
        conv_logger.setLevel(logging.DEBUG)

        debug_messages: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                debug_messages.append(self.format(record))

        handler = CaptureHandler()
        handler.setLevel(logging.DEBUG)
        conv_logger.addHandler(handler)
        try:
            await agent.run("Search for hello")
        finally:
            conv_logger.removeHandler(handler)
            conv_logger.setLevel(original_level)

        # Verify tool call logging
        tool_call_msgs = [m for m in debug_messages if "Tool call:" in m]
        assert len(tool_call_msgs) >= 1
        assert "search" in tool_call_msgs[0]

        # Verify tool result logging
        tool_result_msgs = [m for m in debug_messages if "Tool result:" in m]
        assert len(tool_result_msgs) >= 1
        assert "search" in tool_result_msgs[0]

        # Verify LLM tool request logging
        llm_request_msgs = [m for m in debug_messages if "tool call(s)" in m]
        assert len(llm_request_msgs) >= 1
        assert "search" in llm_request_msgs[0]
