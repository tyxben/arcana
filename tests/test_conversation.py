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
            completion_reason="explicit_marker",
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
            completion_reason="done_marker",
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
        facts = TurnFacts(assistant_text="Hello [DONE]", finish_reason="stop")
        assessment = TurnAssessment(completed=True, answer="Hello [DONE]")
        outcome = TurnOutcome(facts=facts, assessment=assessment)

        # Access through explicit paths
        assert outcome.facts.assistant_text == "Hello [DONE]"
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
            content="The answer is 42 [DONE]",
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

    def test_done_marker_completes(self) -> None:
        """Text containing [DONE] -> completed."""
        facts = TurnFacts(assistant_text="Paris is the capital [DONE]", finish_reason="stop")
        assessment = ConversationAgent._assess_turn(facts, self._make_state(1))
        assert assessment.completed
        assert assessment.answer is not None

    def test_stop_reason_completes(self) -> None:
        """Text with stop finish_reason on non-first turn -> completed (heuristic)."""
        # Heuristic requires: finish_reason=stop, turn > 0, text > 100 chars
        long_answer = (
            "The capital of France is Paris. It has been the capital since "
            "the 10th century and is home to over 2 million residents in "
            "the city proper, making it one of the most important cities."
        )
        facts = TurnFacts(assistant_text=long_answer, finish_reason="stop")
        assessment = ConversationAgent._assess_turn(facts, self._make_state(2))
        assert assessment.completed
        assert assessment.completion_reason == "heuristic_natural_stop"


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
                content="Paris [DONE]",
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
        # Gateway always returns text without [DONE] and without stop
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
        # Should not be completed since we never got [DONE]
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
                content="15 * 37 + 89 = 644 [DONE]",
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
                content="Paris is the capital of France [DONE]",
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
            StreamChunk(type="text_delta", text="world! "),
            StreamChunk(type="text_delta", text="[DONE]"),
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
        assert len(chunk_events) == 3
        assert chunk_events[0].content == "Hello "
        assert chunk_events[1].content == "world! "
        assert chunk_events[2].content == "[DONE]"

    @pytest.mark.asyncio
    async def test_streaming_accumulates_into_completed_state(self) -> None:
        """Streaming chunks should be accumulated into a complete response."""
        turn_chunks = [[
            StreamChunk(type="text_delta", text="The answer "),
            StreamChunk(type="text_delta", text="is 42 "),
            StreamChunk(type="text_delta", text="[DONE]"),
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
                StreamChunk(type="text_delta", text="Found results [DONE]"),
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
            StreamChunk(type="text_delta", text="Answer [DONE]"),
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
                content="Fallback response [DONE]",
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
            StreamChunk(type="text_delta", text="Hello [DONE]"),
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
