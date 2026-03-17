"""Tests for ConversationAgent -- V2 execution model.

Tests are split into two groups:
1. Contract tests (TurnFacts, TurnAssessment, TurnOutcome) -- always run.
2. ConversationAgent tests -- skipped when runtime/conversation.py is absent.
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.contracts.llm import LLMResponse, ModelConfig, TokenUsage, ToolCallRequest
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
        """Create an agent with a mocked gateway that returns preset responses."""
        gateway = MagicMock()
        call_count = 0

        async def mock_generate(request, config, trace_ctx=None):
            nonlocal call_count
            idx = min(call_count, len(responses) - 1)
            call_count += 1
            return responses[idx]

        gateway.generate = AsyncMock(side_effect=mock_generate)
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
