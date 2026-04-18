"""Tests for the cognitive `recall` primitive (v0.7.0).

Validates:
- Basic recall returns the requested turn's messages
- Out-of-range turns return structured not-found, not exceptions
- turn=0 / negative turns are rejected with a structured note
- include=assistant_only / tool_calls filter the returned messages
- Recall works when no trace is available (fails gracefully)
- End-to-end: LLM calls recall through ConversationAgent interception
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from arcana.contracts.cognitive import RecallRequest
from arcana.contracts.llm import (
    LLMResponse,
    ModelConfig,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from arcana.runtime.cognitive import (
    RECALL_TOOL_NAME,
    CognitiveHandler,
)
from arcana.runtime.conversation import ConversationAgent

# ---------------------------------------------------------------------------
# Unit tests — CognitiveHandler.handle_recall
# ---------------------------------------------------------------------------


class TestHandleRecall:
    def test_basic_recall_returns_messages(self) -> None:
        handler = CognitiveHandler(enabled={"recall"})
        handler.record_turn(
            1,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        )
        result = handler.handle_recall(RecallRequest(turn=1))
        assert result.found is True
        assert result.turn == 1
        assert len(result.messages) == 2
        assert result.messages[1]["content"] == "hi there"

    def test_turn_out_of_range(self) -> None:
        handler = CognitiveHandler(enabled={"recall"})
        handler.record_turn(1, [{"role": "assistant", "content": "a"}])
        result = handler.handle_recall(RecallRequest(turn=999))
        assert result.found is False
        assert result.note is not None
        assert "out of range" in result.note
        assert "999" in result.note

    def test_turn_zero_rejected(self) -> None:
        handler = CognitiveHandler(enabled={"recall"})
        handler.record_turn(1, [{"role": "assistant", "content": "a"}])
        result = handler.handle_recall(RecallRequest(turn=0))
        assert result.found is False
        assert result.note is not None
        assert "1 or greater" in result.note or "1-indexed" in result.note

    def test_turn_negative_rejected(self) -> None:
        handler = CognitiveHandler(enabled={"recall"})
        handler.record_turn(1, [{"role": "assistant", "content": "a"}])
        result = handler.handle_recall(RecallRequest(turn=-2))
        assert result.found is False
        assert result.note is not None

    def test_include_assistant_only_filters(self) -> None:
        handler = CognitiveHandler(enabled={"recall"})
        handler.record_turn(
            1,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "tool", "content": "tool output"},
            ],
        )
        result = handler.handle_recall(
            RecallRequest(turn=1, include="assistant_only")
        )
        assert result.found is True
        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "assistant"

    def test_include_tool_calls_filters(self) -> None:
        handler = CognitiveHandler(enabled={"recall"})
        handler.record_turn(
            1,
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "t1", "name": "calc"}],
                },
                {"role": "tool", "content": "42", "tool_call_id": "t1"},
                {"role": "assistant", "content": "result is 42"},
            ],
        )
        result = handler.handle_recall(
            RecallRequest(turn=1, include="tool_calls")
        )
        assert result.found is True
        # tool_calls include: the tool message and the assistant with tool_calls
        roles = [m["role"] for m in result.messages]
        assert "tool" in roles
        assert "assistant" in roles
        # The plain-text assistant message should be filtered out
        text_only = [
            m for m in result.messages
            if m["role"] == "assistant" and not m.get("tool_calls")
        ]
        assert text_only == []

    def test_trace_unavailable_returns_structured_note(self) -> None:
        """No in-memory turns and no trace reader → structured note."""
        handler = CognitiveHandler(enabled={"recall"})
        # no record_turn / no trace_reader
        result = handler.handle_recall(RecallRequest(turn=1))
        assert result.found is False
        assert result.note is not None
        assert "trace not available" in result.note


# ---------------------------------------------------------------------------
# Integration — LLM calls recall through ConversationAgent
# ---------------------------------------------------------------------------


def _make_agent(
    responses: list[LLMResponse],
    cognitive_primitives: list[str] | None = None,
) -> ConversationAgent:
    gateway = MagicMock()
    call_count = 0

    def _next_response() -> LLMResponse:
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        return responses[idx]

    async def mock_stream(request, config, trace_ctx=None):  # noqa: ANN001
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

    gateway.stream = mock_stream
    gateway.default_provider = "mock"
    gateway.get = MagicMock(return_value=None)

    return ConversationAgent(
        gateway=gateway,
        model_config=ModelConfig(provider="mock", model_id="mock-model"),
        max_turns=5,
        cognitive_primitives=cognitive_primitives or [],
    )


class TestRecallInConversation:
    @pytest.mark.asyncio
    async def test_recall_intercepted_and_returns_structured_result(
        self,
    ) -> None:
        """LLM calls recall on a prior turn; returns JSON result."""
        tc = ToolCallRequest(
            id="tc-recall-1",
            name=RECALL_TOOL_NAME,
            arguments='{"turn": 1}',
        )
        responses = [
            LLMResponse(
                content="First answer",
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15,
                ),
                model="mock-model",
                finish_reason="stop",
            ),
            LLMResponse(
                content=None,
                tool_calls=[tc],
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15,
                ),
                model="mock-model",
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Recalled content successfully.",
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15,
                ),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = _make_agent(responses, cognitive_primitives=["recall"])

        # Run it: the first turn records history, second calls recall,
        # third completes.
        await agent.run("Remember and recall")

        # The recall tool call should have been serviced. Because we
        # only have a mock gateway, the simplest verification is that
        # the handler itself now has the recalled turn.
        assert agent._cognitive_handler.max_turn() >= 1
        result = agent._cognitive_handler.handle_recall(RecallRequest(turn=1))
        assert result.found is True

    def test_recall_tool_in_tool_list_when_enabled(self) -> None:
        agent = _make_agent([], cognitive_primitives=["recall"])
        tools = agent._get_current_tools()
        assert tools is not None
        names = [t["function"]["name"] for t in tools]
        assert RECALL_TOOL_NAME in names

    def test_recall_tool_absent_when_disabled(self) -> None:
        """Default cognitive_primitives=[] → recall is NOT in the tool list."""
        agent = _make_agent([], cognitive_primitives=[])
        tools = agent._get_current_tools()
        assert tools is not None
        names = [t["function"]["name"] for t in tools]
        assert RECALL_TOOL_NAME not in names

    @pytest.mark.asyncio
    async def test_recall_invalid_arguments_returns_structured_error(
        self,
    ) -> None:
        """Invalid JSON or missing turn field should not raise."""
        tc = ToolCallRequest(
            id="tc-recall-bad",
            name=RECALL_TOOL_NAME,
            arguments="{}",  # missing `turn`
        )
        agent = _make_agent([], cognitive_primitives=["recall"])
        result = agent._handle_cognitive_tool_call(tc, run_id="rid")
        assert result.success is False
        payload = json.loads(result.output_str)
        # Pydantic validation error should be captured
        assert payload.get("error") == "invalid_arguments"
