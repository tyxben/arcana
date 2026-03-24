"""Tests for the ask_user built-in tool.

Validates that:
- The LLM can call ask_user and receive answers from sync/async handlers
- No input_handler returns a graceful fallback
- ask_user tool schema is always present in the tool list
- ask_user bypasses ToolGateway (handled directly by ConversationAgent)
- INPUT_NEEDED events are yielded when ask_user is called
- ask_user works alongside regular tools
- Multiple ask_user calls work in one run
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arcana.contracts.llm import (
    LLMResponse,
    ModelConfig,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from arcana.contracts.streaming import StreamEventType
from arcana.contracts.tool import ToolCall, ToolResult, ToolSpec
from arcana.runtime.ask_user import ASK_USER_SPEC, AskUserHandler
from arcana.runtime.conversation import ConversationAgent
from arcana.tool_gateway.gateway import ToolGateway
from arcana.tool_gateway.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    responses: list[LLMResponse],
    tool_gateway: object | None = None,
    input_handler: object | None = None,
) -> ConversationAgent:
    """Create an agent with a mocked streaming gateway that returns preset responses."""
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
        tool_gateway=tool_gateway,
        input_handler=input_handler,
    )


def _make_tool_gateway() -> ToolGateway:
    """Create a ToolGateway with a single mock calculator tool."""

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
    return ToolGateway(registry=registry)


# ---------------------------------------------------------------------------
# AskUserHandler unit tests
# ---------------------------------------------------------------------------


class TestAskUserHandler:
    """Direct tests for the AskUserHandler class."""

    @pytest.mark.asyncio
    async def test_sync_handler_provides_answer(self) -> None:
        """Sync input_handler should return its answer."""
        handler = AskUserHandler(input_handler=lambda q: f"Answer to: {q}")
        result = await handler.handle("What color?")
        assert result == "Answer to: What color?"

    @pytest.mark.asyncio
    async def test_async_handler_provides_answer(self) -> None:
        """Async input_handler should be awaited."""

        async def async_handler(q: str) -> str:
            return f"Async answer to: {q}"

        handler = AskUserHandler(input_handler=async_handler)
        result = await handler.handle("What color?")
        assert result == "Async answer to: What color?"

    @pytest.mark.asyncio
    async def test_no_handler_returns_fallback(self) -> None:
        """No input_handler should return the fallback message."""
        handler = AskUserHandler(input_handler=None)
        result = await handler.handle("What color?")
        assert "best judgment" in result.lower()
        assert "no user input" in result.lower()

    @pytest.mark.asyncio
    async def test_handler_result_converted_to_string(self) -> None:
        """Non-string sync handler result should be converted to string."""
        handler = AskUserHandler(input_handler=lambda q: 42)
        result = await handler.handle("How many?")
        assert result == "42"


# ---------------------------------------------------------------------------
# ASK_USER_SPEC contract tests
# ---------------------------------------------------------------------------


class TestAskUserSpec:
    """Validate the ask_user ToolSpec."""

    def test_name(self) -> None:
        assert ASK_USER_SPEC.name == "ask_user"

    def test_has_question_parameter(self) -> None:
        props = ASK_USER_SPEC.input_schema["properties"]
        assert "question" in props
        assert props["question"]["type"] == "string"

    def test_question_is_required(self) -> None:
        assert "question" in ASK_USER_SPEC.input_schema["required"]

    def test_has_affordance_fields(self) -> None:
        assert ASK_USER_SPEC.when_to_use is not None
        assert ASK_USER_SPEC.what_to_expect is not None
        assert ASK_USER_SPEC.failure_meaning is not None

    def test_side_effect_is_read(self) -> None:
        from arcana.contracts.tool import SideEffect

        assert ASK_USER_SPEC.side_effect == SideEffect.READ


# ---------------------------------------------------------------------------
# ConversationAgent integration tests
# ---------------------------------------------------------------------------


class TestAskUserInConversation:
    """Integration tests: ask_user through ConversationAgent."""

    @pytest.mark.asyncio
    async def test_sync_handler_provides_answer_as_tool_result(self) -> None:
        """LLM calls ask_user, sync handler provides answer, answer becomes ToolResult."""
        tc = ToolCallRequest(
            id="tc-ask-1",
            name="ask_user",
            arguments='{"question": "What is your name?"}',
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
                content="Hello, Alice!",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = _make_agent(
            responses,
            input_handler=lambda q: "Alice",
        )
        state = await agent.run("Greet the user by name")
        assert state.status.value == "completed"
        assert "Alice" in str(state.working_memory.get("answer", ""))

    @pytest.mark.asyncio
    async def test_async_handler_provides_answer(self) -> None:
        """LLM calls ask_user, async handler provides answer."""

        async def async_handler(q: str) -> str:
            return "Blue"

        tc = ToolCallRequest(
            id="tc-ask-2",
            name="ask_user",
            arguments='{"question": "Favorite color?"}',
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
                content="Your favorite color is Blue.",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = _make_agent(responses, input_handler=async_handler)
        state = await agent.run("Ask the user their favorite color")
        assert state.status.value == "completed"
        assert "Blue" in str(state.working_memory.get("answer", ""))

    @pytest.mark.asyncio
    async def test_no_handler_returns_fallback(self) -> None:
        """No input_handler returns the fallback message as tool result."""
        tc = ToolCallRequest(
            id="tc-ask-3",
            name="ask_user",
            arguments='{"question": "What do you prefer?"}',
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
                content="I will proceed with my best judgment.",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = _make_agent(responses, input_handler=None)

        events = []
        async for event in agent.astream("Help the user"):
            events.append(event)

        # Should have a TOOL_RESULT event with the fallback
        tool_results = [
            e for e in events if e.event_type == StreamEventType.TOOL_RESULT
        ]
        assert len(tool_results) == 1
        assert "best judgment" in (tool_results[0].content or "").lower()

    @pytest.mark.asyncio
    async def test_ask_user_tool_schema_in_tool_list(self) -> None:
        """ask_user tool schema should be included in the tools sent to LLM."""
        agent = _make_agent(
            [
                LLMResponse(
                    content="Done",
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    model="mock-model",
                    finish_reason="stop",
                ),
            ],
        )
        tools = agent._get_current_tools()
        assert tools is not None
        tool_names = [t["function"]["name"] for t in tools]
        assert "ask_user" in tool_names

    @pytest.mark.asyncio
    async def test_ask_user_schema_present_even_without_tool_gateway(self) -> None:
        """ask_user should be available even when no tool_gateway is set."""
        agent = _make_agent(
            [
                LLMResponse(
                    content="Done",
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    model="mock-model",
                    finish_reason="stop",
                ),
            ],
            tool_gateway=None,
        )
        tools = agent._get_current_tools()
        assert tools is not None
        tool_names = [t["function"]["name"] for t in tools]
        assert "ask_user" in tool_names
        # Only ask_user should be present
        assert len(tool_names) == 1

    @pytest.mark.asyncio
    async def test_ask_user_does_not_go_through_tool_gateway(self) -> None:
        """ask_user calls should NOT be sent to ToolGateway."""
        tool_gw = _make_tool_gateway()
        # Spy on ToolGateway.call_many_concurrent
        original_call = tool_gw.call_many_concurrent
        calls_received: list[list[ToolCall]] = []

        async def spy_call(tool_calls, **kwargs):  # noqa: ANN001, ANN003
            calls_received.append(tool_calls)
            return await original_call(tool_calls, **kwargs)

        tool_gw.call_many_concurrent = spy_call

        tc = ToolCallRequest(
            id="tc-ask-spy",
            name="ask_user",
            arguments='{"question": "Hello?"}',
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
                content="OK",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = _make_agent(
            responses,
            tool_gateway=tool_gw,
            input_handler=lambda q: "Yes",
        )
        await agent.run("Ask something")

        # ToolGateway should NOT have received any calls (ask_user was intercepted)
        assert len(calls_received) == 0

    @pytest.mark.asyncio
    async def test_input_needed_event_is_yielded(self) -> None:
        """INPUT_NEEDED event should be yielded when ask_user is called."""
        tc = ToolCallRequest(
            id="tc-ask-ev",
            name="ask_user",
            arguments='{"question": "Which format?"}',
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
                content="Using JSON format.",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = _make_agent(
            responses,
            input_handler=lambda q: "JSON",
        )

        events = []
        async for event in agent.astream("Generate output"):
            events.append(event)

        input_events = [
            e for e in events if e.event_type == StreamEventType.INPUT_NEEDED
        ]
        assert len(input_events) == 1
        assert input_events[0].content == "Which format?"

    @pytest.mark.asyncio
    async def test_ask_user_works_alongside_regular_tools(self) -> None:
        """ask_user and regular tools should work together in the same run."""
        tool_gw = _make_tool_gateway()

        # Turn 1: LLM calls ask_user
        tc_ask = ToolCallRequest(
            id="tc-ask-mix",
            name="ask_user",
            arguments='{"question": "What expression?"}',
        )
        # Turn 2: LLM calls calculator
        tc_calc = ToolCallRequest(
            id="tc-calc-mix",
            name="calculator",
            arguments='{"expression": "2+2"}',
        )
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[tc_ask],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[tc_calc],
                usage=TokenUsage(prompt_tokens=15, completion_tokens=8, total_tokens=23),
                model="mock-model",
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="2+2 = 4",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = _make_agent(
            responses,
            tool_gateway=tool_gw,
            input_handler=lambda q: "2+2",
        )
        state = await agent.run("Calculate what the user wants")
        assert state.status.value == "completed"
        assert "4" in str(state.working_memory.get("answer", ""))

    @pytest.mark.asyncio
    async def test_multiple_ask_user_calls_in_one_run(self) -> None:
        """Multiple ask_user calls should work across turns."""
        call_count = 0

        def counting_handler(q: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"answer-{call_count}"

        # Turn 1: ask_user
        tc1 = ToolCallRequest(
            id="tc-ask-m1",
            name="ask_user",
            arguments='{"question": "First question"}',
        )
        # Turn 2: ask_user again
        tc2 = ToolCallRequest(
            id="tc-ask-m2",
            name="ask_user",
            arguments='{"question": "Second question"}',
        )
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[tc1],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=None,
                tool_calls=[tc2],
                usage=TokenUsage(prompt_tokens=15, completion_tokens=8, total_tokens=23),
                model="mock-model",
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Got both answers: answer-1 and answer-2",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = _make_agent(responses, input_handler=counting_handler)

        events = []
        async for event in agent.astream("Ask two questions"):
            events.append(event)

        # Should have two INPUT_NEEDED events
        input_events = [
            e for e in events if e.event_type == StreamEventType.INPUT_NEEDED
        ]
        assert len(input_events) == 2
        assert input_events[0].content == "First question"
        assert input_events[1].content == "Second question"

        # Handler should have been called twice
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_ask_user_and_regular_tool_in_same_turn(self) -> None:
        """ask_user and a regular tool called in the same turn should both work."""
        tool_gw = _make_tool_gateway()

        # Single turn with both ask_user and calculator
        tc_ask = ToolCallRequest(
            id="tc-ask-same",
            name="ask_user",
            arguments='{"question": "Confirm?"}',
        )
        tc_calc = ToolCallRequest(
            id="tc-calc-same",
            name="calculator",
            arguments='{"expression": "3*7"}',
        )
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[tc_ask, tc_calc],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Confirmed. 3*7 = 21",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
                finish_reason="stop",
            ),
        ]
        agent = _make_agent(
            responses,
            tool_gateway=tool_gw,
            input_handler=lambda q: "Yes",
        )

        events = []
        async for event in agent.astream("Calculate and confirm"):
            events.append(event)

        # Should have INPUT_NEEDED for ask_user
        input_events = [
            e for e in events if e.event_type == StreamEventType.INPUT_NEEDED
        ]
        assert len(input_events) == 1

        # Should have TOOL_RESULT for both
        tool_results = [
            e for e in events if e.event_type == StreamEventType.TOOL_RESULT
        ]
        assert len(tool_results) == 2

        # Final state should be completed
        state = agent._state
        assert state is not None
        assert state.status.value == "completed"
