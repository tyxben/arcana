"""Tests for prebuilt graph patterns: ReAct agent and Plan-Execute agent."""

from __future__ import annotations

from typing import Any

import pytest

from arcana.contracts.llm import LLMRequest, LLMResponse, ModelConfig, TokenUsage, ToolCallRequest
from arcana.contracts.tool import ToolResult
from arcana.graph.prebuilt.react_agent import create_react_agent
from arcana.graph.prebuilt.plan_execute import create_plan_execute_agent

MOCK_CONFIG = ModelConfig(provider="openai", model_id="mock-model")


# ── Mock Gateways ────────────────────────────────────────────────


class MockGateway:
    """Mock ModelGateway for testing.

    Cycles through a list of pre-configured LLMResponse objects.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.requests: list[Any] = []

    async def generate(self, request: Any, *args: Any, **kwargs: Any) -> LLMResponse:
        self.requests.append(request)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class MockToolGateway:
    """Mock ToolGateway for testing.

    Returns a ToolResult for each tool call with configurable output.
    """

    def __init__(self, output: str = "tool result") -> None:
        self.output = output
        self.call_count = 0
        self.received_calls: list[Any] = []

    async def call_many(
        self, tool_calls: list[Any], **kwargs: Any
    ) -> list[ToolResult]:
        self.received_calls.extend(tool_calls)
        results = []
        for tc in tool_calls:
            call_id = getattr(tc, "id", None) or ""
            name = getattr(tc, "name", None) or ""
            results.append(
                ToolResult(
                    tool_call_id=str(call_id),
                    name=str(name),
                    success=True,
                    output=self.output,
                )
            )
        self.call_count += len(tool_calls)
        return results


# ── Helpers ──────────────────────────────────────────────────────


def _make_response(
    content: str | None = None,
    tool_calls: list[ToolCallRequest] | None = None,
) -> LLMResponse:
    """Create an LLMResponse with sensible defaults."""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="mock-model",
        finish_reason="stop",
    )


def _make_tool_call(name: str = "search", call_id: str = "call_1") -> ToolCallRequest:
    """Create a ToolCallRequest."""
    return ToolCallRequest(id=call_id, name=name, arguments='{"q": "test"}')


def _msg_content(msg: Any) -> str:
    """Extract content from a message dict or object."""
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


# ── ReAct Agent Tests ────────────────────────────────────────────


class TestReActAgent:
    """Tests for create_react_agent."""

    async def test_direct_answer_no_tools(self) -> None:
        """LLM returns content without tool_calls -- agent finishes immediately."""
        gateway = MockGateway([_make_response(content="The answer is 42.")])
        tool_gw = MockToolGateway()

        agent = create_react_agent(gateway, tool_gw, model_config=MOCK_CONFIG)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "What is 42?"}]})

        assert gateway.call_count == 1
        assert tool_gw.call_count == 0

        # Without add_messages reducer, state["messages"] is replaced by last node output.
        # The agent node returns [assistant_msg], so final messages has 1 entry.
        messages = result["messages"]
        assert len(messages) == 1
        assert _msg_content(messages[0]) == "The answer is 42."

    async def test_tool_call_then_answer(self) -> None:
        """LLM returns tool_calls, tools execute, LLM called again with results."""
        tool_call = _make_tool_call(name="search", call_id="call_1")
        responses = [
            # First call: LLM wants to use a tool
            _make_response(content="Let me search.", tool_calls=[tool_call]),
            # Second call: LLM gives final answer after seeing tool result
            _make_response(content="Based on the search, the answer is X."),
        ]

        gateway = MockGateway(responses)
        tool_gw = MockToolGateway(output="search result: X")

        agent = create_react_agent(gateway, tool_gw, model_config=MOCK_CONFIG)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "Search for X"}]})

        # LLM called twice: once producing tool_calls, once producing final answer
        assert gateway.call_count == 2
        assert tool_gw.call_count == 1

        # Final messages from last agent node output
        messages = result["messages"]
        assert _msg_content(messages[-1]) == "Based on the search, the answer is X."

    async def test_multiple_tool_rounds(self) -> None:
        """LLM does two rounds of tool calls before final answer."""
        tc1 = _make_tool_call(name="search", call_id="call_1")
        tc2 = _make_tool_call(name="lookup", call_id="call_2")

        responses = [
            _make_response(content="Searching...", tool_calls=[tc1]),
            _make_response(content="Looking up...", tool_calls=[tc2]),
            _make_response(content="Final answer."),
        ]

        gateway = MockGateway(responses)
        tool_gw = MockToolGateway(output="data")

        agent = create_react_agent(gateway, tool_gw, model_config=MOCK_CONFIG)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "Multi-step"}]})

        assert gateway.call_count == 3
        assert tool_gw.call_count == 2

        messages = result["messages"]
        assert _msg_content(messages[-1]) == "Final answer."

    async def test_max_iterations_safety(self) -> None:
        """Agent stops after max_iterations even if LLM keeps requesting tools."""
        tool_call = _make_tool_call()
        # Always return tool calls -- never a final answer
        gateway = MockGateway([_make_response(content="again", tool_calls=[tool_call])])
        tool_gw = MockToolGateway()

        agent = create_react_agent(gateway, tool_gw, model_config=MOCK_CONFIG, max_iterations=3)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "loop"}]})

        # Should have stopped due to max_iterations
        assert gateway.call_count <= 4  # at most max_iterations + 1
        assert "messages" in result

    async def test_streaming(self) -> None:
        """ReAct agent works with astream."""
        gateway = MockGateway([_make_response(content="Streamed answer.")])
        tool_gw = MockToolGateway()

        agent = create_react_agent(gateway, tool_gw, model_config=MOCK_CONFIG)
        events = []
        async for event in agent.astream(
            {"messages": [{"role": "user", "content": "stream test"}]},
            mode="updates",
        ):
            events.append(event)

        assert len(events) >= 1
        assert events[0]["node"] == "agent"


# ── Plan-Execute Agent Tests ─────────────────────────────────────


class TestPlanExecuteAgent:
    """Tests for create_plan_execute_agent."""

    async def test_plan_execute_verify_complete(self) -> None:
        """Planner plans, executor executes, verifier says COMPLETE -- ends."""
        responses = [
            # Planner
            _make_response(content="Step 1: Do A. Step 2: Do B."),
            # Executor
            _make_response(content="Done A and B."),
            # Verifier
            _make_response(content="COMPLETE - task accomplished."),
        ]

        gateway = MockGateway(responses)
        tool_gw = MockToolGateway()

        agent = create_plan_execute_agent(gateway, tool_gw, model_config=MOCK_CONFIG)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "Do A then B"}]})

        # All three nodes should have been called
        assert gateway.call_count == 3
        assert result.get("is_complete") is True

        # Verifier's message is the last node output (replaces messages)
        messages = result["messages"]
        assert len(messages) >= 1
        assert "COMPLETE" in _msg_content(messages[-1]).upper()

    async def test_plan_execute_replan(self) -> None:
        """Verifier gives feedback, agent replans and succeeds on second attempt."""
        responses = [
            # First planner
            _make_response(content="Plan: Do X"),
            # First executor
            _make_response(content="Did X but incomplete"),
            # First verifier -- NOT complete
            _make_response(content="Needs improvement: missing step Y"),
            # Second planner (replan)
            _make_response(content="Revised plan: Do X then Y"),
            # Second executor
            _make_response(content="Did X and Y"),
            # Second verifier -- COMPLETE
            _make_response(content="COMPLETE"),
        ]

        gateway = MockGateway(responses)
        tool_gw = MockToolGateway()

        agent = create_plan_execute_agent(gateway, tool_gw, model_config=MOCK_CONFIG, max_replans=3)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "Do X and Y"}]})

        assert gateway.call_count == 6
        assert result.get("is_complete") is True

    async def test_plan_execute_max_replans(self) -> None:
        """Agent stops after max_replans even if verifier never says COMPLETE."""
        # Verifier never says COMPLETE
        responses = [
            _make_response(content="Plan"),
            _make_response(content="Executed"),
            _make_response(content="Not done yet, needs more work"),
        ]

        gateway = MockGateway(responses)
        tool_gw = MockToolGateway()

        agent = create_plan_execute_agent(gateway, tool_gw, model_config=MOCK_CONFIG, max_replans=1)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "impossible"}]})

        # Should stop after 1 replan attempt:
        # Round 1: planner + executor + verifier = 3
        # Round 2 (replan): planner + executor + verifier = 3
        # Total: 6 calls max
        assert gateway.call_count <= 6
        assert "messages" in result

    async def test_plan_execute_streaming(self) -> None:
        """Plan-execute agent works with astream in updates mode."""
        responses = [
            _make_response(content="The plan."),
            _make_response(content="Executed."),
            _make_response(content="COMPLETE"),
        ]

        gateway = MockGateway(responses)
        tool_gw = MockToolGateway()

        agent = create_plan_execute_agent(gateway, tool_gw, model_config=MOCK_CONFIG)
        events = []
        async for event in agent.astream(
            {"messages": [{"role": "user", "content": "stream plan-execute"}]},
            mode="updates",
        ):
            events.append(event)

        # Should see updates from planner, executor, verifier
        node_names = [e["node"] for e in events]
        assert "planner" in node_names
        assert "executor" in node_names
        assert "verifier" in node_names
