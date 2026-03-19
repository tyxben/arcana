"""
Arcana: Graph Orchestration (Advanced)

Graph is an **advanced platform capability** — explicit nodes, edges,
reducers, and interrupt/resume for complex control flows.

For most tasks, use ``runtime.run(goal)`` (ConversationAgent, LLM-native).
Use Graph when you need deterministic step ordering, branching, or
human-in-the-loop.

This example shows two patterns:
  1. Custom graph — manual nodes + conditional routing
  2. Prebuilt ReAct — factory function for agent ↔ tools loop

Usage:
    uv run python examples/12_graph_orchestration.py
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from pydantic import BaseModel, Field

from arcana.contracts.llm import LLMResponse, ModelConfig, TokenUsage, ToolCallRequest
from arcana.contracts.tool import ToolResult
from arcana.graph import END, START, StateGraph, append_reducer
from arcana.graph.prebuilt.react_agent import create_react_agent

# ── Mock gateways (replace with real ones in production) ─────────


class DemoGateway:
    """Mock LLM gateway that returns canned responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self._idx = 0

    async def generate(self, request: Any, *args: Any, **kwargs: Any) -> LLMResponse:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp


class DemoToolGateway:
    """Mock tool gateway that echoes tool calls."""

    async def call_many(self, tool_calls: list[Any], **kwargs: Any) -> list[ToolResult]:
        return [
            ToolResult(
                tool_call_id=getattr(tc, "id", ""),
                name=getattr(tc, "name", ""),
                success=True,
                output=f"Result for {getattr(tc, 'name', 'unknown')}",
            )
            for tc in tool_calls
        ]


# ── Pattern 1: Custom graph ─────────────────────────────────────


class AnalysisState(BaseModel):
    """Typed state with message accumulation."""
    messages: Annotated[list, append_reducer] = Field(default_factory=list)
    analysis: str = ""
    decision: str = ""


async def analyze(state: dict[str, Any]) -> dict[str, Any]:
    """Step 1: Analyze input."""
    messages = state.get("messages", [])
    user_msg = messages[-1]["content"] if messages else "nothing"
    return {
        "analysis": f"Analyzed: {user_msg}",
        "messages": [{"role": "assistant", "content": f"Analysis complete for: {user_msg}"}],
    }


async def decide(state: dict[str, Any]) -> dict[str, Any]:
    """Step 2: Route based on analysis."""
    analysis = state.get("analysis", "")
    if "urgent" in analysis.lower():
        return {"decision": "escalate"}
    return {"decision": "respond"}


async def escalate(state: dict[str, Any]) -> dict[str, Any]:
    """Branch A: Escalate."""
    return {"messages": [{"role": "assistant", "content": "Escalated to human."}]}


async def respond(state: dict[str, Any]) -> dict[str, Any]:
    """Branch B: Respond directly."""
    return {"messages": [{"role": "assistant", "content": "Here is your answer."}]}


def route_decision(state: dict[str, Any]) -> str:
    return "escalate" if state.get("decision") == "escalate" else "respond"


async def demo_custom_graph():
    print("=== Pattern 1: Custom Graph ===\n")

    graph = StateGraph(state_schema=AnalysisState)
    graph.add_node("analyze", analyze)
    graph.add_node("decide", decide)
    graph.add_node("escalate", escalate)
    graph.add_node("respond", respond)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "decide")
    graph.add_conditional_edges("decide", route_decision, {
        "escalate": "escalate",
        "respond": "respond",
    })
    graph.add_edge("escalate", END)
    graph.add_edge("respond", END)

    app = graph.compile()

    # Normal case → respond branch
    result = await app.ainvoke({
        "messages": [{"role": "user", "content": "What is Arcana?"}],
    })
    print(f"  Normal:  {result['messages'][-1]['content']}")
    print(f"  Decision: {result['decision']}")

    # Urgent case → escalate branch
    result = await app.ainvoke({
        "messages": [{"role": "user", "content": "URGENT: system down"}],
    })
    print(f"  Urgent:  {result['messages'][-1]['content']}")
    print(f"  Decision: {result['decision']}")


# ── Pattern 2: Prebuilt ReAct ────────────────────────────────────


async def demo_react_agent():
    print("\n=== Pattern 2: Prebuilt ReAct Agent ===\n")

    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    # Gateway returns: tool call → final answer
    gateway = DemoGateway([
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="tc1", name="search", arguments='{"q": "Arcana"}')],
            usage=usage, model="mock", finish_reason="tool_calls",
        ),
        LLMResponse(
            content="Arcana is an Agent Runtime.",
            tool_calls=None,
            usage=usage, model="mock", finish_reason="stop",
        ),
    ])
    tool_gw = DemoToolGateway()

    react = create_react_agent(
        gateway=gateway,
        tool_gateway=tool_gw,
        model_config=ModelConfig(provider="mock", model_id="mock"),
        system_prompt="You are a helpful assistant.",
    )

    result = await react.ainvoke({
        "messages": [{"role": "user", "content": "What is Arcana?"}],
    })

    print("  Messages:")
    for msg in result["messages"]:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg.get("content", "") if isinstance(msg, dict) else msg.content
        if content:
            print(f"    [{role}] {content}")


# ── Streaming ────────────────────────────────────────────────────


async def demo_streaming():
    print("\n=== Streaming Graph Execution ===\n")

    graph = StateGraph()
    graph.add_node("step1", lambda s: {"result": "step1 done"})
    graph.add_node("step2", lambda s: {"result": "step2 done", "final": True})
    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", END)

    app = graph.compile()

    async for event in app.astream({"result": ""}, mode="updates"):
        print(f"  Node: {event.get('node', '?'):10s} → {event.get('output', {})}")


async def main():
    await demo_custom_graph()
    await demo_react_agent()
    await demo_streaming()
    print("\nGraph is an advanced capability. For most tasks, use runtime.run().")


if __name__ == "__main__":
    asyncio.run(main())
