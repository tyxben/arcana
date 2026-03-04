"""ReAct agent example using mock gateways.

Demonstrates how to use create_react_agent with mock LLM and tool gateways
to build an agent that reasons and acts in a loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from arcana.contracts.llm import LLMResponse, ModelConfig, TokenUsage, ToolCallRequest
from arcana.contracts.tool import ToolResult
from arcana.graph.prebuilt.react_agent import create_react_agent

DEMO_CONFIG = ModelConfig(provider="openai", model_id="demo-model")


# ── Mock Gateways ────────────────────────────────────────────────


class DemoModelGateway:
    """A mock LLM gateway that returns scripted responses.

    Simulates an LLM that:
    1. First call: decides to search using a tool
    2. Second call: provides a final answer based on tool results
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def generate(self, request: Any, *args: Any, **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        usage = TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70)

        if self.call_count == 1:
            # First call: LLM decides to use a tool
            print("[LLM] Deciding to search for information...")
            return LLMResponse(
                content="I need to search for this information.",
                tool_calls=[
                    ToolCallRequest(
                        id="call_001",
                        name="web_search",
                        arguments='{"query": "latest Python release"}',
                    )
                ],
                usage=usage,
                model="demo-model",
                finish_reason="tool_calls",
            )
        else:
            # Second call: LLM provides final answer
            print("[LLM] Providing final answer based on tool results...")
            return LLMResponse(
                content="Based on my search, Python 3.12 is the latest stable release. "
                "It includes improvements to error messages, performance optimizations, "
                "and new syntax features like type parameter syntax.",
                tool_calls=None,
                usage=usage,
                model="demo-model",
                finish_reason="stop",
            )


class DemoToolGateway:
    """A mock tool gateway that returns scripted tool results."""

    async def call_many(self, tool_calls: list[Any], **kwargs: Any) -> list[ToolResult]:
        results = []
        for tc in tool_calls:
            call_id = getattr(tc, "id", None) or getattr(tc, "call_id", "unknown")
            name = getattr(tc, "name", None) or getattr(tc, "tool_name", "unknown")
            print(f"[Tool] Executing '{name}' (id={call_id})")
            results.append(
                ToolResult(
                    tool_call_id=str(call_id),
                    name=str(name),
                    success=True,
                    output="Python 3.12 was released on October 2, 2023. "
                    "Key features: improved error messages, type parameter syntax (PEP 695), "
                    "per-interpreter GIL (PEP 684).",
                )
            )
        return results


# ── Main Demo ────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 60)
    print("ReAct Agent Demo (mock gateways)")
    print("=" * 60)
    print()

    # Build the agent
    gateway = DemoModelGateway()
    tool_gw = DemoToolGateway()
    agent = create_react_agent(
        gateway,
        tool_gw,
        model_config=DEMO_CONFIG,
        system_prompt="You are a helpful research assistant.",
        max_iterations=10,
    )

    # Run with ainvoke
    print("--- Running agent with ainvoke ---")
    print()
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is the latest Python release?"}]
    })

    print()
    print("--- Final messages ---")
    for i, msg in enumerate(result.get("messages", [])):
        msg_data = msg if isinstance(msg, dict) else msg.model_dump()
        role = msg_data.get("role", "?")
        content = msg_data.get("content", "")
        preview = content[:80] + "..." if len(content) > 80 else content
        has_tools = " [+tool_calls]" if msg_data.get("tool_calls") else ""
        print(f"  [{i}] {role}: {preview}{has_tools}")

    print()
    print(f"LLM calls: {gateway.call_count}")
    print()

    # Run with astream
    print("--- Running agent with astream (updates mode) ---")
    print()
    gateway2 = DemoModelGateway()
    agent2 = create_react_agent(
        gateway2,
        DemoToolGateway(),
        model_config=DEMO_CONFIG,
        system_prompt="You are a helpful assistant.",
    )

    async for event in agent2.astream(
        {"messages": [{"role": "user", "content": "Tell me about Python 3.12"}]},
        mode="updates",
    ):
        node = event.get("node", "?")
        output_keys = list(event.get("output", {}).keys())
        print(f"  Node '{node}' -> keys: {output_keys}")

    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
