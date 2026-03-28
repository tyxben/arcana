"""Example 13: Interactive Chat Session.

Demonstrates multi-turn conversation using runtime.chat() with
persistent history, shared budget, and automatic context management.

The ChatSession maintains conversation history across send() calls,
so the LLM sees full context from previous turns. Budget accumulates
across the entire session. Tools remain available in every turn.

Usage:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python examples/13_interactive_chat.py
"""

from __future__ import annotations

import asyncio
import os

import arcana


@arcana.tool(
    when_to_use="When you need to perform a calculation",
    what_to_expect="The numeric result",
    failure_meaning="Invalid expression",
)
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    allowed = set("0123456789+-*/(). ")
    if all(c in allowed for c in expression):
        return str(eval(expression))  # noqa: S307
    return "Invalid expression"


async def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Set DEEPSEEK_API_KEY")
        return

    runtime = arcana.Runtime(
        providers={"deepseek": api_key},
        tools=[calculator],
        budget=arcana.Budget(max_cost_usd=1.0),
    )

    print("Chat session started. Type 'exit' to quit.\n")

    async with runtime.chat(
        system_prompt="You are a helpful math tutor. Use the calculator tool when you need to compute.",
    ) as session:
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in ("exit", "quit"):
                break

            response = await session.send(user_input)
            print(f"Agent: {response.content}")
            print(f"  ({response.tokens_used} tokens, ${response.cost_usd:.4f})")
            if response.tool_calls_made:
                print(f"  (used {response.tool_calls_made} tool call(s))")
            print()

        print("\nSession summary:")
        print(f"  Messages: {session.message_count}")
        print(f"  Total cost: ${session.total_cost_usd:.4f}")
        print(f"  Total tokens: {session.total_tokens}")

    await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
