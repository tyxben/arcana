"""
Arcana: Interactive Chat Session

Demonstrates runtime.chat() for multi-turn conversation.
The agent maintains context across messages and can use tools.

This uses the ChatSession API for clean multi-turn interaction.

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

    # --- Programmatic multi-turn via runtime.chat() ---
    # NOTE: runtime.chat() is being implemented by another agent.
    # When available, usage will be:
    #
    #   async with runtime.chat(system_prompt="You are a math tutor.") as chat:
    #       r = await chat.send("What is 15% of 240?")
    #       print(f"Agent: {r.content}")
    #
    #       r = await chat.send("Now add tax of 8% to that")
    #       print(f"Agent: {r.content}")
    #
    #       r = await chat.send("Summarize the calculation steps")
    #       print(f"Agent: {r.content}")
    #
    #       print(f"\nTotal: {chat.total_tokens} tokens, ${chat.total_cost_usd:.4f}")

    # --- Working implementation using raw LLM messages ---
    # This demonstrates multi-turn context without the ChatSession wrapper.
    from arcana.contracts.llm import LLMRequest, Message, MessageRole

    model_config = runtime._resolve_model_config()
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful math tutor. Use the calculator tool when you need to compute."),
    ]

    questions = [
        "What is 15% of 240?",
        "Now add tax of 8% to that",
        "Summarize the calculation steps",
    ]

    total_tokens = 0
    for q in questions:
        print(f"You: {q}")
        messages.append(Message(role=MessageRole.USER, content=q))

        response = await runtime._gateway.generate(
            request=LLMRequest(messages=messages),
            config=model_config,
        )

        assistant_text = response.content or ""
        messages.append(Message(role=MessageRole.ASSISTANT, content=assistant_text))

        tokens = response.usage.total_tokens if response.usage else 0
        total_tokens += tokens

        print(f"Agent: {assistant_text}")
        print(f"  ({tokens} tokens)")
        print()

    print(f"Total: {total_tokens} tokens across {len(questions)} turns")

    await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
