"""
Arcana: Custom Tools with Runtime

Register tools once on Runtime, every run gets access.
Tools automatically get: authorization, validation, audit logging.

Usage:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python examples/10_tool_with_runtime.py
"""

from __future__ import annotations

import asyncio
import os

import arcana


@arcana.tool(
    when_to_use="When you need to perform mathematical calculations",
    what_to_expect="Returns the exact numeric result as a string",
    failure_meaning="The expression was malformed or contained undefined operations",
)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    # In production, use a safe math parser instead of eval
    return str(eval(expression))  # noqa: S307


@arcana.tool(
    when_to_use="When you need to look up a word's definition",
    what_to_expect="Returns a brief dictionary definition",
    failure_meaning="The word was not found in the dictionary",
)
def dictionary(word: str) -> str:
    """Look up a word definition (mock)."""
    definitions = {
        "python": "A high-level programming language known for readability",
        "rust": "A systems programming language focused on safety and performance",
        "agent": "An autonomous entity that perceives and acts upon an environment",
        "runtime": "The period during which a program is executing",
    }
    result = definitions.get(word.lower())
    if result:
        return result
    return f"Definition not found for '{word}'"


async def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Set DEEPSEEK_API_KEY")
        return

    # Runtime holds tools -- every run gets access
    runtime = arcana.Runtime(
        providers={"deepseek": api_key},
        tools=[calculator, dictionary],
        budget=arcana.Budget(max_cost_usd=0.5),
    )

    print(f"Registered tools: {runtime.tools}")
    print()

    # Run 1: Should use calculator
    print("--- Task 1: Math ---")
    result = await runtime.run("What is (25 * 4) + (13 * 7)?")
    print(f"Answer: {result.output}")
    print(f"Steps: {result.steps}, Tokens: {result.tokens_used}")
    print()

    # Run 2: Should use dictionary
    print("--- Task 2: Definition ---")
    result = await runtime.run("What does 'agent' mean in the context of AI?")
    print(f"Answer: {result.output}")
    print(f"Steps: {result.steps}, Tokens: {result.tokens_used}")
    print()

    # Run 3: Direct answer (no tool needed)
    print("--- Task 3: Direct answer ---")
    result = await runtime.run("Say hello in Japanese")
    print(f"Answer: {result.output}")
    print(f"Steps: {result.steps}, Tokens: {result.tokens_used}")


if __name__ == "__main__":
    asyncio.run(main())
