"""
Arcana: Agent with Tools

Uses @arcana.tool decorator and arcana.run() SDK.

Usage:
    DEEPSEEK_API_KEY="sk-xxx" uv run python examples/02_with_tools.py
"""

import asyncio
import os
import sys


async def main():
    from arcana.sdk import run, tool

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Set DEEPSEEK_API_KEY or pass api_key to run()")
        sys.exit(1)

    @tool(
        when_to_use="When you need to calculate math expressions",
        what_to_expect="Returns the numeric result",
    )
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return str(eval(expression))  # noqa: S307

    result = await run(
        "What is (15 * 37) + (89 * 2)?",
        tools=[calculator],
        provider="deepseek",
        api_key=api_key,
        max_turns=5,
    )
    print(f"Answer: {result.output}")
    print(f"Steps: {result.steps}, Tokens: {result.tokens_used}")


if __name__ == "__main__":
    asyncio.run(main())
