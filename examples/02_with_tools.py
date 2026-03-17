"""
Arcana: Agent with Tools

Uses @arcana.tool decorator and arcana.run() SDK.
"""

import asyncio
import os  # noqa: F401 -- remind user to set DEEPSEEK_API_KEY

# Must set: export DEEPSEEK_API_KEY=sk-xxx


async def main():
    from arcana.sdk import run, tool

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
        max_steps=5,
    )
    print(f"Answer: {result.output}")
    print(f"Steps: {result.steps}, Tokens: {result.tokens_used}")


if __name__ == "__main__":
    asyncio.run(main())
