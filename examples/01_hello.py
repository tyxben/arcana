"""
Arcana: Hello World

The simplest possible agent -- one LLM call via the SDK.

Usage:
    # With DeepSeek (default)
    DEEPSEEK_API_KEY="sk-xxx" uv run python examples/01_hello.py

    # With OpenAI
    OPENAI_API_KEY="sk-proj-xxx" uv run python examples/01_hello.py --provider openai
"""

import asyncio
import os
import sys

import arcana


async def main():
    # Pick provider from CLI arg or default to deepseek
    provider = "deepseek"
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        if idx + 1 < len(sys.argv):
            provider = sys.argv[idx + 1]

    # Resolve API key from environment
    env_var = f"{provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var, "")
    if not api_key:
        print(f"Set {env_var} or pass --provider <name>")
        sys.exit(1)

    result = await arcana.run(
        "Say hello in 3 languages",
        provider=provider,
        api_key=api_key,
    )
    print(result.output)
    print(f"Tokens: {result.tokens_used} | Cost: ${result.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
