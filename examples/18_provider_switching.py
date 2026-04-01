"""
Arcana: Provider Switching

Shows how to use different LLM providers: DeepSeek, OpenAI, Anthropic.
Demonstrates both arcana.run() (quick) and Runtime (production) patterns.

Usage:
    # Set one or more API keys
    export DEEPSEEK_API_KEY=sk-xxx
    export OPENAI_API_KEY=sk-proj-xxx
    export ANTHROPIC_API_KEY=sk-ant-xxx

    uv run python examples/18_provider_switching.py
"""

import asyncio
import os

import arcana
from arcana.runtime_core import RuntimeConfig


async def demo_quick_run():
    """arcana.run() -- one-off calls with different providers."""
    print("=" * 60)
    print("Demo 1: arcana.run() with different providers")
    print("=" * 60)

    # --- DeepSeek (default provider) ---
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if deepseek_key:
        result = await arcana.run(
            "What is 2 + 2? Answer in one sentence.",
            provider="deepseek",
            api_key=deepseek_key,
        )
        print(f"  DeepSeek: {result.output}")
        print(f"    Cost: ${result.cost_usd:.4f}")
    else:
        print("  DeepSeek: skipped (DEEPSEEK_API_KEY not set)")

    # --- OpenAI ---
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        result = await arcana.run(
            "What is 2 + 2? Answer in one sentence.",
            provider="openai",
            api_key=openai_key,
        )
        print(f"  OpenAI:   {result.output}")
        print(f"    Cost: ${result.cost_usd:.4f}")
    else:
        print("  OpenAI: skipped (OPENAI_API_KEY not set)")

    # --- OpenAI with specific model ---
    if openai_key:
        result = await arcana.run(
            "What is 2 + 2? Answer in one sentence.",
            provider="openai",
            model="gpt-4o",
            api_key=openai_key,
        )
        print(f"  OpenAI (gpt-4o): {result.output}")
        print(f"    Cost: ${result.cost_usd:.4f}")

    # --- Anthropic (requires pip install arcana-agent[anthropic]) ---
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        result = await arcana.run(
            "What is 2 + 2? Answer in one sentence.",
            provider="anthropic",
            model="claude-sonnet-4-20250514",  # Anthropic has no default model
            api_key=anthropic_key,
        )
        print(f"  Anthropic: {result.output}")
        print(f"    Cost: ${result.cost_usd:.4f}")
    else:
        print("  Anthropic: skipped (ANTHROPIC_API_KEY not set)")


async def demo_runtime_multi_provider():
    """Runtime with multiple providers and automatic fallback."""
    print("\n" + "=" * 60)
    print("Demo 2: Runtime with multiple providers")
    print("=" * 60)

    # Register all available providers -- empty string reads from env var
    providers = {}
    if os.environ.get("DEEPSEEK_API_KEY"):
        providers["deepseek"] = ""  # reads DEEPSEEK_API_KEY
    if os.environ.get("OPENAI_API_KEY"):
        providers["openai"] = ""  # reads OPENAI_API_KEY

    if not providers:
        print("  Skipped: set at least one API key")
        return

    default = next(iter(providers))
    runtime = arcana.Runtime(
        providers=providers,
        config=RuntimeConfig(default_provider=default),
        budget=arcana.Budget(max_cost_usd=1.0),
    )

    # Uses default provider (first in dict)
    result = await runtime.run("Name three programming languages. Be brief.")
    print(f"  Default provider: {result.output}")
    print(f"    Steps: {result.steps}, Cost: ${result.cost_usd:.4f}")

    # Override provider per-run (if openai is registered)
    if "openai" in providers:
        result = await runtime.run(
            "Name three programming languages. Be brief.",
            provider="openai",
        )
        print(f"  OpenAI override: {result.output}")
        print(f"    Cost: ${result.cost_usd:.4f}")

    await runtime.close()


async def demo_runtime_env_only():
    """Runtime using only environment variables -- no keys in code."""
    print("\n" + "=" * 60)
    print("Demo 3: Runtime with env-only keys (no keys in code)")
    print("=" * 60)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("  Skipped: set DEEPSEEK_API_KEY")
        return

    # Pass empty string to read from environment
    runtime = arcana.Runtime(
        providers={"deepseek": ""},
        budget=arcana.Budget(max_cost_usd=0.5),
    )

    result = await runtime.run("Say hello in Japanese. One sentence.")
    print(f"  Result: {result.output}")
    await runtime.close()


async def main():
    print("Arcana Provider Switching Demo")
    print()
    await demo_quick_run()
    await demo_runtime_multi_provider()
    await demo_runtime_env_only()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
