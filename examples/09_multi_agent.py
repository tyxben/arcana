"""
Arcana: Multi-Agent Collaboration

Two agents collaborate: one designs, one reviews.
Runtime handles communication, budget, and trace.

Usage:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python examples/09_multi_agent.py
"""

from __future__ import annotations

import asyncio
import os

import arcana


async def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Set DEEPSEEK_API_KEY")
        return

    runtime = arcana.Runtime(
        providers={"deepseek": api_key},
        budget=arcana.Budget(max_cost_usd=1.0),
    )

    result = await runtime.team(
        goal="Design a simple REST API for a bookmark manager app. "
        "Include endpoints, data models, authentication, and error handling.",
        agents=[
            arcana.AgentConfig(
                name="architect",
                prompt="You are a senior API architect. Design clean, RESTful endpoints. "
                "Be specific about HTTP methods, paths, request/response bodies.",
            ),
            arcana.AgentConfig(
                name="reviewer",
                prompt="You are a security-focused API reviewer. Check for: "
                "authentication gaps, missing validation, inconsistent naming, "
                "missing error codes. If the design is solid, approve with [DONE].",
            ),
        ],
        max_rounds=3,
    )

    print(f"Success: {result.success}")
    print(f"Rounds: {result.rounds}")
    print(f"Tokens: {result.total_tokens}")
    print(f"Cost: ${result.total_cost_usd:.4f}")
    print()
    for entry in result.conversation_log:
        print(f"--- Round {entry['round']} [{entry['agent']}] ---")
        print(entry["content"][:500])
        print()


if __name__ == "__main__":
    asyncio.run(main())
