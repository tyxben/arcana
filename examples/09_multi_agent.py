"""
Arcana: Multi-Agent Collaboration

Two agents collaborate via runtime.collaborate(): one designs, one reviews.
The user code drives turn order — there is no framework-prescribed round
counter or fixed schedule. The pool gives shared budget + trace; the loop
shape is yours.

Usage:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python examples/09_multi_agent.py
"""

from __future__ import annotations

import asyncio
import os

import arcana

GOAL = (
    "Design a simple REST API for a bookmark manager app. "
    "Include endpoints, data models, authentication, and error handling."
)


async def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Set DEEPSEEK_API_KEY")
        return

    runtime = arcana.Runtime(
        providers={"deepseek": api_key},
        budget=arcana.Budget(max_cost_usd=1.0),
    )

    async with runtime.collaborate() as pool:
        architect = pool.add(
            "architect",
            system=(
                "You are a senior API architect. Design clean, RESTful endpoints. "
                "Be specific about HTTP methods, paths, request/response bodies."
            ),
            provider="deepseek",
        )
        reviewer = pool.add(
            "reviewer",
            system=(
                "You are a security-focused API reviewer. Check for: "
                "authentication gaps, missing validation, inconsistent naming, "
                "missing error codes. If the design is solid, approve with [DONE]."
            ),
            provider="deepseek",
        )

        design = await architect.send(GOAL)
        review = await reviewer.send(
            f"Review this API design and reply with feedback "
            f"(end with [DONE] if acceptable):\n\n{design.content}"
        )

        # If the reviewer didn't approve, give the architect one revision pass.
        if "[DONE]" not in review.content:
            revised = await architect.send(
                f"Revise the design based on this feedback:\n\n{review.content}"
            )
        else:
            revised = design

        total_cost = sum(s.total_cost_usd for s in pool.agents.values())
        print(f"Total cost: ${total_cost:.4f}")
        print()
        print("--- architect (final) ---")
        print(revised.content[:500])
        print()
        print("--- reviewer ---")
        print(review.content[:500])


if __name__ == "__main__":
    asyncio.run(main())
