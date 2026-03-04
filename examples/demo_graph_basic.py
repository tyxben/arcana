"""Basic graph example: search -> summarize pipeline.

Demonstrates StateGraph construction, compile, ainvoke, and astream.
"""

from __future__ import annotations

import asyncio
from typing import Any

from arcana.graph.constants import END, START
from arcana.graph.state_graph import StateGraph


# ── Dummy node functions ─────────────────────────────────────────


async def search(state: dict[str, Any]) -> dict[str, Any]:
    """Simulate a search operation."""
    query = state.get("query", "")
    print(f"[search] Searching for: {query}")
    results = [
        f"Result 1 for '{query}': Python is a programming language.",
        f"Result 2 for '{query}': Python was created by Guido van Rossum.",
        f"Result 3 for '{query}': Python 3.11 added performance improvements.",
    ]
    return {"results": results, "messages": [{"role": "system", "content": f"Found {len(results)} results"}]}


async def summarize(state: dict[str, Any]) -> dict[str, Any]:
    """Simulate summarization of search results."""
    results = state.get("results", [])
    print(f"[summarize] Summarizing {len(results)} results")
    summary = "Summary: " + " | ".join(r.split(": ", 1)[-1] for r in results)
    return {"summary": summary, "messages": [{"role": "assistant", "content": summary}]}


# ── Build and run the graph ──────────────────────────────────────


def build_graph() -> Any:
    """Build a search -> summarize pipeline graph."""
    graph = StateGraph()
    graph.add_node("search", search)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile(name="search-summarize")


async def demo_ainvoke() -> None:
    """Demonstrate ainvoke -- run graph to completion."""
    print("=" * 60)
    print("Demo: ainvoke (run to completion)")
    print("=" * 60)

    app = build_graph()
    result = await app.ainvoke({"query": "Python programming", "messages": []})

    print(f"\nFinal state:")
    print(f"  query:   {result.get('query')}")
    print(f"  results: {len(result.get('results', []))} items")
    print(f"  summary: {result.get('summary')}")
    print(f"  messages: {len(result.get('messages', []))} total")
    print()


async def demo_astream() -> None:
    """Demonstrate astream -- stream node updates."""
    print("=" * 60)
    print("Demo: astream (streaming updates)")
    print("=" * 60)

    app = build_graph()

    print("\n--- mode='updates' ---")
    async for event in app.astream(
        {"query": "Python history", "messages": []},
        mode="updates",
    ):
        node = event.get("node", "?")
        output_keys = list(event.get("output", {}).keys())
        print(f"  Node '{node}' produced keys: {output_keys}")

    print("\n--- mode='values' ---")
    app2 = build_graph()
    step = 0
    async for state in app2.astream(
        {"query": "Python performance", "messages": []},
        mode="values",
    ):
        step += 1
        print(f"  Step {step}: state has keys {list(state.keys())}")

    print()


async def main() -> None:
    await demo_ainvoke()
    await demo_astream()


if __name__ == "__main__":
    asyncio.run(main())
