"""Streaming support for graph execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arcana.graph.compiled_graph import CompiledGraph


async def astream(
    compiled_graph: CompiledGraph,
    input: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    mode: str = "values",
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Execute a graph with streaming output.

    Modes:
    - "values": yield full state after each node completes
    - "updates": yield {"node": name, "output": {...}} after each node
    - "messages": yield new messages added at each step
    """
    if mode not in ("values", "updates", "messages"):
        raise ValueError(f"Invalid stream mode '{mode}'. Must be 'values', 'updates', or 'messages'")

    from arcana.graph.executor import GraphExecutor

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    executor = GraphExecutor(
        config=compiled_graph._config,
        nodes=compiled_graph._nodes,
        node_fns=compiled_graph._node_fns,
        edges=compiled_graph._edges,
        conditional_edges=compiled_graph._conditional_edges,
        conditional_fns=compiled_graph._conditional_fns,
        state_schema=compiled_graph._state_schema,
        checkpointer=compiled_graph._checkpointer,
    )
    executor.event_queue = queue

    # Run executor in background task
    task = asyncio.create_task(executor.execute(input, config))

    previous_messages: list[Any] = list(input.get("messages", []))

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
            except TimeoutError:
                if task.done():
                    # Drain remaining events
                    while not queue.empty():
                        event = queue.get_nowait()
                        result = _format_event(event, mode, previous_messages)
                        if result is not None:
                            if mode == "messages":
                                previous_messages = list(
                                    event.get("state", {}).get("messages", [])
                                )
                            yield result
                    break
                continue

            if event.get("type") == "done":
                break

            result = _format_event(event, mode, previous_messages)
            if result is not None:
                if mode == "messages":
                    previous_messages = list(event.get("state", {}).get("messages", []))
                yield result
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        else:
            # Re-raise any exception from the executor
            exc = task.exception() if not task.cancelled() else None
            if exc:
                raise exc


def _format_event(
    event: dict[str, Any],
    mode: str,
    previous_messages: list[Any],
) -> dict[str, Any] | None:
    """Format a raw event according to the streaming mode."""
    if event.get("type") != "node_complete":
        return None

    if mode == "values":
        return dict(event.get("state", {}))
    elif mode == "updates":
        return {"node": event["node"], "output": event.get("output", {})}
    elif mode == "messages":
        current_messages = event.get("state", {}).get("messages", [])
        new_messages = current_messages[len(previous_messages):]
        if new_messages:
            return {"node": event["node"], "messages": new_messages}
        return None
    return None
