"""NodeRunner - executes individual graph nodes with tracing."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from arcana.contracts.trace import AgentRole, EventType, TraceEvent

if TYPE_CHECKING:
    from arcana.contracts.trace import TraceContext
    from arcana.trace.writer import TraceWriter


class NodeRunner:
    """Executes individual graph nodes, handling sync/async and tracing."""

    def __init__(
        self,
        *,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self._trace_writer = trace_writer

    async def run(
        self,
        node_name: str,
        fn: Callable[..., Any],
        state: dict[str, Any],
        trace_ctx: TraceContext | None = None,
    ) -> dict[str, Any]:
        """
        Execute a node function and return its output.

        Handles both sync and async functions.
        Records GRAPH_NODE_START and GRAPH_NODE_COMPLETE trace events.
        """
        run_id = trace_ctx.run_id if trace_ctx else ""
        step_id = trace_ctx.new_step_id() if trace_ctx else ""

        # Emit start event
        if self._trace_writer and trace_ctx:
            self._trace_writer.write(
                TraceEvent(
                    run_id=run_id,
                    step_id=step_id,
                    role=AgentRole.SYSTEM,
                    event_type=EventType.GRAPH_NODE_START,
                    metadata={"node": node_name},
                )
            )

        start_time = time.monotonic()

        # Execute the node function
        try:
            if asyncio.iscoroutinefunction(fn) or (
                callable(fn) and inspect.iscoroutinefunction(fn.__call__)  # type: ignore[operator]
            ):
                result = await fn(state)
            else:
                result = fn(state)
        except Exception:
            # Emit error event
            if self._trace_writer and trace_ctx:
                self._trace_writer.write(
                    TraceEvent(
                        run_id=run_id,
                        step_id=step_id,
                        role=AgentRole.SYSTEM,
                        event_type=EventType.ERROR,
                        metadata={"node": node_name, "phase": "execution"},
                    )
                )
            raise

        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Normalize output to dict
        if result is None:
            result = {}
        elif not isinstance(result, dict):
            raise TypeError(
                f"Node '{node_name}' must return a dict or None, got {type(result).__name__}"
            )

        # Emit complete event
        if self._trace_writer and trace_ctx:
            self._trace_writer.write(
                TraceEvent(
                    run_id=run_id,
                    step_id=step_id,
                    role=AgentRole.SYSTEM,
                    event_type=EventType.GRAPH_NODE_COMPLETE,
                    metadata={
                        "node": node_name,
                        "duration_ms": duration_ms,
                        "output_keys": list(result.keys()),
                    },
                )
            )

        return result
