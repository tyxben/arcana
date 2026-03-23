"""GraphExecutor - core execution engine for graph workflows."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel

from arcana.contracts.graph import (
    ConditionalEdgeSpec,
    GraphConfig,
    GraphEdgeSpec,
    GraphExecutionState,
    GraphNodeSpec,
)
from arcana.contracts.trace import AgentRole, EventType, TraceContext, TraceEvent
from arcana.graph.constants import END
from arcana.graph.interrupt import Command, GraphInterrupt
from arcana.graph.node_runner import NodeRunner
from arcana.graph.reducers import apply_reducers, extract_reducers

if TYPE_CHECKING:
    from arcana.graph.checkpointer import GraphCheckpointer
    from arcana.trace.writer import TraceWriter


class GraphExecutionError(Exception):
    """Raised when graph execution encounters an unrecoverable error."""


class GraphExecutor:
    """
    Core execution engine for graph workflows.

    Implements the main execution loop:
    1. Check interrupt_before
    2. Execute node function
    3. Apply reducers to merge output into state
    4. Check interrupt_after
    5. Record trace events
    6. Route to next node
    """

    def __init__(
        self,
        *,
        config: GraphConfig,
        nodes: dict[str, GraphNodeSpec],
        node_fns: dict[str, Callable[..., Any]],
        edges: list[GraphEdgeSpec],
        conditional_edges: list[ConditionalEdgeSpec],
        conditional_fns: dict[str, Callable[..., str]],
        state_schema: type[BaseModel] | None = None,
        checkpointer: Any | None = None,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self._config = config
        self._nodes = nodes
        self._node_fns = node_fns
        self._edges = edges
        self._conditional_edges = conditional_edges
        self._conditional_fns = conditional_fns
        self._state_schema = state_schema
        self._checkpointer: GraphCheckpointer | None = checkpointer
        self._trace_writer = trace_writer
        self._reducers = extract_reducers(state_schema)
        self._node_runner = NodeRunner(trace_writer=trace_writer)
        self._event_queue: asyncio.Queue[dict[str, Any]] | None = None
        self._execution_state = GraphExecutionState()

    @property
    def event_queue(self) -> asyncio.Queue[dict[str, Any]] | None:
        return self._event_queue

    @event_queue.setter
    def event_queue(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._event_queue = queue

    async def execute(
        self,
        initial_state: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the graph from the entry point to END."""
        state = self._init_state(initial_state)
        trace_ctx = TraceContext(run_id=str(uuid4()))
        current = self._config.entry_point

        while current != END:
            self._execution_state.current_node = current
            self._execution_state.visited_nodes.append(current)

            # 1. interrupt_before check
            if current in self._config.interrupt_before:
                await self._handle_interrupt(current, state, trace_ctx, phase="before")

            # 2. Execute node
            fn = self._node_fns.get(current)
            if fn is None:
                raise GraphExecutionError(f"No function registered for node '{current}'")

            output = await self._node_runner.run(current, fn, state, trace_ctx)

            # 3. Apply reducers to merge output into state
            state = apply_reducers(state, output, self._reducers)
            self._execution_state.node_outputs[current] = output

            # 4. interrupt_after check
            if current in self._config.interrupt_after:
                await self._handle_interrupt(current, state, trace_ctx, phase="after")

            # 5. Emit streaming event
            await self._emit_event(current, state, output)

            # 6. Route to next node
            next_node = self._route(current, state, output)

            # Record transition
            if self._trace_writer:
                self._trace_writer.write(
                    TraceEvent(
                        run_id=trace_ctx.run_id,
                        step_id=trace_ctx.new_step_id(),
                        role=AgentRole.SYSTEM,
                        event_type=EventType.GRAPH_TRANSITION,
                        metadata={"from": current, "to": next_node},
                    )
                )

            current = next_node

        # Signal completion
        if self._event_queue:
            await self._event_queue.put({"type": "done", "state": state})

        return state

    async def resume(
        self,
        checkpoint_data: dict[str, Any],
        command: Command | None = None,
    ) -> dict[str, Any]:
        """Resume execution from a checkpoint."""
        state = dict(checkpoint_data.get("state", {}))
        resume_node = checkpoint_data.get("resume_node", "")
        phase = checkpoint_data.get("phase", "before")

        if command:
            # Apply state updates from command
            if command.update:
                state.update(command.update)

            # Override resume node if goto specified
            if command.goto:
                resume_node = command.goto
                phase = "before"

            # Store resume value in state for the node to access
            if command.resume is not None:
                state["__resume_value__"] = command.resume

        trace_ctx = TraceContext(run_id=str(uuid4()))

        if phase == "before":
            # Execute from the interrupted node
            current = resume_node
        else:
            # Route from the interrupted node (it already executed)
            current = self._route(resume_node, state, {})

        while current != END:
            self._execution_state.current_node = current
            self._execution_state.visited_nodes.append(current)

            if current in self._config.interrupt_before and current != resume_node:
                await self._handle_interrupt(current, state, trace_ctx, phase="before")

            fn = self._node_fns.get(current)
            if fn is None:
                raise GraphExecutionError(f"No function registered for node '{current}'")

            output = await self._node_runner.run(current, fn, state, trace_ctx)
            state = apply_reducers(state, output, self._reducers)

            if current in self._config.interrupt_after:
                await self._handle_interrupt(current, state, trace_ctx, phase="after")

            await self._emit_event(current, state, output)
            current = self._route(current, state, output)

        if self._event_queue:
            await self._event_queue.put({"type": "done", "state": state})

        # Clean up resume value
        state.pop("__resume_value__", None)
        return state

    def _init_state(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        """Initialize the graph state, optionally from a schema."""
        if self._state_schema:
            # Create schema instance with defaults, then overlay initial values
            defaults = self._state_schema.model_fields
            state: dict[str, Any] = {}
            for field_name, field_info in defaults.items():
                if field_info.default is not None:
                    state[field_name] = field_info.default
                elif field_info.default_factory is not None:
                    factory = field_info.default_factory
                    state[field_name] = factory()  # type: ignore[call-arg]
            state.update(initial_state)
            return state
        return dict(initial_state)

    def _route(
        self,
        current: str,
        state: dict[str, Any],
        output: dict[str, Any],
    ) -> str:
        """Determine the next node based on edges and conditions."""
        # Check conditional edges first
        for cond in self._conditional_edges:
            if cond.source == current:
                path_fn = self._conditional_fns.get(current)
                if path_fn is None:
                    raise GraphExecutionError(
                        f"No path function for conditional edge from '{current}'"
                    )

                key = path_fn(state)

                if cond.path_map:
                    target = cond.path_map.get(key)
                    if target is None:
                        raise GraphExecutionError(
                            f"Conditional edge from '{current}' returned unmapped key "
                            f"'{key}'. Available keys: {list(cond.path_map.keys())}"
                        )
                    return target
                else:
                    # Use key directly as node name or END
                    return key

        # Check direct edges
        for edge in self._edges:
            if edge.source == current:
                return edge.target

        raise GraphExecutionError(
            f"No outgoing edge from node '{current}'. "
            f"Add an edge or conditional edge from this node."
        )

    async def _handle_interrupt(
        self,
        node_id: str,
        state: dict[str, Any],
        trace_ctx: TraceContext,
        phase: str,
    ) -> None:
        """Handle an interrupt point (save checkpoint and raise)."""
        checkpoint_id = ""

        if self._checkpointer:
            checkpoint_id = await self._checkpointer.save(
                state=state,
                node_id=node_id,
                metadata={"phase": phase, "resume_node": node_id},
            )

        if self._trace_writer:
            self._trace_writer.write(
                TraceEvent(
                    run_id=trace_ctx.run_id,
                    step_id=trace_ctx.new_step_id(),
                    role=AgentRole.SYSTEM,
                    event_type=EventType.GRAPH_INTERRUPT,
                    metadata={"node": node_id, "phase": phase},
                )
            )

        self._execution_state.is_interrupted = True
        self._execution_state.interrupt_node = node_id

        raise GraphInterrupt(
            node_id=node_id,
            state=dict(state),
            checkpoint_id=checkpoint_id,
        )

    async def _emit_event(
        self,
        node_name: str,
        state: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        """Emit an event to the streaming queue."""
        if self._event_queue:
            await self._event_queue.put({
                "type": "node_complete",
                "node": node_name,
                "output": output,
                "state": dict(state),
            })
