"""CompiledGraph - executable graph with invoke/stream/resume capabilities."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from arcana.contracts.graph import (
    ConditionalEdgeSpec,
    GraphConfig,
    GraphEdgeSpec,
    GraphNodeSpec,
)
from arcana.graph.interrupt import Command

if TYPE_CHECKING:
    from collections.abc import Callable


class CompiledGraph:
    """
    An executable graph compiled from a StateGraph.

    Provides ainvoke (single run), astream (streaming), and aresume (resume from interrupt).
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
    ) -> None:
        self._config = config
        self._nodes = nodes
        self._node_fns = node_fns
        self._edges = edges
        self._conditional_edges = conditional_edges
        self._conditional_fns = conditional_fns
        self._state_schema = state_schema
        self._checkpointer = checkpointer

    @property
    def config(self) -> GraphConfig:
        return self._config

    @property
    def nodes(self) -> dict[str, GraphNodeSpec]:
        return dict(self._nodes)

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the graph to completion and return final state."""
        from arcana.graph.executor import GraphExecutor

        executor = GraphExecutor(
            config=self._config,
            nodes=self._nodes,
            node_fns=self._node_fns,
            edges=self._edges,
            conditional_edges=self._conditional_edges,
            conditional_fns=self._conditional_fns,
            state_schema=self._state_schema,
            checkpointer=self._checkpointer,
        )
        return await executor.execute(input, config)

    async def astream(
        self,
        input: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        mode: str = "values",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Execute the graph with streaming output.

        Modes:
        - "values": yield full state after each node
        - "updates": yield {"node": name, "output": {...}} after each node
        - "messages": yield new messages added at each step
        """
        from arcana.graph.streaming import astream

        async for event in astream(
            compiled_graph=self,
            input=input,
            config=config,
            mode=mode,
        ):
            yield event

    async def aresume(
        self,
        checkpoint_id: str,
        command: Command | None = None,
    ) -> dict[str, Any]:
        """Resume execution from a checkpoint after an interrupt."""
        if not self._checkpointer:
            raise RuntimeError("Cannot resume without a checkpointer")

        from arcana.graph.executor import GraphExecutor

        checkpoint_data = await self._checkpointer.load(checkpoint_id)
        if checkpoint_data is None:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")

        executor = GraphExecutor(
            config=self._config,
            nodes=self._nodes,
            node_fns=self._node_fns,
            edges=self._edges,
            conditional_edges=self._conditional_edges,
            conditional_fns=self._conditional_fns,
            state_schema=self._state_schema,
            checkpointer=self._checkpointer,
        )
        return await executor.resume(checkpoint_data, command)
