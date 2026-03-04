"""SubgraphNode - executes a nested CompiledGraph as a graph node."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arcana.graph.compiled_graph import CompiledGraph


class SubgraphNode:
    """
    Graph node that executes a nested CompiledGraph.

    The nested graph receives the parent state and its output
    is merged back into the parent state.
    """

    def __init__(
        self,
        graph: CompiledGraph,
        *,
        input_map: dict[str, str] | None = None,
        output_map: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            graph: The nested CompiledGraph to execute
            input_map: Map parent state keys to subgraph input keys
            output_map: Map subgraph output keys to parent state keys
        """
        self._graph = graph
        self._input_map = input_map or {}
        self._output_map = output_map or {}

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the nested graph."""
        # Map parent state to subgraph input
        if self._input_map:
            subgraph_input = {
                sub_key: state[parent_key]
                for parent_key, sub_key in self._input_map.items()
                if parent_key in state
            }
        else:
            subgraph_input = dict(state)

        # Execute nested graph
        result = await self._graph.ainvoke(subgraph_input)

        # Map subgraph output to parent state
        if self._output_map:
            return {
                parent_key: result[sub_key]
                for sub_key, parent_key in self._output_map.items()
                if sub_key in result
            }
        return result
