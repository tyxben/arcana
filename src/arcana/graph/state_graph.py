"""StateGraph - declarative graph builder for agent workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arcana.graph.compiled_graph import CompiledGraph

from pydantic import BaseModel

from arcana.contracts.graph import (
    ConditionalEdgeSpec,
    EdgeType,
    GraphConfig,
    GraphEdgeSpec,
    GraphNodeSpec,
    NodeType,
)
from arcana.graph.constants import END, START


class GraphValidationError(Exception):
    """Raised when graph validation fails during compile."""


class StateGraph:
    """
    Declarative graph builder for agent workflows.

    Usage::

        graph = StateGraph(state_schema=MyState)
        graph.add_node("search", search_fn)
        graph.add_node("summarize", summarize_fn)
        graph.add_edge(START, "search")
        graph.add_edge("search", "summarize")
        graph.add_edge("summarize", END)
        app = graph.compile()
    """

    def __init__(self, state_schema: type[BaseModel] | None = None) -> None:
        self._state_schema = state_schema
        self._nodes: dict[str, GraphNodeSpec] = {}
        self._node_fns: dict[str, Callable[..., Any]] = {}
        self._edges: list[GraphEdgeSpec] = []
        self._conditional_edges: list[ConditionalEdgeSpec] = []
        self._conditional_fns: dict[str, Callable[..., str]] = {}
        self._entry_point: str | None = None
        self._finish_points: list[str] = []

    def add_node(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        node_type: NodeType = NodeType.FUNCTION,
        metadata: dict[str, Any] | None = None,
    ) -> StateGraph:
        """Add a node to the graph."""
        if name in (START, END):
            raise GraphValidationError(f"Cannot use reserved name '{name}' as node name")
        if name in self._nodes:
            raise GraphValidationError(f"Node '{name}' already exists")

        self._nodes[name] = GraphNodeSpec(
            id=name,
            name=name,
            node_type=node_type,
            metadata=metadata or {},
        )
        self._node_fns[name] = fn
        return self

    def add_edge(self, source: str, target: str) -> StateGraph:
        """Add a direct edge between two nodes."""
        self._edges.append(
            GraphEdgeSpec(source=source, target=target, edge_type=EdgeType.DIRECT)
        )
        return self

    def add_conditional_edges(
        self,
        source: str,
        path_fn: Callable[..., str],
        path_map: dict[str, str] | None = None,
    ) -> StateGraph:
        """
        Add conditional edges from a source node.

        The path_fn receives the current state and returns a string key.
        If path_map is provided, the key is mapped to a target node name.
        Otherwise, the key is used directly as the target node name.
        """
        spec = ConditionalEdgeSpec(
            source=source,
            path_map=path_map or {},
        )
        self._conditional_edges.append(spec)
        self._conditional_fns[source] = path_fn
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        """Set the entry point node (alternative to add_edge(START, name))."""
        self._entry_point = name
        return self

    def set_finish_point(self, name: str) -> StateGraph:
        """Set a finish point node (alternative to add_edge(name, END))."""
        self._finish_points.append(name)
        return self

    def compile(
        self,
        *,
        checkpointer: Any | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        name: str = "default",
    ) -> CompiledGraph:
        """
        Compile the graph into an executable CompiledGraph.

        Performs full validation:
        - Entry point exists
        - All edge references are valid nodes
        - At least one path reaches END
        """
        # Resolve entry point
        entry = self._resolve_entry_point()

        # Add implicit edges for finish points
        for fp in self._finish_points:
            self._edges.append(
                GraphEdgeSpec(source=fp, target=END, edge_type=EdgeType.DIRECT)
            )

        # Validate
        self._validate(entry)

        config = GraphConfig(
            name=name,
            entry_point=entry,
            interrupt_before=interrupt_before or [],
            interrupt_after=interrupt_after or [],
        )

        from arcana.graph.compiled_graph import CompiledGraph

        return CompiledGraph(
            config=config,
            nodes=dict(self._nodes),
            node_fns=dict(self._node_fns),
            edges=list(self._edges),
            conditional_edges=list(self._conditional_edges),
            conditional_fns=dict(self._conditional_fns),
            state_schema=self._state_schema,
            checkpointer=checkpointer,
        )

    def _resolve_entry_point(self) -> str:
        """Resolve the entry point from explicit setting or START edges."""
        if self._entry_point:
            return self._entry_point

        # Find edges from START
        start_edges = [e for e in self._edges if e.source == START]
        if len(start_edges) == 1:
            return start_edges[0].target
        if len(start_edges) > 1:
            raise GraphValidationError(
                "Multiple edges from START found. Use set_entry_point() to specify one."
            )

        raise GraphValidationError(
            "No entry point defined. Use set_entry_point() or add_edge(START, node)."
        )

    def _validate(self, entry: str) -> None:
        """Validate the graph structure."""
        node_names = set(self._nodes.keys())

        # Entry point must be a valid node
        if entry not in node_names:
            raise GraphValidationError(
                f"Entry point '{entry}' is not a defined node. "
                f"Available nodes: {sorted(node_names)}"
            )

        # Validate all edges reference valid nodes
        for edge in self._edges:
            if edge.source not in node_names and edge.source != START:
                raise GraphValidationError(
                    f"Edge source '{edge.source}' is not a defined node"
                )
            if edge.target not in node_names and edge.target != END:
                raise GraphValidationError(
                    f"Edge target '{edge.target}' is not a defined node"
                )

        # Validate conditional edge sources
        for cond in self._conditional_edges:
            if cond.source not in node_names:
                raise GraphValidationError(
                    f"Conditional edge source '{cond.source}' is not a defined node"
                )
            # Validate path_map targets
            for target in cond.path_map.values():
                if target not in node_names and target != END:
                    raise GraphValidationError(
                        f"Conditional edge target '{target}' is not a defined node"
                    )

        # Check that at least one path reaches END
        if not self._can_reach_end(entry, node_names):
            raise GraphValidationError(
                "No path from entry point to END. "
                "Add edges leading to END or use set_finish_point()."
            )

    def _can_reach_end(self, entry: str, node_names: set[str]) -> bool:
        """Check if END is reachable from entry via BFS."""
        visited: set[str] = set()
        queue = [entry]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Direct edges
            for edge in self._edges:
                if edge.source == current:
                    if edge.target == END:
                        return True
                    if edge.target not in visited:
                        queue.append(edge.target)

            # Conditional edges
            for cond in self._conditional_edges:
                if cond.source == current:
                    if cond.path_map:
                        for target in cond.path_map.values():
                            if target == END:
                                return True
                            if target not in visited:
                                queue.append(target)
                    else:
                        # Without path_map, any node could be a target
                        for name in node_names:
                            if name not in visited:
                                queue.append(name)
                        return True  # Could route to END directly

        return False
