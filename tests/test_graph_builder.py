"""Tests for the StateGraph builder and validation."""

import pytest

from arcana.graph.compiled_graph import CompiledGraph
from arcana.graph.constants import END, START
from arcana.graph.state_graph import GraphValidationError, StateGraph

# --- Simple node functions for testing ---


def node_a(state: dict) -> dict:
    return {"result": "a"}


def node_b(state: dict) -> dict:
    return {"result": "b"}


def node_c(state: dict) -> dict:
    return {"result": "c"}


def router_fn(state: dict) -> str:
    return state.get("route", "default")


class TestStateGraphBuilder:
    """Tests for graph construction (add_node, add_edge, set_entry/finish)."""

    def test_linear_graph_compiles(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        compiled = graph.compile()

        assert isinstance(compiled, CompiledGraph)
        assert compiled.config.entry_point == "a"
        assert set(compiled.nodes.keys()) == {"a", "b"}

    def test_single_node_graph(self):
        graph = StateGraph()
        graph.add_node("only", node_a)
        graph.add_edge(START, "only")
        graph.add_edge("only", END)

        compiled = graph.compile()

        assert compiled.config.entry_point == "only"
        assert set(compiled.nodes.keys()) == {"only"}

    def test_set_entry_point(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        compiled = graph.compile()

        assert compiled.config.entry_point == "a"

    def test_set_finish_point(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")

        compiled = graph.compile()

        assert compiled.config.entry_point == "a"
        assert set(compiled.nodes.keys()) == {"a", "b"}

    def test_set_entry_and_finish_point(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.set_finish_point("b")

        compiled = graph.compile()

        assert compiled.config.entry_point == "a"

    def test_conditional_edges(self):
        graph = StateGraph()
        graph.add_node("router", node_a)
        graph.add_node("path_a", node_b)
        graph.add_node("path_b", node_c)
        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            router_fn,
            path_map={"go_a": "path_a", "go_b": "path_b"},
        )
        graph.add_edge("path_a", END)
        graph.add_edge("path_b", END)

        compiled = graph.compile()

        assert compiled.config.entry_point == "router"
        assert set(compiled.nodes.keys()) == {"router", "path_a", "path_b"}

    def test_conditional_edges_with_end_target(self):
        graph = StateGraph()
        graph.add_node("router", node_a)
        graph.add_node("continue", node_b)
        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            router_fn,
            path_map={"done": END, "more": "continue"},
        )
        graph.add_edge("continue", END)

        compiled = graph.compile()

        assert isinstance(compiled, CompiledGraph)

    def test_conditional_edges_without_path_map(self):
        graph = StateGraph()
        graph.add_node("router", node_a)
        graph.add_node("target", node_b)
        graph.add_edge(START, "router")
        graph.add_conditional_edges("router", router_fn)
        graph.add_edge("target", END)

        compiled = graph.compile()

        assert isinstance(compiled, CompiledGraph)

    def test_compile_with_name(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        compiled = graph.compile(name="my_graph")

        assert compiled.config.name == "my_graph"

    def test_compile_with_interrupt_before(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        compiled = graph.compile(interrupt_before=["a"])

        assert compiled.config.interrupt_before == ["a"]

    def test_compile_with_interrupt_after(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        compiled = graph.compile(interrupt_after=["a"])

        assert compiled.config.interrupt_after == ["a"]

    def test_add_node_returns_self(self):
        graph = StateGraph()
        result = graph.add_node("a", node_a)
        assert result is graph

    def test_add_edge_returns_self(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        result = graph.add_edge(START, "a")
        assert result is graph

    def test_add_conditional_edges_returns_self(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        result = graph.add_conditional_edges("a", router_fn)
        assert result is graph

    def test_set_entry_point_returns_self(self):
        graph = StateGraph()
        result = graph.set_entry_point("a")
        assert result is graph

    def test_set_finish_point_returns_self(self):
        graph = StateGraph()
        result = graph.set_finish_point("a")
        assert result is graph

    def test_compiled_graph_nodes_match(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)

        compiled = graph.compile()

        nodes = compiled.nodes
        assert len(nodes) == 3
        assert nodes["a"].name == "a"
        assert nodes["b"].name == "b"
        assert nodes["c"].name == "c"


class TestStateGraphValidation:
    """Tests for graph validation errors during compile."""

    def test_error_missing_entry_point(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge("a", END)

        with pytest.raises(GraphValidationError, match="No entry point defined"):
            graph.compile()

    def test_error_dangling_edge_target(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", "nonexistent")

        with pytest.raises(GraphValidationError, match="not a defined node"):
            graph.compile()

    def test_error_dangling_edge_source(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("ghost", "a")
        graph.add_edge("a", END)

        with pytest.raises(GraphValidationError, match="not a defined node"):
            graph.compile()

    def test_error_unreachable_end(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        # No edge from b to END

        with pytest.raises(GraphValidationError, match="No path from entry point to END"):
            graph.compile()

    def test_error_duplicate_node_name(self):
        graph = StateGraph()
        graph.add_node("a", node_a)

        with pytest.raises(GraphValidationError, match="already exists"):
            graph.add_node("a", node_b)

    def test_error_reserved_name_start(self):
        graph = StateGraph()

        with pytest.raises(GraphValidationError, match="reserved name"):
            graph.add_node(START, node_a)

    def test_error_reserved_name_end(self):
        graph = StateGraph()

        with pytest.raises(GraphValidationError, match="reserved name"):
            graph.add_node(END, node_a)

    def test_error_entry_point_not_a_node(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.set_entry_point("nonexistent")
        graph.add_edge("a", END)

        with pytest.raises(GraphValidationError, match="not a defined node"):
            graph.compile()

    def test_error_multiple_start_edges(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge("a", END)
        graph.add_edge("b", END)

        with pytest.raises(GraphValidationError, match="Multiple edges from START"):
            graph.compile()

    def test_error_conditional_edge_invalid_source(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_conditional_edges(
            "ghost",
            router_fn,
            path_map={"x": "a"},
        )
        graph.add_edge("a", END)

        with pytest.raises(GraphValidationError, match="not a defined node"):
            graph.compile()

    def test_error_conditional_edge_invalid_target(self):
        graph = StateGraph()
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_conditional_edges(
            "a",
            router_fn,
            path_map={"x": "nonexistent"},
        )

        with pytest.raises(GraphValidationError, match="not a defined node"):
            graph.compile()
