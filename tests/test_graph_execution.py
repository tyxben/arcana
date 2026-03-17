"""Comprehensive tests for the graph execution engine."""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from arcana.graph.constants import END, START
from arcana.graph.reducers import (
    add_messages,
    add_reducer,
    append_reducer,
    apply_reducers,
    extract_reducers,
)
from arcana.graph.state_graph import GraphValidationError, StateGraph

# ---------------------------------------------------------------------------
# Shared state models
# ---------------------------------------------------------------------------


class SimpleState(BaseModel):
    result: str = ""


class ReducerState(BaseModel):
    messages: Annotated[list, add_messages] = []
    counter: Annotated[int, add_reducer] = 0
    items: Annotated[list, append_reducer] = []
    result: str = ""


# ===========================================================================
# Test: Linear Graph Execution
# ===========================================================================


class TestLinearGraphExecution:
    """A -> B -> END with final state verification."""

    async def test_linear_two_nodes(self):
        def node_a(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": state.get("result", "") + "A"}

        def node_b(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": state["result"] + "B"}

        graph = StateGraph(state_schema=SimpleState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        app = graph.compile()
        result = await app.ainvoke({})

        assert result["result"] == "AB"

    async def test_linear_with_initial_state(self):
        def node_a(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": state["result"] + "-processed"}

        graph = StateGraph(state_schema=SimpleState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)

        app = graph.compile()
        result = await app.ainvoke({"result": "init"})

        assert result["result"] == "init-processed"

    async def test_linear_three_nodes(self):
        def step1(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "step1"}

        def step2(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": state["result"] + "->step2"}

        def step3(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": state["result"] + "->step3"}

        graph = StateGraph(state_schema=SimpleState)
        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_node("step3", step3)
        graph.add_edge(START, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.add_edge("step3", END)

        app = graph.compile()
        result = await app.ainvoke({})

        assert result["result"] == "step1->step2->step3"


# ===========================================================================
# Test: Conditional Routing
# ===========================================================================


class TestConditionalRouting:
    """Agent decides between two paths based on state."""

    async def test_conditional_with_path_map(self):
        def classifier(state: dict[str, Any]) -> dict[str, Any]:
            val = state.get("input", "")
            return {"classification": "positive" if "good" in val else "negative"}

        def positive_handler(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "handled_positive"}

        def negative_handler(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "handled_negative"}

        def route_fn(state: dict[str, Any]) -> str:
            return state["classification"]

        graph = StateGraph(state_schema=None)
        graph.add_node("classifier", classifier)
        graph.add_node("positive", positive_handler)
        graph.add_node("negative", negative_handler)
        graph.add_edge(START, "classifier")
        graph.add_conditional_edges(
            "classifier",
            route_fn,
            {"positive": "positive", "negative": "negative"},
        )
        graph.add_edge("positive", END)
        graph.add_edge("negative", END)

        app = graph.compile()

        # Positive path
        res = await app.ainvoke({"input": "this is good"})
        assert res["result"] == "handled_positive"

        # Negative path
        res = await app.ainvoke({"input": "this is bad"})
        assert res["result"] == "handled_negative"

    async def test_conditional_without_path_map(self):
        """path_fn returns node name directly."""

        def router_node(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        def target_a(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "went_a"}

        def target_b(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "went_b"}

        def route_fn(state: dict[str, Any]) -> str:
            return state.get("goto", "target_a")

        graph = StateGraph(state_schema=None)
        graph.add_node("router", router_node)
        graph.add_node("target_a", target_a)
        graph.add_node("target_b", target_b)
        graph.add_edge(START, "router")
        graph.add_conditional_edges("router", route_fn)
        graph.add_edge("target_a", END)
        graph.add_edge("target_b", END)

        app = graph.compile()

        res = await app.ainvoke({"goto": "target_b"})
        assert res["result"] == "went_b"

    async def test_conditional_routing_to_end(self):
        """Conditional edge can route directly to END."""

        def maybe_done(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "done_early"}

        def route_fn(state: dict[str, Any]) -> str:
            return END

        graph = StateGraph(state_schema=None)
        graph.add_node("check", maybe_done)
        graph.add_edge(START, "check")
        graph.add_conditional_edges("check", route_fn, {"__end__": END})

        app = graph.compile()
        res = await app.ainvoke({})
        assert res["result"] == "done_early"


# ===========================================================================
# Test: Cyclic Graph (ReAct pattern)
# ===========================================================================


class TestCyclicGraph:
    """Agent -> Tools -> Agent loop with break condition."""

    async def test_react_loop_with_max_iterations(self):
        """Simulate ReAct: agent calls tools until done."""

        def agent(state: dict[str, Any]) -> dict[str, Any]:
            iteration = state.get("iteration", 0) + 1
            if iteration >= 3:
                return {"iteration": iteration, "should_continue": False, "result": "final"}
            return {"iteration": iteration, "should_continue": True}

        def tools(state: dict[str, Any]) -> dict[str, Any]:
            return {"tool_output": f"result_{state['iteration']}"}

        def should_continue(state: dict[str, Any]) -> str:
            if state.get("should_continue", False):
                return "continue"
            return "end"

        graph = StateGraph(state_schema=None)
        graph.add_node("agent", agent)
        graph.add_node("tools", tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", should_continue, {"continue": "tools", "end": END}
        )
        graph.add_edge("tools", "agent")

        app = graph.compile()
        res = await app.ainvoke({"iteration": 0})

        assert res["iteration"] == 3
        assert res["should_continue"] is False
        assert res["result"] == "final"

    async def test_react_loop_single_iteration(self):
        """Loop exits immediately when condition met on first pass."""

        def agent(state: dict[str, Any]) -> dict[str, Any]:
            return {"done": True, "result": "immediate"}

        def tools(state: dict[str, Any]) -> dict[str, Any]:
            return {"tool_result": "should_not_reach"}

        def should_continue(state: dict[str, Any]) -> str:
            return END if state.get("done") else "tools"

        graph = StateGraph(state_schema=None)
        graph.add_node("agent", agent)
        graph.add_node("tools", tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue)
        graph.add_edge("tools", "agent")

        app = graph.compile()
        res = await app.ainvoke({})

        assert res["result"] == "immediate"
        assert "tool_result" not in res

    async def test_react_accumulates_state_with_reducers(self):
        """Cyclic execution with reducer-based state accumulation."""

        def agent(state: dict[str, Any]) -> dict[str, Any]:
            counter = state.get("counter", 0)
            if counter >= 2:
                return {"result": "done", "counter": 0}
            return {"counter": 1}

        def tools(state: dict[str, Any]) -> dict[str, Any]:
            return {"items": [f"item_{state.get('counter', 0)}"], "counter": 0}

        def should_continue(state: dict[str, Any]) -> str:
            return END if state.get("result") == "done" else "tools"

        graph = StateGraph(state_schema=ReducerState)
        graph.add_node("agent", agent)
        graph.add_node("tools", tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue)
        graph.add_edge("tools", "agent")

        app = graph.compile()
        res = await app.ainvoke({})

        assert res["result"] == "done"
        assert len(res["items"]) > 0


# ===========================================================================
# Test: Reducer Merging
# ===========================================================================


class TestReducerMerging:
    """Test Annotated-type reducers: add_messages, add_reducer, append_reducer."""

    def test_extract_reducers_from_schema(self):
        reducers = extract_reducers(ReducerState)

        assert "messages" in reducers
        assert "counter" in reducers
        assert "items" in reducers
        # 'result' has no reducer annotation
        assert "result" not in reducers

    def test_extract_reducers_none_schema(self):
        assert extract_reducers(None) == {}

    def test_apply_reducers_add_messages(self):
        reducers = extract_reducers(ReducerState)
        state: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}
        output: dict[str, Any] = {"messages": [{"role": "assistant", "content": "hello"}]}

        new_state = apply_reducers(state, output, reducers)

        assert len(new_state["messages"]) == 2
        assert new_state["messages"][0]["content"] == "hi"
        assert new_state["messages"][1]["content"] == "hello"

    def test_apply_reducers_add_messages_dedup_by_id(self):
        reducers = extract_reducers(ReducerState)
        state: dict[str, Any] = {
            "messages": [{"id": "1", "content": "original"}],
        }
        output: dict[str, Any] = {
            "messages": [{"id": "1", "content": "updated"}],
        }

        new_state = apply_reducers(state, output, reducers)

        assert len(new_state["messages"]) == 1
        assert new_state["messages"][0]["content"] == "updated"

    def test_apply_reducers_add_counter(self):
        reducers = extract_reducers(ReducerState)
        state: dict[str, Any] = {"counter": 5}
        output: dict[str, Any] = {"counter": 3}

        new_state = apply_reducers(state, output, reducers)
        assert new_state["counter"] == 8

    def test_apply_reducers_append_items(self):
        reducers = extract_reducers(ReducerState)
        state: dict[str, Any] = {"items": ["a", "b"]}
        output: dict[str, Any] = {"items": ["c"]}

        new_state = apply_reducers(state, output, reducers)
        assert new_state["items"] == ["a", "b", "c"]

    def test_apply_reducers_append_single_item(self):
        reducers = extract_reducers(ReducerState)
        state: dict[str, Any] = {"items": ["a"]}
        output: dict[str, Any] = {"items": "b"}

        new_state = apply_reducers(state, output, reducers)
        assert new_state["items"] == ["a", "b"]

    def test_apply_reducers_replace_for_non_annotated(self):
        reducers = extract_reducers(ReducerState)
        state: dict[str, Any] = {"result": "old"}
        output: dict[str, Any] = {"result": "new"}

        new_state = apply_reducers(state, output, reducers)
        assert new_state["result"] == "new"

    async def test_reducers_during_execution(self):
        """Reducers applied correctly across multiple nodes."""

        def node_a(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "messages": [{"role": "user", "content": "hello"}],
                "counter": 1,
                "items": ["first"],
            }

        def node_b(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "messages": [{"role": "assistant", "content": "hi"}],
                "counter": 2,
                "items": ["second"],
                "result": "done",
            }

        graph = StateGraph(state_schema=ReducerState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        app = graph.compile()
        result = await app.ainvoke({})

        # add_messages: appended both
        assert len(result["messages"]) == 2
        assert result["messages"][0]["content"] == "hello"
        assert result["messages"][1]["content"] == "hi"

        # add_reducer: 0 + 1 + 2 = 3
        assert result["counter"] == 3

        # append_reducer: [] + ["first"] + ["second"]
        assert result["items"] == ["first", "second"]

        # replace: last write wins
        assert result["result"] == "done"


# ===========================================================================
# Test: Error Handling
# ===========================================================================


class TestErrorHandling:
    """Nodes that raise exceptions propagate correctly."""

    async def test_node_raises_exception(self):
        def failing_node(state: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("something went wrong")

        graph = StateGraph(state_schema=None)
        graph.add_node("fail", failing_node)
        graph.add_edge(START, "fail")
        graph.add_edge("fail", END)

        app = graph.compile()

        with pytest.raises(ValueError, match="something went wrong"):
            await app.ainvoke({})

    async def test_error_in_second_node(self):
        def node_a(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "a_done"}

        def node_b(state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("node_b failed")

        graph = StateGraph(state_schema=None)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        app = graph.compile()

        with pytest.raises(RuntimeError, match="node_b failed"):
            await app.ainvoke({})

    async def test_node_returns_non_dict_raises_type_error(self):
        def bad_node(state: dict[str, Any]) -> dict[str, Any]:
            return "not a dict"  # type: ignore[return-value]

        graph = StateGraph(state_schema=None)
        graph.add_node("bad", bad_node)
        graph.add_edge(START, "bad")
        graph.add_edge("bad", END)

        app = graph.compile()

        with pytest.raises(TypeError, match="must return a dict or None"):
            await app.ainvoke({})


# ===========================================================================
# Test: Node Returning None
# ===========================================================================


class TestNodeReturningNone:
    """Node returning None should be treated as empty dict."""

    async def test_none_return_preserves_state(self):
        def noop_node(state: dict[str, Any]) -> None:
            return None

        def final_node(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "final"}

        graph = StateGraph(state_schema=None)
        graph.add_node("noop", noop_node)
        graph.add_node("final", final_node)
        graph.add_edge(START, "noop")
        graph.add_edge("noop", "final")
        graph.add_edge("final", END)

        app = graph.compile()
        res = await app.ainvoke({"existing": "data"})

        assert res["existing"] == "data"
        assert res["result"] == "final"

    async def test_none_return_does_not_alter_reducers(self):
        def noop(state: dict[str, Any]) -> None:
            return None

        def adder(state: dict[str, Any]) -> dict[str, Any]:
            return {"counter": 5}

        graph = StateGraph(state_schema=ReducerState)
        graph.add_node("noop", noop)
        graph.add_node("adder", adder)
        graph.add_edge(START, "noop")
        graph.add_edge("noop", "adder")
        graph.add_edge("adder", END)

        app = graph.compile()
        res = await app.ainvoke({})

        assert res["counter"] == 5


# ===========================================================================
# Test: Sync and Async Node Functions
# ===========================================================================


class TestSyncAndAsyncNodes:
    """Both sync and async node functions are supported."""

    async def test_sync_node(self):
        def sync_fn(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "sync"}

        graph = StateGraph(state_schema=SimpleState)
        graph.add_node("sync", sync_fn)
        graph.add_edge(START, "sync")
        graph.add_edge("sync", END)

        app = graph.compile()
        res = await app.ainvoke({})
        assert res["result"] == "sync"

    async def test_async_node(self):
        async def async_fn(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "async"}

        graph = StateGraph(state_schema=SimpleState)
        graph.add_node("async_node", async_fn)
        graph.add_edge(START, "async_node")
        graph.add_edge("async_node", END)

        app = graph.compile()
        res = await app.ainvoke({})
        assert res["result"] == "async"

    async def test_mixed_sync_and_async_nodes(self):
        def sync_step(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": "sync"}

        async def async_step(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": state["result"] + "+async"}

        graph = StateGraph(state_schema=SimpleState)
        graph.add_node("sync_step", sync_step)
        graph.add_node("async_step", async_step)
        graph.add_edge(START, "sync_step")
        graph.add_edge("sync_step", "async_step")
        graph.add_edge("async_step", END)

        app = graph.compile()
        res = await app.ainvoke({})
        assert res["result"] == "sync+async"


# ===========================================================================
# Test: Graph Validation Errors
# ===========================================================================


class TestGraphValidation:
    """Validation catches structural errors at compile time."""

    def test_no_entry_point(self):
        graph = StateGraph()
        graph.add_node("a", lambda s: {})

        with pytest.raises(GraphValidationError, match="No entry point"):
            graph.compile()

    def test_duplicate_node(self):
        graph = StateGraph()
        graph.add_node("a", lambda s: {})

        with pytest.raises(GraphValidationError, match="already exists"):
            graph.add_node("a", lambda s: {})

    def test_reserved_node_name_start(self):
        graph = StateGraph()

        with pytest.raises(GraphValidationError, match="reserved name"):
            graph.add_node(START, lambda s: {})

    def test_reserved_node_name_end(self):
        graph = StateGraph()

        with pytest.raises(GraphValidationError, match="reserved name"):
            graph.add_node(END, lambda s: {})

    def test_invalid_entry_point(self):
        graph = StateGraph()
        graph.add_node("a", lambda s: {})
        graph.set_entry_point("nonexistent")

        with pytest.raises(GraphValidationError, match="not a defined node"):
            graph.compile()

    def test_no_path_to_end(self):
        graph = StateGraph()
        graph.add_node("a", lambda s: {})
        graph.add_node("b", lambda s: {})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        # b has no outgoing edge to END

        with pytest.raises(GraphValidationError, match="No path from entry point to END"):
            graph.compile()

    def test_set_entry_point_and_finish_point(self):
        """Alternative API using set_entry_point / set_finish_point."""

        graph = StateGraph(state_schema=SimpleState)
        graph.add_node("only", lambda s: {"result": "ok"})
        graph.set_entry_point("only")
        graph.set_finish_point("only")

        graph.compile()  # Should compile without error

    async def test_set_entry_finish_executes(self):
        graph = StateGraph(state_schema=SimpleState)
        graph.add_node("only", lambda s: {"result": "ok"})
        graph.set_entry_point("only")
        graph.set_finish_point("only")

        app = graph.compile()
        res = await app.ainvoke({})
        assert res["result"] == "ok"
