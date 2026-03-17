"""Tests for graph streaming modes (values, updates, messages)."""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from arcana.graph import END, START, StateGraph, add_messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ChatState(BaseModel):
    messages: Annotated[list[dict[str, str]], add_messages] = Field(default_factory=list)
    counter: int = 0


async def greet_node(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [{"role": "assistant", "content": "hello"}],
        "counter": state.get("counter", 0) + 1,
    }


async def farewell_node(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [{"role": "assistant", "content": "goodbye"}],
        "counter": state.get("counter", 0) + 1,
    }


def _build_two_node_graph() -> StateGraph:
    """Build a simple greet -> farewell graph."""
    graph = StateGraph(state_schema=ChatState)
    graph.add_node("greet", greet_node)
    graph.add_node("farewell", farewell_node)
    graph.add_edge(START, "greet")
    graph.add_edge("greet", "farewell")
    graph.add_edge("farewell", END)
    return graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStreamValues:
    """Test streaming in 'values' mode -- yields full state after each node."""

    async def test_yields_full_state_per_node(self) -> None:
        app = _build_two_node_graph().compile()
        events: list[dict[str, Any]] = []
        async for event in app.astream(
            {"messages": [{"role": "user", "content": "hi"}]},
            mode="values",
        ):
            events.append(event)

        assert len(events) == 2

        # After greet node: state has user msg + greeting
        first = events[0]
        assert first["counter"] == 1
        assert len(first["messages"]) == 2
        assert first["messages"][-1]["content"] == "hello"

        # After farewell node: state has all 3 msgs
        second = events[1]
        assert second["counter"] == 2
        assert len(second["messages"]) == 3
        assert second["messages"][-1]["content"] == "goodbye"


class TestStreamUpdates:
    """Test streaming in 'updates' mode -- yields node name + output dict."""

    async def test_yields_node_and_output(self) -> None:
        app = _build_two_node_graph().compile()
        events: list[dict[str, Any]] = []
        async for event in app.astream(
            {"messages": [{"role": "user", "content": "hi"}]},
            mode="updates",
        ):
            events.append(event)

        assert len(events) == 2

        assert events[0]["node"] == "greet"
        assert events[0]["output"]["counter"] == 1

        assert events[1]["node"] == "farewell"
        assert events[1]["output"]["counter"] == 2


class TestStreamMessages:
    """Test streaming in 'messages' mode -- yields only new messages per step."""

    async def test_yields_only_new_messages(self) -> None:
        app = _build_two_node_graph().compile()
        events: list[dict[str, Any]] = []
        async for event in app.astream(
            {"messages": [{"role": "user", "content": "hi"}]},
            mode="messages",
        ):
            events.append(event)

        assert len(events) == 2

        # greet added one message
        assert events[0]["node"] == "greet"
        assert len(events[0]["messages"]) == 1
        assert events[0]["messages"][0]["content"] == "hello"

        # farewell added one message
        assert events[1]["node"] == "farewell"
        assert len(events[1]["messages"]) == 1
        assert events[1]["messages"][0]["content"] == "goodbye"


class TestStreamInvalidMode:
    """Test that an invalid stream mode raises ValueError."""

    async def test_invalid_mode_raises(self) -> None:
        app = _build_two_node_graph().compile()
        with pytest.raises(ValueError, match="Invalid stream mode"):
            async for _ in app.astream({"messages": []}, mode="bad"):
                pass
