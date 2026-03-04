"""Tests for graph checkpointing, interrupt, and resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from arcana.graph import START, END, StateGraph, GraphInterrupt, Command, GraphCheckpointer
from arcana.graph.compiled_graph import CompiledGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def step_a(state: dict[str, Any]) -> dict[str, Any]:
    return {"value": state.get("value", "") + "A"}


async def step_b(state: dict[str, Any]) -> dict[str, Any]:
    resume_val = state.pop("__resume_value__", None)
    suffix = str(resume_val) if resume_val is not None else "B"
    return {"value": state.get("value", "") + suffix}


async def step_c(state: dict[str, Any]) -> dict[str, Any]:
    return {"value": state.get("value", "") + "C"}


def _build_abc_graph(
    checkpointer: GraphCheckpointer | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> CompiledGraph:
    """Build a simple A -> B -> C graph."""
    graph = StateGraph()
    graph.add_node("a", step_a)
    graph.add_node("b", step_b)
    graph.add_node("c", step_c)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", END)
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )


# ---------------------------------------------------------------------------
# Checkpointer unit tests
# ---------------------------------------------------------------------------

class TestGraphCheckpointer:
    """Test GraphCheckpointer save/load/delete lifecycle."""

    async def test_save_and_load(self, tmp_path: Path) -> None:
        cp = GraphCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        cid = await cp.save(state={"x": 1}, node_id="node_a")

        loaded = await cp.load(cid)
        assert loaded is not None
        assert loaded["state"] == {"x": 1}
        assert loaded["node_id"] == "node_a"
        assert loaded["checkpoint_id"] == cid

    async def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        cp = GraphCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        assert await cp.load("does-not-exist") is None

    async def test_delete(self, tmp_path: Path) -> None:
        cp = GraphCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        cid = await cp.save(state={"x": 1}, node_id="n")
        assert await cp.delete(cid) is True
        assert await cp.load(cid) is None
        assert await cp.delete(cid) is False


# ---------------------------------------------------------------------------
# Interrupt tests
# ---------------------------------------------------------------------------

class TestInterruptBefore:
    """Test interrupt_before: graph raises GraphInterrupt before node executes."""

    async def test_interrupt_before_node(self, tmp_path: Path) -> None:
        cp = GraphCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        app = _build_abc_graph(checkpointer=cp, interrupt_before=["b"])

        with pytest.raises(GraphInterrupt) as exc_info:
            await app.ainvoke({"value": ""})

        interrupt = exc_info.value
        assert interrupt.node_id == "b"
        # Node "a" ran but "b" did not yet
        assert interrupt.state["value"] == "A"
        assert interrupt.checkpoint_id != ""

    async def test_interrupt_before_without_checkpointer(self) -> None:
        """Interrupt without a checkpointer still raises, checkpoint_id is empty."""
        app = _build_abc_graph(interrupt_before=["b"])
        with pytest.raises(GraphInterrupt) as exc_info:
            await app.ainvoke({"value": ""})

        assert exc_info.value.checkpoint_id == ""


class TestInterruptAfter:
    """Test interrupt_after: graph raises GraphInterrupt after node executes."""

    async def test_interrupt_after_node(self, tmp_path: Path) -> None:
        cp = GraphCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        app = _build_abc_graph(checkpointer=cp, interrupt_after=["a"])

        with pytest.raises(GraphInterrupt) as exc_info:
            await app.ainvoke({"value": ""})

        interrupt = exc_info.value
        assert interrupt.node_id == "a"
        # Node "a" already ran
        assert interrupt.state["value"] == "A"


# ---------------------------------------------------------------------------
# Resume tests
# ---------------------------------------------------------------------------

class TestResumeFromCheckpoint:
    """Test resuming graph execution from a saved checkpoint."""

    async def test_resume_with_value(self, tmp_path: Path) -> None:
        """Resume from interrupt_before with Command(resume=value)."""
        cp = GraphCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        app = _build_abc_graph(checkpointer=cp, interrupt_before=["b"])

        with pytest.raises(GraphInterrupt) as exc_info:
            await app.ainvoke({"value": ""})

        checkpoint_id = exc_info.value.checkpoint_id
        result = await app.aresume(checkpoint_id, Command(resume="X"))

        # step_b uses resume value as suffix, then step_c appends "C"
        assert result["value"] == "AXC"

    async def test_resume_with_goto(self, tmp_path: Path) -> None:
        """Resume with Command(goto=node) to jump to a different node."""
        cp = GraphCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        app = _build_abc_graph(checkpointer=cp, interrupt_before=["b"])

        with pytest.raises(GraphInterrupt) as exc_info:
            await app.ainvoke({"value": ""})

        checkpoint_id = exc_info.value.checkpoint_id
        # Skip "b" entirely, jump to "c"
        result = await app.aresume(checkpoint_id, Command(goto="c"))

        assert result["value"] == "AC"

    async def test_checkpoint_not_found_raises(self, tmp_path: Path) -> None:
        """Resuming with a nonexistent checkpoint_id raises ValueError."""
        cp = GraphCheckpointer(checkpoint_dir=tmp_path / "ckpt")
        app = _build_abc_graph(checkpointer=cp)

        with pytest.raises(ValueError, match="not found"):
            await app.aresume("nonexistent-id")

    async def test_resume_without_checkpointer_raises(self) -> None:
        """Resuming without a checkpointer raises RuntimeError."""
        app = _build_abc_graph()
        with pytest.raises(RuntimeError, match="Cannot resume without a checkpointer"):
            await app.aresume("any-id")
