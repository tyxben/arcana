"""
Arcana Graph Engine — advanced orchestration for complex control flows.

This is a **platform capability**, not the Runtime default execution path.

  Default path: ``Runtime.run()`` → ConversationAgent (LLM-native, no graph)
  Advanced path: Graph Engine → explicit nodes, edges, reducers, interrupt/resume

Use Graph when you need:
  - Explicit multi-step control flow (branch, loop, replan)
  - Human-in-the-loop (interrupt / resume)
  - Deterministic node ordering
  - Custom state reducers

For most tasks, ``runtime.run(goal)`` is the right entry point.
See ``examples/12_graph_orchestration.py`` for Graph usage.
"""

# ── User-facing API ─────────────────────────────────────────────
from arcana.graph.checkpointer import GraphCheckpointer
from arcana.graph.compiled_graph import CompiledGraph
from arcana.graph.constants import END, START

# ── Internals (re-exported for advanced use) ────────────────────
from arcana.graph.executor import GraphExecutionError, GraphExecutor
from arcana.graph.interrupt import Command, GraphInterrupt
from arcana.graph.node_runner import NodeRunner

# ── Reducers ────────────────────────────────────────────────────
from arcana.graph.reducers import (
    add_messages,
    add_reducer,
    append_reducer,
    apply_reducers,
    extract_reducers,
    merge_reducer,
    replace_reducer,
)
from arcana.graph.state_graph import GraphValidationError, StateGraph

__all__ = [
    # Builder
    "StateGraph",
    "CompiledGraph",
    # Constants
    "START",
    "END",
    # Reducers
    "replace_reducer",
    "append_reducer",
    "add_reducer",
    "merge_reducer",
    "add_messages",
    "extract_reducers",
    "apply_reducers",
    # Checkpoint & Interrupt
    "GraphCheckpointer",
    "GraphInterrupt",
    "Command",
    # Internals
    "GraphExecutor",
    "NodeRunner",
    # Errors
    "GraphValidationError",
    "GraphExecutionError",
]
