"""Graph-related contracts and data models for the graph execution engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Type of graph node."""

    FUNCTION = "function"
    LLM = "llm"
    TOOL = "tool"
    SUBGRAPH = "subgraph"


class GraphNodeSpec(BaseModel):
    """Specification for a node in the graph."""

    id: str
    name: str
    node_type: NodeType = NodeType.FUNCTION
    metadata: dict[str, Any] = Field(default_factory=dict)


class EdgeType(str, Enum):
    """Type of graph edge."""

    DIRECT = "direct"
    CONDITIONAL = "conditional"


class GraphEdgeSpec(BaseModel):
    """Specification for an edge in the graph."""

    source: str
    target: str  # "__end__" for termination
    edge_type: EdgeType = EdgeType.DIRECT


class ConditionalEdgeSpec(BaseModel):
    """Specification for a conditional edge with a path function."""

    source: str
    path_map: dict[str, str] = Field(default_factory=dict)
    # path_fn stored separately (not serializable)


class GraphConfig(BaseModel):
    """Configuration for a compiled graph."""

    name: str = "default"
    entry_point: str = ""
    interrupt_before: list[str] = Field(default_factory=list)
    interrupt_after: list[str] = Field(default_factory=list)


class GraphExecutionState(BaseModel):
    """Internal state tracking during graph execution."""

    current_node: str | None = None
    visited_nodes: list[str] = Field(default_factory=list)
    node_outputs: dict[str, Any] = Field(default_factory=dict)
    is_interrupted: bool = False
    interrupt_node: str | None = None
