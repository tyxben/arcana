"""Streaming-related contracts for real-time agent execution events."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StreamEventType(str, Enum):
    """Unified stream event types for both Runtime and Graph execution."""

    # Lifecycle
    RUN_START = "run_start"
    RUN_COMPLETE = "run_complete"
    # Step-level
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    # LLM
    LLM_CHUNK = "llm_chunk"
    LLM_COMPLETE = "llm_complete"
    LLM_THINKING = "llm_thinking"
    # Tool
    TOOL_CALL_START = "tool_call_start"
    TOOL_RESULT = "tool_result"
    # State
    STATE_UPDATE = "state_update"
    CHECKPOINT = "checkpoint"
    # Errors
    ERROR = "error"
    # User interaction
    INPUT_NEEDED = "input_needed"
    # Graph-specific (for compatibility)
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"


class StreamEvent(BaseModel):
    """Unified streaming event for Runtime and Graph execution."""

    event_type: StreamEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_id: str
    step_id: str | None = None

    # Content (depending on event_type)
    content: str | None = None
    thinking: str | None = None
    node_name: str | None = None

    # Structured data
    step_result_data: dict[str, Any] | None = None
    tool_result_data: dict[str, Any] | None = None
    state_delta: dict[str, Any] | None = None

    # Metrics
    tokens_used: int | None = None
    cost_usd: float | None = None
    budget_remaining_pct: float | None = None

    # Error
    error: str | None = None
    error_type: str | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


# Stream mode filter sets
STEP_EVENTS = frozenset({
    StreamEventType.STEP_START,
    StreamEventType.STEP_COMPLETE,
})

LLM_EVENTS = frozenset({
    StreamEventType.LLM_CHUNK,
    StreamEventType.LLM_COMPLETE,
    StreamEventType.LLM_THINKING,
})

TOOL_EVENTS = frozenset({
    StreamEventType.TOOL_CALL_START,
    StreamEventType.TOOL_RESULT,
})

ALL_EVENTS = frozenset(StreamEventType)

STREAM_MODE_FILTERS: dict[str, frozenset[StreamEventType]] = {
    "all": ALL_EVENTS,
    "steps": STEP_EVENTS,
    "llm": LLM_EVENTS,
    "tools": TOOL_EVENTS,
}


def matches_mode(event: StreamEvent, mode: str) -> bool:
    """Check if an event matches the given stream mode filter. Pure function."""
    allowed = STREAM_MODE_FILTERS.get(mode, ALL_EVENTS)
    return event.event_type in allowed
