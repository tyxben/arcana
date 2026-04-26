"""Trace-related contracts for event logging and auditing."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class StopReason(str, Enum):
    """Reasons for stopping agent execution."""

    GOAL_REACHED = "goal_reached"
    MAX_STEPS = "max_steps"
    MAX_TIME = "max_time"
    MAX_COST = "max_cost"
    MAX_TOKENS = "max_tokens"
    NO_PROGRESS = "no_progress"
    ERROR = "error"
    USER_CANCELLED = "user_cancelled"
    TOOL_BLOCKED = "tool_blocked"


class AgentRole(str, Enum):
    """Role of an agent in the system."""

    SYSTEM = "system"
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"


class EventType(str, Enum):
    """Type of trace event."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    PLAN = "plan"
    VERIFY = "verify"
    MEMORY_WRITE = "memory_write"

    # Context management
    CONTEXT_DECISION = "context_decision"
    PROMPT_SNAPSHOT = "prompt_snapshot"

    # Cognitive primitives (v0.7.0) — recall / pin / unpin invocations
    COGNITIVE_PRIMITIVE = "cognitive_primitive"

    # Session lifecycle — history seeded by user code (cold-start
    # restore from external storage). Emitted by ChatSession.seed_history.
    HISTORY_SEEDED = "history_seeded"

    # Orchestrator events
    TASK_SUBMIT = "task_submit"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAIL = "task_fail"

    # Turn-level events
    TURN = "turn"
    AGENT_TURN = "agent_turn"

    # Graph engine events
    GRAPH_NODE_START = "graph_node_start"
    GRAPH_NODE_COMPLETE = "graph_node_complete"
    GRAPH_TRANSITION = "graph_transition"
    GRAPH_INTERRUPT = "graph_interrupt"


class BudgetSnapshot(BaseModel):
    """Snapshot of budget consumption at a point in time."""

    # Limits
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_time_ms: int | None = None
    max_iterations: int | None = None

    # Consumed
    tokens_used: int = 0
    cost_usd: float = 0.0
    time_ms: int = 0
    iterations_used: int = 0

    @property
    def tokens_remaining(self) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def iterations_remaining(self) -> int | None:
        if self.max_iterations is None:
            return None
        return max(0, self.max_iterations - self.iterations_used)

    @property
    def budget_exhausted(self) -> bool:
        """Check if any budget limit has been reached."""
        if self.max_tokens and self.tokens_used >= self.max_tokens:
            return True
        if self.max_cost_usd and self.cost_usd >= self.max_cost_usd:
            return True
        if self.max_time_ms and self.time_ms >= self.max_time_ms:
            return True
        if self.max_iterations and self.iterations_used >= self.max_iterations:
            return True
        return False


class PromptSnapshot(BaseModel):
    """Full capture of a single LLM prompt for offline replay.

    Stored inside ``TraceEvent.metadata["prompt_snapshot"]`` when
    ``RuntimeConfig.trace_include_prompt_snapshots`` is enabled. Disabled
    by default: snapshots can contain PII / secrets and bloat trace size.

    The schema mirrors what was sent to ``gateway.generate()`` at the
    time of the call — messages, tool schemas, model, budget at that
    moment. It is retrospective evidence, not an input channel.
    """

    turn: int
    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    response_format: dict[str, Any] | None = None
    budget_snapshot: BudgetSnapshot | None = None


class ToolCallRecord(BaseModel):
    """Record of a tool call for tracing."""

    name: str
    args_digest: str  # Canonical hash of arguments
    idempotency_key: str | None = None
    result_digest: str | None = None
    error: str | None = None
    duration_ms: int | None = None
    side_effect: str | None = None  # "read" or "write"


class TraceContext(BaseModel):
    """Context for trace operations."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str | None = None
    parent_step_id: str | None = None

    def new_step_id(self) -> str:
        """Generate a new step ID."""
        return str(uuid4())


class TraceEvent(BaseModel):
    """A single trace event for auditing and debugging."""

    # Identifiers
    run_id: str
    task_id: str | None = None
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    # Causal parent — links an event to the step that caused it.
    # Siblings of a single LLM turn (CONTEXT_DECISION, PROMPT_SNAPSHOT, TURN)
    # share the same parent = the TURN event's step_id. Tool calls emitted
    # from a turn point to that turn's step_id. Absent on root events
    # (e.g. TASK_START) and on legacy traces written before this field.
    parent_step_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Classification
    role: AgentRole = AgentRole.SYSTEM
    event_type: EventType

    # State digests (Canonical JSON SHA-256 truncated to 16 chars)
    state_before_hash: str | None = None
    state_after_hash: str | None = None

    # LLM-related
    llm_request_digest: str | None = None
    llm_response_digest: str | None = None
    model: str | None = None
    llm_latency_ms: int | None = None

    # Tool-related
    tool_call: ToolCallRecord | None = None

    # Budget tracking
    budgets: BudgetSnapshot | None = None

    # Stop information
    stop_reason: StopReason | None = None
    stop_detail: str | None = None

    # Additional context
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
