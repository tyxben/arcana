"""State-related contracts for agent state management."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from arcana.contracts.context import ContextReport


class ExecutionStatus(str, Enum):
    """Status of agent execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentState(BaseModel):
    """Current state of an agent execution."""

    # Identifiers
    run_id: str
    task_id: str | None = None

    # Execution tracking
    status: ExecutionStatus = ExecutionStatus.PENDING
    current_step: int = 0
    max_steps: int = 100

    # Goal and progress
    goal: str | None = None
    current_plan: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)

    # Working memory (key-value store)
    working_memory: dict[str, Any] = Field(default_factory=dict)

    # Conversation history
    messages: list[dict[str, Any]] = Field(default_factory=list)

    # Budget tracking
    tokens_used: int = 0
    cost_usd: float = 0.0
    start_time: datetime | None = None
    elapsed_ms: int = 0

    # Context report (last turn)
    last_context_report: ContextReport | None = None

    # Error tracking
    last_error: str | None = None
    consecutive_errors: int = 0
    consecutive_no_progress: int = 0

    def with_step(self, step: int) -> AgentState:
        """Return a new state with updated step counter."""
        return self.model_copy(update={"current_step": step})

    def increment_step(self) -> AgentState:
        """Return a new state with step counter incremented by 1."""
        return self.with_step(self.current_step + 1)

    @property
    def steps_remaining(self) -> int:
        """Get remaining steps before max_steps is reached."""
        return max(0, self.max_steps - self.current_step)

    @property
    def has_reached_max_steps(self) -> bool:
        """Check if max steps has been reached."""
        return self.current_step >= self.max_steps


class StateSnapshot(BaseModel):
    """A snapshot of agent state for checkpointing."""

    run_id: str
    step_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # State hash for integrity verification
    state_hash: str

    # Full state data
    state: AgentState

    # Checkpoint metadata
    checkpoint_reason: str = ""  # "interval", "error", "plan_step", "verification", "budget"
    plan_progress: dict[str, Any] = Field(default_factory=dict)  # Plan state at checkpoint time
    is_resumable: bool = True

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
