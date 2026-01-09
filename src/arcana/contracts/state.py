"""State-related contracts for agent state management."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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

    # Error tracking
    last_error: str | None = None
    consecutive_errors: int = 0
    consecutive_no_progress: int = 0

    def increment_step(self) -> None:
        """Increment the step counter."""
        self.current_step += 1

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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # State hash for integrity verification
    state_hash: str

    # Full state data
    state: AgentState

    # Checkpoint metadata
    checkpoint_reason: str | None = None
    is_resumable: bool = True

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
