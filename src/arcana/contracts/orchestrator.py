"""Orchestrator contracts for task scheduling and coordination."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Status of a task in the orchestrator."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RetryPolicy(BaseModel):
    """Configuration for task retry behavior."""

    max_retries: int = 0
    delay_ms: int = 1000
    backoff_multiplier: float = 2.0
    max_delay_ms: int = 30000


class TaskBudget(BaseModel):
    """Budget constraints for a single task."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_time_ms: int | None = None


class Task(BaseModel):
    """A unit of work in the orchestrator DAG."""

    id: str
    goal: str
    dependencies: list[str] = Field(default_factory=list)
    priority: int = 0  # higher = more urgent
    budget: TaskBudget | None = None
    deadline: datetime | None = None
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    attempt: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Result of a completed task execution."""

    task_id: str
    status: TaskStatus
    attempt: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str | None = None
    state_summary: dict[str, Any] = Field(default_factory=dict)


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""

    max_concurrent_tasks: int = 4
    global_max_tokens: int | None = None
    global_max_cost_usd: float | None = None
    global_max_time_ms: int | None = None
    default_retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    scheduling_interval_ms: int = 100
