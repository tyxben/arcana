"""Plan-and-Execute contracts for structured planning."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PlanStepStatus(str, Enum):
    """Status of a plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """A single step in a structured plan."""

    id: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)  # IDs of prerequisite steps
    status: PlanStepStatus = PlanStepStatus.PENDING
    result: str | None = None  # Result summary after execution
    metadata: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """A structured execution plan with steps and acceptance criteria."""

    steps: list[PlanStep] = Field(default_factory=list)
    goal: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)  # Global criteria
    metadata: dict[str, Any] = Field(default_factory=dict)

    def next_step(self) -> PlanStep | None:
        """Get the next executable step (all dependencies met, status pending)."""
        completed_ids = {s.id for s in self.steps if s.status == PlanStepStatus.COMPLETED}
        for step in self.steps:
            if step.status != PlanStepStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in step.dependencies):
                return step
        return None

    def mark_step(self, step_id: str, status: PlanStepStatus, result: str | None = None) -> None:
        """Update a step's status and optional result."""
        for step in self.steps:
            if step.id == step_id:
                step.status = status
                if result is not None:
                    step.result = result
                return

    @property
    def is_complete(self) -> bool:
        """Check if all steps are completed or skipped."""
        return all(
            s.status in (PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED)
            for s in self.steps
        )

    @property
    def has_failed(self) -> bool:
        """Check if any step has failed."""
        return any(s.status == PlanStepStatus.FAILED for s in self.steps)

    @property
    def progress_ratio(self) -> float:
        """Ratio of completed steps to total steps."""
        if not self.steps:
            return 0.0
        done = sum(
            1 for s in self.steps
            if s.status in (PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED)
        )
        return done / len(self.steps)


class VerificationOutcome(str, Enum):
    """Outcome of a verification check."""

    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"


class GoalVerificationResult(BaseModel):
    """Result of verifying goal/acceptance criteria."""

    outcome: VerificationOutcome
    criteria_results: dict[str, bool] = Field(default_factory=dict)  # criterion -> pass/fail
    coverage: float = 0.0  # Ratio of criteria passed
    failed_criteria: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)  # Re-plan hints
    metadata: dict[str, Any] = Field(default_factory=dict)
