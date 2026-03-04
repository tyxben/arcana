"""Evaluation-related contracts for eval harness."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OutcomeCriterion(str, Enum):
    """Criterion for evaluating agent outcome."""

    STATUS = "status"
    STOP_REASON = "stop_reason"
    MAX_STEPS = "max_steps"
    MAX_COST = "max_cost_usd"
    CONTAINS_KEYS = "contains_keys"
    KEY_VALUES = "key_values"


class EvalCase(BaseModel, frozen=True):
    """A single evaluation case definition."""

    id: str
    goal: str
    expected_outcome: OutcomeCriterion
    expected_value: Any = None
    max_steps: int = 50
    timeout_ms: int = 30_000
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Result of running a single evaluation case."""

    case_id: str
    passed: bool
    actual_status: str
    actual_stop_reason: str | None = None
    steps: int
    tokens_used: int
    cost_usd: float
    duration_ms: int
    error: str | None = None


class EvalReport(BaseModel):
    """Aggregated report for an evaluation suite."""

    suite_name: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[EvalResult] = Field(default_factory=list)
    aggregate_tokens: int = 0
    aggregate_cost_usd: float = 0.0
    aggregate_duration_ms: int = 0


class GateConfig(BaseModel, frozen=True):
    """Configuration for regression gate checks."""

    min_pass_rate: float = 0.9
    max_regression_pct: float = 0.05
    max_avg_cost_usd: float | None = None
    max_avg_tokens: int | None = None


class RegressionResult(BaseModel):
    """Result of a regression gate check."""

    passed: bool
    current_pass_rate: float
    baseline_pass_rate: float | None = None
    regression_pct: float | None = None
    gate_violations: list[str] = Field(default_factory=list)
