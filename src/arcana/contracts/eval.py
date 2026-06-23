"""Evaluation-related contracts for eval harness."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class OutcomeCriterion(str, Enum):
    """Criterion for evaluating agent outcome."""

    STATUS = "status"
    STOP_REASON = "stop_reason"
    MAX_STEPS = "max_steps"
    MAX_COST = "max_cost_usd"
    CONTAINS_KEYS = "contains_keys"
    KEY_VALUES = "key_values"


class TraceSignals(BaseModel):
    """Trace-derived per-case fitness vector (finding F5).

    A multi-dimensional, tamper-evident summary of *how* a run behaved, derived
    post-hoc from its trace events — not just whether it passed. The dimensions
    are deliberately orthogonal (not collapsed into one score an optimizer
    could game): boundary/authority signals stay separate from cost/quality so
    that "passed the eval by weakening a boundary" is detectable.

    Every field is defaulted. ``trace_available`` is explicit so that the
    *absence* of a trace is never silently read as a clean run.
    """

    # --- authority / boundary signals (the anti-Goodhart core) ---
    permission_denials: int = 0
    permission_asks: int = 0
    guardrail_blocks: int = 0  # counts block + redact + require_approval
    guardrail_warns: int = 0
    tool_error_categories: dict[str, int] = Field(default_factory=dict)
    provider_degraded: bool = False
    degraded_capabilities: list[str] = Field(default_factory=list)
    capability_downgrades: int = 0
    write_tool_calls: int = 0
    imported_capability_calls: int = 0  # tool calls carrying remote provenance

    # --- context signals ---
    context_messages_in: int = 0
    context_messages_out: int = 0
    context_messages_dropped: int = 0
    context_compressed_count: int = 0
    context_loss_ratio: float = 0.0  # dropped / max(messages_in, 1)
    context_fidelity_min: str | None = None  # worst fidelity level reached

    # --- denominators ---
    llm_calls: int = 0
    tool_calls: int = 0

    # --- meta ---
    trace_available: bool = False
    signals_digest: str = ""  # canonical hash of the vector (excl. this field)

    def compute_digest(self) -> str:
        """Canonical hash of the vector, excluding ``signals_digest`` itself.

        A proposal that cites a fabricated signal vector is rejectable because
        the digest will not recompute.
        """
        from arcana.utils.hashing import canonical_hash

        return canonical_hash(self.model_dump(exclude={"signals_digest"}))

    def with_digest(self) -> TraceSignals:
        """Return a copy with ``signals_digest`` filled in."""
        return self.model_copy(update={"signals_digest": self.compute_digest()})


class GoldenTrace(BaseModel):
    """A recorded reference run for golden-trace replay (finding F5).

    Stores a *redacted structural skeleton* (no prose, no args, no PII) plus
    the trace signal vector, so a later run can be diffed against it to detect
    behavioural regressions. Committed to the repo so that relaxing a golden is
    a reviewable git diff.
    """

    case_id: str
    suite_name: str = "default"
    recorded_at: str = ""
    arcana_version: str | None = None
    goal_digest: str = ""  # canonical hash of EvalCase.goal
    signals: TraceSignals = Field(default_factory=TraceSignals)
    event_skeleton: list[dict[str, Any]] = Field(default_factory=list)
    skeleton_digest: str = ""


class GoldenDiff(BaseModel):
    """Outcome of diffing a run against its golden trace."""

    case_id: str
    golden_status: str = "skip"  # skip / match / regressed / improved / new
    structural_changes: list[str] = Field(default_factory=list)
    signal_regressions: list[str] = Field(default_factory=list)
    is_regression: bool = False


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
    # F5: trace-derived fitness vector (None = trace not collected / legacy).
    signals: TraceSignals | None = None
    # F5: golden-replay verdict (skip / match / regressed / improved / new).
    golden_status: str = "skip"


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
    # F5: suite-level merged signal vector + any golden regressions.
    aggregate_signals: TraceSignals | None = None
    golden_regressions: list[str] = Field(default_factory=list)


class GateConfig(BaseModel, frozen=True):
    """Configuration for regression gate checks.

    The F5 signal/golden fields all default to "off" (None / False / 'off'),
    so a default ``GateConfig()`` reproduces the legacy pass-rate/cost/tokens
    behaviour exactly. Hard-fail on a boundary signal is always an explicit
    opt-in — the library never silently becomes a release supervisor
    (Principle 7 Corollary: evals are evidence, not governance).
    """

    min_pass_rate: float = 0.9
    max_regression_pct: float = 0.05
    max_avg_cost_usd: float | None = None
    max_avg_tokens: int | None = None
    # --- F5 signal gates (opt-in hard-fail; None/False = warn-or-ignore) ---
    max_permission_denials: int | None = None
    max_guardrail_blocks: int | None = None
    max_context_loss_ratio: float | None = None
    max_write_tool_calls: int | None = None
    max_imported_capability_calls: int | None = None
    max_capability_downgrades: int | None = None
    max_context_fidelity: str | None = None  # worst acceptable level, e.g. "L1"
    forbid_provider_degradation: bool = False
    forbid_new_tool_error_categories: bool = False  # baseline-relative (compare)
    require_trace: bool = False  # a case with no trace signals is a violation
    # --- F5 golden-trace replay ---
    golden_replay: Literal["off", "warn", "strict"] = "off"
    golden_dir: str | None = None


class RegressionResult(BaseModel):
    """Result of a regression gate check."""

    passed: bool
    current_pass_rate: float
    baseline_pass_rate: float | None = None
    regression_pct: float | None = None
    gate_violations: list[str] = Field(default_factory=list)
    # F5: signal/golden hard-fails are ALSO appended to gate_violations so
    # legacy callers reading only that field still see passed=False.
    signal_violations: list[str] = Field(default_factory=list)
    golden_violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
