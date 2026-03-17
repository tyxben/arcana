"""V2 evaluation runner -- result-oriented."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from arcana.eval.judge import EvalJudge, RuleJudge
from arcana.eval.metrics import EvalMetrics

if TYPE_CHECKING:
    pass


class EvalCaseV2(BaseModel):
    """A single evaluation case -- result-oriented."""

    goal: str
    success_criteria: list[str]
    quality_rubric: dict[str, float] | None = None  # dimension -> weight
    max_acceptable_cost: float | None = None
    max_acceptable_steps: int | None = None
    tags: list[str] = Field(default_factory=list)


class EvalVerdict(BaseModel):
    """Judgment for a single eval case."""

    case: EvalCaseV2
    metrics: EvalMetrics
    passed: bool
    reason: str = ""


class EvalSuiteReport(BaseModel):
    """Aggregate report for a suite of eval cases."""

    verdicts: list[EvalVerdict] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of verdicts that passed."""
        if not self.verdicts:
            return 0.0
        return sum(1 for v in self.verdicts if v.passed) / len(self.verdicts)

    @property
    def avg_goal_achievement(self) -> float:
        """Mean goal achievement rate across all verdicts."""
        if not self.verdicts:
            return 0.0
        return sum(v.metrics.goal_achievement_rate for v in self.verdicts) / len(self.verdicts)

    @property
    def total_cost(self) -> float:
        """Sum of cost_usd across all verdicts."""
        return sum(v.metrics.cost_usd for v in self.verdicts)

    @property
    def avg_cost_per_success(self) -> float:
        """Mean cost_per_success for verdicts with finite cost."""
        costs = [
            v.metrics.cost_per_success
            for v in self.verdicts
            if v.metrics.cost_per_success != float("inf")
        ]
        return sum(costs) / len(costs) if costs else float("inf")


class EvalRunnerV2:
    """Run evaluation cases and produce reports."""

    def __init__(self, judge: EvalJudge | None = None) -> None:
        self.judge = judge or RuleJudge()

    async def run_case(self, case: EvalCaseV2, agent: object) -> EvalVerdict:
        """Run a single eval case against the given agent."""
        start = time.monotonic()

        state = await agent.run(case.goal)  # type: ignore[union-attr]

        wall_clock_ms = int((time.monotonic() - start) * 1000)

        metrics = await self.judge.evaluate(
            goal=case.goal,
            state=state,
            success_criteria=case.success_criteria,
            quality_rubric=case.quality_rubric,
        )
        metrics.wall_clock_ms = wall_clock_ms
        metrics.steps_to_completion = state.current_step
        metrics.tokens_used = state.tokens_used
        metrics.cost_usd = state.cost_usd

        # Determine pass/fail
        passed = metrics.goal_achievement_rate >= 0.5
        if case.max_acceptable_cost and metrics.cost_usd > case.max_acceptable_cost:
            passed = False
        if case.max_acceptable_steps and state.current_step > case.max_acceptable_steps:
            passed = False

        return EvalVerdict(
            case=case,
            metrics=metrics,
            passed=passed,
            reason=f"achievement={metrics.goal_achievement_rate:.1%}",
        )

    async def run_suite(self, cases: list[EvalCaseV2], agent: object) -> EvalSuiteReport:
        """Run all cases and produce aggregate report."""
        verdicts: list[EvalVerdict] = []
        for case in cases:
            verdict = await self.run_case(case, agent)
            verdicts.append(verdict)
        return EvalSuiteReport(verdicts=verdicts)
