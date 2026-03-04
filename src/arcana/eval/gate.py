"""RegressionGate — pass/fail gating for eval reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.eval import RegressionResult

if TYPE_CHECKING:
    from arcana.contracts.eval import EvalReport, GateConfig


class RegressionGate:
    """
    Gate that checks eval reports against thresholds.

    Supports:
    - Absolute checks: pass rate, avg cost, avg tokens
    - Relative checks: regression percentage vs baseline
    """

    def __init__(self, config: GateConfig) -> None:
        self._config = config

    def check(self, report: EvalReport) -> RegressionResult:
        """
        Check a report against absolute thresholds.

        Args:
            report: The evaluation report to check.

        Returns:
            RegressionResult with pass/fail and violations.
        """
        violations: list[str] = []

        if report.pass_rate < self._config.min_pass_rate:
            violations.append(
                f"pass_rate {report.pass_rate:.2%} < {self._config.min_pass_rate:.2%}"
            )

        if self._config.max_avg_cost_usd is not None and report.total > 0:
            avg_cost = report.aggregate_cost_usd / report.total
            if avg_cost > self._config.max_avg_cost_usd:
                violations.append(
                    f"avg_cost ${avg_cost:.4f} > ${self._config.max_avg_cost_usd:.4f}"
                )

        if self._config.max_avg_tokens is not None and report.total > 0:
            avg_tokens = report.aggregate_tokens / report.total
            if avg_tokens > self._config.max_avg_tokens:
                violations.append(
                    f"avg_tokens {avg_tokens:.0f} > {self._config.max_avg_tokens}"
                )

        return RegressionResult(
            passed=len(violations) == 0,
            current_pass_rate=report.pass_rate,
            gate_violations=violations,
        )

    def compare(
        self,
        current: EvalReport,
        baseline: EvalReport,
    ) -> RegressionResult:
        """
        Compare current report against a baseline for regression.

        Args:
            current: Current evaluation report.
            baseline: Baseline evaluation report.

        Returns:
            RegressionResult with regression analysis.
        """
        violations: list[str] = []
        regression_pct: float | None = None

        if baseline.pass_rate > 0:
            regression_pct = (
                (baseline.pass_rate - current.pass_rate) / baseline.pass_rate
            )
            if regression_pct > self._config.max_regression_pct:
                violations.append(
                    f"regression {regression_pct:.2%} > {self._config.max_regression_pct:.2%}"
                )

        # Also apply absolute checks
        if current.pass_rate < self._config.min_pass_rate:
            violations.append(
                f"pass_rate {current.pass_rate:.2%} < {self._config.min_pass_rate:.2%}"
            )

        return RegressionResult(
            passed=len(violations) == 0,
            current_pass_rate=current.pass_rate,
            baseline_pass_rate=baseline.pass_rate,
            regression_pct=regression_pct,
            gate_violations=violations,
        )
