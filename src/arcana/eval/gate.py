"""RegressionGate — pass/fail gating for eval reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.eval import RegressionResult

if TYPE_CHECKING:
    from arcana.contracts.eval import EvalReport, GateConfig

# None (no context decision) and L0 (kept verbatim) both mean "no loss".
_FIDELITY_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def _fidelity_worse(current: str | None, threshold: str) -> bool:
    """True if ``current`` fidelity is worse than the allowed ``threshold``."""
    cur = _FIDELITY_RANK.get(current or "L0", 0)
    thr = _FIDELITY_RANK.get(threshold, 0)
    return cur > thr


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

        signal_violations, warnings = self._signal_checks(report)
        golden_violations, golden_warnings = self._golden_checks(report)
        hard = violations + signal_violations + golden_violations
        return RegressionResult(
            passed=len(hard) == 0,
            current_pass_rate=report.pass_rate,
            # Hard signal/golden fails are mirrored into gate_violations so
            # callers reading only that field still see passed=False.
            gate_violations=hard,
            signal_violations=signal_violations,
            golden_violations=golden_violations,
            warnings=warnings + golden_warnings,
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

        signal_violations, warnings = self._signal_checks(current, baseline)
        golden_violations, golden_warnings = self._golden_checks(current)
        hard = violations + signal_violations + golden_violations
        return RegressionResult(
            passed=len(hard) == 0,
            current_pass_rate=current.pass_rate,
            baseline_pass_rate=baseline.pass_rate,
            regression_pct=regression_pct,
            gate_violations=hard,
            signal_violations=signal_violations,
            golden_violations=golden_violations,
            warnings=warnings + golden_warnings,
        )

    def _golden_checks(self, report: EvalReport) -> tuple[list[str], list[str]]:
        """Golden-replay gate (finding F5). ``strict`` hard-fails on any golden
        regression; ``warn`` only warns; ``off`` (default) ignores."""
        cfg = self._config
        if cfg.golden_replay == "off" or not report.golden_regressions:
            return [], []
        regressions = list(report.golden_regressions)
        if cfg.golden_replay == "strict":
            return regressions, []
        return [], regressions

    def _signal_checks(
        self,
        report: EvalReport,
        baseline: EvalReport | None = None,
    ) -> tuple[list[str], list[str]]:
        """Trace-signal gates (finding F5).

        Returns ``(hard_violations, warnings)``. Hard fails happen only on
        explicit opt-in config (so a default GateConfig never gates on
        signals); everything else is a warning. This keeps the gate evidence,
        not a hidden supervisor (Principle 7 Corollary).
        """
        cfg = self._config
        hard: list[str] = []
        warnings: list[str] = []

        # require_trace is per-case (not the suite-level merge): an errored or
        # untraced case has signals=None and would otherwise be invisible.
        if cfg.require_trace:
            untraced = [
                r.case_id
                for r in report.results
                if r.signals is None or not r.signals.trace_available
            ]
            if untraced:
                hard.append(f"require_trace: untraced/errored cases {untraced}")
            elif not report.results:
                hard.append("require_trace: no cases were run")

        sig = report.aggregate_signals
        if sig is None:
            return hard, warnings

        _ceilings = [
            (cfg.max_permission_denials, sig.permission_denials, "permission_denials"),
            (cfg.max_guardrail_blocks, sig.guardrail_blocks, "guardrail_blocks"),
            (cfg.max_write_tool_calls, sig.write_tool_calls, "write_tool_calls"),
            (cfg.max_imported_capability_calls, sig.imported_capability_calls,
             "imported_capability_calls"),
            (cfg.max_capability_downgrades, sig.capability_downgrades,
             "capability_downgrades"),
        ]
        for ceiling, value, name in _ceilings:
            if ceiling is not None and value > ceiling:
                hard.append(f"{name} {value} > {ceiling}")

        if (
            cfg.max_context_loss_ratio is not None
            and sig.context_loss_ratio > cfg.max_context_loss_ratio
        ):
            hard.append(
                f"context_loss_ratio {sig.context_loss_ratio:.3f} > "
                f"{cfg.max_context_loss_ratio:.3f}"
            )
        if cfg.max_context_fidelity is not None and _fidelity_worse(
            sig.context_fidelity_min, cfg.max_context_fidelity
        ):
            hard.append(
                f"context_fidelity_min {sig.context_fidelity_min} worse than "
                f"{cfg.max_context_fidelity}"
            )
        if cfg.forbid_provider_degradation and sig.provider_degraded:
            hard.append(
                f"provider degraded: {sorted(sig.degraded_capabilities)}"
            )
        elif sig.provider_degraded:
            warnings.append(
                f"provider degraded ({sorted(sig.degraded_capabilities)}) — not gated"
            )

        if cfg.forbid_new_tool_error_categories:
            if baseline is not None and baseline.aggregate_signals is not None:
                base_categories = set(baseline.aggregate_signals.tool_error_categories)
                new_categories = set(sig.tool_error_categories) - base_categories
                if new_categories:
                    hard.append(
                        f"new tool_error_categories vs baseline: "
                        f"{sorted(new_categories)}"
                    )
            elif sig.tool_error_categories:
                # No baseline to diff against (single-report check) — surface
                # the categories present so the knob is not a silent no-op.
                warnings.append(
                    f"tool_error_categories present (no baseline): "
                    f"{sorted(sig.tool_error_categories)}"
                )

        return hard, warnings
