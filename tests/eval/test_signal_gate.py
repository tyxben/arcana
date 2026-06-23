"""Tests for the RegressionGate trace-signal pass (finding F5, slice 5).

The gate gains boundary-signal checks that hard-fail ONLY on explicit opt-in,
so a default GateConfig reproduces legacy behaviour. A green pass-rate that
introduced a permission denial / provider degradation / new tool-error
category is detectable as a regression — the anti-Goodhart property.
"""

from __future__ import annotations

from arcana.contracts.eval import EvalReport, EvalResult, GateConfig, TraceSignals
from arcana.eval.gate import RegressionGate


def _report(*, pass_rate: float = 1.0, signals: TraceSignals | None = None) -> EvalReport:
    # Per-case result carries the signals too (require_trace is per-case).
    result = EvalResult(
        case_id="c1", passed=True, actual_status="completed",
        steps=1, tokens_used=10, cost_usd=0.0, duration_ms=1, signals=signals,
    )
    return EvalReport(
        suite_name="s", total=1, passed=1, failed=0,
        pass_rate=pass_rate, results=[result], aggregate_signals=signals,
    )


class TestDefaultGateUnchanged:
    def test_default_config_ignores_signals(self):
        # A report full of boundary events still passes a default gate (signals
        # are recorded as evidence but never gate unless opted in).
        sig = TraceSignals(permission_denials=5, guardrail_blocks=3,
                           provider_degraded=True, trace_available=True)
        result = RegressionGate(GateConfig()).check(_report(signals=sig))
        assert result.passed is True
        assert result.signal_violations == []


class TestSignalHardFails:
    def test_permission_denial_fails_when_ceilinged(self):
        sig = TraceSignals(permission_denials=1, trace_available=True)
        cfg = GateConfig(max_permission_denials=0)
        result = RegressionGate(cfg).check(_report(signals=sig))
        assert result.passed is False
        assert result.signal_violations
        # mirrored into gate_violations for legacy callers
        assert any("permission_denials" in v for v in result.gate_violations)

    def test_same_report_passes_without_the_ceiling(self):
        # Opt-in proof: identical report, no signal config -> green.
        sig = TraceSignals(permission_denials=1, trace_available=True)
        result = RegressionGate(GateConfig()).check(_report(signals=sig))
        assert result.passed is True

    def test_provider_degradation_forbidden(self):
        sig = TraceSignals(provider_degraded=True,
                           degraded_capabilities=["tool_calls"], trace_available=True)
        result = RegressionGate(
            GateConfig(forbid_provider_degradation=True)
        ).check(_report(signals=sig))
        assert result.passed is False
        assert any("degraded" in v for v in result.signal_violations)

    def test_provider_degradation_warns_when_not_forbidden(self):
        sig = TraceSignals(provider_degraded=True,
                           degraded_capabilities=["tool_calls"], trace_available=True)
        result = RegressionGate(GateConfig()).check(_report(signals=sig))
        assert result.passed is True
        assert any("degraded" in w for w in result.warnings)

    def test_require_trace_fails_on_missing_signals(self):
        result = RegressionGate(
            GateConfig(require_trace=True)
        ).check(_report(signals=None))
        assert result.passed is False
        assert any("require_trace" in v for v in result.signal_violations)

    def test_context_loss_ceiling(self):
        sig = TraceSignals(context_messages_in=10, context_messages_dropped=5,
                           context_loss_ratio=0.5, trace_available=True)
        result = RegressionGate(
            GateConfig(max_context_loss_ratio=0.2)
        ).check(_report(signals=sig))
        assert result.passed is False


class TestBoundaryCeilings:
    def test_write_tool_calls_ceiling(self):
        sig = TraceSignals(write_tool_calls=2, trace_available=True)
        result = RegressionGate(
            GateConfig(max_write_tool_calls=0)
        ).check(_report(signals=sig))
        assert result.passed is False
        assert any("write_tool_calls" in v for v in result.signal_violations)

    def test_imported_capability_ceiling(self):
        sig = TraceSignals(imported_capability_calls=1, trace_available=True)
        result = RegressionGate(
            GateConfig(max_imported_capability_calls=0)
        ).check(_report(signals=sig))
        assert result.passed is False

    def test_capability_downgrade_ceiling(self):
        sig = TraceSignals(capability_downgrades=1, trace_available=True)
        result = RegressionGate(
            GateConfig(max_capability_downgrades=0)
        ).check(_report(signals=sig))
        assert result.passed is False

    def test_fidelity_threshold(self):
        sig = TraceSignals(context_fidelity_min="L3", trace_available=True)
        result = RegressionGate(
            GateConfig(max_context_fidelity="L1")
        ).check(_report(signals=sig))
        assert result.passed is False
        # L0/None is no worse than any threshold
        ok = TraceSignals(context_fidelity_min="L0", trace_available=True)
        assert RegressionGate(
            GateConfig(max_context_fidelity="L1")
        ).check(_report(signals=ok)).passed is True


class TestRequireTracePerCase:
    def test_untraced_case_fails_even_if_aggregate_present(self):
        from arcana.contracts.eval import EvalResult
        # One traced case + one errored (signals None) case.
        traced = EvalResult(case_id="ok", passed=True, actual_status="completed",
                            steps=1, tokens_used=1, cost_usd=0.0, duration_ms=1,
                            signals=TraceSignals(trace_available=True))
        errored = EvalResult(case_id="bad", passed=False, actual_status="error",
                             steps=0, tokens_used=0, cost_usd=0.0, duration_ms=1,
                             signals=None)
        report = EvalReport(
            suite_name="s", total=2, passed=1, failed=1, pass_rate=0.5,
            results=[traced, errored],
            aggregate_signals=TraceSignals(trace_available=True),
        )
        result = RegressionGate(
            GateConfig(min_pass_rate=0.0, require_trace=True)
        ).check(report)
        assert result.passed is False
        assert any("bad" in v for v in result.signal_violations)


class TestImprovementsNeverFail:
    def test_zero_boundary_events_passes_strict_config(self):
        sig = TraceSignals(permission_denials=0, guardrail_blocks=0,
                           provider_degraded=False, trace_available=True)
        cfg = GateConfig(max_permission_denials=0, max_guardrail_blocks=0,
                         forbid_provider_degradation=True, require_trace=True)
        result = RegressionGate(cfg).check(_report(signals=sig))
        assert result.passed is True


class TestNewToolErrorCategoryVsBaseline:
    def test_new_category_fails_compare(self):
        baseline = _report(signals=TraceSignals(
            tool_error_categories={"timeout": 1}, trace_available=True))
        current = _report(signals=TraceSignals(
            tool_error_categories={"timeout": 1, "permission": 2},
            trace_available=True))
        cfg = GateConfig(forbid_new_tool_error_categories=True)
        result = RegressionGate(cfg).compare(current, baseline)
        assert result.passed is False
        assert any("tool_error_categories" in v for v in result.signal_violations)

    def test_same_categories_pass_compare(self):
        baseline = _report(signals=TraceSignals(
            tool_error_categories={"timeout": 1}, trace_available=True))
        current = _report(signals=TraceSignals(
            tool_error_categories={"timeout": 3}, trace_available=True))
        cfg = GateConfig(forbid_new_tool_error_categories=True)
        result = RegressionGate(cfg).compare(current, baseline)
        assert result.passed is True


class TestGoldenReplayGate:
    def _golden_report(self):
        r = _report(signals=TraceSignals(trace_available=True))
        return r.model_copy(update={"golden_regressions": ["c1: guardrail_blocks 0->1"]})

    def test_strict_fails_on_golden_regression(self):
        result = RegressionGate(
            GateConfig(golden_replay="strict")
        ).check(self._golden_report())
        assert result.passed is False
        assert result.golden_violations
        assert any("guardrail_blocks" in v for v in result.gate_violations)

    def test_warn_only_warns(self):
        result = RegressionGate(
            GateConfig(golden_replay="warn")
        ).check(self._golden_report())
        assert result.passed is True
        assert result.warnings

    def test_off_ignores(self):
        result = RegressionGate(GateConfig()).check(self._golden_report())
        assert result.passed is True
        assert result.golden_violations == []
