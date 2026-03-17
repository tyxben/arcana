"""Tests for the eval module — EvalRunner + RegressionGate."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from arcana.contracts.eval import (
    EvalCase,
    EvalReport,
    EvalResult,
    GateConfig,
    OutcomeCriterion,
)
from arcana.contracts.state import AgentState, ExecutionStatus
from arcana.eval.gate import RegressionGate
from arcana.eval.runner import EvalRunner

# ── Helpers ──────────────────────────────────────────────────────────

def _make_completed_state(
    *,
    run_id: str = "eval-run-001",
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
    current_step: int = 5,
    tokens_used: int = 200,
    cost_usd: float = 0.02,
    working_memory: dict | None = None,
) -> AgentState:
    return AgentState(
        run_id=run_id,
        status=status,
        current_step=current_step,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        working_memory=working_memory or {},
    )


def _make_mock_agent(state: AgentState | None = None) -> MagicMock:
    agent = MagicMock()
    agent.trace_writer = None  # No trace writer by default
    state = state or _make_completed_state()
    agent.run = AsyncMock(return_value=state)
    return agent


def _make_report(
    *,
    total: int = 10,
    passed: int = 9,
    aggregate_cost: float = 1.0,
    aggregate_tokens: int = 5000,
) -> EvalReport:
    failed = total - passed
    return EvalReport(
        suite_name="test",
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=passed / total if total > 0 else 0.0,
        aggregate_cost_usd=aggregate_cost,
        aggregate_tokens=aggregate_tokens,
        aggregate_duration_ms=10000,
    )


# ── Contract Tests ───────────────────────────────────────────────────

class TestContracts:
    def test_eval_case_frozen(self):
        case = EvalCase(
            id="case-1",
            goal="do something",
            expected_outcome=OutcomeCriterion.STATUS,
            expected_value="completed",
        )
        with pytest.raises(ValidationError):  # frozen (Pydantic v2)
            case.id = "changed"  # type: ignore[misc]

    def test_eval_result_serialization(self):
        result = EvalResult(
            case_id="case-1",
            passed=True,
            actual_status="completed",
            steps=5,
            tokens_used=200,
            cost_usd=0.02,
            duration_ms=1500,
        )
        data = result.model_dump()
        assert data["case_id"] == "case-1"
        assert data["passed"] is True
        restored = EvalResult.model_validate(data)
        assert restored == result

    def test_gate_config_defaults(self):
        config = GateConfig()
        assert config.min_pass_rate == 0.9
        assert config.max_regression_pct == 0.05
        assert config.max_avg_cost_usd is None
        assert config.max_avg_tokens is None


# ── EvalRunner Tests ─────────────────────────────────────────────────

class TestEvalRunner:
    @pytest.mark.asyncio
    async def test_run_case_pass(self):
        state = _make_completed_state(status=ExecutionStatus.COMPLETED)
        agent = _make_mock_agent(state)
        runner = EvalRunner(agent_factory=lambda: agent)

        case = EvalCase(
            id="case-1",
            goal="achieve goal",
            expected_outcome=OutcomeCriterion.STATUS,
            expected_value="completed",
        )
        result = await runner.run_case(case)

        assert result.passed is True
        assert result.case_id == "case-1"
        assert result.actual_status == "completed"
        agent.run.assert_awaited_once_with("achieve goal")

    @pytest.mark.asyncio
    async def test_run_case_fail(self):
        state = _make_completed_state(status=ExecutionStatus.FAILED)
        agent = _make_mock_agent(state)
        runner = EvalRunner(agent_factory=lambda: agent)

        case = EvalCase(
            id="case-2",
            goal="achieve goal",
            expected_outcome=OutcomeCriterion.STATUS,
            expected_value="completed",
        )
        result = await runner.run_case(case)

        assert result.passed is False
        assert result.actual_status == "failed"

    @pytest.mark.asyncio
    async def test_run_case_exception(self):
        agent = MagicMock()
        agent.trace_writer = None
        agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        runner = EvalRunner(agent_factory=lambda: agent)

        case = EvalCase(
            id="case-err",
            goal="crash",
            expected_outcome=OutcomeCriterion.STATUS,
            expected_value="completed",
        )
        result = await runner.run_case(case)

        assert result.passed is False
        assert result.error == "boom"
        assert result.actual_status == "error"

    @pytest.mark.asyncio
    async def test_run_suite_aggregation(self):
        state = _make_completed_state(tokens_used=100, cost_usd=0.01)
        agent = _make_mock_agent(state)
        runner = EvalRunner(agent_factory=lambda: agent)

        cases = [
            EvalCase(id=f"case-{i}", goal="goal", expected_outcome=OutcomeCriterion.STATUS,
                     expected_value="completed")
            for i in range(3)
        ]
        report = await runner.run_suite(cases, suite_name="my-suite")

        assert report.suite_name == "my-suite"
        assert report.total == 3
        assert report.passed == 3
        assert report.failed == 0
        assert report.pass_rate == pytest.approx(1.0)
        assert len(report.results) == 3

    @pytest.mark.asyncio
    async def test_outcome_contains_keys(self):
        state = _make_completed_state(working_memory={"key_a": 1, "key_b": 2})
        agent = _make_mock_agent(state)
        runner = EvalRunner(agent_factory=lambda: agent)

        case = EvalCase(
            id="case-keys",
            goal="produce keys",
            expected_outcome=OutcomeCriterion.CONTAINS_KEYS,
            expected_value=["key_a", "key_b"],
        )
        result = await runner.run_case(case)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_outcome_contains_keys_missing(self):
        state = _make_completed_state(working_memory={"key_a": 1})
        agent = _make_mock_agent(state)
        runner = EvalRunner(agent_factory=lambda: agent)

        case = EvalCase(
            id="case-keys-miss",
            goal="produce keys",
            expected_outcome=OutcomeCriterion.CONTAINS_KEYS,
            expected_value=["key_a", "key_c"],
        )
        result = await runner.run_case(case)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_outcome_max_steps(self):
        state = _make_completed_state(current_step=8)
        agent = _make_mock_agent(state)
        runner = EvalRunner(agent_factory=lambda: agent)

        case = EvalCase(
            id="case-steps",
            goal="be efficient",
            expected_outcome=OutcomeCriterion.MAX_STEPS,
            expected_value=10,
        )
        result = await runner.run_case(case)
        assert result.passed is True  # 8 <= 10

    @pytest.mark.asyncio
    async def test_outcome_max_steps_exceeded(self):
        state = _make_completed_state(current_step=15)
        agent = _make_mock_agent(state)
        runner = EvalRunner(agent_factory=lambda: agent)

        case = EvalCase(
            id="case-steps-fail",
            goal="be efficient",
            expected_outcome=OutcomeCriterion.MAX_STEPS,
            expected_value=10,
        )
        result = await runner.run_case(case)
        assert result.passed is False  # 15 > 10

    @pytest.mark.asyncio
    async def test_outcome_key_values(self):
        state = _make_completed_state(
            working_memory={"answer": 42, "status": "done"}
        )
        agent = _make_mock_agent(state)
        runner = EvalRunner(agent_factory=lambda: agent)

        case = EvalCase(
            id="case-kv",
            goal="compute answer",
            expected_outcome=OutcomeCriterion.KEY_VALUES,
            expected_value={"answer": 42, "status": "done"},
        )
        result = await runner.run_case(case)
        assert result.passed is True


# ── RegressionGate Tests ─────────────────────────────────────────────

class TestRegressionGate:
    def test_check_pass(self):
        gate = RegressionGate(GateConfig(min_pass_rate=0.8))
        report = _make_report(total=10, passed=9)

        result = gate.check(report)
        assert result.passed is True
        assert result.current_pass_rate == pytest.approx(0.9)
        assert len(result.gate_violations) == 0

    def test_check_fail_pass_rate(self):
        gate = RegressionGate(GateConfig(min_pass_rate=0.9))
        report = _make_report(total=10, passed=7)

        result = gate.check(report)
        assert result.passed is False
        assert len(result.gate_violations) == 1
        assert "pass_rate" in result.gate_violations[0]

    def test_check_cost_threshold(self):
        gate = RegressionGate(
            GateConfig(min_pass_rate=0.5, max_avg_cost_usd=0.05)
        )
        report = _make_report(total=10, passed=10, aggregate_cost=1.0)

        result = gate.check(report)
        assert result.passed is False
        assert any("avg_cost" in v for v in result.gate_violations)

    def test_check_tokens_threshold(self):
        gate = RegressionGate(
            GateConfig(min_pass_rate=0.5, max_avg_tokens=100)
        )
        report = _make_report(total=10, passed=10, aggregate_tokens=5000)

        result = gate.check(report)
        assert result.passed is False
        assert any("avg_tokens" in v for v in result.gate_violations)

    def test_compare_no_regression(self):
        gate = RegressionGate(GateConfig(min_pass_rate=0.5, max_regression_pct=0.1))
        current = _make_report(total=10, passed=9)
        baseline = _make_report(total=10, passed=9)

        result = gate.compare(current, baseline)
        assert result.passed is True
        assert result.regression_pct == pytest.approx(0.0)
        assert result.baseline_pass_rate == pytest.approx(0.9)

    def test_compare_detects_regression(self):
        gate = RegressionGate(GateConfig(min_pass_rate=0.5, max_regression_pct=0.05))
        current = _make_report(total=10, passed=7)
        baseline = _make_report(total=10, passed=9)

        result = gate.compare(current, baseline)
        assert result.passed is False
        # regression: (0.9 - 0.7) / 0.9 ≈ 0.222
        assert result.regression_pct is not None
        assert result.regression_pct > 0.05
        assert any("regression" in v for v in result.gate_violations)

    def test_compare_improvement(self):
        gate = RegressionGate(GateConfig(min_pass_rate=0.5, max_regression_pct=0.05))
        current = _make_report(total=10, passed=10)
        baseline = _make_report(total=10, passed=8)

        result = gate.compare(current, baseline)
        assert result.passed is True
        # regression_pct should be negative (improvement)
        assert result.regression_pct is not None
        assert result.regression_pct < 0


# ── Module Exports Test ──────────────────────────────────────────────

class TestModuleExports:
    def test_eval_exports(self):
        import arcana.eval as ev

        assert hasattr(ev, "EvalRunner")
        assert hasattr(ev, "RegressionGate")
        assert hasattr(ev, "EvalCase")
        assert hasattr(ev, "EvalResult")
        assert hasattr(ev, "EvalReport")
        assert hasattr(ev, "GateConfig")
        assert hasattr(ev, "RegressionResult")
        assert hasattr(ev, "OutcomeCriterion")
