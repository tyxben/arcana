"""Tests for the observability module — MetricsCollector + MetricsHook."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.contracts.runtime import StepResult, StepType
from arcana.contracts.state import AgentState, ExecutionStatus
from arcana.contracts.trace import (
    AgentRole,
    BudgetSnapshot,
    EventType,
    StopReason,
    TraceContext,
    TraceEvent,
)
from arcana.observability.hooks import MetricsHook, StepMetric
from arcana.observability.metrics import AggregateMetrics, MetricsCollector, RunSummary


# ── Helpers ──────────────────────────────────────────────────────────

def _make_event(
    event_type: EventType,
    *,
    run_id: str = "run-001",
    timestamp: datetime | None = None,
    budgets: BudgetSnapshot | None = None,
    stop_reason: StopReason | None = None,
) -> TraceEvent:
    return TraceEvent(
        run_id=run_id,
        event_type=event_type,
        timestamp=timestamp or datetime.now(UTC),
        role=AgentRole.SYSTEM,
        budgets=budgets,
        stop_reason=stop_reason,
    )


def _make_state(
    run_id: str = "run-001",
    current_step: int = 5,
    tokens_used: int = 100,
    cost_usd: float = 0.01,
) -> AgentState:
    return AgentState(
        run_id=run_id,
        status=ExecutionStatus.COMPLETED,
        current_step=current_step,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
    )


# ── MetricsCollector Tests ───────────────────────────────────────────

class TestMetricsCollector:
    def test_summarize_run_empty(self):
        summary = MetricsCollector.summarize_run([])
        assert summary.run_id == "unknown"
        assert summary.total_steps == 0
        assert summary.llm_calls == 0

    def test_summarize_run_full(self):
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        events = [
            _make_event(EventType.LLM_CALL, timestamp=t0),
            _make_event(EventType.LLM_CALL, timestamp=t0 + timedelta(seconds=1)),
            _make_event(EventType.TOOL_CALL, timestamp=t0 + timedelta(seconds=2)),
            _make_event(EventType.ERROR, timestamp=t0 + timedelta(seconds=3)),
            _make_event(
                EventType.STATE_CHANGE,
                timestamp=t0 + timedelta(seconds=4),
                budgets=BudgetSnapshot(tokens_used=500, cost_usd=0.05),
                stop_reason=StopReason.GOAL_REACHED,
            ),
        ]
        summary = MetricsCollector.summarize_run(events)

        assert summary.run_id == "run-001"
        assert summary.llm_calls == 2
        assert summary.tool_calls == 1
        assert summary.errors == 1
        assert summary.tokens_used == 500
        assert summary.cost_usd == pytest.approx(0.05)
        assert summary.duration_ms == 4000
        assert summary.stop_reason == "goal_reached"

    def test_summarize_from_reader(self):
        mock_reader = MagicMock()
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        mock_reader.read_events.return_value = [
            _make_event(EventType.LLM_CALL, run_id="run-x", timestamp=t0),
        ]

        summary = MetricsCollector.summarize_from_reader(mock_reader, "run-x")
        assert summary.run_id == "run-x"
        assert summary.llm_calls == 1
        mock_reader.read_events.assert_called_once_with("run-x")

    def test_summarize_stop_reason_extraction(self):
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        events = [
            _make_event(EventType.LLM_CALL, timestamp=t0),
            _make_event(
                EventType.STATE_CHANGE,
                timestamp=t0 + timedelta(seconds=1),
                stop_reason=StopReason.MAX_STEPS,
            ),
        ]
        summary = MetricsCollector.summarize_run(events)
        assert summary.stop_reason == "max_steps"

    def test_aggregate_averages_and_percentiles(self):
        summaries = [
            RunSummary(run_id=f"run-{i}", total_steps=i * 2, tokens_used=i * 100,
                       cost_usd=i * 0.01, duration_ms=i * 1000)
            for i in range(1, 6)
        ]
        agg = MetricsCollector.aggregate(summaries)

        assert agg.count == 5
        assert agg.avg_steps == pytest.approx(6.0)  # (2+4+6+8+10)/5
        assert agg.avg_tokens == pytest.approx(300.0)  # (100+200+300+400+500)/5
        assert agg.avg_cost_usd == pytest.approx(0.03)
        assert agg.avg_duration_ms == pytest.approx(3000.0)
        # p50 of [100, 200, 300, 400, 500] = 300
        assert agg.p50_tokens == pytest.approx(300.0)
        assert agg.p95_tokens > 0
        assert agg.p99_tokens > 0

    def test_aggregate_single_summary(self):
        summary = RunSummary(run_id="run-1", tokens_used=1000, cost_usd=0.5,
                             duration_ms=5000, total_steps=10)
        agg = MetricsCollector.aggregate([summary])

        assert agg.count == 1
        assert agg.avg_tokens == pytest.approx(1000.0)
        assert agg.p50_tokens == pytest.approx(1000.0)
        assert agg.p95_tokens == pytest.approx(1000.0)

    def test_aggregate_empty(self):
        agg = MetricsCollector.aggregate([])
        assert agg.count == 0
        assert agg.avg_steps == 0.0


# ── MetricsHook Tests ────────────────────────────────────────────────

class TestMetricsHook:
    @pytest.mark.asyncio
    async def test_on_run_start_records_time(self):
        hook = MetricsHook()
        state = _make_state()
        ctx = TraceContext(run_id="run-001")

        await hook.on_run_start(state, ctx)

        assert hook.run_start_time is not None
        assert hook._run_id == "run-001"

    @pytest.mark.asyncio
    async def test_on_step_complete_appends_metric(self):
        hook = MetricsHook()
        state = _make_state(current_step=3)
        ctx = TraceContext(run_id="run-001")

        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=True,
        )

        await hook.on_step_complete(state, step_result, ctx)

        assert len(hook.step_metrics) == 1
        m = hook.step_metrics[0]
        assert m.step_number == 3
        assert m.step_type == "think"
        assert m.success is True

    @pytest.mark.asyncio
    async def test_on_run_end_records_time(self):
        hook = MetricsHook()
        state = _make_state(tokens_used=500, cost_usd=0.05)
        ctx = TraceContext(run_id="run-001")

        await hook.on_run_start(state, ctx)
        await hook.on_run_end(state, ctx)

        assert hook.run_end_time is not None
        assert hook.total_tokens == 500
        assert hook.total_cost == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_on_error_increments_count(self):
        hook = MetricsHook()
        state = _make_state()
        ctx = TraceContext(run_id="run-001")

        await hook.on_error(state, ValueError("test"), ctx)
        await hook.on_error(state, RuntimeError("test2"), ctx)

        assert hook._error_count == 2

    @pytest.mark.asyncio
    async def test_to_summary(self):
        hook = MetricsHook()
        state = _make_state(run_id="run-42", tokens_used=1000, cost_usd=0.1)
        ctx = TraceContext(run_id="run-42")

        await hook.on_run_start(state, ctx)

        # Simulate steps
        for step_type in [StepType.THINK, StepType.ACT, StepType.THINK]:
            step_result = StepResult(
                step_type=step_type,
                step_id=f"step-{step_type.value}",
                success=True,
            )
            await hook.on_step_complete(state, step_result, ctx)

        await hook.on_error(state, ValueError("err"), ctx)
        await hook.on_run_end(state, ctx)

        summary = hook.to_summary()
        assert summary.run_id == "run-42"
        assert summary.total_steps == 3
        assert summary.llm_calls == 2  # 2x think
        assert summary.tool_calls == 1  # 1x act
        assert summary.errors == 1
        assert summary.tokens_used == 1000
        assert summary.cost_usd == pytest.approx(0.1)
        assert summary.duration_ms >= 0


# ── StepMetric Tests ─────────────────────────────────────────────────

class TestStepMetric:
    def test_slots(self):
        m = StepMetric(
            step_number=1,
            step_type="think",
            tokens_used=100,
            duration_ms=50.0,
            success=True,
        )
        assert m.step_number == 1
        assert m.step_type == "think"
        assert m.tokens_used == 100
        assert m.duration_ms == 50.0
        assert m.success is True
        assert hasattr(m, "__slots__")


# ── Module Exports Test ──────────────────────────────────────────────

class TestModuleExports:
    def test_observability_exports(self):
        import arcana.observability as obs

        assert hasattr(obs, "MetricsCollector")
        assert hasattr(obs, "MetricsHook")
        assert hasattr(obs, "RunSummary")
        assert hasattr(obs, "AggregateMetrics")
        assert hasattr(obs, "StepMetric")
