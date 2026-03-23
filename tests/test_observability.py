"""Tests for the observability module — MetricsCollector + MetricsHook + BudgetWarningHook."""

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

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
from arcana.observability.hooks import BudgetWarningHook, MetricsHook, StepMetric
from arcana.observability.metrics import (
    MetricsCollector,
    RunSummary,
)

# ── Helpers ──────────────────────────────────────────────────────────

def _make_event(
    event_type: EventType,
    *,
    run_id: str = "run-001",
    timestamp: datetime | None = None,
    budgets: BudgetSnapshot | None = None,
    stop_reason: StopReason | None = None,
    model: str | None = None,
    metadata: dict | None = None,
) -> TraceEvent:
    return TraceEvent(
        run_id=run_id,
        event_type=event_type,
        timestamp=timestamp or datetime.now(UTC),
        role=AgentRole.SYSTEM,
        budgets=budgets,
        stop_reason=stop_reason,
        model=model,
        metadata=metadata or {},
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
            _make_event(EventType.LLM_CALL, timestamp=t0, model="test-model"),
            _make_event(EventType.LLM_CALL, timestamp=t0 + timedelta(seconds=1)),
            _make_event(EventType.TOOL_CALL, timestamp=t0 + timedelta(seconds=2)),
            _make_event(EventType.ERROR, timestamp=t0 + timedelta(seconds=3),
                        metadata={"error_category": "rate_limit"}),
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
        # Enhanced fields
        assert summary.success is False  # has errors
        assert summary.model_name == "test-model"
        assert summary.error_categories == {"rate_limit": 1}

    def test_summarize_run_success(self):
        """Run with no errors should have success=True."""
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        events = [
            _make_event(EventType.LLM_CALL, timestamp=t0),
            _make_event(
                EventType.STATE_CHANGE,
                timestamp=t0 + timedelta(seconds=1),
                stop_reason=StopReason.GOAL_REACHED,
            ),
        ]
        summary = MetricsCollector.summarize_run(events)
        assert summary.success is True

    def test_summarize_run_extracts_provider_from_metadata(self):
        """provider_name should be extracted from LLM_CALL event metadata (P1 fix)."""
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        events = [
            _make_event(EventType.LLM_CALL, timestamp=t0, model="deepseek-chat",
                        metadata={"provider": "deepseek"}),
            _make_event(EventType.STATE_CHANGE, timestamp=t0 + timedelta(seconds=1)),
        ]
        summary = MetricsCollector.summarize_run(events)
        assert summary.provider_name == "deepseek"
        assert summary.model_name == "deepseek-chat"

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
        # Duration percentiles
        assert agg.p50_duration_ms == pytest.approx(3000.0)
        assert agg.p95_duration_ms > 0
        # Cost percentiles
        assert agg.p50_cost_usd == pytest.approx(0.03)
        assert agg.p95_cost_usd > 0
        # Rates (all successful by default)
        assert agg.success_rate == pytest.approx(1.0)
        assert agg.error_rate == pytest.approx(0.0)

    def test_aggregate_with_errors(self):
        summaries = [
            RunSummary(run_id="s", success=True),
            RunSummary(run_id="f", success=False, error_categories={"timeout": 2}),
        ]
        agg = MetricsCollector.aggregate(summaries)
        assert agg.success_rate == pytest.approx(0.5)
        assert agg.error_rate == pytest.approx(0.5)
        assert agg.error_type_counts == {"timeout": 2}

    def test_aggregate_single_summary(self):
        summary = RunSummary(run_id="run-1", tokens_used=1000, cost_usd=0.5,
                             duration_ms=5000, total_steps=10)
        agg = MetricsCollector.aggregate([summary])

        assert agg.count == 1
        assert agg.avg_tokens == pytest.approx(1000.0)
        assert agg.p50_tokens == pytest.approx(1000.0)
        assert agg.p95_tokens == pytest.approx(1000.0)
        assert agg.p50_duration_ms == pytest.approx(5000.0)
        assert agg.p50_cost_usd == pytest.approx(0.5)

    def test_aggregate_empty(self):
        agg = MetricsCollector.aggregate([])
        assert agg.count == 0
        assert agg.avg_steps == 0.0


# ── ProviderMetrics Tests ───────────────────────────────────────────

class TestProviderMetrics:
    def test_by_provider_groups_correctly(self):
        summaries = [
            RunSummary(run_id="r1", provider_name="deepseek", tokens_used=100,
                       cost_usd=0.01, duration_ms=1000, errors=0),
            RunSummary(run_id="r2", provider_name="deepseek", tokens_used=200,
                       cost_usd=0.02, duration_ms=2000, errors=1),
            RunSummary(run_id="r3", provider_name="openai", tokens_used=300,
                       cost_usd=0.05, duration_ms=500, errors=0),
        ]
        result = MetricsCollector.by_provider(summaries)

        assert "deepseek" in result
        assert "openai" in result

        ds = result["deepseek"]
        assert ds.call_count == 2
        assert ds.total_tokens == 300
        assert ds.total_cost_usd == pytest.approx(0.03)
        assert ds.avg_latency_ms == pytest.approx(1500.0)
        assert ds.error_count == 1

        oai = result["openai"]
        assert oai.call_count == 1
        assert oai.total_tokens == 300
        assert oai.total_cost_usd == pytest.approx(0.05)

    def test_by_provider_unknown_provider(self):
        summaries = [RunSummary(run_id="r1")]  # provider_name=None
        result = MetricsCollector.by_provider(summaries)
        assert "unknown" in result


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
        assert summary.success is False  # has errors


# ── BudgetWarningHook Tests ──────────────────────────────────────────

class TestBudgetWarningHook:
    @pytest.mark.asyncio
    async def test_warn_threshold(self, caplog):
        hook = BudgetWarningHook(
            warn_at_pct=0.8, critical_at_pct=0.95,
            max_tokens=1000,
        )
        state = _make_state(tokens_used=800)
        ctx = TraceContext(run_id="run-001")
        step = StepResult(step_type=StepType.THINK, step_id="s1", success=True)

        await hook.on_run_start(state, ctx)

        with caplog.at_level(logging.WARNING):
            await hook.on_step_complete(state, step, ctx)

        assert "warn" in hook._fired.get("tokens", set())
        assert "critical" not in hook._fired.get("tokens", set())
        assert any("WARNING" in r.message and "tokens" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_critical_threshold(self, caplog):
        hook = BudgetWarningHook(
            warn_at_pct=0.8, critical_at_pct=0.95,
            max_tokens=1000,
        )
        state = _make_state(tokens_used=960)
        ctx = TraceContext(run_id="run-001")
        step = StepResult(step_type=StepType.THINK, step_id="s1", success=True)

        await hook.on_run_start(state, ctx)

        with caplog.at_level(logging.WARNING):
            await hook.on_step_complete(state, step, ctx)

        assert "critical" in hook._fired.get("tokens", set())
        assert any("CRITICAL" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_duplicate_warnings(self, caplog):
        hook = BudgetWarningHook(
            warn_at_pct=0.8, critical_at_pct=0.95,
            max_tokens=1000,
        )
        state = _make_state(tokens_used=850)
        ctx = TraceContext(run_id="run-001")
        step = StepResult(step_type=StepType.THINK, step_id="s1", success=True)

        await hook.on_run_start(state, ctx)

        with caplog.at_level(logging.WARNING):
            await hook.on_step_complete(state, step, ctx)
            await hook.on_step_complete(state, step, ctx)  # second call

        # Should only fire once
        warn_messages = [r for r in caplog.records if "WARNING" in r.message and "tokens" in r.message]
        assert len(warn_messages) == 1

    @pytest.mark.asyncio
    async def test_cost_budget_warning(self, caplog):
        hook = BudgetWarningHook(
            warn_at_pct=0.5, critical_at_pct=0.9,
            max_cost_usd=1.0,
        )
        state = _make_state(cost_usd=0.55)
        ctx = TraceContext(run_id="run-001")
        step = StepResult(step_type=StepType.THINK, step_id="s1", success=True)

        await hook.on_run_start(state, ctx)

        with caplog.at_level(logging.WARNING):
            await hook.on_step_complete(state, step, ctx)

        assert "warn" in hook._fired.get("cost", set())
        assert any("cost" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_below_threshold_no_warning(self, caplog):
        hook = BudgetWarningHook(
            warn_at_pct=0.8, critical_at_pct=0.95,
            max_tokens=1000,
        )
        state = _make_state(tokens_used=100)  # 10%, well below
        ctx = TraceContext(run_id="run-001")
        step = StepResult(step_type=StepType.THINK, step_id="s1", success=True)

        await hook.on_run_start(state, ctx)

        with caplog.at_level(logging.WARNING):
            await hook.on_step_complete(state, step, ctx)

        assert hook._fired.get("tokens", set()) == set()

    @pytest.mark.asyncio
    async def test_reset_on_run_start(self):
        hook = BudgetWarningHook(max_tokens=1000)
        hook._fired = {"tokens": {"warn", "critical"}}

        state = _make_state()
        ctx = TraceContext(run_id="run-001")
        await hook.on_run_start(state, ctx)

        assert hook._fired == {}

    @pytest.mark.asyncio
    async def test_cross_resource_no_suppression(self, caplog):
        """Token warning should NOT suppress cost warning (P2 fix)."""
        hook = BudgetWarningHook(
            warn_at_pct=0.8, critical_at_pct=0.95,
            max_tokens=1000, max_cost_usd=1.0,
        )
        ctx = TraceContext(run_id="run-001")
        step = StepResult(step_type=StepType.THINK, step_id="s1", success=True)

        await hook.on_run_start(_make_state(), ctx)

        # First: tokens hit warning
        state1 = _make_state(tokens_used=850, cost_usd=0.10)
        with caplog.at_level(logging.WARNING):
            await hook.on_step_complete(state1, step, ctx)

        assert "warn" in hook._fired.get("tokens", set())
        assert hook._fired.get("cost", set()) == set()  # cost NOT fired yet

        # Second: cost hits warning — should NOT be suppressed
        state2 = _make_state(tokens_used=850, cost_usd=0.85)
        with caplog.at_level(logging.WARNING):
            await hook.on_step_complete(state2, step, ctx)

        assert "warn" in hook._fired.get("cost", set())
        cost_warnings = [r for r in caplog.records if "cost" in r.message]
        assert len(cost_warnings) == 1


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

    def test_provider_name_slot(self):
        m = StepMetric(
            step_number=1,
            step_type="think",
            tokens_used=0,
            duration_ms=0.0,
            success=True,
            provider_name="deepseek",
        )
        assert m.provider_name == "deepseek"


# ── TraceEvent llm_latency_ms Test ──────────────────────────────────

class TestTraceEventLatency:
    def test_llm_latency_ms_field(self):
        event = TraceEvent(
            run_id="r1",
            event_type=EventType.LLM_CALL,
            llm_latency_ms=150,
        )
        assert event.llm_latency_ms == 150

    def test_llm_latency_ms_default_none(self):
        event = TraceEvent(
            run_id="r1",
            event_type=EventType.LLM_CALL,
        )
        assert event.llm_latency_ms is None


# ── Module Exports Test ──────────────────────────────────────────────

class TestModuleExports:
    def test_observability_exports(self):
        import arcana.observability as obs

        assert hasattr(obs, "MetricsCollector")
        assert hasattr(obs, "MetricsHook")
        assert hasattr(obs, "RunSummary")
        assert hasattr(obs, "AggregateMetrics")
        assert hasattr(obs, "StepMetric")
        assert hasattr(obs, "ProviderMetrics")
        assert hasattr(obs, "BudgetWarningHook")
