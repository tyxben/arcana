"""EvalRunner — execute evaluation cases and produce reports."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from arcana.contracts.eval import (
    EvalReport,
    EvalResult,
    OutcomeCriterion,
)
from arcana.observability.metrics import MetricsCollector

if TYPE_CHECKING:
    from arcana.contracts.eval import EvalCase
    from arcana.contracts.state import AgentState
    from arcana.contracts.trace import TraceEvent
    from arcana.runtime.agent import Agent


class EvalRunner:
    """
    Executes evaluation cases against an agent and produces reports.

    Each case gets a fresh Agent instance via the agent_factory callable.
    """

    def __init__(
        self,
        agent_factory: Callable[[], Agent],
    ) -> None:
        self._agent_factory = agent_factory
        # Accumulates "case_id: regression" strings during a suite run.
        self._golden_regressions: list[str] = []

    def _handle_golden(
        self,
        case: EvalCase,
        events: list[TraceEvent],
        suite_name: str,
        golden_store: Any,
        record_golden: bool,
    ) -> str:
        """Record or replay this case's golden. Returns the golden_status.

        Pure post-hoc evidence — appends any regression strings to
        ``self._golden_regressions`` for the gate to read; it never fails the
        run itself.
        """
        if golden_store is None:
            return "skip"
        from arcana.eval.golden import build_golden, replay_diff

        if record_golden:
            golden = build_golden(case, events, suite_name=suite_name)
            golden_store.record(golden, force=True)
            return "recorded"

        golden = golden_store.load(suite_name, case.id)
        if golden is None:
            return "new"
        diff = replay_diff(golden, events, case=case)
        if diff.is_regression:
            self._golden_regressions.extend(
                f"{case.id}: {r}" for r in diff.signal_regressions
            )
        return diff.golden_status

    async def run_suite(
        self,
        cases: list[EvalCase],
        *,
        suite_name: str = "default",
        golden_dir: str | None = None,
        record_golden: bool = False,
    ) -> EvalReport:
        """
        Run all cases and aggregate into an EvalReport.

        Args:
            cases: List of evaluation cases to run.
            suite_name: Name for the evaluation suite.
            golden_dir: Optional golden-trace directory. When set, each case is
                diffed against its golden (or, with ``record_golden``, recorded).
            record_golden: When True, (re)record goldens instead of diffing.

        Returns:
            EvalReport with aggregated results.
        """
        from arcana.eval.golden import GoldenStore

        store = GoldenStore(golden_dir) if golden_dir else None
        self._golden_regressions = []

        results: list[EvalResult] = []
        for case in cases:
            result = await self.run_case(
                case,
                suite_name=suite_name,
                golden_store=store,
                record_golden=record_golden,
            )
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        # F5: merge per-case signal vectors into a suite-level vector.
        from arcana.eval.signals import merge_signals

        case_signals = [r.signals for r in results if r.signals is not None]
        aggregate_signals = merge_signals(case_signals) if case_signals else None

        return EvalReport(
            suite_name=suite_name,
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total if total > 0 else 0.0,
            results=results,
            aggregate_tokens=sum(r.tokens_used for r in results),
            aggregate_cost_usd=sum(r.cost_usd for r in results),
            aggregate_duration_ms=sum(r.duration_ms for r in results),
            aggregate_signals=aggregate_signals,
            golden_regressions=list(self._golden_regressions),
        )

    async def run_case(
        self,
        case: EvalCase,
        *,
        suite_name: str = "default",
        golden_store: Any = None,
        record_golden: bool = False,
    ) -> EvalResult:
        """
        Run a single evaluation case.

        Args:
            case: The evaluation case to run.
            suite_name: Suite name (for golden lookup).
            golden_store: Optional GoldenStore for record/replay.
            record_golden: Record the golden instead of diffing against it.

        Returns:
            EvalResult with pass/fail and metrics.
        """
        agent = self._agent_factory()
        start_ms = _now_ms()

        try:
            state = await agent.run(case.goal)

            # Collect trace events if trace_writer available
            events: list[TraceEvent] = []
            if agent.trace_writer:
                from arcana.trace.reader import TraceReader

                reader = TraceReader(trace_dir=agent.trace_writer.trace_dir)
                events = reader.read_events(state.run_id)

            # Extract metrics from events
            summary = MetricsCollector.summarize_run(events)

            # F5: trace-derived signal vector (post-hoc, pure).
            from arcana.eval.signals import extract_signals

            signals = extract_signals(events)

            # Check outcome
            passed = self._check_outcome(case, state, events)

            # F5: golden-trace record or replay (post-hoc, never gates here —
            # the gate reads report.golden_regressions and decides).
            golden_status = self._handle_golden(
                case, events, suite_name, golden_store, record_golden
            )

            duration_ms = _now_ms() - start_ms

            return EvalResult(
                case_id=case.id,
                passed=passed,
                actual_status=state.status.value,
                actual_stop_reason=summary.stop_reason,
                steps=state.current_step,
                tokens_used=state.tokens_used or summary.tokens_used,
                cost_usd=state.cost_usd or summary.cost_usd,
                duration_ms=duration_ms,
                signals=signals,
                golden_status=golden_status,
            )

        except Exception as e:
            duration_ms = _now_ms() - start_ms
            return EvalResult(
                case_id=case.id,
                passed=False,
                actual_status="error",
                steps=0,
                tokens_used=0,
                cost_usd=0.0,
                duration_ms=duration_ms,
                error=str(e),
            )

    @staticmethod
    def _check_outcome(
        case: EvalCase,
        state: AgentState,
        events: list[TraceEvent],
    ) -> bool:
        """Check if the agent outcome matches the expected criterion."""
        criterion = case.expected_outcome
        expected = case.expected_value

        if criterion == OutcomeCriterion.STATUS:
            return bool(state.status.value == expected)

        if criterion == OutcomeCriterion.STOP_REASON:
            # Find stop reason from trace events
            for event in reversed(events):
                if event.stop_reason:
                    return bool(event.stop_reason.value == expected)
            return expected is None

        if criterion == OutcomeCriterion.MAX_STEPS:
            return state.current_step <= (expected or case.max_steps)

        if criterion == OutcomeCriterion.MAX_COST:
            return state.cost_usd <= (expected or 0.0)

        if criterion == OutcomeCriterion.CONTAINS_KEYS:
            if not isinstance(expected, list):
                return False
            return all(k in state.working_memory for k in expected)

        if criterion == OutcomeCriterion.KEY_VALUES:
            if not isinstance(expected, dict):
                return False
            return all(
                state.working_memory.get(k) == v for k, v in expected.items()
            )

        return False


def _now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.monotonic() * 1000)
