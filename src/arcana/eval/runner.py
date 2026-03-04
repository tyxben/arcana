"""EvalRunner — execute evaluation cases and produce reports."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

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

    async def run_suite(
        self,
        cases: list[EvalCase],
        *,
        suite_name: str = "default",
    ) -> EvalReport:
        """
        Run all cases and aggregate into an EvalReport.

        Args:
            cases: List of evaluation cases to run.
            suite_name: Name for the evaluation suite.

        Returns:
            EvalReport with aggregated results.
        """
        results: list[EvalResult] = []
        for case in cases:
            result = await self.run_case(case)
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        total = len(results)

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
        )

    async def run_case(self, case: EvalCase) -> EvalResult:
        """
        Run a single evaluation case.

        Args:
            case: The evaluation case to run.

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

            # Check outcome
            passed = self._check_outcome(case, state, events)

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
            return state.status.value == expected

        if criterion == OutcomeCriterion.STOP_REASON:
            # Find stop reason from trace events
            for event in reversed(events):
                if event.stop_reason:
                    return event.stop_reason.value == expected
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
