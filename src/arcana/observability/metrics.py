"""MetricsCollector — extract run metrics from trace events."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel

from arcana.contracts.trace import EventType

if TYPE_CHECKING:
    from arcana.contracts.trace import TraceEvent
    from arcana.trace.reader import TraceReader


class RunSummary(BaseModel):
    """Summary metrics for a single agent run."""

    run_id: str
    total_steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    errors: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    stop_reason: str | None = None


class AggregateMetrics(BaseModel):
    """Aggregated metrics across multiple runs."""

    count: int = 0
    avg_steps: float = 0.0
    avg_tokens: float = 0.0
    avg_cost_usd: float = 0.0
    avg_duration_ms: float = 0.0
    p50_tokens: float = 0.0
    p95_tokens: float = 0.0
    p99_tokens: float = 0.0


class MetricsCollector:
    """Extracts and aggregates metrics from trace events."""

    @staticmethod
    def summarize_run(events: list[TraceEvent]) -> RunSummary:
        """
        Extract a RunSummary from a list of trace events.

        Args:
            events: List of TraceEvent objects for a single run.

        Returns:
            RunSummary with extracted metrics.
        """
        if not events:
            return RunSummary(run_id="unknown")

        run_id = events[0].run_id
        llm_calls = 0
        tool_calls = 0
        errors = 0
        tokens_used = 0
        cost_usd = 0.0
        stop_reason: str | None = None

        step_ids: set[str] = set()

        for event in events:
            step_ids.add(event.step_id)

            if event.event_type == EventType.LLM_CALL:
                llm_calls += 1
            elif event.event_type == EventType.TOOL_CALL:
                tool_calls += 1
            elif event.event_type == EventType.ERROR:
                errors += 1

            # Track budget from latest snapshot
            if event.budgets:
                tokens_used = max(tokens_used, event.budgets.tokens_used)
                cost_usd = max(cost_usd, event.budgets.cost_usd)

            # Capture stop reason from last event that has one
            if event.stop_reason:
                stop_reason = event.stop_reason.value

        # Duration from first to last event
        duration_ms = 0
        if len(events) >= 2:
            delta = events[-1].timestamp - events[0].timestamp
            duration_ms = int(delta.total_seconds() * 1000)

        return RunSummary(
            run_id=run_id,
            total_steps=len(step_ids),
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            errors=errors,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            stop_reason=stop_reason,
        )

    @staticmethod
    def summarize_from_reader(reader: TraceReader, run_id: str) -> RunSummary:
        """
        Extract a RunSummary by reading events from a TraceReader.

        Args:
            reader: TraceReader instance.
            run_id: Run ID to read events for.

        Returns:
            RunSummary with extracted metrics.
        """
        events = reader.read_events(run_id)
        return MetricsCollector.summarize_run(events)

    @staticmethod
    def aggregate(summaries: list[RunSummary]) -> AggregateMetrics:
        """
        Aggregate multiple RunSummary instances into percentile metrics.

        Args:
            summaries: List of RunSummary objects.

        Returns:
            AggregateMetrics with averages and percentiles.
        """
        if not summaries:
            return AggregateMetrics()

        n = len(summaries)
        token_values = sorted(s.tokens_used for s in summaries)

        return AggregateMetrics(
            count=n,
            avg_steps=sum(s.total_steps for s in summaries) / n,
            avg_tokens=sum(s.tokens_used for s in summaries) / n,
            avg_cost_usd=sum(s.cost_usd for s in summaries) / n,
            avg_duration_ms=sum(s.duration_ms for s in summaries) / n,
            p50_tokens=_percentile(token_values, 50),
            p95_tokens=_percentile(token_values, 95),
            p99_tokens=_percentile(token_values, 99),
        )


def _percentile(sorted_values: list[int], p: float) -> float:
    """Calculate percentile from a sorted list using linear interpolation."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    k = (p / 100) * (len(sorted_values) - 1)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return float(sorted_values[int(k)])

    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])
