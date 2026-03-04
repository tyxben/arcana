"""MetricsHook — real-time step-level metrics via RuntimeHook protocol."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from arcana.observability.metrics import RunSummary

if TYPE_CHECKING:
    from arcana.contracts.runtime import StepResult
    from arcana.contracts.state import AgentState
    from arcana.contracts.trace import TraceContext


class StepMetric:
    """Lightweight step-level metric record."""

    __slots__ = ("step_number", "step_type", "tokens_used", "duration_ms", "success")

    def __init__(
        self,
        *,
        step_number: int,
        step_type: str,
        tokens_used: int,
        duration_ms: float,
        success: bool,
    ) -> None:
        self.step_number = step_number
        self.step_type = step_type
        self.tokens_used = tokens_used
        self.duration_ms = duration_ms
        self.success = success


class MetricsHook:
    """
    RuntimeHook implementation that collects step-level metrics.

    Implements the RuntimeHook protocol from runtime/hooks/base.py.
    """

    def __init__(self) -> None:
        self.step_metrics: list[StepMetric] = []
        self.run_start_time: float | None = None
        self.run_end_time: float | None = None
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self._error_count: int = 0
        self._run_id: str | None = None

    async def on_run_start(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
    ) -> None:
        """Record run start time."""
        self.run_start_time = time.monotonic()
        self._run_id = state.run_id

    async def on_step_complete(
        self,
        state: AgentState,
        step_result: StepResult,
        trace_ctx: TraceContext,
    ) -> None:
        """Record step-level metrics."""
        tokens = 0
        if step_result.llm_response and step_result.llm_response.usage:
            tokens = step_result.llm_response.usage.total_tokens
            self.total_tokens += tokens

        metric = StepMetric(
            step_number=state.current_step,
            step_type=step_result.step_type.value,
            tokens_used=tokens,
            duration_ms=0.0,  # Duration tracked at run level
            success=step_result.success,
        )
        self.step_metrics.append(metric)

    async def on_run_end(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
    ) -> None:
        """Record run end time and final state."""
        self.run_end_time = time.monotonic()
        self.total_tokens = state.tokens_used
        self.total_cost = state.cost_usd

    async def on_error(
        self,
        state: AgentState,
        error: Exception,
        trace_ctx: TraceContext,
    ) -> None:
        """Record error occurrence."""
        self._error_count += 1

    async def on_checkpoint(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
    ) -> None:
        """No-op for checkpoints."""

    def to_summary(self) -> RunSummary:
        """Convert collected metrics to a RunSummary."""
        duration_ms = 0
        if self.run_start_time is not None and self.run_end_time is not None:
            duration_ms = int((self.run_end_time - self.run_start_time) * 1000)

        return RunSummary(
            run_id=self._run_id or "unknown",
            total_steps=len(self.step_metrics),
            llm_calls=sum(
                1 for m in self.step_metrics if m.step_type in ("think", "plan")
            ),
            tool_calls=sum(1 for m in self.step_metrics if m.step_type == "act"),
            errors=self._error_count,
            tokens_used=self.total_tokens,
            cost_usd=self.total_cost,
            duration_ms=duration_ms,
        )
