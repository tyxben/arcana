"""MetricsHook — real-time step-level metrics via RuntimeHook protocol."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from arcana.observability.metrics import RunSummary

if TYPE_CHECKING:
    from arcana.contracts.runtime import StepResult
    from arcana.contracts.state import AgentState
    from arcana.contracts.trace import TraceContext

logger = logging.getLogger(__name__)


class StepMetric:
    """Lightweight step-level metric record."""

    __slots__ = ("step_number", "step_type", "tokens_used", "duration_ms", "success",
                 "provider_name")

    def __init__(
        self,
        *,
        step_number: int,
        step_type: str,
        tokens_used: int,
        duration_ms: float,
        success: bool,
        provider_name: str | None = None,
    ) -> None:
        self.step_number = step_number
        self.step_type = step_type
        self.tokens_used = tokens_used
        self.duration_ms = duration_ms
        self.success = success
        self.provider_name = provider_name


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
        self._provider_name: str | None = None
        self._model_name: str | None = None

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
        if step_result.llm_response:
            model = step_result.llm_response.model
            if model and self._model_name is None:
                self._model_name = model
            # Infer provider from model name (e.g. "deepseek-chat" → "deepseek")
            if model and self._provider_name is None:
                self._provider_name = model.split("-")[0] if "-" in model else model

        metric = StepMetric(
            step_number=state.current_step,
            step_type=step_result.step_type.value,
            tokens_used=tokens,
            duration_ms=0.0,  # Duration tracked at run level
            success=step_result.success,
            provider_name=self._provider_name,
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
            success=self._error_count == 0,
            provider_name=self._provider_name,
            model_name=self._model_name,
        )


class BudgetWarningHook:
    """Emits log warnings when budget usage crosses configurable thresholds.

    Usage::

        hook = BudgetWarningHook(
            max_tokens=100_000, max_cost_usd=1.0,
            warn_at_pct=0.8, critical_at_pct=0.95,
        )
        # Register as a RuntimeHook — on_step_complete checks budget.
    """

    def __init__(
        self,
        warn_at_pct: float = 0.8,
        critical_at_pct: float = 0.95,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        self._warn_at = warn_at_pct
        self._critical_at = critical_at_pct
        self._max_tokens = max_tokens
        self._max_cost_usd = max_cost_usd
        # Per-resource threshold flags — prevent cross-resource suppression
        self._fired: dict[str, set[str]] = {}  # resource -> {"warn", "critical"}

    def _check_budget(self, state: AgentState) -> None:
        """Check budget usage against thresholds."""
        if self._max_tokens and self._max_tokens > 0:
            pct = state.tokens_used / self._max_tokens
            self._emit_warnings(pct, "tokens", state.tokens_used, self._max_tokens)

        if self._max_cost_usd and self._max_cost_usd > 0:
            pct = state.cost_usd / self._max_cost_usd
            self._emit_warnings(pct, "cost", state.cost_usd, self._max_cost_usd)

    def _emit_warnings(
        self, pct: float, resource: str, used: float, limit: float
    ) -> None:
        fired = self._fired.setdefault(resource, set())
        if "critical" not in fired and pct >= self._critical_at:
            logger.warning(
                "CRITICAL: %s budget at %.0f%% (%s / %s)",
                resource, pct * 100, used, limit,
            )
            fired.add("critical")
            fired.add("warn")  # skip warn if critical already fired
        elif "warn" not in fired and pct >= self._warn_at:
            logger.warning(
                "WARNING: %s budget at %.0f%% (%s / %s)",
                resource, pct * 100, used, limit,
            )
            fired.add("warn")

    # -- RuntimeHook protocol --

    async def on_run_start(
        self, state: AgentState, trace_ctx: TraceContext
    ) -> None:
        self._fired = {}

    async def on_step_complete(
        self,
        state: AgentState,
        step_result: StepResult,
        trace_ctx: TraceContext,
    ) -> None:
        self._check_budget(state)

    async def on_run_end(
        self, state: AgentState, trace_ctx: TraceContext
    ) -> None:
        pass

    async def on_error(
        self, state: AgentState, error: Exception, trace_ctx: TraceContext
    ) -> None:
        pass

    async def on_checkpoint(
        self, state: AgentState, trace_ctx: TraceContext
    ) -> None:
        pass
