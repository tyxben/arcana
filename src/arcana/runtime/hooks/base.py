"""Hook protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from arcana.contracts.runtime import StepResult
    from arcana.contracts.state import AgentState
    from arcana.contracts.trace import TraceContext


@runtime_checkable
class RuntimeHook(Protocol):
    """
    Protocol for runtime hooks.

    Hooks are called at various points in the execution lifecycle.
    All methods are optional - implement only what you need.
    """

    async def on_run_start(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
    ) -> None:
        """Called when a run starts."""
        ...

    async def on_run_end(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
    ) -> None:
        """Called when a run ends."""
        ...

    async def on_step_complete(
        self,
        state: AgentState,
        step_result: StepResult,
        trace_ctx: TraceContext,
    ) -> None:
        """Called after each step completes."""
        ...

    async def on_checkpoint(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
    ) -> None:
        """Called when a checkpoint is created."""
        ...

    async def on_error(
        self,
        state: AgentState,
        error: Exception,
        trace_ctx: TraceContext,
    ) -> None:
        """Called when an error occurs."""
        ...
