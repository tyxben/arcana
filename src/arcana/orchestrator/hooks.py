"""Orchestrator hook protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from arcana.contracts.orchestrator import Task, TaskResult


@runtime_checkable
class OrchestratorHook(Protocol):
    """
    Protocol for orchestrator lifecycle hooks.

    All methods are optional — implement only what you need.
    Hooks are error-tolerant: exceptions are caught and logged.
    """

    async def on_task_submitted(self, task: Task) -> None:
        """Called when a task is submitted to the orchestrator."""
        ...

    async def on_task_started(self, task: Task) -> None:
        """Called when a task begins execution."""
        ...

    async def on_task_completed(self, task: Task, result: TaskResult) -> None:
        """Called when a task completes successfully."""
        ...

    async def on_task_failed(self, task: Task, result: TaskResult) -> None:
        """Called when a task fails (after retries exhausted)."""
        ...

    async def on_task_retrying(self, task: Task, attempt: int) -> None:
        """Called when a task is about to be retried."""
        ...

    async def on_orchestrator_complete(self) -> None:
        """Called when all tasks are finished."""
        ...
