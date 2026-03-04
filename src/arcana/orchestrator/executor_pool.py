"""ExecutorPool — manages concurrent Agent executions."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from arcana.contracts.orchestrator import TaskResult, TaskStatus

if TYPE_CHECKING:
    from arcana.contracts.orchestrator import Task
    from arcana.runtime.agent import Agent


class AgentFactory(ABC):
    """
    Factory for creating Agent instances per task.

    Users implement this to configure Agent with appropriate
    policy, reducer, gateway, and per-task budget.
    """

    @abstractmethod
    def create_agent(self, task: Task) -> Agent:
        """Create an Agent configured for the given task."""


class ExecutorPool:
    """
    Manages concurrent Agent executions with semaphore-based control.

    Each task runs as an asyncio.Task wrapping Agent.run().
    """

    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        max_concurrent: int = 4,
    ) -> None:
        self._factory = agent_factory
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._running: dict[str, asyncio.Task[TaskResult]] = {}

    @property
    def running_count(self) -> int:
        """Number of currently running tasks."""
        return len(self._running)

    @property
    def available_slots(self) -> int:
        """Number of available concurrency slots."""
        return self._max_concurrent - len(self._running)

    def submit(self, task: Task) -> asyncio.Task[TaskResult]:
        """
        Submit a task for execution.

        Returns the asyncio.Task wrapping the execution.
        """
        async_task = asyncio.create_task(
            self._execute(task),
            name=f"task-{task.id}",
        )
        self._running[task.id] = async_task
        return async_task

    async def _execute(self, task: Task) -> TaskResult:
        """Execute a single task under semaphore control."""
        async with self._semaphore:
            start_time = datetime.now(UTC)
            agent = self._factory.create_agent(task)

            try:
                state = await agent.run(goal=task.goal, task_id=task.id)

                duration_ms = int(
                    (datetime.now(UTC) - start_time).total_seconds() * 1000
                )

                if state.status.value == "completed":
                    return TaskResult(
                        task_id=task.id,
                        status=TaskStatus.COMPLETED,
                        attempt=task.attempt,
                        tokens_used=state.tokens_used,
                        cost_usd=state.cost_usd,
                        duration_ms=duration_ms,
                        state_summary={
                            "completed_steps": state.completed_steps,
                            "working_memory_keys": list(
                                state.working_memory.keys()
                            ),
                        },
                    )
                else:
                    return TaskResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        attempt=task.attempt,
                        tokens_used=state.tokens_used,
                        cost_usd=state.cost_usd,
                        duration_ms=duration_ms,
                        error=state.last_error,
                    )

            except Exception as e:
                duration_ms = int(
                    (datetime.now(UTC) - start_time).total_seconds() * 1000
                )
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    attempt=task.attempt,
                    duration_ms=duration_ms,
                    error=str(e),
                )
            finally:
                self._running.pop(task.id, None)

    async def wait_any(self) -> list[TaskResult]:
        """
        Wait for at least one running task to complete.

        Returns list of TaskResults for completed tasks.
        """
        if not self._running:
            return []

        done, _ = await asyncio.wait(
            self._running.values(),
            return_when=asyncio.FIRST_COMPLETED,
        )

        return [f.result() for f in done]

    async def wait_all(self) -> list[TaskResult]:
        """Wait for all running tasks to complete."""
        if not self._running:
            return []

        done, _ = await asyncio.wait(self._running.values())
        return [f.result() for f in done]

    async def cancel_all(self) -> None:
        """Cancel all running tasks."""
        for async_task in self._running.values():
            async_task.cancel()
        if self._running:
            await asyncio.wait(self._running.values())
        self._running.clear()
