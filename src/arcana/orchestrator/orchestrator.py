"""Orchestrator — main coordinator for task scheduling and execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from arcana.contracts.orchestrator import OrchestratorConfig, TaskResult, TaskStatus
from arcana.contracts.trace import AgentRole, EventType, TraceEvent
from arcana.orchestrator.executor_pool import ExecutorPool
from arcana.orchestrator.scheduler import TaskScheduler
from arcana.orchestrator.task_graph import TaskGraph

if TYPE_CHECKING:
    from arcana.contracts.orchestrator import Task
    from arcana.gateway.budget import BudgetTracker
    from arcana.orchestrator.executor_pool import AgentFactory
    from arcana.orchestrator.hooks import OrchestratorHook
    from arcana.storage.base import StorageBackend
    from arcana.trace.writer import TraceWriter


class Orchestrator:
    """
    Main coordinator for task DAG scheduling and concurrent execution.

    Lifecycle:
    1. submit() tasks to build the DAG
    2. run() the scheduling loop
    3. Loop: schedule ready → execute concurrently → handle results → repeat
    4. Complete when all tasks finished or graph is stuck
    """

    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        config: OrchestratorConfig | None = None,
        storage: StorageBackend | None = None,
        trace_writer: TraceWriter | None = None,
        global_budget: BudgetTracker | None = None,
        hooks: list[OrchestratorHook] | None = None,
    ) -> None:
        self._config = config or OrchestratorConfig()
        self._storage = storage
        self._trace_writer = trace_writer
        self._hooks = hooks or []

        self._graph = TaskGraph()
        self._scheduler = TaskScheduler(
            self._graph,
            global_budget=global_budget,
        )
        self._pool = ExecutorPool(
            agent_factory,
            max_concurrent=self._config.max_concurrent_tasks,
        )

        self._run_id = str(uuid4())
        self._results: dict[str, TaskResult] = {}

    @property
    def run_id(self) -> str:
        """Orchestrator run ID."""
        return self._run_id

    @property
    def graph(self) -> TaskGraph:
        """The task graph."""
        return self._graph

    @property
    def results(self) -> dict[str, TaskResult]:
        """Results for completed tasks."""
        return dict(self._results)

    async def submit(self, task: Task) -> None:
        """
        Submit a task to the orchestrator.

        Adds the task to the DAG, persists state, and fires hooks.
        """
        self._graph.add_task(task)

        # Persist task state
        if self._storage:
            await self._storage.put(
                f"orchestrator:{self._run_id}",
                f"task:{task.id}",
                task.model_dump(mode="json"),
            )

        # Trace event
        self._write_trace_event(
            event_type=EventType.TASK_SUBMIT,
            metadata={"task_id": task.id, "goal": task.goal},
        )

        # Hook
        await self._call_hooks("on_task_submitted", task)

    async def run(self) -> dict[str, TaskResult]:
        """
        Run the scheduling loop until all tasks complete or graph is stuck.

        Returns:
            Dict mapping task_id → TaskResult for all completed/failed tasks.
        """
        while not self._graph.is_complete:
            # Check for stuck state
            if self._graph.is_stuck:
                break

            # Schedule: pick tasks to run
            available = self._pool.available_slots
            if available > 0:
                to_run = self._scheduler.select_tasks(available)
                for task in to_run:
                    self._graph.mark_task(task.id, TaskStatus.RUNNING)
                    task.attempt += 1

                    self._write_trace_event(
                        event_type=EventType.TASK_START,
                        metadata={"task_id": task.id, "attempt": task.attempt},
                    )
                    await self._call_hooks("on_task_started", task)

                    self._pool.submit(task)

            # Wait for at least one task to complete
            if self._pool.running_count > 0:
                completed_results = await self._pool.wait_any()
                for result in completed_results:
                    await self._handle_result(result)
            else:
                # Brief sleep to avoid busy-waiting
                await asyncio.sleep(self._config.scheduling_interval_ms / 1000)

        # Hook: orchestrator complete
        await self._call_hooks("on_orchestrator_complete")

        return dict(self._results)

    async def cancel(self) -> None:
        """Cancel all running tasks and mark pending as cancelled."""
        await self._pool.cancel_all()
        for task in self._graph.all_tasks:
            if task.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
                self._graph.mark_task(task.id, TaskStatus.CANCELLED)

    async def _handle_result(self, result: TaskResult) -> None:
        """Handle a task completion result: retry or finalize."""
        task = self._graph.get_task(result.task_id)
        if task is None:
            return

        if result.status == TaskStatus.COMPLETED:
            self._graph.mark_task(
                task.id,
                TaskStatus.COMPLETED,
                result=result.state_summary,
            )
            self._results[task.id] = result

            self._write_trace_event(
                event_type=EventType.TASK_COMPLETE,
                metadata={"task_id": task.id, "tokens_used": result.tokens_used},
            )
            await self._call_hooks("on_task_completed", task, result)

        elif result.status == TaskStatus.FAILED:
            if task.attempt < task.retry_policy.max_retries:
                # Retry with backoff
                delay_ms = min(
                    task.retry_policy.delay_ms
                    * (task.retry_policy.backoff_multiplier ** (task.attempt - 1)),
                    task.retry_policy.max_delay_ms,
                )
                await asyncio.sleep(delay_ms / 1000)

                # Reset to PENDING for re-scheduling
                self._graph.mark_task(task.id, TaskStatus.PENDING)
                await self._call_hooks("on_task_retrying", task, task.attempt + 1)
            else:
                # Final failure
                self._graph.mark_task(
                    task.id,
                    TaskStatus.FAILED,
                    error=result.error,
                )
                self._results[task.id] = result

                self._write_trace_event(
                    event_type=EventType.TASK_FAIL,
                    metadata={
                        "task_id": task.id,
                        "error": result.error,
                        "attempts": task.attempt,
                    },
                )
                await self._call_hooks("on_task_failed", task, result)

        # Persist updated task state
        if self._storage:
            await self._storage.put(
                f"orchestrator:{self._run_id}",
                f"task:{task.id}",
                task.model_dump(mode="json"),
            )

    def _write_trace_event(
        self,
        event_type: EventType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a trace event for orchestrator actions."""
        if self._trace_writer is None:
            return

        event = TraceEvent(
            run_id=self._run_id,
            role=AgentRole.SYSTEM,
            event_type=event_type,
            metadata=metadata or {},
        )
        self._trace_writer.write(event)

    async def _call_hooks(self, hook_name: str, *args: object) -> None:
        """Call all registered hooks, swallowing errors."""
        for hook in self._hooks:
            method = getattr(hook, hook_name, None)
            if method:
                try:
                    await method(*args)
                except Exception:
                    pass
