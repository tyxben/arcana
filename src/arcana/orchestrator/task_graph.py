"""TaskGraph — DAG management for task dependencies."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from arcana.contracts.orchestrator import Task, TaskStatus


class CycleError(Exception):
    """Raised when adding a task would create a dependency cycle."""

    def __init__(self, task_id: str, cycle_path: list[str]) -> None:
        super().__init__(
            f"Cycle detected involving task {task_id}: {' -> '.join(cycle_path)}"
        )
        self.task_id = task_id
        self.cycle_path = cycle_path


class TaskGraph:
    """
    DAG of tasks with dependency resolution.

    Unlike Plan.next_step() which returns a single step,
    ready_tasks() returns ALL tasks whose dependencies are satisfied,
    enabling parallel execution.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        """
        Add a task to the graph.

        Raises:
            CycleError: If adding the task would create a dependency cycle.
            ValueError: If a task with the same ID already exists, or
                        if a dependency references a non-existent task.
        """
        if task.id in self._tasks:
            msg = f"Task with ID '{task.id}' already exists"
            raise ValueError(msg)

        # Temporarily add task for cycle detection
        self._tasks[task.id] = task

        cycle = self._detect_cycle(task.id)
        if cycle:
            del self._tasks[task.id]
            raise CycleError(task.id, cycle)

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def remove_task(self, task_id: str) -> bool:
        """Remove a task. Returns True if it existed."""
        return self._tasks.pop(task_id, None) is not None

    def ready_tasks(self) -> list[Task]:
        """
        Get all tasks whose dependencies are satisfied and status is PENDING.

        A task is ready when:
        - status == PENDING
        - all dependency task IDs have status COMPLETED
        """
        completed_ids = {
            tid
            for tid, t in self._tasks.items()
            if t.status == TaskStatus.COMPLETED
        }
        return [
            t
            for t in self._tasks.values()
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.dependencies)
        ]

    def mark_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update a task's status and optional result/error."""
        task = self._tasks.get(task_id)
        if task is None:
            return

        task.status = status
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        if status == TaskStatus.RUNNING:
            task.started_at = datetime.now(UTC)
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            task.completed_at = datetime.now(UTC)

    @property
    def is_complete(self) -> bool:
        """True when all tasks are COMPLETED or CANCELLED."""
        if not self._tasks:
            return True
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
            for t in self._tasks.values()
        )

    @property
    def has_failed(self) -> bool:
        """True if any task has FAILED status."""
        return any(t.status == TaskStatus.FAILED for t in self._tasks.values())

    @property
    def is_stuck(self) -> bool:
        """
        True when no tasks are running/queued and no tasks are ready,
        but work remains incomplete.
        """
        if self.is_complete:
            return False
        running_or_queued = any(
            t.status in (TaskStatus.RUNNING, TaskStatus.QUEUED)
            for t in self._tasks.values()
        )
        return not running_or_queued and len(self.ready_tasks()) == 0

    @property
    def progress_ratio(self) -> float:
        """Ratio of completed tasks to total tasks."""
        if not self._tasks:
            return 0.0
        done = sum(
            1
            for t in self._tasks.values()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
        )
        return done / len(self._tasks)

    @property
    def all_tasks(self) -> list[Task]:
        """Return all tasks."""
        return list(self._tasks.values())

    def _detect_cycle(self, task_id: str) -> list[str] | None:
        """DFS cycle detection starting from task_id. Returns cycle path or None."""
        visited: set[str] = set()
        path: list[str] = []

        def dfs(current: str) -> list[str] | None:
            if current in visited:
                # Found cycle — extract the cycle portion
                if current in path:
                    idx = path.index(current)
                    return [*path[idx:], current]
                return None

            visited.add(current)
            path.append(current)

            task = self._tasks.get(current)
            if task:
                for dep in task.dependencies:
                    dep_task = self._tasks.get(dep)
                    if dep_task:
                        cycle = dfs(dep)
                        if cycle:
                            return cycle

            path.pop()
            return None

        return dfs(task_id)
