"""TaskScheduler — priority-based scheduling with admission control."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcana.contracts.orchestrator import Task
    from arcana.gateway.budget import BudgetTracker
    from arcana.orchestrator.task_graph import TaskGraph

_MAX_DATETIME = datetime.max.replace(tzinfo=UTC)


class TaskScheduler:
    """
    Selects next tasks to run from ready tasks in the graph.

    Selection criteria (in order):
    1. Dependencies satisfied (enforced by TaskGraph.ready_tasks())
    2. Budget available (admission control via global BudgetTracker)
    3. Deadline urgency (tasks with earlier deadlines first)
    4. Priority (higher priority value first)
    5. Creation order (FIFO tiebreaker)
    """

    def __init__(
        self,
        task_graph: TaskGraph,
        *,
        global_budget: BudgetTracker | None = None,
    ) -> None:
        self._graph = task_graph
        self._global_budget = global_budget

    def select_tasks(self, max_count: int) -> list[Task]:
        """
        Select up to max_count tasks to run.

        Args:
            max_count: Maximum number of tasks to return.

        Returns:
            List of tasks to execute, sorted by scheduling priority.
        """
        ready = self._graph.ready_tasks()
        if not ready:
            return []

        # Admission control: filter by global budget
        admissible = [t for t in ready if self._can_admit(t)]

        # Sort: deadline ASC (None last), priority DESC, created_at ASC
        admissible.sort(
            key=lambda t: (
                t.deadline if t.deadline is not None else _MAX_DATETIME,
                -t.priority,
                t.created_at,
            )
        )

        return admissible[:max_count]

    def _can_admit(self, task: Task) -> bool:
        """Check if the global budget can accommodate this task."""
        if self._global_budget is None:
            return True
        if task.budget and task.budget.max_tokens:
            return self._global_budget.can_afford(task.budget.max_tokens)
        return True
