"""Tests for the orchestrator module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from arcana.contracts.llm import LLMResponse, TokenUsage
from arcana.contracts.orchestrator import (
    OrchestratorConfig,
    RetryPolicy,
    Task,
    TaskBudget,
    TaskResult,
    TaskStatus,
)
from arcana.gateway.budget import BudgetTracker
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.orchestrator.executor_pool import AgentFactory, ExecutorPool
from arcana.orchestrator.orchestrator import Orchestrator
from arcana.orchestrator.scheduler import TaskScheduler
from arcana.orchestrator.task_graph import CycleError, TaskGraph
from arcana.runtime.agent import Agent
from arcana.runtime.policies.react import ReActPolicy
from arcana.runtime.reducers.default import DefaultReducer

# ── Helpers ─────────────────────────────────────────────────────


def _make_task(
    task_id: str = "t1",
    goal: str = "Test goal",
    dependencies: list[str] | None = None,
    priority: int = 0,
    status: TaskStatus = TaskStatus.PENDING,
    deadline: datetime | None = None,
    budget: TaskBudget | None = None,
    retry_policy: RetryPolicy | None = None,
) -> Task:
    kwargs: dict = {
        "id": task_id,
        "goal": goal,
        "dependencies": dependencies or [],
        "priority": priority,
        "status": status,
    }
    if deadline is not None:
        kwargs["deadline"] = deadline
    if budget is not None:
        kwargs["budget"] = budget
    if retry_policy is not None:
        kwargs["retry_policy"] = retry_policy
    return Task(**kwargs)


class _MockModelGateway:
    """Mock LLM gateway that always returns FINISH."""

    default_model = "mock-model"

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    async def generate(self, request, config, trace_ctx=None):
        if self._fail:
            msg = "Mock LLM failure"
            raise RuntimeError(msg)
        return LLMResponse(
            content="Thought: Done\nAction: FINISH",
            model="mock",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class _TestAgentFactory(AgentFactory):
    """Factory that creates mock agents for testing."""

    def __init__(self, *, fail_tasks: set[str] | None = None) -> None:
        self.fail_tasks = fail_tasks or set()
        self.created: list[str] = []

    def create_agent(self, task: Task) -> Agent:
        self.created.append(task.id)
        gateway = ModelGatewayRegistry()
        gw = _MockModelGateway(fail=task.id in self.fail_tasks)
        gateway._providers["mock"] = gw
        gateway._default_provider = "mock"

        return Agent(
            policy=ReActPolicy(),
            reducer=DefaultReducer(),
            gateway=gateway,
            config=__import__(
                "arcana.contracts.runtime", fromlist=["RuntimeConfig"]
            ).RuntimeConfig(max_steps=3),
        )


class _RecordingHook:
    """Hook that records all lifecycle events."""

    def __init__(self) -> None:
        self.events: list[tuple[str, ...]] = []

    async def on_task_submitted(self, task):
        self.events.append(("submitted", task.id))

    async def on_task_started(self, task):
        self.events.append(("started", task.id))

    async def on_task_completed(self, task, result):
        self.events.append(("completed", task.id))

    async def on_task_failed(self, task, result):
        self.events.append(("failed", task.id))

    async def on_task_retrying(self, task, attempt):
        self.events.append(("retrying", task.id, str(attempt)))

    async def on_orchestrator_complete(self):
        self.events.append(("orchestrator_complete",))


# ── Contract Tests ──────────────────────────────────────────────


class TestContracts:
    """Test orchestrator data models."""

    def test_task_serialization_roundtrip(self):
        task = _make_task(priority=5, budget=TaskBudget(max_tokens=1000))
        data = task.model_dump(mode="json")
        restored = Task.model_validate(data)
        assert restored.id == task.id
        assert restored.priority == 5
        assert restored.budget is not None
        assert restored.budget.max_tokens == 1000

    def test_task_result_serialization(self):
        result = TaskResult(
            task_id="t1",
            status=TaskStatus.COMPLETED,
            attempt=1,
            tokens_used=100,
            state_summary={"key": "value"},
        )
        data = result.model_dump(mode="json")
        restored = TaskResult.model_validate(data)
        assert restored.task_id == "t1"
        assert restored.tokens_used == 100

    def test_retry_policy_defaults(self):
        policy = RetryPolicy()
        assert policy.max_retries == 0
        assert policy.delay_ms == 1000
        assert policy.backoff_multiplier == 2.0

    def test_orchestrator_config_defaults(self):
        config = OrchestratorConfig()
        assert config.max_concurrent_tasks == 4
        assert config.scheduling_interval_ms == 100
        assert config.default_retry_policy.max_retries == 0


# ── TaskGraph Tests ─────────────────────────────────────────────


class TestTaskGraph:
    """Test DAG management."""

    def test_add_and_get_task(self):
        graph = TaskGraph()
        task = _make_task("t1")
        graph.add_task(task)
        assert graph.get_task("t1") is task

    def test_add_duplicate_raises(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        with pytest.raises(ValueError, match="already exists"):
            graph.add_task(_make_task("t1"))

    def test_ready_tasks_no_dependencies(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.add_task(_make_task("t2"))
        ready = graph.ready_tasks()
        assert len(ready) == 2
        assert {t.id for t in ready} == {"t1", "t2"}

    def test_ready_tasks_with_dependencies(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.add_task(_make_task("t2", dependencies=["t1"]))
        ready = graph.ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t1"

    def test_ready_tasks_after_dependency_completed(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.add_task(_make_task("t2", dependencies=["t1"]))
        graph.mark_task("t1", TaskStatus.COMPLETED)
        ready = graph.ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t2"

    def test_ready_tasks_multiple_ready(self):
        """Core test: multiple tasks can be ready simultaneously."""
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.add_task(_make_task("t2"))
        graph.add_task(_make_task("t3", dependencies=["t1"]))
        ready = graph.ready_tasks()
        assert {t.id for t in ready} == {"t1", "t2"}

    def test_mark_task_updates_status(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.mark_task("t1", TaskStatus.RUNNING)
        assert graph.get_task("t1").status == TaskStatus.RUNNING

    def test_mark_task_sets_timestamps(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.mark_task("t1", TaskStatus.RUNNING)
        assert graph.get_task("t1").started_at is not None
        graph.mark_task("t1", TaskStatus.COMPLETED)
        assert graph.get_task("t1").completed_at is not None

    def test_is_complete(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.add_task(_make_task("t2"))
        assert not graph.is_complete
        graph.mark_task("t1", TaskStatus.COMPLETED)
        assert not graph.is_complete
        graph.mark_task("t2", TaskStatus.COMPLETED)
        assert graph.is_complete

    def test_is_complete_with_cancelled(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.mark_task("t1", TaskStatus.CANCELLED)
        assert graph.is_complete

    def test_has_failed(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        assert not graph.has_failed
        graph.mark_task("t1", TaskStatus.FAILED)
        assert graph.has_failed

    def test_is_stuck_when_deps_failed(self):
        """Graph is stuck when a dependency fails and its dependent can't run."""
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.add_task(_make_task("t2", dependencies=["t1"]))
        graph.mark_task("t1", TaskStatus.FAILED)
        assert graph.is_stuck

    def test_progress_ratio(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.add_task(_make_task("t2"))
        graph.add_task(_make_task("t3"))
        assert graph.progress_ratio == pytest.approx(0.0)
        graph.mark_task("t1", TaskStatus.COMPLETED)
        assert graph.progress_ratio == pytest.approx(1 / 3)
        graph.mark_task("t2", TaskStatus.COMPLETED)
        graph.mark_task("t3", TaskStatus.CANCELLED)
        assert graph.progress_ratio == pytest.approx(1.0)

    def test_cycle_detection_direct(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1", dependencies=["t2"]))
        with pytest.raises(CycleError):
            graph.add_task(_make_task("t2", dependencies=["t1"]))

    def test_cycle_detection_transitive(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1", dependencies=["t3"]))
        graph.add_task(_make_task("t2", dependencies=["t1"]))
        with pytest.raises(CycleError):
            graph.add_task(_make_task("t3", dependencies=["t2"]))

    def test_remove_task(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        assert graph.remove_task("t1")
        assert graph.get_task("t1") is None
        assert not graph.remove_task("nonexistent")

    def test_empty_graph_is_complete(self):
        graph = TaskGraph()
        assert graph.is_complete
        assert graph.progress_ratio == 0.0


# ── TaskScheduler Tests ─────────────────────────────────────────


class TestTaskScheduler:
    """Test priority scheduling."""

    def test_select_from_empty_graph(self):
        graph = TaskGraph()
        scheduler = TaskScheduler(graph)
        assert scheduler.select_tasks(5) == []

    def test_select_respects_max_count(self):
        graph = TaskGraph()
        graph.add_task(_make_task("t1"))
        graph.add_task(_make_task("t2"))
        graph.add_task(_make_task("t3"))
        scheduler = TaskScheduler(graph)
        selected = scheduler.select_tasks(2)
        assert len(selected) == 2

    def test_select_priority_ordering(self):
        graph = TaskGraph()
        graph.add_task(_make_task("low", priority=1))
        graph.add_task(_make_task("high", priority=10))
        graph.add_task(_make_task("mid", priority=5))
        scheduler = TaskScheduler(graph)
        selected = scheduler.select_tasks(3)
        assert selected[0].id == "high"
        assert selected[1].id == "mid"
        assert selected[2].id == "low"

    def test_select_deadline_ordering(self):
        now = datetime.now(UTC)
        graph = TaskGraph()
        graph.add_task(_make_task("later", deadline=now + timedelta(hours=2)))
        graph.add_task(_make_task("sooner", deadline=now + timedelta(hours=1)))
        graph.add_task(_make_task("no_deadline"))
        scheduler = TaskScheduler(graph)
        selected = scheduler.select_tasks(3)
        assert selected[0].id == "sooner"
        assert selected[1].id == "later"
        assert selected[2].id == "no_deadline"

    def test_admission_control_with_budget(self):
        from arcana.contracts.llm import Budget

        graph = TaskGraph()
        graph.add_task(
            _make_task("t1", budget=TaskBudget(max_tokens=5000))
        )
        graph.add_task(
            _make_task("t2", budget=TaskBudget(max_tokens=100))
        )

        tracker = BudgetTracker.from_budget(Budget(max_tokens=200))
        scheduler = TaskScheduler(graph, global_budget=tracker)
        selected = scheduler.select_tasks(5)
        # t1 needs 5000 tokens but budget only has 200 → rejected
        # t2 needs 100 tokens → admitted
        assert len(selected) == 1
        assert selected[0].id == "t2"


# ── ExecutorPool Tests ──────────────────────────────────────────


class TestExecutorPool:
    """Test concurrent execution management."""

    @pytest.mark.asyncio
    async def test_submit_and_wait_any(self):
        factory = _TestAgentFactory()
        pool = ExecutorPool(factory, max_concurrent=4)
        task = _make_task("t1")
        task.attempt = 1
        pool.submit(task)
        assert pool.running_count == 1

        results = await pool.wait_any()
        assert len(results) == 1
        assert results[0].task_id == "t1"

    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """Verify semaphore limits concurrency."""
        factory = _TestAgentFactory()
        pool = ExecutorPool(factory, max_concurrent=2)

        tasks = [_make_task(f"t{i}") for i in range(4)]
        for t in tasks:
            t.attempt = 1
            pool.submit(t)

        assert pool.running_count == 4  # All submitted as asyncio tasks
        assert pool.available_slots == -2  # Over-subscribed in tasks, semaphore limits actual execution

        results = await pool.wait_all()
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_failed_task_returns_result(self):
        factory = _TestAgentFactory(fail_tasks={"t1"})
        pool = ExecutorPool(factory, max_concurrent=4)
        task = _make_task("t1")
        task.attempt = 1
        pool.submit(task)

        results = await pool.wait_any()
        assert len(results) == 1
        assert results[0].status == TaskStatus.FAILED
        assert results[0].error is not None

    @pytest.mark.asyncio
    async def test_cancel_all(self):
        factory = _TestAgentFactory()
        pool = ExecutorPool(factory, max_concurrent=4)
        task = _make_task("t1")
        task.attempt = 1
        pool.submit(task)
        await pool.cancel_all()
        assert pool.running_count == 0


# ── Orchestrator Integration Tests ──────────────────────────────


class TestOrchestrator:
    """Test the main coordinator."""

    @pytest.mark.asyncio
    async def test_single_task_execution(self):
        factory = _TestAgentFactory()
        orch = Orchestrator(factory)
        await orch.submit(_make_task("t1"))
        results = await orch.run()
        assert "t1" in results
        assert results["t1"].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dag_dependency_order(self):
        """Tasks with dependencies execute after their prerequisites."""
        factory = _TestAgentFactory()
        orch = Orchestrator(factory)
        await orch.submit(_make_task("t1"))
        await orch.submit(_make_task("t2", dependencies=["t1"]))

        results = await orch.run()
        assert results["t1"].status == TaskStatus.COMPLETED
        assert results["t2"].status == TaskStatus.COMPLETED
        # t1 should have been created before t2
        assert factory.created.index("t1") < factory.created.index("t2")

    @pytest.mark.asyncio
    async def test_parallel_independent_tasks(self):
        """Independent tasks can run in parallel."""
        factory = _TestAgentFactory()
        orch = Orchestrator(
            factory,
            config=OrchestratorConfig(max_concurrent_tasks=4),
        )
        await orch.submit(_make_task("t1"))
        await orch.submit(_make_task("t2"))
        await orch.submit(_make_task("t3"))

        results = await orch.run()
        assert len(results) == 3
        assert all(r.status == TaskStatus.COMPLETED for r in results.values())

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Failed tasks are retried according to retry policy."""
        # Factory that fails first attempt but succeeds on retry
        # (Since mock always fails for fail_tasks, we test final failure path instead)
        factory = _TestAgentFactory(fail_tasks={"t1"})
        orch = Orchestrator(factory)
        await orch.submit(
            _make_task(
                "t1",
                retry_policy=RetryPolicy(max_retries=2, delay_ms=1),
            )
        )
        results = await orch.run()
        assert "t1" in results
        # Task should have been attempted multiple times
        task = orch.graph.get_task("t1")
        assert task.attempt == 2  # Tried twice (max_retries=2)
        assert results["t1"].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_retry_exhausted_marks_failed(self):
        factory = _TestAgentFactory(fail_tasks={"t1"})
        orch = Orchestrator(factory)
        await orch.submit(
            _make_task("t1", retry_policy=RetryPolicy(max_retries=1, delay_ms=1))
        )
        results = await orch.run()
        assert results["t1"].status == TaskStatus.FAILED
        assert orch.graph.get_task("t1").status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_hooks_called(self):
        hook = _RecordingHook()
        factory = _TestAgentFactory()
        orch = Orchestrator(factory, hooks=[hook])
        await orch.submit(_make_task("t1"))
        await orch.run()

        event_types = [e[0] for e in hook.events]
        assert "submitted" in event_types
        assert "started" in event_types
        assert "completed" in event_types
        assert "orchestrator_complete" in event_types

    @pytest.mark.asyncio
    async def test_cancel_orchestrator(self):
        factory = _TestAgentFactory()
        orch = Orchestrator(factory)
        await orch.submit(_make_task("t1"))
        await orch.submit(_make_task("t2"))
        await orch.cancel()
        # Pending tasks should be cancelled
        assert orch.graph.get_task("t1").status == TaskStatus.CANCELLED
        assert orch.graph.get_task("t2").status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_stuck_detection(self):
        """Orchestrator detects when graph is stuck (dep failed)."""
        factory = _TestAgentFactory(fail_tasks={"t1"})
        orch = Orchestrator(factory)
        await orch.submit(_make_task("t1"))
        await orch.submit(_make_task("t2", dependencies=["t1"]))
        results = await orch.run()
        # t1 failed, t2 can never run → stuck
        assert "t1" in results
        assert results["t1"].status == TaskStatus.FAILED
        assert "t2" not in results  # Never executed
        assert orch.graph.is_stuck


# ── Module Export Tests ─────────────────────────────────────────


class TestModuleExports:
    """Test that all public APIs are exported."""

    def test_exports(self):
        from arcana.orchestrator import (
            CycleError,
            Orchestrator,
            Task,
            TaskGraph,
        )

        assert Task is not None
        assert Orchestrator is not None
        assert TaskGraph is not None
        assert CycleError is not None
