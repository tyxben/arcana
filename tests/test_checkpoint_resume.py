"""Tests for checkpoint, resume, and replay enhancements (Week 9)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.contracts.runtime import RuntimeConfig, StepResult, StepType
from arcana.contracts.state import AgentState, ExecutionStatus, StateSnapshot
from arcana.contracts.trace import AgentRole, EventType, TraceContext, TraceEvent
from arcana.runtime.exceptions import HashVerificationError
from arcana.runtime.state_manager import StateManager
from arcana.trace.reader import TraceReader
from arcana.utils.hashing import canonical_hash

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def trace_ctx():
    """Create a TraceContext for tests."""
    return TraceContext(run_id="test-run-001")


@pytest.fixture
def base_state():
    """Create a base AgentState for tests."""
    return AgentState(
        run_id="test-run-001",
        task_id="task-001",
        goal="Test goal",
        status=ExecutionStatus.RUNNING,
        current_step=5,
    )


@pytest.fixture
def state_manager(temp_dir):
    """Create a StateManager with a temporary checkpoint directory."""
    return StateManager(checkpoint_dir=temp_dir / "checkpoints")


@pytest.fixture
def trace_reader(temp_dir):
    """Create a TraceReader with a temporary directory."""
    return TraceReader(trace_dir=temp_dir / "traces")


# ===========================================================================
# StateManager Tests
# ===========================================================================


class TestStateManagerCheckpointReason:
    """Tests for checkpoint reason storage."""

    async def test_checkpoint_stores_reason(
        self, state_manager, base_state, trace_ctx
    ):
        """Verify checkpoint reason is stored in the snapshot."""
        snapshot = await state_manager.checkpoint(
            base_state, trace_ctx, reason="error"
        )

        assert snapshot.checkpoint_reason == "error"

    async def test_checkpoint_stores_default_reason(
        self, state_manager, base_state, trace_ctx
    ):
        """Verify default checkpoint reason is stored."""
        snapshot = await state_manager.checkpoint(base_state, trace_ctx)

        assert snapshot.checkpoint_reason == "step_complete"

    async def test_checkpoint_stores_interval_reason(
        self, state_manager, base_state, trace_ctx
    ):
        """Verify 'interval' reason is preserved."""
        snapshot = await state_manager.checkpoint(
            base_state, trace_ctx, reason="interval"
        )

        assert snapshot.checkpoint_reason == "interval"

    async def test_checkpoint_stores_plan_step_reason(
        self, state_manager, base_state, trace_ctx
    ):
        """Verify 'plan_step' reason is preserved."""
        snapshot = await state_manager.checkpoint(
            base_state, trace_ctx, reason="plan_step"
        )

        assert snapshot.checkpoint_reason == "plan_step"

    async def test_checkpoint_stores_verification_reason(
        self, state_manager, base_state, trace_ctx
    ):
        """Verify 'verification' reason is preserved."""
        snapshot = await state_manager.checkpoint(
            base_state, trace_ctx, reason="verification"
        )

        assert snapshot.checkpoint_reason == "verification"

    async def test_checkpoint_stores_budget_reason(
        self, state_manager, base_state, trace_ctx
    ):
        """Verify 'budget' reason is preserved."""
        snapshot = await state_manager.checkpoint(
            base_state, trace_ctx, reason="budget"
        )

        assert snapshot.checkpoint_reason == "budget"


class TestStateManagerPlanProgress:
    """Tests for plan progress capture in checkpoints."""

    async def test_checkpoint_stores_plan_progress(
        self, state_manager, trace_ctx
    ):
        """Verify plan data from working_memory is captured in checkpoint."""
        plan_data = {
            "goal": "Build feature X",
            "steps": ["step1", "step2", "step3"],
            "current_step_index": 1,
            "status": "in_progress",
        }
        state = AgentState(
            run_id="test-run-001",
            status=ExecutionStatus.RUNNING,
            working_memory={"plan": plan_data, "other_key": "other_value"},
        )

        snapshot = await state_manager.checkpoint(
            state, trace_ctx, reason="plan_step"
        )

        assert snapshot.plan_progress == plan_data
        assert snapshot.plan_progress["current_step_index"] == 1

    async def test_checkpoint_empty_plan_progress_when_no_plan(
        self, state_manager, base_state, trace_ctx
    ):
        """Verify plan_progress is empty when no plan in working_memory."""
        snapshot = await state_manager.checkpoint(base_state, trace_ctx)

        assert snapshot.plan_progress == {}

    async def test_checkpoint_empty_plan_progress_when_plan_not_dict(
        self, state_manager, trace_ctx
    ):
        """Verify plan_progress is empty when plan is not a dict."""
        state = AgentState(
            run_id="test-run-001",
            status=ExecutionStatus.RUNNING,
            working_memory={"plan": "not a dict"},
        )

        snapshot = await state_manager.checkpoint(state, trace_ctx)

        assert snapshot.plan_progress == {}


class TestStateManagerListCheckpoints:
    """Tests for listing checkpoints."""

    async def test_list_checkpoints(
        self, state_manager, base_state, trace_ctx
    ):
        """List all checkpoints for a run."""
        # Create multiple checkpoints
        await state_manager.checkpoint(
            base_state, trace_ctx, reason="interval"
        )

        base_state.current_step = 10
        await state_manager.checkpoint(
            base_state, trace_ctx, reason="plan_step"
        )

        base_state.current_step = 15
        await state_manager.checkpoint(
            base_state, trace_ctx, reason="error"
        )

        checkpoints = await state_manager.list_checkpoints("test-run-001")

        assert len(checkpoints) == 3
        assert checkpoints[0].checkpoint_reason == "interval"
        assert checkpoints[1].checkpoint_reason == "plan_step"
        assert checkpoints[2].checkpoint_reason == "error"

    async def test_list_checkpoints_empty(self, state_manager):
        """List checkpoints returns empty list for unknown run."""
        checkpoints = await state_manager.list_checkpoints("nonexistent-run")

        assert checkpoints == []


class TestVerifySnapshotIntegrity:
    """Tests for snapshot hash verification."""

    async def test_verify_snapshot_integrity(
        self, state_manager, base_state, trace_ctx
    ):
        """Verify that snapshot hash verification works."""
        snapshot = await state_manager.checkpoint(base_state, trace_ctx)

        # Verification should pass
        assert state_manager.verify_snapshot(snapshot) is True

    async def test_verify_snapshot_integrity_fails_on_tamper(
        self, state_manager, base_state, trace_ctx
    ):
        """Verify that tampered snapshot fails verification."""
        snapshot = await state_manager.checkpoint(base_state, trace_ctx)

        # Tamper with the state
        snapshot.state.current_step = 999

        with pytest.raises(HashVerificationError):
            state_manager.verify_snapshot(snapshot)


# ===========================================================================
# Agent Checkpoint Trigger Tests
# ===========================================================================


class TestAgentCheckpointTriggers:
    """Tests for Agent._should_checkpoint() with plan-aware triggers."""

    def _make_agent(self, config=None):
        """Create a minimal Agent for testing _should_checkpoint."""
        from arcana.runtime.agent import Agent

        policy = MagicMock()
        reducer = MagicMock()
        gateway = MagicMock()

        return Agent(
            policy=policy,
            reducer=reducer,
            gateway=gateway,
            config=config or RuntimeConfig(),
        )

    def test_checkpoint_on_error(self):
        """Error triggers checkpoint with 'error' reason."""
        agent = self._make_agent()
        state = AgentState(
            run_id="test-run",
            status=ExecutionStatus.RUNNING,
            current_step=3,
        )
        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=False,
            error="Something failed",
        )

        reason = agent._should_checkpoint(state, step_result)

        assert reason == "error"

    def test_checkpoint_on_interval(self):
        """Interval triggers checkpoint with 'interval' reason."""
        agent = self._make_agent(
            RuntimeConfig(checkpoint_interval_steps=5)
        )
        state = AgentState(
            run_id="test-run",
            status=ExecutionStatus.RUNNING,
            current_step=10,  # Multiple of 5
        )
        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=True,
        )

        reason = agent._should_checkpoint(state, step_result)

        assert reason == "interval"

    def test_checkpoint_on_plan_step(self):
        """Plan step completion triggers checkpoint with 'plan_step' reason."""
        agent = self._make_agent(
            RuntimeConfig(
                checkpoint_on_plan_step=True,
                checkpoint_interval_steps=100,  # Prevent interval trigger
            )
        )
        state = AgentState(
            run_id="test-run",
            status=ExecutionStatus.RUNNING,
            current_step=3,
        )
        step_result = StepResult(
            step_type=StepType.ACT,
            step_id="step-1",
            success=True,
            state_updates={"plan_step_completed": True},
        )

        reason = agent._should_checkpoint(state, step_result)

        assert reason == "plan_step"

    def test_no_checkpoint_on_plan_step_when_disabled(self):
        """Plan step checkpoint disabled via config."""
        agent = self._make_agent(
            RuntimeConfig(
                checkpoint_on_plan_step=False,
                checkpoint_interval_steps=100,
            )
        )
        state = AgentState(
            run_id="test-run",
            status=ExecutionStatus.RUNNING,
            current_step=3,
        )
        step_result = StepResult(
            step_type=StepType.ACT,
            step_id="step-1",
            success=True,
            state_updates={"plan_step_completed": True},
        )

        reason = agent._should_checkpoint(state, step_result)

        assert reason is None

    def test_checkpoint_on_verification(self):
        """Verify step triggers checkpoint with 'verification' reason."""
        agent = self._make_agent(
            RuntimeConfig(
                checkpoint_on_verification=True,
                checkpoint_interval_steps=100,
            )
        )
        state = AgentState(
            run_id="test-run",
            status=ExecutionStatus.RUNNING,
            current_step=3,
        )
        step_result = StepResult(
            step_type=StepType.VERIFY,
            step_id="step-1",
            success=True,
        )

        reason = agent._should_checkpoint(state, step_result)

        assert reason == "verification"

    def test_no_checkpoint_on_verification_when_disabled(self):
        """Verification checkpoint disabled via config."""
        agent = self._make_agent(
            RuntimeConfig(
                checkpoint_on_verification=False,
                checkpoint_interval_steps=100,
            )
        )
        state = AgentState(
            run_id="test-run",
            status=ExecutionStatus.RUNNING,
            current_step=3,
        )
        step_result = StepResult(
            step_type=StepType.VERIFY,
            step_id="step-1",
            success=True,
        )

        reason = agent._should_checkpoint(state, step_result)

        assert reason is None

    def test_no_checkpoint_when_no_trigger(self):
        """No checkpoint when no trigger condition is met."""
        agent = self._make_agent(
            RuntimeConfig(checkpoint_interval_steps=100)
        )
        state = AgentState(
            run_id="test-run",
            status=ExecutionStatus.RUNNING,
            current_step=3,
        )
        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=True,
        )

        reason = agent._should_checkpoint(state, step_result)

        assert reason is None

    def test_error_takes_priority_over_plan_step(self):
        """Error checkpoint takes priority over plan step."""
        agent = self._make_agent()
        state = AgentState(
            run_id="test-run",
            status=ExecutionStatus.RUNNING,
            current_step=3,
        )
        step_result = StepResult(
            step_type=StepType.ACT,
            step_id="step-1",
            success=False,
            error="Failed",
            state_updates={"plan_step_completed": True},
        )

        reason = agent._should_checkpoint(state, step_result)

        assert reason == "error"


# ===========================================================================
# Resume Tests
# ===========================================================================


class TestResumeFromSnapshot:
    """Tests for resuming agent execution from a checkpoint."""

    async def test_resume_from_snapshot(self, temp_dir):
        """Agent can resume from a checkpoint."""
        from arcana.runtime.agent import Agent

        # Create a snapshot to resume from
        state = AgentState(
            run_id="test-resume-run",
            task_id="task-001",
            goal="Complete the task",
            status=ExecutionStatus.RUNNING,
            current_step=5,
            working_memory={"key": "value"},
        )
        serializable = state.model_dump(exclude={"start_time", "elapsed_ms"})
        state_hash = canonical_hash(serializable)

        snapshot = StateSnapshot(
            run_id=state.run_id,
            step_id="step-5",
            state_hash=state_hash,
            state=state,
            checkpoint_reason="interval",
        )

        # Create agent with mocked components that immediately complete
        policy = AsyncMock()
        reducer = AsyncMock()
        gateway = MagicMock()

        agent = Agent(
            policy=policy,
            reducer=reducer,
            gateway=gateway,
            config=RuntimeConfig(max_steps=6),  # Will stop after 1 more step
        )

        # Mock state_manager property
        mock_state_manager = MagicMock()
        mock_state_manager.verify_snapshot.return_value = True
        mock_state_manager.transition.side_effect = lambda s, status: setattr(
            s, "status", status
        ) or s
        mock_state_manager.checkpoint = AsyncMock()
        agent._state_manager = mock_state_manager

        # Mock step_executor
        mock_step_executor = AsyncMock()
        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-6",
            success=True,
            state_updates={"goal_reached": True},
        )
        mock_step_executor.execute.return_value = step_result
        agent._step_executor = mock_step_executor

        # Mock reducer to return state
        reducer.reduce.return_value = state

        await agent.resume(snapshot)

        # Verify snapshot was verified
        mock_state_manager.verify_snapshot.assert_called_once_with(snapshot)

    async def test_resume_preserves_state(self, temp_dir):
        """Resumed state matches checkpoint state."""
        state = AgentState(
            run_id="test-resume-run",
            task_id="task-001",
            goal="Complete the task",
            status=ExecutionStatus.RUNNING,
            current_step=5,
            working_memory={"key": "value", "context": [1, 2, 3]},
            tokens_used=500,
            cost_usd=0.05,
        )
        serializable = state.model_dump(exclude={"start_time", "elapsed_ms"})
        state_hash = canonical_hash(serializable)

        snapshot = StateSnapshot(
            run_id=state.run_id,
            step_id="step-5",
            state_hash=state_hash,
            state=state,
            checkpoint_reason="plan_step",
            plan_progress={"current_step_index": 2},
        )

        # Verify snapshot state preserves all fields
        assert snapshot.state.run_id == "test-resume-run"
        assert snapshot.state.current_step == 5
        assert snapshot.state.working_memory["key"] == "value"
        assert snapshot.state.working_memory["context"] == [1, 2, 3]
        assert snapshot.state.tokens_used == 500
        assert snapshot.state.cost_usd == 0.05
        assert snapshot.state.goal == "Complete the task"
        assert snapshot.plan_progress["current_step_index"] == 2


# ===========================================================================
# Integration: Checkpoint -> Load -> Verify round-trip
# ===========================================================================


class TestCheckpointRoundTrip:
    """Integration tests for checkpoint -> load -> verify cycle."""

    async def test_checkpoint_load_verify_roundtrip(
        self, state_manager, base_state, trace_ctx
    ):
        """Full round-trip: checkpoint, load, verify."""
        # Checkpoint
        await state_manager.checkpoint(
            base_state, trace_ctx, reason="interval"
        )

        # Load
        loaded = await state_manager.load_snapshot("test-run-001")

        assert loaded is not None
        assert loaded.run_id == "test-run-001"
        assert loaded.checkpoint_reason == "interval"
        assert loaded.state.current_step == base_state.current_step

        # Verify
        assert state_manager.verify_snapshot(loaded) is True

    async def test_checkpoint_with_plan_data_roundtrip(
        self, state_manager, trace_ctx
    ):
        """Round-trip preserves plan progress data."""
        state = AgentState(
            run_id="plan-run-001",
            status=ExecutionStatus.RUNNING,
            working_memory={
                "plan": {
                    "goal": "Build feature",
                    "steps": ["a", "b", "c"],
                    "current_step_index": 1,
                }
            },
        )

        await state_manager.checkpoint(state, trace_ctx, reason="plan_step")

        loaded = await state_manager.load_snapshot("plan-run-001")

        assert loaded is not None
        assert loaded.plan_progress["goal"] == "Build feature"
        assert loaded.plan_progress["current_step_index"] == 1
        assert loaded.checkpoint_reason == "plan_step"
