"""Tests for the runtime module."""

from __future__ import annotations

import pytest

from arcana.contracts.runtime import RuntimeConfig, StepResult, StepType
from arcana.contracts.state import AgentState, ExecutionStatus
from arcana.runtime.exceptions import StateTransitionError
from arcana.runtime.policies.react import ReActPolicy
from arcana.runtime.progress import ProgressDetector
from arcana.runtime.reducers.default import DefaultReducer
from arcana.runtime.state_manager import VALID_TRANSITIONS, StateManager


class TestRuntimeConfig:
    """Tests for RuntimeConfig."""

    def test_default_values(self) -> None:
        config = RuntimeConfig()
        assert config.max_steps == 100
        assert config.max_consecutive_errors == 3
        assert config.max_consecutive_no_progress == 3
        assert config.checkpoint_interval_steps == 5

    def test_custom_values(self) -> None:
        config = RuntimeConfig(max_steps=50, max_consecutive_errors=5)
        assert config.max_steps == 50
        assert config.max_consecutive_errors == 5


class TestStepResult:
    """Tests for StepResult."""

    def test_create_successful_step(self) -> None:
        result = StepResult(
            step_type=StepType.THINK,
            step_id="test-step-1",
            success=True,
            thought="I should do X",
            action="do_x",
        )
        assert result.success is True
        assert result.step_type == StepType.THINK
        assert result.thought == "I should do X"
        assert result.action == "do_x"

    def test_create_failed_step(self) -> None:
        result = StepResult(
            step_type=StepType.ACT,
            step_id="test-step-2",
            success=False,
            error="Something went wrong",
            is_recoverable=True,
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.is_recoverable is True


class TestStateManager:
    """Tests for StateManager."""

    def test_valid_transitions(self) -> None:
        manager = StateManager()

        # PENDING -> RUNNING
        state = AgentState(run_id="test-1")
        assert state.status == ExecutionStatus.PENDING
        state = manager.transition(state, ExecutionStatus.RUNNING)
        assert state.status == ExecutionStatus.RUNNING

        # RUNNING -> COMPLETED
        state = manager.transition(state, ExecutionStatus.COMPLETED)
        assert state.status == ExecutionStatus.COMPLETED

    def test_invalid_transition(self) -> None:
        manager = StateManager()
        state = AgentState(run_id="test-2")

        # PENDING -> COMPLETED is invalid
        with pytest.raises(StateTransitionError):
            manager.transition(state, ExecutionStatus.COMPLETED)

    def test_all_valid_transitions_defined(self) -> None:
        """Ensure all statuses have defined transitions."""
        for status in ExecutionStatus:
            assert status in VALID_TRANSITIONS


class TestProgressDetector:
    """Tests for ProgressDetector."""

    def test_initial_state_is_making_progress(self) -> None:
        detector = ProgressDetector()
        assert detector.is_making_progress() is True

    def test_detects_duplicate_steps(self) -> None:
        detector = ProgressDetector(window_size=5)

        # Add same step multiple times
        step = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=True,
            thought="Same thought",
            action="same_action",
        )

        detector.record_step(step)
        assert detector.is_making_progress() is True  # First step

        detector.record_step(step)
        assert detector.is_making_progress() is True  # Second same step

        detector.record_step(step)
        assert detector.is_making_progress() is False  # Third same step - stuck

    def test_reset(self) -> None:
        detector = ProgressDetector()
        step = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=True,
            thought="Test",
        )
        detector.record_step(step)
        detector.record_step(step)
        detector.record_step(step)

        detector.reset()
        assert detector.is_making_progress() is True


class TestReActPolicy:
    """Tests for ReActPolicy."""

    @pytest.mark.asyncio
    async def test_decide_returns_llm_call(self) -> None:
        policy = ReActPolicy()
        state = AgentState(run_id="test-1", goal="Test goal")

        decision = await policy.decide(state)

        assert decision.action_type == "llm_call"
        assert len(decision.messages) == 2
        assert decision.messages[0]["role"] == "system"
        assert "Test goal" in decision.messages[0]["content"]

    def test_policy_name(self) -> None:
        policy = ReActPolicy()
        assert policy.name == "react"


class TestDefaultReducer:
    """Tests for DefaultReducer."""

    @pytest.mark.asyncio
    async def test_reduce_updates_state(self) -> None:
        reducer = DefaultReducer()
        state = AgentState(run_id="test-1")

        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=True,
            thought="I analyzed the problem",
            action="search",
            memory_updates={"key1": "value1"},
        )

        new_state = await reducer.reduce(state, step_result)

        assert len(new_state.completed_steps) == 1
        assert new_state.working_memory["key1"] == "value1"
        assert new_state.consecutive_errors == 0

    @pytest.mark.asyncio
    async def test_reduce_tracks_errors(self) -> None:
        reducer = DefaultReducer()
        state = AgentState(run_id="test-1")

        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=False,
            error="Something failed",
        )

        new_state = await reducer.reduce(state, step_result)

        assert new_state.consecutive_errors == 1
        assert new_state.last_error == "Something failed"

    def test_reducer_name(self) -> None:
        reducer = DefaultReducer()
        assert reducer.name == "default"
