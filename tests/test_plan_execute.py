"""Tests for Plan-and-Execute: contracts, policy, verifier, reducer, and integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from arcana.contracts.plan import (
    Plan,
    PlanStep,
    PlanStepStatus,
    VerificationOutcome,
)
from arcana.contracts.runtime import PolicyDecision, StepResult, StepType
from arcana.contracts.state import AgentState
from arcana.contracts.trace import TraceContext
from arcana.runtime.policies.plan_execute import PlanExecutePolicy
from arcana.runtime.reducers.plan_reducer import PlanReducer
from arcana.runtime.step import StepExecutor
from arcana.runtime.verifiers.goal_verifier import GoalVerifier

# ── Helpers ──────────────────────────────────────────────────────


def _make_plan(
    *,
    steps: list[PlanStep] | None = None,
    goal: str = "Test goal",
    acceptance_criteria: list[str] | None = None,
) -> Plan:
    """Create a Plan for testing."""
    if steps is None:
        steps = [
            PlanStep(
                id="step_1",
                description="Research the topic",
                acceptance_criteria=["Topic researched"],
            ),
            PlanStep(
                id="step_2",
                description="Write the report",
                acceptance_criteria=["Report written"],
                dependencies=["step_1"],
            ),
            PlanStep(
                id="step_3",
                description="Review and finalize",
                acceptance_criteria=["Report finalized"],
                dependencies=["step_2"],
            ),
        ]
    return Plan(
        steps=steps,
        goal=goal,
        acceptance_criteria=acceptance_criteria or ["Goal achieved successfully"],
    )


def _make_state(
    *,
    goal: str = "Test goal",
    plan: Plan | None = None,
    completed_steps: list[str] | None = None,
) -> AgentState:
    """Create an AgentState for testing."""
    state = AgentState(
        run_id="test-run-1",
        goal=goal,
        completed_steps=completed_steps or [],
    )
    if plan is not None:
        state.working_memory["plan"] = plan.model_dump(mode="json")
    return state


# ── Plan Contract Tests ──────────────────────────────────────────


class TestPlanContract:
    """Tests for Plan, PlanStep, and related contracts."""

    def test_next_step_returns_first_pending(self) -> None:
        plan = _make_plan()
        step = plan.next_step()
        assert step is not None
        assert step.id == "step_1"

    def test_next_step_respects_dependencies(self) -> None:
        plan = _make_plan()
        # step_2 depends on step_1, so it shouldn't be next while step_1 is pending
        step = plan.next_step()
        assert step is not None
        assert step.id == "step_1"

        # Complete step_1
        plan.mark_step("step_1", PlanStepStatus.COMPLETED)
        step = plan.next_step()
        assert step is not None
        assert step.id == "step_2"

    def test_next_step_returns_none_when_all_done(self) -> None:
        plan = _make_plan()
        for s in plan.steps:
            plan.mark_step(s.id, PlanStepStatus.COMPLETED)
        assert plan.next_step() is None

    def test_next_step_returns_none_when_blocked(self) -> None:
        plan = _make_plan()
        # Fail step_1 — step_2 depends on it and is blocked
        plan.mark_step("step_1", PlanStepStatus.FAILED)
        step = plan.next_step()
        # step_2 and step_3 are blocked because their deps aren't completed
        assert step is None

    def test_mark_step_updates_status(self) -> None:
        plan = _make_plan()
        plan.mark_step("step_1", PlanStepStatus.COMPLETED, result="Done")
        assert plan.steps[0].status == PlanStepStatus.COMPLETED
        assert plan.steps[0].result == "Done"

    def test_mark_step_nonexistent_is_noop(self) -> None:
        plan = _make_plan()
        plan.mark_step("nonexistent", PlanStepStatus.COMPLETED)
        # Should not raise, just no-op

    def test_is_complete_all_done(self) -> None:
        plan = _make_plan()
        for s in plan.steps:
            plan.mark_step(s.id, PlanStepStatus.COMPLETED)
        assert plan.is_complete is True

    def test_is_complete_with_skipped(self) -> None:
        plan = _make_plan()
        plan.mark_step("step_1", PlanStepStatus.COMPLETED)
        plan.mark_step("step_2", PlanStepStatus.SKIPPED)
        plan.mark_step("step_3", PlanStepStatus.COMPLETED)
        assert plan.is_complete is True

    def test_is_complete_not_done(self) -> None:
        plan = _make_plan()
        plan.mark_step("step_1", PlanStepStatus.COMPLETED)
        assert plan.is_complete is False

    def test_has_failed_with_failed_step(self) -> None:
        plan = _make_plan()
        plan.mark_step("step_1", PlanStepStatus.FAILED)
        assert plan.has_failed is True

    def test_has_failed_no_failures(self) -> None:
        plan = _make_plan()
        assert plan.has_failed is False

    def test_progress_ratio_empty(self) -> None:
        plan = Plan(steps=[], goal="empty")
        assert plan.progress_ratio == 0.0

    def test_progress_ratio_partial(self) -> None:
        plan = _make_plan()  # 3 steps
        plan.mark_step("step_1", PlanStepStatus.COMPLETED)
        assert abs(plan.progress_ratio - 1 / 3) < 0.01

    def test_progress_ratio_full(self) -> None:
        plan = _make_plan()
        for s in plan.steps:
            plan.mark_step(s.id, PlanStepStatus.COMPLETED)
        assert plan.progress_ratio == 1.0

    def test_progress_ratio_counts_skipped(self) -> None:
        plan = _make_plan()
        plan.mark_step("step_1", PlanStepStatus.COMPLETED)
        plan.mark_step("step_2", PlanStepStatus.SKIPPED)
        # 2 out of 3
        assert abs(plan.progress_ratio - 2 / 3) < 0.01

    def test_plan_serialization_roundtrip(self) -> None:
        plan = _make_plan()
        plan.mark_step("step_1", PlanStepStatus.COMPLETED, result="Done")
        data = plan.model_dump(mode="json")
        restored = Plan.model_validate(data)
        assert restored.steps[0].status == PlanStepStatus.COMPLETED
        assert restored.steps[0].result == "Done"
        assert restored.goal == plan.goal


# ── PlanExecutePolicy Tests ─────────────────────────────────────


class TestPlanExecutePolicy:
    """Tests for PlanExecutePolicy."""

    def test_policy_name(self) -> None:
        policy = PlanExecutePolicy()
        assert policy.name == "plan_execute"

    @pytest.mark.asyncio
    async def test_plan_phase_when_no_plan(self) -> None:
        """When no plan exists, should return an LLM call to create one."""
        policy = PlanExecutePolicy()
        state = _make_state(goal="Write a report")

        decision = await policy.decide(state)

        assert decision.action_type == "llm_call"
        assert len(decision.messages) == 2
        assert decision.messages[0]["role"] == "system"
        assert "Write a report" in decision.messages[0]["content"]
        assert decision.metadata.get("phase") == "plan"
        assert "plan" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_plan_phase_when_invalid_plan(self) -> None:
        """When plan data is invalid JSON string, should trigger plan phase."""
        policy = PlanExecutePolicy()
        state = _make_state(goal="Test")
        state.working_memory["plan"] = "not valid json"

        decision = await policy.decide(state)

        assert decision.action_type == "llm_call"
        assert decision.metadata.get("phase") == "plan"

    @pytest.mark.asyncio
    async def test_execute_phase_with_pending_steps(self) -> None:
        """When plan exists with pending steps, should execute next step."""
        policy = PlanExecutePolicy()
        plan = _make_plan()
        state = _make_state(goal="Test goal", plan=plan)

        decision = await policy.decide(state)

        assert decision.action_type == "llm_call"
        assert decision.metadata.get("phase") == "execute"
        assert decision.metadata.get("current_step_id") == "step_1"
        assert "Research the topic" in decision.messages[1]["content"]

    @pytest.mark.asyncio
    async def test_execute_phase_respects_step_order(self) -> None:
        """Execute phase should follow dependency order."""
        policy = PlanExecutePolicy()
        plan = _make_plan()
        plan.mark_step("step_1", PlanStepStatus.COMPLETED)
        state = _make_state(goal="Test goal", plan=plan)

        decision = await policy.decide(state)

        assert decision.action_type == "llm_call"
        assert decision.metadata.get("current_step_id") == "step_2"

    @pytest.mark.asyncio
    async def test_verify_phase_when_all_steps_done(self) -> None:
        """When all plan steps are complete, should trigger verification."""
        policy = PlanExecutePolicy()
        plan = _make_plan()
        for s in plan.steps:
            plan.mark_step(s.id, PlanStepStatus.COMPLETED)
        state = _make_state(goal="Test goal", plan=plan)

        decision = await policy.decide(state)

        assert decision.action_type == "verify"
        assert decision.metadata.get("phase") == "verify"
        assert "plan" in decision.metadata
        assert decision.messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_fail_when_plan_has_failed_step(self) -> None:
        """When a plan step has failed, should return fail decision."""
        policy = PlanExecutePolicy()
        plan = _make_plan()
        plan.mark_step("step_1", PlanStepStatus.FAILED)
        state = _make_state(goal="Test goal", plan=plan)

        decision = await policy.decide(state)

        assert decision.action_type == "fail"

    @pytest.mark.asyncio
    async def test_fail_when_steps_blocked(self) -> None:
        """When no steps are executable (all blocked), should fail."""
        policy = PlanExecutePolicy()
        # Create steps where step_2 depends on step_1, but step_1 is skipped (not completed)
        steps = [
            PlanStep(id="s1", description="First", status=PlanStepStatus.SKIPPED),
            PlanStep(id="s2", description="Second", dependencies=["s1"]),
        ]
        plan = Plan(steps=steps, goal="test")
        # s1 is skipped (counts as complete), so s2 should be executable
        state = _make_state(goal="test", plan=plan)

        decision = await policy.decide(state)
        # s1 is skipped which counts toward is_complete=False because s2 is still pending
        # But next_step checks if deps are COMPLETED, and skipped is not completed
        # So s2 is blocked
        assert decision.action_type == "fail"

    @pytest.mark.asyncio
    async def test_execute_phase_includes_history(self) -> None:
        """Execute phase should include completed step history in prompt."""
        policy = PlanExecutePolicy()
        plan = _make_plan()
        state = _make_state(
            goal="Test goal",
            plan=plan,
            completed_steps=["Step 1: researched topic"],
        )

        decision = await policy.decide(state)

        system_content = decision.messages[0]["content"]
        assert "researched topic" in system_content

    @pytest.mark.asyncio
    async def test_plan_loaded_from_dict(self) -> None:
        """Plan can be loaded when stored as dict in working memory."""
        policy = PlanExecutePolicy()
        plan = _make_plan()
        state = AgentState(run_id="test", goal="Test")
        state.working_memory["plan"] = plan.model_dump(mode="json")

        decision = await policy.decide(state)

        # Should proceed to execute phase (plan exists)
        assert decision.action_type == "llm_call"
        assert decision.metadata.get("phase") == "execute"

    @pytest.mark.asyncio
    async def test_plan_loaded_from_json_string(self) -> None:
        """Plan can be loaded when stored as JSON string in working memory."""
        policy = PlanExecutePolicy()
        plan = _make_plan()
        state = AgentState(run_id="test", goal="Test")
        state.working_memory["plan"] = json.dumps(plan.model_dump(mode="json"))

        decision = await policy.decide(state)

        assert decision.action_type == "llm_call"
        assert decision.metadata.get("phase") == "execute"


# ── GoalVerifier Tests ───────────────────────────────────────────


class TestGoalVerifier:
    """Tests for GoalVerifier."""

    @pytest.mark.asyncio
    async def test_all_criteria_met_returns_passed(self) -> None:
        verifier = GoalVerifier()
        plan = _make_plan()
        for s in plan.steps:
            plan.mark_step(s.id, PlanStepStatus.COMPLETED)

        state = _make_state(
            goal="Test goal",
            plan=plan,
            completed_steps=["Goal achieved successfully with great results"],
        )

        result = await verifier.verify(state, plan)

        assert result.outcome == VerificationOutcome.PASSED
        assert result.coverage == 1.0
        assert len(result.failed_criteria) == 0

    @pytest.mark.asyncio
    async def test_no_criteria_met_returns_failed(self) -> None:
        verifier = GoalVerifier()
        plan = _make_plan()
        # All steps still pending
        state = _make_state(goal="Test goal", plan=plan)

        result = await verifier.verify(state, plan)

        assert result.outcome == VerificationOutcome.FAILED
        assert result.coverage == 0.0
        assert len(result.failed_criteria) > 0

    @pytest.mark.asyncio
    async def test_partial_criteria_met_returns_partial(self) -> None:
        verifier = GoalVerifier()
        plan = _make_plan()
        plan.mark_step("step_1", PlanStepStatus.COMPLETED)
        # step_2, step_3 still pending; global criteria unmatched

        state = _make_state(goal="Test goal", plan=plan)

        result = await verifier.verify(state, plan)

        assert result.outcome == VerificationOutcome.PARTIAL
        assert 0.0 < result.coverage < 1.0

    @pytest.mark.asyncio
    async def test_no_plan_with_completed_steps(self) -> None:
        """Without a plan, verifier checks if any steps were completed."""
        verifier = GoalVerifier()
        state = _make_state(
            goal="Complete the task",
            completed_steps=["Did something useful"],
        )

        result = await verifier.verify(state, plan=None)

        assert result.outcome == VerificationOutcome.PASSED

    @pytest.mark.asyncio
    async def test_no_plan_no_steps_returns_failed(self) -> None:
        """Without a plan and no completed steps, should fail."""
        verifier = GoalVerifier()
        state = _make_state(goal="Complete the task")

        result = await verifier.verify(state, plan=None)

        assert result.outcome == VerificationOutcome.FAILED

    @pytest.mark.asyncio
    async def test_no_plan_no_goal_returns_passed(self) -> None:
        """Without plan or goal, no criteria to check — passes vacuously."""
        verifier = GoalVerifier()
        state = AgentState(run_id="test")

        result = await verifier.verify(state, plan=None)

        assert result.outcome == VerificationOutcome.PASSED
        assert result.coverage == 1.0

    @pytest.mark.asyncio
    async def test_suggestions_generated_for_failures(self) -> None:
        verifier = GoalVerifier()
        plan = _make_plan()
        state = _make_state(goal="Test", plan=plan)

        result = await verifier.verify(state, plan)

        assert len(result.suggestions) > 0
        for suggestion in result.suggestions:
            assert suggestion.startswith("Re-attempt:")

    @pytest.mark.asyncio
    async def test_global_criteria_matched_by_completed_steps(self) -> None:
        """Global acceptance criteria should match against completed step text."""
        verifier = GoalVerifier()
        plan = Plan(
            steps=[
                PlanStep(
                    id="s1",
                    description="Do task",
                    status=PlanStepStatus.COMPLETED,
                ),
            ],
            goal="Test",
            acceptance_criteria=["Task completed successfully"],
        )
        state = _make_state(
            goal="Test",
            plan=plan,
            completed_steps=["Task completed successfully with output"],
        )

        result = await verifier.verify(state, plan)

        # step is completed AND global criterion matches completed steps
        assert result.outcome == VerificationOutcome.PASSED

    @pytest.mark.asyncio
    async def test_global_criteria_not_matched(self) -> None:
        """Global criteria that don't match any completed step should fail."""
        verifier = GoalVerifier()
        plan = Plan(
            steps=[
                PlanStep(
                    id="s1",
                    description="Do task",
                    status=PlanStepStatus.COMPLETED,
                ),
            ],
            goal="Test",
            acceptance_criteria=["Completely unrelated criterion xyz"],
        )
        state = _make_state(
            goal="Test",
            plan=plan,
            completed_steps=["Finished doing the task"],
        )

        result = await verifier.verify(state, plan)

        # Step passed but global criterion did not match
        assert result.outcome == VerificationOutcome.PARTIAL


# ── PlanReducer Tests ────────────────────────────────────────────


class TestPlanReducer:
    """Tests for PlanReducer."""

    def test_reducer_name(self) -> None:
        reducer = PlanReducer()
        assert reducer.name == "plan"

    @pytest.mark.asyncio
    async def test_plan_stored_from_state_updates(self) -> None:
        """Plan data in state_updates should be stored in working_memory."""
        reducer = PlanReducer()
        state = AgentState(run_id="test")
        plan = _make_plan()

        step_result = StepResult(
            step_type=StepType.PLAN,
            step_id="step-1",
            success=True,
            state_updates={"plan": plan.model_dump(mode="json")},
        )

        new_state = await reducer.reduce(state, step_result)

        assert "plan" in new_state.working_memory
        loaded = Plan.model_validate(new_state.working_memory["plan"])
        assert loaded.goal == "Test goal"
        assert len(loaded.steps) == 3

    @pytest.mark.asyncio
    async def test_plan_stored_from_json_string(self) -> None:
        """Plan as JSON string in state_updates should be parsed and stored."""
        reducer = PlanReducer()
        state = AgentState(run_id="test")
        plan = _make_plan()

        step_result = StepResult(
            step_type=StepType.PLAN,
            step_id="step-1",
            success=True,
            state_updates={"plan": json.dumps(plan.model_dump(mode="json"))},
        )

        new_state = await reducer.reduce(state, step_result)

        assert "plan" in new_state.working_memory
        loaded = Plan.model_validate(new_state.working_memory["plan"])
        assert len(loaded.steps) == 3

    @pytest.mark.asyncio
    async def test_step_completion_tracked(self) -> None:
        """Memory update with plan_step_completed should mark the step."""
        reducer = PlanReducer()
        plan = _make_plan()
        state = _make_state(plan=plan)

        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=True,
            thought="Done researching",
            memory_updates={"plan_step_completed": "step_1"},
        )

        new_state = await reducer.reduce(state, step_result)

        loaded = Plan.model_validate(new_state.working_memory["plan"])
        assert loaded.steps[0].status == PlanStepStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_step_result_text_stored(self) -> None:
        """Memory update with plan_step_result should set step result text."""
        reducer = PlanReducer()
        plan = _make_plan()
        state = _make_state(plan=plan)

        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=True,
            memory_updates={
                "plan_step_completed": "step_1",
                "plan_step_result": "Research complete",
            },
        )

        new_state = await reducer.reduce(state, step_result)

        loaded = Plan.model_validate(new_state.working_memory["plan"])
        assert loaded.steps[0].result == "Research complete"

    @pytest.mark.asyncio
    async def test_progress_tracking_updated(self) -> None:
        """Plan progress should be tracked in working memory."""
        reducer = PlanReducer()
        plan = _make_plan()
        state = _make_state(plan=plan)

        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=True,
            memory_updates={"plan_step_completed": "step_1"},
        )

        new_state = await reducer.reduce(state, step_result)

        assert "plan_progress" in new_state.working_memory
        assert abs(new_state.working_memory["plan_progress"] - 1 / 3) < 0.01
        assert new_state.working_memory["plan_complete"] is False
        assert new_state.working_memory["plan_failed"] is False

    @pytest.mark.asyncio
    async def test_inherits_default_reducer_behavior(self) -> None:
        """PlanReducer should still track errors and completed steps like DefaultReducer."""
        reducer = PlanReducer()
        state = AgentState(run_id="test")

        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=False,
            error="Something went wrong",
        )

        new_state = await reducer.reduce(state, step_result)

        assert new_state.consecutive_errors == 1
        assert new_state.last_error == "Something went wrong"

    @pytest.mark.asyncio
    async def test_no_plan_in_state_handles_gracefully(self) -> None:
        """When no plan exists, reducer should still work without errors."""
        reducer = PlanReducer()
        state = AgentState(run_id="test")

        step_result = StepResult(
            step_type=StepType.THINK,
            step_id="step-1",
            success=True,
            thought="Just thinking",
        )

        new_state = await reducer.reduce(state, step_result)

        assert new_state.consecutive_errors == 0
        assert "plan" not in new_state.working_memory


# ── StepExecutor Verify Tests ───────────────────────────────────


class TestStepExecutorVerify:
    """Tests for StepExecutor verify action_type handling."""

    def _make_executor(
        self, *, verifier: GoalVerifier | None = None
    ) -> StepExecutor:
        mock_gateway = MagicMock()
        mock_gateway.default_provider = "mock"
        mock_provider = MagicMock()
        mock_provider.default_model = "mock-model"
        mock_gateway.get.return_value = mock_provider
        return StepExecutor(
            gateway=mock_gateway,
            verifier=verifier,
        )

    def _make_trace_ctx(self) -> TraceContext:
        return TraceContext(run_id="test-run")

    @pytest.mark.asyncio
    async def test_verify_without_verifier(self) -> None:
        """Verify action without a verifier should succeed with goal_reached."""
        executor = self._make_executor()
        state = AgentState(run_id="test", goal="Test")
        decision = PolicyDecision(
            action_type="verify",
            messages=[],
            metadata={"phase": "verify"},
        )
        trace_ctx = self._make_trace_ctx()

        result = await executor.execute(
            state=state, decision=decision, trace_ctx=trace_ctx
        )

        assert result.step_type == StepType.VERIFY
        assert result.success is True
        assert result.state_updates.get("goal_reached") is True

    @pytest.mark.asyncio
    async def test_verify_with_verifier_passed(self) -> None:
        """Verify with verifier that passes should return success."""
        verifier = GoalVerifier()
        executor = self._make_executor(verifier=verifier)
        plan = _make_plan()
        for s in plan.steps:
            plan.mark_step(s.id, PlanStepStatus.COMPLETED)
        state = _make_state(
            goal="Test",
            plan=plan,
            completed_steps=["Goal achieved successfully"],
        )
        decision = PolicyDecision(
            action_type="verify",
            messages=[],
            metadata={"phase": "verify", "plan": plan.model_dump(mode="json")},
        )
        trace_ctx = self._make_trace_ctx()

        result = await executor.execute(
            state=state, decision=decision, trace_ctx=trace_ctx
        )

        assert result.success is True
        assert "verification_result" in result.state_updates

    @pytest.mark.asyncio
    async def test_verify_with_verifier_failed(self) -> None:
        """Verify with verifier that fails should return failure."""
        verifier = GoalVerifier()
        executor = self._make_executor(verifier=verifier)
        plan = _make_plan()
        # No steps completed
        state = _make_state(goal="Test", plan=plan)
        decision = PolicyDecision(
            action_type="verify",
            messages=[],
            metadata={"phase": "verify", "plan": plan.model_dump(mode="json")},
        )
        trace_ctx = self._make_trace_ctx()

        result = await executor.execute(
            state=state, decision=decision, trace_ctx=trace_ctx
        )

        assert result.success is False
        assert result.step_type == StepType.VERIFY
        assert result.is_recoverable is True
        assert "verification_result" in result.state_updates


# ── Integration Tests ────────────────────────────────────────────


class TestPlanExecuteIntegration:
    """Integration tests for the full plan-execute-verify cycle."""

    @pytest.mark.asyncio
    async def test_full_cycle_plan_execute_verify(self) -> None:
        """Test full cycle: plan -> execute steps -> verify."""
        policy = PlanExecutePolicy()
        reducer = PlanReducer()
        verifier = GoalVerifier()

        # Phase 1: Plan
        state = AgentState(run_id="integration-test", goal="Write a report")
        decision = await policy.decide(state)
        assert decision.action_type == "llm_call"
        assert decision.metadata.get("phase") == "plan"

        # Simulate LLM returning a plan — apply via reducer
        plan = Plan(
            steps=[
                PlanStep(
                    id="s1",
                    description="Research topic",
                    acceptance_criteria=["Topic researched"],
                ),
                PlanStep(
                    id="s2",
                    description="Write draft",
                    acceptance_criteria=["Draft written"],
                    dependencies=["s1"],
                ),
            ],
            goal="Write a report",
            acceptance_criteria=["Report is complete and reviewed"],
        )

        plan_result = StepResult(
            step_type=StepType.PLAN,
            step_id="plan-step",
            success=True,
            state_updates={"plan": plan.model_dump(mode="json")},
        )
        state = await reducer.reduce(state, plan_result)

        # Phase 2: Execute step 1
        decision = await policy.decide(state)
        assert decision.action_type == "llm_call"
        assert decision.metadata.get("phase") == "execute"
        assert decision.metadata.get("current_step_id") == "s1"

        # Simulate step 1 completion
        exec_result_1 = StepResult(
            step_type=StepType.THINK,
            step_id="exec-1",
            success=True,
            thought="Researched the topic",
            action="STEP_COMPLETE",
            memory_updates={
                "plan_step_completed": "s1",
                "plan_step_result": "Topic thoroughly researched",
            },
        )
        state = await reducer.reduce(state, exec_result_1)

        # Phase 2 continued: Execute step 2
        decision = await policy.decide(state)
        assert decision.action_type == "llm_call"
        assert decision.metadata.get("current_step_id") == "s2"

        # Simulate step 2 completion
        exec_result_2 = StepResult(
            step_type=StepType.THINK,
            step_id="exec-2",
            success=True,
            thought="Wrote the draft",
            action="STEP_COMPLETE",
            memory_updates={
                "plan_step_completed": "s2",
                "plan_step_result": "Draft written and formatted",
            },
        )
        state = await reducer.reduce(state, exec_result_2)

        # Phase 3: Verify
        decision = await policy.decide(state)
        assert decision.action_type == "verify"
        assert decision.metadata.get("phase") == "verify"

        # Run verification
        plan_data = decision.metadata.get("plan")
        verified_plan = Plan.model_validate(plan_data)
        result = await verifier.verify(state, verified_plan)

        # Both steps completed, but global criterion might not match
        assert result.outcome in (
            VerificationOutcome.PASSED,
            VerificationOutcome.PARTIAL,
        )
        assert result.coverage > 0.0

    @pytest.mark.asyncio
    async def test_plan_reducer_then_policy_cycle(self) -> None:
        """Reducer updates state so policy makes correct next decision."""
        policy = PlanExecutePolicy()
        reducer = PlanReducer()

        state = AgentState(run_id="cycle-test", goal="Do task")

        # Store a plan
        plan = Plan(
            steps=[
                PlanStep(id="a", description="First action"),
                PlanStep(id="b", description="Second action", dependencies=["a"]),
            ],
            goal="Do task",
        )
        plan_result = StepResult(
            step_type=StepType.PLAN,
            step_id="p1",
            success=True,
            state_updates={"plan": plan.model_dump(mode="json")},
        )
        state = await reducer.reduce(state, plan_result)

        # Policy should pick step "a"
        decision = await policy.decide(state)
        assert decision.metadata.get("current_step_id") == "a"

        # Complete step "a" via reducer
        complete_a = StepResult(
            step_type=StepType.THINK,
            step_id="e1",
            success=True,
            memory_updates={"plan_step_completed": "a"},
        )
        state = await reducer.reduce(state, complete_a)

        # Policy should now pick step "b"
        decision = await policy.decide(state)
        assert decision.metadata.get("current_step_id") == "b"

        # Complete step "b"
        complete_b = StepResult(
            step_type=StepType.THINK,
            step_id="e2",
            success=True,
            memory_updates={"plan_step_completed": "b"},
        )
        state = await reducer.reduce(state, complete_b)

        # Policy should now trigger verify
        decision = await policy.decide(state)
        assert decision.action_type == "verify"


# ── Module Export Tests ──────────────────────────────────────────


class TestModuleExports:
    """Tests that modules are properly exported."""

    def test_policy_export(self) -> None:
        from arcana.runtime.policies import PlanExecutePolicy

        assert PlanExecutePolicy is not None

    def test_verifier_export(self) -> None:
        from arcana.runtime.verifiers import BaseVerifier, GoalVerifier

        assert BaseVerifier is not None
        assert GoalVerifier is not None

    def test_reducer_export(self) -> None:
        from arcana.runtime.reducers import PlanReducer

        assert PlanReducer is not None
