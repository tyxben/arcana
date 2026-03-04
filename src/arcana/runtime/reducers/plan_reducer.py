"""Plan-aware reducer that tracks plan state updates."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from arcana.contracts.plan import Plan, PlanStepStatus
from arcana.runtime.reducers.default import DefaultReducer

if TYPE_CHECKING:
    from arcana.contracts.runtime import StepResult
    from arcana.contracts.state import AgentState


class PlanReducer(DefaultReducer):
    """
    Reducer that extends DefaultReducer with plan-aware state updates.

    Additional behavior on top of DefaultReducer:
    - Parses plan data from step_result.state_updates["plan"]
    - Marks plan steps as completed via memory_updates["plan_step_completed"]
    - Tracks plan progress in working memory
    """

    @property
    def name(self) -> str:
        return "plan"

    async def reduce(
        self,
        state: AgentState,
        step_result: StepResult,
    ) -> AgentState:
        """Apply step result to state with plan-aware updates."""
        # Run parent reducer first
        state = await super().reduce(state, step_result)

        # Check for plan data in state_updates
        plan_data = step_result.state_updates.get("plan")
        if plan_data is not None:
            self._update_plan(state, plan_data)

        # Check for plan step completion in memory_updates
        completed_step_id = step_result.memory_updates.get("plan_step_completed")
        if completed_step_id is not None:
            self._mark_step_completed(state, completed_step_id)

        # Check for plan step completion result
        step_result_text = step_result.memory_updates.get("plan_step_result")

        if step_result_text and completed_step_id:
            self._set_step_result(state, completed_step_id, step_result_text)

        # Update plan progress tracking
        self._update_progress(state)

        return state

    def _update_plan(self, state: AgentState, plan_data: object) -> None:
        """Parse and store plan data in working memory."""
        if isinstance(plan_data, str):
            try:
                parsed = json.loads(plan_data)
                # Validate it can be parsed as a Plan
                Plan.model_validate(parsed)
                state.working_memory["plan"] = parsed
            except (json.JSONDecodeError, Exception):
                # Store raw string if it can't be parsed
                state.working_memory["plan"] = plan_data
        elif isinstance(plan_data, dict):
            try:
                Plan.model_validate(plan_data)
                state.working_memory["plan"] = plan_data
            except Exception:
                state.working_memory["plan"] = plan_data
        elif isinstance(plan_data, Plan):
            state.working_memory["plan"] = plan_data.model_dump(mode="json")

    def _mark_step_completed(self, state: AgentState, step_id: str) -> None:
        """Mark a plan step as completed."""
        plan = self._load_plan(state)
        if plan is None:
            return

        plan.mark_step(step_id, PlanStepStatus.COMPLETED)
        state.working_memory["plan"] = plan.model_dump(mode="json")

    def _set_step_result(
        self,
        state: AgentState,
        step_id: str,
        result: str,
    ) -> None:
        """Set the result text for a plan step."""
        plan = self._load_plan(state)
        if plan is None:
            return

        for step in plan.steps:
            if step.id == step_id:
                step.result = result
                break

        state.working_memory["plan"] = plan.model_dump(mode="json")

    def _update_progress(self, state: AgentState) -> None:
        """Update plan progress tracking in working memory."""
        plan = self._load_plan(state)
        if plan is None:
            return

        state.working_memory["plan_progress"] = plan.progress_ratio
        state.working_memory["plan_complete"] = plan.is_complete
        state.working_memory["plan_failed"] = plan.has_failed

    def _load_plan(self, state: AgentState) -> Plan | None:
        """Load plan from working memory."""
        plan_data = state.working_memory.get("plan")
        if plan_data is None:
            return None

        if isinstance(plan_data, dict):
            try:
                return Plan.model_validate(plan_data)
            except Exception:
                return None
        elif isinstance(plan_data, str):
            try:
                parsed = json.loads(plan_data)
                return Plan.model_validate(parsed)
            except (json.JSONDecodeError, Exception):
                return None
        elif isinstance(plan_data, Plan):
            return plan_data

        return None
