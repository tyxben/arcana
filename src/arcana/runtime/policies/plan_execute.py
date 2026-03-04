"""Plan-and-Execute policy implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from arcana.contracts.plan import Plan
from arcana.contracts.runtime import PolicyDecision
from arcana.runtime.policies.base import BasePolicy

if TYPE_CHECKING:
    from arcana.contracts.state import AgentState


PLAN_SYSTEM_PROMPT = """You are an AI assistant that follows the Plan-and-Execute framework.

You must create a structured plan to achieve the goal. Respond with a JSON object:

{{
  "goal": "<the goal to achieve>",
  "acceptance_criteria": ["<criterion 1>", "<criterion 2>"],
  "steps": [
    {{
      "id": "step_1",
      "description": "<what to do>",
      "acceptance_criteria": ["<how to verify this step>"],
      "dependencies": []
    }},
    {{
      "id": "step_2",
      "description": "<what to do next>",
      "acceptance_criteria": ["<how to verify>"],
      "dependencies": ["step_1"]
    }}
  ]
}}

Goal: {goal}

Respond ONLY with the JSON plan, no additional text."""

EXECUTE_SYSTEM_PROMPT = """You are an AI assistant executing a plan step-by-step.

Current goal: {goal}

Plan progress: {progress}

Current step to execute:
- ID: {step_id}
- Description: {step_description}
- Acceptance criteria: {step_criteria}

Previous steps completed:
{history}

Working memory:
{memory}

Execute this step. Format your response as:
Thought: <your reasoning about how to execute this step>
Action: <the action to take>

If the step is complete, respond with:
Thought: <summary of what was accomplished>
Action: STEP_COMPLETE"""

VERIFY_SYSTEM_PROMPT = """You are an AI assistant verifying that a plan has been completed.

Goal: {goal}

Plan acceptance criteria:
{acceptance_criteria}

Completed steps:
{completed_steps}

Evaluate whether the goal has been achieved based on the acceptance criteria.
Respond with your assessment."""


class PlanExecutePolicy(BasePolicy):
    """
    Plan-and-Execute policy: structured planning followed by step execution.

    Three phases:
    1. Plan: Generate a structured plan via LLM call
    2. Execute: Execute plan steps one at a time
    3. Verify: Verify goal completion when all steps are done
    """

    @property
    def name(self) -> str:
        return "plan_execute"

    async def decide(self, state: AgentState) -> PolicyDecision:
        """Generate Plan-Execute-Verify decision based on current state."""
        plan = self._load_plan(state)

        if plan is None:
            return self._plan_phase(state)

        if plan.has_failed:
            return PolicyDecision(
                action_type="fail",
                stop_reason="Plan has failed steps",
                reasoning="A plan step has failed",
            )

        if not plan.is_complete:
            next_step = plan.next_step()
            if next_step is not None:
                return self._execute_phase(state, plan, next_step.id)
            # All steps either completed, failed, or blocked
            return PolicyDecision(
                action_type="fail",
                stop_reason="No executable steps remaining (blocked dependencies)",
                reasoning="Plan has pending steps but none are executable",
            )

        # All steps complete — verify
        return self._verify_phase(state, plan)

    def _load_plan(self, state: AgentState) -> Plan | None:
        """Load plan from working memory."""
        plan_data = state.working_memory.get("plan")
        if plan_data is None:
            return None

        if isinstance(plan_data, str):
            try:
                parsed = json.loads(plan_data)
                return Plan.model_validate(parsed)
            except (json.JSONDecodeError, Exception):
                return None
        elif isinstance(plan_data, dict):
            try:
                return Plan.model_validate(plan_data)
            except Exception:
                return None
        elif isinstance(plan_data, Plan):
            return plan_data

        return None

    def _plan_phase(self, state: AgentState) -> PolicyDecision:
        """Generate a planning decision — ask LLM to create a structured plan."""
        system_prompt = PLAN_SYSTEM_PROMPT.format(
            goal=state.goal or "No goal specified",
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Create a plan to achieve this goal."},
        ]

        return PolicyDecision(
            action_type="llm_call",
            messages=messages,
            reasoning="Plan phase: generating structured plan via LLM",
            metadata={"phase": "plan"},
        )

    def _execute_phase(
        self,
        state: AgentState,
        plan: Plan,
        step_id: str,
    ) -> PolicyDecision:
        """Generate an execution decision for the next plan step."""
        # Find the step
        current_step = None
        for s in plan.steps:
            if s.id == step_id:
                current_step = s
                break

        if current_step is None:
            return PolicyDecision(
                action_type="fail",
                stop_reason=f"Step {step_id} not found in plan",
                reasoning="Could not find the next step to execute",
            )

        history = self._format_history(state)
        memory = self._format_memory(state)
        progress = f"{plan.progress_ratio:.0%} ({sum(1 for s in plan.steps if s.status.value == 'completed')}/{len(plan.steps)} steps)"

        criteria_str = ", ".join(current_step.acceptance_criteria) if current_step.acceptance_criteria else "None specified"

        system_prompt = EXECUTE_SYSTEM_PROMPT.format(
            goal=state.goal or "No goal specified",
            progress=progress,
            step_id=current_step.id,
            step_description=current_step.description,
            step_criteria=criteria_str,
            history=history or "No previous steps",
            memory=memory or "Empty",
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Execute step: {current_step.description}"},
        ]

        return PolicyDecision(
            action_type="llm_call",
            messages=messages,
            reasoning=f"Execute phase: executing step {current_step.id}",
            metadata={
                "phase": "execute",
                "current_step_id": current_step.id,
            },
        )

    def _verify_phase(
        self,
        state: AgentState,
        plan: Plan,
    ) -> PolicyDecision:
        """Generate a verification decision."""
        criteria = plan.acceptance_criteria
        completed_summaries = []
        for step in plan.steps:
            result_str = f" -> {step.result}" if step.result else ""
            completed_summaries.append(
                f"- [{step.status.value}] {step.description}{result_str}"
            )

        system_prompt = VERIFY_SYSTEM_PROMPT.format(
            goal=state.goal or "No goal specified",
            acceptance_criteria="\n".join(f"- {c}" for c in criteria) if criteria else "None specified",
            completed_steps="\n".join(completed_summaries),
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Verify that the goal has been achieved."},
        ]

        return PolicyDecision(
            action_type="verify",
            messages=messages,
            reasoning="Verify phase: all plan steps complete, verifying goal",
            metadata={
                "phase": "verify",
                "plan": plan.model_dump(mode="json"),
            },
        )

    def _format_history(self, state: AgentState) -> str:
        """Format completed steps as history."""
        if not state.completed_steps:
            return ""

        return "\n".join(
            f"Step {i + 1}: {step}"
            for i, step in enumerate(state.completed_steps[-5:])
        )

    def _format_memory(self, state: AgentState) -> str:
        """Format working memory for context."""
        if not state.working_memory:
            return ""

        items = []
        for key, value in state.working_memory.items():
            if key == "plan":
                items.append("- plan: <structured plan loaded>")
                continue
            value_str = str(value)
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            items.append(f"- {key}: {value_str}")

        return "\n".join(items)
