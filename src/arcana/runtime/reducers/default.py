"""Default reducer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.runtime import StepType
from arcana.runtime.reducers.base import BaseReducer

if TYPE_CHECKING:
    from arcana.contracts.runtime import StepResult
    from arcana.contracts.state import AgentState


class DefaultReducer(BaseReducer):
    """
    Default reducer that applies standard state updates.

    Updates:
    - Completed steps
    - Working memory
    - Error tracking
    - Budget tracking
    """

    @property
    def name(self) -> str:
        return "default"

    async def reduce(
        self,
        state: AgentState,
        step_result: StepResult,
    ) -> AgentState:
        """Apply step result to state."""
        # Track completed step
        step_summary = self._summarize_step(step_result)
        if step_summary:
            state.completed_steps.append(step_summary)

        # Apply state updates from step.
        # Keys that match AgentState attributes are set directly on
        # the state object. All other keys are written into
        # working_memory so that policies can read them on subsequent
        # decide() calls (e.g. "answer", "pending_tool_call",
        # "adaptive_state").
        for key, value in step_result.state_updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                state.working_memory[key] = value

        # Apply memory updates
        for key, value in step_result.memory_updates.items():
            if value is None:
                # None means delete
                state.working_memory.pop(key, None)
            else:
                state.working_memory[key] = value

        # After a tool execution step, clear pending tool call keys from
        # working_memory so the adaptive policy does not re-execute them.
        if step_result.step_type == StepType.ACT:
            state.working_memory.pop("pending_tool_call", None)
            state.working_memory.pop("pending_parallel_calls", None)

        # Track errors
        if step_result.success:
            state.consecutive_errors = 0
            state.last_error = None
        else:
            state.consecutive_errors += 1
            state.last_error = step_result.error

        # Update messages with LLM response
        if step_result.llm_response and step_result.llm_response.content:
            state.messages.append(
                {
                    "role": "assistant",
                    "content": step_result.llm_response.content,
                }
            )

        return state

    def _summarize_step(self, step_result: StepResult) -> str | None:
        """Create a summary of the step for history."""
        parts = []

        if step_result.thought:
            parts.append(f"Thought: {step_result.thought[:100]}")

        if step_result.action:
            parts.append(f"Action: {step_result.action}")

        if step_result.observation:
            parts.append(f"Observation: {step_result.observation[:100]}")

        if step_result.error:
            parts.append(f"Error: {step_result.error[:50]}")

        return " | ".join(parts) if parts else None
