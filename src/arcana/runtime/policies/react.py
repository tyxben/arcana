"""ReAct (Reasoning + Acting) policy implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.runtime import PolicyDecision
from arcana.runtime.policies.base import BasePolicy

if TYPE_CHECKING:
    from arcana.contracts.state import AgentState


REACT_SYSTEM_PROMPT = """You are an AI assistant that follows the ReAct framework.

For each step, you must:
1. Think about what to do next
2. Decide on an action (or conclude if the goal is reached)

Format your response EXACTLY as:
Thought: <your reasoning about the current situation and what to do>
Action: <the action to take, or "FINISH" if the goal is achieved>

Goal: {goal}

Previous steps:
{history}

Working memory:
{memory}
"""


class ReActPolicy(BasePolicy):
    """
    ReAct policy: Reasoning + Acting in an interleaved manner.

    Reference: https://arxiv.org/abs/2210.03629
    """

    @property
    def name(self) -> str:
        return "react"

    async def decide(self, state: AgentState) -> PolicyDecision:
        """Generate ReAct-style decision."""
        # Build context
        history = self._format_history(state)
        memory = self._format_memory(state)

        system_prompt = REACT_SYSTEM_PROMPT.format(
            goal=state.goal or "No goal specified",
            history=history or "No previous steps",
            memory=memory or "Empty",
        )

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is your next step?"},
        ]

        return PolicyDecision(
            action_type="llm_call",
            messages=messages,
            reasoning="ReAct step: generate thought and action",
        )

    def _format_history(self, state: AgentState) -> str:
        """Format completed steps as history."""
        if not state.completed_steps:
            return ""

        return "\n".join(
            f"Step {i + 1}: {step}"
            for i, step in enumerate(state.completed_steps[-5:])  # Last 5 steps
        )

    def _format_memory(self, state: AgentState) -> str:
        """Format working memory for context."""
        if not state.working_memory:
            return ""

        items = []
        for key, value in state.working_memory.items():
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            items.append(f"- {key}: {value_str}")

        return "\n".join(items)
