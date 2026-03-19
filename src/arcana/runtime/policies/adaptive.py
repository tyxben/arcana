"""Adaptive policy that delegates strategy decisions to the LLM.

.. deprecated::
    Legacy V1 component. AdaptivePolicy forces the LLM to emit structured
    JSON strategy declarations, which is premature structuring. V2
    ConversationAgent lets the LLM respond naturally instead.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from arcana.contracts.runtime import PolicyDecision
from arcana.contracts.strategy import AdaptiveState, StrategyDecision, StrategyType
from arcana.runtime.policies.base import BasePolicy

if TYPE_CHECKING:
    from arcana.contracts.state import AgentState

logger = logging.getLogger(__name__)

ADAPTIVE_SYSTEM_PROMPT = """You are an AI agent with adaptive strategy selection.

At each step, you decide HOW to proceed, not just WHAT to do. Choose the strategy that fits the current situation:

- direct_answer: You already have enough information to answer. Use this to end the loop.
- single_tool: You need exactly one tool call. Specify which tool and arguments.
- sequential: You need to do multiple things in order. Describe your next action.
- parallel: You need to do multiple independent things. List them all.
- plan_and_execute: The task is complex enough to warrant an explicit plan. Generate the plan.
- pivot: Your current approach is wrong. Explain why and what you'll try instead.

Respond with a JSON object:
{
  "strategy": "<strategy_type>",
  "reasoning": "<why this strategy>",
  "action": "<for direct_answer: your response>",
  "tool_name": "<for single_tool: tool name>",
  "tool_arguments": {"<for single_tool: arguments>"},
  "parallel_actions": [{"tool_name": "...", "arguments": {...}}, ...],
  "plan": ["step 1 description", "step 2 description", ...],
  "pivot_reason": "<for pivot: why the current approach fails>",
  "pivot_new_approach": "<for pivot: what to try instead>"
}

Include only the fields relevant to your chosen strategy. Omit the rest.

Rules:
- If you can answer now, answer now. Do not add unnecessary steps.
- If your plan is not working, pivot. Do not repeat failed approaches.
- Parallel actions must be truly independent (no dependencies between them).
- Plans should be short (3-7 steps). If you need more, break the problem into sub-problems."""


class AdaptivePolicy(BasePolicy):
    """
    Adaptive policy that delegates strategy decisions to the LLM.

    Instead of imposing ReAct or PlanExecute structure, this policy
    asks the LLM: "Given what you know, what do you want to do next?"

    The LLM can:
    - Answer directly (short-circuit the loop)
    - Call a single tool
    - Request sequential steps
    - Request parallel tool calls
    - Generate an explicit plan
    - Pivot to a completely different approach
    """

    def __init__(
        self,
        *,
        default_strategy: StrategyType = StrategyType.SEQUENTIAL,
        max_pivots: int = 3,
        allow_parallel: bool = True,
    ):
        self.default_strategy = default_strategy
        self.max_pivots = max_pivots
        self.allow_parallel = allow_parallel

    @property
    def name(self) -> str:
        return "adaptive"

    async def decide(self, state: AgentState) -> PolicyDecision:
        """
        Ask the LLM what to do next.

        Before asking the LLM for a new strategy, check if a previous
        strategy decision left a pending tool call or parallel calls
        in working_memory. If so, execute those directly.

        The prompt provides:
        1. Current goal and progress
        2. Available strategies
        3. Recent history (compressed)
        4. Error context if recovering
        """
        # Check if there's a pending tool call from a previous strategy decision
        pending = state.working_memory.get("pending_tool_call")
        if pending and isinstance(pending, dict):
            return PolicyDecision(
                action_type="tool_call",
                tool_calls=[{
                    "name": pending["name"],
                    "arguments": pending.get("arguments", {}),
                }],
                reasoning="Executing tool call from previous strategy decision",
            )

        # Check if there are pending parallel calls
        parallel = state.working_memory.get("pending_parallel_calls")
        if parallel and isinstance(parallel, list) and len(parallel) > 0:
            return PolicyDecision(
                action_type="tool_call",
                tool_calls=[
                    {
                        "name": a.get("tool_name", ""),
                        "arguments": a.get("arguments", {}),
                    }
                    for a in parallel
                ],
                reasoning="Executing parallel tool calls from previous strategy decision",
            )

        # Normal path: ask LLM for strategy
        system_prompt = self._build_strategy_prompt(state)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_step_prompt(state)},
        ]

        return PolicyDecision(
            action_type="llm_call",
            messages=messages,
            reasoning="Adaptive: asking LLM for strategy decision",
            metadata={
                "phase": "strategy",
                "expected_format": "strategy_decision",
            },
        )

    def _build_strategy_prompt(self, state: AgentState) -> str:
        """Build the system prompt that presents available strategies."""
        adaptive_state = self._load_adaptive_state(state)

        prompt_parts = [
            ADAPTIVE_SYSTEM_PROMPT,
            f"\nGoal: {state.goal}",
            f"\nCurrent step: {state.current_step} / {state.max_steps}",
        ]

        # Add strategy options
        strategies = self._available_strategies(adaptive_state)
        prompt_parts.append(
            f"\nAvailable strategies: {', '.join(s.value for s in strategies)}"
        )

        # Add plan context if in plan-and-execute mode
        if adaptive_state.plan_steps:
            completed = len(adaptive_state.completed_plan_steps)
            total = len(adaptive_state.plan_steps)
            prompt_parts.append(f"\nActive plan: {completed}/{total} steps completed")
            next_steps = [
                s
                for s in adaptive_state.plan_steps
                if s not in adaptive_state.completed_plan_steps
            ]
            if next_steps:
                prompt_parts.append(f"Next planned: {next_steps[0]}")

        # Add error context if recovering
        if state.last_error:
            prompt_parts.append(f"\nLast error: {state.last_error}")
            prompt_parts.append(
                "Consider: retry with changes, different tool, narrower scope, or pivot."
            )

        # Add pivot budget
        if adaptive_state.pivot_count > 0:
            remaining = self.max_pivots - adaptive_state.pivot_count
            prompt_parts.append(f"\nPivots remaining: {remaining}")

        return "\n".join(prompt_parts)

    def _build_step_prompt(self, state: AgentState) -> str:
        """Build the user-turn prompt for this step."""
        if state.current_step == 0:
            return "Assess this goal. Choose your strategy and take your first action."

        # Include recent history (last 3 steps, compressed)
        history = self._format_recent_history(state, max_steps=3)

        if state.last_error:
            return f"Previous steps:\n{history}\n\nThe last action failed. Decide how to recover."

        return f"Previous steps:\n{history}\n\nDecide your next action."

    def _available_strategies(self, adaptive_state: AdaptiveState) -> list[StrategyType]:
        """Determine which strategies are available given current state."""
        strategies = [
            StrategyType.DIRECT_ANSWER,
            StrategyType.SINGLE_TOOL,
            StrategyType.SEQUENTIAL,
            StrategyType.PLAN_AND_EXECUTE,
        ]

        if self.allow_parallel:
            strategies.append(StrategyType.PARALLEL)

        if adaptive_state.pivot_count < self.max_pivots:
            strategies.append(StrategyType.PIVOT)

        return strategies

    def _load_adaptive_state(self, state: AgentState) -> AdaptiveState:
        """Load or create adaptive policy state from working memory."""
        data = state.working_memory.get("adaptive_state")
        if isinstance(data, dict):
            return AdaptiveState.model_validate(data)
        if isinstance(data, AdaptiveState):
            return data
        return AdaptiveState(max_pivots=self.max_pivots)

    def _format_recent_history(self, state: AgentState, max_steps: int = 3) -> str:
        """Format recent completed steps, compressed."""
        if not state.completed_steps:
            return "No previous steps."
        recent = state.completed_steps[-max_steps:]
        return "\n".join(f"- {step}" for step in recent)

    @staticmethod
    def parse_strategy_response(response: str) -> StrategyDecision:
        """
        Parse an LLM response string into a StrategyDecision.

        Pure function: extracts JSON from the response and validates it
        against the StrategyDecision schema.

        Args:
            response: Raw LLM response text (may contain markdown fences).

        Returns:
            Validated StrategyDecision.

        Raises:
            ValueError: If the response cannot be parsed or validated.
        """
        text = response.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Remove opening fence (possibly with language tag)
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
            # Remove closing fence
            if text.endswith("```"):
                text = text[: -len("```")]
            text = text.strip()

        # Try direct JSON parse
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # Try to find a JSON object in the text
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                msg = f"No valid JSON object found in response: {text[:200]}"
                raise ValueError(msg) from exc
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError as e:
                msg = f"Failed to parse JSON from response: {e}"
                raise ValueError(msg) from e

        try:
            return StrategyDecision.model_validate(data)
        except Exception as e:
            msg = f"Invalid strategy decision: {e}"
            raise ValueError(msg) from e

    @staticmethod
    def strategy_to_policy_decision(decision: StrategyDecision) -> PolicyDecision:
        """
        Convert a StrategyDecision into a PolicyDecision.

        Pure function: maps each strategy type to the appropriate
        PolicyDecision action_type and fields.

        Args:
            decision: A validated StrategyDecision.

        Returns:
            PolicyDecision ready for the runtime to execute.
        """
        match decision.strategy:
            case StrategyType.DIRECT_ANSWER:
                return PolicyDecision(
                    action_type="complete",
                    stop_reason="direct_answer",
                    reasoning=decision.reasoning,
                    metadata={
                        "strategy": decision.strategy.value,
                        "answer": decision.action,
                    },
                )

            case StrategyType.SINGLE_TOOL:
                tool_calls: list[dict[str, Any]] = []
                if decision.tool_name:
                    tool_calls.append(
                        {
                            "name": decision.tool_name,
                            "arguments": decision.tool_arguments or {},
                        }
                    )
                return PolicyDecision(
                    action_type="tool_call",
                    tool_calls=tool_calls,
                    reasoning=decision.reasoning,
                    metadata={"strategy": decision.strategy.value},
                )

            case StrategyType.PARALLEL:
                tool_calls_parallel: list[dict[str, Any]] = [
                    {
                        "name": action.get("tool_name", ""),
                        "arguments": action.get("arguments", {}),
                    }
                    for action in (decision.parallel_actions or [])
                ]
                return PolicyDecision(
                    action_type="tool_call",
                    tool_calls=tool_calls_parallel,
                    reasoning=decision.reasoning,
                    metadata={
                        "strategy": decision.strategy.value,
                        "parallel": True,
                    },
                )

            case StrategyType.PLAN_AND_EXECUTE:
                return PolicyDecision(
                    action_type="llm_call",
                    reasoning=decision.reasoning,
                    metadata={
                        "strategy": decision.strategy.value,
                        "phase": "plan",
                        "plan": decision.plan or [],
                    },
                )

            case StrategyType.PIVOT:
                return PolicyDecision(
                    action_type="llm_call",
                    reasoning=decision.reasoning,
                    metadata={
                        "strategy": decision.strategy.value,
                        "pivot_reason": decision.pivot_reason,
                        "pivot_new_approach": decision.pivot_new_approach,
                    },
                )

            case StrategyType.SEQUENTIAL:
                return PolicyDecision(
                    action_type="llm_call",
                    reasoning=decision.reasoning,
                    metadata={"strategy": decision.strategy.value},
                )
