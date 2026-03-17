"""Evaluation judges -- rule-based, LLM-based, and hybrid."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from arcana.eval.metrics import EvalMetrics

if TYPE_CHECKING:
    from arcana.contracts.state import AgentState


class EvalJudge(ABC):
    """Base class for evaluation judges."""

    @abstractmethod
    async def evaluate(
        self,
        goal: str,
        state: AgentState,
        success_criteria: list[str],
        quality_rubric: dict[str, float] | None = None,
    ) -> EvalMetrics:
        """Evaluate agent state against goal and criteria, returning metrics."""
        ...


class RuleJudge(EvalJudge):
    """Deterministic rule-based judge.

    Checks state.status == COMPLETED, scans working_memory for keyword
    matches against success_criteria, and computes efficiency metrics.
    """

    async def evaluate(
        self,
        goal: str,
        state: AgentState,
        success_criteria: list[str],
        quality_rubric: dict[str, float] | None = None,
    ) -> EvalMetrics:
        from arcana.contracts.state import ExecutionStatus

        # Basic completion check
        completed = state.status == ExecutionStatus.COMPLETED
        no_errors = state.consecutive_errors == 0

        # Check success_criteria keywords against working_memory values
        memory_text = " ".join(str(v) for v in state.working_memory.values())
        matched = 0
        for criterion in success_criteria:
            # Simple keyword presence check
            if criterion.lower() in memory_text.lower():
                matched += 1

        total = len(success_criteria)
        achievement = matched / total if total > 0 else (1.0 if completed else 0.0)

        return EvalMetrics(
            first_attempt_success=completed and no_errors,
            goal_achievement_rate=achievement,
            result_verifiability=0.0,
            cost_usd=state.cost_usd,
            tokens_used=state.tokens_used,
            steps_to_completion=state.current_step,
            wall_clock_ms=state.elapsed_ms,
            errors_encountered=state.consecutive_errors,
        )


class LLMJudge(EvalJudge):
    """LLM-based semantic judge.

    Constructs a judgment prompt from the goal, criteria, and agent output,
    then asks an LLM to return structured scores.
    """

    def __init__(self, gateway: object, model_config: object | None = None) -> None:
        self.gateway = gateway
        self.model_config = model_config

    async def evaluate(
        self,
        goal: str,
        state: AgentState,
        success_criteria: list[str],
        quality_rubric: dict[str, float] | None = None,
    ) -> EvalMetrics:
        # Placeholder: build judge prompt, call LLM, parse JSON response.
        # Full implementation will construct:
        #   - goal + criteria + agent output -> prompt
        #   - LLM returns JSON: {goal_achievement, rubric_scores, reasoning}
        raise NotImplementedError("LLMJudge requires gateway integration (not yet wired)")


class HybridJudge(EvalJudge):
    """Rule-first, LLM for uncertain criteria.

    Runs RuleJudge first. If goal_achievement_rate < 0.5 and an LLM judge
    is available, delegates to LLMJudge for a re-evaluation.
    """

    def __init__(
        self,
        gateway: object | None = None,
        model_config: object | None = None,
    ) -> None:
        self.rule_judge = RuleJudge()
        self.llm_judge = LLMJudge(gateway, model_config) if gateway else None

    async def evaluate(
        self,
        goal: str,
        state: AgentState,
        success_criteria: list[str],
        quality_rubric: dict[str, float] | None = None,
    ) -> EvalMetrics:
        metrics = await self.rule_judge.evaluate(goal, state, success_criteria, quality_rubric)

        # If rule-based judge is uncertain and LLM judge is available, re-evaluate
        if metrics.goal_achievement_rate < 0.5 and self.llm_judge is not None:
            metrics = await self.llm_judge.evaluate(goal, state, success_criteria, quality_rubric)

        return metrics
