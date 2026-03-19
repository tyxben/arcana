"""LLM-as-judge evaluation — sends goal + output to an LLM for scoring."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from arcana.contracts.llm import (
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
    TokenUsage,
)

if TYPE_CHECKING:
    from arcana.gateway.registry import ModelGatewayRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an evaluation judge. Score the following output against each criterion.

## Goal
{goal}

## Output to evaluate
{output}

## Criteria
{criteria_text}

## Instructions
For each criterion, provide a score from 1 to 5:
  1 = completely fails
  2 = mostly fails
  3 = partially meets
  4 = mostly meets
  5 = fully meets

Respond with ONLY valid JSON in this exact format:
{{
  "scores": [
    {{"criterion": "<criterion text>", "score": <1-5>, "reasoning": "<brief explanation>"}}
  ],
  "overall_reasoning": "<brief overall assessment>"
}}
"""


class CriterionScore(BaseModel):
    """Score for a single evaluation criterion."""

    criterion: str
    score: int = Field(ge=1, le=5)
    reasoning: str = ""


class JudgeResult(BaseModel):
    """Result of LLM-as-judge evaluation."""

    scores: list[CriterionScore] = Field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False
    overall_reasoning: str = ""
    error: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Evaluate agent output using an LLM as judge.

    Sends the goal, output, and criteria to an LLM and parses structured
    scores. Each criterion is scored 1-5. The result passes if the average
    score meets the threshold (default 3.0).
    """

    def __init__(
        self,
        gateway: ModelGatewayRegistry,
        model_config: ModelConfig | None = None,
        *,
        pass_threshold: float = 3.0,
    ) -> None:
        self.gateway = gateway
        self.model_config = model_config or ModelConfig(
            provider="default",
            model_id="default",
            temperature=0.0,
            max_tokens=2048,
        )
        self.pass_threshold = pass_threshold

    async def evaluate(
        self,
        goal: str,
        output: str,
        criteria: list[str],
    ) -> JudgeResult:
        """Evaluate output against criteria using an LLM judge.

        Args:
            goal: The original goal or task description.
            output: The agent/system output to evaluate.
            criteria: List of criteria to score against (each scored 1-5).

        Returns:
            JudgeResult with per-criterion scores and overall assessment.
        """
        if not criteria:
            return JudgeResult(
                passed=True,
                overall_score=5.0,
                overall_reasoning="No criteria specified.",
            )

        criteria_text = "\n".join(f"- {c}" for c in criteria)
        prompt = _JUDGE_PROMPT.format(
            goal=goal,
            output=output,
            criteria_text=criteria_text,
        )

        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content=prompt)],
        )

        try:
            response = await self.gateway.generate(request, self.model_config)
        except Exception as e:
            logger.warning("LLMJudge: gateway call failed: %s", e)
            return JudgeResult(error=f"LLM call failed: {e}")

        return self._parse_response(
            response.content or "",
            criteria,
            usage=response.usage,
        )

    def _parse_response(
        self,
        raw: str,
        criteria: list[str],
        *,
        usage: TokenUsage | None = None,
    ) -> JudgeResult:
        """Parse the LLM JSON response into a JudgeResult."""
        usage = usage or TokenUsage()

        # Try to extract JSON from the response (handle markdown fences)
        json_str = raw.strip()
        if json_str.startswith("```"):
            # Strip ```json ... ```
            lines = json_str.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            json_str = "\n".join(lines)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("LLMJudge: failed to parse JSON: %s", e)
            return JudgeResult(
                error=f"Failed to parse LLM response as JSON: {e}",
                usage=usage,
            )

        # Parse per-criterion scores
        raw_scores = data.get("scores", [])
        parsed_scores: list[CriterionScore] = []
        for item in raw_scores:
            try:
                score = CriterionScore(
                    criterion=str(item.get("criterion", "")),
                    score=int(item.get("score", 1)),
                    reasoning=str(item.get("reasoning", "")),
                )
                parsed_scores.append(score)
            except (ValueError, TypeError) as e:
                logger.debug("LLMJudge: skipping malformed score entry: %s", e)

        # Compute overall score
        if parsed_scores:
            overall = sum(s.score for s in parsed_scores) / len(parsed_scores)
        else:
            overall = 0.0

        return JudgeResult(
            scores=parsed_scores,
            overall_score=round(overall, 2),
            passed=overall >= self.pass_threshold,
            overall_reasoning=data.get("overall_reasoning", ""),
            usage=usage,
        )
