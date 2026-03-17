"""Result-oriented evaluation metrics."""

from __future__ import annotations

from pydantic import BaseModel


class EvalMetrics(BaseModel):
    """Comprehensive metrics for a single evaluation run."""

    # Primary metrics
    first_attempt_success: bool = False  # Goal reached without any errors?
    goal_achievement_rate: float = 0.0  # 0.0 to 1.0, determined by judge
    result_verifiability: float = 0.0  # 0.0 to 1.0, checkable assertions?

    # Efficiency metrics
    cost_usd: float = 0.0
    tokens_used: int = 0
    steps_to_completion: int = 0
    wall_clock_ms: int = 0

    # Recovery metrics
    errors_encountered: int = 0
    recovery_attempts: int = 0
    recovery_success_rate: float = -1.0  # -1.0 = N/A (no recoveries)

    # Quality rubric scores
    rubric_scores: dict[str, float] | None = None  # dimension -> score
    rubric_weighted_average: float = 0.0

    @property
    def cost_per_success(self) -> float:
        """Cost normalized by goal achievement. Infinite when achievement is zero."""
        if self.goal_achievement_rate <= 0:
            return float("inf")
        return self.cost_usd / self.goal_achievement_rate

    @property
    def tokens_per_success(self) -> float:
        """Tokens normalized by goal achievement. Infinite when achievement is zero."""
        if self.goal_achievement_rate <= 0:
            return float("inf")
        return self.tokens_used / self.goal_achievement_rate
