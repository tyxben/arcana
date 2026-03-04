"""Goal verifier that checks acceptance criteria against completed steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.plan import (
    GoalVerificationResult,
    PlanStepStatus,
    VerificationOutcome,
)
from arcana.runtime.verifiers.base import BaseVerifier

if TYPE_CHECKING:
    from arcana.contracts.plan import Plan
    from arcana.contracts.state import AgentState


class GoalVerifier(BaseVerifier):
    """
    Verifies goal completion by matching completed steps against plan criteria.

    For each plan step, checks if a corresponding completed step exists
    using simple string matching. Also evaluates global acceptance criteria.
    """

    async def verify(
        self,
        state: AgentState,
        plan: Plan | None = None,
    ) -> GoalVerificationResult:
        """
        Verify whether the goal criteria are met.

        Checks:
        1. Plan step completion status (if plan provided)
        2. Global acceptance criteria against completed steps
        3. Coverage ratio calculation

        Args:
            state: Current agent state with completed_steps
            plan: Optional plan with steps and acceptance criteria

        Returns:
            GoalVerificationResult with outcome and details
        """
        criteria_results: dict[str, bool] = {}
        failed_criteria: list[str] = []
        suggestions: list[str] = []

        if plan is not None:
            # Check plan step completion
            for step in plan.steps:
                criterion = step.description
                is_met = step.status == PlanStepStatus.COMPLETED
                criteria_results[criterion] = is_met
                if not is_met:
                    failed_criteria.append(criterion)

            # Check global acceptance criteria against completed steps
            for criterion in plan.acceptance_criteria:
                is_met = self._criterion_matched(criterion, state.completed_steps)
                criteria_results[criterion] = is_met
                if not is_met:
                    failed_criteria.append(criterion)
        else:
            # No plan — check completed steps against goal
            if state.goal:
                has_steps = len(state.completed_steps) > 0
                criteria_results[state.goal] = has_steps
                if not has_steps:
                    failed_criteria.append(state.goal)

        # Calculate coverage
        total = len(criteria_results)
        passed = sum(1 for v in criteria_results.values() if v)
        coverage = passed / total if total > 0 else 0.0

        # Determine outcome
        if total == 0:
            outcome = VerificationOutcome.PASSED
            coverage = 1.0
        elif passed == total:
            outcome = VerificationOutcome.PASSED
        elif passed > 0:
            outcome = VerificationOutcome.PARTIAL
        else:
            outcome = VerificationOutcome.FAILED

        # Generate suggestions for failed criteria
        for criterion in failed_criteria:
            suggestions.append(f"Re-attempt: {criterion}")

        return GoalVerificationResult(
            outcome=outcome,
            criteria_results=criteria_results,
            coverage=coverage,
            failed_criteria=failed_criteria,
            suggestions=suggestions,
        )

    def _criterion_matched(
        self,
        criterion: str,
        completed_steps: list[str],
    ) -> bool:
        """
        Check if a criterion is matched by any completed step.

        Uses simple word overlap matching: a criterion is considered met
        if any completed step shares significant content words with it.

        Args:
            criterion: The criterion text to match
            completed_steps: List of completed step summaries

        Returns:
            True if the criterion appears to be met
        """
        if not completed_steps:
            return False

        criterion_words = self._extract_content_words(criterion)
        if not criterion_words:
            return False

        for step in completed_steps:
            step_words = self._extract_content_words(step)
            overlap = criterion_words & step_words
            # Consider matched if at least 2 content words overlap
            # or if the criterion is very short (1-2 words)
            min_overlap = min(2, len(criterion_words))
            if len(overlap) >= min_overlap:
                return True

        return False

    def _extract_content_words(self, text: str) -> set[str]:
        """Extract meaningful content words from text (length > 3)."""
        return {w.lower() for w in text.split() if len(w) > 3}
