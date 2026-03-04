"""Base verifier interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcana.contracts.plan import GoalVerificationResult, Plan
    from arcana.contracts.state import AgentState


class BaseVerifier(ABC):
    """
    Abstract base class for goal/plan verifiers.

    A verifier checks whether an agent's goal or plan criteria
    have been satisfied.
    """

    @abstractmethod
    async def verify(
        self,
        state: AgentState,
        plan: Plan | None = None,
    ) -> GoalVerificationResult:
        """
        Verify whether the goal/plan criteria are met.

        Args:
            state: Current agent state
            plan: Optional plan with acceptance criteria

        Returns:
            GoalVerificationResult with outcome, coverage, and details
        """
        ...
