"""Base reducer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcana.contracts.runtime import StepResult
    from arcana.contracts.state import AgentState


class BaseReducer(ABC):
    """
    Abstract base class for state reducers.

    A reducer takes the current state and a step result,
    producing a new state.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Reducer name for identification."""
        ...

    @abstractmethod
    async def reduce(
        self,
        state: AgentState,
        step_result: StepResult,
    ) -> AgentState:
        """
        Apply step result to state, producing new state.

        Args:
            state: Current state
            step_result: Result of executed step

        Returns:
            Updated state
        """
        ...
