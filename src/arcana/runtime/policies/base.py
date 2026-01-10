"""Base policy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcana.contracts.runtime import PolicyDecision
    from arcana.contracts.state import AgentState


class BasePolicy(ABC):
    """
    Abstract base class for agent policies.

    A policy decides what action to take given the current state.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for identification."""
        ...

    @abstractmethod
    async def decide(self, state: AgentState) -> PolicyDecision:
        """
        Decide the next action based on current state.

        Args:
            state: Current agent state

        Returns:
            PolicyDecision describing the next action
        """
        ...

    def build_system_prompt(self, state: AgentState) -> str:
        """
        Build system prompt for LLM.

        Override in subclasses for custom prompts.
        """
        return f"You are a helpful AI assistant. Your goal is: {state.goal}"
