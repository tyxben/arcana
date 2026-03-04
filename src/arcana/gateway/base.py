"""Base class for Model Gateway providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcana.contracts.llm import LLMRequest, LLMResponse, ModelConfig
    from arcana.contracts.trace import TraceContext


class ModelGateway(ABC):
    """
    Abstract base class for LLM provider implementations.

    All providers must implement the generate method to provide
    a unified interface for LLM interactions.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of this provider."""
        ...

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """Get list of model IDs supported by this provider."""
        ...

    @abstractmethod
    async def generate(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            request: The LLM request containing messages and options
            config: Model configuration (model ID, temperature, etc.)
            trace_ctx: Optional trace context for logging

        Returns:
            LLMResponse containing the model's response

        Raises:
            ProviderError: If the provider encounters an error
            TimeoutError: If the request times out
            BudgetExceededError: If budget limits are exceeded
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        return True


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        retryable: bool = False,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class BudgetExceededError(Exception):
    """Exception raised when budget limits are exceeded."""

    def __init__(self, message: str, budget_type: str):
        super().__init__(message)
        self.budget_type = budget_type
