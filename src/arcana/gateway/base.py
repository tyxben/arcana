"""Base class for Model Gateway providers."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from arcana.contracts.llm import LLMRequest, LLMResponse, ModelConfig, StreamChunk

if TYPE_CHECKING:
    from arcana.contracts.trace import TraceContext


# ---------------------------------------------------------------------------
# Protocol (new, preferred)
# ---------------------------------------------------------------------------


@runtime_checkable
class BaseProvider(Protocol):
    """Protocol for LLM providers. Maps to Rust trait."""

    @property
    def provider_name(self) -> str: ...

    @property
    def supported_models(self) -> list[str]: ...

    async def generate(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> LLMResponse: ...

    async def stream(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    async def health_check(self) -> bool: ...


# ---------------------------------------------------------------------------
# ABC (deprecated, kept for backward compatibility)
# ---------------------------------------------------------------------------


class ModelGateway(ABC):
    """Abstract base class for LLM provider implementations.

    .. deprecated::
        Use :class:`BaseProvider` protocol instead. ``ModelGateway`` will be
        removed in a future release.

    All providers must implement the generate method to provide
    a unified interface for LLM interactions.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        warnings.warn(
            "ModelGateway is deprecated, implement the BaseProvider protocol instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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
        """Generate a response from the LLM.

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

    async def stream(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Default: generate() wrapped as a single-chunk stream."""
        response = await self.generate(request, config, trace_ctx)
        yield StreamChunk(type="text_delta", text=response.content)
        if response.tool_calls:
            for tc in response.tool_calls:
                yield StreamChunk(
                    type="tool_call_delta",
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    arguments_delta=tc.arguments,
                )
        yield StreamChunk(type="done", usage=response.usage)

    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        return True


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        retryable: bool = False,
        status_code: int | None = None,
        retry_after_ms: int | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_ms = retry_after_ms


class RateLimitError(ProviderError):
    """Raised when the provider returns a rate-limit (429) response."""

    def __init__(self, message: str, provider: str, retry_after_ms: int | None = None):
        super().__init__(
            message,
            provider=provider,
            retryable=True,
            status_code=429,
            retry_after_ms=retry_after_ms,
        )


class AuthenticationError(ProviderError):
    """Raised when provider authentication fails (401)."""

    def __init__(self, message: str, provider: str):
        super().__init__(message, provider=provider, retryable=False, status_code=401)


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not found (404)."""

    def __init__(self, message: str, provider: str):
        super().__init__(message, provider=provider, retryable=False, status_code=404)


class ContentFilterError(ProviderError):
    """Raised when content is rejected by the provider's safety filter (400)."""

    def __init__(self, message: str, provider: str):
        super().__init__(message, provider=provider, retryable=False, status_code=400)


class ContextLengthError(ProviderError):
    """Raised when the request exceeds the model's context window (400)."""

    def __init__(self, message: str, provider: str):
        super().__init__(message, provider=provider, retryable=False, status_code=400)


class BudgetExceededError(Exception):
    """Exception raised when budget limits are exceeded."""

    def __init__(self, message: str, budget_type: str):
        super().__init__(message)
        self.budget_type = budget_type
