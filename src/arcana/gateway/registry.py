"""Registry for managing multiple LLM providers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from arcana.contracts.llm import LLMRequest, LLMResponse, ModelConfig, StreamChunk
from arcana.contracts.trace import TraceContext
from arcana.gateway.base import ModelGateway, ProviderError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ModelGatewayRegistry:
    """
    Registry for managing multiple LLM providers.

    Supports:
    - Registering multiple providers
    - Routing requests to the appropriate provider
    - Fallback chains for reliability
    - Provider-level retry with exponential backoff
    """

    def __init__(
        self,
        max_retries: int = 2,
        retry_base_delay_ms: int = 500,
    ) -> None:
        """Initialize the registry.

        Args:
            max_retries: Max retry attempts before falling back (0 = no retries)
            retry_base_delay_ms: Base delay for exponential backoff in ms
        """
        self._providers: dict[str, ModelGateway] = {}
        self._fallback_chains: dict[str, list[str]] = {}
        self._default_provider: str | None = None
        self._max_retries = max_retries
        self._retry_base_delay_ms = retry_base_delay_ms

    def register(self, name: str, provider: ModelGateway) -> None:
        """
        Register a provider.

        Args:
            name: Provider name (e.g., "gemini", "deepseek")
            provider: The provider instance
        """
        self._providers[name] = provider

    def unregister(self, name: str) -> bool:
        """
        Unregister a provider.

        Args:
            name: Provider name

        Returns:
            True if removed, False if not found
        """
        if name in self._providers:
            del self._providers[name]
            return True
        return False

    def set_default(self, name: str) -> None:
        """
        Set the default provider.

        Args:
            name: Provider name to use as default
        """
        if name not in self._providers:
            registered = self.list_providers()
            raise KeyError(
                f"Provider '{name}' is not registered. "
                f"Registered providers: {registered}. "
                f"Register it first: Runtime(providers={{'{name}': 'your-api-key'}})"
            )
        self._default_provider = name

    @property
    def default_provider(self) -> str | None:
        """Get the default provider name."""
        return self._default_provider

    def get(self, name: str) -> ModelGateway | None:
        """
        Get a provider by name.

        Args:
            name: Provider name

        Returns:
            The provider instance or None
        """
        return self._providers.get(name)

    def set_fallback_chain(self, primary: str, fallbacks: list[str]) -> None:
        """
        Set a fallback chain for a provider.

        Args:
            primary: Primary provider name
            fallbacks: List of fallback provider names in priority order
        """
        self._fallback_chains[primary] = fallbacks

    def list_providers(self) -> list[str]:
        """Get list of registered provider names."""
        return list(self._providers.keys())

    def get_fallback_chain(self, primary: str) -> list[str]:
        """Get the fallback chain for a provider.

        Returns:
            List of fallback provider names in priority order, or empty list.
        """
        return list(self._fallback_chains.get(primary, []))

    def _resolve_provider(
        self, config: ModelConfig
    ) -> tuple[str, ModelGateway]:
        """Resolve provider from config, falling back to default."""
        provider_name = config.provider
        provider = self._providers.get(provider_name)

        if provider is None and self._default_provider:
            provider_name = self._default_provider
            provider = self._providers.get(provider_name)

        if provider is None:
            registered = self.list_providers()
            raise KeyError(
                f"Provider '{config.provider}' is not registered. "
                f"Registered providers: {registered}. "
                f"Register it first: Runtime(providers={{'{config.provider}': 'your-api-key'}})"
            )
        return provider_name, provider

    def _compute_retry_delay(self, attempt: int, error: ProviderError) -> float:
        """Compute retry delay with exponential backoff."""
        delay: float = self._retry_base_delay_ms * (2 ** (attempt - 1)) / 1000
        if error.retry_after_ms is not None:
            hint: float = error.retry_after_ms / 1000
            delay = max(delay, hint)
        return delay

    async def generate(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
        use_fallback: bool = True,
    ) -> LLMResponse:
        """
        Generate a response using the specified provider.

        Retries with exponential backoff on retryable errors, then
        iterates the fallback chain if configured.

        Args:
            request: The LLM request
            config: Model configuration
            trace_ctx: Optional trace context
            use_fallback: Whether to use fallback providers on failure

        Returns:
            LLMResponse from the model

        Raises:
            ProviderError: If all providers fail
            KeyError: If the provider is not registered
        """
        provider_name, provider = self._resolve_provider(config)

        # Try primary provider with retries
        last_error: ProviderError | None = None
        for attempt in range(1 + self._max_retries):
            try:
                if attempt > 0 and last_error is not None:
                    delay = self._compute_retry_delay(attempt, last_error)
                    logger.info(
                        "Retry %d/%d for provider '%s' (delay=%.1fs)",
                        attempt, self._max_retries, provider_name, delay,
                    )
                    await asyncio.sleep(delay)
                return await provider.generate(request, config, trace_ctx)
            except ProviderError as e:
                if not e.retryable:
                    raise
                last_error = e

        # Retries exhausted — try fallback chain
        if not use_fallback or last_error is None:
            raise last_error  # type: ignore[misc]

        fallbacks = self._fallback_chains.get(provider_name, [])
        for fallback_name in fallbacks:
            fallback = self._providers.get(fallback_name)
            if fallback is None:
                continue
            logger.info(
                "Provider '%s' exhausted retries; falling back to '%s'",
                provider_name, fallback_name,
            )
            fallback_config = config.model_copy(update={"provider": fallback_name})
            for attempt in range(1 + self._max_retries):
                try:
                    if attempt > 0 and last_error is not None:
                        delay = self._compute_retry_delay(attempt, last_error)
                        logger.info(
                            "Retry %d/%d for fallback provider '%s' (delay=%.1fs)",
                            attempt, self._max_retries, fallback_name, delay,
                        )
                        await asyncio.sleep(delay)
                    return await fallback.generate(request, fallback_config, trace_ctx)
                except ProviderError as fallback_error:
                    last_error = fallback_error
                    if not fallback_error.retryable:
                        raise

        raise last_error

    async def batch_generate(
        self,
        requests: list[LLMRequest],
        config: ModelConfig,
        *,
        concurrency: int = 5,
        trace_ctx: TraceContext | None = None,
    ) -> list[LLMResponse | ProviderError]:
        """Generate responses for multiple requests concurrently.

        Delegates to the provider's ``batch_generate`` if available,
        otherwise falls back to running ``generate`` calls concurrently
        with a semaphore.

        Each request is independent -- failures in one don't affect others.
        Failed requests return the ProviderError instance in that position.

        Args:
            requests: List of LLM requests to process.
            config: Model configuration (shared across all requests).
            concurrency: Maximum number of concurrent API calls (default 5).
            trace_ctx: Optional trace context for logging.

        Returns:
            List of LLMResponse or ProviderError, one per request,
            preserving input order.
        """
        if not requests:
            return []

        _provider_name, provider = self._resolve_provider(config)

        # Prefer provider-level batch if available
        if hasattr(provider, "batch_generate"):
            return await provider.batch_generate(
                requests, config, concurrency=concurrency, trace_ctx=trace_ctx,
            )

        # Fallback: semaphore + gather over self.generate (includes retry/fallback)
        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded(request: LLMRequest) -> LLMResponse | ProviderError:
            async with semaphore:
                try:
                    return await self.generate(request, config, trace_ctx)
                except ProviderError as e:
                    return e

        results = await asyncio.gather(*[_guarded(req) for req in requests])
        return list(results)

    async def stream(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
        use_fallback: bool = True,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response chunks from the specified provider.

        Retries with exponential backoff on retryable errors, then
        iterates the fallback chain if configured. Fallback only occurs
        if no chunks have been yielded yet (mid-stream errors propagate).
        """
        provider_name, provider = self._resolve_provider(config)

        # Build ordered list: (name, provider, config) — primary + fallbacks
        candidates: list[tuple[str, ModelGateway, ModelConfig]] = [
            (provider_name, provider, config),
        ]
        if use_fallback:
            for fb_name in self._fallback_chains.get(provider_name, []):
                fb = self._providers.get(fb_name)
                if fb is not None:
                    candidates.append(
                        (fb_name, fb, config.model_copy(update={"provider": fb_name}))
                    )

        last_error: ProviderError | None = None

        for cand_name, cand_provider, cand_config in candidates:
            for attempt in range(1 + self._max_retries):
                if attempt > 0 and last_error is not None:
                    delay = self._compute_retry_delay(attempt, last_error)
                    logger.info(
                        "Stream retry %d/%d for provider '%s' (delay=%.1fs)",
                        attempt, self._max_retries, cand_name, delay,
                    )
                    await asyncio.sleep(delay)

                yielded = False
                try:
                    async for chunk in cand_provider.stream(
                        request, cand_config, trace_ctx
                    ):
                        yielded = True
                        yield chunk
                    return  # stream completed successfully
                except ProviderError as e:
                    if not e.retryable or yielded:
                        raise
                    last_error = e
            # Retries exhausted for this provider, try next candidate

        if last_error is not None:
            raise last_error

    async def close(self) -> None:
        """Close all registered providers that support it.

        Iterates every provider and calls ``close()`` if available,
        releasing HTTP connection pools and other resources.
        """
        for provider in self._providers.values():
            if hasattr(provider, "close"):
                await provider.close()

    async def health_check_all(self) -> dict[str, bool]:
        """
        Check health of all registered providers.

        Returns:
            Dict mapping provider name to health status
        """
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception:
                results[name] = False
        return results
