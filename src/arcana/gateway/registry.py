"""Registry for managing multiple LLM providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.llm import LLMRequest, LLMResponse, ModelConfig
from arcana.contracts.trace import TraceContext
from arcana.gateway.base import ModelGateway, ProviderError

if TYPE_CHECKING:
    pass


class ModelGatewayRegistry:
    """
    Registry for managing multiple LLM providers.

    Supports:
    - Registering multiple providers
    - Routing requests to the appropriate provider
    - Fallback chains for reliability
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._providers: dict[str, ModelGateway] = {}
        self._fallback_chains: dict[str, list[str]] = {}
        self._default_provider: str | None = None

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

    async def generate(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
        use_fallback: bool = True,
    ) -> LLMResponse:
        """
        Generate a response using the specified provider.

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
        provider_name = config.provider
        provider = self._providers.get(provider_name)

        # Fall back to default provider if specified provider not found
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

        # Try primary provider
        try:
            return await provider.generate(request, config, trace_ctx)
        except ProviderError as e:
            if not use_fallback or not e.retryable:
                raise

            # Try fallback chain
            fallbacks = self._fallback_chains.get(provider_name, [])
            last_error = e

            for fallback_name in fallbacks:
                fallback = self._providers.get(fallback_name)
                if fallback is None:
                    continue

                # Create new config for fallback
                fallback_config = config.model_copy(update={"provider": fallback_name})

                try:
                    return await fallback.generate(request, fallback_config, trace_ctx)
                except ProviderError as fallback_error:
                    last_error = fallback_error
                    if not fallback_error.retryable:
                        raise

            # All providers failed
            raise last_error from last_error

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
