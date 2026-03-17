"""LLM Provider implementations.

All providers inherit from OpenAICompatibleProvider, which provides
a unified implementation for OpenAI-compatible APIs.

Usage:
    # Use specific provider classes
    from arcana.gateway.providers import GeminiProvider, DeepSeekProvider

    gemini = GeminiProvider(api_key="...")
    deepseek = DeepSeekProvider(api_key="...")

    # Or use the base class directly for any OpenAI-compatible API
    from arcana.gateway.providers import OpenAICompatibleProvider

    custom = OpenAICompatibleProvider(
        provider_name="custom",
        api_key="...",
        base_url="https://api.example.com/v1",
        default_model="model-name",
    )

    # Or use factory functions
    from arcana.gateway.providers import create_gemini_provider, create_ollama_provider

    gemini = create_gemini_provider(api_key="...")
    ollama = create_ollama_provider()  # Local, no API key needed
"""

from arcana.gateway.providers.deepseek import DeepSeekProvider, create_deepseek_provider
from arcana.gateway.providers.gemini import GeminiProvider, create_gemini_provider
from arcana.gateway.providers.openai_compatible import (
    OpenAICompatibleProvider,
    create_glm_provider,
    create_kimi_provider,
    create_minimax_provider,
    create_ollama_provider,
)

__all__ = [
    # Base class
    "OpenAICompatibleProvider",
    # Specific providers
    "GeminiProvider",
    "DeepSeekProvider",
    # Factory functions
    "create_gemini_provider",
    "create_deepseek_provider",
    "create_glm_provider",
    "create_kimi_provider",
    "create_minimax_provider",
    "create_ollama_provider",
]
