"""Google Gemini provider.

This is a convenience wrapper around OpenAICompatibleProvider
for the Gemini API's OpenAI-compatible endpoint.
"""

from __future__ import annotations

from arcana.gateway.providers.openai_compatible import (
    OpenAICompatibleProvider,
    create_gemini_provider,
)
from arcana.trace.writer import TraceWriter

# Re-export for backwards compatibility
__all__ = ["GeminiProvider", "create_gemini_provider"]


class GeminiProvider(OpenAICompatibleProvider):
    """
    Google Gemini provider using the OpenAI-compatible API.

    Uses the endpoint: https://generativelanguage.googleapis.com/v1beta/openai

    This is a thin wrapper around OpenAICompatibleProvider with
    Gemini-specific defaults.
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    DEFAULT_MODEL = "gemini-2.0-flash"
    SUPPORTED_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        trace_writer: TraceWriter | None = None,
    ):
        """
        Initialize the Gemini provider.

        Args:
            api_key: Google AI API key
            base_url: API base URL (optional, uses default Gemini endpoint)
            trace_writer: Optional trace writer for logging
        """
        super().__init__(
            provider_name="gemini",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            default_model=self.DEFAULT_MODEL,
            supported_models=self.SUPPORTED_MODELS,
            trace_writer=trace_writer,
        )
