"""DeepSeek provider.

This is a convenience wrapper around OpenAICompatibleProvider
for the DeepSeek API.
"""

from __future__ import annotations

from arcana.gateway.providers.openai_compatible import (
    OpenAICompatibleProvider,
    create_deepseek_provider,
)
from arcana.trace.writer import TraceWriter

# Re-export for backwards compatibility
__all__ = ["DeepSeekProvider", "create_deepseek_provider"]


class DeepSeekProvider(OpenAICompatibleProvider):
    """
    DeepSeek provider using the OpenAI-compatible API.

    Uses the endpoint: https://api.deepseek.com

    This is a thin wrapper around OpenAICompatibleProvider with
    DeepSeek-specific defaults.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"
    SUPPORTED_MODELS = [
        "deepseek-chat",
        "deepseek-coder",
        "deepseek-reasoner",
    ]

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        trace_writer: TraceWriter | None = None,
    ):
        """
        Initialize the DeepSeek provider.

        Args:
            api_key: DeepSeek API key
            base_url: API base URL (optional, uses default DeepSeek endpoint)
            trace_writer: Optional trace writer for logging
        """
        super().__init__(
            provider_name="deepseek",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            default_model=self.DEFAULT_MODEL,
            supported_models=self.SUPPORTED_MODELS,
            trace_writer=trace_writer,
        )
