"""Capability registry for LLM providers.

Defines the set of capabilities each provider supports and provides
a registry for querying provider capabilities at runtime.
"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    """All known provider capabilities."""

    # Universal
    CHAT = "chat"
    STREAMING = "streaming"
    TOOL_USE = "tool_use"
    MULTIMODAL_INPUT = "multimodal_input"
    STRUCTURED_OUTPUT = "structured_output"

    # Anthropic
    EXTENDED_THINKING = "extended_thinking"
    PROMPT_CACHING = "prompt_caching"
    COMPUTER_USE = "computer_use"
    PDF_INPUT = "pdf_input"

    # Gemini
    GROUNDING = "grounding"
    CODE_EXECUTION = "code_execution"
    SAFETY_SETTINGS = "safety_settings"
    CACHED_CONTENT = "cached_content"

    # OpenAI
    JSON_SCHEMA_OUTPUT = "json_schema_output"
    PARALLEL_TOOL_CALLS = "parallel_tool_calls"
    LOGPROBS = "logprobs"
    PREDICTED_OUTPUT = "predicted_output"

    # Ollama
    LOCAL_EXECUTION = "local_execution"
    MODEL_MANAGEMENT = "model_management"
    RAW_GENERATE = "raw_generate"

    # Chinese providers
    DEEP_THINKING = "deep_thinking"
    WEB_SEARCH = "web_search"
    LONG_CONTEXT = "long_context"
    CODE_INTERPRETER = "code_interpreter"
    TEXT_TO_AUDIO = "text_to_audio"


# ---------------------------------------------------------------------------
# Static capability map per provider
# ---------------------------------------------------------------------------

PROVIDER_CAPABILITIES: dict[str, frozenset[Capability]] = {
    "anthropic": frozenset({
        Capability.CHAT,
        Capability.STREAMING,
        Capability.TOOL_USE,
        Capability.MULTIMODAL_INPUT,
        Capability.STRUCTURED_OUTPUT,
        Capability.EXTENDED_THINKING,
        Capability.PROMPT_CACHING,
        Capability.COMPUTER_USE,
        Capability.PDF_INPUT,
    }),
    "gemini": frozenset({
        Capability.CHAT,
        Capability.STREAMING,
        Capability.TOOL_USE,
        Capability.MULTIMODAL_INPUT,
        Capability.STRUCTURED_OUTPUT,
        Capability.GROUNDING,
        Capability.CODE_EXECUTION,
        Capability.SAFETY_SETTINGS,
        Capability.CACHED_CONTENT,
    }),
    "openai": frozenset({
        Capability.CHAT,
        Capability.STREAMING,
        Capability.TOOL_USE,
        Capability.MULTIMODAL_INPUT,
        Capability.STRUCTURED_OUTPUT,
        Capability.JSON_SCHEMA_OUTPUT,
        Capability.PARALLEL_TOOL_CALLS,
        Capability.LOGPROBS,
        Capability.PREDICTED_OUTPUT,
    }),
    "ollama": frozenset({
        Capability.CHAT,
        Capability.STREAMING,
        Capability.TOOL_USE,
        Capability.LOCAL_EXECUTION,
        Capability.MODEL_MANAGEMENT,
        Capability.RAW_GENERATE,
    }),
    # Chinese providers
    "deepseek": frozenset({
        Capability.CHAT,
        Capability.STREAMING,
        Capability.TOOL_USE,
        Capability.STRUCTURED_OUTPUT,
        Capability.DEEP_THINKING,
    }),
    "kimi": frozenset({
        Capability.CHAT,
        Capability.STREAMING,
        Capability.TOOL_USE,
        Capability.LONG_CONTEXT,
        Capability.WEB_SEARCH,
    }),
    "glm": frozenset({
        Capability.CHAT,
        Capability.STREAMING,
        Capability.TOOL_USE,
        Capability.MULTIMODAL_INPUT,
        Capability.WEB_SEARCH,
        Capability.CODE_INTERPRETER,
    }),
    "minimax": frozenset({
        Capability.CHAT,
        Capability.STREAMING,
        Capability.TOOL_USE,
        Capability.LONG_CONTEXT,
        Capability.TEXT_TO_AUDIO,
    }),
}


# ---------------------------------------------------------------------------
# CapabilityRegistry
# ---------------------------------------------------------------------------


class CapabilityRegistry:
    """Runtime registry for querying and managing provider capabilities.

    Initialised from ``PROVIDER_CAPABILITIES`` by default but allows
    runtime registration of additional providers.
    """

    def __init__(
        self,
        capabilities: dict[str, frozenset[Capability]] | None = None,
    ) -> None:
        self._caps: dict[str, frozenset[Capability]] = dict(
            capabilities or PROVIDER_CAPABILITIES
        )

    # -- queries ------------------------------------------------------------

    def supports(self, provider: str, capability: Capability) -> bool:
        """Return ``True`` if *provider* supports *capability*."""
        return capability in self._caps.get(provider, frozenset())

    def capabilities_of(self, provider: str) -> frozenset[Capability]:
        """Return the full capability set of *provider* (empty if unknown)."""
        return self._caps.get(provider, frozenset())

    def providers_with(self, capability: Capability) -> list[str]:
        """Return all providers that support *capability*."""
        return [
            name
            for name, caps in self._caps.items()
            if capability in caps
        ]

    # -- mutations ----------------------------------------------------------

    def register(
        self, provider: str, capabilities: frozenset[Capability]
    ) -> None:
        """Register (or replace) the capability set for *provider*."""
        self._caps[provider] = capabilities

    # -- selection ----------------------------------------------------------

    def best_provider_for(
        self,
        required: set[Capability],
        preferred: list[str] | None = None,
    ) -> str | None:
        """Return the best provider that satisfies all *required* capabilities.

        Selection strategy:
        1. Filter to providers that have **all** required capabilities.
        2. Among those, return the first match from *preferred* (if given).
        3. Otherwise return the provider with the largest overall capability
           set (breaking ties by name for determinism).
        4. Return ``None`` if no provider qualifies.
        """
        qualified = {
            name: caps
            for name, caps in self._caps.items()
            if required <= caps
        }
        if not qualified:
            return None

        # Honour preference order
        if preferred:
            for name in preferred:
                if name in qualified:
                    return name

        # Fall back to the provider with the most capabilities
        return max(qualified, key=lambda n: (len(qualified[n]), n))
