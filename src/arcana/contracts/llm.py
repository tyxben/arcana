"""LLM-related contracts and data models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core enums and base types
# ---------------------------------------------------------------------------


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Multi-modal content blocks
# ---------------------------------------------------------------------------


class ImageSource(BaseModel):
    """Source specification for an image in a content block."""

    type: Literal["base64", "url"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class ContentBlock(BaseModel):
    """A typed content block within a message.

    Supports text, image, image_url, tool_use, tool_result, thinking, and
    document types.  All type-specific fields are optional to allow flexible
    construction.

    For images there are two canonical representations:

    * **``image``** -- Anthropic-native format using ``source`` (base64 / URL).
    * **``image_url``** -- OpenAI-compatible format using ``image_url`` dict.

    Both are first-class; the gateway providers convert as needed.
    """

    type: Literal["text", "image", "image_url", "tool_use", "tool_result", "thinking", "document"]

    # text
    text: str | None = None

    # image (Anthropic-native)
    source: ImageSource | None = None

    # image_url (OpenAI-compatible: {"url": "data:image/png;base64,..."})
    image_url: dict[str, str] | None = None

    # tool_use
    tool_use_id: str | None = None
    name: str | None = None
    arguments: str | None = None  # JSON string

    # tool_result
    tool_call_id: str | None = None
    content: str | None = None
    is_error: bool | None = None

    # thinking
    thinking: str | None = None

    # document
    document_type: str | None = None
    document_data: str | None = None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str | list[ContentBlock] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    # For assistant messages that contain tool calls (OpenAI native format)
    tool_calls: list[ToolCallRequest] | None = None
    # Provider-specific cache control (e.g. Anthropic prompt caching)
    cache_control: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Token usage and budget
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token usage statistics for an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Prompt caching metrics (provider-specific)
    cache_creation_input_tokens: int | None = None  # Anthropic: tokens written to cache
    cache_read_input_tokens: int | None = None  # Anthropic: tokens read from cache
    cached_tokens: int | None = None  # OpenAI: cached prompt tokens

    @property
    def cost_estimate(self) -> float:
        """Estimate cost based on typical pricing (placeholder)."""
        # This is a rough estimate; actual costs vary by provider/model
        return (self.prompt_tokens * 0.001 + self.completion_tokens * 0.002) / 1000


class Budget(BaseModel):
    """Budget constraints for an LLM call or run."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_time_ms: int | None = None


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    model_config = {"protected_namespaces": ()}

    provider: str  # Provider name, validated at registry level
    model_id: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    max_tokens: int = Field(default=4096, gt=0)
    timeout_ms: int = Field(default=30000, gt=0)

    # Provider-specific options
    extra_params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


class ToolCallRequest(BaseModel):
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: str  # JSON string


# ---------------------------------------------------------------------------
# Provider-specific request extensions
# ---------------------------------------------------------------------------


class ThinkingConfig(BaseModel):
    """Configuration for extended thinking / chain-of-thought."""

    enabled: bool = False
    budget_tokens: int | None = None


class GroundingConfig(BaseModel):
    """Gemini grounding / Google Search configuration."""

    google_search: bool = False
    dynamic_retrieval_threshold: float | None = None


class SafetySetting(BaseModel):
    """A single Gemini safety setting."""

    category: str
    threshold: str


class AnthropicRequestExt(BaseModel):
    """Anthropic-specific request extensions."""

    system: str | None = None
    thinking: ThinkingConfig | None = None
    prompt_caching: bool | None = None


class GeminiRequestExt(BaseModel):
    """Gemini-specific request extensions."""

    grounding: GroundingConfig | None = None
    code_execution: bool | None = None
    safety_settings: list[SafetySetting] | None = None
    thinking: ThinkingConfig | None = None
    cached_content: str | None = None


class OpenAIRequestExt(BaseModel):
    """OpenAI-specific request extensions."""

    json_schema: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    prediction: dict[str, Any] | None = None


class OllamaRequestExt(BaseModel):
    """Ollama-specific request extensions."""

    keep_alive: str | None = None
    num_ctx: int | None = None
    num_gpu: int | None = None
    raw_mode: bool | None = None
    options: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# LLM Request
# ---------------------------------------------------------------------------


class LLMRequest(BaseModel):
    """Request to an LLM."""

    messages: list[Message]
    response_schema: dict[str, Any] | None = None  # JSON Schema for structured output (json_object mode)
    response_format: dict[str, Any] | None = None  # JSON Schema for structured output (json_schema mode)
    tools: list[dict[str, Any]] | None = None  # Tool definitions
    budget: Budget | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Provider-specific extensions (all optional for backward compatibility)
    anthropic: AnthropicRequestExt | None = None
    gemini: GeminiRequestExt | None = None
    openai: OpenAIRequestExt | None = None
    ollama: OllamaRequestExt | None = None


# ---------------------------------------------------------------------------
# Provider-specific response extensions
# ---------------------------------------------------------------------------


class ThinkingBlock(BaseModel):
    """A block of extended thinking output."""

    thinking: str


class GroundingChunk(BaseModel):
    """A grounding source reference."""

    uri: str
    title: str | None = None


class GroundingMetadata(BaseModel):
    """Metadata about grounding / search results from Gemini."""

    search_queries: list[str] | None = None
    grounding_chunks: list[GroundingChunk] | None = None
    web_search_queries: list[str] | None = None


class CodeExecutionResult(BaseModel):
    """Result of Gemini code execution."""

    code: str
    output: str
    outcome: Literal["ok", "error"]


class AnthropicResponseExt(BaseModel):
    """Anthropic-specific response extensions."""

    thinking_blocks: list[ThinkingBlock] | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    stop_reason: str | None = None


class GeminiResponseExt(BaseModel):
    """Gemini-specific response extensions."""

    grounding_metadata: GroundingMetadata | None = None
    code_execution_results: list[CodeExecutionResult] | None = None
    safety_ratings: list[dict[str, Any]] | None = None
    thinking_text: str | None = None


class OpenAIResponseExt(BaseModel):
    """OpenAI-specific response extensions."""

    logprobs: dict[str, Any] | None = None
    system_fingerprint: str | None = None


class OllamaResponseExt(BaseModel):
    """Ollama-specific response extensions."""

    eval_count: int | None = None
    eval_duration_ns: int | None = None
    load_duration_ns: int | None = None
    model_name: str | None = None


# ---------------------------------------------------------------------------
# LLM Response
# ---------------------------------------------------------------------------


class LLMResponse(BaseModel):
    """Response from an LLM."""

    model_config = {"protected_namespaces": ()}

    content: str | None = None
    tool_calls: list[ToolCallRequest] | None = None
    usage: TokenUsage
    model: str
    finish_reason: str

    # For tracing
    raw_response: dict[str, Any] | None = None

    # Provider-specific extensions (all optional for backward compatibility)
    anthropic: AnthropicResponseExt | None = None
    gemini: GeminiResponseExt | None = None
    openai: OpenAIResponseExt | None = None
    ollama: OllamaResponseExt | None = None


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class StreamChunk(BaseModel):
    """A single chunk in a streaming LLM response."""

    type: Literal[
        "text_delta",
        "tool_call_delta",
        "thinking_delta",
        "usage",
        "done",
        "error",
    ]

    # text_delta
    text: str | None = None

    # tool_call_delta
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str | None = None

    # thinking_delta
    thinking: str | None = None

    # usage (sent with "done" or standalone)
    usage: TokenUsage | None = None

    # error
    error: str | None = None

    # Provider-specific metadata
    metadata: dict[str, Any] | None = None
