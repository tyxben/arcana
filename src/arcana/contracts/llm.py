"""LLM-related contracts and data models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None


class TokenUsage(BaseModel):
    """Token usage statistics for an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

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


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    model_config = {"protected_namespaces": ()}

    provider: Literal["gemini", "deepseek", "openai", "anthropic", "ollama"]
    model_id: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    max_tokens: int = Field(default=4096, gt=0)
    timeout_ms: int = Field(default=30000, gt=0)

    # Provider-specific options
    extra_params: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: str  # JSON string


class LLMRequest(BaseModel):
    """Request to an LLM."""

    messages: list[Message]
    response_schema: dict[str, Any] | None = None  # JSON Schema for structured output
    tools: list[dict[str, Any]] | None = None  # Tool definitions
    budget: Budget | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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
