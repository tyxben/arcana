"""Tool-related contracts for the Tool Gateway."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

ASK_USER_TOOL_NAME = "ask_user"


class SideEffect(str, Enum):
    """Type of side effect a tool may have."""

    READ = "read"
    WRITE = "write"
    NONE = "none"


class ErrorType(str, Enum):
    """Classification of tool errors."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    REQUIRES_HUMAN = "requires_human"


class ToolSpec(BaseModel):
    """Specification of a tool."""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema
    output_schema: dict[str, Any] | None = None  # JSON Schema

    # Capabilities and constraints
    side_effect: SideEffect = SideEffect.READ
    requires_confirmation: bool = False
    capabilities: list[str] = Field(default_factory=list)

    # Retry configuration
    max_retries: int = 3
    retry_delay_ms: int = 1000
    timeout_ms: int = 30000

    # Affordance fields (for LLM understanding)
    # These are optional and backward-compatible. They are consumed only by
    # format_tool_for_llm() and ToolMatcher, NOT sent to OpenAI function calling API.
    when_to_use: str | None = None
    what_to_expect: str | None = None
    failure_meaning: str | None = None
    success_next_step: str | None = None

    # Categorization
    category: str | None = None  # "search", "file", "code", "web", "data", "shell"
    related_tools: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    """A tool call to be executed."""

    id: str
    name: str
    arguments: dict[str, Any]
    idempotency_key: str | None = None

    # Context
    run_id: str | None = None
    step_id: str | None = None
    # Causal parent for trace — the step_id of the LLM turn that emitted
    # this call. Propagated into TOOL_CALL trace events so ``arcana trace
    # flow`` and ``arcana trace explain`` can stitch the DAG together.
    parent_step_id: str | None = None


class ToolError(BaseModel):
    """Error from tool execution."""

    error_type: ErrorType
    message: str
    code: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_retryable(self) -> bool:
        return self.error_type == ErrorType.RETRYABLE


class ToolResult(BaseModel):
    """Result of tool execution."""

    tool_call_id: str
    name: str
    success: bool
    output: Any | None = None
    error: ToolError | None = None

    # Execution metadata
    duration_ms: int | None = None
    retry_count: int = 0

    @property
    def output_str(self) -> str:
        """Get output as string for LLM consumption."""
        if self.error:
            return f"Error: {self.error.message}"
        if self.output is None:
            return "Success (no output)"
        if isinstance(self.output, str):
            return self.output
        import json

        return json.dumps(self.output, ensure_ascii=False)
