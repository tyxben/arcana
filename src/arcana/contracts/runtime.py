"""Runtime-related contracts for agent execution."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from arcana.contracts.llm import LLMResponse
from arcana.contracts.tool import ToolResult


class StepType(str, Enum):
    """Type of step being executed."""

    THINK = "think"  # LLM reasoning
    ACT = "act"  # Tool execution
    OBSERVE = "observe"  # Process results
    PLAN = "plan"  # Planning step
    VERIFY = "verify"  # Verification step


class StepResult(BaseModel):
    """Result of executing a single step."""

    step_type: StepType
    step_id: str
    success: bool

    # Outputs
    thought: str | None = None
    action: str | None = None
    observation: str | None = None

    # LLM interaction
    llm_response: LLMResponse | None = None

    # Tool interaction
    tool_results: list[ToolResult] = Field(default_factory=list)

    # State changes
    state_updates: dict[str, Any] = Field(default_factory=dict)
    memory_updates: dict[str, Any] = Field(default_factory=dict)

    # Error handling
    error: str | None = None
    is_recoverable: bool = True


class PolicyDecision(BaseModel):
    """Decision made by a policy about what to do next."""

    action_type: str  # "llm_call", "tool_call", "complete", "fail"

    # For LLM calls
    prompt_template: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)

    # For tool calls
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    # For completion
    stop_reason: str | None = None

    # Metadata
    reasoning: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    """Configuration for the agent runtime."""

    # Execution limits
    max_steps: int = 100
    max_consecutive_errors: int = 3
    max_consecutive_no_progress: int = 3

    # Checkpointing
    checkpoint_interval_steps: int = 5
    checkpoint_on_error: bool = True
    checkpoint_budget_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 0.9]
    )

    # Retry configuration
    step_retry_count: int = 2
    step_retry_delay_ms: int = 1000

    # Progress detection
    progress_window_size: int = 5
    similarity_threshold: float = 0.95
