"""Context management contracts for Working Set."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContextLayer(str, Enum):
    IDENTITY = "identity"
    TASK = "task"
    WORKING = "working"
    EXTERNAL = "external"


class ContextBlock(BaseModel):
    """A discrete block of context content."""

    layer: ContextLayer
    key: str  # e.g., "tool:file_read", "history:step_3"
    content: str
    token_count: int
    priority: float = 0.5  # 0.0 (drop first) to 1.0 (drop last)
    compressible: bool = True
    source: str | None = None


class TokenBudget(BaseModel):
    """Token allocation for a single LLM call."""

    total_window: int = 128000
    identity_tokens: int = 200
    task_tokens: int = 300
    response_reserve: int = 4096

    @property
    def working_budget(self) -> int:
        return (
            self.total_window
            - self.identity_tokens
            - self.task_tokens
            - self.response_reserve
        )


class StepContext(BaseModel):
    """What the current step needs."""

    step_type: str = "think"
    needs_tools: bool = False
    needs_memory: bool = False
    relevant_tool_names: list[str] | None = None
    memory_query: str | None = None
    previous_error: dict[str, Any] | None = None  # ErrorDiagnosis dict
    focus_instruction: str | None = None


class WorkingSet(BaseModel):
    """The assembled context for a single LLM call."""

    identity: ContextBlock
    task: ContextBlock
    working_blocks: list[ContextBlock] = Field(default_factory=list)
    total_tokens: int = 0
    dropped_keys: list[str] = Field(default_factory=list)
    compressed_keys: list[str] = Field(default_factory=list)
