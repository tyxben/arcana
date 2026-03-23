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

    # Per-layer hard caps (None = no cap, use remaining budget)
    tool_budget: int | None = None
    history_budget: int | None = None
    memory_budget: int | None = None

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


class ContextDecision(BaseModel):
    """Record of why context was composed this way for a single LLM call.

    Every turn, the WorkingSetBuilder produces one of these. It answers:
    - Was anything compressed or dropped?
    - How full is the context window?
    - Where did the tokens go?
    - What information was lost?
    """

    turn: int = 0

    # Budget breakdown (tokens)
    budget_total: int = 0
    budget_used: int = 0
    budget_tools: int = 0
    budget_reserve: int = 0

    # Message counts
    messages_in: int = 0
    messages_out: int = 0
    compressed_count: int = 0

    # Flags
    memory_injected: bool = False
    history_compressed: bool = False

    # What was compressed/dropped (message role:summary pairs)
    compressed_messages: list[str] = Field(default_factory=list)

    # Human-readable explanation
    explanation: str = ""
