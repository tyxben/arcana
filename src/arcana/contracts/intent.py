"""Intent classification contracts and data models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """The four execution paths."""

    DIRECT_ANSWER = "direct_answer"  # Single LLM call, no tools
    SINGLE_TOOL = "single_tool"  # One tool call + response
    AGENT_LOOP = "agent_loop"  # Multi-step adaptive execution
    COMPLEX_PLAN = "complex_plan"  # Explicit planning + execution


class IntentClassification(BaseModel):
    """Result of intent classification."""

    intent: IntentType
    confidence: float  # 0.0 to 1.0
    reasoning: str | None = None  # Why this classification
    suggested_tools: list[str] = Field(default_factory=list)  # For SINGLE_TOOL: which tool
    complexity_estimate: int = 1  # 1-5 scale for budget hints
