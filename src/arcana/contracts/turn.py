"""Turn-level contracts for ConversationAgent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from arcana.contracts.llm import TokenUsage, ToolCallRequest


class TurnFacts(BaseModel):
    """Raw facts from LLM response. NO runtime interpretation.

    _parse_turn() ONLY populates this -- no inference, no heuristics.
    This is a 1:1 mapping of what the provider API returned.
    """

    assistant_text: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    thinking: str | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class TurnAssessment(BaseModel):
    """Runtime's interpretation of the turn. Separate from facts.

    _assess_turn() ONLY populates this. This is where completion detection,
    failure detection, and other runtime judgments live. Never mixed into
    TurnFacts.
    """

    completed: bool = False
    failed: bool = False
    needs_clarification: bool = False
    answer: str | None = None
    completion_reason: str | None = None
    confidence: float = 0.0


class TurnOutcome(BaseModel):
    """Facts + Assessment, kept visibly separate.

    Code that reads TurnOutcome always knows whether it's looking at
    "what the LLM said" (facts) or "what runtime thinks" (assessment).
    """

    facts: TurnFacts
    assessment: TurnAssessment = Field(default_factory=TurnAssessment)
