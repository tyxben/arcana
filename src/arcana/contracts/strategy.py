"""Strategy-related contracts for adaptive policy decision making."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    """How the LLM wants to proceed."""

    DIRECT_ANSWER = "direct_answer"
    SINGLE_TOOL = "single_tool"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PLAN_AND_EXECUTE = "plan_and_execute"
    PIVOT = "pivot"


class StrategyDecision(BaseModel):
    """The LLM's strategy decision for the current step."""

    strategy: StrategyType
    reasoning: str
    action: str | None = None  # For DIRECT_ANSWER: the answer
    tool_name: str | None = None  # For SINGLE_TOOL
    tool_arguments: dict[str, Any] | None = None  # For SINGLE_TOOL
    parallel_actions: list[dict[str, Any]] | None = None  # For PARALLEL: list of tool calls
    plan: list[str] | None = None  # For PLAN_AND_EXECUTE: step descriptions
    pivot_reason: str | None = None  # For PIVOT: why changing direction
    pivot_new_approach: str | None = None  # For PIVOT: what to try instead


class AdaptiveState(BaseModel):
    """Adaptive policy's internal tracking state."""

    current_strategy: StrategyType = StrategyType.SEQUENTIAL
    plan_steps: list[str] = Field(default_factory=list)
    completed_plan_steps: list[str] = Field(default_factory=list)
    pivot_count: int = 0
    max_pivots: int = 3
