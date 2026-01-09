"""Core contracts and data models for Arcana."""

from arcana.contracts.llm import (
    Budget,
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    TokenUsage,
)
from arcana.contracts.state import AgentState, StateSnapshot
from arcana.contracts.tool import ToolCall, ToolResult, ToolSpec
from arcana.contracts.trace import (
    BudgetSnapshot,
    StopReason,
    ToolCallRecord,
    TraceContext,
    TraceEvent,
)

__all__ = [
    # LLM
    "ModelConfig",
    "Message",
    "LLMRequest",
    "LLMResponse",
    "TokenUsage",
    "Budget",
    # Tool
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    # State
    "AgentState",
    "StateSnapshot",
    # Trace
    "TraceEvent",
    "TraceContext",
    "ToolCallRecord",
    "BudgetSnapshot",
    "StopReason",
]
