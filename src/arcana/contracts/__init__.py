"""Core contracts and data models for Arcana."""

from arcana.contracts.channel import ExecutionChannel
from arcana.contracts.llm import (
    Budget,
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    TokenUsage,
)
from arcana.contracts.permission import (
    PermissionAction,
    PermissionDecision,
    PermissionMatch,
    PermissionPolicy,
    PermissionRequest,
    PermissionRule,
    PermissionScope,
)
from arcana.contracts.runtime import (
    PolicyDecision,
    RuntimeConfig,
    StepResult,
    StepType,
)
from arcana.contracts.state import AgentState, ExecutionStatus, StateSnapshot
from arcana.contracts.strategy import AdaptiveState, StrategyDecision, StrategyType
from arcana.contracts.tool import ToolCall, ToolProvenance, ToolResult, ToolSpec
from arcana.contracts.trace import (
    BudgetSnapshot,
    StopReason,
    ToolCallRecord,
    TraceContext,
    TraceEvent,
)

__all__ = [
    # Channel
    "ExecutionChannel",
    # LLM
    "ModelConfig",
    "Message",
    "LLMRequest",
    "LLMResponse",
    "TokenUsage",
    "Budget",
    # Permission
    "PermissionAction",
    "PermissionDecision",
    "PermissionMatch",
    "PermissionPolicy",
    "PermissionRequest",
    "PermissionRule",
    "PermissionScope",
    # Tool
    "ToolSpec",
    "ToolProvenance",
    "ToolCall",
    "ToolResult",
    # State
    "AgentState",
    "StateSnapshot",
    "ExecutionStatus",
    # Strategy
    "StrategyType",
    "StrategyDecision",
    "AdaptiveState",
    # Runtime
    "RuntimeConfig",
    "PolicyDecision",
    "StepResult",
    "StepType",
    # Trace
    "TraceEvent",
    "TraceContext",
    "ToolCallRecord",
    "BudgetSnapshot",
    "StopReason",
]
