"""Core contracts and data models for Arcana."""

from arcana.contracts.channel import ExecutionChannel
from arcana.contracts.evolution import (
    AcceptanceDecision,
    ApproverKind,
    AuthorityClass,
    EvidenceBundle,
    EvolutionProposal,
    EvolutionTarget,
    EvolutionTargetKind,
    MonitoringAnchor,
    PatchRef,
    PromotionRecord,
    ProposalStatus,
    RollbackPointer,
    SandboxVerificationRef,
    classify_authority,
)
from arcana.contracts.guardrail import (
    GuardrailAction,
    GuardrailDecision,
    ToolGuardrailRequest,
)
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
from arcana.contracts.skill import (
    SkillLifecycleState,
    SkillRegistry,
    SkillSelectionRecord,
    SkillSpec,
)
from arcana.contracts.state import AgentState, ExecutionStatus, StateSnapshot
from arcana.contracts.strategy import AdaptiveState, StrategyDecision, StrategyType
from arcana.contracts.tool import ToolCall, ToolProvenance, ToolResult, ToolSpec
from arcana.contracts.trace import (
    BudgetSnapshot,
    GuardrailDecisionRecord,
    ProtocolDiscoveryRecord,
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
    # Guardrails
    "GuardrailAction",
    "GuardrailDecision",
    "ToolGuardrailRequest",
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
    # Skill
    "SkillSpec",
    "SkillSelectionRecord",
    "SkillRegistry",
    "SkillLifecycleState",
    # Self-evolution (Amendment 6 contracts — no running loop)
    "EvolutionTargetKind",
    "AuthorityClass",
    "ProposalStatus",
    "AcceptanceDecision",
    "ApproverKind",
    "EvidenceBundle",
    "EvolutionProposal",
    "EvolutionTarget",
    "PatchRef",
    "PromotionRecord",
    "SandboxVerificationRef",
    "RollbackPointer",
    "MonitoringAnchor",
    "classify_authority",
    # Trace
    "TraceEvent",
    "TraceContext",
    "ToolCallRecord",
    "ProtocolDiscoveryRecord",
    "GuardrailDecisionRecord",
    "BudgetSnapshot",
    "StopReason",
]
