"""Multi-Agent — role-based collaboration protocol."""

from arcana.contracts.multi_agent import (
    AgentMessage,
    CollaborationSession,
    HandoffResult,
    MessageType,
)
from arcana.multi_agent.message_bus import MessageBus
from arcana.multi_agent.team import (
    APPROVED_VERDICTS,
    WM_KEY_FEEDBACK,
    WM_KEY_PLAN,
    WM_KEY_RESULT,
    WM_KEY_VERDICT,
    RoleConfig,
    TeamOrchestrator,
)

__all__ = [
    # Constants
    "APPROVED_VERDICTS",
    "WM_KEY_FEEDBACK",
    "WM_KEY_PLAN",
    "WM_KEY_RESULT",
    "WM_KEY_VERDICT",
    # Contracts
    "AgentMessage",
    "CollaborationSession",
    "HandoffResult",
    "MessageType",
    # Core
    "MessageBus",
    "RoleConfig",
    "TeamOrchestrator",
]
