"""Multi-Agent — role-based collaboration protocol + name-addressed primitives."""

from arcana.contracts.multi_agent import (
    AgentMessage,
    ChannelMessage,
    CollaborationSession,
    HandoffResult,
    MessageType,
)
from arcana.multi_agent.agent_pool import AgentPool
from arcana.multi_agent.channel import Channel
from arcana.multi_agent.message_bus import MessageBus
from arcana.multi_agent.shared_context import SharedContext
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
    "ChannelMessage",
    "CollaborationSession",
    "HandoffResult",
    "MessageType",
    # Core (legacy)
    "MessageBus",
    "RoleConfig",
    "TeamOrchestrator",
    # Core (new primitives)
    "AgentPool",
    "Channel",
    "SharedContext",
]
