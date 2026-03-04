"""Multi-Agent — role-based collaboration protocol."""

from arcana.contracts.multi_agent import (
    AgentMessage,
    CollaborationSession,
    HandoffResult,
    MessageType,
)
from arcana.multi_agent.message_bus import MessageBus
from arcana.multi_agent.team import RoleConfig, TeamOrchestrator

__all__ = [
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
