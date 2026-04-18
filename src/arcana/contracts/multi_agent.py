"""Multi-agent collaboration contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from arcana.contracts.trace import AgentRole


class MessageType(str, Enum):
    """Type of inter-agent message."""

    PLAN = "plan"
    RESULT = "result"
    FEEDBACK = "feedback"
    HANDOFF = "handoff"
    ESCALATE = "escalate"
    CHAT = "chat"


class AgentMessage(BaseModel):
    """Message passed between agents in a collaboration session."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    sender_role: AgentRole
    recipient_role: AgentRole
    message_type: MessageType
    content: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChannelMessage(BaseModel):
    """Message in a name-addressed communication channel.

    Immutable: ``Channel.send`` broadcasts a single instance to every
    recipient (and to ``history``), so shared-state mutations would bleed
    across receivers. Use :meth:`model_copy(update=...)` to derive a
    modified message rather than mutating in place.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    sender: str  # agent name
    recipient: str | None = None  # agent name, None = broadcast
    content: str  # message body (simple string, not dict)
    message_type: MessageType = MessageType.CHAT
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str = ""


class CollaborationSession(BaseModel):
    """Configuration for a multi-agent collaboration session."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    roles: list[AgentRole] = Field(
        default_factory=lambda: [
            AgentRole.PLANNER,
            AgentRole.EXECUTOR,
            AgentRole.CRITIC,
        ]
    )
    max_rounds: int = 5
    shared_memory_ns: str = ""
    status: str = "active"


class HandoffResult(BaseModel):
    """Result of a multi-agent collaboration session."""

    session_id: str
    final_status: str  # "completed" / "failed" / "escalated"
    rounds: int
    messages: list[AgentMessage] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
