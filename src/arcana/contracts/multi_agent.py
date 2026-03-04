"""Multi-agent collaboration contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from arcana.contracts.trace import AgentRole


class MessageType(str, Enum):
    """Type of inter-agent message."""

    PLAN = "plan"
    RESULT = "result"
    FEEDBACK = "feedback"
    HANDOFF = "handoff"
    ESCALATE = "escalate"


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
