"""Multi-agent collaboration contracts.

Trimmed in the post-Amendment-3 cleanup (commit landing 2026-05-03):
the original ``AgentMessage`` / ``CollaborationSession`` / ``HandoffResult``
were tightly coupled to the role-addressed ``MessageBus`` + Planner/
Executor/Critic ``TeamOrchestrator`` shape that Amendment 3 (v3.4)
rejected. Those types and their consumers have been removed.

What remains is the name-addressed surface: ``ChannelMessage`` (used by
``arcana.multi_agent.channel.Channel``). ``MessageType`` is retained
because ``ChannelMessage.message_type`` defaults to ``MessageType.CHAT``;
the legacy enum members (``PLAN``/``RESULT``/``FEEDBACK``/``HANDOFF``/
``ESCALATE``) are kept for downstream code that may still parse historic
trace messages and will be re-evaluated as part of a future enum cleanup.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class MessageType(str, Enum):
    """Type of inter-agent message."""

    PLAN = "plan"
    RESULT = "result"
    FEEDBACK = "feedback"
    HANDOFF = "handoff"
    ESCALATE = "escalate"
    CHAT = "chat"


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
