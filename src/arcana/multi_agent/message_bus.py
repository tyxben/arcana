"""MessageBus — in-process async message passing between agents."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcana.contracts.multi_agent import AgentMessage
    from arcana.contracts.trace import AgentRole


class MessageBus:
    """
    In-process async message bus for agent-to-agent communication.

    Messages are routed by recipient role. Each role has an async queue.
    All messages are also stored in a session-keyed history for auditing.
    """

    def __init__(self) -> None:
        self._queues: dict[AgentRole, asyncio.Queue[AgentMessage]] = {}
        self._history: dict[str, list[AgentMessage]] = defaultdict(list)

    async def publish(self, message: AgentMessage) -> None:
        """
        Publish a message to the recipient's queue.

        Args:
            message: The message to publish.
        """
        role = message.recipient_role
        if role not in self._queues:
            self._queues[role] = asyncio.Queue()
        await self._queues[role].put(message)
        self._history[message.session_id].append(message)

    async def subscribe(self, role: AgentRole) -> list[AgentMessage]:
        """
        Retrieve all pending messages for a role (non-blocking drain).

        Args:
            role: The agent role to get messages for.

        Returns:
            List of pending messages, empty if none.
        """
        if role not in self._queues:
            return []

        messages: list[AgentMessage] = []
        queue = self._queues[role]
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def history(self, session_id: str) -> list[AgentMessage]:
        """
        Get the full message history for a session.

        Args:
            session_id: The collaboration session ID.

        Returns:
            List of all messages in chronological order.
        """
        return list(self._history.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """
        Clear message history for a session.

        Args:
            session_id: The collaboration session ID to clear.
        """
        self._history.pop(session_id, None)
