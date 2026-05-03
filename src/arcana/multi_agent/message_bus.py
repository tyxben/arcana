"""MessageBus — in-process role-addressed async message passing.

.. deprecated::
    Slated for removal in a v1.x minor following Constitution Amendment 3
    (v3.4, 2026-05-03). ``MessageBus`` is role-addressed via the
    ``AgentRole`` enum, which is the same prescribed-topology shape that
    Amendment 3 rejects. Use the name-addressed ``Channel`` (in
    ``arcana.multi_agent.channel``) instead — it provides the same
    publish/subscribe primitive but addressing is by free-form agent
    name, not by a framework-fixed role enum.

    ``arcana.multi_agent.*`` is internal-not-stable per
    ``specs/v1.0.0-stability.md`` §2; one minor with ``DeprecationWarning``
    is sufficient courtesy.
"""

from __future__ import annotations

import asyncio
import warnings
from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcana.contracts.multi_agent import AgentMessage
    from arcana.contracts.trace import AgentRole


_DEPRECATION_MSG = (
    "MessageBus is deprecated and slated for removal in a v1.x minor. It is "
    "role-addressed via the AgentRole enum, which encodes a "
    "framework-prescribed topology rejected by Constitution Amendment 3. "
    "Use arcana.multi_agent.channel.Channel — same primitive, name-addressed "
    "instead of role-addressed."
)


class MessageBus:
    """
    In-process async message bus for agent-to-agent communication.

    Messages are routed by recipient role. Each role has an async queue.
    All messages are also stored in a session-keyed history for auditing.

    .. deprecated:: see module docstring.

    Args:
        history_limit: Maximum number of past messages to retain per session
            in :meth:`history`. ``None`` (default) keeps unbounded history --
            matches pre-v0.8.2 behaviour. Set a positive ``int`` for
            long-running owners (e.g. :class:`TeamOrchestrator` reusing a
            single bus across runs) to bound memory. ``0`` disables history
            retention entirely (queues still deliver live messages).
            Negative values raise ``ValueError``. Only per-session history
            is bounded; per-role delivery queues are driven by consumer
            ``subscribe()`` calls and are not affected.
    """

    def __init__(self, *, history_limit: int | None = None) -> None:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        if history_limit is not None and history_limit < 0:
            raise ValueError(
                f"history_limit must be None or >= 0, got {history_limit}"
            )
        self._queues: dict[AgentRole, asyncio.Queue[AgentMessage]] = {}
        self._history_limit = history_limit
        self._history: dict[str, deque[AgentMessage]] = defaultdict(
            lambda: deque(maxlen=history_limit)
        )

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

        Note: This method is provided for custom orchestrator implementations.
        The default ``TeamOrchestrator`` does not use ``subscribe()``; it
        communicates between roles via direct state passing and only uses
        ``publish()`` / ``history()`` for audit logging.

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

    def reset(self) -> None:
        """
        Clear all history AND drain every per-role queue.

        Use this when an owner reuses a single ``MessageBus`` across
        independent runs (e.g. :class:`TeamOrchestrator`): without it, the
        per-role ``asyncio.Queue`` s accumulate every published message
        forever because the orchestrator never calls ``subscribe()``. Call
        this at the end of each run to release both history and queue
        state.
        """
        self._history.clear()
        for queue in self._queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
