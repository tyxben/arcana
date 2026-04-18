"""Channel -- name-addressed message passing between agents."""

from __future__ import annotations

import asyncio

from arcana.contracts.multi_agent import ChannelMessage


class Channel:
    """Name-addressed async message channel.

    Unlike :class:`MessageBus` (role-addressed), ``Channel`` uses agent names
    as addresses.  Supports point-to-point and broadcast messaging.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[ChannelMessage]] = {}
        self._history: list[ChannelMessage] = []

    # -- sending / receiving ---------------------------------------------------

    async def send(self, message: ChannelMessage) -> None:
        """Send a message to a specific agent or broadcast."""
        self._history.append(message)
        if message.recipient is not None:
            # Point-to-point
            self._ensure_queue(message.recipient)
            await self._queues[message.recipient].put(message)
        else:
            # Broadcast to all known agents except sender
            for name, queue in self._queues.items():
                if name != message.sender:
                    await queue.put(message)

    async def receive(self, agent_name: str) -> list[ChannelMessage]:
        """Drain all pending messages for an agent (non-blocking)."""
        self._ensure_queue(agent_name)
        messages: list[ChannelMessage] = []
        queue = self._queues[agent_name]
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    # -- registration / introspection ------------------------------------------

    def register(self, agent_name: str) -> None:
        """Register an agent name (creates its queue)."""
        self._ensure_queue(agent_name)

    @property
    def agents(self) -> list[str]:
        """List registered agent names."""
        return list(self._queues.keys())

    @property
    def history(self) -> list[ChannelMessage]:
        """Full message history (read-only copy)."""
        return list(self._history)

    def clear(self) -> None:
        """Clear all queues and history."""
        self._queues.clear()
        self._history.clear()

    # -- internal --------------------------------------------------------------

    def _ensure_queue(self, name: str) -> None:
        if name not in self._queues:
            self._queues[name] = asyncio.Queue()
