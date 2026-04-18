"""AgentPool -- manages named agents with shared communication primitives.

The pool provides infrastructure (Channel, SharedContext, Budget).
The user provides orchestration (who talks to whom, in what order).
The framework NEVER decides topology or turn order.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arcana.gateway.budget import BudgetTracker
    from arcana.multi_agent.channel import Channel
    from arcana.multi_agent.shared_context import SharedContext
    from arcana.runtime_core import ChatSession, Runtime

logger = logging.getLogger(__name__)


class AgentPool:
    """Manages named agents with shared bus, context, and budget.

    Usage::

        pool = AgentPool(runtime)
        planner = pool.add("planner", system="You plan tasks")
        executor = pool.add("executor", system="You execute plans", tools=[...])

        plan = await planner.send("Create a plan for: ...")
        result = await executor.send(f"Execute: {plan.content}")

    The pool provides:

    - Named ChatSessions (pool.add)
    - Shared budget (pool-level BudgetTracker)
    - Shared context (pool.shared) -- key-value store
    - Message channel (pool.channel) -- for complex communication patterns

    The pool does NOT provide:

    - Turn order (user code decides)
    - Topology (user code decides)
    - Stop conditions (user code decides)
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        budget_tracker: BudgetTracker | None = None,
    ) -> None:
        from arcana.multi_agent.channel import Channel
        from arcana.multi_agent.shared_context import SharedContext

        self._runtime = runtime
        self._budget_tracker = budget_tracker
        self._agents: dict[str, ChatSession] = {}
        self._channel = Channel()
        self._shared = SharedContext()

    def add(
        self,
        name: str,
        *,
        system: str = "",
        tools: list[Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_history: int | None = None,
    ) -> ChatSession:
        """Create a named agent (ChatSession) in the pool.

        The agent shares the pool's budget and has access to the
        channel and shared context through the pool.

        Raises ValueError if name already exists.
        """
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already exists in pool")

        # Register in channel so it can receive messages
        self._channel.register(name)

        # Create ChatSession via runtime helper
        session = self._runtime._create_pool_session(
            name=name,
            system=system or "You are a helpful assistant.",
            tools=tools,
            provider=provider,
            model=model,
            max_history=max_history,
            budget_tracker=self._budget_tracker,
        )
        self._agents[name] = session
        return session

    @property
    def channel(self) -> Channel:
        """The shared message channel."""
        return self._channel

    @property
    def shared(self) -> SharedContext:
        """The shared key-value context."""
        return self._shared

    @property
    def agents(self) -> dict[str, ChatSession]:
        """All named agents in the pool."""
        return dict(self._agents)

    async def close(self) -> None:
        """Clean up all agents."""
        self._agents.clear()
        self._channel.clear()
        self._shared.clear()

    async def __aenter__(self) -> AgentPool:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
