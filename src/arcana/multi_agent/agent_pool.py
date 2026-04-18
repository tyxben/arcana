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

# Cognitive primitive names are reserved — a user-supplied tool with the
# same name would be shadowed by the runtime's interception. Raise instead
# of silently losing the user's tool (Principle 5: structured feedback).
_COGNITIVE_PRIMITIVE_NAMES = frozenset({"recall", "pin", "unpin"})


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

    Per-agent cognitive primitives (v0.8.0): each :meth:`add` call takes an
    optional ``cognitive_primitives`` list. If omitted, the agent inherits
    the pool-level default (set via ``runtime.collaborate(...)``). Each pool
    agent's cognitive state (pins, recall log) is fully private — never
    shared across the pool. See Principle 8 and Principle 9 in
    ``CONSTITUTION.md``.

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
        default_cognitive_primitives: list[str] | None = None,
    ) -> None:
        from arcana.multi_agent.channel import Channel
        from arcana.multi_agent.shared_context import SharedContext

        self._runtime = runtime
        self._budget_tracker = budget_tracker
        self._agents: dict[str, ChatSession] = {}
        self._channel = Channel()
        self._shared = SharedContext()
        # Pool-level default; inherited by pool.add(...) when it does not
        # specify its own cognitive_primitives. None means "fall back to
        # runtime config default" (the same behaviour as a bare
        # runtime.chat() session).
        self._default_cognitive_primitives = default_cognitive_primitives

    def add(
        self,
        name: str,
        *,
        system: str = "",
        tools: list[Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_history: int | None = None,
        cognitive_primitives: list[str] | None = None,
    ) -> ChatSession:
        """Create a named agent (ChatSession) in the pool.

        The agent shares the pool's budget and has access to the
        channel and shared context through the pool.

        Args:
            name: Unique name within the pool. Raises ``ValueError`` if
                a sibling already uses it.
            system: System prompt for this agent.
            tools: Per-agent tool list. A user tool whose name collides
                with an active cognitive primitive (``recall``, ``pin``,
                ``unpin``) raises ``ValueError`` — the runtime would
                otherwise silently intercept one at the expense of the
                other.
            provider / model: Per-agent provider/model override.
            max_history: Per-agent history retention cap.
            cognitive_primitives: Per-agent override of the cognitive
                primitives this agent sees as intercepted tools. ``None``
                (default) inherits the pool-level default; ``[]``
                explicitly opts out; ``["recall", "pin"]`` explicitly
                opts in. Each agent gets its own ``PinState`` /
                recall log — cognitive state is never shared across pool
                members.
        """
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already exists in pool")

        # Effective cognitive primitives for this agent: per-agent override
        # beats pool default; None at both levels means "inherit runtime
        # config" (handled downstream by ChatSession).
        effective_primitives: list[str] | None
        if cognitive_primitives is not None:
            effective_primitives = list(cognitive_primitives)
        elif self._default_cognitive_primitives is not None:
            effective_primitives = list(self._default_cognitive_primitives)
        else:
            effective_primitives = None

        # Tool-name / primitive-name collision is a configuration error.
        if effective_primitives and tools:
            active = _COGNITIVE_PRIMITIVE_NAMES & set(effective_primitives)
            # recall/pin implicitly activate unpin too — guard its name as well
            if "pin" in effective_primitives:
                active = active | {"unpin"}
            for tool in tools:
                tool_name = _extract_tool_name(tool)
                if tool_name and tool_name in active:
                    raise ValueError(
                        f"Agent '{name}': user-supplied tool '{tool_name}' "
                        f"collides with active cognitive primitive "
                        f"'{tool_name}'. Rename the tool or drop "
                        f"'{tool_name}' from cognitive_primitives."
                    )

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
            cognitive_primitives=effective_primitives,
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


def _extract_tool_name(tool: Any) -> str | None:
    """Best-effort extraction of a tool's user-visible name.

    Supports ``@arcana.tool`` decorated functions (carry ``_arcana_tool_spec``),
    raw ``ToolSpec`` instances, and anything exposing ``.name`` / ``__name__``.
    Returns ``None`` if no name can be determined — the collision check
    falls back to "don't know, don't block."
    """
    spec = getattr(tool, "_arcana_tool_spec", None)
    if spec is not None and hasattr(spec, "name"):
        return spec.name  # type: ignore[no-any-return]
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    fn_name = getattr(tool, "__name__", None)
    if isinstance(fn_name, str):
        return fn_name
    return None
