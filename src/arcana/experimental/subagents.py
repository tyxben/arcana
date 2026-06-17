"""Optional subtask isolation -- the experimental subagent service.

Phase 3 of ``specs/kimi-code-response-roadmap.md``. This is an **explicit,
user-directed** facade over the existing ``ChatSession`` / ``AgentPool``
machinery, deliberately incubated under :mod:`arcana.experimental` until its
semantics harden (see roadmap open question #3 and the stability note in the
package docstring).

Design law (mirrors the constitution and the roadmap non-goals):

- **No default topology.** There is no main agent, scheduler, or supervisor.
  The caller registers named subagents and decides who is asked, in what
  order, with what task. The framework never picks.
- **Isolation.** Every :meth:`SubagentService.ask` runs a *fresh* single-shot
  task: the subagent sees only its own system prompt, its granted tools, and
  the task packet (plus optional caller-supplied context). Repeated asks to
  the same name do **not** share conversation history.
- **No recursion (v1).** A subagent cannot spawn another subagent. Delegation
  tools produced by :meth:`SubagentService.as_tool` refuse with structured
  feedback when invoked from inside an active delegation.
- **Honest accounting.** Each ask returns a :class:`SubagentResult` carrying
  the run id, token/cost usage, and trace refs. Per-subtask budget caps are
  enforced; trace events are stamped with ``source_agent`` / ``bundle_id`` so
  a parent run can be correlated to its delegated children.

Authority is governed: a subagent's granted tools run under an optional
``PermissionPolicy`` (service-level default, overridable per subagent to
narrow), and ASK / write / confirmation-required tools gate behind a
service-level ``approval_handler`` (without one they surface
CONFIRMATION_REQUIRED rather than executing silently). A delegation tool can
be marked ``requires_approval`` so the *parent* gateway confirms before the
subagent runs.

Still deferred: auto-capture of ``delegated_by_run_id`` across arbitrary
parent agents (currently caller-threaded), and graduation of this surface
from ``arcana.experimental`` to the stable ``Runtime`` API.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from arcana.contracts.permission import PermissionPolicy
    from arcana.contracts.tool import ToolCall, ToolSpec
    from arcana.runtime_core import Budget, ChatSession, Runtime
    from arcana.sdk import Tool

    ApprovalHandler = Callable[[ToolCall, ToolSpec], Awaitable[bool]]

# True while a delegation (ask) is executing in the current async context.
# A delegation tool that fires while this is set is a recursive spawn attempt
# and is refused -- this enforces the "no recursion in v1" rule independently
# of run-id threading. Concurrent *independent* top-level asks each run in
# their own copied context (default False), so they are not blocked; only a
# nested ask-within-ask sees True.
_DELEGATION_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_arcana_subagent_delegation_active", default=False
)


class SubagentRecursionError(RuntimeError):
    """Raised when a subagent ask is attempted from inside an active delegation.

    v1 of the subagent service forbids recursive subtask spawning. The
    delegation tool returned by :meth:`SubagentService.as_tool` catches this
    and returns structured feedback to the calling LLM rather than crashing.
    """


class SubagentResult(BaseModel):
    """Outcome of a single :meth:`SubagentService.ask`.

    This is the only value that crosses the isolation boundary back to the
    caller -- the subagent's internal conversation does not leak out.
    """

    agent: str
    """Name of the subagent that produced this result."""

    content: str = ""
    """The subagent's final answer text."""

    run_id: str = ""
    """run_id of the isolated run (correlatable in the session bundle)."""

    tokens: int = 0
    """Tokens consumed by this ask."""

    cost: float = 0.0
    """USD cost of this ask."""

    trace_refs: list[str] = Field(default_factory=list)
    """Trace correlation handles for this ask (currently the run id)."""


class _SubagentConfig(BaseModel):
    """Internal registration record for a named subagent."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    system: str
    tools: list[Any] | None = None
    provider: str | None = None
    model: str | None = None
    budget: Any = None  # Budget | None -- per-subtask cap
    permission_policy: Any = None  # PermissionPolicy | None -- per-subagent override


class SubagentService:
    """User-directed registry of isolated subagents.

    Usage::

        async with subagents(runtime, budget=Budget(max_cost_usd=1.0)) as svc:
            svc.add("researcher", system="You research.", tools=[search],
                    budget=Budget(max_cost_usd=0.3))
            res = await svc.ask("researcher", "Find the latest on X")
            print(res.content, res.cost, res.run_id)

            # Expose delegation to *another* agent only when you choose to:
            delegate = svc.as_tool("researcher")
            answer = await runtime.run("Use research when needed",
                                       tools=[delegate])

    The service owns no orchestration policy. It provides named, isolated
    runs and (optionally) delegation tools; the caller is the orchestrator.
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        budget: Budget | None = None,
        bundle_id: str | None = None,
        permission_policy: PermissionPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self._runtime = runtime
        self._configs: dict[str, _SubagentConfig] = {}
        # Stable correlation id shared by every ask routed through this
        # service, stamped into trace metadata as ``bundle_id``.
        self._bundle_id = bundle_id or f"bundle-{uuid4().hex[:16]}"
        # Service-level default permission policy inherited by every subagent
        # that does not supply its own (override, not merge). Governs each
        # subagent's granted tools at its session gateway.
        self._permission_policy = permission_policy
        # Human-in-the-loop approval callback for ASK-action / write /
        # confirmation-required tools inside a subagent run. When None, such
        # tools surface CONFIRMATION_REQUIRED rather than silently executing.
        self._approval_handler = approval_handler
        # Optional service-level shared budget. Subagents without their own
        # ``budget`` accumulate against this tracker across asks; subagents
        # *with* a budget get a fresh per-ask cap instead.
        self._service_tracker: Any = None
        if budget is not None:
            from arcana.gateway.budget import BudgetTracker

            self._service_tracker = BudgetTracker(
                max_cost_usd=budget.max_cost_usd,
                max_tokens=budget.max_tokens,
            )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        *,
        system: str = "",
        tools: list[Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
        budget: Budget | None = None,
        permission_policy: PermissionPolicy | None = None,
    ) -> None:
        """Register a named subagent capability.

        Registration only records a profile (system prompt, granted tools,
        provider/model, optional per-subtask budget cap, optional per-subagent
        permission policy). No run happens until :meth:`ask` (or a delegation
        tool from :meth:`as_tool`) is called. Raises ``ValueError`` on a
        duplicate name.

        ``permission_policy`` narrows this subagent's authority over its
        granted tools. It **overrides** (does not merge with) the
        service-level policy -- supply a stricter policy to narrow, omit to
        inherit the service default.
        """
        if name in self._configs:
            raise ValueError(f"Subagent '{name}' already registered")
        self._configs[name] = _SubagentConfig(
            name=name,
            system=system or "You are a focused subtask agent.",
            tools=tools,
            provider=provider,
            model=model,
            budget=budget,
            permission_policy=permission_policy,
        )

    @property
    def bundle_id(self) -> str:
        """Correlation id shared by every ask routed through this service."""
        return self._bundle_id

    @property
    def names(self) -> list[str]:
        """Registered subagent names."""
        return list(self._configs)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def ask(
        self,
        name: str,
        task: str,
        *,
        context: dict[str, Any] | str | None = None,
        delegated_by_run_id: str | None = None,
    ) -> SubagentResult:
        """Run ``task`` on the named subagent as an isolated single-shot.

        The subagent sees only its own system prompt, its granted tools, and
        the task packet (plus ``context`` when supplied). A fresh session is
        built per call -- no conversation history carries across asks.

        Args:
            name: A name previously passed to :meth:`add`.
            task: The task packet for this run.
            context: Optional extra context injected as a ``<context>`` block,
                mirroring ``Runtime.run(context=...)``.
            delegated_by_run_id: Optional parent run id to record in trace
                metadata when this ask was triggered by another run. Auto
                capture across arbitrary parent agents is deferred; callers
                that want the link thread it explicitly.

        Raises:
            KeyError: If ``name`` was never registered.
            SubagentRecursionError: If called from inside an active delegation
                (v1 forbids recursive subtask spawning).
        """
        if name not in self._configs:
            raise KeyError(f"Unknown subagent '{name}'")
        if _DELEGATION_ACTIVE.get():
            raise SubagentRecursionError(
                f"Cannot ask subagent '{name}': recursive subtask spawning is "
                f"not allowed in v1."
            )

        cfg = self._configs[name]
        session = self._build_session(cfg, delegated_by_run_id)

        message = self._compose_message(task, context)

        token = _DELEGATION_ACTIVE.set(True)
        try:
            response = await session.send(message)
        finally:
            _DELEGATION_ACTIVE.reset(token)

        # Internal: ChatSession records the run_id of its last turn. The
        # service is already coupled to runtime internals (it builds sessions
        # via _create_pool_session), so reading this private attr is in scope.
        run_id = session._last_run_id
        return SubagentResult(
            agent=name,
            content=response.content,
            run_id=run_id,
            tokens=response.tokens_used,
            cost=response.cost_usd,
            trace_refs=[run_id] if run_id else [],
        )

    def as_tool(
        self,
        name: str,
        *,
        tool_name: str | None = None,
        description: str | None = None,
        side_effect: str = "write",
        requires_approval: bool = False,
    ) -> Tool:
        """Expose delegation to the named subagent as a callable tool.

        Delegation is *opt-in*: the caller chooses to hand this tool to an
        LLM. The returned tool runs :meth:`ask` with the model-supplied task
        string. Invoking it from inside an active delegation returns a refusal
        message (structured feedback) rather than spawning recursively.

        The tool defaults to a ``write`` side-effect because a delegated run
        consumes budget and may itself invoke write tools -- callers should
        not mistake it for a pure read.

        ``requires_approval=True`` marks the delegation as high-authority:
        the tool spec carries ``requires_confirmation`` so the *parent*
        gateway gates the delegation behind its confirmation handler before
        the subagent ever runs. (The service cannot force the parent gateway
        to have a handler; without one, the parent surfaces
        CONFIRMATION_REQUIRED rather than delegating silently.)
        """
        if name not in self._configs:
            raise KeyError(f"Unknown subagent '{name}'")
        from arcana.sdk import Tool

        async def _delegate(task: str) -> str:
            if _DELEGATION_ACTIVE.get():
                return (
                    f"Delegation to '{name}' refused: recursive subtask "
                    f"spawning is not allowed. Complete this task directly."
                )
            result = await self.ask(name, task)
            return result.content

        _delegate.__name__ = tool_name or f"ask_{name}"
        return Tool(
            _delegate,
            name=tool_name or f"ask_{name}",
            description=(
                description
                or f"Delegate a self-contained subtask to the '{name}' subagent "
                f"and return its answer. Pass the full task as 'task'."
            ),
            side_effect=side_effect,
            requires_confirmation=requires_approval,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_session(
        self, cfg: _SubagentConfig, delegated_by_run_id: str | None
    ) -> ChatSession:
        """Construct a fresh, isolated ChatSession for one ask."""
        tracker = self._tracker_for(cfg)
        extra_metadata: dict[str, Any] = {"bundle_id": self._bundle_id}
        if delegated_by_run_id:
            extra_metadata["delegated_by_run_id"] = delegated_by_run_id
        # Per-subagent policy overrides the service-level default (narrowing);
        # absent both, the subagent's granted tools are ungoverned.
        policy = (
            cfg.permission_policy
            if cfg.permission_policy is not None
            else self._permission_policy
        )
        return self._runtime._create_pool_session(
            name=cfg.name,
            system=cfg.system,
            tools=cfg.tools,
            provider=cfg.provider,
            model=cfg.model,
            budget_tracker=tracker,
            extra_trace_metadata=extra_metadata,
            permission_policy=policy,
            confirmation_callback=self._approval_handler,
        )

    def _tracker_for(self, cfg: _SubagentConfig) -> Any:
        """Resolve the budget tracker for an ask.

        A subagent with its own ``budget`` gets a *fresh* per-ask cap. Without
        one, it accumulates against the service-level shared tracker (or, when
        the service has no budget, ``None`` -- the session falls back to the
        runtime's default budget policy).
        """
        if cfg.budget is not None:
            from arcana.gateway.budget import BudgetTracker

            return BudgetTracker(
                max_cost_usd=cfg.budget.max_cost_usd,
                max_tokens=cfg.budget.max_tokens,
            )
        return self._service_tracker

    @staticmethod
    def _compose_message(
        task: str, context: dict[str, Any] | str | None
    ) -> str:
        if context is None:
            return task
        import json as _json

        if isinstance(context, dict):
            ctx = _json.dumps(context, ensure_ascii=False, indent=2)
        else:
            ctx = str(context)
        return f"{task}\n\n<context>\n{ctx}\n</context>"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Drop all registrations. Idempotent."""
        self._configs.clear()

    async def __aenter__(self) -> SubagentService:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


def subagents(
    runtime: Runtime,
    *,
    budget: Budget | None = None,
    bundle_id: str | None = None,
    permission_policy: PermissionPolicy | None = None,
    approval_handler: ApprovalHandler | None = None,
) -> SubagentService:
    """Create an experimental :class:`SubagentService` over ``runtime``.

    This is a synchronous factory mirroring ``runtime.collaborate()``
    ergonomics, but it lives under :mod:`arcana.experimental` rather than on
    the stable ``Runtime`` surface while its semantics incubate. Enter the
    returned service with ``async with`` for automatic cleanup.

    ``permission_policy`` is the service-level default governing every
    subagent's granted tools (override per subagent via :meth:`add`).
    ``approval_handler`` is the human-in-the-loop callback for ASK / write /
    confirmation-required tools inside subagent runs.
    """
    return SubagentService(
        runtime,
        budget=budget,
        bundle_id=bundle_id,
        permission_policy=permission_policy,
        approval_handler=approval_handler,
    )
