"""
Arcana Runtime -- the core product.

Create once, use across your application. Provides budget, tools, trace,
and LLM access as managed services.

Usage:
    # Create runtime (once, at app startup)
    runtime = arcana.Runtime(
        providers={"deepseek": "sk-xxx", "openai": "sk-proj-xxx"},
        tools=[my_search, my_calculator],
        budget=arcana.Budget(max_cost_usd=10.0),
        trace=True,
    )

    # Simple: run a task
    result = await runtime.run("Analyze this data")

    # Multi-turn chat
    async with runtime.chat() as c:
        r = await c.send("Hello")
        r = await c.send("Tell me more about X")
        print(c.total_cost_usd)

    # Advanced: manual session control
    async with runtime.session() as s:
        response = await s.llm("What should I search for?")
        data = await s.tool("web_search", query="quantum computing")
        print(s.budget.remaining_usd)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from arcana.graph.nodes.llm_node import LLMNode
    from arcana.graph.nodes.tool_node import ToolNode
    from arcana.graph.state_graph import StateGraph

from pydantic import BaseModel, Field

from arcana.contracts.llm import ModelConfig
from arcana.contracts.mcp import MCPServerConfig
from arcana.contracts.state import AgentState
from arcana.contracts.streaming import StreamEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re as _re  # noqa: E402

_CODE_FENCE_RE = _re.compile(
    r"^\s*```(?:json|JSON)?\s*\n(.*?)\n\s*```\s*$",
    _re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences wrapping JSON output.

    Some providers (GLM, MiniMax) return JSON wrapped like::

        ```json
        {"key": "value"}
        ```

    This helper removes the fences so ``json.loads()`` succeeds.
    If the text does not match the fence pattern it is returned unchanged.
    """
    m = _CODE_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Event hook system
# ---------------------------------------------------------------------------

EventCallback = Callable[..., Any]


class _EventBus:
    """Lightweight pub/sub for runtime events."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[EventCallback]] = defaultdict(list)

    def on(self, event: str, callback: EventCallback) -> None:
        self._listeners[event].append(callback)

    def off(self, event: str, callback: EventCallback) -> None:
        listeners = self._listeners.get(event, [])
        if callback in listeners:
            listeners.remove(callback)

    async def emit(self, event: str, **kwargs: Any) -> None:
        for cb in self._listeners.get(event, []):
            result = cb(**kwargs)
            if asyncio.iscoroutine(result):
                await result


class Budget(BaseModel):
    """Budget configuration for a Runtime."""

    max_cost_usd: float = 10.0
    max_tokens: int = 500_000


class ChainStep(BaseModel):
    """A step in a sequential chain/pipeline."""

    name: str
    goal: str  # Prompt for this step
    system: str | None = None  # System prompt for this step
    response_format: Any = None  # type[BaseModel] | None
    tools: list[Any] | None = None  # list[Callable] | None
    provider: str | None = None  # Override provider for this step
    model: str | None = None  # Override model for this step
    on_parse_error: Any = None  # Callable[[str, Exception], BaseModel | None] | None
    budget: Budget | None = None  # Per-step budget cap (soft, within chain budget)

    model_config = {"arbitrary_types_allowed": True}


class ChainResult(BaseModel):
    """Result of a chain/pipeline execution."""

    output: Any = None  # Final step's output
    success: bool = False
    steps: dict[str, Any] = Field(default_factory=dict)  # name -> output
    total_tokens: int = 0
    total_cost_usd: float = 0.0


class BatchResult(BaseModel):
    """Result of a batch execution (multiple independent runs)."""

    results: list[Any] = Field(default_factory=list)  # list[RunResult]
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    succeeded: int = 0
    failed: int = 0


class BudgetScope:
    """Scoped budget context — runs inside deduct from both scope and runtime.

    Usage::

        async with runtime.budget_scope(max_cost_usd=0.10) as scoped:
            result = await scoped.run("Filter items", ...)
            print(scoped.budget_used_usd)
    """

    def __init__(
        self,
        runtime: Runtime,
        max_cost_usd: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._runtime = runtime
        self._max_cost_usd = max_cost_usd
        self._max_tokens = max_tokens
        self._cost_used: float = 0.0
        self._tokens_used: int = 0

    @property
    def budget_used_usd(self) -> float:
        return self._cost_used

    @property
    def budget_remaining_usd(self) -> float | None:
        if self._max_cost_usd is None:
            return None
        return max(0.0, self._max_cost_usd - self._cost_used)

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def tokens_remaining(self) -> int | None:
        if self._max_tokens is None:
            return None
        return max(0, self._max_tokens - self._tokens_used)

    async def run(self, goal: str, **kwargs: Any) -> RunResult:
        """Run with scoped budget enforcement."""
        from arcana.gateway.base import BudgetExceededError

        if self._max_cost_usd is not None and self._cost_used > self._max_cost_usd:
            raise BudgetExceededError("Scoped budget exhausted (cost)", budget_type="cost")
        if self._max_tokens is not None and self._tokens_used > self._max_tokens:
            raise BudgetExceededError("Scoped token budget exhausted", budget_type="tokens")

        scope_budget: Budget | None = None
        if self._max_cost_usd is not None or self._max_tokens is not None:
            remaining_cost = (self._max_cost_usd - self._cost_used if self._max_cost_usd is not None else 10.0)
            remaining_tokens = (self._max_tokens - self._tokens_used if self._max_tokens is not None else 500_000)
            scope_budget = Budget(max_cost_usd=remaining_cost, max_tokens=remaining_tokens)

        user_budget: Budget | None = kwargs.pop("budget", None)
        if user_budget and scope_budget:
            scope_budget = Budget(
                max_cost_usd=min(scope_budget.max_cost_usd, user_budget.max_cost_usd),
                max_tokens=min(scope_budget.max_tokens, user_budget.max_tokens),
            )
        elif user_budget:
            scope_budget = user_budget

        result = await self._runtime.run(goal, budget=scope_budget, **kwargs)
        self._cost_used += result.cost_usd
        self._tokens_used += result.tokens_used
        return result


class AgentConfig(BaseModel):
    """Configuration for a single agent in a team."""

    name: str
    prompt: str  # System prompt defining this agent's role/personality
    model: str | None = None  # Override model for this agent
    provider: str | None = None  # Override provider for this agent


class TeamResult(BaseModel):
    """Result of a multi-agent run."""

    output: Any = None
    success: bool = False
    rounds: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    agent_outputs: dict[str, str] = Field(default_factory=dict)  # name -> last output
    conversation_log: list[dict[str, Any]] = Field(default_factory=list)  # full history


class RuntimeConfig(BaseModel):
    """Configuration for an Arcana Runtime."""

    default_provider: str = "deepseek"
    default_model: str | None = None
    max_turns: int = 20
    trace_dir: str = "./traces"
    system_prompt: str | None = None

    # When True, trace emits PROMPT_SNAPSHOT events containing the full
    # messages/tools/model for each LLM call. Off by default: prompts can
    # carry PII / secrets and bloat trace files. Opt in for deep replay.
    trace_include_prompt_snapshots: bool = False


class Runtime:
    """
    Arcana Agent Runtime -- create once, use many times.

    Holds long-lived resources:
    - Provider connections (gateway registry, with automatic fallback chain)
    - Tool registry + gateway
    - Trace backend
    - Default budget policy
    - Default engine config

    Provider fallback behavior:
        When multiple providers are registered (e.g.
        ``providers={"deepseek": "sk-xxx", "openai": "sk-yyy"}``),
        the runtime automatically builds a fallback chain based on
        registration order (dict key order). The first provider is
        primary; subsequent providers serve as fallbacks. If the
        primary provider fails with a retryable error after exhausting
        retries, the request is automatically forwarded to the next
        provider in the chain.

        Use ``runtime.fallback_order`` to inspect the resolved order.

    Args:
        providers: Mapping of provider name to API key. The first key
            becomes the default provider; remaining keys form the
            automatic fallback chain in insertion order.
        namespace: Optional namespace for tenant isolation. When set,
            memory and trace are partitioned so that multiple Runtimes
            sharing the same backing stores don't see each other's data.
            When ``None`` (the default), behavior is unchanged.
    """

    def __init__(
        self,
        *,
        providers: dict[str, str | dict[str, str]] | None = None,  # {"deepseek": "sk-xxx"} or {"deepseek": {"api_key": "sk-xxx", "model": "deepseek-reasoner"}}
        tools: list[Callable] | None = None,  # type: ignore[type-arg]
        mcp_servers: list[MCPServerConfig] | None = None,
        budget: Budget | None = None,
        trace: bool = False,
        memory: bool = False,
        memory_budget_tokens: int = 800,
        config: RuntimeConfig | None = None,
        namespace: str | None = None,
        context_strategy: Any = None,  # ContextStrategy | str | None
    ) -> None:
        self._config = config or RuntimeConfig()
        self._budget_policy = budget or Budget()
        self._namespace = namespace

        # Parse context strategy
        from arcana.contracts.context import ContextStrategy

        if isinstance(context_strategy, str):
            self._context_strategy = ContextStrategy(mode=context_strategy)
        elif isinstance(context_strategy, ContextStrategy):
            self._context_strategy = context_strategy
        else:
            self._context_strategy = ContextStrategy()  # default adaptive

        # Setup providers (long-lived)
        self._gateway = self._setup_providers(providers or {})

        # Setup tools (long-lived)
        self._tool_registry = None
        self._tool_gateway = None
        if tools:
            self._tool_registry, self._tool_gateway = self._setup_tools(tools)

        # Store MCP configs for lazy connection
        self._mcp_configs = mcp_servers or []
        self._mcp_client: Any = None  # MCPClient, set after connect_mcp()

        # Cumulative budget tracking across runs (lock protects concurrent run() calls)
        self._totals_lock = asyncio.Lock()
        self._total_tokens_used: int = 0
        self._total_cost_usd: float = 0.0

        # Setup trace (long-lived)
        self._trace_writer = None
        if trace:
            from arcana.trace.writer import TraceWriter

            self._trace_writer = TraceWriter(
                trace_dir=self._config.trace_dir,
                namespace=self._namespace,
            )

        # Memory (cross-run context)
        self._memory_store: Any = None
        if memory:
            from arcana.memory.run_memory import RunMemoryStore

            self._memory_store = RunMemoryStore(
                default_budget_tokens=memory_budget_tokens,
                namespace=self._namespace,
            )

        # Event hooks
        self._events = _EventBus()

    # ------------------------------------------------------------------
    # Context manager (ensures cleanup of HTTP connections etc.)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Runtime:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def on(self, event: str, callback: EventCallback) -> Runtime:
        """Subscribe to runtime events. Returns self for chaining.

        Events:
            "run_start": (run_id: str, goal: str)
            "run_end": (run_id: str, result: RunResult)
            "error": (run_id: str, error: Exception)
        """
        self._events.on(event, callback)
        return self

    def off(self, event: str, callback: EventCallback) -> Runtime:
        """Unsubscribe from runtime events."""
        self._events.off(event, callback)
        return self

    async def run(
        self,
        goal: str,
        *,
        engine: str = "conversation",
        max_turns: int | None = None,
        budget: Budget | None = None,
        tools: list[Callable] | None = None,  # type: ignore[type-arg]
        response_format: type[BaseModel] | None = None,
        images: list[str] | None = None,
        input_handler: Callable | None = None,  # type: ignore[type-arg]
        system: str | None = None,
        context: dict[str, Any] | str | None = None,
        provider: str | None = None,
        model: str | None = None,
        on_parse_error: Callable | None = None,  # type: ignore[type-arg]
    ) -> RunResult:
        """
        Run a task to completion.

        Args:
            goal: What to accomplish
            engine: "conversation" (V2, default) or "adaptive" (V1)
            max_turns: Override default max turns
            budget: Override default budget for this run
            tools: Additional tools for this run only
            response_format: Pydantic model class for structured output.
                When provided, the LLM response is parsed and validated
                against this model. ``result.output`` will be an instance
                of the model rather than a plain string. Tools remain
                available — the agent uses tools during reasoning and
                returns structured output on the final turn.
            images: Optional list of image inputs (URLs, file paths, or
                data URIs) to include in the initial user message.
            input_handler: Optional callback for the ask_user built-in tool.
                Can be sync or async. When None, the LLM receives a fallback
                message and proceeds with best judgment.
            system: System prompt for this run. Overrides
                ``RuntimeConfig.system_prompt``. When None, falls back to
                the config value or the engine's default.
            context: Additional context for the agent. A dict is serialized
                as JSON; a string is used as-is. Injected into the goal as
                a ``<context>`` block so the agent can reference prior step
                outputs or external data.
            provider: Override the default provider for this run only.
            model: Override the default model for this run only.
            on_parse_error: Optional callback invoked when the LLM returns
                text that cannot be parsed into the ``response_format``
                model.  Receives ``(raw_string, error)`` where *error* is
                a ``json.JSONDecodeError`` or ``pydantic.ValidationError``.
                Return a fixed ``BaseModel`` instance to recover, or
                ``None`` to preserve the failure.  Supports async.

                Does NOT fire for provider-level rejections (e.g. the
                provider does not support ``json_schema`` mode) -- those
                surface as ``ProviderError`` and are handled by provider
                capability detection / auto-downgrade.
        """
        import json as _json

        # Auto-connect MCP on first run
        if self._mcp_configs and not self._mcp_client:
            await self.connect_mcp()

        # Inject context into goal
        if context is not None:
            if isinstance(context, dict):
                context_str = _json.dumps(context, ensure_ascii=False, indent=2)
            else:
                context_str = str(context)
            goal = f"{goal}\n\n<context>\n{context_str}\n</context>"

        # Hint when many tools are registered
        if self._tool_registry and len(self._tool_registry.list_tools()) > 5:
            logger.info(
                "Runtime has %d tools registered; consider using "
                "LazyToolRegistry to reduce prompt size.",
                len(self._tool_registry.list_tools()),
            )

        # Memory: retrieve relevant facts for this goal
        memory_context = ""
        if self._memory_store and self._memory_store.fact_count > 0:
            memory_context = self._memory_store.retrieve(goal)

        session = self._create_session(
            engine=engine,
            max_turns=max_turns,
            budget=budget,
            extra_tools=tools,
            memory_context=memory_context,
            response_format=response_format,
            images=images,
            input_handler=input_handler,
            system=system,
            provider=provider,
            model=model,
            on_parse_error=on_parse_error,
        )

        run_id = getattr(session, "run_id", "unknown")
        await self._events.emit("run_start", run_id=run_id, goal=goal)

        result: RunResult | None = None
        try:
            result = await session.run(goal)
        except (asyncio.CancelledError, KeyboardInterrupt) as exc:
            # Ensure partial budget is still tracked before re-raising
            logger.info("Run %s cancelled; recording partial budget.", run_id)
            await self._events.emit("error", run_id=run_id, error=exc)
            raise
        except Exception as exc:
            await self._events.emit("error", run_id=run_id, error=exc)
            raise
        finally:
            # Always accumulate whatever budget was used, even on cancellation.
            # Session's budget tracker has the authoritative partial usage.
            if result is not None:
                async with self._totals_lock:
                    self._total_tokens_used += result.tokens_used
                    self._total_cost_usd += result.cost_usd
            elif session.state is not None:
                # Cancelled before RunResult was built -- pull from AgentState
                async with self._totals_lock:
                    self._total_tokens_used += session.state.tokens_used
                    self._total_cost_usd += session.state.cost_usd

        # Memory: store result facts
        if self._memory_store and result.success:
            self._memory_store.store_run_result(
                goal=goal,
                answer=str(result.output) if result.output else "",
                run_id=result.run_id,
            )

        await self._events.emit("run_end", run_id=run_id, result=result)

        return result

    async def stream(
        self,
        goal: str,
        *,
        engine: str = "conversation",
        max_turns: int | None = None,
        input_handler: Callable | None = None,  # type: ignore[type-arg]
        system: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream agent execution events.

        Usage::

            async for event in runtime.stream("Analyze this"):
                print(event.event_type, event.content)

        Only the ``conversation`` engine supports streaming.
        """
        if engine != "conversation":
            raise ValueError("Streaming only supported with engine='conversation'")

        # Auto-connect MCP on first stream
        if self._mcp_configs and not self._mcp_client:
            await self.connect_mcp()

        # Memory: retrieve relevant facts (same path as run())
        memory_context = ""
        if self._memory_store and self._memory_store.fact_count > 0:
            memory_context = self._memory_store.retrieve(goal)

        from arcana.routing.classifier import RuleBasedClassifier
        from arcana.runtime.conversation import ConversationAgent

        if provider or model:
            resolved_provider = provider or self._config.default_provider
            resolved_model = model or self._config.default_model
            if not resolved_model:
                p = self._gateway.get(resolved_provider)
                if p and hasattr(p, "default_model") and isinstance(p.default_model, str):
                    resolved_model = p.default_model
                else:
                    raise ValueError(f"No default model for provider '{resolved_provider}'")
            model_config = ModelConfig(provider=resolved_provider, model_id=resolved_model)
        else:
            model_config = self._resolve_model_config()
        classifier = RuleBasedClassifier()

        agent_kwargs: dict[str, Any] = {
            "gateway": self._gateway,
            "model_config": model_config,
            "tool_gateway": self._tool_gateway,
            "budget_tracker": self._create_budget_tracker(),
            "trace_writer": self._trace_writer,
            "intent_classifier": classifier,
            "max_turns": max_turns or self._config.max_turns,
            "trace_include_prompt_snapshots": self._config.trace_include_prompt_snapshots,
        }
        if memory_context:
            agent_kwargs["memory_context"] = memory_context
        if input_handler is not None:
            agent_kwargs["input_handler"] = input_handler
        resolved_system = system or self._config.system_prompt
        if resolved_system:
            agent_kwargs["system_prompt"] = resolved_system

        # Create WorkingSetBuilder with strategy from Runtime
        from arcana.context.builder import WorkingSetBuilder
        from arcana.contracts.context import ContextStrategy, TokenBudget

        stream_system = resolved_system or "You are a helpful assistant. Answer the user's request directly and completely."
        ctx_strategy = getattr(self, "_context_strategy", None) or ContextStrategy()
        ctx_builder = WorkingSetBuilder(
            identity=stream_system,
            token_budget=TokenBudget(total_window=128_000),
            goal=None,
            gateway=self._gateway,
            strategy=ctx_strategy,
        )
        agent_kwargs["context_builder"] = ctx_builder

        agent = ConversationAgent(**agent_kwargs)

        async for event in agent.astream(goal):
            yield event

    async def connect_mcp(self) -> list[str]:
        """Connect to configured MCP servers and register tools.

        Returns list of registered MCP tool names.
        """
        if not self._mcp_configs:
            return []

        from arcana.mcp.setup import setup_mcp_tools

        # Ensure we have a tool registry/gateway
        if not self._tool_registry:
            from arcana.tool_gateway.gateway import ToolGateway
            from arcana.tool_gateway.registry import ToolRegistry

            self._tool_registry = ToolRegistry()
            self._tool_gateway = ToolGateway(registry=self._tool_registry)

        self._mcp_client = await setup_mcp_tools(
            self._mcp_configs, self._tool_registry
        )

        return [name for name, _ in self._mcp_client.get_all_tools()]

    async def close(self) -> None:
        """Cleanup runtime resources."""
        if self._mcp_client:
            await self._mcp_client.disconnect_all()
            self._mcp_client = None

        # Close tool gateway (releases execution backend resources)
        if self._tool_gateway is not None:
            await self._tool_gateway.close()

        # Close provider HTTP clients to release connection pools
        if self._gateway is not None:
            await self._gateway.close()

    @asynccontextmanager
    async def session(
        self,
        *,
        engine: str = "conversation",
        max_turns: int | None = None,
        budget: Budget | None = None,
        tools: list[Callable] | None = None,  # type: ignore[type-arg]
        system: str | None = None,
    ) -> AsyncGenerator[Session, None]:
        """
        Create a session for manual control.

        Usage:
            async with runtime.session() as s:
                result = await s.run("Do something")
                print(s.state)
                print(s.trace_events)
        """
        s = self._create_session(
            engine=engine,
            max_turns=max_turns,
            budget=budget,
            extra_tools=tools,
            system=system,
        )
        try:
            yield s
        finally:
            pass  # Session cleanup if needed

    @asynccontextmanager
    async def chat(
        self,
        *,
        system_prompt: str | None = None,
        max_turns_per_message: int = 10,
        budget: Budget | None = None,
        input_handler: Callable | None = None,  # type: ignore[type-arg]
        max_history: int | None = None,
    ) -> AsyncGenerator[ChatSession, None]:
        """Create a multi-turn chat session.

        Unlike ``run()`` (single goal -> result), ``chat()`` maintains
        conversation history across multiple user messages.  Each ``send()``
        is one conversation turn where the agent may use tools before
        responding.

        Args:
            max_history: Maximum number of non-system messages to retain.
                When set, older non-system messages are trimmed after each
                ``send()`` / ``stream()`` call.  System messages are always
                preserved.  ``None`` (default) means unlimited -- no trimming.

        Usage::

            async with runtime.chat() as c:
                r = await c.send("Hello")
                r = await c.send("Tell me more about X")
                print(c.total_cost_usd)
        """
        session = ChatSession(
            runtime=self,
            system_prompt=system_prompt,
            max_turns_per_message=max_turns_per_message,
            budget=budget,
            input_handler=input_handler,
            max_history=max_history,
        )
        try:
            yield session
        finally:
            pass  # Cleanup if needed

    def create_chat_session(
        self,
        *,
        system_prompt: str | None = None,
        max_turns_per_message: int = 10,
        budget: Budget | None = None,
        input_handler: Callable | None = None,  # type: ignore[type-arg]
        max_history: int | None = None,
    ) -> ChatSession:
        """Create a chat session without a context manager.

        Use this when you need to hold a session across HTTP requests
        or other boundaries where ``async with runtime.chat()`` is
        inconvenient::

            session = runtime.create_chat_session()
            r = await session.send("Hello")
            # ... later, in another request ...
            r = await session.send("Follow up")

        The session has the same API as one from ``runtime.chat()``.
        """
        return ChatSession(
            runtime=self,
            system_prompt=system_prompt,
            max_turns_per_message=max_turns_per_message,
            budget=budget,
            input_handler=input_handler,
            max_history=max_history,
        )

    async def team(
        self,
        goal: str,
        agents: list[AgentConfig],
        *,
        max_rounds: int = 3,
        budget: Budget | None = None,
        mode: str = "shared",
    ) -> TeamResult:
        """
        Run a team of agents on a shared goal.

        Two collaboration modes:

        - ``"shared"`` (default): All agents share one conversation
          history. Each agent sees everything every other agent said.
          Best for open discussion.

        - ``"session"``: Each agent has an independent conversation
          context. Other agents' messages arrive as user messages in
          the recipient's history. Best for focused, role-based work.

        Args:
            goal: The shared objective
            agents: List of agent configurations
            max_rounds: Maximum conversation rounds (each agent speaks once per round)
            budget: Budget for the entire team run
            mode: Collaboration mode — ``"shared"`` or ``"session"``
        """
        if mode not in ("shared", "session"):
            raise ValueError(f"Invalid team mode '{mode}'. Use 'shared' or 'session'.")

        from arcana.contracts.llm import (
            LLMRequest,
            Message,
            MessageRole,
        )
        from arcana.gateway.budget import BudgetTracker

        team_budget = budget or self._budget_policy
        budget_tracker = BudgetTracker(
            max_cost_usd=team_budget.max_cost_usd,
            max_tokens=team_budget.max_tokens,
        )

        conversation_log: list[dict[str, Any]] = []
        agent_outputs: dict[str, str] = {}
        total_tokens = 0

        # --- Mode-specific state ---
        if mode == "shared":
            shared_messages: list[Message] = [
                Message(role=MessageRole.USER, content=f"Team goal: {goal}"),
            ]
        else:
            # Session mode: per-agent independent histories + inboxes
            agent_histories: dict[str, list[Message]] = {}
            agent_inboxes: dict[str, list[Message]] = {}
            for ac in agents:
                agent_histories[ac.name] = [
                    Message(
                        role=MessageRole.SYSTEM,
                        content=(
                            f"{ac.prompt}\n\n"
                            f"You are '{ac.name}' in a team. "
                            f"Other agents will send you messages. "
                            f"When the team has fully addressed the goal, "
                            f"end your response with [DONE]."
                        ),
                    ),
                    Message(role=MessageRole.USER, content=f"Team goal: {goal}"),
                ]
                agent_inboxes[ac.name] = []

        for round_num in range(max_rounds):
            for agent_config in agents:
                # Budget check before each agent's turn
                budget_tracker.check_budget()

                # --- Build messages per mode ---
                if mode == "shared":
                    agent_messages = [
                        Message(
                            role=MessageRole.SYSTEM,
                            content=(
                                f"{agent_config.prompt}\n\n"
                                f"You are '{agent_config.name}' in a team discussion. "
                                f"Other agents can see your response. "
                                f"When the team has fully addressed the goal, "
                                f"the last agent should end with [DONE]."
                            ),
                        ),
                        *shared_messages,
                    ]
                else:
                    # Session: deliver inbox then use agent's own history
                    for inbox_msg in agent_inboxes[agent_config.name]:
                        agent_histories[agent_config.name].append(inbox_msg)
                    agent_inboxes[agent_config.name] = []
                    agent_messages = agent_histories[agent_config.name]

                # Resolve model config for this agent
                provider = agent_config.provider or self._config.default_provider
                model_id = agent_config.model or self._config.default_model
                if not model_id:
                    p = self._gateway.get(provider)
                    if (
                        p
                        and hasattr(p, "default_model")
                        and isinstance(p.default_model, str)
                    ):
                        model_id = p.default_model
                    else:
                        msg = (
                            f"No default model configured for provider '{provider}'. "
                            "Set model on AgentConfig or register a provider with a default_model."
                        )
                        raise ValueError(msg)

                config = ModelConfig(provider=provider, model_id=model_id)

                # Make LLM call
                request = LLMRequest(messages=agent_messages)
                response = await self._gateway.generate(
                    request=request, config=config
                )

                # Track budget
                if response.usage:
                    budget_tracker.add_usage(response.usage)
                    total_tokens += response.usage.total_tokens

                agent_text = response.content or ""

                # --- Update state per mode ---
                if mode == "shared":
                    shared_messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=f"[{agent_config.name}]: {agent_text}",
                        )
                    )
                else:
                    # Add own response to own history
                    agent_histories[agent_config.name].append(
                        Message(role=MessageRole.ASSISTANT, content=agent_text)
                    )
                    # Deliver to other agents' inboxes
                    for other in agents:
                        if other.name != agent_config.name:
                            agent_inboxes[other.name].append(
                                Message(
                                    role=MessageRole.USER,
                                    content=f"Message from {agent_config.name}:\n{agent_text}",
                                )
                            )

                # Log
                agent_outputs[agent_config.name] = agent_text
                conversation_log.append(
                    {
                        "round": round_num + 1,
                        "agent": agent_config.name,
                        "content": agent_text,
                        "tokens": (
                            response.usage.total_tokens if response.usage else 0
                        ),
                    }
                )

                # Trace
                if self._trace_writer:
                    from arcana.contracts.trace import EventType, TraceEvent

                    self._trace_writer.write(TraceEvent(
                        run_id=f"team-{uuid4()}",
                        event_type=EventType.AGENT_TURN,
                        metadata={
                            "round": round_num + 1,
                            "agent": agent_config.name,
                            "content_length": len(agent_text),
                            "tokens": (
                                response.usage.total_tokens
                                if response.usage
                                else 0
                            ),
                            "mode": mode,
                        },
                    ))

                # Check if done (any agent says [DONE])
                if "[done]" in agent_text.lower():
                    return TeamResult(
                        output=agent_text.replace("[DONE]", "")
                        .replace("[done]", "")
                        .strip(),
                        success=True,
                        rounds=round_num + 1,
                        total_tokens=total_tokens,
                        total_cost_usd=budget_tracker.to_snapshot().cost_usd,
                        agent_outputs=agent_outputs,
                        conversation_log=conversation_log,
                    )

        # Max rounds reached
        last_output = conversation_log[-1]["content"] if conversation_log else ""
        return TeamResult(
            output=last_output,
            success=False,
            rounds=max_rounds,
            total_tokens=total_tokens,
            total_cost_usd=budget_tracker.to_snapshot().cost_usd,
            agent_outputs=agent_outputs,
            conversation_log=conversation_log,
        )

    async def chain(
        self,
        steps: list[ChainStep | list[ChainStep]],
        *,
        input: str = "",
        budget: Budget | None = None,
    ) -> ChainResult:
        """
        Run a pipeline of agent steps with optional parallel branches.

        Each step's output is automatically passed as context to the next.
        Use nested lists for parallel execution::

            steps=[
                ChainStep(name="filter", ...),
                [ChainStep(name="classify", ...), ChainStep(name="analyze", ...)],
                ChainStep(name="integrate", ...),
            ]

        Args:
            steps: Ordered list. Each element is a ``ChainStep`` (sequential)
                or a ``list[ChainStep]`` (parallel branch).
            input: Initial input text fed as context to the first step
            budget: Shared budget across all steps
        """
        step_outputs: dict[str, Any] = {}
        total_tokens = 0
        total_cost_usd = 0.0
        current_context: str = input

        def _effective_budget(step_budget: Budget | None) -> Budget | None:
            """Compute effective budget: min(step_budget, chain_remaining)."""
            if budget is None and step_budget is None:
                return None
            chain_remaining: Budget | None = None
            if budget is not None:
                chain_remaining = Budget(
                    max_cost_usd=max(0.0, budget.max_cost_usd - total_cost_usd),
                    max_tokens=max(0, budget.max_tokens - total_tokens),
                )
            if step_budget is not None and chain_remaining is not None:
                return Budget(
                    max_cost_usd=min(step_budget.max_cost_usd, chain_remaining.max_cost_usd),
                    max_tokens=min(step_budget.max_tokens, chain_remaining.max_tokens),
                )
            return step_budget if step_budget is not None else chain_remaining

        for step_or_group in steps:
            if isinstance(step_or_group, list):
                # Parallel execution
                parallel_steps = step_or_group

                async def _run_parallel(s: ChainStep, ctx: str, eff_budget: Budget | None) -> tuple[str, RunResult]:
                    ctx_val: dict[str, Any] | str | None = ctx if ctx else None
                    r = await self.run(
                        s.goal, system=s.system, response_format=s.response_format,
                        tools=s.tools, budget=eff_budget, context=ctx_val,
                        provider=s.provider, model=s.model,
                        on_parse_error=s.on_parse_error,
                    )
                    return s.name, r

                results = await asyncio.gather(
                    *[_run_parallel(s, current_context, _effective_budget(s.budget)) for s in parallel_steps],
                )

                failed = False
                context_parts: list[str] = []
                for name, result in results:
                    step_outputs[name] = result.output
                    total_tokens += result.tokens_used
                    total_cost_usd += result.cost_usd
                    if not result.success:
                        failed = True
                    if result.output is not None:
                        if isinstance(result.output, BaseModel):
                            output_str = result.output.model_dump_json(indent=2)
                        else:
                            output_str = str(result.output)
                        context_parts.append(f"[{name}]:\n{output_str}")

                if failed:
                    return ChainResult(
                        output=None, success=False, steps=step_outputs,
                        total_tokens=total_tokens, total_cost_usd=total_cost_usd,
                    )
                current_context = "\n\n".join(context_parts)
            else:
                # Sequential execution
                step = step_or_group
                ctx: dict[str, Any] | str | None = current_context if current_context else None

                result = await self.run(
                    step.goal, system=step.system, response_format=step.response_format,
                    tools=step.tools, budget=_effective_budget(step.budget), context=ctx,
                    provider=step.provider, model=step.model,
                    on_parse_error=step.on_parse_error,
                )

                step_outputs[step.name] = result.output
                total_tokens += result.tokens_used
                total_cost_usd += result.cost_usd

                if not result.success:
                    return ChainResult(
                        output=result.output, success=False, steps=step_outputs,
                        total_tokens=total_tokens, total_cost_usd=total_cost_usd,
                    )

                if result.output is not None:
                    if isinstance(result.output, BaseModel):
                        current_context = result.output.model_dump_json(indent=2)
                    else:
                        current_context = str(result.output)
                else:
                    current_context = ""

        if steps:
            last = steps[-1]
            if isinstance(last, list):
                output = {s.name: step_outputs[s.name] for s in last}
            else:
                output = step_outputs.get(last.name)
        else:
            output = None

        return ChainResult(
            output=output, success=True, steps=step_outputs,
            total_tokens=total_tokens, total_cost_usd=total_cost_usd,
        )

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    async def run_batch(
        self,
        tasks: list[dict[str, Any]],
        *,
        concurrency: int = 5,
    ) -> BatchResult:
        """Run multiple independent tasks concurrently.

        Each task dict must have a ``"goal"`` key and may include any
        keyword arguments accepted by :meth:`run` (``tools``, ``system``,
        ``provider``, ``model``, ``response_format``, etc.).

        Individual failures do not crash the batch -- the corresponding
        ``RunResult`` will have ``success=False``.

        Args:
            tasks: List of task dicts, each with ``"goal"`` key and
                optional kwargs matching ``run()`` parameters.
            concurrency: Maximum number of concurrent runs (default 5).

        Returns:
            BatchResult with all results preserving input order.
        """
        if not tasks:
            return BatchResult()

        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded_run(task: dict[str, Any]) -> RunResult:
            goal = task.pop("goal") if "goal" in task else task.get("goal", "")
            # Rebuild task dict without 'goal' for kwargs
            kwargs = {k: v for k, v in task.items() if k != "goal"}
            async with semaphore:
                try:
                    return await self.run(goal, **kwargs)
                except Exception as exc:
                    logger.warning("Batch task failed: %s", exc)
                    return RunResult(
                        output=str(exc),
                        success=False,
                    )

        # Copy each task dict so pop doesn't mutate caller's data
        results = await asyncio.gather(
            *[_guarded_run(dict(t)) for t in tasks],
        )

        total_tokens = sum(r.tokens_used for r in results)
        total_cost = sum(r.cost_usd for r in results)
        succeeded = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        return BatchResult(
            results=list(results),
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            succeeded=succeeded,
            failed=failed,
        )

    # ------------------------------------------------------------------
    # Budget scoping
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def budget_scope(
        self,
        max_cost_usd: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[BudgetScope, None]:
        """Create a scoped budget context.

        Runs inside the scope deduct from both the scope budget and
        the runtime's global budget. When the scope is exhausted,
        only runs inside the scope are affected.
        """
        scope = BudgetScope(self, max_cost_usd=max_cost_usd, max_tokens=max_tokens)
        yield scope

    # ------------------------------------------------------------------
    # Graph node factories
    # ------------------------------------------------------------------

    def make_llm_node(self, *, system_prompt: str | None = None) -> LLMNode:
        """Create an LLMNode pre-wired with this Runtime's gateway and model config."""
        from arcana.graph.nodes.llm_node import LLMNode

        model_config = self._resolve_model_config()
        return LLMNode(
            self._gateway,
            model_config=model_config,
            system_prompt=system_prompt,
        )

    def make_tool_node(self) -> ToolNode:
        """Create a ToolNode pre-wired with this Runtime's tool gateway."""
        from arcana.graph.nodes.tool_node import ToolNode

        if self._tool_gateway is None:
            raise ValueError(
                "No tools registered. Pass tools= to Runtime() before calling make_tool_node()."
            )
        return ToolNode(self._tool_gateway)

    # ------------------------------------------------------------------
    # Graph factory
    # ------------------------------------------------------------------

    def graph(self, state_schema: type | None = None) -> StateGraph:
        """
        Create a StateGraph connected to this Runtime's resources.

        The graph uses Runtime's gateway for LLM calls and Runtime's
        tool_gateway for tool execution when called from node functions.

        Args:
            state_schema: Pydantic model for graph state (optional)

        Returns:
            StateGraph ready for node/edge configuration

        Example::

            graph = runtime.graph(state_schema=MyState)
            graph.add_node("search", search_fn)
            graph.add_edge(START, "search")
            app = graph.compile()
            result = await app.ainvoke(initial_state)
        """
        from arcana.graph.state_graph import StateGraph

        return StateGraph(state_schema=state_schema)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def budget_remaining_usd(self) -> float | None:
        """Remaining USD budget, or None if no budget limit is set."""
        if self._budget_policy.max_cost_usd is None:
            return None
        return max(0.0, self._budget_policy.max_cost_usd - self._total_cost_usd)

    @property
    def budget_used_usd(self) -> float:
        """Total USD spent across all runs."""
        return self._total_cost_usd

    @property
    def tokens_remaining(self) -> int | None:
        """Remaining token budget, or None if no token limit is set."""
        if self._budget_policy.max_tokens is None:
            return None
        return max(0, self._budget_policy.max_tokens - self._total_tokens_used)

    @property
    def tokens_used(self) -> int:
        """Total tokens used across all runs."""
        return self._total_tokens_used

    @property
    def providers(self) -> list[str]:
        """List registered provider names."""
        return list(self._gateway.list_providers())

    @property
    def fallback_order(self) -> list[str]:
        """Return the provider fallback order for the default provider.

        The first element is the default (primary) provider. Subsequent
        elements are fallback providers in the order they will be tried
        when the primary exhausts retries on a retryable error.
        """
        default = self._gateway.default_provider
        if default is None:
            return []
        chain = self._gateway.get_fallback_chain(default)
        return [default, *chain]

    @property
    def tools(self) -> list[str]:
        """List registered tool names."""
        if self._tool_registry:
            return list(self._tool_registry.list_tools())
        return []

    @property
    def namespace(self) -> str | None:
        """The namespace for tenant isolation, or None."""
        return self._namespace

    @property
    def memory(self) -> Any:
        """Access the memory store (if enabled)."""
        return self._memory_store

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_providers(self, providers: dict[str, str | dict[str, str]]) -> Any:
        """Setup provider registry from {name: api_key} or {name: {api_key, model}} dict."""
        from arcana.gateway.providers.openai_compatible import (
            OpenAICompatibleProvider,
            create_deepseek_provider,
            create_gemini_provider,
            create_glm_provider,
            create_kimi_provider,
            create_minimax_provider,
            create_ollama_provider,
        )
        from arcana.gateway.registry import ModelGatewayRegistry

        gateway = ModelGatewayRegistry()

        factory_map: dict[str, Callable[[str], Any]] = {
            "deepseek": lambda key: create_deepseek_provider(key),
            "gemini": lambda key: create_gemini_provider(key),
            "kimi": lambda key: create_kimi_provider(key),
            "glm": lambda key: create_glm_provider(key),
            "minimax": lambda key: create_minimax_provider(key),
            "ollama": lambda _: create_ollama_provider(),
            "openai": lambda key: OpenAICompatibleProvider(
                provider_name="openai",
                api_key=key,
                base_url="https://api.openai.com/v1",
                default_model="gpt-4o-mini",
            ),
        }

        # Handle Anthropic separately (needs SDK)
        def _create_anthropic(key: str) -> Any:
            from arcana.gateway.providers.anthropic import AnthropicProvider

            return AnthropicProvider(api_key=key)

        factory_map["anthropic"] = _create_anthropic

        for name, config_value in providers.items():
            if isinstance(config_value, dict):
                api_key = config_value.get("api_key", "")
                model_override = config_value.get("model")
            else:
                api_key = config_value
                model_override = None

            # Resolve from env if empty
            if not api_key:
                env_var = f"{name.upper()}_API_KEY"
                api_key = os.environ.get(env_var, "")

            if name in factory_map:
                provider_instance = factory_map[name](api_key)
                # Override default model if specified
                if model_override and hasattr(provider_instance, "_default_model"):
                    provider_instance._default_model = model_override
                gateway.register(name, provider_instance)
            elif isinstance(config_value, dict) and config_value.get("base_url"):
                # Custom OpenAI-compatible provider with explicit base_url
                from arcana.gateway.providers.openai_compatible import ProviderProfile

                custom_profile = ProviderProfile(
                    tool_calls=config_value.get("tool_calls", True),
                    json_schema=config_value.get("json_schema", False),
                    json_mode=config_value.get("json_mode", True),
                    stream_options=config_value.get("stream_options", False),
                )
                provider_instance = OpenAICompatibleProvider(
                    provider_name=name,
                    api_key=api_key,
                    base_url=config_value["base_url"],
                    default_model=config_value.get("model"),
                    profile=custom_profile,
                )
                gateway.register(name, provider_instance)
            else:
                raise ValueError(
                    f"Unknown provider '{name}'. "
                    f"Supported: {list(factory_map.keys())}. "
                    f"For custom providers, pass a dict with 'base_url': "
                    f"providers={{'{name}': {{'api_key': '...', 'base_url': '...', 'model': '...'}}}}"
                )

        if providers:
            gateway.set_default(self._config.default_provider)

            # Auto-build fallback chain: default provider falls back to the
            # remaining registered providers in dict insertion order.
            provider_names = list(providers.keys())
            default = self._config.default_provider
            fallbacks = [n for n in provider_names if n != default]
            if fallbacks:
                gateway.set_fallback_chain(default, fallbacks)
                logger.info(
                    "Provider fallback chain: %s -> %s",
                    default,
                    " -> ".join(fallbacks),
                )

        return gateway

    def _setup_tools(
        self, tools: list[Callable]  # type: ignore[type-arg]
    ) -> tuple[Any, Any]:
        """Setup tool registry + gateway from decorated functions."""
        from arcana.sdk import _FunctionToolProvider
        from arcana.tool_gateway.gateway import ToolGateway
        from arcana.tool_gateway.registry import ToolRegistry

        registry = ToolRegistry()
        for func in tools:
            # Support Tool wrapper instances
            if hasattr(func, "_fn"):
                func = func._fn
            spec = getattr(func, "_arcana_tool_spec", None)
            if spec is None:
                continue
            registry.register(_FunctionToolProvider(spec=spec, func=func))

        gateway = ToolGateway(registry=registry)
        return registry, gateway

    def _create_session(
        self,
        engine: str = "conversation",
        max_turns: int | None = None,
        budget: Budget | None = None,
        extra_tools: list[Callable] | None = None,  # type: ignore[type-arg]
        memory_context: str = "",
        response_format: type[BaseModel] | None = None,
        images: list[str] | None = None,
        input_handler: Callable | None = None,  # type: ignore[type-arg]
        system: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        on_parse_error: Callable | None = None,  # type: ignore[type-arg]
    ) -> Session:
        """Create a new session with per-run resources."""
        return Session(
            runtime=self,
            engine=engine,
            max_turns=max_turns or self._config.max_turns,
            budget=budget or self._budget_policy,
            extra_tools=extra_tools,
            memory_context=memory_context,
            response_format=response_format,
            images=images,
            input_handler=input_handler,
            system=system,
            provider=provider,
            model=model,
            on_parse_error=on_parse_error,
        )

    def _create_budget_tracker(self) -> Any:
        """Create a fresh BudgetTracker from the runtime's budget policy."""
        from arcana.gateway.budget import BudgetTracker

        return BudgetTracker(
            max_cost_usd=self._budget_policy.max_cost_usd,
            max_tokens=self._budget_policy.max_tokens,
        )

    def _resolve_model_config(self) -> ModelConfig:
        """Get default ModelConfig.

        Resolution order:
        1. RuntimeConfig.default_model (user-provided)
        2. Provider's default_model attribute
        3. Raise ValueError -- never guess a hardcoded model name
        """
        provider_name = self._config.default_provider
        model_id = self._config.default_model
        if not model_id:
            provider = self._gateway.get(provider_name)
            if (
                provider
                and hasattr(provider, "default_model")
                and isinstance(provider.default_model, str)
            ):
                model_id = provider.default_model
            else:
                msg = (
                    f"No default model configured for provider '{provider_name}'. "
                    "Pass default_model in RuntimeConfig or register a provider with a default_model."
                )
                raise ValueError(msg)
        return ModelConfig(provider=provider_name, model_id=model_id)


class Session:
    """
    Per-run execution context.

    Holds run-scoped resources:
    - run_id
    - per-run budget tracker
    - per-run trace context
    - per-run state
    """

    def __init__(
        self,
        runtime: Runtime,
        engine: str = "conversation",
        max_turns: int = 20,
        budget: Budget | None = None,
        extra_tools: list[Callable] | None = None,  # type: ignore[type-arg]
        memory_context: str = "",
        response_format: type[BaseModel] | None = None,
        images: list[str] | None = None,
        input_handler: Callable | None = None,  # type: ignore[type-arg]
        system: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        on_parse_error: Callable | None = None,  # type: ignore[type-arg]
    ) -> None:
        self._runtime = runtime
        self._engine = engine
        self._max_turns = max_turns
        self._budget_config = budget or Budget()
        self._extra_tools = extra_tools
        self._memory_context = memory_context
        self._response_format = response_format
        self._images = images
        self._input_handler = input_handler
        self._system = system
        self._provider_override = provider
        self._model_override = model
        self._on_parse_error = on_parse_error

        # Per-run resources
        self.run_id = str(uuid4())
        self.state: AgentState | None = None

        # Per-run budget tracker
        from arcana.gateway.budget import BudgetTracker

        self.budget = BudgetTracker(
            max_cost_usd=self._budget_config.max_cost_usd,
            max_tokens=self._budget_config.max_tokens,
        )

    async def run(self, goal: str) -> RunResult:
        """Run a task in this session."""
        import json as _json

        # Merge tools: runtime tools + session extra tools
        tool_gateway = self._runtime._tool_gateway
        # TODO: merge extra_tools if provided

        if self._provider_override or self._model_override:
            resolved_provider = self._provider_override or self._runtime._config.default_provider
            resolved_model_id = self._model_override or self._runtime._config.default_model
            if not resolved_model_id:
                p = self._runtime._gateway.get(resolved_provider)
                if p and hasattr(p, "default_model") and isinstance(p.default_model, str):
                    resolved_model_id = p.default_model
                else:
                    raise ValueError(f"No default model for provider '{resolved_provider}'")
            model_config = ModelConfig(provider=resolved_provider, model_id=resolved_model_id)
        else:
            model_config = self._runtime._resolve_model_config()

        # Convert Pydantic model to JSON schema for the LLM request
        response_format_schema: dict[str, Any] | None = None
        if self._response_format is not None:
            response_format_schema = self._response_format.model_json_schema()

        if self._engine == "conversation":
            from arcana.routing.classifier import RuleBasedClassifier
            from arcana.runtime.conversation import ConversationAgent
            from arcana.sdk import build_content_blocks

            classifier = RuleBasedClassifier()

            # Build kwargs; pass memory_context through to WorkingSetBuilder
            agent_kwargs: dict[str, Any] = {
                "gateway": self._runtime._gateway,
                "model_config": model_config,
                "tool_gateway": tool_gateway,
                "budget_tracker": self.budget,
                "trace_writer": self._runtime._trace_writer,
                "intent_classifier": classifier,
                "max_turns": self._max_turns,
                "trace_include_prompt_snapshots": self._runtime._config.trace_include_prompt_snapshots,
            }
            # Resolve system prompt: run(system=) > RuntimeConfig > default
            resolved_system = self._system or self._runtime._config.system_prompt
            if resolved_system:
                agent_kwargs["system_prompt"] = resolved_system
            if self._memory_context:
                agent_kwargs["memory_context"] = self._memory_context
            if response_format_schema is not None:
                agent_kwargs["response_format_schema"] = response_format_schema
            if self._input_handler is not None:
                agent_kwargs["input_handler"] = self._input_handler

            # Build multimodal content blocks when images are provided
            user_content = build_content_blocks(goal, self._images)
            if isinstance(user_content, list):
                agent_kwargs["initial_user_content"] = user_content

            # Create WorkingSetBuilder with strategy from Runtime
            from arcana.context.builder import WorkingSetBuilder
            from arcana.contracts.context import ContextStrategy, TokenBudget

            ctx_system = resolved_system or "You are a helpful assistant. Answer the user's request directly and completely."
            ctx_strategy = getattr(self._runtime, "_context_strategy", None) or ContextStrategy()
            ctx_builder = WorkingSetBuilder(
                identity=ctx_system,
                token_budget=TokenBudget(total_window=128_000),
                goal=None,
                gateway=self._runtime._gateway,
                strategy=ctx_strategy,
            )
            agent_kwargs["context_builder"] = ctx_builder

            agent = ConversationAgent(**agent_kwargs)
            self.state = await agent.run(goal)

        else:
            # V1: Agent + AdaptivePolicy
            from arcana.contracts.runtime import RuntimeConfig as V1Config
            from arcana.runtime.agent import Agent
            from arcana.runtime.policies.adaptive import AdaptivePolicy
            from arcana.runtime.reducers.default import DefaultReducer

            v1_agent = Agent(
                policy=AdaptivePolicy(),
                reducer=DefaultReducer(),
                gateway=self._runtime._gateway,
                config=V1Config(max_steps=self._max_turns),
                tool_gateway=tool_gateway,
                budget_tracker=self.budget,
                trace_writer=self._runtime._trace_writer,
            )
            self.state = await v1_agent.run(goal)

        raw_output = self.state.working_memory.get(
            "answer", self.state.working_memory.get("result", "")
        )
        # Strip completion markers that the LLM may include
        clean_output = raw_output.replace("[DONE]", "").replace("[done]", "").strip() if isinstance(raw_output, str) else raw_output

        # Parse and validate structured output when response_format is set
        parsed_model = None
        if self._response_format is not None:
            # Normalise: if raw_output is already a dict (e.g. provider
            # pre-parsed), skip json.loads; otherwise parse the string.
            if isinstance(clean_output, dict):
                parsed_json = clean_output
            elif isinstance(clean_output, str):
                try:
                    parsed_json = _json.loads(strip_code_fences(clean_output))
                except _json.JSONDecodeError as parse_error:
                    parsed_json = None
                    _first_parse_error: Exception = parse_error
                else:
                    _first_parse_error = None  # type: ignore[assignment]
            else:
                parsed_json = None
                _first_parse_error = TypeError(  # type: ignore[assignment]
                    f"Expected str or dict, got {type(clean_output).__name__}"
                )

            # Validate against response_format (always, whether from
            # json.loads or from a pre-parsed dict).
            if parsed_json is not None:
                try:
                    parsed_model = self._response_format.model_validate(parsed_json)
                    # Backward compat: output holds the validated model too
                    clean_output = parsed_model
                except Exception as validate_error:
                    parsed_json = None
                    _first_parse_error = validate_error  # type: ignore[assignment]

            # If we still don't have a parsed model, try the error callback
            if parsed_model is None:
                if self._on_parse_error is not None:
                    try:
                        if asyncio.iscoroutinefunction(self._on_parse_error):
                            fixed = await self._on_parse_error(
                                clean_output, _first_parse_error,  # type: ignore[possibly-undefined]
                            )
                        else:
                            fixed = self._on_parse_error(
                                clean_output, _first_parse_error,  # type: ignore[possibly-undefined]
                            )
                        # Ensure callback result is a model, not a raw dict
                        if fixed is not None:
                            if isinstance(fixed, dict):
                                fixed = self._response_format.model_validate(fixed)
                            elif not isinstance(fixed, BaseModel):
                                fixed = self._response_format.model_validate(fixed)
                            return RunResult(
                                output=fixed,
                                parsed=fixed,
                                success=True,
                                steps=self.state.current_step,
                                tokens_used=self.state.tokens_used,
                                cost_usd=self.state.cost_usd,
                                run_id=self.state.run_id,
                                context_report=self.state.last_context_report,
                            )
                    except Exception:
                        logger.debug("on_parse_error callback failed", exc_info=True)
                return RunResult(
                    output=clean_output,
                    parsed=None,
                    success=False,
                    steps=self.state.current_step,
                    tokens_used=self.state.tokens_used,
                    cost_usd=self.state.cost_usd,
                    run_id=self.state.run_id,
                    context_report=self.state.last_context_report,
                )

        return RunResult(
            output=clean_output,
            parsed=parsed_model,
            success=self.state.status.value == "completed",
            steps=self.state.current_step,
            tokens_used=self.state.tokens_used,
            cost_usd=self.state.cost_usd,
            run_id=self.state.run_id,
            context_report=self.state.last_context_report,
        )


class RunResult(BaseModel):
    """Result of a runtime run."""

    output: Any = None
    parsed: Any = None
    success: bool = False
    steps: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    run_id: str = ""
    context_report: Any = None  # ContextReport | None (Any to avoid circular import)


class ChatResponse(BaseModel):
    """Response from a single chat turn."""

    content: str = ""
    tool_calls_made: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    context_report: Any = None  # ContextReport | None


class ChatSession:
    """Multi-turn conversational session.

    Unlike Session (single goal -> result), ChatSession maintains conversation
    history across multiple user messages. Each ``send()`` is one conversation
    turn where the agent may use tools before responding.

    The LLM sees the full conversation history (subject to context compression)
    and can use tools across turns. Budget accumulates across the entire chat.
    """

    def __init__(
        self,
        runtime: Runtime,
        *,
        system_prompt: str | None = None,
        max_turns_per_message: int = 10,
        budget: Budget | None = None,
        input_handler: Callable | None = None,  # type: ignore[type-arg]
        max_history: int | None = None,
    ) -> None:
        self._runtime = runtime
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._max_turns = max_turns_per_message
        self._budget_config = budget or runtime._budget_policy
        self._input_handler = input_handler
        self._max_history: int | None = max_history  # None = unlimited (backward compat)
        self._session_id = str(uuid4())

        # Persistent state across send() calls
        from arcana.contracts.llm import Message, MessageRole

        self._messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=self._system_prompt),
        ]

        # Shared budget tracker for the entire chat session
        from arcana.gateway.budget import BudgetTracker

        self._budget_tracker = BudgetTracker(
            max_cost_usd=self._budget_config.max_cost_usd,
            max_tokens=self._budget_config.max_tokens,
        )

        self._total_tokens = 0
        self._total_cost_usd = 0.0
        self._turn_count = 0

    def _trim_history(self) -> None:
        """Trim message history to max_history non-system messages.

        System messages are always preserved. When ``max_history`` is ``None``
        (default), no trimming occurs.
        """
        if self._max_history is None or len(self._messages) <= self._max_history:
            return
        from arcana.contracts.llm import MessageRole

        system_msgs = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        non_system = [m for m in self._messages if m.role != MessageRole.SYSTEM]
        if len(non_system) <= self._max_history:
            return
        self._messages = system_msgs + non_system[-self._max_history:]

    async def send(
        self,
        message: str,
        *,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Send a message and get the agent's response.

        Delegates to ``ConversationAgent`` for the full turn loop, gaining
        all V2 features: ask_user, lazy tools, diagnostics, fidelity
        compression, thinking assessment, and rich
        streaming events.

        Args:
            message: The user's message text.
            images: Optional list of image inputs (URLs, file paths, or
                data URIs) to include with this message.

        Returns:
            ChatResponse with the agent's reply and usage metrics.
        """
        from arcana.contracts.llm import Message, MessageRole
        from arcana.contracts.streaming import StreamEventType

        # 1. Append user message to persistent history
        if images:
            from arcana.sdk import build_content_blocks

            content_blocks = build_content_blocks(message, images)
            self._messages.append(Message(role=MessageRole.USER, content=content_blocks))
        else:
            self._messages.append(Message(role=MessageRole.USER, content=message))
        self._turn_count += 1

        # 2. Build ConversationAgent for this send()
        agent = self._build_agent(message)

        # 3. Run agent (consume all stream events)
        tool_calls_made = 0
        async for event in agent.astream(message):
            if event.event_type == StreamEventType.TOOL_END:
                tool_calls_made += 1

        # 4. Extract results from agent state
        state = agent._state
        turn_tokens = state.tokens_used if state else 0
        turn_cost = state.cost_usd if state else 0.0
        answer = state.working_memory.get("answer", "") if state else ""

        # 5. Update persistent state from agent's final messages
        self._messages = agent.final_messages

        # ConversationAgent does not append the final answer to messages
        # when the turn is assessed as "completed" (it stores it in
        # working_memory instead). For multi-turn chat we need the
        # assistant reply in the history so the next send() sees it.
        if answer:
            from arcana.contracts.llm import Message, MessageRole

            last_is_assistant = (
                self._messages
                and self._messages[-1].role == MessageRole.ASSISTANT
            )
            if not last_is_assistant:
                self._messages.append(
                    Message(role=MessageRole.ASSISTANT, content=answer)
                )

        self._total_tokens += turn_tokens
        self._total_cost_usd += turn_cost

        # 6. Trim history
        self._trim_history()

        # 7. Build response
        context_report = state.last_context_report if state else None

        return ChatResponse(
            content=answer,
            tool_calls_made=tool_calls_made,
            tokens_used=turn_tokens,
            cost_usd=turn_cost,
            context_report=context_report,
        )

    async def stream(self, message: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream version of send(). Yields events including the response.

        Delegates to ``ConversationAgent`` for the full turn loop, gaining
        all V2 features. Yields ``StreamEvent`` objects for LLM chunks,
        tool results, and the final response. After this generator is
        exhausted the message history is updated just like ``send()``.
        """
        from arcana.contracts.llm import Message, MessageRole

        # 1. Append user message to persistent history
        self._messages.append(Message(role=MessageRole.USER, content=message))
        self._turn_count += 1

        # 2. Build ConversationAgent for this send
        agent = self._build_agent(message)

        # 3. Stream events from agent
        async for event in agent.astream(message):
            yield event

        # 4. Update persistent state
        state = agent._state
        turn_tokens = state.tokens_used if state else 0
        turn_cost = state.cost_usd if state else 0.0

        self._messages = agent.final_messages

        # Ensure final assistant answer is in message history (see send())
        answer = state.working_memory.get("answer", "") if state else ""
        if answer:
            from arcana.contracts.llm import Message, MessageRole

            last_is_assistant = (
                self._messages
                and self._messages[-1].role == MessageRole.ASSISTANT
            )
            if not last_is_assistant:
                self._messages.append(
                    Message(role=MessageRole.ASSISTANT, content=answer)
                )

        self._total_tokens += turn_tokens
        self._total_cost_usd += turn_cost

        # 5. Trim history
        self._trim_history()

    # ------------------------------------------------------------------
    # Agent builder
    # ------------------------------------------------------------------

    def _build_agent(self, goal: str) -> Any:
        """Build a ConversationAgent for a single send/stream call."""
        from arcana.context.builder import WorkingSetBuilder
        from arcana.contracts.context import ContextStrategy, TokenBudget
        from arcana.runtime.conversation import ConversationAgent

        model_config = self._runtime._resolve_model_config()

        # Resolve context window from provider
        context_window = 128_000
        provider = self._runtime._gateway.get(model_config.provider)
        if provider and hasattr(provider, "profile") and provider.profile:
            context_window = getattr(
                provider.profile, "context_window", context_window,
            )

        context_builder = WorkingSetBuilder(
            identity=self._system_prompt,
            token_budget=TokenBudget(total_window=context_window),
            goal=goal,
            gateway=self._runtime._gateway,
            strategy=(
                getattr(self._runtime, "_context_strategy", None)
                or ContextStrategy()
            ),
        )

        agent_kwargs: dict[str, Any] = {
            "gateway": self._runtime._gateway,
            "model_config": model_config,
            "budget_tracker": self._budget_tracker,
            "trace_writer": self._runtime._trace_writer,
            "max_turns": self._max_turns,
            "system_prompt": self._system_prompt,
            "context_builder": context_builder,
            "initial_messages": list(self._messages),  # Copy
            "input_handler": self._input_handler,
            "trace_include_prompt_snapshots": self._runtime._config.trace_include_prompt_snapshots,
        }

        # Tool gateway
        if self._runtime._tool_gateway:
            agent_kwargs["tool_gateway"] = self._runtime._tool_gateway

        return ConversationAgent(**agent_kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        """Total cost across all send() calls in this session."""
        return self._total_cost_usd

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all send() calls in this session."""
        return self._total_tokens

    @property
    def message_count(self) -> int:
        """Number of messages in the conversation (including system prompt)."""
        return len(self._messages)

    @property
    def history(self) -> list[dict[str, str]]:
        """Return conversation history as a list of role/content dicts.

        Only includes user and assistant messages (excludes system, tool).
        """
        result = []
        for msg in self._messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role in ("user", "assistant"):
                content = msg.content if isinstance(msg.content, str) else ""
                result.append({"role": role, "content": content})
        return result

    @property
    def session_id(self) -> str:
        """Unique identifier for this chat session."""
        return self._session_id
