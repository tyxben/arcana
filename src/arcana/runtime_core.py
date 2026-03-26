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

    model_config = {"arbitrary_types_allowed": True}


class ChainResult(BaseModel):
    """Result of a chain/pipeline execution."""

    output: Any = None  # Final step's output
    success: bool = False
    steps: dict[str, Any] = Field(default_factory=dict)  # name -> output
    total_tokens: int = 0
    total_cost_usd: float = 0.0


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

        if self._max_cost_usd is not None and self._cost_used >= self._max_cost_usd:
            raise BudgetExceededError("Scoped budget exhausted (cost)", budget_type="cost")
        if self._max_tokens is not None and self._tokens_used >= self._max_tokens:
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
    ) -> None:
        self._config = config or RuntimeConfig()
        self._budget_policy = budget or Budget()
        self._namespace = namespace

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

        # Cumulative budget tracking across runs
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
        result = await session.run(goal)

        # Accumulate usage into runtime-level totals
        self._total_tokens_used += result.tokens_used
        self._total_cost_usd += result.cost_usd

        # Memory: store result facts
        if self._memory_store and result.success:
            self._memory_store.store_run_result(
                goal=goal,
                answer=str(result.output) if result.output else "",
                run_id=result.run_id,
            )

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
        }
        if memory_context:
            agent_kwargs["memory_context"] = memory_context
        if input_handler is not None:
            agent_kwargs["input_handler"] = input_handler
        resolved_system = system or self._config.system_prompt
        if resolved_system:
            agent_kwargs["system_prompt"] = resolved_system

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
    ) -> AsyncGenerator[ChatSession, None]:
        """Create a multi-turn chat session.

        Unlike ``run()`` (single goal -> result), ``chat()`` maintains
        conversation history across multiple user messages.  Each ``send()``
        is one conversation turn where the agent may use tools before
        responding.

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
        )
        try:
            yield session
        finally:
            pass  # Cleanup if needed

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

        for step_or_group in steps:
            if isinstance(step_or_group, list):
                # Parallel execution
                parallel_steps = step_or_group

                async def _run_parallel(s: ChainStep, ctx: str) -> tuple[str, RunResult]:
                    ctx_val: dict[str, Any] | str | None = ctx if ctx else None
                    r = await self.run(
                        s.goal, system=s.system, response_format=s.response_format,
                        tools=s.tools, budget=budget, context=ctx_val,
                        provider=s.provider, model=s.model,
                        on_parse_error=s.on_parse_error,
                    )
                    return s.name, r

                results = await asyncio.gather(
                    *[_run_parallel(s, current_context) for s in parallel_steps],
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
                    tools=step.tools, budget=budget, context=ctx,
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
            else:
                raise ValueError(
                    f"Unknown provider '{name}'. "
                    f"Supported: {list(factory_map.keys())}"
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
        if self._response_format is not None and isinstance(clean_output, str):
            try:
                parsed_json = _json.loads(clean_output)
                parsed_model = self._response_format.model_validate(parsed_json)
                # Backward compat: output holds the validated model too
                clean_output = parsed_model
            except (_json.JSONDecodeError, Exception) as parse_error:
                if self._on_parse_error is not None:
                    try:
                        if asyncio.iscoroutinefunction(self._on_parse_error):
                            fixed = await self._on_parse_error(clean_output, parse_error)
                        else:
                            fixed = self._on_parse_error(clean_output, parse_error)
                        if fixed is not None:
                            return RunResult(
                                output=fixed,
                                parsed=fixed,
                                success=True,
                                steps=self.state.current_step,
                                tokens_used=self.state.tokens_used,
                                cost_usd=self.state.cost_usd,
                                run_id=self.state.run_id,
                            )
                    except Exception:
                        pass
                return RunResult(
                    output=clean_output,
                    parsed=None,
                    success=False,
                    steps=self.state.current_step,
                    tokens_used=self.state.tokens_used,
                    cost_usd=self.state.cost_usd,
                    run_id=self.state.run_id,
                )

        return RunResult(
            output=clean_output,
            parsed=parsed_model,
            success=self.state.status.value == "completed",
            steps=self.state.current_step,
            tokens_used=self.state.tokens_used,
            cost_usd=self.state.cost_usd,
            run_id=self.state.run_id,
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


class ChatResponse(BaseModel):
    """Response from a single chat turn."""

    content: str = ""
    tool_calls_made: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0


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
    ) -> None:
        self._runtime = runtime
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._max_turns = max_turns_per_message
        self._budget_config = budget or runtime._budget_policy
        self._input_handler = input_handler
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

    async def send(self, message: str) -> ChatResponse:
        """Send a message and get the agent's response.

        The agent may use tools before responding. Each ``send()`` allows
        up to ``max_turns_per_message`` agent turns (tool calls count as
        turns).

        Args:
            message: The user's message text.

        Returns:
            ChatResponse with the agent's reply and usage metrics.
        """
        import json

        from arcana.contracts.llm import (
            LLMRequest,
            Message,
            MessageRole,
        )
        from arcana.contracts.tool import ToolCall

        # 1. Append user message to history
        self._messages.append(Message(role=MessageRole.USER, content=message))
        self._turn_count += 1

        model_config = self._runtime._resolve_model_config()

        # Resolve tools
        tool_defs: list[dict[str, object]] | None = None
        if self._runtime._tool_gateway:
            tool_defs = self._runtime._tool_gateway.registry.to_openai_tools() or None

        # Context management
        from arcana.context.builder import WorkingSetBuilder
        from arcana.contracts.context import TokenBudget

        context_builder = WorkingSetBuilder(
            identity=self._system_prompt,
            token_budget=TokenBudget(total_window=128_000),
            goal=message,
            gateway=self._runtime._gateway,
        )

        turn_tokens = 0
        turn_cost = 0.0
        tool_calls_made = 0
        assistant_text = ""

        # 2. Agent conversation loop (tool calls may require multiple turns)
        for _turn in range(self._max_turns):
            # Budget check
            self._budget_tracker.check_budget()

            # Context compression
            from arcana.context.builder import estimate_tokens

            tool_token_cost = 0
            if tool_defs:
                tool_token_cost = sum(
                    estimate_tokens(json.dumps(t)) for t in tool_defs
                )

            curated = await context_builder.abuild_conversation_context(
                self._messages,
                tool_token_estimate=tool_token_cost,
                turn=_turn,
            )

            # LLM call
            request = LLMRequest(
                messages=curated,
                tools=tool_defs,
            )

            # Use generate() for send() — reliable usage tracking.
            # stream() method handles streaming separately.
            response = await self._runtime._gateway.generate(
                request=request, config=model_config,
            )

            # Track usage
            if response.usage:
                turn_tokens += response.usage.total_tokens
                turn_cost += response.usage.cost_estimate
                self._budget_tracker.add_usage(response.usage)

            # Trace
            if self._runtime._trace_writer:
                from arcana.contracts.trace import EventType, TraceEvent

                self._runtime._trace_writer.write(TraceEvent(
                    run_id=self._session_id,
                    event_type=EventType.TURN,
                    metadata={
                        "chat_turn": self._turn_count,
                        "agent_turn": _turn,
                        "content_length": len(response.content or ""),
                        "tool_calls": len(response.tool_calls or []),
                        "tokens": response.usage.total_tokens if response.usage else 0,
                    },
                ))

            # Handle tool calls
            if response.tool_calls:
                tool_calls_made += len(response.tool_calls)

                # Append assistant message with tool_calls to history
                self._messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                # Execute tools
                if self._runtime._tool_gateway:
                    gateway_calls: list[ToolCall] = []
                    for tc in response.tool_calls:
                        try:
                            args = json.loads(tc.arguments) if tc.arguments else {}
                        except json.JSONDecodeError:
                            args = {"_raw": tc.arguments}
                        gateway_calls.append(
                            ToolCall(id=tc.id, name=tc.name, arguments=args)
                        )
                    results = await self._runtime._tool_gateway.call_many_concurrent(
                        gateway_calls
                    )

                    # Append tool result messages
                    for result in results:
                        self._messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=result.output_str,
                                tool_call_id=result.tool_call_id,
                            )
                        )
                else:
                    # No tool gateway -- return synthetic error results
                    for tc in response.tool_calls:
                        self._messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=f"Tool '{tc.name}' cannot be executed: no tools registered.",
                                tool_call_id=tc.id,
                            )
                        )

                # Continue the loop to let the LLM process tool results
                continue

            # No tool calls -- the LLM produced a text response
            assistant_text = (response.content or "").strip()
            self._messages.append(
                Message(role=MessageRole.ASSISTANT, content=assistant_text)
            )
            break

        # Update cumulative totals
        self._total_tokens += turn_tokens
        self._total_cost_usd += turn_cost

        return ChatResponse(
            content=assistant_text,
            tool_calls_made=tool_calls_made,
            tokens_used=turn_tokens,
            cost_usd=turn_cost,
        )

    async def stream(self, message: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream version of send(). Yields events including the response.

        Yields ``StreamEvent`` objects for LLM chunks, tool results, and
        the final response. After this generator is exhausted the message
        history is updated just like ``send()``.
        """
        import json

        from arcana.contracts.llm import (
            LLMRequest,
            LLMResponse,
            Message,
            MessageRole,
            ToolCallRequest,
        )
        from arcana.contracts.streaming import StreamEventType
        from arcana.contracts.tool import ToolCall

        self._messages.append(Message(role=MessageRole.USER, content=message))
        self._turn_count += 1

        model_config = self._runtime._resolve_model_config()

        tool_defs: list[dict[str, object]] | None = None
        if self._runtime._tool_gateway:
            tool_defs = self._runtime._tool_gateway.registry.to_openai_tools() or None

        from arcana.context.builder import WorkingSetBuilder, estimate_tokens
        from arcana.contracts.context import TokenBudget

        context_builder = WorkingSetBuilder(
            identity=self._system_prompt,
            token_budget=TokenBudget(total_window=128_000),
            goal=message,
            gateway=self._runtime._gateway,
        )

        turn_tokens = 0
        turn_cost = 0.0
        tool_calls_made = 0
        assistant_text = ""

        yield StreamEvent(
            event_type=StreamEventType.RUN_START,
            run_id=self._session_id,
            content=message,
        )

        for _turn in range(self._max_turns):
            self._budget_tracker.check_budget()

            tool_token_cost = 0
            if tool_defs:
                tool_token_cost = sum(
                    estimate_tokens(json.dumps(t)) for t in tool_defs
                )

            curated = await context_builder.abuild_conversation_context(
                self._messages,
                tool_token_estimate=tool_token_cost,
                turn=_turn,
            )

            request = LLMRequest(messages=curated, tools=tool_defs)

            response: LLMResponse
            try:
                from arcana.contracts.llm import TokenUsage

                text_parts: list[str] = []
                tc_names: dict[str, str] = {}
                tc_args: dict[str, list[str]] = {}
                stream_usage: TokenUsage | None = None
                stream_finish = "stop"
                stream_model = model_config.model_id

                async for chunk in self._runtime._gateway.stream(
                    request=request, config=model_config,
                ):
                    if chunk.type == "text_delta" and chunk.text:
                        text_parts.append(chunk.text)
                        yield StreamEvent(
                            event_type=StreamEventType.LLM_CHUNK,
                            run_id=self._session_id,
                            content=chunk.text,
                        )
                    elif chunk.type == "tool_call_delta" and chunk.tool_call_id:
                        if chunk.tool_call_id not in tc_names:
                            tc_names[chunk.tool_call_id] = chunk.tool_name or ""
                        if chunk.tool_name:
                            tc_names[chunk.tool_call_id] = chunk.tool_name
                        if chunk.arguments_delta:
                            tc_args.setdefault(
                                chunk.tool_call_id, [],
                            ).append(chunk.arguments_delta)
                    elif chunk.type == "usage" and chunk.usage:
                        stream_usage = chunk.usage
                    elif chunk.type == "done":
                        if chunk.metadata:
                            stream_finish = chunk.metadata.get(
                                "finish_reason", stream_finish,
                            )
                            stream_model = chunk.metadata.get(
                                "model", stream_model,
                            )
                        if chunk.usage:
                            stream_usage = chunk.usage

                full_text = "".join(text_parts) if text_parts else None
                tool_calls_list = [
                    ToolCallRequest(
                        id=tc_id,
                        name=tc_names.get(tc_id, ""),
                        arguments="".join(tc_args.get(tc_id, [])),
                    )
                    for tc_id in tc_names
                ] or None
                response = LLMResponse(
                    content=full_text,
                    tool_calls=tool_calls_list,
                    usage=stream_usage or TokenUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    ),
                    model=stream_model,
                    finish_reason=stream_finish,
                )
            except (AttributeError, TypeError, NotImplementedError):
                response = await self._runtime._gateway.generate(
                    request=request, config=model_config,
                )

            if response.usage:
                turn_tokens += response.usage.total_tokens
                turn_cost += response.usage.cost_estimate
                self._budget_tracker.add_usage(response.usage)

            if response.tool_calls:
                tool_calls_made += len(response.tool_calls)

                self._messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                if self._runtime._tool_gateway:
                    gateway_calls: list[ToolCall] = []
                    for tc in response.tool_calls:
                        try:
                            args = json.loads(tc.arguments) if tc.arguments else {}
                        except json.JSONDecodeError:
                            args = {"_raw": tc.arguments}
                        gateway_calls.append(
                            ToolCall(id=tc.id, name=tc.name, arguments=args)
                        )
                    results = await self._runtime._tool_gateway.call_many_concurrent(
                        gateway_calls
                    )

                    for result in results:
                        self._messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=result.output_str,
                                tool_call_id=result.tool_call_id,
                            )
                        )
                        yield StreamEvent(
                            event_type=StreamEventType.TOOL_RESULT,
                            run_id=self._session_id,
                            content=result.output_str,
                            tool_result_data=result.model_dump(),
                        )
                else:
                    for tc in response.tool_calls:
                        err = f"Tool '{tc.name}' cannot be executed: no tools registered."
                        self._messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=err,
                                tool_call_id=tc.id,
                            )
                        )
                        yield StreamEvent(
                            event_type=StreamEventType.TOOL_RESULT,
                            run_id=self._session_id,
                            content=err,
                        )
                continue

            assistant_text = (response.content or "").strip()
            self._messages.append(
                Message(role=MessageRole.ASSISTANT, content=assistant_text)
            )
            break

        self._total_tokens += turn_tokens
        self._total_cost_usd += turn_cost

        yield StreamEvent(
            event_type=StreamEventType.RUN_COMPLETE,
            run_id=self._session_id,
            content=assistant_text,
            tokens_used=turn_tokens,
            cost_usd=turn_cost,
        )

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
