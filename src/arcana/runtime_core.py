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

    # Advanced: manual session control
    async with runtime.session() as s:
        response = await s.llm("What should I search for?")
        data = await s.tool("web_search", query="quantum computing")
        print(s.budget.remaining_usd)
"""

from __future__ import annotations

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
    - Provider connections (gateway registry)
    - Tool registry + gateway
    - Trace backend
    - Default budget policy
    - Default engine config

    Args:
        namespace: Optional namespace for tenant isolation. When set,
            memory and trace are partitioned so that multiple Runtimes
            sharing the same backing stores don't see each other's data.
            When ``None`` (the default), behavior is unchanged.
    """

    def __init__(
        self,
        *,
        providers: dict[str, str] | None = None,  # {"deepseek": "sk-xxx"}
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
    ) -> RunResult:
        """
        Run a task to completion.

        Args:
            goal: What to accomplish
            engine: "conversation" (V2, default) or "adaptive" (V1)
            max_turns: Override default max turns
            budget: Override default budget for this run
            tools: Additional tools for this run only
        """
        # Auto-connect MCP on first run
        if self._mcp_configs and not self._mcp_client:
            await self.connect_mcp()

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
        )
        result = await session.run(goal)

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
        )
        try:
            yield s
        finally:
            pass  # Session cleanup if needed

    async def team(
        self,
        goal: str,
        agents: list[AgentConfig],
        *,
        max_rounds: int = 3,
        budget: Budget | None = None,
    ) -> TeamResult:
        """
        Run a team of agents on a shared goal.

        Each agent gets its own system prompt and takes turns responding
        to a shared conversation. Runtime manages: resource isolation,
        communication, budget, trace. Runtime does NOT: decide strategy,
        assign roles, parse outputs for orchestration decisions.

        Args:
            goal: The shared objective
            agents: List of agent configurations
            max_rounds: Maximum conversation rounds (each agent speaks once per round)
            budget: Budget for the entire team run
        """
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

        # Shared conversation history -- all agents see what others said
        shared_messages: list[Message] = [
            Message(role=MessageRole.USER, content=f"Team goal: {goal}"),
        ]

        conversation_log: list[dict[str, Any]] = []
        agent_outputs: dict[str, str] = {}
        total_tokens = 0

        for round_num in range(max_rounds):
            for agent_config in agents:
                # Budget check before each agent's turn
                budget_tracker.check_budget()

                # Build this agent's messages:
                # System prompt (agent identity) + shared conversation
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

                # Add to shared conversation
                agent_text = response.content or ""
                shared_messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=f"[{agent_config.name}]: {agent_text}",
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
    def providers(self) -> list[str]:
        """List registered provider names."""
        return list(self._gateway.list_providers())

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

    def _setup_providers(self, providers: dict[str, str]) -> Any:
        """Setup provider registry from {name: api_key} dict."""
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

        for name, api_key in providers.items():
            # Resolve from env if empty
            if not api_key:
                env_var = f"{name.upper()}_API_KEY"
                api_key = os.environ.get(env_var, "")

            if name in factory_map:
                gateway.register(name, factory_map[name](api_key))
            else:
                raise ValueError(
                    f"Unknown provider '{name}'. "
                    f"Supported: {list(factory_map.keys())}"
                )

        if providers:
            gateway.set_default(self._config.default_provider)

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
    ) -> Session:
        """Create a new session with per-run resources."""
        return Session(
            runtime=self,
            engine=engine,
            max_turns=max_turns or self._config.max_turns,
            budget=budget or self._budget_policy,
            extra_tools=extra_tools,
            memory_context=memory_context,
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
    ) -> None:
        self._runtime = runtime
        self._engine = engine
        self._max_turns = max_turns
        self._budget_config = budget or Budget()
        self._extra_tools = extra_tools
        self._memory_context = memory_context

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
        # Merge tools: runtime tools + session extra tools
        tool_gateway = self._runtime._tool_gateway
        # TODO: merge extra_tools if provided

        model_config = self._runtime._resolve_model_config()

        if self._engine == "conversation":
            from arcana.routing.classifier import RuleBasedClassifier
            from arcana.runtime.conversation import ConversationAgent

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
            if self._memory_context:
                agent_kwargs["memory_context"] = self._memory_context

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

        return RunResult(
            output=clean_output,
            success=self.state.status.value == "completed",
            steps=self.state.current_step,
            tokens_used=self.state.tokens_used,
            cost_usd=self.state.cost_usd,
            run_id=self.state.run_id,
        )


class RunResult(BaseModel):
    """Result of a runtime run."""

    output: Any = None
    success: bool = False
    steps: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    run_id: str = ""
