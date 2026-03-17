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

import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from arcana.contracts.llm import ModelConfig
from arcana.contracts.state import AgentState


class Budget(BaseModel):
    """Budget configuration for a Runtime."""

    max_cost_usd: float = 10.0
    max_tokens: int = 500_000


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
    """

    def __init__(
        self,
        *,
        providers: dict[str, str] | None = None,  # {"deepseek": "sk-xxx"}
        tools: list[Callable] | None = None,  # type: ignore[type-arg]
        budget: Budget | None = None,
        trace: bool = False,
        config: RuntimeConfig | None = None,
    ) -> None:
        self._config = config or RuntimeConfig()
        self._budget_policy = budget or Budget()

        # Setup providers (long-lived)
        self._gateway = self._setup_providers(providers or {})

        # Setup tools (long-lived)
        self._tool_registry = None
        self._tool_gateway = None
        if tools:
            self._tool_registry, self._tool_gateway = self._setup_tools(tools)

        # Setup trace (long-lived)
        self._trace_writer = None
        if trace:
            from arcana.trace.writer import TraceWriter

            self._trace_writer = TraceWriter(output_dir=self._config.trace_dir)

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
        session = self._create_session(
            engine=engine,
            max_turns=max_turns,
            budget=budget,
            extra_tools=tools,
        )
        return await session.run(goal)

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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def providers(self) -> list[str]:
        """List registered provider names."""
        return self._gateway.list_providers()

    @property
    def tools(self) -> list[str]:
        """List registered tool names."""
        if self._tool_registry:
            return self._tool_registry.list_tools()
        return []

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
    ) -> Session:
        """Create a new session with per-run resources."""
        return Session(
            runtime=self,
            engine=engine,
            max_turns=max_turns or self._config.max_turns,
            budget=budget or self._budget_policy,
            extra_tools=extra_tools,
        )

    def _resolve_model_config(self) -> ModelConfig:
        """Get default ModelConfig."""
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
                model_id = "deepseek-chat"
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
    ) -> None:
        self._runtime = runtime
        self._engine = engine
        self._max_turns = max_turns
        self._budget_config = budget or Budget()
        self._extra_tools = extra_tools

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
            from arcana.runtime.conversation import ConversationAgent

            agent = ConversationAgent(
                gateway=self._runtime._gateway,
                model_config=model_config,
                tool_gateway=tool_gateway,
                budget_tracker=self.budget,
                trace_writer=self._runtime._trace_writer,
                max_turns=self._max_turns,
            )
            self.state = await agent.run(goal)

        else:
            # V1: Agent + AdaptivePolicy
            from arcana.contracts.runtime import RuntimeConfig as V1Config
            from arcana.runtime.agent import Agent
            from arcana.runtime.policies.adaptive import AdaptivePolicy
            from arcana.runtime.reducers.default import DefaultReducer

            agent = Agent(
                policy=AdaptivePolicy(),
                reducer=DefaultReducer(),
                gateway=self._runtime._gateway,
                config=V1Config(max_steps=self._max_turns),
                tool_gateway=tool_gateway,
                budget_tracker=self.budget,
                trace_writer=self._runtime._trace_writer,
            )
            self.state = await agent.run(goal)

        return RunResult(
            output=self.state.working_memory.get(
                "answer", self.state.working_memory.get("result", "")
            ),
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
