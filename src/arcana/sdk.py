"""
Arcana SDK -- Public API for building LLM agents.

Quick start:
    import arcana
    result = await arcana.run("Research quantum computing trends")
    print(result.output)

With tools:
    @arcana.tool(
        when_to_use="When you need to search the web for information",
        what_to_expect="Returns search results that may need filtering",
    )
    async def web_search(query: str) -> str:
        ...

    result = await arcana.run("Find latest AI news", tools=[web_search])
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from arcana.contracts.tool import (
    ErrorType,
    SideEffect,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)

# --- Tool Decorator ---


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    when_to_use: str | None = None,
    what_to_expect: str | None = None,
    failure_meaning: str | None = None,
    side_effect: str = "read",
    requires_confirmation: bool = False,
) -> Callable:  # type: ignore[type-arg]
    """
    Decorator to register a function as an Arcana tool.

    Usage:
        @arcana.tool(when_to_use="When you need current information")
        async def search(query: str) -> str:
            return await do_search(query)
    """

    def decorator(func: Callable) -> Callable:  # type: ignore[type-arg]
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"

        # Infer input_schema from function signature
        sig = inspect.signature(func)
        input_schema = _signature_to_json_schema(sig)

        spec = ToolSpec(
            name=tool_name,
            description=tool_desc,
            input_schema=input_schema,
            when_to_use=when_to_use,
            what_to_expect=what_to_expect,
            failure_meaning=failure_meaning,
            side_effect=SideEffect(side_effect),
            requires_confirmation=requires_confirmation,
        )

        # Attach spec to function for later registration
        func._arcana_tool_spec = spec  # type: ignore[attr-defined]
        func._arcana_tool_func = func  # type: ignore[attr-defined]
        return func

    return decorator


def _signature_to_json_schema(sig: inspect.Signature) -> dict[str, Any]:
    """Convert function signature to JSON Schema. Pure function."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    type_map: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = param.annotation
        json_type = type_map.get(annotation, "string")
        properties[param_name] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


# --- Result Model ---


class RunResult(BaseModel):
    """Result of an arcana.run() call."""

    output: Any = None
    success: bool = False
    steps: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    run_id: str = ""


# --- Run Function ---


async def run(
    goal: str,
    *,
    tools: list[Callable] | None = None,  # type: ignore[type-arg]
    provider: str = "deepseek",
    model: str | None = None,
    api_key: str | None = None,
    max_turns: int = 20,
    max_cost_usd: float = 1.0,
    auto_route: bool = True,
    engine: str = "conversation",
    stream: bool = False,
) -> RunResult:
    """
    Run an agent to accomplish a goal.

    This is the simplest way to use Arcana. It handles provider setup,
    tool registration, intent routing, and execution automatically.

    Args:
        goal: What you want the agent to accomplish
        tools: Optional list of @arcana.tool decorated functions
        provider: LLM provider name (default: "deepseek")
        model: Model ID (auto-selected if None)
        api_key: API key for the provider. If None, reads from environment variable.
        max_turns: Maximum execution turns (default: 20)
        max_cost_usd: Maximum cost in USD (default: 1.0)
        auto_route: Enable intent routing (default: True)
        engine: Execution engine - "conversation" (V2, default) or "adaptive" (V1)
        stream: Enable streaming output (reserved for future use)

    Returns:
        RunResult with output and execution metadata

    Examples:
        # Simplest usage
        result = await arcana.run("What is 2+2?", api_key="sk-xxx")

        # With tools
        @arcana.tool(when_to_use="For math")
        def calc(expression: str) -> str:
            return str(eval(expression))

        result = await arcana.run("15*37+89?", tools=[calc], api_key="sk-xxx")

        # With OpenAI
        result = await arcana.run("Hello", provider="openai", api_key="sk-proj-xxx")
    """
    from arcana.contracts.llm import ModelConfig

    # Setup gateway
    gateway = _setup_gateway(provider, model, api_key)

    # Setup tools
    tool_gateway = None
    if tools:
        tool_gateway = _setup_tools(tools)

    if engine == "conversation":
        # V2: ConversationAgent (recommended)
        from arcana.runtime.conversation import ConversationAgent

        # Resolve model_id
        model_id = model
        if not model_id:
            p = gateway.get(provider)
            if p and hasattr(p, "default_model") and isinstance(p.default_model, str):
                model_id = p.default_model
            else:
                model_id = "deepseek-chat"

        agent = ConversationAgent(
            gateway=gateway,
            model_config=ModelConfig(provider=provider, model_id=model_id),
            tool_gateway=tool_gateway,
            max_turns=max_turns,
        )

        state = await agent.run(goal)

    else:
        # V1: Agent + AdaptivePolicy (legacy compatible)
        from arcana.contracts.runtime import RuntimeConfig
        from arcana.gateway.budget import BudgetTracker
        from arcana.routing.classifier import HybridClassifier
        from arcana.runtime.agent import Agent
        from arcana.runtime.policies.adaptive import AdaptivePolicy
        from arcana.runtime.reducers.default import DefaultReducer

        classifier = HybridClassifier(gateway=gateway) if auto_route else None
        budget_tracker = BudgetTracker(
            max_cost_usd=max_cost_usd,
            max_tokens=max_turns * 10000,
        )

        agent = Agent(
            policy=AdaptivePolicy(),
            reducer=DefaultReducer(),
            gateway=gateway,
            config=RuntimeConfig(max_steps=max_turns),
            tool_gateway=tool_gateway,
            intent_classifier=classifier,
            auto_route=auto_route,
            budget_tracker=budget_tracker,
        )

        state = await agent.run(goal)

    return RunResult(
        output=state.working_memory.get("answer", state.working_memory.get("result", "")),
        success=state.status.value == "completed",
        steps=state.current_step,
        tokens_used=state.tokens_used,
        cost_usd=state.cost_usd,
        run_id=state.run_id,
    )


# --- Internal Helpers ---


def _setup_gateway(provider: str, model: str | None, api_key: str | None = None) -> Any:
    """Lazy setup of model gateway.

    Args:
        provider: Provider name.
        model: Model ID (unused here, resolved later).
        api_key: Explicit API key. Falls back to environment variable if None.
    """
    import os

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

    # Resolve API key: explicit > env var
    env_var_map: dict[str, str] = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "kimi": "KIMI_API_KEY",
        "glm": "GLM_API_KEY",
        "minimax": "MINIMAX_API_KEY",
    }

    resolved_key = api_key
    if not resolved_key and provider in env_var_map:
        resolved_key = os.environ.get(env_var_map[provider], "")

    if not resolved_key and provider != "ollama":
        env_var = env_var_map.get(provider, f"{provider.upper()}_API_KEY")
        msg = (
            f"API key required for provider '{provider}'.\n"
            f"Either pass api_key='sk-xxx' or set {env_var} environment variable."
        )
        raise ValueError(msg)

    gateway = ModelGatewayRegistry()

    # Factory for providers with explicit key
    if provider == "openai":
        gateway.register(provider, OpenAICompatibleProvider(
            provider_name="openai",
            api_key=resolved_key or "",
            base_url="https://api.openai.com/v1",
            default_model="gpt-4o-mini",
        ))
    elif provider == "anthropic":
        try:
            from arcana.gateway.providers.anthropic import AnthropicProvider
            gateway.register(provider, AnthropicProvider(api_key=resolved_key or ""))
        except ImportError:
            msg = "anthropic SDK not installed. Install with: pip install arcana-agent[anthropic]"
            raise ImportError(msg) from None
    elif provider == "deepseek":
        gateway.register(provider, create_deepseek_provider(resolved_key or ""))
    elif provider == "gemini":
        gateway.register(provider, create_gemini_provider(resolved_key or ""))
    elif provider == "kimi":
        gateway.register(provider, create_kimi_provider(resolved_key or ""))
    elif provider == "glm":
        gateway.register(provider, create_glm_provider(resolved_key or ""))
    elif provider == "minimax":
        gateway.register(provider, create_minimax_provider(resolved_key or ""))
    elif provider == "ollama":
        gateway.register(provider, create_ollama_provider())
    else:
        msg = f"Unknown provider '{provider}'. Supported: deepseek, openai, anthropic, gemini, kimi, glm, minimax, ollama"
        raise ValueError(msg)

    gateway.set_default(provider)
    return gateway


def _setup_tools(tools: list[Callable]) -> Any:  # type: ignore[type-arg]
    """Setup tool gateway from decorated functions."""
    from arcana.tool_gateway.gateway import ToolGateway
    from arcana.tool_gateway.registry import ToolRegistry

    registry = ToolRegistry()

    for func in tools:
        spec = getattr(func, "_arcana_tool_spec", None)
        if spec is None:
            continue

        # Create a ToolProvider wrapping the function
        provider = _FunctionToolProvider(spec=spec, func=func)
        registry.register(provider)

    return ToolGateway(registry=registry)


class _FunctionToolProvider:
    """Wraps a decorated function as a ToolProvider."""

    def __init__(self, spec: ToolSpec, func: Callable) -> None:  # type: ignore[type-arg]
        self._spec = spec
        self._func = func

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(**call.arguments)
            else:
                result = self._func(**call.arguments)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=True,
                output=result,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=ToolError(error_type=ErrorType.NON_RETRYABLE, message=str(e)),
            )

    async def health_check(self) -> bool:
        return True
