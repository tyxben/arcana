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
    max_steps: int = 20,
    max_cost_usd: float = 1.0,
    auto_route: bool = True,
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
        max_steps: Maximum execution steps
        max_cost_usd: Maximum cost in USD
        auto_route: Enable intent routing (default: True)
        stream: Enable streaming output (reserved for future use)

    Returns:
        RunResult with output and execution metadata
    """
    from arcana.contracts.runtime import RuntimeConfig
    from arcana.gateway.budget import BudgetTracker
    from arcana.routing.classifier import HybridClassifier
    from arcana.runtime.agent import Agent
    from arcana.runtime.policies.adaptive import AdaptivePolicy
    from arcana.runtime.reducers.default import DefaultReducer

    # Setup gateway (lazy -- only import/create what's needed)
    gateway = _setup_gateway(provider, model)

    # Setup tools
    tool_gateway = None
    if tools:
        tool_gateway = _setup_tools(tools)

    # Setup classifier
    classifier = HybridClassifier(gateway=gateway) if auto_route else None

    # Setup budget tracker
    budget_tracker = BudgetTracker(
        max_cost_usd=max_cost_usd,
        max_tokens=max_steps * 10000,  # rough estimate per step
    )

    # Create agent
    agent = Agent(
        policy=AdaptivePolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=max_steps),
        tool_gateway=tool_gateway,
        intent_classifier=classifier,
        auto_route=auto_route,
        budget_tracker=budget_tracker,
    )

    # Run
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


def _setup_gateway(provider: str, model: str | None) -> Any:
    """Lazy setup of model gateway."""
    import os

    from arcana.gateway.providers.openai_compatible import (
        create_deepseek_provider,
        create_gemini_provider,
        create_glm_provider,
        create_kimi_provider,
        create_minimax_provider,
        create_ollama_provider,
    )
    from arcana.gateway.registry import ModelGatewayRegistry

    # Validate API key for cloud providers
    api_key_env: dict[str, str] = {
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "kimi": "KIMI_API_KEY",
        "glm": "GLM_API_KEY",
        "minimax": "MINIMAX_API_KEY",
    }

    if provider in api_key_env:
        env_var = api_key_env[provider]
        key = os.environ.get(env_var, "")
        if not key:
            msg = (
                f"API key not found. Set the {env_var} environment variable.\n"
                f"Example: export {env_var}=sk-xxx"
            )
            raise ValueError(msg)

    gateway = ModelGatewayRegistry()

    factories: dict[str, Callable] = {  # type: ignore[type-arg]
        "deepseek": lambda: create_deepseek_provider(os.environ.get("DEEPSEEK_API_KEY", "")),
        "gemini": lambda: create_gemini_provider(os.environ.get("GEMINI_API_KEY", "")),
        "kimi": lambda: create_kimi_provider(os.environ.get("KIMI_API_KEY", "")),
        "glm": lambda: create_glm_provider(os.environ.get("GLM_API_KEY", "")),
        "minimax": lambda: create_minimax_provider(os.environ.get("MINIMAX_API_KEY", "")),
        "ollama": lambda: create_ollama_provider(),
    }

    if provider in factories:
        gateway.register(provider, factories[provider]())
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
