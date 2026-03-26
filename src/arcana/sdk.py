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
import base64
import inspect
import mimetypes
import os
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from arcana.contracts.llm import ContentBlock
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


class Tool:
    """Non-decorator tool registration.

    Use this when you want to register tools without importing arcana
    at module level, or when the tool function is defined elsewhere.

    Example::

        from arcana import Tool

        def my_search(query: str) -> str:
            \"\"\"Search the web.\"\"\"
            return requests.get(f"https://api.example.com/search?q={query}").text

        search_tool = Tool(fn=my_search, when_to_use="Search the web for information")
        runtime = Runtime(tools=[search_tool])
    """

    def __init__(
        self,
        fn: Callable,  # type: ignore[type-arg]
        *,
        name: str | None = None,
        description: str | None = None,
        when_to_use: str | None = None,
        what_to_expect: str | None = None,
        failure_meaning: str | None = None,
        side_effect: str = "read",
        requires_confirmation: bool = False,
    ) -> None:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or f"Tool: {tool_name}"

        sig = inspect.signature(fn)
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

        self._fn = fn
        self._spec = spec
        # Make it look like a decorated function for the registry
        fn._arcana_tool_spec = spec  # type: ignore[attr-defined]
        fn._arcana_tool_func = fn  # type: ignore[attr-defined]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


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
    parsed: Any = None
    success: bool = False
    steps: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    run_id: str = ""


# --- Image helpers ---


_MIME_FALLBACK = "image/png"


def _detect_mime(path: str) -> str:
    """Detect MIME type from a file path, falling back to image/png."""
    mime, _ = mimetypes.guess_type(path)
    return mime or _MIME_FALLBACK


def build_content_blocks(
    text: str,
    images: list[str] | None = None,
) -> str | list[ContentBlock]:
    """Build message content from text and optional images.

    When *images* is empty or ``None``, returns plain ``str`` for maximum
    backward compatibility.  Otherwise returns a ``list[ContentBlock]``
    with one text block followed by one ``image_url`` block per image.

    Each image string can be:

    * An HTTP(S) URL -- used directly.
    * A local file path -- read, base64-encoded, MIME-type detected.
    * Raw data (anything else) -- assumed to be a ``data:`` URI or base64.

    The output uses the **OpenAI-compatible** ``image_url`` block format
    which is the canonical multimodal format in Arcana.  Provider-specific
    gateways (e.g. Anthropic) convert internally.
    """
    if not images:
        return text

    blocks: list[ContentBlock] = [ContentBlock(type="text", text=text)]

    for img in images:
        if img.startswith(("http://", "https://")):
            blocks.append(
                ContentBlock(type="image_url", image_url={"url": img})
            )
        elif os.path.isfile(img):
            mime = _detect_mime(img)
            with open(img, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            data_uri = f"data:{mime};base64,{b64}"
            blocks.append(
                ContentBlock(type="image_url", image_url={"url": data_uri})
            )
        else:
            # Assume it's already a data URI or raw base64
            blocks.append(
                ContentBlock(type="image_url", image_url={"url": img})
            )

    return blocks


# --- Run Function ---


async def run(
    goal: str,
    *,
    images: list[str] | None = None,
    tools: list[Callable] | None = None,  # type: ignore[type-arg]
    provider: str = "deepseek",
    model: str | None = None,
    api_key: str | None = None,
    max_turns: int = 20,
    max_cost_usd: float = 1.0,
    auto_route: bool = True,
    engine: str = "conversation",
    stream: bool = False,
    response_format: type[BaseModel] | None = None,
    input_handler: Callable | None = None,  # type: ignore[type-arg]
    system: str | None = None,
    context: dict | str | None = None,
    on_parse_error: Callable | None = None,  # type: ignore[type-arg]
) -> RunResult:
    """
    Run an agent to accomplish a goal.

    This is the simplest way to use Arcana. It handles provider setup,
    tool registration, intent routing, and execution automatically.

    Quick run -- creates a temporary Runtime. For scripts and demos.

    Args:
        goal: What you want the agent to accomplish
        images: Optional list of image inputs. Each can be a URL, local file
            path, or ``data:`` URI / raw base64 string.
        tools: Optional list of @arcana.tool decorated functions
        provider: LLM provider name (default: "deepseek")
        model: Model ID (auto-selected if None)
        api_key: API key for the provider. If None, reads from environment variable.
        max_turns: Maximum execution turns (default: 20)
        max_cost_usd: Maximum cost in USD (default: 1.0)
        auto_route: Enable intent routing (default: True)
        engine: Execution engine - "conversation" (V2, default) or "adaptive" (V1)
        stream: Enable streaming output (reserved for future use)
        response_format: Pydantic BaseModel class for structured output.
            When provided, the LLM is instructed to return JSON matching
            the model's schema, and ``result.output`` will be a validated
            instance of the model rather than a plain string. Tools and
            structured output can be used together — the agent uses tools
            during reasoning and returns structured output on the final turn.
        input_handler: Optional callback for the ask_user built-in tool.
            Can be sync or async. When None, the LLM receives a fallback
            message and proceeds with best judgment.
        system: System prompt defining the agent's role/persona for this
            run. When None, the engine's default is used.
        context: Additional context for the agent. A dict is serialized
            as JSON; a string is used as-is. Injected as a ``<context>``
            block so the agent can reference prior outputs or external data.
        on_parse_error: Optional callback invoked when the LLM returns
            text that cannot be parsed into the ``response_format`` model.
            Receives ``(raw_string, error)`` where *error* is a
            ``json.JSONDecodeError`` or ``pydantic.ValidationError``.
            Return a fixed ``BaseModel`` instance to recover, or ``None``
            to preserve the failure.  Supports async.

            Does NOT fire for provider-level rejections (e.g. the provider
            does not support ``json_schema`` mode) -- those surface as
            ``ProviderError`` and are handled by provider capability
            detection / auto-downgrade.

    Returns:
        RunResult with output and execution metadata

    Examples:
        # Simplest usage
        result = await arcana.run("What is 2+2?", api_key="sk-xxx")

        # With an image
        result = await arcana.run(
            "Describe this image",
            images=["https://example.com/photo.jpg"],
            provider="openai",
            api_key="sk-proj-xxx",
        )

        # With tools
        @arcana.tool(when_to_use="For math")
        def calc(expression: str) -> str:
            return str(eval(expression))

        result = await arcana.run("15*37+89?", tools=[calc], api_key="sk-xxx")

        # With OpenAI
        result = await arcana.run("Hello", provider="openai", api_key="sk-proj-xxx")
    """
    from arcana.runtime_core import Budget as RuntimeBudget
    from arcana.runtime_core import Runtime, RuntimeConfig

    providers = {provider: api_key or ""}
    config = RuntimeConfig(default_provider=provider, default_model=model)
    rt = Runtime(
        providers=providers,
        tools=tools,
        budget=RuntimeBudget(max_cost_usd=max_cost_usd),
        config=config,
    )
    result = await rt.run(
        goal,
        engine=engine,
        max_turns=max_turns,
        images=images,
        response_format=response_format,
        input_handler=input_handler,
        system=system,
        context=context,
        on_parse_error=on_parse_error,
    )

    # Convert to sdk.RunResult (keep backward compat)
    return RunResult(
        output=result.output,
        parsed=result.parsed,
        success=result.success,
        steps=result.steps,
        tokens_used=result.tokens_used,
        cost_usd=result.cost_usd,
        run_id=result.run_id,
    )


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
