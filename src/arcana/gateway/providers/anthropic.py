"""Native Anthropic Claude provider.

Unlike the OpenAI-compatible providers, this module talks directly to the
Anthropic Messages API so that Anthropic-specific features (extended thinking,
prompt caching, etc.) are first-class citizens.

All conversion logic is implemented as **pure functions** that accept and
return plain dicts / Pydantic models, making them easy to unit-test without
network calls.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from arcana.contracts.llm import (
    AnthropicResponseExt,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    StreamChunk,
    ThinkingBlock,
    TokenUsage,
    ToolCallRequest,
)
from arcana.gateway.base import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)

if TYPE_CHECKING:
    from arcana.contracts.trace import TraceContext
    from arcana.trace.writer import TraceWriter

# ---------------------------------------------------------------------------
# Import guard -- anthropic SDK is optional
# ---------------------------------------------------------------------------

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure conversion functions
# ---------------------------------------------------------------------------


def to_anthropic_request(request: LLMRequest, config: ModelConfig) -> dict[str, Any]:
    """Convert an ``LLMRequest`` + ``ModelConfig`` into Anthropic API params.

    This is a **pure function** with no side-effects, making it trivial to
    test in isolation.
    """

    # -- system message extraction ------------------------------------------
    system_text: str | None = None

    # Prefer the explicit Anthropic extension field
    if request.anthropic and request.anthropic.system:
        system_text = request.anthropic.system

    # Separate non-system messages and extract system from the list
    api_messages: list[dict[str, Any]] = []
    for msg in request.messages:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

        if role == "system":
            # Use the first system message if no explicit one was set
            if system_text is None:
                if isinstance(msg.content, str):
                    system_text = msg.content
                elif isinstance(msg.content, list):
                    # Concatenate text blocks from content list
                    parts = []
                    for block in msg.content:
                        if hasattr(block, "text") and block.text:
                            parts.append(block.text)
                    system_text = "\n".join(parts)
            continue

        api_messages.append(_convert_message(msg, role))

    # -- build params -------------------------------------------------------
    params: dict[str, Any] = {
        "model": config.model_id,
        "max_tokens": config.max_tokens,
        "messages": api_messages,
    }

    if system_text:
        # Prompt caching: convert system to content block list with cache_control
        # on the last block. This tells Anthropic to cache the system prompt prefix,
        # saving ~90% of input tokens on subsequent turns.
        prompt_caching = (
            request.anthropic
            and request.anthropic.prompt_caching is not False
        ) or (request.anthropic is None)
        if prompt_caching:
            params["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            params["system"] = system_text

    # Temperature -- but NOT when extended thinking is enabled
    thinking_enabled = (
        request.anthropic
        and request.anthropic.thinking
        and request.anthropic.thinking.enabled
    )

    if not thinking_enabled:
        params["temperature"] = config.temperature

    # Extended thinking
    if thinking_enabled:
        budget = (
            request.anthropic.thinking.budget_tokens
            if request.anthropic
            and request.anthropic.thinking
            and request.anthropic.thinking.budget_tokens
            else config.max_tokens
        )
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget,
        }

    # Tools
    if request.tools:
        converted_tools = [_convert_tool_def(t) for t in request.tools]
        # Prompt caching: mark the last tool definition with cache_control
        # so that the entire tool schema prefix is cached across turns.
        prompt_caching = (
            request.anthropic
            and request.anthropic.prompt_caching is not False
        ) or (request.anthropic is None)
        if prompt_caching and converted_tools:
            converted_tools[-1]["cache_control"] = {"type": "ephemeral"}
        params["tools"] = converted_tools

    # Seed
    if config.seed is not None:
        params["metadata"] = {"user_id": str(config.seed)}

    # Extra params pass-through
    if config.extra_params:
        params.update(config.extra_params)

    return params


def from_anthropic_response(raw: Any) -> LLMResponse:
    """Convert an Anthropic ``Message`` object into an ``LLMResponse``.

    Pure function -- accepts *any* object with the right attributes so that
    it can be tested with simple mocks.
    """

    content_text: str | None = None
    tool_calls: list[ToolCallRequest] = []
    thinking_blocks: list[ThinkingBlock] = []

    for block in raw.content:
        block_type = getattr(block, "type", None)

        if block_type == "text":
            # Concatenate multiple text blocks (rare but possible)
            if content_text is None:
                content_text = block.text
            else:
                content_text += block.text

        elif block_type == "thinking":
            thinking_blocks.append(ThinkingBlock(thinking=block.thinking))

        elif block_type == "tool_use":
            tool_calls.append(
                ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                )
            )

    # Usage -- including prompt caching metrics when available
    cache_creation = getattr(raw.usage, "cache_creation_input_tokens", None)
    cache_read = getattr(raw.usage, "cache_read_input_tokens", None)

    usage = TokenUsage(
        prompt_tokens=raw.usage.input_tokens,
        completion_tokens=raw.usage.output_tokens,
        total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
        cache_creation_input_tokens=cache_creation if cache_creation else None,
        cache_read_input_tokens=cache_read if cache_read else None,
    )

    # stop_reason mapping
    stop_reason_map: dict[str, str] = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    finish_reason = stop_reason_map.get(raw.stop_reason, raw.stop_reason or "stop")

    # Build response extensions
    anthropic_ext = AnthropicResponseExt(
        thinking_blocks=thinking_blocks if thinking_blocks else None,
        cache_creation_input_tokens=cache_creation if cache_creation else None,
        cache_read_input_tokens=cache_read if cache_read else None,
        stop_reason=raw.stop_reason,
    )

    return LLMResponse(
        content=content_text,
        tool_calls=tool_calls if tool_calls else None,
        usage=usage,
        model=raw.model,
        finish_reason=finish_reason,
        anthropic=anthropic_ext,
    )


def from_anthropic_stream_event(event: Any) -> StreamChunk | None:
    """Convert an Anthropic stream event into a ``StreamChunk`` or ``None``.

    Pure function. Returns ``None`` for events we don't care about
    (e.g. ``message_start``, ``content_block_start``).
    """

    event_type = getattr(event, "type", None)

    # -- content_block_delta ------------------------------------------------
    if event_type == "content_block_delta":
        delta = event.delta
        delta_type = getattr(delta, "type", None)

        if delta_type == "text_delta":
            return StreamChunk(type="text_delta", text=delta.text)

        if delta_type == "thinking_delta":
            return StreamChunk(type="thinking_delta", thinking=delta.thinking)

        if delta_type == "input_json_delta":
            return StreamChunk(
                type="tool_call_delta",
                arguments_delta=delta.partial_json,
            )

    # -- content_block_start (capture tool_use id + name) -------------------
    if event_type == "content_block_start":
        block = getattr(event, "content_block", None)
        if block and getattr(block, "type", None) == "tool_use":
            return StreamChunk(
                type="tool_call_delta",
                tool_call_id=block.id,
                tool_name=block.name,
            )

    # -- message_delta (final usage) ----------------------------------------
    if event_type == "message_delta":
        usage_info = getattr(event, "usage", None)
        if usage_info:
            return StreamChunk(
                type="usage",
                usage=TokenUsage(
                    prompt_tokens=0,  # input tokens come from message_start
                    completion_tokens=getattr(usage_info, "output_tokens", 0),
                    total_tokens=getattr(usage_info, "output_tokens", 0),
                ),
            )

    # -- message_stop -------------------------------------------------------
    if event_type == "message_stop":
        return StreamChunk(type="done")

    return None


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _map_anthropic_error(exc: Exception) -> ProviderError:
    """Map an ``anthropic`` SDK exception to our ``ProviderError`` hierarchy."""

    if not ANTHROPIC_AVAILABLE:
        return ProviderError(str(exc), provider="anthropic")

    if isinstance(exc, anthropic.RateLimitError):
        return RateLimitError(str(exc), provider="anthropic")

    if isinstance(exc, anthropic.AuthenticationError):
        return AuthenticationError(
            f"Anthropic authentication failed. Check your API key. "
            f"Pass it directly: Runtime(providers={{'anthropic': 'your-key'}}) "
            f"or set the ANTHROPIC_API_KEY environment variable. "
            f"Original error: {exc}",
            provider="anthropic",
        )

    if isinstance(exc, anthropic.NotFoundError):
        return ModelNotFoundError(
            f"{exc}. Known Anthropic models: claude-opus-4-20250514, claude-sonnet-4-20250514, claude-haiku-4-20250414. "
            f"Check available models at https://docs.anthropic.com/en/docs/about-claude/models",
            provider="anthropic",
        )

    if isinstance(exc, anthropic.BadRequestError):
        msg = str(exc).lower()
        if "content" in msg and ("filter" in msg or "safety" in msg or "block" in msg):
            return ContentFilterError(str(exc), provider="anthropic")
        if "context" in msg or "token" in msg:
            return ContextLengthError(str(exc), provider="anthropic")
        return ProviderError(str(exc), provider="anthropic", retryable=False, status_code=400)

    # Connection / timeout errors are retryable
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return ProviderError(str(exc), provider="anthropic", retryable=True)

    # Overloaded (529) is retryable
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", None)
        retryable = status in (429, 502, 503, 504, 529) if status else False
        return ProviderError(
            str(exc), provider="anthropic", retryable=retryable, status_code=status
        )

    # Fallback
    return ProviderError(str(exc), provider="anthropic")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_message(msg: Any, role: str) -> dict[str, Any]:
    """Convert a single ``Message`` into an Anthropic API message dict."""

    # Tool results are sent as ``user`` messages with tool_result blocks
    if role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content if isinstance(msg.content, str) else "",
                }
            ],
        }

    # Build content
    content: str | list[dict[str, Any]]
    if isinstance(msg.content, str):
        content = [{"type": "text", "text": msg.content}]
    elif isinstance(msg.content, list):
        content = [_convert_content_block(b) for b in msg.content]
    else:
        content = [{"type": "text", "text": ""}]

    return {"role": role, "content": content}


def _convert_content_block(block: Any) -> dict[str, Any]:
    """Convert a ``ContentBlock`` into an Anthropic content block dict."""

    block_type = getattr(block, "type", "text")

    if block_type == "text":
        return {"type": "text", "text": block.text or ""}

    if block_type == "image":
        source = block.source
        if source and source.type == "base64":
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": source.media_type or "image/png",
                    "data": source.data or "",
                },
            }
        if source and source.type == "url":
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": source.url or "",
                },
            }
        return {"type": "text", "text": "[image]"}

    if block_type == "image_url":
        # OpenAI-compatible image_url -> Anthropic image format
        image_url = getattr(block, "image_url", None) or {}
        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
        if url.startswith("data:"):
            # Parse data URI: data:<mime>;base64,<data>
            # Format: "data:image/png;base64,iVBOR..."
            try:
                header, data = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
            except (ValueError, IndexError):
                mime = "image/png"
                data = ""
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": data,
                },
            }
        if url.startswith(("http://", "https://")):
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": url,
                },
            }
        return {"type": "text", "text": "[image]"}

    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.tool_use_id or "",
            "name": block.name or "",
            "input": json.loads(block.arguments) if block.arguments else {},
        }

    if block_type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_call_id or "",
            "content": block.content or "",
        }

    if block_type == "thinking":
        return {"type": "thinking", "thinking": block.thinking or ""}

    if block_type == "document":
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": block.document_type or "application/pdf",
                "data": block.document_data or "",
            },
        }

    # Fallback
    return {"type": "text", "text": getattr(block, "text", "") or ""}


def _convert_tool_def(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-format tool definition to Anthropic format.

    OpenAI format::

        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    Anthropic format::

        {"name": ..., "description": ..., "input_schema": ...}
    """

    if "function" in tool:
        func = tool["function"]
        return {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        }

    # Already in Anthropic format or unknown -- pass through
    return tool


# ---------------------------------------------------------------------------
# AnthropicProvider class
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """Native Anthropic Claude provider.

    Implements the ``BaseProvider`` protocol defined in ``gateway.base``
    by talking directly to the Anthropic Messages API.  This enables
    first-class support for extended thinking, prompt caching, and other
    Anthropic-only features.
    """

    def __init__(self, api_key: str, trace_writer: TraceWriter | None = None) -> None:
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic SDK is required for the Anthropic provider but not installed. "
                "Install with: pip install anthropic  (or: uv add anthropic)"
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._trace_writer = trace_writer

    # -- BaseProvider protocol properties -----------------------------------

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def supported_models(self) -> list[str]:
        return [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250414",
        ]

    # -- generate -----------------------------------------------------------

    async def generate(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> LLMResponse:
        """Send a non-streaming request and return the full response."""

        params = to_anthropic_request(request, config)
        try:
            raw = await self._client.messages.create(**params)
        except Exception as exc:
            raise _map_anthropic_error(exc) from exc
        return from_anthropic_response(raw)

    # -- stream -------------------------------------------------------------

    async def stream(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream the response, yielding ``StreamChunk`` objects."""

        params = to_anthropic_request(request, config)
        try:
            async with self._client.messages.stream(**params) as stream:
                async for event in stream:
                    chunk = from_anthropic_stream_event(event)
                    if chunk is not None:
                        yield chunk
        except Exception as exc:
            raise _map_anthropic_error(exc) from exc

    # -- health_check -------------------------------------------------------

    async def health_check(self) -> bool:
        """Verify connectivity by sending a minimal request."""

        try:
            await self._client.messages.create(
                model="claude-haiku-4-20250414",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False
