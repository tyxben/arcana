"""OpenAI-compatible provider base class.

This module provides a unified implementation for any LLM API that follows
the OpenAI chat completions format. Most modern LLM providers support this:
- DeepSeek
- Gemini (via v1beta/openai endpoint)
- Ollama
- vLLM
- LiteLLM
- Azure OpenAI
- Together AI
- Groq
- etc.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from arcana.contracts.llm import (
    LLMRequest,
    LLMResponse,
    ModelConfig,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from arcana.contracts.trace import EventType, TraceContext, TraceEvent
from arcana.gateway.base import ModelGateway, ProviderError
from arcana.trace.writer import TraceWriter
from arcana.utils.hashing import canonical_hash

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AsyncOpenAI,
        RateLimitError,
    )

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    APIConnectionError = None  # type: ignore
    APITimeoutError = None  # type: ignore
    RateLimitError = None  # type: ignore


class OpenAICompatibleProvider(ModelGateway):
    """
    Universal provider for OpenAI-compatible APIs.

    This single implementation can be used for any LLM API that follows
    the OpenAI chat completions format. Just provide different base_url
    and api_key for each provider.

    Example:
        # DeepSeek
        deepseek = OpenAICompatibleProvider(
            provider_name="deepseek",
            api_key="sk-xxx",
            base_url="https://api.deepseek.com",
            default_model="deepseek-chat",
        )

        # Gemini
        gemini = OpenAICompatibleProvider(
            provider_name="gemini",
            api_key="AIza-xxx",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            default_model="gemini-2.0-flash",
        )

        # Ollama (local)
        ollama = OpenAICompatibleProvider(
            provider_name="ollama",
            api_key="ollama",  # Ollama doesn't need real key
            base_url="http://localhost:11434/v1",
            default_model="llama3.2",
        )
    """

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        base_url: str,
        default_model: str | None = None,
        supported_models: list[str] | None = None,
        trace_writer: TraceWriter | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        """
        Initialize the OpenAI-compatible provider.

        Args:
            provider_name: Name of this provider (e.g., "deepseek", "gemini")
            api_key: API key for authentication
            base_url: Base URL of the API (e.g., "https://api.deepseek.com")
            default_model: Default model ID to use
            supported_models: List of supported model IDs (for documentation)
            trace_writer: Optional trace writer for logging
            extra_headers: Optional extra headers to include in requests
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required but not installed. "
                "Install with: pip install openai  (or: uv add openai)"
            )

        self._provider_name = provider_name
        self._default_model = default_model
        self._supported_models = supported_models or []
        self.trace_writer = trace_writer

        # Create AsyncOpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=extra_headers,
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def supported_models(self) -> list[str]:
        return self._supported_models

    @property
    def default_model(self) -> str | None:
        return self._default_model

    def _convert_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert Arcana messages to OpenAI format."""
        result = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()

            # Handle role conversion
            role = msg.get("role", "user")
            if hasattr(role, "value"):  # Enum
                role = role.value

            converted: dict[str, Any] = {
                "role": role,
                "content": msg.get("content", ""),
            }

            # Tool messages require tool_call_id (OpenAI format)
            if role == "tool" and msg.get("tool_call_id"):
                converted["tool_call_id"] = msg["tool_call_id"]

            # Assistant messages with tool_calls (OpenAI native format)
            if role == "assistant" and msg.get("tool_calls"):
                raw_calls = msg["tool_calls"]
                openai_calls = []
                for tc in raw_calls:
                    if isinstance(tc, dict):
                        tc_id = tc.get("id", "")
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("arguments", "")
                    else:
                        # Pydantic model
                        tc_id = tc.id if hasattr(tc, "id") else ""
                        tc_name = tc.name if hasattr(tc, "name") else ""
                        tc_args = tc.arguments if hasattr(tc, "arguments") else ""
                    openai_calls.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": tc_name, "arguments": tc_args},
                    })
                converted["tool_calls"] = openai_calls
                # OpenAI requires content to be null/None for tool_call messages
                if not converted.get("content"):
                    converted["content"] = None

            # Assistant messages with name field
            if msg.get("name"):
                converted["name"] = msg["name"]

            result.append(converted)
        return result

    async def generate(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        # Convert messages
        messages = self._convert_messages(request.messages)

        # Log request digest
        request_digest = canonical_hash({
            "messages": messages,
            "config": config.model_dump(),
        })

        try:
            # Build request parameters
            params: dict[str, Any] = {
                "model": config.model_id,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }

            # Add seed if specified (for reproducibility)
            if config.seed is not None:
                params["seed"] = config.seed

            # Add response format if schema specified
            if request.response_schema:
                params["response_format"] = {"type": "json_object"}

            # Add tools if specified
            if request.tools:
                params["tools"] = request.tools

            # Add any extra params from config
            if config.extra_params:
                params.update(config.extra_params)

            # Make API call
            response = await self.client.chat.completions.create(**params)

        except RateLimitError as e:
            raise ProviderError(
                f"Rate limit hit on provider '{self._provider_name}': {e}. "
                f"Wait a moment and retry, or set up a fallback provider.",
                provider=self._provider_name,
                retryable=True,
                status_code=429,
            ) from e
        except (APIConnectionError, APITimeoutError) as e:
            raise ProviderError(
                f"Connection error with provider '{self._provider_name}': {e}. "
                f"Check your network connection and the provider's status page.",
                provider=self._provider_name,
                retryable=True,
            ) from e
        except Exception as e:
            error_msg = str(e)
            error_lower = error_msg.lower()
            # Detect model-not-found errors
            if any(phrase in error_lower for phrase in ["model not found", "model_not_found", "does not exist", "invalid model"]):
                model_hint = (
                    f"Model '{config.model_id}' not found on provider '{self._provider_name}'. "
                )
                if self._supported_models:
                    model_hint += f"Known models: {self._supported_models}. "
                model_hint += "Check available models in your provider's documentation."
                raise ProviderError(
                    model_hint,
                    provider=self._provider_name,
                    retryable=False,
                    status_code=404,
                ) from e
            # Detect auth errors
            if any(phrase in error_lower for phrase in ["401", "unauthorized", "invalid api key", "invalid_api_key", "authentication"]):
                env_var = f"{self._provider_name.upper()}_API_KEY"
                raise ProviderError(
                    f"Authentication failed for provider '{self._provider_name}'. "
                    f"Check your API key. Pass it directly: Runtime(providers={{'{self._provider_name}': 'your-key'}}) "
                    f"or set the {env_var} environment variable.",
                    provider=self._provider_name,
                    retryable=False,
                    status_code=401,
                ) from e
            # Check for retryable status codes in error message as fallback
            retryable = any(
                code in error_msg for code in ["503", "529", "502", "504"]
            )
            raise ProviderError(
                f"Provider '{self._provider_name}' error: {error_msg}",
                provider=self._provider_name,
                retryable=retryable,
            ) from e

        # Parse response
        choice = response.choices[0]
        content = choice.message.content

        # Parse tool calls if any
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in choice.message.tool_calls
            ]

        # Get usage
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        # Create response
        llm_response = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
        )

        # Log to trace
        if self.trace_writer and trace_ctx:
            response_digest = canonical_hash({
                "content": content,
                "usage": usage.model_dump(),
            })

            event = TraceEvent(
                run_id=trace_ctx.run_id,
                task_id=trace_ctx.task_id,
                step_id=trace_ctx.new_step_id(),
                timestamp=datetime.now(UTC),
                event_type=EventType.LLM_CALL,
                llm_request_digest=request_digest,
                llm_response_digest=response_digest,
                model=response.model,
                metadata={"provider": self.provider_name},
            )
            self.trace_writer.write(event)

        return llm_response

    async def stream(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response chunks using the OpenAI streaming API.

        Yields StreamChunk objects as tokens arrive, enabling real-time
        token-level streaming. Tool call deltas are tracked across chunks
        so each yielded chunk includes the tool_call_id.
        """
        messages = self._convert_messages(request.messages)

        params: dict[str, Any] = {
            "model": config.model_id,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if config.seed is not None:
            params["seed"] = config.seed
        if request.response_schema:
            params["response_format"] = {"type": "json_object"}
        if request.tools:
            params["tools"] = request.tools
        if config.extra_params:
            params.update(config.extra_params)

        stream_response = await self.client.chat.completions.create(**params)

        # Track tool call state across incremental deltas
        tool_call_state: dict[int, dict[str, str]] = {}

        async for chunk in stream_response:
            # Usage-only chunk (typically the last one with stream_options)
            if not chunk.choices:
                if chunk.usage:
                    yield StreamChunk(
                        type="usage",
                        usage=TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        ),
                    )
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Text content
            if delta.content:
                yield StreamChunk(type="text_delta", text=delta.content)

            # Tool call deltas (accumulated by index)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_state:
                        tool_call_state[idx] = {"id": "", "name": ""}
                    entry = tool_call_state[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        entry["name"] = tc_delta.function.name
                    yield StreamChunk(
                        type="tool_call_delta",
                        tool_call_id=entry["id"] or None,
                        tool_name=entry["name"] or None,
                        arguments_delta=(
                            tc_delta.function.arguments
                            if tc_delta.function and tc_delta.function.arguments
                            else None
                        ),
                    )

            # Finish reason signals end of this choice
            if choice.finish_reason:
                yield StreamChunk(
                    type="done",
                    metadata={
                        "finish_reason": choice.finish_reason,
                        "model": chunk.model or config.model_id,
                    },
                )

    async def health_check(self) -> bool:
        """Check if the API is accessible."""
        try:
            await self.client.models.list()
            return True
        except Exception:
            # Some APIs don't support models.list, try a minimal completion
            try:
                await self.client.chat.completions.create(
                    model=self._default_model or "gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
                return True
            except Exception:
                return False


# Pre-configured provider factories for convenience
def create_deepseek_provider(
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    trace_writer: TraceWriter | None = None,
) -> OpenAICompatibleProvider:
    """Create a DeepSeek provider."""
    return OpenAICompatibleProvider(
        provider_name="deepseek",
        api_key=api_key,
        base_url=base_url,
        default_model="deepseek-chat",
        supported_models=["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        trace_writer=trace_writer,
    )


def create_gemini_provider(
    api_key: str,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
    trace_writer: TraceWriter | None = None,
) -> OpenAICompatibleProvider:
    """Create a Gemini provider (via OpenAI-compatible endpoint)."""
    return OpenAICompatibleProvider(
        provider_name="gemini",
        api_key=api_key,
        base_url=base_url,
        default_model="gemini-2.0-flash",
        supported_models=[
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        trace_writer=trace_writer,
    )


def create_ollama_provider(
    base_url: str = "http://localhost:11434/v1",
    default_model: str = "llama3.2",
    trace_writer: TraceWriter | None = None,
) -> OpenAICompatibleProvider:
    """Create an Ollama provider (local)."""
    return OpenAICompatibleProvider(
        provider_name="ollama",
        api_key="ollama",  # Ollama doesn't require a real API key
        base_url=base_url,
        default_model=default_model,
        trace_writer=trace_writer,
    )


# ---------------------------------------------------------------------------
# Chinese provider factories
# ---------------------------------------------------------------------------


def create_kimi_provider(
    api_key: str,
    trace_writer: TraceWriter | None = None,
) -> OpenAICompatibleProvider:
    """Create a Kimi (Moonshot) provider."""
    return OpenAICompatibleProvider(
        provider_name="kimi",
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        supported_models=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        trace_writer=trace_writer,
    )


def create_glm_provider(
    api_key: str,
    trace_writer: TraceWriter | None = None,
) -> OpenAICompatibleProvider:
    """Create a GLM (Zhipu AI) provider."""
    return OpenAICompatibleProvider(
        provider_name="glm",
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        supported_models=["glm-4", "glm-4-flash", "glm-4v", "glm-4-long"],
        trace_writer=trace_writer,
    )


def create_minimax_provider(
    api_key: str,
    trace_writer: TraceWriter | None = None,
) -> OpenAICompatibleProvider:
    """Create a MiniMax provider."""
    return OpenAICompatibleProvider(
        provider_name="minimax",
        api_key=api_key,
        base_url="https://api.minimax.chat/v1",
        default_model="abab6.5s-chat",
        supported_models=["abab6.5s-chat", "abab6.5-chat", "abab5.5-chat"],
        trace_writer=trace_writer,
    )
