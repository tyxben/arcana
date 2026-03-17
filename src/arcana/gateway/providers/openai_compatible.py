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

from datetime import UTC, datetime
from typing import Any

from arcana.contracts.llm import (
    LLMRequest,
    LLMResponse,
    ModelConfig,
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
                "openai is not installed. Install with: pip install openai"
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

            result.append({
                "role": role,
                "content": msg.get("content", ""),
            })
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
                str(e),
                provider=self._provider_name,
                retryable=True,
                status_code=429,
            ) from e
        except (APIConnectionError, APITimeoutError) as e:
            raise ProviderError(
                str(e),
                provider=self._provider_name,
                retryable=True,
            ) from e
        except Exception as e:
            error_msg = str(e)
            # Check for retryable status codes in error message as fallback
            retryable = any(
                code in error_msg for code in ["503", "529", "502", "504"]
            )
            raise ProviderError(
                error_msg,
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
            )
            self.trace_writer.write(event)

        return llm_response

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
