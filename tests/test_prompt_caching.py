"""Tests for prompt caching support across providers.

Verifies that:
- Anthropic provider automatically adds cache_control to system messages and tools
- Anthropic cache metrics (cache_creation_input_tokens, cache_read_input_tokens)
  are tracked in TokenUsage and AnthropicResponseExt
- OpenAI provider tracks cached_tokens from prompt_tokens_details
- Cache_control field on Message is optional and backward compatible
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.contracts.llm import (
    AnthropicRequestExt,
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
    TokenUsage,
)
from arcana.gateway.providers.anthropic import (
    from_anthropic_response,
    to_anthropic_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: object) -> ModelConfig:
    defaults = {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_request(
    *,
    system: str = "You are a helpful assistant.",
    user: str = "Hello",
    tools: list[dict] | None = None,
    prompt_caching: bool | None = None,
) -> LLMRequest:
    messages = [
        Message(role=MessageRole.SYSTEM, content=system),
        Message(role=MessageRole.USER, content=user),
    ]
    anthropic_ext = None
    if prompt_caching is not None:
        anthropic_ext = AnthropicRequestExt(prompt_caching=prompt_caching)
    return LLMRequest(
        messages=messages,
        tools=tools,
        anthropic=anthropic_ext,
    )


def _make_raw_response(
    *,
    text: str = "Hello!",
    cache_creation: int | None = None,
    cache_read: int | None = None,
) -> SimpleNamespace:
    """Create a mock Anthropic raw response object."""
    usage_fields = {
        "input_tokens": 100,
        "output_tokens": 50,
    }
    if cache_creation is not None:
        usage_fields["cache_creation_input_tokens"] = cache_creation
    if cache_read is not None:
        usage_fields["cache_read_input_tokens"] = cache_read

    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(**usage_fields),
        model="claude-sonnet-4-20250514",
        stop_reason="end_turn",
    )


# =========================================================================
# 1. Anthropic: cache_control injection on system message
# =========================================================================


class TestAnthropicSystemCaching:
    """Verify that to_anthropic_request adds cache_control to system prompt."""

    def test_system_message_gets_cache_control_by_default(self) -> None:
        """System prompt should be wrapped in a content block with cache_control."""
        request = _make_request(system="You are a test assistant.")
        config = _make_config()
        params = to_anthropic_request(request, config)

        # System should be a list of content blocks, not a plain string
        system = params["system"]
        assert isinstance(system, list), f"Expected list, got {type(system)}"
        assert len(system) == 1
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "You are a test assistant."
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_system_caching_with_explicit_true(self) -> None:
        """Explicit prompt_caching=True should also add cache_control."""
        request = _make_request(prompt_caching=True)
        config = _make_config()
        params = to_anthropic_request(request, config)

        system = params["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_system_caching_disabled_when_false(self) -> None:
        """prompt_caching=False should send system as plain string."""
        request = _make_request(prompt_caching=False)
        config = _make_config()
        params = to_anthropic_request(request, config)

        system = params["system"]
        assert isinstance(system, str)
        assert system == "You are a helpful assistant."

    def test_no_system_message_no_crash(self) -> None:
        """If there is no system message, params should not have system key."""
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="Hi")],
        )
        config = _make_config()
        params = to_anthropic_request(request, config)
        assert "system" not in params


# =========================================================================
# 2. Anthropic: cache_control injection on tool definitions
# =========================================================================


class TestAnthropicToolCaching:
    """Verify that to_anthropic_request adds cache_control to the last tool."""

    def test_last_tool_gets_cache_control(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "search", "description": "Search", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "read", "description": "Read file", "parameters": {"type": "object", "properties": {}}}},
        ]
        request = _make_request(tools=tools)
        config = _make_config()
        params = to_anthropic_request(request, config)

        converted_tools = params["tools"]
        assert len(converted_tools) == 2

        # First tool should NOT have cache_control
        assert "cache_control" not in converted_tools[0]

        # Last tool SHOULD have cache_control
        assert converted_tools[-1]["cache_control"] == {"type": "ephemeral"}

    def test_single_tool_gets_cache_control(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "search", "description": "Search", "parameters": {"type": "object", "properties": {}}}},
        ]
        request = _make_request(tools=tools)
        config = _make_config()
        params = to_anthropic_request(request, config)

        assert params["tools"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_tools_no_crash(self) -> None:
        request = _make_request(tools=None)
        config = _make_config()
        params = to_anthropic_request(request, config)
        assert "tools" not in params

    def test_tool_caching_disabled(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "search", "description": "Search", "parameters": {"type": "object", "properties": {}}}},
        ]
        request = _make_request(tools=tools, prompt_caching=False)
        config = _make_config()
        params = to_anthropic_request(request, config)

        # Tools should be present but without cache_control
        assert "cache_control" not in params["tools"][0]


# =========================================================================
# 3. Anthropic: cache metrics in response
# =========================================================================


class TestAnthropicCacheMetrics:
    """Verify that from_anthropic_response extracts cache metrics."""

    def test_cache_creation_tokens_tracked(self) -> None:
        raw = _make_raw_response(cache_creation=500, cache_read=0)
        response = from_anthropic_response(raw)

        assert response.usage.cache_creation_input_tokens == 500
        assert response.usage.cache_read_input_tokens is None  # 0 is falsy
        assert response.anthropic is not None
        assert response.anthropic.cache_creation_input_tokens == 500

    def test_cache_read_tokens_tracked(self) -> None:
        raw = _make_raw_response(cache_creation=0, cache_read=450)
        response = from_anthropic_response(raw)

        assert response.usage.cache_read_input_tokens == 450
        assert response.anthropic is not None
        assert response.anthropic.cache_read_input_tokens == 450

    def test_both_cache_metrics(self) -> None:
        raw = _make_raw_response(cache_creation=200, cache_read=300)
        response = from_anthropic_response(raw)

        assert response.usage.cache_creation_input_tokens == 200
        assert response.usage.cache_read_input_tokens == 300

    def test_no_cache_metrics_backward_compat(self) -> None:
        """Responses without cache fields should still work (backward compat)."""
        raw = _make_raw_response()
        response = from_anthropic_response(raw)

        assert response.usage.cache_creation_input_tokens is None
        assert response.usage.cache_read_input_tokens is None
        assert response.usage.prompt_tokens == 100
        assert response.usage.completion_tokens == 50

    def test_base_usage_still_correct(self) -> None:
        raw = _make_raw_response(cache_creation=500, cache_read=300)
        response = from_anthropic_response(raw)

        assert response.usage.prompt_tokens == 100
        assert response.usage.completion_tokens == 50
        assert response.usage.total_tokens == 150


# =========================================================================
# 4. OpenAI: cached_tokens tracking
# =========================================================================


class TestOpenAICachedTokens:
    """Verify that OpenAI provider extracts cached_tokens from usage."""

    @pytest.mark.asyncio
    async def test_cached_tokens_extracted(self) -> None:
        """When OpenAI returns cached_tokens in prompt_tokens_details, track it."""
        from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider

        # Build a mock response matching OpenAI's structure
        mock_choice = SimpleNamespace(
            message=SimpleNamespace(
                content="Hello!",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
        mock_usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=50,
            total_tokens=1050,
            prompt_tokens_details=SimpleNamespace(cached_tokens=800),
        )
        mock_response = SimpleNamespace(
            choices=[mock_choice],
            usage=mock_usage,
            model="gpt-4o",
        )

        provider = OpenAICompatibleProvider(
            provider_name="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

        # Mock the client's create method
        provider.client = MagicMock()
        provider.client.chat = MagicMock()
        provider.client.chat.completions = MagicMock()
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="Hi")],
        )
        config = ModelConfig(provider="openai", model_id="gpt-4o")

        response = await provider.generate(request, config)

        assert response.usage.cached_tokens == 800
        assert response.usage.prompt_tokens == 1000

    @pytest.mark.asyncio
    async def test_no_cached_tokens_backward_compat(self) -> None:
        """When OpenAI response has no prompt_tokens_details, cached_tokens is None."""
        from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider

        mock_choice = SimpleNamespace(
            message=SimpleNamespace(
                content="Hello!",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
        mock_usage = SimpleNamespace(
            prompt_tokens=500,
            completion_tokens=50,
            total_tokens=550,
        )
        mock_response = SimpleNamespace(
            choices=[mock_choice],
            usage=mock_usage,
            model="gpt-4o",
        )

        provider = OpenAICompatibleProvider(
            provider_name="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        provider.client = MagicMock()
        provider.client.chat = MagicMock()
        provider.client.chat.completions = MagicMock()
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="Hi")],
        )
        config = ModelConfig(provider="openai", model_id="gpt-4o")

        response = await provider.generate(request, config)

        assert response.usage.cached_tokens is None
        assert response.usage.prompt_tokens == 500


# =========================================================================
# 5. Message model: cache_control field
# =========================================================================


class TestMessageCacheControl:
    """Verify the cache_control field on Message is optional and backward compatible."""

    def test_default_is_none(self) -> None:
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.cache_control is None

    def test_can_set_cache_control(self) -> None:
        msg = Message(
            role=MessageRole.SYSTEM,
            content="System prompt",
            cache_control={"type": "ephemeral"},
        )
        assert msg.cache_control == {"type": "ephemeral"}

    def test_serialization_backward_compat(self) -> None:
        """Messages without cache_control should serialize cleanly."""
        msg = Message(role=MessageRole.USER, content="Hi")
        data = msg.model_dump()
        assert data["cache_control"] is None

    def test_deserialization_without_cache_control(self) -> None:
        """Old serialized messages (without cache_control) should still load."""
        data = {"role": "user", "content": "Hi"}
        msg = Message(**data)
        assert msg.cache_control is None


# =========================================================================
# 6. TokenUsage: cache fields
# =========================================================================


class TestTokenUsageCacheFields:
    """Verify that TokenUsage cache fields are optional and backward compatible."""

    def test_defaults_are_none(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None
        assert usage.cached_tokens is None

    def test_anthropic_cache_fields(self) -> None:
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=300,
        )
        assert usage.cache_creation_input_tokens == 500
        assert usage.cache_read_input_tokens == 300
        assert usage.cached_tokens is None

    def test_openai_cache_fields(self) -> None:
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=50,
            total_tokens=1050,
            cached_tokens=800,
        )
        assert usage.cached_tokens == 800
        assert usage.cache_creation_input_tokens is None

    def test_cost_estimate_unchanged(self) -> None:
        """Cache fields should not affect cost_estimate (it's a placeholder anyway)."""
        usage_no_cache = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage_with_cache = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cache_read_input_tokens=80,
        )
        assert usage_no_cache.cost_estimate == usage_with_cache.cost_estimate
