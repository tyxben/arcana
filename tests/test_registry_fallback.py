"""Tests for ModelGatewayRegistry fallback chains and retry logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from arcana.contracts.llm import (
    LLMRequest,
    LLMResponse,
    ModelConfig,
    StreamChunk,
    TokenUsage,
)
from arcana.gateway.base import ProviderError, RateLimitError
from arcana.gateway.registry import ModelGatewayRegistry

# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(provider: str = "primary") -> ModelConfig:
    return ModelConfig(provider=provider, model_id="test-model")


def _make_request() -> LLMRequest:
    return LLMRequest(messages=[{"role": "user", "content": "hello"}])


def _ok_response(text: str = "ok") -> LLMResponse:
    return LLMResponse(
        content=text,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test-model",
        finish_reason="stop",
    )


class FakeProvider:
    """Minimal provider for testing."""

    def __init__(self, name: str, responses: list | None = None, errors: list | None = None):
        self.provider_name = name
        self.supported_models = ["test-model"]
        self._responses = list(responses or [])
        self._errors = list(errors or [])
        self.call_count = 0

    async def generate(self, request, config, trace_ctx=None):
        self.call_count += 1
        if self._errors:
            raise self._errors.pop(0)
        if self._responses:
            return self._responses.pop(0)
        return _ok_response()

    async def stream(self, request, config, trace_ctx=None):
        self.call_count += 1
        if self._errors:
            raise self._errors.pop(0)
        yield StreamChunk(type="text_delta", text="streamed")
        yield StreamChunk(type="done", usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8))

    async def health_check(self):
        return True


# ── Retry Tests ──────────────────────────────────────────────────────


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        registry = ModelGatewayRegistry(max_retries=2)
        provider = FakeProvider("primary")
        registry.register("primary", provider)

        result = await registry.generate(_make_request(), _make_config())
        assert result.content == "ok"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        """Should retry up to max_retries before failing."""
        registry = ModelGatewayRegistry(max_retries=2, retry_base_delay_ms=1)
        provider = FakeProvider(
            "primary",
            errors=[
                RateLimitError("rate limited", "primary"),
                RateLimitError("rate limited", "primary"),
            ],
        )
        registry.register("primary", provider)

        # 2 errors + 1 success = 3 calls total
        result = await registry.generate(_make_request(), _make_config())
        assert result.content == "ok"
        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises(self):
        """After max_retries, should raise the last error."""
        registry = ModelGatewayRegistry(max_retries=1, retry_base_delay_ms=1)
        provider = FakeProvider(
            "primary",
            errors=[
                RateLimitError("r1", "primary"),
                RateLimitError("r2", "primary"),
            ],
        )
        registry.register("primary", provider)

        with pytest.raises(RateLimitError):
            await registry.generate(_make_request(), _make_config(), use_fallback=False)
        assert provider.call_count == 2  # 1 initial + 1 retry

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable(self):
        """Non-retryable errors should raise immediately."""
        registry = ModelGatewayRegistry(max_retries=2, retry_base_delay_ms=1)
        provider = FakeProvider(
            "primary",
            errors=[ProviderError("auth fail", "primary", retryable=False)],
        )
        registry.register("primary", provider)

        with pytest.raises(ProviderError, match="auth fail"):
            await registry.generate(_make_request(), _make_config())
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_respects_retry_after_ms(self):
        """Should use the larger of computed backoff and retry_after_ms."""
        registry = ModelGatewayRegistry(max_retries=1, retry_base_delay_ms=1)
        provider = FakeProvider(
            "primary",
            errors=[RateLimitError("wait", "primary", retry_after_ms=50)],
        )
        registry.register("primary", provider)

        with patch("arcana.gateway.registry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await registry.generate(_make_request(), _make_config())
            assert result.content == "ok"
            # Should have slept with max(0.001, 0.05) = 0.05
            mock_sleep.assert_called_once()
            delay = mock_sleep.call_args[0][0]
            assert delay >= 0.05


# ── Fallback Tests ───────────────────────────────────────────────────


class TestFallbackChain:
    @pytest.mark.asyncio
    async def test_fallback_on_retryable_error(self):
        """After retry exhaustion, should try fallback chain."""
        registry = ModelGatewayRegistry(max_retries=0, retry_base_delay_ms=1)
        primary = FakeProvider(
            "primary",
            errors=[RateLimitError("fail", "primary")],
        )
        fallback = FakeProvider("fallback", responses=[_ok_response("from fallback")])
        registry.register("primary", primary)
        registry.register("fallback", fallback)
        registry.set_fallback_chain("primary", ["fallback"])

        result = await registry.generate(_make_request(), _make_config())
        assert result.content == "from fallback"
        assert primary.call_count == 1
        assert fallback.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_disabled(self):
        """use_fallback=False should skip fallback chain."""
        registry = ModelGatewayRegistry(max_retries=0, retry_base_delay_ms=1)
        primary = FakeProvider(
            "primary",
            errors=[RateLimitError("fail", "primary")],
        )
        fallback = FakeProvider("fallback")
        registry.register("primary", primary)
        registry.register("fallback", fallback)
        registry.set_fallback_chain("primary", ["fallback"])

        with pytest.raises(RateLimitError):
            await registry.generate(_make_request(), _make_config(), use_fallback=False)
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_all_providers_fail(self):
        """Should raise last error when all providers fail."""
        registry = ModelGatewayRegistry(max_retries=0, retry_base_delay_ms=1)
        primary = FakeProvider("primary", errors=[RateLimitError("p-fail", "primary")])
        fallback = FakeProvider("fallback", errors=[RateLimitError("f-fail", "fallback")])
        registry.register("primary", primary)
        registry.register("fallback", fallback)
        registry.set_fallback_chain("primary", ["fallback"])

        with pytest.raises(RateLimitError, match="f-fail"):
            await registry.generate(_make_request(), _make_config())


# ── Streaming Fallback Tests ────────────────────────────────────────


class TestStreamingFallback:
    @pytest.mark.asyncio
    async def test_stream_success(self):
        registry = ModelGatewayRegistry(max_retries=0)
        provider = FakeProvider("primary")
        registry.register("primary", provider)

        chunks = []
        async for chunk in registry.stream(_make_request(), _make_config()):
            chunks.append(chunk)
        assert any(c.text == "streamed" for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_retry_on_retryable_error(self):
        registry = ModelGatewayRegistry(max_retries=1, retry_base_delay_ms=1)
        provider = FakeProvider(
            "primary",
            errors=[RateLimitError("rate limited", "primary")],
        )
        registry.register("primary", provider)

        chunks = []
        async for chunk in registry.stream(_make_request(), _make_config()):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert provider.call_count == 2  # 1 failed + 1 success

    @pytest.mark.asyncio
    async def test_stream_fallback(self):
        registry = ModelGatewayRegistry(max_retries=0, retry_base_delay_ms=1)
        primary = FakeProvider(
            "primary",
            errors=[RateLimitError("fail", "primary")],
        )
        fallback = FakeProvider("fallback")
        registry.register("primary", primary)
        registry.register("fallback", fallback)
        registry.set_fallback_chain("primary", ["fallback"])

        chunks = []
        async for chunk in registry.stream(_make_request(), _make_config()):
            chunks.append(chunk)
        assert any(c.text == "streamed" for c in chunks)
        assert fallback.call_count == 1

    @pytest.mark.asyncio
    async def test_stream_no_fallback_after_yield(self):
        """Mid-stream errors should propagate, not trigger fallback."""

        class MidStreamProvider:
            provider_name = "midstream"
            supported_models = ["test"]

            async def generate(self, *a, **kw):
                pass

            async def stream(self, *a, **kw):
                yield StreamChunk(type="text_delta", text="partial")
                raise RateLimitError("mid-stream fail", "midstream")

            async def health_check(self):
                return True

        registry = ModelGatewayRegistry(max_retries=0)
        registry.register("primary", MidStreamProvider())
        fallback = FakeProvider("fallback")
        registry.register("fallback", fallback)
        registry.set_fallback_chain("primary", ["fallback"])

        with pytest.raises(RateLimitError, match="mid-stream"):
            chunks = []
            async for chunk in registry.stream(_make_request(), _make_config()):
                chunks.append(chunk)
        # Fallback should NOT have been called since we already yielded
        assert fallback.call_count == 0


# ── Constructor Tests ────────────────────────────────────────────────


class TestRegistryInit:
    def test_default_retry_params(self):
        registry = ModelGatewayRegistry()
        assert registry._max_retries == 2
        assert registry._retry_base_delay_ms == 500

    def test_custom_retry_params(self):
        registry = ModelGatewayRegistry(max_retries=5, retry_base_delay_ms=100)
        assert registry._max_retries == 5
        assert registry._retry_base_delay_ms == 100
