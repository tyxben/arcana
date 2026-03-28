"""Tests for provider client cleanup (connection pool leak fix)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arcana.gateway.registry import ModelGatewayRegistry

# ---------------------------------------------------------------------------
# OpenAICompatibleProvider.close()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_provider_close_calls_client_close():
    """OpenAICompatibleProvider.close() should call client.close()."""
    from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        provider_name="test",
        api_key="fake-key",
        base_url="https://example.com",
    )
    # Replace the real client with a mock
    mock_client = AsyncMock()
    provider.client = mock_client

    await provider.close()

    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_provider_close_with_none_client():
    """close() should be safe when client is None."""
    from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        provider_name="test",
        api_key="fake-key",
        base_url="https://example.com",
    )
    provider.client = None  # type: ignore[assignment]

    # Should not raise
    await provider.close()


# ---------------------------------------------------------------------------
# AnthropicProvider.close()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_provider_close_calls_client_close():
    """AnthropicProvider.close() should call _client.close()."""
    from arcana.gateway.providers.anthropic import AnthropicProvider

    with patch("arcana.gateway.providers.anthropic.ANTHROPIC_AVAILABLE", True):
        with patch("arcana.gateway.providers.anthropic.anthropic") as mock_anthropic:
            mock_async_client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = mock_async_client

            provider = AnthropicProvider(api_key="fake-key")

    await provider.close()

    mock_async_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_provider_close_with_none_client():
    """close() should be safe when _client is None."""
    from arcana.gateway.providers.anthropic import AnthropicProvider

    with patch("arcana.gateway.providers.anthropic.ANTHROPIC_AVAILABLE", True):
        with patch("arcana.gateway.providers.anthropic.anthropic") as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = MagicMock()
            provider = AnthropicProvider(api_key="fake-key")

    provider._client = None  # type: ignore[assignment]

    # Should not raise
    await provider.close()


# ---------------------------------------------------------------------------
# ModelGatewayRegistry.close()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_close_calls_all_providers():
    """Registry.close() should call close() on every registered provider."""
    registry = ModelGatewayRegistry()

    provider_a = AsyncMock()
    provider_a.close = AsyncMock()

    provider_b = AsyncMock()
    provider_b.close = AsyncMock()

    registry.register("a", provider_a)
    registry.register("b", provider_b)

    await registry.close()

    provider_a.close.assert_awaited_once()
    provider_b.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_registry_close_skips_providers_without_close():
    """Registry.close() should skip providers that lack a close() method."""
    registry = ModelGatewayRegistry()

    # Provider with close()
    provider_with = AsyncMock()
    provider_with.close = AsyncMock()

    # Provider without close() -- use a plain object with no close attr
    provider_without = MagicMock(spec=["provider_name", "generate"])

    registry.register("with_close", provider_with)
    registry.register("without_close", provider_without)

    # Should not raise
    await registry.close()

    provider_with.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_registry_close_empty():
    """Registry.close() should work with no providers registered."""
    registry = ModelGatewayRegistry()
    # Should not raise
    await registry.close()


# ---------------------------------------------------------------------------
# Runtime.close() integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_close_closes_gateway():
    """Runtime.close() should call gateway.close()."""
    from arcana.runtime_core import Runtime

    runtime = Runtime(providers={"deepseek": "fake-key"})

    # Replace gateway with a mock
    mock_gateway = AsyncMock()
    mock_gateway.close = AsyncMock()
    runtime._gateway = mock_gateway

    await runtime.close()

    mock_gateway.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_close_handles_mcp_and_gateway():
    """Runtime.close() should clean up both MCP and gateway."""
    from arcana.runtime_core import Runtime

    runtime = Runtime(providers={"deepseek": "fake-key"})

    # Mock MCP client
    mock_mcp = AsyncMock()
    mock_mcp.disconnect_all = AsyncMock()
    runtime._mcp_client = mock_mcp

    # Mock gateway
    mock_gateway = AsyncMock()
    mock_gateway.close = AsyncMock()
    runtime._gateway = mock_gateway

    await runtime.close()

    mock_mcp.disconnect_all.assert_awaited_once()
    mock_gateway.close.assert_awaited_once()
