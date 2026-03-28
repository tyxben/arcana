"""Tests for timeout wiring and cancellation safety.

Part 1: Verify timeout_ms is passed to provider SDK calls.
Part 2: Verify cancellation during run() still records budget.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arcana.contracts.llm import (
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
)

# ---------------------------------------------------------------------------
# Part 1: Timeout wiring
# ---------------------------------------------------------------------------


class TestOpenAITimeoutWiring:
    """Verify timeout is passed to OpenAI-compatible provider generate/stream."""

    @pytest.fixture()
    def provider(self):
        """Create an OpenAI-compatible provider with a mock client."""
        with patch("arcana.gateway.providers.openai_compatible.OPENAI_AVAILABLE", True):
            with patch("arcana.gateway.providers.openai_compatible.AsyncOpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client

                from arcana.gateway.providers.openai_compatible import (
                    OpenAICompatibleProvider,
                )

                p = OpenAICompatibleProvider(
                    provider_name="test",
                    api_key="fake-key",
                    base_url="http://localhost",
                    default_model="test-model",
                )
                # Replace client with our mock
                p.client = mock_client
                return p

    def _make_request(self) -> LLMRequest:
        return LLMRequest(
            messages=[Message(role=MessageRole.USER, content="hello")],
        )

    @pytest.mark.asyncio()
    async def test_generate_passes_timeout(self, provider):
        """generate() should pass timeout=config.timeout_ms/1000 to SDK."""
        # Setup mock response
        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )
        mock_response.usage.prompt_tokens_details = None
        mock_response.model = "test-model"

        provider.client.chat.completions.create = AsyncMock(
            return_value=mock_response,
        )

        config = ModelConfig(
            provider="test", model_id="test-model", timeout_ms=15000,
        )
        request = self._make_request()
        await provider.generate(request, config)

        # Verify timeout was passed as seconds
        call_kwargs = provider.client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("timeout") == 15.0, (
            f"Expected timeout=15.0, got {call_kwargs.kwargs.get('timeout')}"
        )

    @pytest.mark.asyncio()
    async def test_generate_default_timeout(self, provider):
        """Default timeout_ms=30000 should result in timeout=30.0."""
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(
            prompt_tokens=5, completion_tokens=3, total_tokens=8,
        )
        mock_response.usage.prompt_tokens_details = None
        mock_response.model = "test-model"

        provider.client.chat.completions.create = AsyncMock(
            return_value=mock_response,
        )

        config = ModelConfig(provider="test", model_id="test-model")
        await provider.generate(self._make_request(), config)

        call_kwargs = provider.client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("timeout") == 30.0

    @pytest.mark.asyncio()
    async def test_stream_passes_timeout(self, provider):
        """stream() should pass timeout to SDK."""
        # Create an async iterable mock for the stream response
        async def _fake_stream():
            return
            yield  # noqa: F841 -- makes this an async generator

        provider.client.chat.completions.create = AsyncMock(
            return_value=_fake_stream(),
        )

        config = ModelConfig(
            provider="test", model_id="test-model", timeout_ms=45000,
        )
        request = self._make_request()

        # Consume the stream
        async for _chunk in provider.stream(request, config):
            pass

        call_kwargs = provider.client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("timeout") == 45.0


class TestAnthropicTimeoutWiring:
    """Verify timeout is passed to Anthropic provider generate/stream."""

    @pytest.fixture()
    def provider(self):
        """Create an Anthropic provider with a mock client."""
        anthropic_available = pytest.importorskip("anthropic")  # noqa: F841

        from arcana.gateway.providers.anthropic import AnthropicProvider

        p = AnthropicProvider(api_key="fake-key")
        p._client = MagicMock()
        return p

    def _make_request(self) -> LLMRequest:
        return LLMRequest(
            messages=[Message(role=MessageRole.USER, content="hello")],
        )

    @pytest.mark.asyncio()
    async def test_generate_passes_timeout(self, provider):
        """generate() should pass timeout to Anthropic SDK."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="hi")]
        mock_response.usage = MagicMock(
            input_tokens=10, output_tokens=5,
        )
        mock_response.usage.cache_creation_input_tokens = None
        mock_response.usage.cache_read_input_tokens = None
        mock_response.model = "claude-haiku-4-20250414"
        mock_response.stop_reason = "end_turn"

        provider._client.messages.create = AsyncMock(
            return_value=mock_response,
        )

        config = ModelConfig(
            provider="anthropic",
            model_id="claude-haiku-4-20250414",
            timeout_ms=20000,
        )
        await provider.generate(self._make_request(), config)

        call_kwargs = provider._client.messages.create.call_args
        assert call_kwargs.kwargs.get("timeout") == 20.0


# ---------------------------------------------------------------------------
# Part 2: Cancellation safety
# ---------------------------------------------------------------------------


class TestRuntimeCancellationSafety:
    """Runtime.run() should record budget even when cancelled."""

    @pytest.mark.asyncio()
    async def test_cancellation_records_partial_budget(self):
        """When a task is cancelled, partial tokens/cost should be tracked."""
        from arcana.contracts.state import AgentState, ExecutionStatus

        # Build a Runtime with a fake provider
        runtime = self._build_runtime()

        # Make session.run() simulate cancellation after partial work
        partial_state = AgentState(
            run_id="test-run",
            goal="test",
            tokens_used=500,
            cost_usd=0.05,
            status=ExecutionStatus.FAILED,
            last_error="cancelled",
        )

        async def _fake_session_run(goal):
            # Simulate that Session set its state before cancellation
            return None  # We'll never reach this

        # We'll patch _create_session to return a session that cancels
        with patch.object(runtime, "_create_session") as mock_create:
            mock_session = MagicMock()
            mock_session.run_id = "test-run"
            mock_session.state = partial_state

            async def _cancelled_run(goal):
                raise asyncio.CancelledError()

            mock_session.run = _cancelled_run
            mock_create.return_value = mock_session

            with pytest.raises(asyncio.CancelledError):
                await runtime.run("test goal")

        # Budget from the partial state should be recorded
        assert runtime.tokens_used == 500
        assert runtime.budget_used_usd == pytest.approx(0.05)

    @pytest.mark.asyncio()
    async def test_cancellation_propagates(self):
        """CancelledError should propagate after cleanup."""
        runtime = self._build_runtime()

        with patch.object(runtime, "_create_session") as mock_create:
            mock_session = MagicMock()
            mock_session.run_id = "test-run"
            mock_session.state = None

            async def _cancelled_run(goal):
                raise asyncio.CancelledError()

            mock_session.run = _cancelled_run
            mock_create.return_value = mock_session

            with pytest.raises(asyncio.CancelledError):
                await runtime.run("test goal")

    @pytest.mark.asyncio()
    async def test_normal_run_still_records_budget(self):
        """Normal (non-cancelled) runs should still record budget."""
        from arcana.runtime_core import RunResult

        runtime = self._build_runtime()

        with patch.object(runtime, "_create_session") as mock_create:
            mock_session = MagicMock()
            mock_session.run_id = "test-run"
            mock_session.state = None

            fake_result = RunResult(
                output="done",
                success=True,
                tokens_used=1000,
                cost_usd=0.10,
                run_id="test-run",
            )

            mock_session.run = AsyncMock(return_value=fake_result)
            mock_create.return_value = mock_session

            result = await runtime.run("test goal")

        assert result.tokens_used == 1000
        assert runtime.tokens_used == 1000
        assert runtime.budget_used_usd == pytest.approx(0.10)

    def _build_runtime(self):
        """Build a Runtime with a fake provider for testing."""
        from arcana.gateway.registry import ModelGatewayRegistry
        from arcana.runtime_core import Runtime

        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        mock_provider.default_model = "test-model"
        mock_provider.supported_models = ["test-model"]

        with patch.object(
            Runtime, "_setup_providers", return_value=ModelGatewayRegistry(),
        ):
            runtime = Runtime(providers={"test": "fake-key"})

        # Register the mock provider manually
        runtime._gateway.register("test", mock_provider)
        runtime._gateway.set_default("test")
        return runtime


class TestConversationAgentCancellation:
    """ConversationAgent.run() should preserve state on cancellation."""

    @pytest.mark.asyncio()
    async def test_run_cancellation_preserves_state(self):
        """When cancelled, run() should still set _state."""
        from arcana.contracts.state import ExecutionStatus
        from arcana.runtime.conversation import ConversationAgent

        # Create agent with mock gateway
        mock_gateway = MagicMock()

        agent = ConversationAgent(
            gateway=mock_gateway,
            model_config=ModelConfig(
                provider="test", model_id="test-model",
            ),
            max_turns=5,
        )

        # Make the gateway.stream raise CancelledError on first call
        async def _cancel_on_stream(*args, **kwargs):
            raise asyncio.CancelledError()
            yield  # noqa: F841 -- makes it an async generator

        mock_gateway.stream = _cancel_on_stream
        mock_gateway.generate = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await agent.run("test goal")

        # _state should be set even after cancellation
        assert agent._state is not None
        assert agent._state.status == ExecutionStatus.FAILED
