"""Tests for LLM-driven context compression in WorkingSetBuilder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.context.builder import WorkingSetBuilder, estimate_tokens
from arcana.contracts.context import TokenBudget
from arcana.contracts.llm import (
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    TokenUsage,
)


def _make_messages(count: int, content_len: int = 100) -> list[Message]:
    """Create a list of alternating user/assistant messages with a system head."""
    msgs = [Message(role=MessageRole.SYSTEM, content="You are helpful.")]
    for i in range(count):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        msgs.append(Message(role=role, content=f"Message {i}: " + "x" * content_len))
    return msgs


def _make_mock_gateway(summary_text: str = "Summary of conversation.") -> MagicMock:
    """Create a mock ModelGatewayRegistry that returns a summary."""
    gateway = MagicMock()
    gateway.default_provider = "mock_provider"
    gateway.list_providers.return_value = ["mock_provider"]

    mock_provider = MagicMock()
    mock_provider.default_model = "mock-model"
    gateway.get.return_value = mock_provider

    response = LLMResponse(
        content=summary_text,
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        model="mock-model",
        finish_reason="stop",
    )
    gateway.generate = AsyncMock(return_value=response)
    return gateway


def _make_large_messages() -> list[Message]:
    """Create messages large enough to guarantee the middle section exceeds the
    LLM compression threshold (2000 tokens).

    Structure: 1 system + 30 middle messages (each ~250 tokens) + 6 tail = 37 total.
    Middle tokens: 30 * 250 = 7500 tokens, well above threshold.
    """
    msgs = [Message(role=MessageRole.SYSTEM, content="System prompt.")]
    # 30 middle messages with ~1000 chars each (~250 tokens)
    for i in range(30):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        msgs.append(Message(role=role, content=f"Turn {i}: " + "a" * 1000))
    # 6 tail messages (short, so they fit in budget easily)
    for i in range(6):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        msgs.append(Message(role=role, content=f"Recent {i}: short"))
    return msgs


# ---- Test: sync compression still works (no gateway) ----


class TestSyncCompressionBackwardCompat:
    """Sync build_conversation_context must work unchanged without a gateway."""

    def test_no_gateway_under_budget(self):
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=128000),
        )
        msgs = _make_messages(4)
        result = builder.build_conversation_context(msgs, turn=0)
        assert len(result) == len(msgs)
        assert builder.last_decision is not None
        assert not builder.last_decision.history_compressed

    def test_no_gateway_over_budget_uses_keyword_compression(self):
        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=800, response_reserve=50),
        )
        msgs = _make_messages(20, content_len=200)
        result = builder.build_conversation_context(msgs, turn=3)
        assert len(result) < len(msgs)
        has_keyword_summary = any(
            "Earlier conversation" in (m.content or "") for m in result
        )
        assert has_keyword_summary
        assert builder.last_decision is not None
        assert builder.last_decision.history_compressed
        # Should NOT be LLM-compressed since no gateway
        assert "LLM" not in builder.last_decision.explanation


# ---- Test: async compression with mocked gateway ----


class TestAsyncLLMCompression:
    @pytest.mark.asyncio
    async def test_llm_compression_triggered(self):
        """When gateway is set and middle section is large, LLM compression is used."""
        gateway = _make_mock_gateway(
            "The user asked about Python files. Key findings: config.py found."
        )

        # Use a large enough total_window so summary_budget is positive,
        # but small enough that the total message tokens exceed it.
        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=3000, response_reserve=200),
            goal="find Python files",
            gateway=gateway,
        )
        msgs = _make_large_messages()
        result = await builder.abuild_conversation_context(msgs, turn=5)

        # Should have compressed
        assert len(result) < len(msgs)
        # Should contain LLM summary
        has_llm_summary = any("LLM summary" in (m.content or "") for m in result)
        assert has_llm_summary
        # Decision should note LLM compression
        assert builder.last_decision is not None
        assert builder.last_decision.history_compressed
        assert "LLM-compressed" in builder.last_decision.explanation
        # Gateway should have been called
        gateway.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_compression_includes_goal_in_prompt(self):
        """The summarization prompt should include the goal for relevance."""
        gateway = _make_mock_gateway("Summary with goal context.")

        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=3000, response_reserve=200),
            goal="debug the authentication module",
            gateway=gateway,
        )
        msgs = _make_large_messages()
        await builder.abuild_conversation_context(msgs, turn=5)

        # Check the request sent to the gateway
        gateway.generate.assert_called_once()
        call_args = gateway.generate.call_args
        request = call_args[0][0]  # First positional arg is the LLMRequest
        user_msg = request.messages[1]  # Second message is the user prompt
        assert "debug the authentication module" in (user_msg.content or "")

    @pytest.mark.asyncio
    async def test_under_budget_skips_compression(self):
        """When under budget, async version should pass through like sync."""
        gateway = _make_mock_gateway()

        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=128000),
            gateway=gateway,
        )
        msgs = _make_messages(4)
        result = await builder.abuild_conversation_context(msgs, turn=0)

        assert len(result) == len(msgs)
        assert builder.last_decision is not None
        assert not builder.last_decision.history_compressed
        # Gateway should NOT be called when under budget
        gateway.generate.assert_not_called()


# ---- Test: LLM summary respects token budget ----


class TestLLMSummaryBudget:
    @pytest.mark.asyncio
    async def test_summary_fits_within_budget(self):
        """The LLM summary message should fit within the available token budget."""
        gateway = _make_mock_gateway("Key points: user searched files, found config.py.")

        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=3000, response_reserve=200),
            gateway=gateway,
        )
        msgs = _make_large_messages()
        await builder.abuild_conversation_context(msgs, turn=5)

        assert builder.last_decision is not None
        assert builder.last_decision.budget_used <= builder.last_decision.budget_total

    @pytest.mark.asyncio
    async def test_long_summary_gets_truncated(self):
        """If the LLM returns a very long summary, it should be truncated to fit."""
        # Return a summary that is much larger than summary_budget.
        # With total_window=3000, response_reserve=200, budget=2800.
        # Head ~5 tokens, tail ~30 tokens, margin 100 -> summary_budget ~2665.
        # Use a 5000-token summary to force truncation.
        long_summary = "Important detail about the system. " * 1500  # ~13000+ tokens
        gateway = _make_mock_gateway(long_summary)

        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=3000, response_reserve=200),
            gateway=gateway,
        )
        msgs = _make_large_messages()
        result = await builder.abuild_conversation_context(msgs, turn=5)

        # The summary message should exist
        summary_msgs = [m for m in result if "LLM summary" in (m.content or "")]
        assert len(summary_msgs) == 1
        summary_tokens = estimate_tokens(summary_msgs[0].content or "")
        # It should be drastically shorter than the raw long_summary
        assert summary_tokens < estimate_tokens(long_summary) // 2


# ---- Test: fallback to keyword compression when gateway fails ----


class TestGatewayFailureFallback:
    @pytest.mark.asyncio
    async def test_gateway_exception_falls_back_to_keyword(self):
        """When the gateway raises an exception, fall back to keyword compression."""
        gateway = _make_mock_gateway()
        gateway.generate = AsyncMock(side_effect=RuntimeError("API error"))

        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=3000, response_reserve=200),
            gateway=gateway,
        )
        msgs = _make_large_messages()
        result = await builder.abuild_conversation_context(msgs, turn=5)

        # Should still produce a compressed result using keyword fallback
        assert len(result) < len(msgs)
        has_keyword_summary = any(
            "relevance-compressed" in (m.content or "") for m in result
        )
        assert has_keyword_summary
        assert builder.last_decision is not None
        assert builder.last_decision.history_compressed
        # Explanation should NOT mention LLM since it failed
        assert "LLM" not in builder.last_decision.explanation

    @pytest.mark.asyncio
    async def test_gateway_returns_empty_falls_back(self):
        """When the gateway returns empty content, the compression still works."""
        gateway = _make_mock_gateway("")  # Empty summary

        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=3000, response_reserve=200),
            gateway=gateway,
        )
        msgs = _make_large_messages()
        result = await builder.abuild_conversation_context(msgs, turn=5)

        # Should still produce a valid result (empty summary is still valid)
        assert len(result) < len(msgs)
        assert builder.last_decision is not None
        assert builder.last_decision.history_compressed


# ---- Test: threshold check ----


class TestCompressionThreshold:
    @pytest.mark.asyncio
    async def test_small_middle_uses_keyword_not_llm(self):
        """When middle section is below threshold, keyword compression is used even with gateway."""
        gateway = _make_mock_gateway("This should not be called.")

        # Use a budget that forces compression but with few/short middle messages
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=300, response_reserve=50),
            gateway=gateway,
        )
        # Create messages where middle section is small (below _LLM_COMPRESSION_THRESHOLD)
        msgs = [
            Message(role=MessageRole.SYSTEM, content="System"),
            # These will be in the middle (compressed zone)
            Message(role=MessageRole.USER, content="Short message 1"),
            Message(role=MessageRole.ASSISTANT, content="Short reply 1"),
            # Recent tail (6 messages)
            Message(role=MessageRole.USER, content="Recent " + "x" * 100),
            Message(role=MessageRole.ASSISTANT, content="Recent " + "y" * 100),
            Message(role=MessageRole.USER, content="Recent " + "x" * 100),
            Message(role=MessageRole.ASSISTANT, content="Recent " + "y" * 100),
            Message(role=MessageRole.USER, content="Recent " + "x" * 100),
            Message(role=MessageRole.ASSISTANT, content="Recent " + "y" * 100),
        ]

        await builder.abuild_conversation_context(msgs, turn=3)

        # Gateway should NOT have been called (middle too small)
        gateway.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_middle_triggers_llm(self):
        """When middle section exceeds threshold, LLM compression is triggered."""
        gateway = _make_mock_gateway("Compressed summary.")

        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=3000, response_reserve=200),
            gateway=gateway,
        )
        msgs = _make_large_messages()

        await builder.abuild_conversation_context(msgs, turn=5)

        # Gateway should have been called
        gateway.generate.assert_called_once()
        assert builder.last_decision is not None
        assert "LLM-compressed" in builder.last_decision.explanation


# ---- Test: compression model config ----


class TestCompressionModelConfig:
    @pytest.mark.asyncio
    async def test_custom_compression_model(self):
        """When compression_model is provided, it should be used for the LLM call."""
        gateway = _make_mock_gateway("Custom model summary.")
        custom_config = ModelConfig(
            provider="custom_provider",
            model_id="cheap-model-v1",
            temperature=0.0,
            max_tokens=200,
        )

        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=3000, response_reserve=200),
            gateway=gateway,
            compression_model=custom_config,
        )
        msgs = _make_large_messages()
        await builder.abuild_conversation_context(msgs, turn=5)

        # Check the config passed to gateway.generate
        gateway.generate.assert_called_once()
        call_args = gateway.generate.call_args
        config_used = call_args[0][1]  # Second positional arg is the ModelConfig
        assert config_used.provider == "custom_provider"
        assert config_used.model_id == "cheap-model-v1"
        assert config_used.temperature == 0.0

    @pytest.mark.asyncio
    async def test_default_config_derived_from_gateway(self):
        """When no compression_model is set, config is derived from gateway's default provider."""
        gateway = _make_mock_gateway("Default provider summary.")

        builder = WorkingSetBuilder(
            identity="System prompt.",
            token_budget=TokenBudget(total_window=3000, response_reserve=200),
            gateway=gateway,
        )
        msgs = _make_large_messages()
        await builder.abuild_conversation_context(msgs, turn=5)

        gateway.generate.assert_called_once()
        call_args = gateway.generate.call_args
        config_used = call_args[0][1]
        assert config_used.provider == "mock_provider"
        assert config_used.model_id == "mock-model"
        assert config_used.temperature == 0.0


# ---- Test: no gateway async still works ----


class TestAsyncWithoutGateway:
    @pytest.mark.asyncio
    async def test_abuild_without_gateway_uses_keyword(self):
        """abuild_conversation_context without gateway falls back to keyword compression."""
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=800, response_reserve=50),
        )
        msgs = _make_messages(20, content_len=200)
        result = await builder.abuild_conversation_context(msgs, turn=3)

        assert len(result) < len(msgs)
        has_keyword_summary = any(
            "relevance-compressed" in (m.content or "") for m in result
        )
        assert has_keyword_summary

    @pytest.mark.asyncio
    async def test_abuild_under_budget_passthrough(self):
        """abuild_conversation_context under budget passes through all messages."""
        builder = WorkingSetBuilder(
            identity="System",
            token_budget=TokenBudget(total_window=128000),
        )
        msgs = _make_messages(4)
        result = await builder.abuild_conversation_context(msgs, turn=0)
        assert len(result) == len(msgs)
