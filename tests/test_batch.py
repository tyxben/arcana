"""Tests for batch execution at provider, registry, and runtime levels."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.contracts.llm import (
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    TokenUsage,
)
from arcana.gateway.base import ProviderError
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.runtime_core import (
    BatchResult,
    RunResult,
    Runtime,
    RuntimeConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(content: str = "hello") -> LLMRequest:
    return LLMRequest(messages=[Message(role=MessageRole.USER, content=content)])


def _make_response(text: str = "world", tokens: int = 10) -> LLMResponse:
    return LLMResponse(
        content=text,
        usage=TokenUsage(prompt_tokens=tokens, completion_tokens=tokens, total_tokens=tokens * 2),
        model="test-model",
        finish_reason="stop",
    )


def _make_config() -> ModelConfig:
    return ModelConfig(provider="test", model_id="test-model")


# ---------------------------------------------------------------------------
# Provider-level batch_generate
# ---------------------------------------------------------------------------


class TestProviderBatchGenerate:
    """Tests for OpenAICompatibleProvider.batch_generate."""

    @pytest.mark.asyncio
    async def test_batch_generates_all_requests(self):
        """batch_generate runs N requests and returns N responses."""
        from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_key="test-key",
            base_url="http://localhost:1234",
        )

        responses = [_make_response(f"response-{i}") for i in range(5)]
        call_count = 0

        async def mock_generate(request, config, trace_ctx=None):
            nonlocal call_count
            idx = call_count
            call_count += 1
            return responses[idx]

        provider.generate = mock_generate  # type: ignore[assignment]

        requests = [_make_request(f"request-{i}") for i in range(5)]
        config = _make_config()
        results = await provider.batch_generate(requests, config)

        assert len(results) == 5
        assert call_count == 5
        for i, result in enumerate(results):
            assert isinstance(result, LLMResponse)
            assert result.content == f"response-{i}"

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        """batch_generate with empty list returns empty list."""
        from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_key="test-key",
            base_url="http://localhost:1234",
        )
        results = await provider.batch_generate([], _make_config())
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_respects_concurrency_limit(self):
        """batch_generate never exceeds the concurrency limit."""
        from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_key="test-key",
            base_url="http://localhost:1234",
        )

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_generate(request, config, trace_ctx=None):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.01)  # Simulate API latency
            async with lock:
                current_concurrent -= 1
            return _make_response("ok")

        provider.generate = mock_generate  # type: ignore[assignment]

        requests = [_make_request(f"req-{i}") for i in range(10)]
        concurrency = 3
        results = await provider.batch_generate(
            requests, _make_config(), concurrency=concurrency,
        )

        assert len(results) == 10
        assert max_concurrent <= concurrency

    @pytest.mark.asyncio
    async def test_batch_individual_failure_returns_error(self):
        """Failed requests return ProviderError, others still succeed."""
        from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_key="test-key",
            base_url="http://localhost:1234",
        )

        call_index = 0

        async def mock_generate(request, config, trace_ctx=None):
            nonlocal call_index
            idx = call_index
            call_index += 1
            if idx == 2:
                raise ProviderError("Rate limit", provider="test", retryable=True)
            return _make_response(f"ok-{idx}")

        provider.generate = mock_generate  # type: ignore[assignment]

        requests = [_make_request(f"req-{i}") for i in range(5)]
        results = await provider.batch_generate(requests, _make_config())

        assert len(results) == 5
        # Requests 0, 1, 3, 4 should succeed
        succeeded = [r for r in results if isinstance(r, LLMResponse)]
        failed = [r for r in results if isinstance(r, ProviderError)]
        assert len(succeeded) == 4
        assert len(failed) == 1
        assert "Rate limit" in str(failed[0])

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self):
        """Results are in the same order as requests, even with concurrency."""
        from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_key="test-key",
            base_url="http://localhost:1234",
        )

        async def mock_generate(request, config, trace_ctx=None):
            # Extract request content to verify ordering
            content = request.messages[0].content
            # Variable delays to test ordering
            delay = 0.01 if "3" in content else 0.001
            await asyncio.sleep(delay)
            return _make_response(f"reply-to-{content}")

        provider.generate = mock_generate  # type: ignore[assignment]

        requests = [_make_request(f"msg-{i}") for i in range(5)]
        results = await provider.batch_generate(requests, _make_config(), concurrency=5)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert isinstance(result, LLMResponse)
            assert result.content == f"reply-to-msg-{i}"


# ---------------------------------------------------------------------------
# Registry-level batch_generate
# ---------------------------------------------------------------------------


class TestRegistryBatchGenerate:
    """Tests for ModelGatewayRegistry.batch_generate."""

    @pytest.mark.asyncio
    async def test_delegates_to_provider_batch(self):
        """Registry delegates to provider's batch_generate when available."""
        registry = ModelGatewayRegistry()

        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        expected_results = [_make_response("a"), _make_response("b")]
        mock_provider.batch_generate = AsyncMock(return_value=expected_results)

        registry.register("test", mock_provider)
        registry.set_default("test")

        requests = [_make_request("x"), _make_request("y")]
        config = _make_config()
        results = await registry.batch_generate(requests, config, concurrency=3)

        assert results == expected_results
        mock_provider.batch_generate.assert_called_once_with(
            requests, config, concurrency=3, trace_ctx=None,
        )

    @pytest.mark.asyncio
    async def test_fallback_to_generate_when_no_batch(self):
        """Registry uses generate() with semaphore when no batch_generate."""
        registry = ModelGatewayRegistry()

        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        # Remove batch_generate so fallback path is used
        del mock_provider.batch_generate
        mock_provider.generate = AsyncMock(return_value=_make_response("ok"))

        registry.register("test", mock_provider)
        registry.set_default("test")

        requests = [_make_request(f"req-{i}") for i in range(3)]
        config = _make_config()
        results = await registry.batch_generate(requests, config)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, LLMResponse)
            assert r.content == "ok"

    @pytest.mark.asyncio
    async def test_registry_batch_empty_list(self):
        """Registry batch with empty list returns empty."""
        registry = ModelGatewayRegistry()
        # No provider needed for empty list
        results = await registry.batch_generate([], _make_config())
        assert results == []

    @pytest.mark.asyncio
    async def test_registry_batch_failure_isolation(self):
        """Failures in registry fallback path return ProviderError."""
        registry = ModelGatewayRegistry()

        call_count = 0

        async def mock_generate(request, config, trace_ctx=None, use_fallback=True):
            nonlocal call_count
            idx = call_count
            call_count += 1
            if idx == 1:
                raise ProviderError("Server error", provider="test", retryable=False)
            return _make_response(f"ok-{idx}")

        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        del mock_provider.batch_generate  # Force fallback path

        registry.register("test", mock_provider)
        registry.set_default("test")

        # Override generate on the registry itself for the fallback path
        registry.generate = mock_generate  # type: ignore[assignment]

        requests = [_make_request(f"req-{i}") for i in range(3)]
        config = _make_config()
        results = await registry.batch_generate(requests, config)

        assert len(results) == 3
        succeeded = [r for r in results if isinstance(r, LLMResponse)]
        failed = [r for r in results if isinstance(r, ProviderError)]
        assert len(succeeded) == 2
        assert len(failed) == 1


# ---------------------------------------------------------------------------
# Runtime.run_batch
# ---------------------------------------------------------------------------


class TestRuntimeRunBatch:
    """Tests for Runtime.run_batch."""

    @pytest.mark.asyncio
    async def test_run_batch_multiple_goals(self):
        """run_batch processes multiple goals concurrently."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        call_count = 0
        goals_seen: list[str] = []

        async def mock_run(goal, **kwargs):
            nonlocal call_count
            call_count += 1
            goals_seen.append(goal)
            return RunResult(
                output=f"result-for-{goal}",
                success=True,
                tokens_used=50,
                cost_usd=0.001,
                run_id=f"run-{call_count}",
            )

        rt.run = mock_run  # type: ignore[assignment]

        tasks = [
            {"goal": "Summarize doc A"},
            {"goal": "Summarize doc B"},
            {"goal": "Summarize doc C"},
        ]
        batch_result = await rt.run_batch(tasks, concurrency=3)

        assert isinstance(batch_result, BatchResult)
        assert len(batch_result.results) == 3
        assert batch_result.succeeded == 3
        assert batch_result.failed == 0
        assert batch_result.total_tokens == 150
        assert batch_result.total_cost_usd == pytest.approx(0.003)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_run_batch_empty(self):
        """run_batch with empty task list returns empty BatchResult."""
        rt = Runtime()
        batch_result = await rt.run_batch([])
        assert isinstance(batch_result, BatchResult)
        assert batch_result.results == []
        assert batch_result.succeeded == 0
        assert batch_result.failed == 0

    @pytest.mark.asyncio
    async def test_run_batch_individual_failure(self):
        """Individual failures don't crash the batch."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        call_count = 0

        async def mock_run(goal, **kwargs):
            nonlocal call_count
            call_count += 1
            if "fail" in goal:
                raise RuntimeError("Simulated failure")
            return RunResult(
                output=f"ok-{goal}",
                success=True,
                tokens_used=50,
                cost_usd=0.001,
                run_id=f"run-{call_count}",
            )

        rt.run = mock_run  # type: ignore[assignment]

        tasks = [
            {"goal": "normal task 1"},
            {"goal": "this should fail"},
            {"goal": "normal task 2"},
        ]
        batch_result = await rt.run_batch(tasks)

        assert len(batch_result.results) == 3
        assert batch_result.succeeded == 2
        assert batch_result.failed == 1

        # Check the failed result
        failed_result = batch_result.results[1]
        assert not failed_result.success
        assert "Simulated failure" in str(failed_result.output)

        # Successful results are intact
        assert batch_result.results[0].success
        assert batch_result.results[2].success

    @pytest.mark.asyncio
    async def test_run_batch_respects_concurrency(self):
        """run_batch never exceeds the concurrency limit."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_run(goal, **kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return RunResult(
                output="ok", success=True, tokens_used=10, cost_usd=0.001,
            )

        rt.run = mock_run  # type: ignore[assignment]

        tasks = [{"goal": f"task-{i}"} for i in range(10)]
        batch_result = await rt.run_batch(tasks, concurrency=2)

        assert len(batch_result.results) == 10
        assert max_concurrent <= 2
        assert batch_result.succeeded == 10

    @pytest.mark.asyncio
    async def test_run_batch_passes_kwargs(self):
        """run_batch forwards task kwargs to run()."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        captured_kwargs: list[dict] = []

        async def mock_run(goal, **kwargs):
            captured_kwargs.append({"goal": goal, **kwargs})
            return RunResult(output="ok", success=True, tokens_used=10, cost_usd=0.001)

        rt.run = mock_run  # type: ignore[assignment]

        tasks = [
            {"goal": "Task A", "system": "You are a helper", "provider": "openai"},
            {"goal": "Task B", "model": "gpt-4o"},
        ]
        await rt.run_batch(tasks)

        assert len(captured_kwargs) == 2
        # Find the task A kwargs (order may vary due to concurrency)
        task_a = next(k for k in captured_kwargs if k["goal"] == "Task A")
        task_b = next(k for k in captured_kwargs if k["goal"] == "Task B")

        assert task_a["system"] == "You are a helper"
        assert task_a["provider"] == "openai"
        assert task_b["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_run_batch_does_not_mutate_input(self):
        """run_batch does not mutate the caller's task dicts."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        async def mock_run(goal, **kwargs):
            return RunResult(output="ok", success=True, tokens_used=10, cost_usd=0.001)

        rt.run = mock_run  # type: ignore[assignment]

        original_task = {"goal": "my task", "system": "be nice"}
        tasks = [original_task]
        await rt.run_batch(tasks)

        # Original dict should be untouched
        assert "goal" in original_task
        assert original_task["goal"] == "my task"
        assert original_task["system"] == "be nice"

    @pytest.mark.asyncio
    async def test_run_batch_budget_accumulates(self):
        """Budget tracking works across batch results."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        async def mock_run(goal, **kwargs):
            return RunResult(
                output="ok", success=True,
                tokens_used=100, cost_usd=0.01,
            )

        rt.run = mock_run  # type: ignore[assignment]

        tasks = [{"goal": f"task-{i}"} for i in range(5)]
        batch_result = await rt.run_batch(tasks)

        assert batch_result.total_tokens == 500
        assert batch_result.total_cost_usd == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_run_batch_preserves_order(self):
        """Results are in the same order as tasks, even with varied delays."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        async def mock_run(goal, **kwargs):
            # Tasks with higher index complete faster
            delay = 0.01 if "0" in goal else 0.001
            await asyncio.sleep(delay)
            return RunResult(
                output=f"result-{goal}", success=True,
                tokens_used=10, cost_usd=0.001,
            )

        rt.run = mock_run  # type: ignore[assignment]

        tasks = [{"goal": f"task-{i}"} for i in range(5)]
        batch_result = await rt.run_batch(tasks, concurrency=5)

        for i, result in enumerate(batch_result.results):
            assert result.output == f"result-task-{i}"


# ---------------------------------------------------------------------------
# BatchResult model
# ---------------------------------------------------------------------------


class TestBatchResult:
    def test_defaults(self):
        b = BatchResult()
        assert b.results == []
        assert b.total_tokens == 0
        assert b.total_cost_usd == 0.0
        assert b.succeeded == 0
        assert b.failed == 0

    def test_populated(self):
        b = BatchResult(
            results=[RunResult(output="a", success=True)],
            total_tokens=100,
            total_cost_usd=0.01,
            succeeded=1,
            failed=0,
        )
        assert len(b.results) == 1
        assert b.total_tokens == 100
        assert b.succeeded == 1
