"""Baseline eval gate tests — CI-safe, no API keys needed.

Tests the eval infrastructure (mock provider, EvalGate runner, scoring)
and the four baseline cases:
  1. Direct answer (simple math)
  2. Tool use (calculator)
  3. Multi-turn context retention
  4. Context compression survival
"""

from __future__ import annotations

import pytest

from arcana.eval.baseline import EvalCase, EvalGate, EvalResult, build_baseline_cases
from arcana.eval.mock_provider import MockProvider

# ---------------------------------------------------------------------------
# Mock provider unit tests
# ---------------------------------------------------------------------------


class TestMockProvider:
    """Verify the mock provider is deterministic and protocol-compliant."""

    @pytest.mark.asyncio
    async def test_deterministic_response(self):
        """Same input always gives same output."""
        from arcana.contracts.llm import LLMRequest, Message, MessageRole, ModelConfig

        mock = MockProvider()
        config = ModelConfig(provider="mock", model_id="mock-v1")
        request = LLMRequest(messages=[
            Message(role=MessageRole.USER, content="What is 2+2?"),
        ])

        r1 = await mock.generate(request, config)
        r2 = await mock.generate(request, config)
        assert r1.content == r2.content
        assert "4" in (r1.content or "")

    @pytest.mark.asyncio
    async def test_call_count_tracking(self):
        from arcana.contracts.llm import LLMRequest, Message, MessageRole, ModelConfig

        mock = MockProvider()
        config = ModelConfig(provider="mock", model_id="mock-v1")
        request = LLMRequest(messages=[
            Message(role=MessageRole.USER, content="Hello"),
        ])

        assert mock.call_count == 0
        await mock.generate(request, config)
        assert mock.call_count == 1
        await mock.generate(request, config)
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        """When tools are available and pattern matches, return tool calls."""
        from arcana.contracts.llm import LLMRequest, Message, MessageRole, ModelConfig

        mock = MockProvider()
        config = ModelConfig(provider="mock", model_id="mock-v1")
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.USER, content="Calculate 5+3"),
            ],
            tools=[{
                "type": "function",
                "function": {"name": "calculator", "description": "calc", "parameters": {}},
            }],
        )

        response = await mock.generate(request, config)
        assert response.tool_calls is not None
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0].name == "calculator"

    @pytest.mark.asyncio
    async def test_custom_rule(self):
        from arcana.contracts.llm import LLMRequest, Message, MessageRole, ModelConfig

        mock = MockProvider()
        mock.add_response_rule(r"weather", "It's sunny today.")
        config = ModelConfig(provider="mock", model_id="mock-v1")
        request = LLMRequest(messages=[
            Message(role=MessageRole.USER, content="What's the weather?"),
        ])

        response = await mock.generate(request, config)
        assert "sunny" in (response.content or "")

    @pytest.mark.asyncio
    async def test_stream_protocol(self):
        """Stream should yield chunks and end with done."""
        from arcana.contracts.llm import LLMRequest, Message, MessageRole, ModelConfig

        mock = MockProvider()
        config = ModelConfig(provider="mock", model_id="mock-v1")
        request = LLMRequest(messages=[
            Message(role=MessageRole.USER, content="Hello"),
        ])

        chunks = []
        async for chunk in mock.stream(request, config):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[-1].type == "done"

    @pytest.mark.asyncio
    async def test_health_check(self):
        mock = MockProvider()
        assert await mock.health_check() is True

    @pytest.mark.asyncio
    async def test_provider_properties(self):
        mock = MockProvider()
        assert mock.provider_name == "mock"
        assert "mock-v1" in mock.supported_models


# ---------------------------------------------------------------------------
# EvalGate unit tests
# ---------------------------------------------------------------------------


class TestEvalGate:
    """Verify the eval gate runner and reporting."""

    @pytest.mark.asyncio
    async def test_empty_suite(self):
        gate = EvalGate()
        report = await gate.run_all()
        assert report.total == 0
        assert report.score == 0.0

    @pytest.mark.asyncio
    async def test_all_pass(self):
        gate = EvalGate()

        async def _passing() -> EvalResult:
            return EvalResult(passed=True, score=1.0, detail="ok")

        gate.register(EvalCase(name="a", category="cat1", run=_passing))
        gate.register(EvalCase(name="b", category="cat1", run=_passing))

        report = await gate.run_all()
        assert report.total == 2
        assert report.passed == 2
        assert report.failed == 0
        assert report.score == 1.0
        assert report.by_category["cat1"] == 1.0

    @pytest.mark.asyncio
    async def test_mixed_results(self):
        gate = EvalGate()

        async def _pass() -> EvalResult:
            return EvalResult(passed=True, score=1.0, detail="ok")

        async def _fail() -> EvalResult:
            return EvalResult(passed=False, score=0.0, detail="nope")

        gate.register(EvalCase(name="p", category="a", run=_pass))
        gate.register(EvalCase(name="f", category="b", run=_fail))

        report = await gate.run_all()
        assert report.total == 2
        assert report.passed == 1
        assert report.failed == 1
        assert report.score == 0.5
        assert report.by_category["a"] == 1.0
        assert report.by_category["b"] == 0.0

    @pytest.mark.asyncio
    async def test_category_filter(self):
        gate = EvalGate()

        async def _pass() -> EvalResult:
            return EvalResult(passed=True, score=1.0, detail="ok")

        async def _fail() -> EvalResult:
            return EvalResult(passed=False, score=0.0, detail="nope")

        gate.register(EvalCase(name="p", category="good", run=_pass))
        gate.register(EvalCase(name="f", category="bad", run=_fail))

        report = await gate.run_all(category="good")
        assert report.total == 1
        assert report.passed == 1
        assert report.score == 1.0

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Cases that raise exceptions should be marked as failed."""
        gate = EvalGate()

        async def _boom() -> EvalResult:
            raise RuntimeError("boom")

        gate.register(EvalCase(name="err", category="x", run=_boom))

        report = await gate.run_all()
        assert report.total == 1
        assert report.failed == 1
        assert "Exception" in report.results[0][1].detail

    @pytest.mark.asyncio
    async def test_duration_tracking(self):
        import asyncio

        gate = EvalGate()

        async def _slow() -> EvalResult:
            await asyncio.sleep(0.05)
            return EvalResult(passed=True, score=1.0, detail="slow")

        gate.register(EvalCase(name="slow", category="perf", run=_slow))

        report = await gate.run_all()
        assert report.results[0][1].duration_ms >= 40  # at least ~50ms


# ---------------------------------------------------------------------------
# Baseline eval cases — integration tests with mock provider
# ---------------------------------------------------------------------------


class TestBaselineCases:
    """Run each baseline eval case individually."""

    @pytest.mark.asyncio
    async def test_direct_answer(self):
        cases = build_baseline_cases()
        case = next(c for c in cases if c.name == "direct_answer_math")
        result = await case.run()
        assert result.passed, f"Direct answer failed: {result.detail}"
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_tool_use(self):
        cases = build_baseline_cases()
        case = next(c for c in cases if c.name == "tool_use_calculator")
        result = await case.run()
        assert result.passed, f"Tool use failed: {result.detail}"
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_multi_turn_context(self):
        cases = build_baseline_cases()
        case = next(c for c in cases if c.name == "multi_turn_context")
        result = await case.run()
        assert result.passed, f"Multi-turn failed: {result.detail}"

    @pytest.mark.asyncio
    async def test_context_compression(self):
        cases = build_baseline_cases()
        case = next(c for c in cases if c.name == "context_compression")
        result = await case.run()
        assert result.passed, f"Context compression failed: {result.detail}"


class TestFullSuite:
    """Run the complete baseline suite through EvalGate."""

    @pytest.mark.asyncio
    async def test_full_baseline_suite(self):
        gate = EvalGate()
        for case in build_baseline_cases():
            gate.register(case)

        report = await gate.run_all()

        assert report.total == 4
        assert report.score >= 0.75, f"Baseline suite score too low: {report.score}"
        # All categories should be represented
        assert "direct_answer" in report.by_category
        assert "tool_use" in report.by_category
        assert "multi_turn" in report.by_category
        assert "context" in report.by_category

    @pytest.mark.asyncio
    async def test_category_filter_integration(self):
        gate = EvalGate()
        for case in build_baseline_cases():
            gate.register(case)

        report = await gate.run_all(category="direct_answer")
        assert report.total == 1
        assert report.results[0][0] == "direct_answer_math"
