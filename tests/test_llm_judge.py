"""Tests for LLMJudge — LLM-as-judge evaluation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.contracts.llm import LLMResponse, ModelConfig, TokenUsage
from arcana.eval.llm_judge import CriterionScore, JudgeResult, LLMJudge

# ── Helpers ──────────────────────────────────────────────────────────


def _make_gateway(content: str, usage: TokenUsage | None = None) -> MagicMock:
    """Create a mock gateway that returns a fixed response."""
    gateway = MagicMock()
    response = LLMResponse(
        content=content,
        usage=usage or TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        model="test-model",
        finish_reason="stop",
    )
    gateway.generate = AsyncMock(return_value=response)
    return gateway


def _make_judge_response(
    scores: list[tuple[str, int, str]],
    overall_reasoning: str = "Good output.",
) -> str:
    """Build a valid JSON response string."""
    return json.dumps({
        "scores": [
            {"criterion": c, "score": s, "reasoning": r}
            for c, s, r in scores
        ],
        "overall_reasoning": overall_reasoning,
    })


def _config() -> ModelConfig:
    return ModelConfig(provider="test", model_id="test-model")


# ── Core Evaluation Tests ────────────────────────────────────────────


class TestLLMJudgeEvaluate:
    @pytest.mark.asyncio
    async def test_basic_scoring(self):
        content = _make_judge_response([
            ("accuracy", 5, "Perfectly accurate"),
            ("completeness", 4, "Mostly complete"),
        ])
        gateway = _make_gateway(content)
        judge = LLMJudge(gateway, _config())

        result = await judge.evaluate(
            goal="Summarize the document",
            output="This is a summary.",
            criteria=["accuracy", "completeness"],
        )

        assert isinstance(result, JudgeResult)
        assert len(result.scores) == 2
        assert result.overall_score == 4.5
        assert result.passed is True
        assert result.error is None
        gateway.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failing_score(self):
        content = _make_judge_response([
            ("accuracy", 1, "Inaccurate"),
            ("completeness", 2, "Incomplete"),
        ])
        gateway = _make_gateway(content)
        judge = LLMJudge(gateway, _config(), pass_threshold=3.0)

        result = await judge.evaluate(
            goal="Summarize",
            output="Wrong.",
            criteria=["accuracy", "completeness"],
        )

        assert result.overall_score == 1.5
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_empty_criteria(self):
        gateway = _make_gateway("")
        judge = LLMJudge(gateway, _config())

        result = await judge.evaluate(
            goal="Do something",
            output="Done.",
            criteria=[],
        )

        assert result.passed is True
        assert result.overall_score == 5.0
        gateway.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_custom_pass_threshold(self):
        content = _make_judge_response([
            ("quality", 4, "Good"),
        ])
        gateway = _make_gateway(content)
        judge = LLMJudge(gateway, _config(), pass_threshold=4.5)

        result = await judge.evaluate(
            goal="Write well",
            output="Content.",
            criteria=["quality"],
        )

        assert result.overall_score == 4.0
        assert result.passed is False  # 4.0 < 4.5

    @pytest.mark.asyncio
    async def test_usage_propagated(self):
        content = _make_judge_response([("clarity", 3, "OK")])
        usage = TokenUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280)
        gateway = _make_gateway(content, usage=usage)
        judge = LLMJudge(gateway, _config())

        result = await judge.evaluate(
            goal="Be clear",
            output="Output.",
            criteria=["clarity"],
        )

        assert result.usage.total_tokens == 280


# ── Error Handling Tests ─────────────────────────────────────────────


class TestLLMJudgeErrorHandling:
    @pytest.mark.asyncio
    async def test_gateway_exception(self):
        gateway = MagicMock()
        gateway.generate = AsyncMock(side_effect=RuntimeError("connection failed"))
        judge = LLMJudge(gateway, _config())

        result = await judge.evaluate(
            goal="Test",
            output="Output.",
            criteria=["accuracy"],
        )

        assert result.error is not None
        assert "connection failed" in result.error
        assert result.passed is False
        assert result.scores == []

    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        gateway = _make_gateway("This is not JSON at all")
        judge = LLMJudge(gateway, _config())

        result = await judge.evaluate(
            goal="Test",
            output="Output.",
            criteria=["accuracy"],
        )

        assert result.error is not None
        assert "JSON" in result.error
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_json_in_markdown_fence(self):
        raw_json = _make_judge_response([("accuracy", 5, "Perfect")])
        fenced = f"```json\n{raw_json}\n```"
        gateway = _make_gateway(fenced)
        judge = LLMJudge(gateway, _config())

        result = await judge.evaluate(
            goal="Test",
            output="Output.",
            criteria=["accuracy"],
        )

        assert result.error is None
        assert len(result.scores) == 1
        assert result.scores[0].score == 5

    @pytest.mark.asyncio
    async def test_malformed_score_entries_skipped(self):
        content = json.dumps({
            "scores": [
                {"criterion": "good", "score": 4, "reasoning": "OK"},
                {"criterion": "bad", "score": "not_a_number", "reasoning": "x"},
                {"criterion": "ok", "score": 3, "reasoning": "Fine"},
            ],
            "overall_reasoning": "Mixed.",
        })
        gateway = _make_gateway(content)
        judge = LLMJudge(gateway, _config())

        result = await judge.evaluate(
            goal="Test",
            output="Output.",
            criteria=["good", "bad", "ok"],
        )

        # Malformed entry skipped, 2 valid scores remain
        assert len(result.scores) == 2
        assert result.overall_score == 3.5

    @pytest.mark.asyncio
    async def test_empty_scores_array(self):
        content = json.dumps({"scores": [], "overall_reasoning": "Nothing."})
        gateway = _make_gateway(content)
        judge = LLMJudge(gateway, _config())

        result = await judge.evaluate(
            goal="Test",
            output="Output.",
            criteria=["accuracy"],
        )

        assert result.overall_score == 0.0
        assert result.passed is False
        assert result.scores == []


# ── Model Tests ──────────────────────────────────────────────────────


class TestJudgeModels:
    def test_criterion_score_validation(self):
        score = CriterionScore(criterion="test", score=3, reasoning="OK")
        assert score.score == 3

    def test_criterion_score_bounds(self):
        with pytest.raises(ValueError):  # Pydantic validation
            CriterionScore(criterion="test", score=0, reasoning="too low")

        with pytest.raises(ValueError):
            CriterionScore(criterion="test", score=6, reasoning="too high")

    def test_judge_result_defaults(self):
        result = JudgeResult()
        assert result.scores == []
        assert result.overall_score == 0.0
        assert result.passed is False
        assert result.error is None

    def test_judge_result_serialization(self):
        result = JudgeResult(
            scores=[CriterionScore(criterion="a", score=4, reasoning="good")],
            overall_score=4.0,
            passed=True,
            overall_reasoning="Solid.",
        )
        data = result.model_dump()
        restored = JudgeResult.model_validate(data)
        assert restored == result


# ── Default Config Tests ─────────────────────────────────────────────


class TestLLMJudgeConfig:
    def test_default_model_config(self):
        gateway = MagicMock()
        judge = LLMJudge(gateway)

        assert judge.model_config.temperature == 0.0
        assert judge.pass_threshold == 3.0

    def test_custom_model_config(self):
        gateway = MagicMock()
        config = ModelConfig(provider="openai", model_id="gpt-4o", temperature=0.2)
        judge = LLMJudge(gateway, config, pass_threshold=4.0)

        assert judge.model_config.provider == "openai"
        assert judge.pass_threshold == 4.0


# ── Export Tests ─────────────────────────────────────────────────────


class TestExports:
    def test_llm_judge_exported(self):
        import arcana.eval as ev

        assert hasattr(ev, "LLMJudge")
        assert hasattr(ev, "JudgeResult")
        assert hasattr(ev, "CriterionScore")
