"""Tests for thinking-informed turn assessment.

When the LLM uses extended thinking, its internal reasoning (uncertainty,
verification intent, incomplete info signals) should influence the
TurnAssessment produced by _assess_turn.
"""

from __future__ import annotations

import importlib

import pytest

from arcana.contracts.llm import ToolCallRequest
from arcana.contracts.state import AgentState
from arcana.contracts.turn import TurnFacts

_conversation_available = importlib.util.find_spec("arcana.runtime.conversation") is not None

if _conversation_available:
    from arcana.runtime.conversation import ConversationAgent

needs_conversation = pytest.mark.skipif(
    not _conversation_available,
    reason="arcana.runtime.conversation not implemented yet",
)


@needs_conversation
class TestThinkingInformedAssessment:
    """Thinking content should adjust confidence and completion of Rule 3."""

    @staticmethod
    def _state() -> AgentState:
        return AgentState(run_id="test-thinking", current_step=1)

    # ------------------------------------------------------------------
    # Backward compatibility: no thinking → behavior unchanged
    # ------------------------------------------------------------------

    def test_no_thinking_behavior_unchanged(self) -> None:
        """Without thinking, Rule 3 should behave exactly as before."""
        facts = TurnFacts(
            assistant_text="The answer is 42.",
            finish_reason="stop",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.answer == "The answer is 42."
        assert assessment.completion_reason == "natural_stop"
        assert assessment.confidence == pytest.approx(0.85)

    # ------------------------------------------------------------------
    # Uncertainty signals → lower confidence, still completed
    # ------------------------------------------------------------------

    def test_uncertainty_lowers_confidence(self) -> None:
        facts = TurnFacts(
            assistant_text="I think it's Paris.",
            finish_reason="stop",
            thinking="I'm not sure about this. The user asked about a capital city.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85 * 0.6)

    def test_uncertainty_pattern_uncertain(self) -> None:
        facts = TurnFacts(
            assistant_text="Maybe Berlin.",
            finish_reason="stop",
            thinking="This is uncertain, I recall different answers.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence < 0.85

    def test_uncertainty_pattern_dont_know(self) -> None:
        facts = TurnFacts(
            assistant_text="Could be either.",
            finish_reason="stop",
            thinking="I don't know the exact answer here.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85 * 0.6)

    # ------------------------------------------------------------------
    # Verification intent → NOT completed
    # ------------------------------------------------------------------

    def test_verification_intent_not_completed(self) -> None:
        facts = TurnFacts(
            assistant_text="Let me check the database.",
            finish_reason="stop",
            thinking="I need to verify this against the actual data before answering.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is False
        assert assessment.completion_reason == "thinking_wants_verification"

    def test_verification_should_check(self) -> None:
        facts = TurnFacts(
            assistant_text="The file should exist.",
            finish_reason="stop",
            thinking="I should check whether this file really exists.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is False
        assert assessment.completion_reason == "thinking_wants_verification"

    def test_verification_double_check(self) -> None:
        facts = TurnFacts(
            assistant_text="Here's the config.",
            finish_reason="stop",
            thinking="Wait, I need to double-check the port number.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is False
        assert assessment.completion_reason == "thinking_wants_verification"

    # ------------------------------------------------------------------
    # Incomplete information → lower confidence, still completed
    # ------------------------------------------------------------------

    def test_incomplete_info_lowers_confidence(self) -> None:
        facts = TurnFacts(
            assistant_text="Based on what I have, the answer is X.",
            finish_reason="stop",
            thinking="I need more information to give a definitive answer.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85 * 0.5)

    def test_incomplete_missing_data(self) -> None:
        facts = TurnFacts(
            assistant_text="Here's my best guess.",
            finish_reason="stop",
            thinking="There's missing data in the user's request.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85 * 0.5)

    # ------------------------------------------------------------------
    # Combined: uncertainty + incomplete → both multipliers apply
    # ------------------------------------------------------------------

    def test_uncertainty_and_incomplete_stack(self) -> None:
        facts = TurnFacts(
            assistant_text="My best guess.",
            finish_reason="stop",
            thinking="I'm not sure and I need more information to be precise.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        # 0.85 * 0.6 (uncertainty) * 0.5 (incomplete) = 0.255
        assert assessment.confidence == pytest.approx(0.85 * 0.6 * 0.5)

    # ------------------------------------------------------------------
    # Verification takes priority over uncertainty
    # ------------------------------------------------------------------

    def test_verification_priority_over_uncertainty(self) -> None:
        """If both uncertainty and verification are present, verification wins."""
        facts = TurnFacts(
            assistant_text="Not sure yet.",
            finish_reason="stop",
            thinking="I'm not sure about this. I should check the docs first.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        # Verification pattern found → not completed (returns early)
        assert assessment.completed is False
        assert assessment.completion_reason == "thinking_wants_verification"

    # ------------------------------------------------------------------
    # Thinking with positive/neutral content → confidence unchanged
    # ------------------------------------------------------------------

    def test_positive_thinking_confidence_unchanged(self) -> None:
        """Thinking that contains no negative signals should not alter confidence."""
        facts = TurnFacts(
            assistant_text="Paris is the capital of France.",
            finish_reason="stop",
            thinking="The user asked about France's capital. That's clearly Paris.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85)
        assert assessment.completion_reason == "natural_stop"

    def test_neutral_thinking_confidence_unchanged(self) -> None:
        facts = TurnFacts(
            assistant_text="The function returns a list.",
            finish_reason="stop",
            thinking="Let me analyze this code. The return type is list[str].",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85)

    # ------------------------------------------------------------------
    # Chinese language patterns
    # ------------------------------------------------------------------

    def test_chinese_uncertainty(self) -> None:
        facts = TurnFacts(
            assistant_text="可能是北京。",
            finish_reason="stop",
            thinking="我不确定这个问题的答案。",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85 * 0.6)

    def test_chinese_uncertainty_alternate(self) -> None:
        facts = TurnFacts(
            assistant_text="答案可能是A。",
            finish_reason="stop",
            thinking="这个答案可能不对，但我先回答了。",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence < 0.85

    def test_chinese_uncertainty_not_very_sure(self) -> None:
        facts = TurnFacts(
            assistant_text="我觉得是这样。",
            finish_reason="stop",
            thinking="我不太确定，但倾向于这个答案。",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85 * 0.6)

    def test_chinese_verification(self) -> None:
        facts = TurnFacts(
            assistant_text="让我看看。",
            finish_reason="stop",
            thinking="这个数据需要验证一下。",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is False
        assert assessment.completion_reason == "thinking_wants_verification"

    def test_chinese_verification_confirm(self) -> None:
        facts = TurnFacts(
            assistant_text="配置如下。",
            finish_reason="stop",
            thinking="应该确认一下端口号。",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is False
        assert assessment.completion_reason == "thinking_wants_verification"

    def test_chinese_incomplete(self) -> None:
        facts = TurnFacts(
            assistant_text="根据已有信息，答案是X。",
            finish_reason="stop",
            thinking="信息不足，无法给出完整答案。",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85 * 0.5)

    def test_chinese_need_more(self) -> None:
        facts = TurnFacts(
            assistant_text="请提供更多信息。",
            finish_reason="stop",
            thinking="需要更多上下文才能准确回答。",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85 * 0.5)

    # ------------------------------------------------------------------
    # Thinking should NOT affect other rules
    # ------------------------------------------------------------------

    def test_thinking_does_not_affect_tool_calls(self) -> None:
        """Rule 1: tool calls → not done, regardless of thinking."""
        tc = ToolCallRequest(id="1", name="search", arguments='{}')
        facts = TurnFacts(
            tool_calls=[tc],
            finish_reason="tool_calls",
            thinking="I'm not sure but let me search.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert not assessment.completed
        assert not assessment.failed

    def test_thinking_does_not_affect_empty_response(self) -> None:
        """Rule 2: no text → failed, regardless of thinking."""
        facts = TurnFacts(
            finish_reason="stop",
            thinking="I had some thoughts but no output.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.failed is True

    def test_thinking_does_not_affect_length_finish(self) -> None:
        """Rule 4: finish_reason=length → not done, regardless of thinking."""
        facts = TurnFacts(
            assistant_text="Partial answer...",
            finish_reason="length",
            thinking="I was confident about this answer.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert not assessment.completed
        assert not assessment.failed

    # ------------------------------------------------------------------
    # Case insensitivity
    # ------------------------------------------------------------------

    def test_case_insensitive_matching(self) -> None:
        """Pattern matching should be case insensitive."""
        facts = TurnFacts(
            assistant_text="Some answer.",
            finish_reason="stop",
            thinking="I'm NOT SURE about this AT ALL.",
        )
        assessment = ConversationAgent._assess_turn(facts, self._state())
        assert assessment.completed is True
        assert assessment.confidence == pytest.approx(0.85 * 0.6)
