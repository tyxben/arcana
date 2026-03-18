"""Tests for Intent Router -- classifiers and executor."""

import pytest

from arcana.contracts.intent import IntentClassification, IntentType
from arcana.routing.classifier import RuleBasedClassifier


class TestRuleBasedClassifier:
    async def test_simple_question(self):
        c = RuleBasedClassifier()
        r = await c.classify("What is Python?")
        assert r.intent == IntentType.DIRECT_ANSWER

    async def test_tool_trigger(self):
        c = RuleBasedClassifier()
        r = await c.classify("search for quantum computing", available_tools=["web_search"])
        assert r.intent == IntentType.SINGLE_TOOL
        assert "web_search" in r.suggested_tools

    async def test_tool_trigger_no_available(self):
        """Tool trigger without matching available tool does not route to SINGLE_TOOL."""
        c = RuleBasedClassifier()
        r = await c.classify("search for something", available_tools=["calculator"])
        # search tools not in available_tools, shouldn't route to SINGLE_TOOL
        assert r.intent != IntentType.SINGLE_TOOL or len(r.suggested_tools) == 0

    async def test_complex_task(self):
        c = RuleBasedClassifier()
        r = await c.classify(
            "Refactor the authentication module and then migrate the database, "
            "after that update the API endpoints and finally run the tests"
        )
        assert r.intent in (IntentType.AGENT_LOOP, IntentType.COMPLEX_PLAN)

    async def test_short_default(self):
        """Short non-question defaults to direct."""
        c = RuleBasedClassifier()
        r = await c.classify("hello")
        assert r.intent == IntentType.DIRECT_ANSWER

    async def test_confidence_range(self):
        c = RuleBasedClassifier()
        r = await c.classify("What is 1+1?")
        assert 0.0 <= r.confidence <= 1.0


class TestIntentClassification:
    def test_model(self):
        ic = IntentClassification(
            intent=IntentType.DIRECT_ANSWER,
            confidence=0.8,
            reasoning="simple question",
        )
        assert ic.intent == IntentType.DIRECT_ANSWER
        assert ic.suggested_tools == []

    def test_with_tools(self):
        ic = IntentClassification(
            intent=IntentType.SINGLE_TOOL,
            confidence=0.6,
            suggested_tools=["web_search"],
        )
        assert len(ic.suggested_tools) == 1
