"""Tests for Intent Router -- classifiers and executor."""


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

    async def test_direct_pattern_no_tools_full_confidence(self):
        """Without tools, direct patterns get normal confidence."""
        c = RuleBasedClassifier()
        r = await c.classify("What is Python?")
        assert r.intent == IntentType.DIRECT_ANSWER
        assert r.confidence == 0.7

    async def test_direct_pattern_with_tools_lower_confidence(self):
        """With tools available, direct patterns get lower confidence."""
        c = RuleBasedClassifier()
        r = await c.classify("What is the weather?", available_tools=["weather_api"])
        assert r.intent == IntentType.DIRECT_ANSWER
        assert r.confidence == 0.5

    async def test_explicit_tool_mention_skips_direct(self):
        """When the goal mentions a tool by name, never route to direct_answer."""
        c = RuleBasedClassifier()
        r = await c.classify(
            "What is the result? use the calculator",
            available_tools=["calculator"],
        )
        assert r.intent != IntentType.DIRECT_ANSWER

    async def test_explicit_tool_mention_routes_to_tool(self):
        """Mentioning a tool name + matching trigger -> single_tool."""
        c = RuleBasedClassifier()
        r = await c.classify(
            "calculate 2+2 with calculator",
            available_tools=["calculator"],
        )
        assert r.intent == IntentType.SINGLE_TOOL
        assert "calculator" in r.suggested_tools

    async def test_tool_mention_without_trigger_not_direct(self):
        """Mentioning a tool name even without trigger words skips direct."""
        c = RuleBasedClassifier()
        r = await c.classify(
            "What is 5*5? use my_tool please",
            available_tools=["my_tool"],
        )
        # Should NOT be direct_answer since the user explicitly asked for a tool
        assert r.intent != IntentType.DIRECT_ANSWER


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
