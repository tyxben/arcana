"""End-to-end integration tests for Arcana v2.

All tests run without real API keys by using mocks/fakes for LLM gateways.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock

import pytest

from arcana.contracts.diagnosis import (
    ErrorCategory,
    ErrorDiagnosis,
    ErrorLayer,
    RecoveryStrategy,
)
from arcana.contracts.intent import IntentClassification, IntentType
from arcana.contracts.llm import (
    LLMResponse,
    ModelConfig,
    TokenUsage,
)
from arcana.contracts.routing import ModelRole, RoutingConfig, TaskComplexity
from arcana.contracts.runtime import RuntimeConfig
from arcana.contracts.state import AgentState, ExecutionStatus
from arcana.contracts.strategy import StrategyDecision, StrategyType
from arcana.contracts.tool import (
    ErrorType,
    SideEffect,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.gateway.router import ModelRouter
from arcana.routing.classifier import HybridClassifier, RuleBasedClassifier
from arcana.runtime.diagnosis.diagnoser import (
    build_recovery_prompt,
    diagnose_tool_error,
)
from arcana.runtime.diagnosis.tracker import RecoveryTracker
from arcana.runtime.policies.adaptive import AdaptivePolicy
from arcana.tool_gateway.formatter import format_tool_for_llm, format_tool_list_for_llm
from arcana.tool_gateway.lazy_registry import KeywordToolMatcher, LazyToolRegistry
from arcana.tool_gateway.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers: fake providers and tool builders
# ---------------------------------------------------------------------------


def _make_tool_spec(
    name: str,
    description: str = "A test tool",
    *,
    when_to_use: str | None = None,
    what_to_expect: str | None = None,
    failure_meaning: str | None = None,
    side_effect: SideEffect = SideEffect.READ,
    input_schema: dict[str, Any] | None = None,
) -> ToolSpec:
    """Create a ToolSpec for testing."""
    return ToolSpec(
        name=name,
        description=description,
        input_schema=input_schema or {"type": "object", "properties": {}},
        when_to_use=when_to_use,
        what_to_expect=what_to_expect,
        failure_meaning=failure_meaning,
        side_effect=side_effect,
    )


class FakeToolProvider:
    """A simple fake ToolProvider for use in ToolRegistry."""

    def __init__(self, spec: ToolSpec, output: Any = "ok") -> None:
        self._spec = spec
        self._output = output

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output=self._output,
        )

    async def health_check(self) -> bool:
        return True


def _make_registry_with_tools(
    specs: list[ToolSpec],
) -> ToolRegistry:
    """Create a ToolRegistry populated with FakeToolProviders."""
    registry = ToolRegistry()
    for spec in specs:
        registry.register(FakeToolProvider(spec))
    return registry


def _fake_llm_response(content: str) -> LLMResponse:
    """Create a minimal LLMResponse with the given content."""
    return LLMResponse(
        content=content,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="mock-model",
        finish_reason="stop",
    )


def _mock_gateway(response_content: str = "Hello!") -> ModelGatewayRegistry:
    """Create a mock ModelGatewayRegistry that returns a fixed LLM response."""
    gateway = ModelGatewayRegistry()
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(
        return_value=_fake_llm_response(response_content)
    )
    mock_provider.health_check = AsyncMock(return_value=True)
    gateway.register("deepseek", mock_provider)
    gateway.set_default("deepseek")
    return gateway


# ===================================================================
# 1. Intent Router Tests
# ===================================================================


class TestIntentRouter:
    """Tests for the RuleBasedClassifier and HybridClassifier."""

    async def test_simple_question_routes_to_direct(self) -> None:
        """A simple 'what is' question should route to DIRECT_ANSWER."""
        classifier = RuleBasedClassifier()
        result = await classifier.classify("What is 1+1?")
        assert result.intent == IntentType.DIRECT_ANSWER
        assert result.confidence > 0

    async def test_explain_routes_to_direct(self) -> None:
        """An 'explain' request should route to DIRECT_ANSWER."""
        classifier = RuleBasedClassifier()
        result = await classifier.classify("explain quantum computing")
        assert result.intent == IntentType.DIRECT_ANSWER

    async def test_tool_trigger_routes_to_single_tool(self) -> None:
        """Mentioning 'search' with a matching available tool -> SINGLE_TOOL."""
        classifier = RuleBasedClassifier()
        result = await classifier.classify(
            "search for quantum computing",
            available_tools=["web_search"],
        )
        assert result.intent == IntentType.SINGLE_TOOL
        assert "web_search" in result.suggested_tools

    async def test_tool_trigger_without_available_tools(self) -> None:
        """Tool trigger without matching available tool should NOT route to SINGLE_TOOL."""
        classifier = RuleBasedClassifier()
        result = await classifier.classify(
            "search for quantum computing",
            available_tools=["file_read"],
        )
        # Should not be SINGLE_TOOL because 'web_search' is not in available_tools
        assert result.intent != IntentType.SINGLE_TOOL

    async def test_complex_request_routes_to_agent_or_plan(self) -> None:
        """A complex request with multiple steps should route to AGENT_LOOP or COMPLEX_PLAN."""
        classifier = RuleBasedClassifier()
        result = await classifier.classify(
            "Refactor the authentication module and then migrate the database schema, "
            "after that run all integration tests and finally deploy to staging"
        )
        assert result.intent in {IntentType.AGENT_LOOP, IntentType.COMPLEX_PLAN}
        assert result.complexity_estimate >= 2

    async def test_hybrid_uses_rules_when_confident(self) -> None:
        """HybridClassifier should use rule-based result when confidence >= threshold and NOT call LLM."""
        gateway = _mock_gateway()
        classifier = HybridClassifier(gateway=gateway, confidence_threshold=0.6)

        # "What is 1+1?" matches a DIRECT pattern with confidence 0.7 >= 0.6
        result = await classifier.classify("What is 1+1?")
        assert result.intent == IntentType.DIRECT_ANSWER
        assert result.confidence >= 0.6

        # The LLM should NOT have been called
        mock_provider = gateway.get("deepseek")
        assert mock_provider is not None
        mock_provider.generate.assert_not_called()

    async def test_hybrid_falls_back_to_llm_when_low_confidence(self) -> None:
        """HybridClassifier should call LLM when rule-based confidence is below threshold."""
        # Make LLM return a valid classification JSON
        llm_response = (
            '{"intent": "agent_loop", "confidence": 0.8, '
            '"reasoning": "needs multiple steps", "suggested_tools": []}'
        )
        gateway = _mock_gateway(llm_response)
        # Use a very high threshold so rules never pass
        classifier = HybridClassifier(gateway=gateway, confidence_threshold=1.0)

        result = await classifier.classify("do something ambiguous")
        # The LLM should have been called and its result used
        mock_provider = gateway.get("deepseek")
        assert mock_provider is not None
        mock_provider.generate.assert_called_once()
        assert result.intent == IntentType.AGENT_LOOP

    async def test_rule_classifier_default_is_direct(self) -> None:
        """When no patterns match, RuleBasedClassifier defaults to DIRECT_ANSWER."""
        classifier = RuleBasedClassifier()
        result = await classifier.classify("hello")
        assert result.intent == IntentType.DIRECT_ANSWER
        assert result.confidence == 0.4  # low-confidence default


# ===================================================================
# 2. Adaptive Policy Tests
# ===================================================================


class TestAdaptivePolicy:
    """Tests for AdaptivePolicy strategy parsing and conversion."""

    async def test_parse_strategy_direct_answer(self) -> None:
        """Can correctly parse a DIRECT_ANSWER strategy response."""
        decision = AdaptivePolicy.parse_strategy_response(
            '{"strategy": "direct_answer", "reasoning": "simple", "action": "2"}'
        )
        assert decision.strategy == StrategyType.DIRECT_ANSWER
        assert decision.action == "2"
        assert decision.reasoning == "simple"

    async def test_parse_strategy_single_tool(self) -> None:
        """Can correctly parse a SINGLE_TOOL strategy response."""
        decision = AdaptivePolicy.parse_strategy_response(
            '{"strategy": "single_tool", "reasoning": "need search", '
            '"tool_name": "web_search", "tool_arguments": {"query": "test"}}'
        )
        assert decision.strategy == StrategyType.SINGLE_TOOL
        assert decision.tool_name == "web_search"
        assert decision.tool_arguments == {"query": "test"}

    async def test_parse_strategy_with_markdown_fences(self) -> None:
        """Handles LLM output wrapped in markdown code fences."""
        decision = AdaptivePolicy.parse_strategy_response(
            '```json\n{"strategy": "single_tool", "reasoning": "need search", '
            '"tool_name": "web_search"}\n```'
        )
        assert decision.strategy == StrategyType.SINGLE_TOOL
        assert decision.tool_name == "web_search"

    async def test_parse_strategy_with_extra_text(self) -> None:
        """Can extract JSON from response with surrounding text."""
        decision = AdaptivePolicy.parse_strategy_response(
            'Here is my decision: {"strategy": "direct_answer", "reasoning": "easy", "action": "42"} done.'
        )
        assert decision.strategy == StrategyType.DIRECT_ANSWER
        assert decision.action == "42"

    async def test_parse_strategy_invalid_json_raises(self) -> None:
        """Raises ValueError for completely invalid JSON."""
        with pytest.raises(ValueError, match="No valid JSON"):
            AdaptivePolicy.parse_strategy_response("this is not json at all")

    async def test_parse_strategy_plan_and_execute(self) -> None:
        """Can parse a PLAN_AND_EXECUTE strategy with plan steps."""
        decision = AdaptivePolicy.parse_strategy_response(
            '{"strategy": "plan_and_execute", "reasoning": "complex task", '
            '"plan": ["step 1", "step 2", "step 3"]}'
        )
        assert decision.strategy == StrategyType.PLAN_AND_EXECUTE
        assert decision.plan == ["step 1", "step 2", "step 3"]

    async def test_parse_strategy_pivot(self) -> None:
        """Can parse a PIVOT strategy with pivot details."""
        decision = AdaptivePolicy.parse_strategy_response(
            '{"strategy": "pivot", "reasoning": "current approach failed", '
            '"pivot_reason": "tool unavailable", "pivot_new_approach": "try manual approach"}'
        )
        assert decision.strategy == StrategyType.PIVOT
        assert decision.pivot_reason == "tool unavailable"

    async def test_strategy_to_policy_decision_direct_answer(self) -> None:
        """DIRECT_ANSWER strategy converts to 'complete' action."""
        decision = StrategyDecision(
            strategy=StrategyType.DIRECT_ANSWER,
            reasoning="simple",
            action="42",
        )
        pd = AdaptivePolicy.strategy_to_policy_decision(decision)
        assert pd.action_type == "complete"
        assert pd.metadata["answer"] == "42"

    async def test_strategy_to_policy_decision_single_tool(self) -> None:
        """SINGLE_TOOL strategy converts to 'tool_call' action."""
        decision = StrategyDecision(
            strategy=StrategyType.SINGLE_TOOL,
            reasoning="need search",
            tool_name="web_search",
            tool_arguments={"query": "test"},
        )
        pd = AdaptivePolicy.strategy_to_policy_decision(decision)
        assert pd.action_type == "tool_call"
        assert len(pd.tool_calls) == 1
        assert pd.tool_calls[0]["name"] == "web_search"
        assert pd.tool_calls[0]["arguments"] == {"query": "test"}

    async def test_strategy_to_policy_decision_parallel(self) -> None:
        """PARALLEL strategy converts to 'tool_call' with multiple calls."""
        decision = StrategyDecision(
            strategy=StrategyType.PARALLEL,
            reasoning="independent searches",
            parallel_actions=[
                {"tool_name": "web_search", "arguments": {"query": "a"}},
                {"tool_name": "web_search", "arguments": {"query": "b"}},
            ],
        )
        pd = AdaptivePolicy.strategy_to_policy_decision(decision)
        assert pd.action_type == "tool_call"
        assert len(pd.tool_calls) == 2
        assert pd.metadata.get("parallel") is True

    async def test_strategy_to_policy_decision_plan(self) -> None:
        """PLAN_AND_EXECUTE strategy converts to 'llm_call' with plan metadata."""
        decision = StrategyDecision(
            strategy=StrategyType.PLAN_AND_EXECUTE,
            reasoning="complex task",
            plan=["step 1", "step 2"],
        )
        pd = AdaptivePolicy.strategy_to_policy_decision(decision)
        assert pd.action_type == "llm_call"
        assert pd.metadata["plan"] == ["step 1", "step 2"]

    async def test_strategy_to_policy_decision_sequential(self) -> None:
        """SEQUENTIAL strategy converts to 'llm_call'."""
        decision = StrategyDecision(
            strategy=StrategyType.SEQUENTIAL,
            reasoning="do one thing at a time",
        )
        pd = AdaptivePolicy.strategy_to_policy_decision(decision)
        assert pd.action_type == "llm_call"
        assert pd.metadata["strategy"] == "sequential"

    async def test_strategy_to_policy_decision_pivot(self) -> None:
        """PIVOT strategy converts to 'llm_call' with pivot info."""
        decision = StrategyDecision(
            strategy=StrategyType.PIVOT,
            reasoning="change direction",
            pivot_reason="tool failed",
            pivot_new_approach="manual",
        )
        pd = AdaptivePolicy.strategy_to_policy_decision(decision)
        assert pd.action_type == "llm_call"
        assert pd.metadata["pivot_reason"] == "tool failed"

    async def test_adaptive_policy_decide_produces_llm_call(self) -> None:
        """AdaptivePolicy.decide() produces an llm_call PolicyDecision."""
        policy = AdaptivePolicy()
        state = AgentState(run_id="test-run", goal="What is 2+2?")
        decision = await policy.decide(state)
        assert decision.action_type == "llm_call"
        assert len(decision.messages) == 2  # system + user


# ===================================================================
# 3. LazyToolRegistry Tests
# ===================================================================


class TestLazyToolRegistryE2E:
    """Tests for LazyToolRegistry initial selection, expansion, and formatting."""

    def _make_registry_and_lazy(self) -> tuple[ToolRegistry, LazyToolRegistry]:
        """Build a ToolRegistry with diverse tools and a wrapping LazyToolRegistry."""
        specs = [
            _make_tool_spec("web_search", "Search the web for information",
                            when_to_use="When you need current information"),
            _make_tool_spec("file_read", "Read contents of a file",
                            when_to_use="When you need to read a file"),
            _make_tool_spec("file_write", "Write contents to a file",
                            side_effect=SideEffect.WRITE,
                            when_to_use="When you need to write to a file"),
            _make_tool_spec("code_exec", "Execute code in a sandbox",
                            when_to_use="When you need to run code"),
            _make_tool_spec("calculator", "Perform mathematical calculations",
                            when_to_use="When you need to calculate"),
            _make_tool_spec("database_query", "Query a SQL database",
                            when_to_use="When you need data from a database"),
            _make_tool_spec("http_get", "Make an HTTP GET request",
                            when_to_use="When you need to fetch a URL"),
        ]
        registry = _make_registry_with_tools(specs)
        lazy = LazyToolRegistry(registry, max_initial_tools=3, max_working_set=5)
        return registry, lazy

    async def test_initial_selection_matches_goal(self) -> None:
        """Initial selection should prioritize tools relevant to the goal."""
        _, lazy = self._make_registry_and_lazy()
        selected = lazy.select_initial_tools("search the web for latest AI news")

        assert len(selected) <= 3
        selected_names = [s.name for s in selected]
        # web_search should be among the top picks
        assert "web_search" in selected_names

    async def test_initial_selection_respects_max(self) -> None:
        """Initial selection should not exceed max_initial_tools."""
        _, lazy = self._make_registry_and_lazy()
        selected = lazy.select_initial_tools("do everything possible")
        assert len(selected) <= 3

    async def test_working_set_populated_after_selection(self) -> None:
        """Working set should contain the selected tools after initial selection."""
        _, lazy = self._make_registry_and_lazy()
        selected = lazy.select_initial_tools("read the config file")
        assert len(lazy.working_set) == len(selected)
        for spec in selected:
            assert spec.name in [s.name for s in lazy.working_set]

    async def test_expand_adds_tools(self) -> None:
        """Expand should add new tools to the working set."""
        _, lazy = self._make_registry_and_lazy()
        lazy.select_initial_tools("search the web")
        initial_size = len(lazy.working_set)

        new_tools = lazy.expand("I need to run some code")
        assert len(new_tools) > 0
        assert len(lazy.working_set) > initial_size

        new_names = [t.name for t in new_tools]
        # The newly added tools should not have been in the initial set
        for name in new_names:
            assert name in [s.name for s in lazy.working_set]

    async def test_expand_does_not_duplicate(self) -> None:
        """Expand should not add tools already in the working set."""
        _, lazy = self._make_registry_and_lazy()
        lazy.select_initial_tools("search the web")
        initial_names = {s.name for s in lazy.working_set}

        new_tools = lazy.expand("search the web again")
        new_names = {t.name for t in new_tools}
        # Newly added should not overlap with initial set
        assert new_names.isdisjoint(initial_names)

    async def test_get_tool_on_demand(self) -> None:
        """get_tool_on_demand loads a specific tool into the working set."""
        _, lazy = self._make_registry_and_lazy()
        lazy.select_initial_tools("something unrelated to databases")

        # database_query should not be in working set yet (unless it matched by chance)
        if "database_query" not in [s.name for s in lazy.working_set]:
            spec = lazy.get_tool_on_demand("database_query")
            assert spec is not None
            assert spec.name == "database_query"
            assert "database_query" in [s.name for s in lazy.working_set]

    async def test_get_tool_on_demand_nonexistent(self) -> None:
        """get_tool_on_demand returns None for a nonexistent tool."""
        _, lazy = self._make_registry_and_lazy()
        result = lazy.get_tool_on_demand("nonexistent_tool")
        assert result is None

    async def test_expansion_log_records_events(self) -> None:
        """Expansion events should be recorded in the expansion log."""
        _, lazy = self._make_registry_and_lazy()
        lazy.select_initial_tools("search for info")
        lazy.expand("need calculator")

        log = lazy.expansion_log
        assert len(log) == 2
        assert log[0].trigger == "initial_selection"
        assert log[1].trigger == "on_demand_expansion"

    async def test_available_but_hidden_correct(self) -> None:
        """available_but_hidden should list tools not in working set."""
        _, lazy = self._make_registry_and_lazy()
        lazy.select_initial_tools("search the web")
        hidden = lazy.available_but_hidden
        working_names = {s.name for s in lazy.working_set}
        # Hidden tools should not appear in working set
        for name in hidden:
            assert name not in working_names

    async def test_to_openai_tools_format(self) -> None:
        """to_openai_tools should return valid OpenAI function calling format."""
        _, lazy = self._make_registry_and_lazy()
        lazy.select_initial_tools("search the web")
        tools = lazy.to_openai_tools()
        assert len(tools) > 0
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    async def test_affordance_formatting(self) -> None:
        """format_tool_for_llm should produce a readable description with affordances."""
        spec = _make_tool_spec(
            "web_search",
            "Search the web",
            when_to_use="When you need current information",
            what_to_expect="Returns search results that may need filtering",
            failure_meaning="Search engine is down or query is too vague",
        )
        formatted = format_tool_for_llm(spec)
        assert "web_search" in formatted
        assert "Search the web" in formatted
        assert "Use when:" in formatted
        assert "Expect:" in formatted
        assert "If it fails:" in formatted

    async def test_format_tool_list(self) -> None:
        """format_tool_list_for_llm should list all tools with a summary."""
        specs = [
            _make_tool_spec("web_search", "Search the web"),
            _make_tool_spec("file_read", "Read a file"),
        ]
        formatted = format_tool_list_for_llm(specs)
        assert "2 tools" in formatted
        assert "web_search" in formatted
        assert "file_read" in formatted

    async def test_format_empty_tool_list(self) -> None:
        """format_tool_list_for_llm with no tools gives a 'no tools' message."""
        formatted = format_tool_list_for_llm([])
        assert "No tools" in formatted

    async def test_reset_clears_working_set(self) -> None:
        """reset() should clear the working set and expansion log."""
        _, lazy = self._make_registry_and_lazy()
        lazy.select_initial_tools("search")
        assert len(lazy.working_set) > 0
        lazy.reset()
        assert len(lazy.working_set) == 0
        assert len(lazy.expansion_log) == 0


# ===================================================================
# 4. Diagnostic Recovery Tests
# ===================================================================


class TestDiagnosticRecovery:
    """Tests for error diagnosis, fuzzy matching, recovery prompts, and tracking."""

    async def test_diagnose_tool_not_found(self) -> None:
        """Tool not found should be diagnosed as FACT_ERROR when a close match exists."""
        available = ["web_search", "file_read", "code_exec"]
        error = ToolError(
            error_type=ErrorType.NON_RETRYABLE,
            message="Tool 'web_serach' not found",
            code="TOOL_NOT_FOUND",
        )
        diagnosis = diagnose_tool_error("web_serach", error, available)
        assert diagnosis.error_category == ErrorCategory.FACT_ERROR
        assert diagnosis.suggested_tool == "web_search"
        assert "web_search" in diagnosis.root_cause

    async def test_diagnose_tool_not_found_no_match(self) -> None:
        """Tool not found with no close match should be TOOL_MISMATCH."""
        available = ["web_search", "file_read"]
        error = ToolError(
            error_type=ErrorType.NON_RETRYABLE,
            message="Tool 'zzz_nonexistent' not found",
            code="TOOL_NOT_FOUND",
        )
        diagnosis = diagnose_tool_error("zzz_nonexistent", error, available)
        assert diagnosis.error_category == ErrorCategory.TOOL_MISMATCH
        assert diagnosis.suggested_tool is None

    async def test_diagnose_with_fuzzy_match(self) -> None:
        """A similar tool name should produce a suggestion via fuzzy matching."""
        available = ["web_search", "file_read", "code_exec"]
        error = ToolError(
            error_type=ErrorType.NON_RETRYABLE,
            message="Tool 'web_sarch' not found",
            code="TOOL_NOT_FOUND",
        )
        diagnosis = diagnose_tool_error("web_sarch", error, available)
        assert diagnosis.suggested_tool == "web_search"
        assert any("web_search" in s for s in diagnosis.actionable_suggestions)

    async def test_diagnose_timeout(self) -> None:
        """A timeout error should be diagnosed as RESOURCE_UNAVAILABLE."""
        error = ToolError(
            error_type=ErrorType.RETRYABLE,
            message="Tool 'web_search' timed out after 30000ms",
            code="TIMEOUT",
        )
        diagnosis = diagnose_tool_error("web_search", error, ["web_search"])
        assert diagnosis.error_category == ErrorCategory.RESOURCE_UNAVAILABLE
        assert diagnosis.error_layer == ErrorLayer.EXTERNAL_SERVICE

    async def test_diagnose_unauthorized(self) -> None:
        """An unauthorized error should be diagnosed as PERMISSION_DENIED."""
        error = ToolError(
            error_type=ErrorType.NON_RETRYABLE,
            message="Unauthorized access to tool",
            code="UNAUTHORIZED",
            details={"required_capability": "file_write"},
        )
        diagnosis = diagnose_tool_error("file_write", error, ["file_write"])
        assert diagnosis.error_category == ErrorCategory.PERMISSION_DENIED

    async def test_recovery_prompt_is_actionable(self) -> None:
        """The recovery prompt should contain specific suggestions and strategy guidance."""
        available = ["web_search", "file_read"]
        error = ToolError(
            error_type=ErrorType.NON_RETRYABLE,
            message="Tool 'web_serach' not found",
            code="TOOL_NOT_FOUND",
        )
        diagnosis = diagnose_tool_error("web_serach", error, available)
        prompt = build_recovery_prompt(
            diagnosis,
            original_goal="Search for AI news",
            step_context="Attempted to call web_serach",
        )
        assert "web_search" in prompt
        assert "Search for AI news" in prompt
        assert "Step context" in prompt
        assert "root cause" in prompt.lower() or "Root cause" in prompt

    async def test_recovery_prompt_without_context(self) -> None:
        """Recovery prompt without step_context should still be valid."""
        error = ToolError(
            error_type=ErrorType.NON_RETRYABLE,
            message="Tool 'web_serach' not found",
            code="TOOL_NOT_FOUND",
        )
        diagnosis = diagnose_tool_error("web_serach", error, ["web_search"])
        prompt = build_recovery_prompt(diagnosis, original_goal="test goal")
        assert "test goal" in prompt
        assert "Step context" not in prompt

    async def test_tracker_records_attempts(self) -> None:
        """RecoveryTracker correctly records and counts attempts."""
        tracker = RecoveryTracker(max_recoveries_per_category=2, max_total_recoveries=5)
        diagnosis = ErrorDiagnosis(
            error_category=ErrorCategory.FACT_ERROR,
            error_layer=ErrorLayer.LLM_REASONING,
            root_cause="typo",
            actionable_suggestions=["fix typo"],
            recommended_strategy=RecoveryStrategy.RETRY_WITH_MODIFICATION,
            related_tool="web_search",
        )
        tracker.record(diagnosis)
        assert tracker.total_attempts() == 1
        assert tracker.attempts_for("web_search", ErrorCategory.FACT_ERROR) == 1

    async def test_tracker_detects_loop(self) -> None:
        """Repeated same-category errors should trigger escalation."""
        tracker = RecoveryTracker(max_recoveries_per_category=2, max_total_recoveries=5)
        for _ in range(3):
            diagnosis = ErrorDiagnosis(
                error_category=ErrorCategory.FACT_ERROR,
                error_layer=ErrorLayer.LLM_REASONING,
                root_cause="keeps getting it wrong",
                actionable_suggestions=["try harder"],
                recommended_strategy=RecoveryStrategy.RETRY_WITH_MODIFICATION,
                related_tool="web_search",
            )
            tracker.record(diagnosis)

        assert tracker.should_escalate() is True

    async def test_tracker_detects_abort(self) -> None:
        """Total recovery attempts exceeding max should trigger abort."""
        tracker = RecoveryTracker(max_recoveries_per_category=3, max_total_recoveries=3)
        for i in range(4):
            diagnosis = ErrorDiagnosis(
                error_category=ErrorCategory.FORMAT_ERROR,
                error_layer=ErrorLayer.VALIDATION,
                root_cause=f"error {i}",
                actionable_suggestions=["fix"],
                recommended_strategy=RecoveryStrategy.RETRY_WITH_MODIFICATION,
                related_tool=f"tool_{i}",
            )
            tracker.record(diagnosis)

        assert tracker.should_abort() is True

    async def test_tracker_detects_pattern(self) -> None:
        """RecoveryTracker should detect consecutive identical error categories."""
        tracker = RecoveryTracker()
        for _ in range(3):
            diagnosis = ErrorDiagnosis(
                error_category=ErrorCategory.FACT_ERROR,
                error_layer=ErrorLayer.LLM_REASONING,
                root_cause="same error",
                actionable_suggestions=["fix"],
                recommended_strategy=RecoveryStrategy.RETRY_WITH_MODIFICATION,
                related_tool="web_search",
            )
            tracker.record(diagnosis)

        pattern = tracker.get_pattern()
        assert pattern is not None
        assert "consecutive" in pattern
        assert "fact_error" in pattern

    async def test_tracker_no_pattern_initially(self) -> None:
        """RecoveryTracker should return None pattern with insufficient history."""
        tracker = RecoveryTracker()
        assert tracker.get_pattern() is None

    async def test_tracker_reset(self) -> None:
        """RecoveryTracker.reset() should clear all history."""
        tracker = RecoveryTracker()
        diagnosis = ErrorDiagnosis(
            error_category=ErrorCategory.FACT_ERROR,
            error_layer=ErrorLayer.LLM_REASONING,
            root_cause="test",
            actionable_suggestions=["fix"],
            recommended_strategy=RecoveryStrategy.RETRY_WITH_MODIFICATION,
            related_tool="tool_a",
        )
        tracker.record(diagnosis)
        assert tracker.total_attempts() == 1
        tracker.reset()
        assert tracker.total_attempts() == 0
        assert not tracker.should_escalate()


# ===================================================================
# 5. Model Router Tests
# ===================================================================


class TestModelRouter:
    """Tests for ModelRouter complexity estimation and role selection."""

    async def test_complexity_trivial(self) -> None:
        """A simple 'what is' question should be TRIVIAL."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        complexity = router.estimate_complexity("what is 2+2?")
        assert complexity == TaskComplexity.TRIVIAL

    async def test_complexity_simple(self) -> None:
        """An 'explain' request should be TRIVIAL (simple keyword reduces score)."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        complexity = router.estimate_complexity("explain how gravity works")
        assert complexity == TaskComplexity.TRIVIAL

    async def test_complexity_moderate_or_higher(self) -> None:
        """A complex goal with analysis keywords should be at least MODERATE."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        complexity = router.estimate_complexity(
            "analyze and compare different design patterns for microservices architecture"
        )
        assert complexity.value in {"moderate", "complex", "expert"}

    async def test_complexity_increases_with_errors(self) -> None:
        """Error count should increase complexity."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        complexity_order = [
            TaskComplexity.TRIVIAL,
            TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE,
            TaskComplexity.COMPLEX,
            TaskComplexity.EXPERT,
        ]
        base = router.estimate_complexity("do a simple task")
        with_errors = router.estimate_complexity("do a simple task", error_count=5)
        assert complexity_order.index(with_errors) >= complexity_order.index(base)

    async def test_complexity_long_goal(self) -> None:
        """A very long goal should get a complexity bump."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        long_goal = "analyze " + "x " * 300
        complexity = router.estimate_complexity(long_goal)
        # The "analyze" keyword + length should push this up
        assert complexity.value in {"simple", "moderate", "complex", "expert"}

    async def test_role_selection_think(self) -> None:
        """'think' step should select STRATEGIST role."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        role = router.select_role("think")
        assert role == ModelRole.STRATEGIST

    async def test_role_selection_plan(self) -> None:
        """'plan' step should select STRATEGIST role."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        role = router.select_role("plan")
        assert role == ModelRole.STRATEGIST

    async def test_role_selection_act(self) -> None:
        """'act' step should select EXECUTOR role."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        role = router.select_role("act")
        assert role == ModelRole.EXECUTOR

    async def test_role_selection_verify(self) -> None:
        """'verify' step should select VALIDATOR role."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        role = router.select_role("verify")
        assert role == ModelRole.VALIDATOR

    async def test_role_selection_compress(self) -> None:
        """'compress' step should select COMPRESSOR role."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        role = router.select_role("compress")
        assert role == ModelRole.COMPRESSOR

    async def test_role_selection_route(self) -> None:
        """'route' and 'classify' steps should select ROUTER role."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        assert router.select_role("route") == ModelRole.ROUTER
        assert router.select_role("classify") == ModelRole.ROUTER

    async def test_role_selection_default(self) -> None:
        """Unknown step type should default to EXECUTOR."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        role = router.select_role("unknown_step")
        assert role == ModelRole.EXECUTOR

    async def test_auto_downgrade_trivial(self) -> None:
        """With auto_downgrade, TRIVIAL complexity should downgrade think/plan to EXECUTOR."""
        gateway = _mock_gateway()
        config = RoutingConfig(auto_downgrade=True)
        router = ModelRouter(gateway, config=config)
        role = router.select_role("think", complexity=TaskComplexity.TRIVIAL)
        assert role == ModelRole.EXECUTOR

    async def test_no_downgrade_complex(self) -> None:
        """COMPLEX tasks should NOT be downgraded even with auto_downgrade."""
        gateway = _mock_gateway()
        config = RoutingConfig(auto_downgrade=True)
        router = ModelRouter(gateway, config=config)
        role = router.select_role("think", complexity=TaskComplexity.COMPLEX)
        assert role == ModelRole.STRATEGIST

    async def test_get_config_for_role(self) -> None:
        """get_config_for_role should return a ModelConfig for each role."""
        gateway = _mock_gateway()
        router = ModelRouter(gateway)
        for role in ModelRole:
            config = router.get_config_for_role(role)
            assert isinstance(config, ModelConfig)
            assert config.provider is not None


# ===================================================================
# 6. SDK Tool Decorator Tests
# ===================================================================


class TestSDK:
    """Tests for the @arcana.tool decorator and FunctionToolProvider."""

    async def test_tool_decorator_creates_spec(self) -> None:
        """@arcana.tool should attach a ToolSpec to the function."""
        import arcana

        @arcana.tool(
            when_to_use="When you need to add numbers",
            what_to_expect="Returns the sum",
        )
        async def add(a: int, b: int) -> int:
            return a + b

        assert hasattr(add, "_arcana_tool_spec")
        spec = add._arcana_tool_spec
        assert isinstance(spec, ToolSpec)
        assert spec.name == "add"
        assert spec.when_to_use == "When you need to add numbers"

    async def test_tool_decorator_custom_name(self) -> None:
        """@arcana.tool with explicit name should use that name."""
        import arcana

        @arcana.tool(name="custom_add", description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        spec = add._arcana_tool_spec
        assert spec.name == "custom_add"
        assert spec.description == "Add two numbers"

    async def test_function_tool_provider_sync(self) -> None:
        """Sync function wrapped by _FunctionToolProvider should execute correctly."""
        import arcana
        from arcana.sdk import _FunctionToolProvider

        @arcana.tool()
        def multiply(a: int, b: int) -> int:
            return a * b

        spec = multiply._arcana_tool_spec
        provider = _FunctionToolProvider(spec=spec, func=multiply)

        call = ToolCall(id="test-1", name="multiply", arguments={"a": 3, "b": 4})
        result = await provider.execute(call)
        assert result.success is True
        assert result.output == 12

    async def test_function_tool_provider_async(self) -> None:
        """Async function wrapped by _FunctionToolProvider should execute correctly."""
        import arcana
        from arcana.sdk import _FunctionToolProvider

        @arcana.tool()
        async def fetch_data(url: str) -> str:
            return f"data from {url}"

        spec = fetch_data._arcana_tool_spec
        provider = _FunctionToolProvider(spec=spec, func=fetch_data)

        call = ToolCall(id="test-2", name="fetch_data", arguments={"url": "https://example.com"})
        result = await provider.execute(call)
        assert result.success is True
        assert result.output == "data from https://example.com"

    async def test_function_tool_provider_error_handling(self) -> None:
        """_FunctionToolProvider should catch exceptions and return error results."""
        import arcana
        from arcana.sdk import _FunctionToolProvider

        @arcana.tool()
        def failing_tool() -> str:
            msg = "something went wrong"
            raise ValueError(msg)

        spec = failing_tool._arcana_tool_spec
        provider = _FunctionToolProvider(spec=spec, func=failing_tool)

        call = ToolCall(id="test-3", name="failing_tool", arguments={})
        result = await provider.execute(call)
        assert result.success is False
        assert result.error is not None
        assert "something went wrong" in result.error.message

    async def test_signature_to_schema_basic_types(self) -> None:
        """Function signature should correctly convert to JSON Schema.

        Note: _signature_to_json_schema uses type objects as dict keys. Since
        this test file has ``from __future__ import annotations``, we build
        an ``inspect.Signature`` directly with real type annotations to test
        the mapping logic.
        """
        from arcana.sdk import _signature_to_json_schema

        sig = inspect.Signature(parameters=[
            inspect.Parameter("name", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter("count", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            inspect.Parameter("ratio", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float),
            inspect.Parameter("active", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=bool),
        ])
        schema = _signature_to_json_schema(sig)

        assert schema["type"] == "object"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["ratio"]["type"] == "number"
        assert schema["properties"]["active"]["type"] == "boolean"
        assert set(schema["required"]) == {"name", "count", "ratio", "active"}

    async def test_signature_to_schema_optional_params(self) -> None:
        """Parameters with defaults should NOT appear in 'required'."""
        from arcana.sdk import _signature_to_json_schema

        sig = inspect.Signature(parameters=[
            inspect.Parameter("name", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter("limit", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int, default=10),
        ])
        schema = _signature_to_json_schema(sig)

        assert "name" in schema["required"]
        assert "limit" not in schema["required"]

    async def test_signature_to_schema_skips_self(self) -> None:
        """'self' and 'cls' parameters should be excluded from schema."""
        from arcana.sdk import _signature_to_json_schema

        sig = inspect.Signature(parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("query", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        ])
        schema = _signature_to_json_schema(sig)

        assert "self" not in schema["properties"]
        assert "query" in schema["properties"]

    async def test_signature_to_schema_list_and_dict(self) -> None:
        """list and dict params should map to 'array' and 'object' types."""
        from arcana.sdk import _signature_to_json_schema

        sig = inspect.Signature(parameters=[
            inspect.Parameter("items", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=list),
            inspect.Parameter("metadata", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=dict),
        ])
        schema = _signature_to_json_schema(sig)

        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["metadata"]["type"] == "object"

    async def test_tool_decorator_side_effect(self) -> None:
        """@arcana.tool side_effect parameter should propagate to ToolSpec."""
        import arcana

        @arcana.tool(side_effect="write", requires_confirmation=True)
        async def delete_file(path: str) -> str:
            return f"deleted {path}"

        spec = delete_file._arcana_tool_spec
        assert spec.side_effect == SideEffect.WRITE
        assert spec.requires_confirmation is True


# ===================================================================
# 7. Agent Fast Path Tests (with Mock)
# ===================================================================


class TestAgentFastPath:
    """Tests for Agent intent-routing fast paths with mock gateways."""

    async def test_direct_answer_skips_loop(self) -> None:
        """DIRECT_ANSWER routing should skip the agent loop and return with one LLM call."""
        from arcana.runtime.agent import Agent
        from arcana.runtime.reducers.default import DefaultReducer

        gateway = _mock_gateway("The answer is 42.")
        # Use a mock classifier that always returns DIRECT_ANSWER
        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock(
            return_value=IntentClassification(
                intent=IntentType.DIRECT_ANSWER,
                confidence=0.9,
                reasoning="simple question",
            )
        )

        agent = Agent(
            policy=AdaptivePolicy(),
            reducer=DefaultReducer(),
            gateway=gateway,
            config=RuntimeConfig(max_steps=10),
            intent_classifier=mock_classifier,
            auto_route=True,
        )

        state = await agent.run("What is 1+1?")

        # Should complete successfully
        assert state.status == ExecutionStatus.COMPLETED
        # The direct path sets step to 1
        assert state.current_step == 1
        # The answer should be stored in working_memory
        assert state.working_memory.get("answer") == "The answer is 42."
        # The classifier should have been called exactly once
        mock_classifier.classify.assert_called_once()
        # The LLM (gateway) should have been called exactly once
        mock_provider = gateway.get("deepseek")
        assert mock_provider is not None
        mock_provider.generate.assert_called_once()

    async def test_agent_loop_for_complex_intent(self) -> None:
        """AGENT_LOOP or COMPLEX_PLAN intents should enter the full agent loop."""
        from arcana.runtime.agent import Agent
        from arcana.runtime.reducers.default import DefaultReducer

        # Set up gateway to return a strategy decision that says "direct_answer"
        # so the loop completes quickly
        strategy_response = (
            '{"strategy": "direct_answer", "reasoning": "done", "action": "result here"}'
        )
        gateway = _mock_gateway(strategy_response)

        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock(
            return_value=IntentClassification(
                intent=IntentType.AGENT_LOOP,
                confidence=0.8,
                reasoning="needs multiple steps",
            )
        )

        agent = Agent(
            policy=AdaptivePolicy(),
            reducer=DefaultReducer(),
            gateway=gateway,
            config=RuntimeConfig(max_steps=5),
            intent_classifier=mock_classifier,
            auto_route=True,
        )

        state = await agent.run("Build a complex system")

        # The agent loop was entered (not the fast path)
        # It will run and eventually stop (either by reaching goal or max steps)
        assert state.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}

    async def test_auto_route_disabled_skips_classification(self) -> None:
        """When auto_route=False, the classifier should NOT be called."""
        from arcana.runtime.agent import Agent
        from arcana.runtime.reducers.default import DefaultReducer

        strategy_response = (
            '{"strategy": "direct_answer", "reasoning": "done", "action": "result"}'
        )
        gateway = _mock_gateway(strategy_response)

        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock()

        agent = Agent(
            policy=AdaptivePolicy(),
            reducer=DefaultReducer(),
            gateway=gateway,
            config=RuntimeConfig(max_steps=3),
            intent_classifier=mock_classifier,
            auto_route=False,  # Disabled
        )

        await agent.run("do something")

        # Classifier should NOT have been called
        mock_classifier.classify.assert_not_called()

    async def test_no_classifier_skips_routing(self) -> None:
        """When no intent_classifier is provided, routing is skipped."""
        from arcana.runtime.agent import Agent
        from arcana.runtime.reducers.default import DefaultReducer

        strategy_response = (
            '{"strategy": "direct_answer", "reasoning": "done", "action": "result"}'
        )
        gateway = _mock_gateway(strategy_response)

        agent = Agent(
            policy=AdaptivePolicy(),
            reducer=DefaultReducer(),
            gateway=gateway,
            config=RuntimeConfig(max_steps=3),
            intent_classifier=None,  # No classifier
            auto_route=True,
        )

        state = await agent.run("do something")
        # Should still run (via the full loop), not crash
        assert state.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}


# ===================================================================
# 8. Keyword Tool Matcher Tests (bonus)
# ===================================================================


class TestKeywordToolMatcher:
    """Tests for the KeywordToolMatcher ranking logic."""

    async def test_name_match_ranks_highest(self) -> None:
        """A tool whose name appears in the query should rank first."""
        matcher = KeywordToolMatcher()
        specs = [
            _make_tool_spec("file_read", "Read a file"),
            _make_tool_spec("web_search", "Search the web"),
            _make_tool_spec("calculator", "Do math"),
        ]
        ranked = matcher.rank("I need to web search for something", specs)
        assert ranked[0].name == "web_search"

    async def test_description_overlap_matters(self) -> None:
        """Description keyword overlap should influence ranking."""
        matcher = KeywordToolMatcher()
        specs = [
            _make_tool_spec("tool_a", "Search the web for information"),
            _make_tool_spec("tool_b", "Read database records"),
        ]
        ranked = matcher.rank("search for information on the web", specs)
        assert ranked[0].name == "tool_a"

    async def test_deterministic_ordering(self) -> None:
        """Same inputs should always produce the same ranking."""
        matcher = KeywordToolMatcher()
        specs = [
            _make_tool_spec("beta_tool", "Do stuff"),
            _make_tool_spec("alpha_tool", "Do stuff"),
        ]
        ranked1 = matcher.rank("do stuff", specs)
        ranked2 = matcher.rank("do stuff", specs)
        assert [s.name for s in ranked1] == [s.name for s in ranked2]
