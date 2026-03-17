"""
Arcana v2 Demo -- End-to-end integration test.

Demonstrates:
1. Intent Router (direct answer path)
2. Adaptive Policy (LLM-driven strategy)
3. Tool usage with @arcana.tool
4. Lazy tool loading
5. Full SDK API

Usage:
    # Set API key
    export DEEPSEEK_API_KEY="sk-xxx"

    # Run demo
    uv run python examples/demo_v2.py
"""

import asyncio
import os
import sys

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def demo_1_direct_answer():
    """Demo 1: Simple question -> Intent Router -> Direct Answer (1 LLM call)"""
    print("\n" + "=" * 60)
    print("Demo 1: Direct Answer (via Intent Router)")
    print("=" * 60)

    from arcana.contracts.llm import ModelConfig
    from arcana.gateway.providers.openai_compatible import create_deepseek_provider
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.routing.classifier import RuleBasedClassifier
    from arcana.routing.executor import DirectExecutor

    # Setup
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("  DEEPSEEK_API_KEY not set, skipping real LLM call")

        # Still test the routing
        classifier = RuleBasedClassifier()
        result = await classifier.classify("What is 2 + 2?")
        print(f"  Intent: {result.intent.value}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reasoning: {result.reasoning}")
        assert result.intent.value == "direct_answer", (
            f"Expected direct_answer, got {result.intent.value}"
        )
        print("  [OK] Intent routing works (no API key for LLM call)")
        return

    gateway = ModelGatewayRegistry()
    gateway.register("deepseek", create_deepseek_provider(api_key))
    gateway.set_default("deepseek")

    # Classify
    classifier = RuleBasedClassifier()
    result = await classifier.classify("What is 2 + 2?")
    print(f"  Intent: {result.intent.value}")
    print(f"  Confidence: {result.confidence}")

    # Execute fast path
    executor = DirectExecutor()
    config = ModelConfig(provider="deepseek", model_id="deepseek-chat")
    answer = await executor.direct_answer("What is 2 + 2?", gateway, config)

    print(f"  Answer: {answer}")
    print("  [OK] Direct answer works!")


async def demo_2_adaptive_policy():
    """Demo 2: Complex task -> Agent Loop -> Adaptive Policy"""
    print("\n" + "=" * 60)
    print("Demo 2: Adaptive Policy (LLM chooses strategy)")
    print("=" * 60)

    from arcana.runtime.policies.adaptive import AdaptivePolicy

    # Test strategy parsing (no API needed)
    _policy = AdaptivePolicy()

    # Simulate LLM response
    test_responses = [
        '{"strategy": "direct_answer", "reasoning": "I know the answer", "action": "Python was created by Guido van Rossum in 1991"}',
        '{"strategy": "single_tool", "reasoning": "Need to search", "tool_name": "web_search", "tool_arguments": {"query": "latest AI news"}}',
        '{"strategy": "pivot", "reasoning": "Wrong approach", "pivot_reason": "Current method too slow", "pivot_new_approach": "Use parallel search instead"}',
    ]

    for resp in test_responses:
        decision = AdaptivePolicy.parse_strategy_response(resp)
        print(
            f"  Strategy: {decision.strategy.value:20s} | Reasoning: {decision.reasoning}"
        )

    print("  [OK] Adaptive Policy parsing works!")


async def demo_3_tool_decorator():
    """Demo 3: @arcana.tool decorator + FunctionToolProvider"""
    print("\n" + "=" * 60)
    print("Demo 3: SDK Tool Decorator")
    print("=" * 60)

    from arcana.contracts.tool import ToolCall
    from arcana.sdk import _FunctionToolProvider, tool

    @tool(
        when_to_use="When you need to do math",
        what_to_expect="Returns the numeric result",
    )
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return str(eval(expression))  # noqa: S307 -- demo only

    # Check spec was attached
    spec = calculator._arcana_tool_spec
    print(f"  Tool name: {spec.name}")
    print(f"  When to use: {spec.when_to_use}")
    print(f"  Input schema: {spec.input_schema}")

    # Test execution
    provider = _FunctionToolProvider(spec=spec, func=calculator)
    call = ToolCall(id="test-1", name="calculator", arguments={"expression": "2 + 3 * 4"})
    result = await provider.execute(call)

    print(f"  Result: {result.output}")
    assert result.success
    assert result.output == "14"
    print("  [OK] Tool decorator + execution works!")


async def demo_4_lazy_tools():
    """Demo 4: LazyToolRegistry selects relevant tools"""
    print("\n" + "=" * 60)
    print("Demo 4: Lazy Tool Loading")
    print("=" * 60)

    from arcana.contracts.tool import ToolCall, ToolResult, ToolSpec
    from arcana.tool_gateway.formatter import format_tool_for_llm
    from arcana.tool_gateway.lazy_registry import LazyToolRegistry
    from arcana.tool_gateway.registry import ToolRegistry

    # Create a registry with many tools
    registry = ToolRegistry()

    tool_defs = [
        ("web_search", "Search the web", "search", "When you need current information"),
        ("file_read", "Read a file", "file", "When you need to read file contents"),
        ("file_write", "Write a file", "file", "When you need to save data"),
        ("database_query", "Query a database", "database", "When you need structured data"),
        ("code_execute", "Execute code", "code", "When you need to run code"),
        ("image_generate", "Generate images", "media", "When you need visual content"),
        ("email_send", "Send an email", "communication", None),
        ("calendar_check", "Check calendar", "productivity", None),
        ("translate", "Translate text", "language", None),
        ("summarize", "Summarize text", "language", None),
    ]

    class _SimpleProvider:
        """Minimal ToolProvider implementation for testing."""

        def __init__(self, spec: ToolSpec) -> None:
            self._spec = spec

        @property
        def spec(self) -> ToolSpec:
            return self._spec

        async def execute(self, call: ToolCall) -> ToolResult:
            return ToolResult(
                tool_call_id=call.id, name=call.name, success=True, output="mock"
            )

        async def health_check(self) -> bool:
            return True

    for name, desc, category, when in tool_defs:
        spec = ToolSpec(
            name=name,
            description=desc,
            input_schema={"type": "object", "properties": {}},
            category=category,
            when_to_use=when,
        )
        registry.register(_SimpleProvider(spec))

    print(f"  Total tools: {len(registry.list_tools())}")

    # Lazy select for a search goal
    lazy = LazyToolRegistry(registry, max_initial_tools=3)
    selected = lazy.select_initial_tools("Search the web for quantum computing papers")

    print(f"  Selected for 'search': {[t.name for t in selected]}")
    print(f"  Hidden: {lazy.available_but_hidden[:5]}...")

    # Format with affordance
    if selected:
        formatted = format_tool_for_llm(selected[0])
        print(f"  Formatted:\n{formatted}")

    print("  [OK] Lazy tool loading works!")


async def demo_5_diagnostic_recovery():
    """Demo 5: Diagnostic error recovery"""
    print("\n" + "=" * 60)
    print("Demo 5: Diagnostic Error Recovery")
    print("=" * 60)

    from arcana.contracts.tool import ErrorType, ToolError
    from arcana.runtime.diagnosis.diagnoser import build_recovery_prompt, diagnose_tool_error
    from arcana.runtime.diagnosis.tracker import RecoveryTracker

    # Simulate a tool-not-found error
    error = ToolError(
        error_type=ErrorType.NON_RETRYABLE,
        message="Tool 'search_web' not found in registry",
        code="TOOL_NOT_FOUND",
    )

    diagnosis = diagnose_tool_error(
        tool_name="search_web",
        tool_error=error,
        available_tools=["web_search", "file_read", "database_query"],
    )

    print(f"  Error category: {diagnosis.error_category.value}")
    print(f"  Error layer: {diagnosis.error_layer.value}")
    print(f"  Root cause: {diagnosis.root_cause}")
    print(f"  Strategy: {diagnosis.recommended_strategy.value}")
    print(f"  Suggestions: {diagnosis.actionable_suggestions}")

    # Build recovery prompt
    prompt = build_recovery_prompt(diagnosis, original_goal="Search for AI news")
    print(f"  Recovery prompt (first 200 chars): {prompt[:200]}...")

    # Test tracker
    tracker = RecoveryTracker()
    tracker.record(diagnosis)
    tracker.record(diagnosis)
    print(f"  Should escalate after 2 same errors: {tracker.should_escalate()}")

    print("  [OK] Diagnostic recovery works!")


async def demo_6_model_router():
    """Demo 6: Model router complexity estimation"""
    print("\n" + "=" * 60)
    print("Demo 6: Multi-Model Routing")
    print("=" * 60)

    from arcana.contracts.routing import RoutingConfig
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.gateway.router import ModelRouter

    router = ModelRouter(
        registry=ModelGatewayRegistry(),
        config=RoutingConfig(),
    )

    test_goals = [
        "What is 1+1?",
        "Explain quantum computing",
        "Design a microservice architecture for an e-commerce platform with high availability, compare three approaches, and implement the chosen one",
    ]

    for goal in test_goals:
        complexity = router.estimate_complexity(goal)
        role = router.select_role("think", complexity)
        config = router.get_config_for_role(role)
        print(
            f"  Goal: {goal[:60]:60s} -> {complexity.value:10s} -> {role.value:12s} -> {config.model_id}"
        )

    print("  [OK] Model routing works!")


async def main():
    """Run all demos."""
    print("Arcana v2 Demo")
    print("=" * 60)

    demos = [
        demo_1_direct_answer,
        demo_2_adaptive_policy,
        demo_3_tool_decorator,
        demo_4_lazy_tools,
        demo_5_diagnostic_recovery,
        demo_6_model_router,
    ]

    passed = 0
    for demo in demos:
        try:
            await demo()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(demos)} demos passed")
    if passed == len(demos):
        print("All demos passed! Arcana v2 is working.")
    else:
        print("Some demos failed. Check output above.")


if __name__ == "__main__":
    asyncio.run(main())
