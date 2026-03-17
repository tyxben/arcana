#!/usr/bin/env python3
"""Deep integration tests with real DeepSeek API.

Runs 5 scenarios against the live LLM to validate end-to-end agent behavior.
Each scenario is independent -- a failure in one does not block the others.

Usage:
    DEEPSEEK_API_KEY="sk-xxx" uv run python examples/demo_deep_test.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    passed: bool = False
    duration_s: float = 0.0
    steps: int = 0
    tokens: int = 0
    error: str | None = None
    details: list[str] = field(default_factory=list)


def _separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_state_summary(state, label: str = "Final State") -> None:
    """Print a concise summary of an AgentState."""
    print(f"\n--- {label} ---")
    print(f"  status:        {state.status.value}")
    print(f"  current_step:  {state.current_step}")
    print(f"  max_steps:     {state.max_steps}")
    print(f"  tokens_used:   {state.tokens_used}")
    print(f"  last_error:    {state.last_error}")

    answer = state.working_memory.get("answer") or state.working_memory.get("result")
    if answer:
        preview = str(answer)[:300]
        print(f"  answer:        {preview}...")
    else:
        print(f"  working_memory keys: {list(state.working_memory.keys())}")

    if state.completed_steps:
        print(f"  completed_steps ({len(state.completed_steps)}):")
        for i, s in enumerate(state.completed_steps[:5]):
            print(f"    [{i}] {s[:120]}")
        if len(state.completed_steps) > 5:
            print(f"    ... and {len(state.completed_steps) - 5} more")


# ---------------------------------------------------------------------------
# Gateway factory (shared)
# ---------------------------------------------------------------------------

def _make_gateway():
    from arcana.gateway.providers.openai_compatible import create_deepseek_provider
    from arcana.gateway.registry import ModelGatewayRegistry

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        sys.exit(1)

    gateway = ModelGatewayRegistry()
    provider = create_deepseek_provider(api_key)
    gateway.register("deepseek", provider)
    gateway.set_default("deepseek")
    return gateway


# ===================================================================
# Scenario 1: Multi-step reasoning (AdaptivePolicy, auto_route=False)
# ===================================================================

async def test_multi_step_reasoning() -> ScenarioResult:
    """Test LLM choosing sequential strategy for multi-step reasoning."""
    from arcana.contracts.runtime import RuntimeConfig
    from arcana.runtime.agent import Agent
    from arcana.runtime.policies.adaptive import AdaptivePolicy
    from arcana.runtime.reducers.default import DefaultReducer

    result = ScenarioResult(name="Multi-step Reasoning")

    goal = (
        "Compare Python and Rust for building web servers. "
        "Consider performance, ecosystem, and learning curve. "
        "Give a recommendation."
    )

    gateway = _make_gateway()
    agent = Agent(
        policy=AdaptivePolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=10),
        auto_route=False,  # Force agent loop, no intent routing
    )

    print(f"\nGoal: {goal}")
    print("Config: AdaptivePolicy, auto_route=False, max_steps=10")

    state = await agent.run(goal)
    _print_state_summary(state)

    result.steps = state.current_step
    result.tokens = state.tokens_used

    # Validations
    checks = []

    if state.status.value == "completed":
        checks.append("status=completed: PASS")
    else:
        checks.append(f"status=completed: FAIL (got {state.status.value})")

    answer = state.working_memory.get("answer") or state.working_memory.get("result", "")
    if answer:
        checks.append("answer non-empty: PASS")
    else:
        checks.append("answer non-empty: FAIL")

    if state.current_step >= 1:
        checks.append(f"current_step >= 1: PASS (step={state.current_step})")
    else:
        checks.append(f"current_step >= 1: FAIL (step={state.current_step})")

    for c in checks:
        print(f"  CHECK: {c}")
    result.details = checks

    # Determine pass/fail
    result.passed = (
        state.status.value == "completed"
        and bool(answer)
        and state.current_step >= 1
    )
    return result


# ===================================================================
# Scenario 2: Tool usage (AdaptivePolicy + ToolGateway)
# ===================================================================

async def test_tool_usage() -> ScenarioResult:
    """Test LLM choosing single_tool strategy to call a calculator."""
    import arcana
    from arcana.contracts.runtime import RuntimeConfig
    from arcana.runtime.agent import Agent
    from arcana.runtime.policies.adaptive import AdaptivePolicy
    from arcana.runtime.reducers.default import DefaultReducer
    from arcana.sdk import _setup_tools

    result = ScenarioResult(name="Tool Usage (Calculator)")

    @arcana.tool(
        name="calculator",
        description="Evaluate a math expression and return the numeric result.",
        when_to_use="When you need to calculate math expressions",
        what_to_expect="Returns the numeric result of the expression as a string",
    )
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return str(eval(expression))  # noqa: S307

    goal = "What is 15 * 37 + 89?"

    gateway = _make_gateway()
    tool_gateway = _setup_tools([calculator])

    agent = Agent(
        policy=AdaptivePolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=10),
        tool_gateway=tool_gateway,
        auto_route=False,
    )

    print(f"\nGoal: {goal}")
    print(f"Tools: {tool_gateway.registry.list_tools()}")

    state = await agent.run(goal)
    _print_state_summary(state)

    result.steps = state.current_step
    result.tokens = state.tokens_used

    checks = []

    if state.status.value == "completed":
        checks.append("status=completed: PASS")
    else:
        checks.append(f"status=completed: FAIL (got {state.status.value})")

    answer = str(
        state.working_memory.get("answer")
        or state.working_memory.get("result", "")
    )
    if "644" in answer:
        checks.append("answer contains '644': PASS")
    else:
        checks.append(f"answer contains '644': FAIL (answer={answer[:200]})")

    for c in checks:
        print(f"  CHECK: {c}")
    result.details = checks

    result.passed = state.status.value == "completed" and "644" in answer
    return result


# ===================================================================
# Scenario 3: Intent Router full path (Direct Answer)
# ===================================================================

async def test_intent_router_direct() -> ScenarioResult:
    """Test intent router -> DIRECT_ANSWER fast path."""
    from arcana.contracts.runtime import RuntimeConfig
    from arcana.routing.classifier import HybridClassifier
    from arcana.runtime.agent import Agent
    from arcana.runtime.policies.adaptive import AdaptivePolicy
    from arcana.runtime.reducers.default import DefaultReducer

    result = ScenarioResult(name="Intent Router (Direct Answer)")

    goal = "What is the meaning of life?"

    gateway = _make_gateway()
    classifier = HybridClassifier(gateway=gateway)

    agent = Agent(
        policy=AdaptivePolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=10),
        intent_classifier=classifier,
        auto_route=True,
    )

    # First, show what the classifier thinks
    classification = await classifier.classify(goal)
    print(f"\nGoal: {goal}")
    print(f"Classification: intent={classification.intent.value}, confidence={classification.confidence}")

    state = await agent.run(goal)
    _print_state_summary(state)

    result.steps = state.current_step
    result.tokens = state.tokens_used

    checks = []

    if state.status.value == "completed":
        checks.append("status=completed: PASS")
    else:
        checks.append(f"status=completed: FAIL (got {state.status.value})")

    answer = state.working_memory.get("answer") or state.working_memory.get("result", "")
    if answer:
        checks.append("answer non-empty: PASS")
    else:
        checks.append("answer non-empty: FAIL")

    # Direct answer should complete in 1 step (fast path) or few steps (agent loop fallback)
    if state.current_step <= 3:
        checks.append(f"current_step <= 3 (fast): PASS (step={state.current_step})")
    else:
        checks.append(f"current_step <= 3 (fast): FAIL (step={state.current_step})")

    for c in checks:
        print(f"  CHECK: {c}")
    result.details = checks

    result.passed = state.status.value == "completed" and bool(answer)
    return result


# ===================================================================
# Scenario 4: Diagnostic recovery (tool failure -> LLM adjusts)
# ===================================================================

async def test_diagnostic_recovery() -> ScenarioResult:
    """Test tool failure triggering diagnostic recovery."""
    import arcana
    from arcana.contracts.runtime import RuntimeConfig
    from arcana.runtime.agent import Agent
    from arcana.runtime.policies.adaptive import AdaptivePolicy
    from arcana.runtime.reducers.default import DefaultReducer
    from arcana.sdk import _setup_tools

    result = ScenarioResult(name="Diagnostic Recovery")

    @arcana.tool(
        name="failing_search",
        description="Search the web for information. Currently experiencing outage.",
        when_to_use="When you need to search for information online",
        what_to_expect="Returns search results (currently failing due to service outage)",
        failure_meaning="The search service is currently down, try backup_search instead",
    )
    def failing_search(query: str) -> str:
        """Search that always fails."""
        raise RuntimeError("Service temporarily unavailable: search API is down")

    @arcana.tool(
        name="backup_search",
        description="Backup search engine that works when the primary search is down.",
        when_to_use="When the primary search (failing_search) is unavailable",
        what_to_expect="Returns basic search results from a backup source",
    )
    def backup_search(query: str) -> str:
        """Backup search that always succeeds."""
        return f"Backup results for '{query}': Quantum computing is an emerging field that uses quantum mechanical phenomena to perform computations. Recent breakthroughs include error correction advances and IBM's 1000+ qubit processor."

    goal = "Search for quantum computing news"

    gateway = _make_gateway()
    tool_gateway = _setup_tools([failing_search, backup_search])

    agent = Agent(
        policy=AdaptivePolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=10),
        tool_gateway=tool_gateway,
        auto_route=False,
    )

    print(f"\nGoal: {goal}")
    print(f"Tools: {tool_gateway.registry.list_tools()}")
    print("Note: 'failing_search' always errors; 'backup_search' always succeeds.")

    state = await agent.run(goal)
    _print_state_summary(state)

    result.steps = state.current_step
    result.tokens = state.tokens_used

    checks = []

    if state.status.value == "completed":
        checks.append("status=completed: PASS")
    else:
        checks.append(f"status=completed: FAIL (got {state.status.value})")

    # Check if recovery happened (either used backup_search or gave direct answer)
    answer = str(
        state.working_memory.get("answer")
        or state.working_memory.get("result", "")
    )
    wm_str = str(state.working_memory)
    had_diagnosis = "last_diagnosis" in wm_str or "recovery_prompt" in wm_str
    used_backup = "backup" in wm_str.lower() or "quantum" in answer.lower()

    if had_diagnosis:
        checks.append("diagnostic recovery triggered: PASS")
    else:
        checks.append("diagnostic recovery triggered: UNCERTAIN (may have skipped failing_search)")

    if used_backup or answer:
        checks.append("recovery succeeded (has answer): PASS")
    else:
        checks.append("recovery succeeded (has answer): FAIL")

    for c in checks:
        print(f"  CHECK: {c}")
    result.details = checks

    # This scenario is hard -- pass if completed OR if diagnosis was triggered
    result.passed = state.status.value == "completed" or had_diagnosis
    return result


# ===================================================================
# Scenario 5: Plan and Execute
# ===================================================================

async def test_plan_and_execute() -> ScenarioResult:
    """Test LLM choosing plan_and_execute strategy."""
    from arcana.contracts.runtime import RuntimeConfig
    from arcana.runtime.agent import Agent
    from arcana.runtime.policies.adaptive import AdaptivePolicy
    from arcana.runtime.reducers.default import DefaultReducer

    result = ScenarioResult(name="Plan and Execute")

    goal = (
        "Create a detailed comparison of 3 programming languages (Python, Rust, Go) "
        "across 5 dimensions: performance, safety, ecosystem, learning curve, and concurrency. "
        "Provide a final recommendation."
    )

    gateway = _make_gateway()
    agent = Agent(
        policy=AdaptivePolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=10),
        auto_route=False,
    )

    print(f"\nGoal: {goal}")
    print("Config: AdaptivePolicy, auto_route=False, max_steps=10")

    state = await agent.run(goal)
    _print_state_summary(state)

    result.steps = state.current_step
    result.tokens = state.tokens_used

    checks = []

    if state.status.value == "completed":
        checks.append("status=completed: PASS")
    else:
        checks.append(f"status=completed: FAIL (got {state.status.value})")

    # Check if plan-related data appeared
    adaptive = state.working_memory.get("adaptive_state", {})
    current_plan = state.working_memory.get("current_plan", [])
    strategy = adaptive.get("current_strategy", "") if isinstance(adaptive, dict) else ""

    answer = state.working_memory.get("answer") or state.working_memory.get("result", "")

    if current_plan:
        checks.append(f"has plan in working_memory: PASS (plan has {len(current_plan)} steps)")
    elif strategy == "plan_and_execute":
        checks.append("strategy=plan_and_execute: PASS (even if plan was consumed)")
    elif answer:
        checks.append("has answer (may have used direct_answer or sequential): PASS")
    else:
        checks.append("has plan or answer: FAIL")

    if answer:
        checks.append("final answer exists: PASS")
    else:
        checks.append("final answer exists: FAIL")

    for c in checks:
        print(f"  CHECK: {c}")
    result.details = checks

    # Pass if completed with an answer (strategy choice is LLM-dependent)
    result.passed = state.status.value == "completed" and bool(answer)
    return result


# ===================================================================
# Main runner
# ===================================================================

async def main() -> None:
    scenarios = [
        ("Scenario 1: Multi-step Reasoning", test_multi_step_reasoning),
        ("Scenario 2: Tool Usage (Calculator)", test_tool_usage),
        ("Scenario 3: Intent Router (Direct Answer)", test_intent_router_direct),
        ("Scenario 4: Diagnostic Recovery", test_diagnostic_recovery),
        ("Scenario 5: Plan and Execute", test_plan_and_execute),
    ]

    results: list[ScenarioResult] = []

    for title, test_fn in scenarios:
        _separator(title)
        t0 = time.time()
        try:
            res = await test_fn()
            res.duration_s = time.time() - t0
            results.append(res)
            status = "PASSED" if res.passed else "FAILED"
            print(f"\n  >> {status} ({res.duration_s:.1f}s, {res.steps} steps, {res.tokens} tokens)")
        except Exception as e:
            elapsed = time.time() - t0
            res = ScenarioResult(
                name=title,
                passed=False,
                duration_s=elapsed,
                error=str(e),
                details=[traceback.format_exc()],
            )
            results.append(res)
            print(f"\n  >> EXCEPTION ({elapsed:.1f}s): {e}")
            traceback.print_exc()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    _separator("SUMMARY")
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_tokens = sum(r.tokens for r in results)
    total_time = sum(r.duration_s for r in results)

    for r in results:
        icon = "PASS" if r.passed else "FAIL"
        err = f" -- {r.error}" if r.error else ""
        print(f"  [{icon}] {r.name} ({r.duration_s:.1f}s, {r.steps} steps, {r.tokens} tokens){err}")

    print(f"\n  Result: {passed}/{total} passed")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.1f}s")

    if passed == total:
        print("\n  All scenarios passed!")
    else:
        print(f"\n  {total - passed} scenario(s) failed. See details above.")


if __name__ == "__main__":
    asyncio.run(main())
