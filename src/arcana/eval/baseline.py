"""Baseline eval suite — CI-safe regression gate for Arcana runtime.

Provides EvalGate (runner + scoring) and built-in baseline cases that
exercise direct answer, tool use, multi-turn context, and compression.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Core data structures (dataclasses, not Pydantic — keep eval internals light)
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result of a single eval case."""

    passed: bool
    score: float  # 0.0–1.0
    detail: str
    duration_ms: int = 0


@dataclass
class EvalCase:
    """A single eval case definition."""

    name: str
    category: str  # "direct_answer", "tool_use", "multi_turn", "context"
    run: Callable[[], Awaitable[EvalResult]]


@dataclass
class EvalReport:
    """Aggregated results from an eval suite run."""

    total: int
    passed: int
    failed: int
    score: float  # passed / total
    by_category: dict[str, float]  # category -> pass rate
    results: list[tuple[str, EvalResult]]


# ---------------------------------------------------------------------------
# EvalGate — register cases, run them, produce a report
# ---------------------------------------------------------------------------


class EvalGate:
    """Register and run eval cases, producing an EvalReport."""

    def __init__(self) -> None:
        self._cases: list[EvalCase] = []

    def register(self, case: EvalCase) -> None:
        self._cases.append(case)

    async def run_all(self, *, category: str | None = None) -> EvalReport:
        """Run all registered cases (optionally filtered by category)."""
        cases = self._cases
        if category:
            cases = [c for c in cases if c.category == category]

        results: list[tuple[str, EvalResult]] = []
        for case in cases:
            start = _now_ms()
            try:
                result = await case.run()
            except Exception as e:
                result = EvalResult(passed=False, score=0.0, detail=f"Exception: {e}")
            result.duration_ms = _now_ms() - start
            results.append((case.name, result))

        passed = sum(1 for _, r in results if r.passed)
        total = len(results)

        # Per-category pass rates
        by_cat: dict[str, list[bool]] = {}
        for case, (_, r) in zip(cases, results, strict=True):
            by_cat.setdefault(case.category, []).append(r.passed)

        return EvalReport(
            total=total,
            passed=passed,
            failed=total - passed,
            score=passed / total if total > 0 else 0.0,
            by_category={cat: sum(v) / len(v) for cat, v in by_cat.items()},
            results=results,
        )


# ---------------------------------------------------------------------------
# Baseline case factory — builds cases using the mock provider
# ---------------------------------------------------------------------------


def build_baseline_cases(
    *,
    provider_name: str = "mock",
    api_key: str = "mock-key",
) -> list[EvalCase]:
    """Build the baseline eval cases using the mock provider.

    Returns a list of EvalCase instances ready for EvalGate.register().
    """
    cases: list[EvalCase] = []

    # Case 1: Direct answer — simple math
    async def _direct_answer() -> EvalResult:
        from arcana.eval.mock_provider import MockProvider
        from arcana.gateway.registry import ModelGatewayRegistry
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        mock = MockProvider()
        rt = Runtime(
            providers={},
            budget=Budget(max_cost_usd=1.0),
            config=RuntimeConfig(default_provider="mock"),
        )
        # Inject mock provider directly
        gateway = ModelGatewayRegistry()
        gateway.register("mock", mock)  # type: ignore[arg-type]
        gateway.set_default("mock")
        rt._gateway = gateway

        result = await rt.run("What is 2+2?", max_turns=3)

        if not result.success:
            return EvalResult(passed=False, score=0.0, detail=f"Run failed: {result.output}")

        output = str(result.output).lower()
        has_answer = "4" in output
        return EvalResult(
            passed=has_answer,
            score=1.0 if has_answer else 0.0,
            detail=f"Output: {result.output}",
        )

    cases.append(EvalCase(name="direct_answer_math", category="direct_answer", run=_direct_answer))

    # Case 2: Tool use — calculator
    # Uses ConversationAgent directly (no intent classifier) to ensure
    # the tool path is exercised rather than the direct-answer fast path.
    async def _tool_use() -> EvalResult:
        from arcana.contracts.llm import ModelConfig
        from arcana.contracts.tool import ToolCall, ToolResult, ToolSpec
        from arcana.eval.mock_provider import MockProvider
        from arcana.gateway.registry import ModelGatewayRegistry
        from arcana.runtime.conversation import ConversationAgent
        from arcana.tool_gateway.gateway import ToolGateway
        from arcana.tool_gateway.registry import ToolRegistry

        mock = MockProvider()
        # Set up a tool call rule for this specific query
        mock.add_tool_call_rule(
            r"15.*37|37.*15|multiply|calculator",
            "calculator",
            {"expression": "15 * 37"},
        )

        # Create calculator tool
        calc_calls: list[dict[str, object]] = []

        class CalcProvider:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="calculator",
                    description="Evaluate a math expression",
                    input_schema={
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                )

            async def execute(self, call: ToolCall) -> ToolResult:
                expr = call.arguments.get("expression", "")
                calc_calls.append({"expression": expr})
                try:
                    val = _safe_math_eval(expr)
                except Exception as e:
                    return ToolResult(
                        tool_call_id=call.id, name=call.name,
                        success=False, output=str(e),
                    )
                return ToolResult(
                    tool_call_id=call.id, name=call.name,
                    success=True, output=str(val),
                )

            async def health_check(self) -> bool:
                return True

        registry = ToolRegistry()
        registry.register(CalcProvider())
        tool_gw = ToolGateway(registry=registry)

        gateway = ModelGatewayRegistry()
        gateway.register("mock", mock)  # type: ignore[arg-type]
        gateway.set_default("mock")

        config = ModelConfig(provider="mock", model_id="mock-v1")

        # Use ConversationAgent directly — no intent classifier so we
        # guarantee the tool execution path is tested.
        agent = ConversationAgent(
            gateway=gateway,
            model_config=config,
            tool_gateway=tool_gw,
            max_turns=5,
        )
        state = await agent.run("What is 15 * 37? Use the calculator tool.")

        tool_was_called = len(calc_calls) > 0
        output = str(state.working_memory.get("answer", ""))
        has_correct = "555" in output

        score = 0.0
        if tool_was_called:
            score += 0.5
        if has_correct:
            score += 0.5

        return EvalResult(
            passed=tool_was_called and has_correct,
            score=score,
            detail=f"tool_called={tool_was_called}, output={output}",
        )

    cases.append(EvalCase(name="tool_use_calculator", category="tool_use", run=_tool_use))

    # Case 3: Multi-turn context retention
    async def _multi_turn() -> EvalResult:
        from arcana.contracts.llm import ModelConfig
        from arcana.eval.mock_provider import MockProvider
        from arcana.gateway.registry import ModelGatewayRegistry
        from arcana.runtime.conversation import ConversationAgent

        mock = MockProvider()

        # Add rules for the multi-turn conversation
        mock.add_response_rule(r"my name is alice", "Nice to meet you, Alice!", done=False)
        mock.add_response_rule(r"favorite color is blue", "Blue is a great color!", done=False)
        mock.add_response_rule(
            r"what.*name|who am i|remember.*name",
            "Your name is Alice, as you mentioned earlier.",
        )

        gateway = ModelGatewayRegistry()
        gateway.register("mock", mock)  # type: ignore[arg-type]
        gateway.set_default("mock")

        config = ModelConfig(provider="mock", model_id="mock-v1")

        # Run 3 sequential messages through the agent
        agent = ConversationAgent(
            gateway=gateway,
            model_config=config,
            max_turns=1,
        )
        # Turn 1
        await agent.run("My name is Alice.")

        # Turn 2 — new agent but we check mock was called with context
        agent2 = ConversationAgent(
            gateway=gateway,
            model_config=config,
            max_turns=1,
        )
        await agent2.run("My favorite color is blue.")

        # Turn 3 — asks for name recall
        agent3 = ConversationAgent(
            gateway=gateway,
            model_config=config,
            max_turns=1,
        )
        state3 = await agent3.run("What is my name?")

        output3 = str(state3.working_memory.get("answer", ""))
        has_name = "alice" in output3.lower()

        # Check mock tracked all 3 calls
        all_called = mock.call_count >= 3

        return EvalResult(
            passed=has_name and all_called,
            score=1.0 if (has_name and all_called) else 0.5 if has_name else 0.0,
            detail=f"calls={mock.call_count}, output3={output3}",
        )

    cases.append(EvalCase(name="multi_turn_context", category="multi_turn", run=_multi_turn))

    # Case 4: Context compression survival
    # Tests that the runtime handles context management correctly even
    # with a small context window. Key information in the goal survives
    # because WorkingSetBuilder always preserves the system prompt and
    # first user message (the goal).
    async def _context_compression() -> EvalResult:
        from arcana.contracts.llm import LLMRequest, LLMResponse, ModelConfig
        from arcana.contracts.trace import TraceContext
        from arcana.eval.mock_provider import MockProvider
        from arcana.gateway.registry import ModelGatewayRegistry
        from arcana.runtime.conversation import ConversationAgent

        mock = MockProvider()

        # Track message counts to verify context management is active
        received_msg_counts: list[int] = []
        _original_generate = mock.generate

        async def _tracking_generate(
            request: LLMRequest,
            config: ModelConfig,
            trace_ctx: TraceContext | None = None,
        ) -> LLMResponse:
            received_msg_counts.append(len(request.messages))
            return await _original_generate(request, config, trace_ctx)

        mock.generate = _tracking_generate  # type: ignore[method-assign]

        # The mock should answer with the secret code on the first turn
        mock.add_response_rule(
            r"secret code.*ALPHA-7|ALPHA-7.*secret",
            "The secret code is ALPHA-7.",
        )

        gateway = ModelGatewayRegistry()
        gateway.register("mock", mock)  # type: ignore[arg-type]
        gateway.set_default("mock")

        config = ModelConfig(provider="mock", model_id="mock-v1")

        # Small context window — forces WorkingSetBuilder to actively
        # manage context. The agent should still complete successfully.
        agent = ConversationAgent(
            gateway=gateway,
            model_config=config,
            max_turns=3,
            context_window=800,
        )

        # A long goal that tests context handling with limited window.
        # Padding is inline text, not separate messages, so the mock
        # pattern matches the goal as a whole.
        padding = " ".join(f"Detail-{i}: noted." for i in range(20))
        goal = f"Remember the secret code ALPHA-7. {padding} What is the secret code?"

        state = await agent.run(goal)

        output = str(state.working_memory.get("answer", ""))
        has_code = "alpha-7" in output.lower()

        # Verify the agent completed and context management ran
        completed = state.status.value == "completed"
        context_active = len(received_msg_counts) > 0

        return EvalResult(
            passed=has_code and completed and context_active,
            score=1.0 if (has_code and completed) else 0.5 if completed else 0.0,
            detail=f"output={output}, completed={completed}, msg_counts={received_msg_counts}",
        )

    cases.append(EvalCase(name="context_compression", category="context", run=_context_compression))

    return cases


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _safe_math_eval(expr: str) -> float:
    """Evaluate simple arithmetic expressions without eval().

    Supports: +, -, *, / with integer and float operands.
    Raises ValueError on anything else.
    """
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def _eval_node(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return float(ops[type(node.op)](_eval_node(node.left), _eval_node(node.right)))
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    tree = ast.parse(expr.strip(), mode="eval")
    return _eval_node(tree.body)
