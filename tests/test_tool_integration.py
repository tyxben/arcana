"""Integration tests for Tool Gateway + Runtime."""

import pytest

from arcana.contracts.llm import LLMResponse, TokenUsage
from arcana.contracts.runtime import RuntimeConfig
from arcana.contracts.state import ExecutionStatus
from arcana.contracts.tool import SideEffect, ToolCall, ToolResult, ToolSpec
from arcana.contracts.trace import TraceContext
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.runtime.agent import Agent
from arcana.runtime.policies.react import ReActPolicy
from arcana.runtime.reducers.default import DefaultReducer
from arcana.tool_gateway.base import ToolProvider
from arcana.tool_gateway.gateway import ToolGateway
from arcana.tool_gateway.registry import ToolRegistry
from arcana.trace.writer import TraceWriter


# ── Mock Providers ───────────────────────────────────────────────


class CalculatorTool(ToolProvider):
    """Simple calculator for integration tests."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Add two numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            side_effect=SideEffect.NONE,
            capabilities=[],
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        a = call.arguments["a"]
        b = call.arguments["b"]
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output={"result": a + b},
        )


class MockModelGateway:
    """Mock LLM gateway for integration tests."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["Thought: Done\nAction: FINISH"]
        self.call_count = 0

    async def generate(self, request, config, trace_ctx=None):
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return LLMResponse(
            content=response,
            model="mock",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            finish_reason="stop",
        )


# ── Integration Tests ────────────────────────────────────────────


class TestFullPipeline:
    """Test the complete tool execution pipeline."""

    @pytest.fixture
    def tool_registry(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        return reg

    @pytest.fixture
    def tool_gateway(self, tool_registry):
        return ToolGateway(
            registry=tool_registry,
            granted_capabilities=set(),
        )

    async def test_full_pipeline_read_tool(self, tool_gateway):
        """Test: register → authorize → validate → execute → result."""
        call = ToolCall(
            id="test-1",
            name="calculator",
            arguments={"a": 15, "b": 27},
        )
        ctx = TraceContext(run_id="integration-run")
        result = await tool_gateway.call(call, trace_ctx=ctx)

        assert result.success
        assert result.output == {"result": 42}

    async def test_pipeline_with_trace(self, tool_registry, tmp_path):
        """Test that trace events are written for tool calls."""
        trace_writer = TraceWriter(trace_dir=tmp_path)
        gw = ToolGateway(
            registry=tool_registry,
            trace_writer=trace_writer,
        )

        call = ToolCall(
            id="traced-1",
            name="calculator",
            arguments={"a": 1, "b": 2},
        )
        ctx = TraceContext(run_id="trace-run")
        result = await gw.call(call, trace_ctx=ctx)

        assert result.success
        trace_files = list(tmp_path.glob("*.jsonl"))
        assert len(trace_files) > 0

    async def test_pipeline_validation_rejects_bad_args(self, tool_gateway):
        """Test that validation rejects invalid arguments."""
        call = ToolCall(
            id="bad-args",
            name="calculator",
            arguments={"a": "not_a_number", "b": 2},
        )
        ctx = TraceContext(run_id="validation-run")
        result = await tool_gateway.call(call, trace_ctx=ctx)

        assert not result.success
        assert result.error is not None
        assert result.error.code == "VALIDATION_ERROR"


class TestStepExecutorWithToolGateway:
    """Test StepExecutor integration with ToolGateway."""

    async def test_step_executor_tool_call(self):
        """Test that StepExecutor correctly routes tool calls through ToolGateway."""
        from arcana.contracts.runtime import PolicyDecision, StepResult
        from arcana.contracts.state import AgentState
        from arcana.contracts.trace import TraceContext
        from arcana.runtime.step import StepExecutor

        # Setup
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        tool_gw = ToolGateway(registry=reg)

        model_gw = ModelGatewayRegistry()
        model_gw._providers["mock"] = MockModelGateway()
        model_gw._default_provider = "mock"

        executor = StepExecutor(
            gateway=model_gw,
            tool_gateway=tool_gw,
        )

        state = AgentState(run_id="test-run", max_steps=5)
        trace_ctx = TraceContext(run_id="test-run")

        decision = PolicyDecision(
            action_type="tool_call",
            tool_calls=[
                {"name": "calculator", "arguments": {"a": 10, "b": 20}},
            ],
        )

        result = await executor.execute(
            state=state,
            decision=decision,
            trace_ctx=trace_ctx,
        )

        assert result.success
        assert result.step_type.value == "act"
        assert len(result.tool_results) == 1
        assert result.tool_results[0].output == {"result": 30}

    async def test_step_executor_no_tool_gateway(self):
        """Test that StepExecutor returns error when ToolGateway not configured."""
        from arcana.contracts.runtime import PolicyDecision
        from arcana.contracts.state import AgentState
        from arcana.contracts.trace import TraceContext
        from arcana.runtime.step import StepExecutor

        model_gw = ModelGatewayRegistry()
        model_gw._providers["mock"] = MockModelGateway()
        model_gw._default_provider = "mock"

        executor = StepExecutor(gateway=model_gw)  # No tool_gateway

        state = AgentState(run_id="test-run", max_steps=5)
        trace_ctx = TraceContext(run_id="test-run")

        decision = PolicyDecision(
            action_type="tool_call",
            tool_calls=[{"name": "calculator", "arguments": {"a": 1, "b": 2}}],
        )

        result = await executor.execute(
            state=state,
            decision=decision,
            trace_ctx=trace_ctx,
        )

        assert not result.success
        assert "not configured" in result.error


class TestAgentWithToolGateway:
    """Test Agent with ToolGateway integration."""

    async def test_agent_accepts_tool_gateway(self):
        """Test that Agent can be constructed with tool_gateway parameter."""
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        tool_gw = ToolGateway(registry=reg)

        model_gw = ModelGatewayRegistry()
        model_gw._providers["mock"] = MockModelGateway()
        model_gw._default_provider = "mock"

        agent = Agent(
            policy=ReActPolicy(),
            reducer=DefaultReducer(),
            gateway=model_gw,
            tool_gateway=tool_gw,
            config=RuntimeConfig(max_steps=3),
        )

        # Verify tool_gateway is passed through to step_executor
        assert agent.tool_gateway is tool_gw
        assert agent.step_executor.tool_gateway is tool_gw

    async def test_agent_runs_without_tool_gateway(self):
        """Test that Agent still works without tool_gateway (backward compatible)."""
        model_gw = ModelGatewayRegistry()
        model_gw._providers["mock"] = MockModelGateway()
        model_gw._default_provider = "mock"

        agent = Agent(
            policy=ReActPolicy(),
            reducer=DefaultReducer(),
            gateway=model_gw,
            config=RuntimeConfig(max_steps=3),
        )

        state = await agent.run("Simple test")
        assert state.status == ExecutionStatus.COMPLETED
