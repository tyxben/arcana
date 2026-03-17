"""Integration tests for Agent Runtime."""

import pytest

from arcana.contracts.llm import LLMResponse, TokenUsage
from arcana.contracts.runtime import RuntimeConfig
from arcana.contracts.state import ExecutionStatus
from arcana.gateway.budget import BudgetTracker
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.runtime.agent import Agent
from arcana.runtime.policies.react import ReActPolicy
from arcana.runtime.reducers.default import DefaultReducer
from arcana.trace.writer import TraceWriter


class MockModelGateway:
    """Mock gateway for testing."""

    default_model = "mock-model"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["Thought: Test thought\nAction: FINISH"]
        self.call_count = 0

    async def generate(self, request, config, trace_ctx=None):
        """Generate mock response."""
        response_content = self.responses[
            min(self.call_count, len(self.responses) - 1)
        ]
        self.call_count += 1

        return LLMResponse(
            content=response_content,
            model="mock-model",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            finish_reason="stop",
        )


class TestAgentExecution:
    """Test basic agent execution."""

    @pytest.fixture
    def mock_gateway(self):
        """Create mock gateway."""
        gateway = ModelGatewayRegistry()
        gateway._providers["mock"] = MockModelGateway()
        gateway._default_provider = "mock"
        return gateway

    @pytest.fixture
    def agent(self, mock_gateway):
        """Create agent with mock components."""
        return Agent(
            policy=ReActPolicy(),
            reducer=DefaultReducer(),
            gateway=mock_gateway,
            config=RuntimeConfig(max_steps=5),
        )

    async def test_simple_execution(self, agent):
        """Test simple agent execution."""
        state = await agent.run("Test goal")

        assert state.status == ExecutionStatus.COMPLETED
        assert state.current_step >= 1

    async def test_max_steps_limit(self, mock_gateway):
        """Test max steps limit is enforced."""
        # Gateway that never finishes
        mock_gateway._providers["mock"] = MockModelGateway(
            responses=["Thought: Thinking\nAction: Continue"] * 10
        )

        agent = Agent(
            policy=ReActPolicy(),
            reducer=DefaultReducer(),
            gateway=mock_gateway,
            config=RuntimeConfig(max_steps=3),
        )

        state = await agent.run("Test goal")

        assert state.status == ExecutionStatus.FAILED
        assert state.current_step == 3

    async def test_token_budget_limit(self, mock_gateway, tmp_path):
        """Test token budget enforcement."""
        budget_tracker = BudgetTracker(max_tokens=50)

        agent = Agent(
            policy=ReActPolicy(),
            reducer=DefaultReducer(),
            gateway=mock_gateway,
            config=RuntimeConfig(max_steps=10),
            budget_tracker=budget_tracker,
        )

        # Each call uses 30 tokens, so should stop after 1-2 calls
        state = await agent.run("Test goal")

        assert state.current_step <= 2
        assert budget_tracker.to_snapshot().tokens_used <= 60


class TestProgressDetection:
    """Test progress detection."""

    @pytest.fixture
    def stuck_agent(self):
        """Create agent that gets stuck."""
        gateway = ModelGatewayRegistry()
        gateway._providers["mock"] = MockModelGateway(
            responses=["Thought: Same thought\nAction: Same action"] * 10
        )
        gateway._default_provider = "mock"

        return Agent(
            policy=ReActPolicy(),
            reducer=DefaultReducer(),
            gateway=gateway,
            config=RuntimeConfig(
                max_steps=10,
                max_consecutive_no_progress=3,
            ),
        )

    async def test_no_progress_detection(self, stuck_agent):
        """Test that no progress is detected."""
        state = await stuck_agent.run("Test goal")

        # Should stop due to no progress
        assert state.status == ExecutionStatus.FAILED
        assert state.consecutive_no_progress >= 3


class TestCheckpointing:
    """Test checkpointing functionality."""

    @pytest.fixture
    def agent_with_checkpoints(self, tmp_path):
        """Create agent with checkpoint directory."""
        gateway = ModelGatewayRegistry()
        gateway._providers["mock"] = MockModelGateway(
            responses=[
                "Thought: Step 1\nAction: Action 1",
                "Thought: Step 2\nAction: Action 2",
                "Thought: Step 3\nAction: FINISH",
            ]
        )
        gateway._default_provider = "mock"

        agent = Agent(
            policy=ReActPolicy(),
            reducer=DefaultReducer(),
            gateway=gateway,
            config=RuntimeConfig(
                max_steps=10,
                checkpoint_interval_steps=1,
            ),
        )

        # Configure checkpoint directory
        agent.state_manager.checkpoint_dir = tmp_path / "checkpoints"

        return agent

    async def test_creates_checkpoints(self, agent_with_checkpoints):
        """Test that checkpoints are created."""
        await agent_with_checkpoints.run("Test goal")

        # Check that checkpoints were created
        checkpoint_dir = agent_with_checkpoints.state_manager.checkpoint_dir
        checkpoint_files = list(checkpoint_dir.glob("*.checkpoints.jsonl"))

        assert len(checkpoint_files) > 0

    async def test_resume_from_checkpoint(self, agent_with_checkpoints, tmp_path):
        """Test resuming from checkpoint."""
        # First run
        state1 = await agent_with_checkpoints.run("Test goal")

        # Load latest checkpoint
        snapshot = await agent_with_checkpoints.state_manager.load_snapshot(
            state1.run_id
        )

        assert snapshot is not None
        assert snapshot.is_resumable or snapshot.state.status == ExecutionStatus.COMPLETED


class TestTracing:
    """Test trace logging."""

    async def test_trace_events_logged(self, tmp_path):
        """Test that trace events are logged."""
        trace_dir = tmp_path / "traces"
        trace_writer = TraceWriter(trace_dir=trace_dir)

        gateway = ModelGatewayRegistry()
        gateway._providers["mock"] = MockModelGateway()
        gateway._default_provider = "mock"

        agent = Agent(
            policy=ReActPolicy(),
            reducer=DefaultReducer(),
            gateway=gateway,
            config=RuntimeConfig(max_steps=5),
            trace_writer=trace_writer,
        )

        await agent.run("Test goal")

        # Check trace file was created
        trace_files = list(trace_dir.glob("*.jsonl"))
        assert len(trace_files) > 0


class TestErrorHandling:
    """Test error handling."""

    async def test_llm_error_handling(self):
        """Test handling of LLM errors."""

        class FailingGateway:
            """Gateway that always fails."""

            default_model = "mock-model"

            async def generate(self, request, config, trace_ctx=None):
                raise Exception("API Error")

        gateway = ModelGatewayRegistry()
        gateway._providers["failing"] = FailingGateway()
        gateway._default_provider = "failing"

        agent = Agent(
            policy=ReActPolicy(),
            reducer=DefaultReducer(),
            gateway=gateway,
            config=RuntimeConfig(max_steps=3, max_consecutive_errors=2),
        )

        state = await agent.run("Test goal")

        assert state.status == ExecutionStatus.FAILED
        assert state.last_error is not None


class TestValidator:
    """Test output validator."""

    def test_json_validation(self):
        """Test JSON validation."""
        from pydantic import BaseModel

        from arcana.runtime.validator import OutputValidator

        class TestSchema(BaseModel):
            name: str
            age: int

        validator = OutputValidator()

        # Valid JSON
        response = LLMResponse(
            content='{"name": "Alice", "age": 30}',
            model="test",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            finish_reason="stop",
        )

        result = validator.validate_json(response, TestSchema)
        assert result.valid
        assert result.data == {"name": "Alice", "age": 30}

    def test_json_validation_failure(self):
        """Test JSON validation with invalid data."""
        from pydantic import BaseModel

        from arcana.runtime.validator import OutputValidator

        class TestSchema(BaseModel):
            name: str
            age: int

        validator = OutputValidator()

        # Invalid JSON (missing field)
        response = LLMResponse(
            content='{"name": "Alice"}',
            model="test",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            finish_reason="stop",
        )

        result = validator.validate_json(response, TestSchema)
        assert not result.valid
        assert len(result.errors) > 0

    def test_structured_format_validation(self):
        """Test structured format validation."""
        from arcana.runtime.validator import OutputValidator

        validator = OutputValidator()

        response = LLMResponse(
            content="Thought: I need to do something\nAction: Take action",
            model="test",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            finish_reason="stop",
        )

        result = validator.validate_structured_format(response, ["thought", "action"])
        assert result.valid
        assert "thought" in result.data
        assert "action" in result.data
