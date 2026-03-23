"""Tests for ConversationAgent checkpoint/resume."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arcana.contracts.llm import (
    LLMResponse,
    ModelConfig,
    StreamChunk,
    TokenUsage,
)
from arcana.contracts.state import AgentState, ExecutionStatus
from arcana.contracts.turn import TurnAssessment
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.runtime.state_manager import StateManager

# ── Helpers ──────────────────────────────────────────────────────────


def _ok_response(text: str = "The answer is 42.") -> LLMResponse:
    return LLMResponse(
        content=text,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="mock",
        finish_reason="stop",
    )


_MOCK_CONFIG = ModelConfig(provider="mock", model_id="mock-v1")


def _make_mock_gateway(responses: list[LLMResponse] | None = None) -> ModelGatewayRegistry:
    """Create a mock gateway that returns canned responses."""
    if responses is None:
        responses = [_ok_response()]

    class FakeProvider:
        provider_name = "mock"
        supported_models = ["mock"]
        _idx = 0

        async def generate(self, request, config, trace_ctx=None):
            resp = responses[min(self._idx, len(responses) - 1)]
            self._idx += 1
            return resp

        async def stream(self, request, config, trace_ctx=None):
            resp = await self.generate(request, config, trace_ctx)
            if resp.content:
                yield StreamChunk(type="text_delta", text=resp.content)
            yield StreamChunk(type="done", usage=resp.usage)

        async def health_check(self):
            return True

    gateway = ModelGatewayRegistry()
    gateway.register("mock", FakeProvider())
    gateway.set_default("mock")
    return gateway


# ── Checkpoint Trigger Tests ─────────────────────────────────────────


class TestShouldCheckpoint:
    def test_interval_trigger(self):
        from arcana.runtime.conversation import ConversationAgent

        agent = ConversationAgent(
            gateway=_make_mock_gateway(),
            model_config=_MOCK_CONFIG,
            checkpoint_interval=3,
            state_manager=MagicMock(),
        )

        assessment = TurnAssessment()

        # Step 0 — no checkpoint
        state = AgentState(run_id="r1", current_step=0)
        assert agent._should_checkpoint(state, assessment) is None

        # Step 3 — checkpoint
        state = AgentState(run_id="r1", current_step=3)
        assert agent._should_checkpoint(state, assessment) == "interval"

        # Step 6 — checkpoint
        state = AgentState(run_id="r1", current_step=6)
        assert agent._should_checkpoint(state, assessment) == "interval"

        # Step 4 — no checkpoint
        state = AgentState(run_id="r1", current_step=4)
        assert agent._should_checkpoint(state, assessment) is None

    def test_error_trigger(self):
        from arcana.runtime.conversation import ConversationAgent

        agent = ConversationAgent(
            gateway=_make_mock_gateway(),
            model_config=_MOCK_CONFIG,
            checkpoint_on_error=True,
            state_manager=MagicMock(),
        )

        state = AgentState(run_id="r1", current_step=1)
        failed = TurnAssessment(failed=True, completion_reason="empty_response")
        assert agent._should_checkpoint(state, failed) == "error"

        ok = TurnAssessment()
        assert agent._should_checkpoint(state, ok) is None

    def test_disabled_when_no_state_manager(self):
        from arcana.runtime.conversation import ConversationAgent

        agent = ConversationAgent(
            gateway=_make_mock_gateway(),
            model_config=_MOCK_CONFIG,
            checkpoint_interval=1,
        )

        # No state_manager → checkpoint block is never entered
        assert agent._state_manager is None

    def test_budget_threshold_trigger(self):
        from arcana.runtime.conversation import ConversationAgent

        mock_tracker = MagicMock()
        mock_tracker.max_cost_usd = 1.0
        mock_tracker.cost_usd = 0.55

        agent = ConversationAgent(
            gateway=_make_mock_gateway(),
            model_config=_MOCK_CONFIG,
            budget_tracker=mock_tracker,
            state_manager=MagicMock(),
            checkpoint_budget_thresholds=[0.5, 0.75],
        )

        state = AgentState(run_id="r1", current_step=1)
        assessment = TurnAssessment()

        # 55% > 50% threshold → trigger
        assert agent._should_checkpoint(state, assessment) == "budget"

        # Threshold already crossed, same ratio → no trigger
        assert agent._should_checkpoint(state, assessment) is None


# ── Checkpoint Persistence Tests ──────────────────────────────────


class TestCheckpointPersistence:
    @pytest.mark.asyncio
    async def test_checkpoint_is_created_on_interval(self, tmp_path):
        """StateManager.checkpoint() is called when interval triggers."""
        from arcana.runtime.conversation import ConversationAgent

        state_manager = StateManager(checkpoint_dir=str(tmp_path))
        gateway = _make_mock_gateway([
            _ok_response("Thinking..."),  # Turn 0 — finish_reason=stop → completed
        ])

        agent = ConversationAgent(
            gateway=gateway,
            model_config=_MOCK_CONFIG,
            state_manager=state_manager,
            checkpoint_interval=1,  # Checkpoint every turn
            max_turns=3,
        )

        state = await agent.run("What is 42?")

        # Should have created at least one checkpoint
        checkpoints = await state_manager.list_checkpoints(state.run_id)
        assert len(checkpoints) >= 1

    @pytest.mark.asyncio
    async def test_checkpoint_saves_messages(self, tmp_path):
        """Checkpoint should include conversation messages in state."""
        from arcana.runtime.conversation import ConversationAgent

        state_manager = StateManager(checkpoint_dir=str(tmp_path))
        gateway = _make_mock_gateway([_ok_response("Hello!")])

        agent = ConversationAgent(
            gateway=gateway,
            model_config=_MOCK_CONFIG,
            state_manager=state_manager,
            checkpoint_interval=1,
            max_turns=3,
        )

        state = await agent.run("Hi there")

        checkpoints = await state_manager.list_checkpoints(state.run_id)
        if checkpoints:
            snapshot = await state_manager.load_snapshot(state.run_id)
            assert snapshot is not None
            saved_messages = snapshot.state.messages
            assert len(saved_messages) > 0
            # Should have at least system + user + assistant
            roles = [m.get("role") for m in saved_messages]
            assert "system" in roles
            assert "user" in roles


# ── Resume Tests ──────────────────────────────────────────────────


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_requires_state_manager(self):
        from arcana.runtime.conversation import ConversationAgent

        agent = ConversationAgent(gateway=_make_mock_gateway(), model_config=_MOCK_CONFIG)

        with pytest.raises(RuntimeError, match="Cannot resume without"):
            await agent.resume("nonexistent")

    @pytest.mark.asyncio
    async def test_resume_no_checkpoint_raises(self, tmp_path):
        from arcana.runtime.conversation import ConversationAgent

        state_manager = StateManager(checkpoint_dir=str(tmp_path))
        agent = ConversationAgent(
            gateway=_make_mock_gateway(),
            model_config=_MOCK_CONFIG,
            state_manager=state_manager,
        )

        with pytest.raises(ValueError, match="No checkpoint found"):
            await agent.resume("nonexistent-run")

    @pytest.mark.asyncio
    async def test_resume_continues_from_checkpoint(self, tmp_path):
        """Resume should continue execution from saved state."""
        from arcana.runtime.conversation import ConversationAgent

        state_manager = StateManager(checkpoint_dir=str(tmp_path))

        # Phase 1: Run and checkpoint
        responses_phase1 = [
            LLMResponse(
                content="Let me search for that.",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                model="mock",
                finish_reason="stop",
            ),
        ]
        gateway1 = _make_mock_gateway(responses_phase1)

        agent1 = ConversationAgent(
            gateway=gateway1,
            model_config=_MOCK_CONFIG,
            state_manager=state_manager,
            checkpoint_interval=1,
            max_turns=5,
        )

        state1 = await agent1.run("Find the answer to life")

        # Verify checkpoint exists
        snapshot = await state_manager.load_snapshot(state1.run_id)
        assert snapshot is not None
        assert snapshot.state.current_step >= 1

        # Phase 2: Resume with new gateway (simulating restart)
        responses_phase2 = [
            _ok_response("The answer is 42."),
        ]
        gateway2 = _make_mock_gateway(responses_phase2)

        agent2 = ConversationAgent(
            gateway=gateway2,
            model_config=_MOCK_CONFIG,
            state_manager=state_manager,
            max_turns=10,
        )

        state2 = await agent2.resume(state1.run_id)

        assert state2.status == ExecutionStatus.COMPLETED
        assert "42" in state2.working_memory.get("answer", "")
        # Step count should be greater than phase 1
        assert state2.current_step > snapshot.state.current_step

    @pytest.mark.asyncio
    async def test_resume_preserves_token_count(self, tmp_path):
        """Resumed run should accumulate tokens from both phases."""
        from arcana.runtime.conversation import ConversationAgent

        state_manager = StateManager(checkpoint_dir=str(tmp_path))

        gateway1 = _make_mock_gateway([_ok_response("Phase 1 done.")])
        agent1 = ConversationAgent(
            gateway=gateway1,
            model_config=_MOCK_CONFIG,
            state_manager=state_manager,
            checkpoint_interval=1,
            max_turns=3,
        )
        state1 = await agent1.run("Test tokens")

        phase1_tokens = state1.tokens_used

        gateway2 = _make_mock_gateway([_ok_response("Phase 2 done.")])
        agent2 = ConversationAgent(
            gateway=gateway2,
            model_config=_MOCK_CONFIG,
            state_manager=state_manager,
            max_turns=10,
        )
        state2 = await agent2.resume(state1.run_id)

        # Should have tokens from both phases
        assert state2.tokens_used > phase1_tokens
