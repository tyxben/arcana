"""Tests for the multi-agent module — MessageBus + TeamOrchestrator."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arcana.contracts.llm import LLMResponse, TokenUsage
from arcana.contracts.multi_agent import (
    AgentMessage,
    CollaborationSession,
    HandoffResult,
    MessageType,
)
from arcana.contracts.runtime import RuntimeConfig
from arcana.contracts.state import AgentState, ExecutionStatus
from arcana.contracts.trace import AgentRole, EventType
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.multi_agent.message_bus import MessageBus
from arcana.multi_agent.team import RoleConfig, TeamOrchestrator
from arcana.runtime.policies.react import ReActPolicy
from arcana.runtime.reducers.default import DefaultReducer


# ── Mock Helpers ─────────────────────────────────────────────────────


class _MockGateway:
    """Mock LLM gateway with configurable per-role responses."""

    def __init__(
        self,
        *,
        verdict: str = "pass",
        plan: str = "Step 1: Do thing",
        result: str = "Done",
    ) -> None:
        self._verdict = verdict
        self._plan = plan
        self._result = result
        self._call_count = 0

    async def generate(self, request, config, trace_ctx=None):
        self._call_count += 1
        # Detect role from the goal in messages
        content = "Thought: Done\nAction: FINISH"
        messages = getattr(request, "messages", [])
        goal_text = ""
        for msg in messages:
            if isinstance(msg, dict):
                goal_text += msg.get("content", "")

        if "Verify" in goal_text or "verify" in goal_text:
            # Critic response
            content = f"Thought: Verified\nAction: FINISH"
        elif "Execute" in goal_text or "execute" in goal_text:
            # Executor response
            content = f"Thought: Executing\nAction: FINISH"
        else:
            # Planner response
            content = f"Thought: Planning\nAction: FINISH"

        return LLMResponse(
            content=content,
            model="mock",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class _VerdictReducer(DefaultReducer):
    """Reducer that injects verdict into working_memory for Critic role."""

    def __init__(self, verdict: str = "pass") -> None:
        self._verdict = verdict

    @property
    def name(self) -> str:
        return "verdict-reducer"

    async def reduce(self, state, step_result):
        state = await super().reduce(state, step_result)
        state.working_memory["verdict"] = self._verdict
        return state


class _PlanReducerMock(DefaultReducer):
    """Reducer that injects a plan into working_memory for Planner role."""

    @property
    def name(self) -> str:
        return "plan-reducer-mock"

    async def reduce(self, state, step_result):
        state = await super().reduce(state, step_result)
        state.working_memory["plan"] = "Step 1: Do the thing"
        return state


class _ResultReducerMock(DefaultReducer):
    """Reducer that injects a result into working_memory for Executor role."""

    @property
    def name(self) -> str:
        return "result-reducer-mock"

    async def reduce(self, state, step_result):
        state = await super().reduce(state, step_result)
        state.working_memory["result"] = "Execution completed successfully"
        return state


def _make_gateway() -> ModelGatewayRegistry:
    registry = ModelGatewayRegistry()
    registry._providers["mock"] = _MockGateway()
    registry._default_provider = "mock"
    return registry


def _make_role_configs(
    verdict: str = "pass",
) -> dict[AgentRole, RoleConfig]:
    return {
        AgentRole.PLANNER: RoleConfig(
            role=AgentRole.PLANNER,
            policy=ReActPolicy(),
            reducer=_PlanReducerMock(),
            max_steps=3,
        ),
        AgentRole.EXECUTOR: RoleConfig(
            role=AgentRole.EXECUTOR,
            policy=ReActPolicy(),
            reducer=_ResultReducerMock(),
            max_steps=3,
        ),
        AgentRole.CRITIC: RoleConfig(
            role=AgentRole.CRITIC,
            policy=ReActPolicy(),
            reducer=_VerdictReducer(verdict=verdict),
            max_steps=3,
        ),
    }


# ── Contract Tests ───────────────────────────────────────────────────


class TestContracts:
    def test_agent_message_serialization(self):
        msg = AgentMessage(
            sender_role=AgentRole.PLANNER,
            recipient_role=AgentRole.EXECUTOR,
            message_type=MessageType.PLAN,
            content={"plan": "do stuff"},
            session_id="session-1",
        )
        data = msg.model_dump(mode="json")
        assert data["sender_role"] == "planner"
        assert data["message_type"] == "plan"
        restored = AgentMessage.model_validate(data)
        assert restored.session_id == "session-1"

    def test_collaboration_session_defaults(self):
        session = CollaborationSession(goal="test goal")
        assert session.max_rounds == 5
        assert AgentRole.PLANNER in session.roles
        assert AgentRole.EXECUTOR in session.roles
        assert AgentRole.CRITIC in session.roles
        assert session.status == "active"

    def test_handoff_result_serialization(self):
        result = HandoffResult(
            session_id="s1",
            final_status="completed",
            rounds=2,
            total_tokens=100,
            total_cost_usd=0.01,
        )
        data = result.model_dump()
        assert data["final_status"] == "completed"
        assert data["rounds"] == 2

    def test_message_type_values(self):
        assert MessageType.PLAN == "plan"
        assert MessageType.RESULT == "result"
        assert MessageType.FEEDBACK == "feedback"
        assert MessageType.HANDOFF == "handoff"
        assert MessageType.ESCALATE == "escalate"


# ── MessageBus Tests ─────────────────────────────────────────────────


class TestMessageBus:
    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self):
        bus = MessageBus()
        msg = AgentMessage(
            sender_role=AgentRole.PLANNER,
            recipient_role=AgentRole.EXECUTOR,
            message_type=MessageType.PLAN,
            content={"plan": "test"},
            session_id="s1",
        )
        await bus.publish(msg)
        messages = await bus.subscribe(AgentRole.EXECUTOR)
        assert len(messages) == 1
        assert messages[0].content["plan"] == "test"

    @pytest.mark.asyncio
    async def test_role_isolation(self):
        bus = MessageBus()
        msg = AgentMessage(
            sender_role=AgentRole.PLANNER,
            recipient_role=AgentRole.EXECUTOR,
            message_type=MessageType.PLAN,
            content={},
            session_id="s1",
        )
        await bus.publish(msg)

        # Critic should get nothing
        critic_msgs = await bus.subscribe(AgentRole.CRITIC)
        assert len(critic_msgs) == 0

        # Executor should get the message
        executor_msgs = await bus.subscribe(AgentRole.EXECUTOR)
        assert len(executor_msgs) == 1

    @pytest.mark.asyncio
    async def test_history(self):
        bus = MessageBus()
        for i in range(3):
            msg = AgentMessage(
                sender_role=AgentRole.PLANNER,
                recipient_role=AgentRole.EXECUTOR,
                message_type=MessageType.PLAN,
                content={"step": i},
                session_id="s1",
            )
            await bus.publish(msg)

        history = bus.history("s1")
        assert len(history) == 3
        assert history[0].content["step"] == 0

    @pytest.mark.asyncio
    async def test_empty_subscribe(self):
        bus = MessageBus()
        messages = await bus.subscribe(AgentRole.PLANNER)
        assert messages == []

    @pytest.mark.asyncio
    async def test_clear(self):
        bus = MessageBus()
        msg = AgentMessage(
            sender_role=AgentRole.PLANNER,
            recipient_role=AgentRole.EXECUTOR,
            message_type=MessageType.PLAN,
            content={},
            session_id="s1",
        )
        await bus.publish(msg)
        assert len(bus.history("s1")) == 1

        bus.clear("s1")
        assert len(bus.history("s1")) == 0


# ── TeamOrchestrator Tests ───────────────────────────────────────────


class TestTeamOrchestrator:
    @pytest.mark.asyncio
    async def test_single_round_success(self):
        """Planner→Executor→Critic all pass in one round."""
        configs = _make_role_configs(verdict="pass")
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=3)

        result = await team.run("Build a widget")

        assert result.final_status == "completed"
        assert result.rounds == 1
        assert result.total_tokens > 0
        assert len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_critic_reject_triggers_replan(self):
        """Critic rejects → Planner replans → eventually passes."""
        # Each agent runs 1 step (FINISH immediately), so reducer called once per agent run.
        # Critic reducer tracks its own call count.
        call_count = 0

        class _FlipVerdictReducer(DefaultReducer):
            @property
            def name(self):
                return "flip-verdict"

            async def reduce(self, state, step_result):
                nonlocal call_count
                state = await super().reduce(state, step_result)
                call_count += 1
                # Fail on first call (round 1), pass on second (round 2)
                state.working_memory["verdict"] = (
                    "pass" if call_count >= 2 else "fail"
                )
                return state

        configs = _make_role_configs()
        configs[AgentRole.CRITIC] = RoleConfig(
            role=AgentRole.CRITIC,
            policy=ReActPolicy(),
            reducer=_FlipVerdictReducer(),
            max_steps=3,
        )
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=5)

        result = await team.run("Build a widget")

        assert result.final_status == "completed"
        assert result.rounds == 2

    @pytest.mark.asyncio
    async def test_max_rounds_escalate(self):
        """All rounds fail → escalate."""
        configs = _make_role_configs(verdict="fail")
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=2)

        result = await team.run("Impossible task")

        assert result.final_status == "escalated"
        assert result.rounds == 2
        # Should have escalate message
        escalate_msgs = [
            m for m in result.messages if m.message_type == MessageType.ESCALATE
        ]
        assert len(escalate_msgs) == 1

    @pytest.mark.asyncio
    async def test_plan_passed_to_executor(self):
        """Planner's plan content is passed to Executor via message bus."""
        configs = _make_role_configs(verdict="pass")
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=1)

        result = await team.run("Test plan passing")

        plan_msgs = [
            m for m in result.messages if m.message_type == MessageType.PLAN
        ]
        assert len(plan_msgs) == 1
        assert "plan" in plan_msgs[0].content

    @pytest.mark.asyncio
    async def test_result_passed_to_critic(self):
        """Executor's result is passed to Critic via message bus."""
        configs = _make_role_configs(verdict="pass")
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=1)

        result = await team.run("Test result passing")

        result_msgs = [
            m for m in result.messages if m.message_type == MessageType.RESULT
        ]
        assert len(result_msgs) == 1
        assert "result" in result_msgs[0].content

    @pytest.mark.asyncio
    async def test_feedback_passed_to_planner(self):
        """Critic's feedback is included in next Planner round."""
        configs = _make_role_configs(verdict="fail")
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=2)

        result = await team.run("Test feedback")

        feedback_msgs = [
            m for m in result.messages if m.message_type == MessageType.FEEDBACK
        ]
        # Should have feedback from round 1 (round 2 also fails → escalate)
        assert len(feedback_msgs) >= 1

    @pytest.mark.asyncio
    async def test_handoff_result_token_tracking(self):
        """HandoffResult tracks total tokens and cost."""
        configs = _make_role_configs(verdict="pass")
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=1)

        result = await team.run("Track tokens")

        assert result.total_tokens > 0
        assert result.total_cost_usd >= 0.0

    @pytest.mark.asyncio
    async def test_missing_role_config_raises(self):
        """Missing role config raises ValueError."""
        configs = {
            AgentRole.PLANNER: RoleConfig(
                role=AgentRole.PLANNER,
                policy=ReActPolicy(),
                reducer=DefaultReducer(),
            ),
            # Missing EXECUTOR and CRITIC
        }
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway)

        with pytest.raises(ValueError, match="No configuration for role"):
            await team.run("Will fail")

    @pytest.mark.asyncio
    async def test_trace_events_written(self):
        """Trace events are written for team orchestration."""
        import tempfile
        from pathlib import Path

        from arcana.trace.writer import TraceWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TraceWriter(trace_dir=tmpdir)
            configs = _make_role_configs(verdict="pass")
            gateway = _make_gateway()
            team = TeamOrchestrator(
                configs, gateway, max_rounds=1, trace_writer=writer
            )

            result = await team.run("Trace test")

            # Check trace files exist
            from arcana.trace.reader import TraceReader

            reader = TraceReader(trace_dir=tmpdir)
            events = reader.read_events(result.session_id)
            assert len(events) > 0

    @pytest.mark.asyncio
    async def test_verdict_extraction_variants(self):
        """_extract_verdict handles various verdict formats."""
        # Boolean
        state = AgentState(run_id="r1", working_memory={"verdict": True})
        assert TeamOrchestrator._extract_verdict(state) is True

        state = AgentState(run_id="r1", working_memory={"verdict": False})
        assert TeamOrchestrator._extract_verdict(state) is False

        # String variants
        for v in ("pass", "true", "yes", "approved"):
            state = AgentState(run_id="r1", working_memory={"verdict": v})
            assert TeamOrchestrator._extract_verdict(state) is True

        for v in ("fail", "false", "no", "rejected"):
            state = AgentState(run_id="r1", working_memory={"verdict": v})
            assert TeamOrchestrator._extract_verdict(state) is False

        # Missing verdict
        state = AgentState(run_id="r1", working_memory={})
        assert TeamOrchestrator._extract_verdict(state) is False


# ── Integration Tests ────────────────────────────────────────────────


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_collaboration_flow(self):
        """Full Planner→Executor→Critic flow with message history."""
        configs = _make_role_configs(verdict="pass")
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=3)

        result = await team.run("Complete a 3-step task")

        assert result.final_status == "completed"
        assert result.rounds == 1

        # Verify message flow
        messages = result.messages
        types = [m.message_type for m in messages]
        assert MessageType.PLAN in types
        assert MessageType.RESULT in types

    @pytest.mark.asyncio
    async def test_multi_round_with_feedback(self):
        """Multi-round collaboration where Critic provides feedback."""
        # Critic reducer called once per round (FINISH on first step)
        round_counter = {"n": 0}

        class _RoundCountReducer(DefaultReducer):
            @property
            def name(self):
                return "round-count"

            async def reduce(self, state, step_result):
                state = await super().reduce(state, step_result)
                round_counter["n"] += 1
                # Pass on round 3 (3rd critic call)
                state.working_memory["verdict"] = (
                    "pass" if round_counter["n"] >= 3 else "fail"
                )
                return state

        configs = _make_role_configs()
        configs[AgentRole.CRITIC] = RoleConfig(
            role=AgentRole.CRITIC,
            policy=ReActPolicy(),
            reducer=_RoundCountReducer(),
            max_steps=3,
        )
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=5)

        result = await team.run("Multi-round task")

        assert result.final_status == "completed"
        assert result.rounds == 3

        # Should have feedback messages from rounds 1 and 2
        feedback_msgs = [
            m for m in result.messages if m.message_type == MessageType.FEEDBACK
        ]
        assert len(feedback_msgs) == 2

    @pytest.mark.asyncio
    async def test_message_history_completeness(self):
        """All messages are preserved in history."""
        configs = _make_role_configs(verdict="pass")
        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=1)

        result = await team.run("History test")

        # Single round: 1 plan + 1 result = 2 messages minimum
        assert len(result.messages) >= 2
        # All messages have the same session_id
        assert all(m.session_id == result.session_id for m in result.messages)

    @pytest.mark.asyncio
    async def test_independent_role_configs(self):
        """Each role gets its own policy/reducer configuration."""
        configs = _make_role_configs(verdict="pass")

        # Verify configs are independent
        assert configs[AgentRole.PLANNER].reducer.name == "plan-reducer-mock"
        assert configs[AgentRole.EXECUTOR].reducer.name == "result-reducer-mock"
        assert configs[AgentRole.CRITIC].reducer.name == "verdict-reducer"

        gateway = _make_gateway()
        team = TeamOrchestrator(configs, gateway, max_rounds=1)

        result = await team.run("Config test")
        assert result.final_status == "completed"


# ── Module Exports Test ──────────────────────────────────────────────


class TestModuleExports:
    def test_multi_agent_exports(self):
        import arcana.multi_agent as ma

        assert hasattr(ma, "TeamOrchestrator")
        assert hasattr(ma, "RoleConfig")
        assert hasattr(ma, "MessageBus")
        assert hasattr(ma, "AgentMessage")
        assert hasattr(ma, "CollaborationSession")
        assert hasattr(ma, "HandoffResult")
        assert hasattr(ma, "MessageType")
