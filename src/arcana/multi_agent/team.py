"""TeamOrchestrator — coordinates Planner→Executor→Critic collaboration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcana.contracts.multi_agent import (
    AgentMessage,
    CollaborationSession,
    HandoffResult,
    MessageType,
)
from arcana.contracts.trace import AgentRole, EventType, TraceEvent
from arcana.multi_agent.message_bus import MessageBus
from arcana.runtime.agent import Agent

if TYPE_CHECKING:
    from arcana.contracts.state import AgentState
    from arcana.gateway.budget import BudgetTracker
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.runtime.hooks.base import RuntimeHook
    from arcana.runtime.policies.base import BasePolicy
    from arcana.runtime.reducers.base import BaseReducer
    from arcana.trace.writer import TraceWriter


class RoleConfig:
    """Configuration for a single agent role."""

    def __init__(
        self,
        *,
        role: AgentRole,
        policy: BasePolicy,
        reducer: BaseReducer,
        max_steps: int = 50,
    ) -> None:
        self.role = role
        self.policy = policy
        self.reducer = reducer
        self.max_steps = max_steps


class TeamOrchestrator:
    """
    Coordinates Planner→Executor→Critic collaboration loop.

    Flow per round:
    1. Planner creates plan from goal (+ optional feedback)
    2. Executor executes the plan
    3. Critic verifies the execution result
    4. If Critic rejects → loop back to step 1 with feedback
    5. If max_rounds reached → escalate
    """

    def __init__(
        self,
        role_configs: dict[AgentRole, RoleConfig],
        gateway: ModelGatewayRegistry,
        *,
        max_rounds: int = 5,
        trace_writer: TraceWriter | None = None,
        global_budget: BudgetTracker | None = None,
        hooks: list[RuntimeHook] | None = None,
    ) -> None:
        self._role_configs = role_configs
        self._gateway = gateway
        self._max_rounds = max_rounds
        self._trace_writer = trace_writer
        self._global_budget = global_budget
        self._hooks = hooks or []
        self._bus = MessageBus()

    async def run(self, goal: str) -> HandoffResult:
        """
        Run the collaboration loop until Critic approves or max rounds.

        Args:
            goal: The high-level goal to achieve.

        Returns:
            HandoffResult with final status and message history.
        """
        session = CollaborationSession(goal=goal, max_rounds=self._max_rounds)
        session_id = session.session_id

        total_tokens = 0
        total_cost = 0.0
        feedback: AgentMessage | None = None

        for round_num in range(1, self._max_rounds + 1):
            # 1. Planner
            planner_goal = goal
            if feedback:
                planner_goal = (
                    f"{goal}\n\nFeedback from previous round: "
                    f"{feedback.content.get('feedback', '')}"
                )

            planner_state = await self._run_role(
                AgentRole.PLANNER, planner_goal, session_id
            )
            total_tokens += planner_state.tokens_used
            total_cost += planner_state.cost_usd

            # Publish plan message
            plan_content = planner_state.working_memory.get("plan", planner_goal)
            plan_msg = AgentMessage(
                sender_role=AgentRole.PLANNER,
                recipient_role=AgentRole.EXECUTOR,
                message_type=MessageType.PLAN,
                content={"plan": plan_content},
                session_id=session_id,
            )
            await self._bus.publish(plan_msg)

            # 2. Executor
            executor_goal = f"Execute the following plan: {plan_content}"
            executor_state = await self._run_role(
                AgentRole.EXECUTOR, executor_goal, session_id
            )
            total_tokens += executor_state.tokens_used
            total_cost += executor_state.cost_usd

            # Publish result message
            result_content = executor_state.working_memory.get(
                "result", "Execution completed"
            )
            result_msg = AgentMessage(
                sender_role=AgentRole.EXECUTOR,
                recipient_role=AgentRole.CRITIC,
                message_type=MessageType.RESULT,
                content={"result": result_content},
                session_id=session_id,
            )
            await self._bus.publish(result_msg)

            # 3. Critic
            critic_goal = (
                f"Verify the following execution result: {result_content}\n"
                f"Original goal: {goal}"
            )
            critic_state = await self._run_role(
                AgentRole.CRITIC, critic_goal, session_id
            )
            total_tokens += critic_state.tokens_used
            total_cost += critic_state.cost_usd

            # Check verdict
            verdict = self._extract_verdict(critic_state)

            if verdict:
                # Critic approved
                self._write_trace_event(
                    session_id,
                    EventType.TASK_COMPLETE,
                    {"round": round_num, "verdict": "pass"},
                )
                return HandoffResult(
                    session_id=session_id,
                    final_status="completed",
                    rounds=round_num,
                    messages=self._bus.history(session_id),
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                )

            # Critic rejected — create feedback for next round
            feedback_content = critic_state.working_memory.get(
                "feedback", "Verification failed"
            )
            feedback = AgentMessage(
                sender_role=AgentRole.CRITIC,
                recipient_role=AgentRole.PLANNER,
                message_type=MessageType.FEEDBACK,
                content={"feedback": feedback_content},
                session_id=session_id,
            )
            await self._bus.publish(feedback)

            self._write_trace_event(
                session_id,
                EventType.TASK_FAIL,
                {"round": round_num, "verdict": "fail"},
            )

        # Max rounds exhausted — escalate
        escalate_msg = AgentMessage(
            sender_role=AgentRole.SYSTEM,
            recipient_role=AgentRole.SYSTEM,
            message_type=MessageType.ESCALATE,
            content={"reason": "max_rounds_exhausted", "goal": goal},
            session_id=session_id,
        )
        await self._bus.publish(escalate_msg)

        return HandoffResult(
            session_id=session_id,
            final_status="escalated",
            rounds=self._max_rounds,
            messages=self._bus.history(session_id),
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
        )

    async def _run_role(
        self,
        role: AgentRole,
        goal: str,
        session_id: str,
    ) -> AgentState:
        """Run a single agent for a given role."""
        agent = self._create_agent(role)

        self._write_trace_event(
            session_id,
            EventType.TASK_START,
            {"role": role.value, "goal": goal[:200]},
        )

        state = await agent.run(goal)
        return state

    def _create_agent(self, role: AgentRole) -> Agent:
        """Create an Agent instance configured for the given role."""
        from arcana.contracts.runtime import RuntimeConfig

        config = self._role_configs.get(role)
        if config is None:
            msg = f"No configuration for role: {role.value}"
            raise ValueError(msg)

        return Agent(
            policy=config.policy,
            reducer=config.reducer,
            gateway=self._gateway,
            config=RuntimeConfig(max_steps=config.max_steps),
            trace_writer=self._trace_writer,
            budget_tracker=self._global_budget,
            hooks=list(self._hooks),
        )

    @staticmethod
    def _extract_verdict(critic_state: AgentState) -> bool:
        """Extract pass/fail verdict from Critic's state."""
        verdict = critic_state.working_memory.get("verdict", "")
        if isinstance(verdict, bool):
            return verdict
        if isinstance(verdict, str):
            return verdict.lower() in ("pass", "true", "yes", "approved")
        return False

    def _write_trace_event(
        self,
        session_id: str,
        event_type: EventType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a trace event for team orchestration."""
        if self._trace_writer is None:
            return

        event = TraceEvent(
            run_id=session_id,
            role=AgentRole.SYSTEM,
            event_type=event_type,
            metadata=metadata or {},
        )
        self._trace_writer.write(event)
