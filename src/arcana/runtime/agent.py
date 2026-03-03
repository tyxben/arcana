"""Main Agent class - execution orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from arcana.contracts.runtime import RuntimeConfig, StepResult
from arcana.contracts.state import AgentState, ExecutionStatus
from arcana.contracts.trace import (
    AgentRole,
    EventType,
    StopReason,
    TraceContext,
    TraceEvent,
)
from arcana.utils.hashing import canonical_hash

if TYPE_CHECKING:
    from arcana.contracts.state import StateSnapshot
    from arcana.gateway.budget import BudgetTracker
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.runtime.hooks.base import RuntimeHook
    from arcana.runtime.policies.base import BasePolicy
    from arcana.runtime.progress import ProgressDetector
    from arcana.runtime.reducers.base import BaseReducer
    from arcana.runtime.state_manager import StateManager
    from arcana.runtime.step import StepExecutor
    from arcana.tool_gateway.gateway import ToolGateway
    from arcana.trace.writer import TraceWriter


class Agent:
    """
    Main agent execution orchestrator.

    Implements the execution loop:
    Initialize -> Execute Step -> Update State -> Checkpoint -> Check Stop -> Repeat
    """

    def __init__(
        self,
        *,
        policy: BasePolicy,
        reducer: BaseReducer,
        gateway: ModelGatewayRegistry,
        config: RuntimeConfig | None = None,
        trace_writer: TraceWriter | None = None,
        budget_tracker: BudgetTracker | None = None,
        tool_gateway: ToolGateway | None = None,
        hooks: list[RuntimeHook] | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            policy: Policy that decides next actions
            reducer: Reducer that updates state
            gateway: Model gateway for LLM calls
            config: Runtime configuration
            trace_writer: Optional trace writer for event logging
            budget_tracker: Optional budget tracker for resource enforcement
            tool_gateway: Optional tool gateway for tool execution
            hooks: Optional list of runtime hooks
        """
        self.policy = policy
        self.reducer = reducer
        self.gateway = gateway
        self.config = config or RuntimeConfig()
        self.trace_writer = trace_writer
        self.budget_tracker = budget_tracker
        self.tool_gateway = tool_gateway
        self.hooks = hooks or []

        # Internal components (lazy initialized)
        self._step_executor: StepExecutor | None = None
        self._state_manager: StateManager | None = None
        self._progress_detector: ProgressDetector | None = None

        # Track last checkpoint budget for threshold detection
        self._last_checkpoint_budget_ratio: float = 0.0

    @property
    def step_executor(self) -> StepExecutor:
        """Get or create step executor."""
        if self._step_executor is None:
            from arcana.runtime.step import StepExecutor

            self._step_executor = StepExecutor(
                gateway=self.gateway,
                tool_gateway=self.tool_gateway,
                trace_writer=self.trace_writer,
                budget_tracker=self.budget_tracker,
            )
        return self._step_executor

    @property
    def state_manager(self) -> StateManager:
        """Get or create state manager."""
        if self._state_manager is None:
            from arcana.runtime.state_manager import StateManager

            self._state_manager = StateManager(
                trace_writer=self.trace_writer,
                config=self.config,
            )
        return self._state_manager

    @property
    def progress_detector(self) -> ProgressDetector:
        """Get or create progress detector."""
        if self._progress_detector is None:
            from arcana.runtime.progress import ProgressDetector

            self._progress_detector = ProgressDetector(
                window_size=self.config.progress_window_size,
                similarity_threshold=self.config.similarity_threshold,
            )
        return self._progress_detector

    async def run(
        self,
        goal: str,
        *,
        initial_state: AgentState | None = None,
        task_id: str | None = None,
    ) -> AgentState:
        """
        Run the agent until completion or stop condition.

        Args:
            goal: The goal to achieve
            initial_state: Optional initial state (for resume)
            task_id: Optional task ID for grouping

        Returns:
            Final agent state
        """
        # Initialize state
        state = initial_state or self._create_initial_state(goal, task_id)
        trace_ctx = TraceContext(run_id=state.run_id, task_id=task_id)

        # Start execution
        state = self.state_manager.transition(state, ExecutionStatus.RUNNING)
        state.start_time = datetime.now(UTC)

        # Call hooks: on_run_start
        await self._call_hooks("on_run_start", state, trace_ctx)

        try:
            # Main execution loop
            while True:
                # Check stop conditions before step
                stop_reason = self._check_stop_conditions(state)
                if stop_reason:
                    state = await self._handle_stop(state, stop_reason, trace_ctx)
                    break

                # Execute single step
                step_result = await self._execute_step(state, trace_ctx)

                # Update state via reducer
                state = await self.reducer.reduce(state, step_result)

                # Update progress tracking
                self.progress_detector.record_step(step_result)
                if not self.progress_detector.is_making_progress():
                    state.consecutive_no_progress += 1
                else:
                    state.consecutive_no_progress = 0

                # Check if step indicated goal completion
                if step_result.state_updates.get("goal_reached"):
                    state = await self._handle_stop(
                        state, StopReason.GOAL_REACHED, trace_ctx
                    )
                    break

                # Checkpoint if needed
                if self._should_checkpoint(state, step_result):
                    await self.state_manager.checkpoint(state, trace_ctx)

                # Call hooks: on_step_complete
                await self._call_hooks("on_step_complete", state, step_result, trace_ctx)

        except Exception as e:
            state = await self._handle_error(state, e, trace_ctx)

        # Call hooks: on_run_end
        await self._call_hooks("on_run_end", state, trace_ctx)

        return state

    async def resume(
        self,
        snapshot: StateSnapshot,
    ) -> AgentState:
        """
        Resume execution from a checkpoint.

        Args:
            snapshot: State snapshot to resume from

        Returns:
            Final agent state
        """
        # Verify hash integrity
        self.state_manager.verify_snapshot(snapshot)

        # Reset progress detector for resumed run
        self.progress_detector.reset()

        # Resume from snapshot state
        return await self.run(
            goal=snapshot.state.goal or "",
            initial_state=snapshot.state,
            task_id=snapshot.state.task_id,
        )

    async def _execute_step(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
    ) -> StepResult:
        """Execute a single step."""

        # Get policy decision
        decision = await self.policy.decide(state)

        # Execute based on decision
        step_result = await self.step_executor.execute(
            state=state,
            decision=decision,
            trace_ctx=trace_ctx,
        )

        state.increment_step()
        return step_result

    def _check_stop_conditions(self, state: AgentState) -> StopReason | None:
        """Check all stop conditions."""
        # Max steps
        if state.has_reached_max_steps:
            return StopReason.MAX_STEPS

        # No progress
        if state.consecutive_no_progress >= self.config.max_consecutive_no_progress:
            return StopReason.NO_PROGRESS

        # Budget checks
        if self.budget_tracker:
            try:
                self.budget_tracker.check_budget()
            except Exception as e:
                error_msg = str(e).lower()
                if "token" in error_msg:
                    return StopReason.MAX_TOKENS
                if "cost" in error_msg:
                    return StopReason.MAX_COST
                if "time" in error_msg:
                    return StopReason.MAX_TIME

        # Consecutive errors
        if state.consecutive_errors >= self.config.max_consecutive_errors:
            return StopReason.ERROR

        return None

    def _should_checkpoint(
        self,
        state: AgentState,
        step_result: StepResult,
    ) -> bool:
        """Determine if we should create a checkpoint."""
        # Checkpoint on error
        if not step_result.success and self.config.checkpoint_on_error:
            return True

        # Checkpoint on interval
        if state.current_step % self.config.checkpoint_interval_steps == 0:
            return True

        # Checkpoint on budget thresholds
        if self.budget_tracker:
            current_ratio = self._get_budget_ratio()
            for threshold in self.config.checkpoint_budget_thresholds:
                if (
                    self._last_checkpoint_budget_ratio < threshold
                    and current_ratio >= threshold
                ):
                    self._last_checkpoint_budget_ratio = current_ratio
                    return True

        return False

    def _get_budget_ratio(self) -> float:
        """Get current budget consumption ratio (0.0 to 1.0)."""
        if not self.budget_tracker:
            return 0.0

        snapshot = self.budget_tracker.to_snapshot()

        # Calculate ratio based on most constrained resource
        ratios = []
        if snapshot.max_tokens and snapshot.max_tokens > 0:
            ratios.append(snapshot.tokens_used / snapshot.max_tokens)
        if snapshot.max_cost_usd and snapshot.max_cost_usd > 0:
            ratios.append(snapshot.cost_usd / snapshot.max_cost_usd)
        if snapshot.max_time_ms and snapshot.max_time_ms > 0:
            ratios.append(snapshot.time_ms / snapshot.max_time_ms)

        return max(ratios) if ratios else 0.0

    def _create_initial_state(
        self,
        goal: str,
        task_id: str | None,
    ) -> AgentState:
        """Create initial agent state."""
        return AgentState(
            run_id=str(uuid4()),
            task_id=task_id,
            goal=goal,
            max_steps=self.config.max_steps,
        )

    async def _handle_stop(
        self,
        state: AgentState,
        reason: StopReason,
        trace_ctx: TraceContext,
    ) -> AgentState:
        """Handle stop condition."""
        if reason == StopReason.GOAL_REACHED:
            state = self.state_manager.transition(state, ExecutionStatus.COMPLETED)
        else:
            state = self.state_manager.transition(state, ExecutionStatus.FAILED)

        # Log stop event
        if self.trace_writer:
            event = TraceEvent(
                run_id=state.run_id,
                task_id=state.task_id,
                step_id=trace_ctx.new_step_id(),
                role=AgentRole.SYSTEM,
                event_type=EventType.STATE_CHANGE,
                stop_reason=reason,
                state_after_hash=canonical_hash(state.model_dump()),
            )
            self.trace_writer.write(event)

        return state

    async def _handle_error(
        self,
        state: AgentState,
        error: Exception,
        trace_ctx: TraceContext,
    ) -> AgentState:
        """Handle execution error."""
        state.last_error = str(error)
        state.consecutive_errors += 1
        state = self.state_manager.transition(state, ExecutionStatus.FAILED)

        # Log error event
        if self.trace_writer:
            event = TraceEvent(
                run_id=state.run_id,
                task_id=state.task_id,
                step_id=trace_ctx.new_step_id(),
                role=AgentRole.SYSTEM,
                event_type=EventType.ERROR,
                stop_reason=StopReason.ERROR,
                stop_detail=str(error),
                state_after_hash=canonical_hash(state.model_dump()),
            )
            self.trace_writer.write(event)

        return state

    async def _call_hooks(
        self,
        hook_name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Call all registered hooks."""
        for hook in self.hooks:
            method = getattr(hook, hook_name, None)
            if method:
                try:
                    await method(*args, **kwargs)
                except Exception:
                    # Log but don't fail on hook errors
                    pass
