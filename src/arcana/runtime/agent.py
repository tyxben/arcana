"""Main Agent class - execution orchestrator."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from arcana.contracts.intent import IntentType
from arcana.contracts.runtime import RuntimeConfig, StepResult, StepType
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
    from arcana.context.builder import WorkingSetBuilder
    from arcana.contracts.intent import IntentClassification
    from arcana.contracts.llm import ModelConfig
    from arcana.contracts.state import StateSnapshot
    from arcana.contracts.streaming import StreamEvent
    from arcana.gateway.budget import BudgetTracker
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.routing.classifier import IntentClassifier
    from arcana.runtime.hooks.base import RuntimeHook
    from arcana.runtime.policies.base import BasePolicy
    from arcana.runtime.progress import ProgressDetector
    from arcana.runtime.reducers.base import BaseReducer
    from arcana.runtime.state_manager import StateManager
    from arcana.runtime.step import StepExecutor
    from arcana.tool_gateway.gateway import ToolGateway
    from arcana.tool_gateway.lazy_registry import LazyToolRegistry
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
        intent_classifier: IntentClassifier | None = None,
        auto_route: bool = True,
        lazy_tools: bool = False,
        working_set_identity: str | None = None,
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
            intent_classifier: Optional intent classifier for fast-path routing
            auto_route: Whether to enable automatic intent routing (default True).
                        Routing only activates when an intent_classifier is also provided.
            lazy_tools: Whether to use lazy tool selection. When True and a
                        tool_gateway is provided, a LazyToolRegistry is created
                        at the start of each run to select an initial working
                        set of tools based on the goal.
            working_set_identity: Optional agent identity description for the
                        WorkingSetBuilder. When provided, the builder assembles
                        minimal context for each LLM call following the four-layer
                        model (identity/task/working/external). Token counts and
                        dropped-context keys are stored in working_memory for
                        policies to reference.
        """
        self.policy = policy
        self.reducer = reducer
        self.gateway = gateway
        self.config = config or RuntimeConfig()
        self.trace_writer = trace_writer
        self.budget_tracker = budget_tracker
        self.tool_gateway = tool_gateway
        self.hooks = hooks or []
        self.intent_classifier = intent_classifier
        self.auto_route = auto_route
        self.lazy_tools = lazy_tools

        # Internal components (lazy initialized)
        self._step_executor: StepExecutor | None = None
        self._state_manager: StateManager | None = None
        self._progress_detector: ProgressDetector | None = None
        self._lazy_registry: LazyToolRegistry | None = None

        # Working set builder (opt-in via working_set_identity)
        self._working_set_builder: WorkingSetBuilder | None = None
        if working_set_identity:
            from arcana.context.builder import WorkingSetBuilder as _WSBuilder

            self._working_set_builder = _WSBuilder(identity=working_set_identity)

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
        # V2: Intent routing (before agent loop)
        # Only activate when: auto_route is enabled, classifier is provided,
        # and this is a fresh run (not a resume with existing state).
        if self.auto_route and self.intent_classifier and initial_state is None:
            classification = await self.intent_classifier.classify(
                goal, available_tools=self._get_tool_names()
            )

            if classification.intent == IntentType.DIRECT_ANSWER:
                return await self._direct_answer(goal, task_id)

            if (
                classification.intent == IntentType.SINGLE_TOOL
                and classification.suggested_tools
            ):
                return await self._single_tool_answer(goal, classification, task_id)

            # AGENT_LOOP and COMPLEX_PLAN fall through to the existing loop

        # Lazy tool selection: narrow the tool working set before entering the loop
        if self.lazy_tools and self.tool_gateway:
            from arcana.tool_gateway.lazy_registry import LazyToolRegistry

            self._lazy_registry = LazyToolRegistry(self.tool_gateway.registry)
            initial_tools = self._lazy_registry.select_initial_tools(goal)
            # Record the curated tool names in working_memory for downstream use.
            # Note: this does NOT modify ToolGateway itself; the actual filtering
            # of exposed tools will be done at the StepExecutor layer in a future step.
            self._initial_tool_names = [t.name for t in initial_tools]

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

                # Build working set context (no-op when builder is not configured)
                state = self._enrich_state_with_working_set(state)

                # Execute single step
                step_result = await self._execute_step(state, trace_ctx)
                state = state.increment_step()

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
                checkpoint_reason = self._should_checkpoint(state, step_result)
                if checkpoint_reason:
                    await self.state_manager.checkpoint(
                        state, trace_ctx, reason=checkpoint_reason
                    )

                # Call hooks: on_step_complete
                await self._call_hooks("on_step_complete", state, step_result, trace_ctx)

        except Exception as e:
            state = await self._handle_error(state, e, trace_ctx)

        # Call hooks: on_run_end
        await self._call_hooks("on_run_end", state, trace_ctx)

        return state

    async def astream(
        self,
        goal: str,
        *,
        initial_state: AgentState | None = None,
        task_id: str | None = None,
        mode: str = "all",
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream agent execution events.

        Mirrors the logic of ``run()`` but yields ``StreamEvent`` objects at
        each execution milestone instead of returning a single final state.

        Args:
            goal: The goal to achieve.
            initial_state: Optional initial state (for resume).
            task_id: Optional task ID for grouping.
            mode: Stream filter mode. One of ``"all"``, ``"steps"``,
                ``"llm"``, ``"tools"``.

        Yields:
            StreamEvent objects at each execution milestone.
        """
        from arcana.contracts.streaming import StreamEvent, StreamEventType, matches_mode

        # ------------------------------------------------------------------
        # V2: Intent routing (before agent loop)
        # ------------------------------------------------------------------
        if self.auto_route and self.intent_classifier and initial_state is None:
            classification = await self.intent_classifier.classify(
                goal, available_tools=self._get_tool_names()
            )

            if classification.intent == IntentType.DIRECT_ANSWER:
                state = await self._direct_answer(goal, task_id)
                answer = state.working_memory.get("answer", "")

                yield StreamEvent(
                    event_type=StreamEventType.STEP_COMPLETE,
                    run_id=state.run_id,
                    content=answer,
                )
                yield StreamEvent(
                    event_type=StreamEventType.RUN_COMPLETE,
                    run_id=state.run_id,
                    content=answer,
                )
                return

            if (
                classification.intent == IntentType.SINGLE_TOOL
                and classification.suggested_tools
            ):
                state = await self._single_tool_answer(goal, classification, task_id)
                answer = state.working_memory.get("answer", "")

                yield StreamEvent(
                    event_type=StreamEventType.STEP_COMPLETE,
                    run_id=state.run_id,
                    content=answer,
                )
                yield StreamEvent(
                    event_type=StreamEventType.RUN_COMPLETE,
                    run_id=state.run_id,
                    content=answer,
                )
                return

            # AGENT_LOOP and COMPLEX_PLAN fall through to the loop

        # ------------------------------------------------------------------
        # Lazy tool selection
        # ------------------------------------------------------------------
        if self.lazy_tools and self.tool_gateway:
            from arcana.tool_gateway.lazy_registry import LazyToolRegistry

            self._lazy_registry = LazyToolRegistry(self.tool_gateway.registry)
            initial_tools = self._lazy_registry.select_initial_tools(goal)
            self._initial_tool_names = [t.name for t in initial_tools]

        # ------------------------------------------------------------------
        # Initialise state
        # ------------------------------------------------------------------
        state = initial_state or self._create_initial_state(goal, task_id)
        trace_ctx = TraceContext(run_id=state.run_id, task_id=task_id)

        # Emit RUN_START
        run_start_event = StreamEvent(
            event_type=StreamEventType.RUN_START,
            run_id=state.run_id,
            content=goal,
        )
        if matches_mode(run_start_event, mode):
            yield run_start_event

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

                # Build working set context (no-op when builder is not configured)
                state = self._enrich_state_with_working_set(state)

                # Emit STEP_START
                step_start_event = StreamEvent(
                    event_type=StreamEventType.STEP_START,
                    run_id=state.run_id,
                    step_id=str(state.current_step),
                    content=f"Step {state.current_step}",
                )
                if matches_mode(step_start_event, mode):
                    yield step_start_event

                # Execute single step
                step_result = await self._execute_step(state, trace_ctx)
                state = state.increment_step()

                # Update state via reducer
                state = await self.reducer.reduce(state, step_result)

                # Emit STEP_COMPLETE
                step_complete_event = StreamEvent(
                    event_type=StreamEventType.STEP_COMPLETE,
                    run_id=state.run_id,
                    step_id=str(state.current_step),
                    content=step_result.thought or step_result.action or "",
                    step_result_data=step_result.model_dump() if step_result else None,
                )
                if matches_mode(step_complete_event, mode):
                    yield step_complete_event

                # Emit TOOL_RESULT events
                if step_result.tool_results:
                    for tr in step_result.tool_results:
                        tool_event = StreamEvent(
                            event_type=StreamEventType.TOOL_RESULT,
                            run_id=state.run_id,
                            content=tr.output_str,
                            tool_result_data=tr.model_dump(),
                        )
                        if matches_mode(tool_event, mode):
                            yield tool_event

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
                checkpoint_reason = self._should_checkpoint(state, step_result)
                if checkpoint_reason:
                    await self.state_manager.checkpoint(
                        state, trace_ctx, reason=checkpoint_reason
                    )
                    checkpoint_event = StreamEvent(
                        event_type=StreamEventType.CHECKPOINT,
                        run_id=state.run_id,
                        content=checkpoint_reason,
                    )
                    if matches_mode(checkpoint_event, mode):
                        yield checkpoint_event

                # Call hooks: on_step_complete
                await self._call_hooks(
                    "on_step_complete", state, step_result, trace_ctx
                )

        except Exception as e:
            state = await self._handle_error(state, e, trace_ctx)
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                run_id=state.run_id,
                error=str(e),
            )

        # Emit RUN_COMPLETE
        run_complete_event = StreamEvent(
            event_type=StreamEventType.RUN_COMPLETE,
            run_id=state.run_id,
            content=state.working_memory.get("answer", ""),
            tokens_used=state.tokens_used,
            cost_usd=state.cost_usd,
        )
        if matches_mode(run_complete_event, mode):
            yield run_complete_event

        # Call hooks: on_run_end
        await self._call_hooks("on_run_end", state, trace_ctx)

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

    def _enrich_state_with_working_set(self, state: AgentState) -> AgentState:
        """Build working set context and store metadata in working_memory.

        When a WorkingSetBuilder is configured, this assembles the four-layer
        context (identity / task / working / external) for the current step
        and records token usage and dropped keys in ``working_memory`` so that
        downstream policies can adapt their behaviour.

        When no builder is configured this is a no-op and returns ``state``
        unchanged, preserving full backward compatibility.
        """
        if self._working_set_builder is None:
            return state

        from arcana.contracts.context import StepContext

        step_context = StepContext(
            step_type="think",
            needs_tools=self.tool_gateway is not None,
            previous_error=(
                {"recovery_prompt": state.last_error}
                if state.last_error
                else None
            ),
        )

        # Get tool descriptions if a lazy registry is active
        tool_descriptions: str | None = None
        if self.tool_gateway and self._lazy_registry is not None:
            from arcana.tool_gateway.formatter import format_tool_list_for_llm

            tool_descriptions = format_tool_list_for_llm(
                self._lazy_registry.working_set
            )

        # Serialise recent messages as plain dicts for the builder
        recent_history: list[dict[str, object]] | None = None
        if state.messages:
            recent_history = list(state.messages[-6:])

        working_set = self._working_set_builder.build(
            state=state,
            step_context=step_context,
            tool_descriptions=tool_descriptions,
            recent_history=recent_history,
        )

        # Merge working-set metadata into working_memory without clobbering
        # existing entries that other subsystems may rely on.
        return state.model_copy(
            update={
                "working_memory": {
                    **state.working_memory,
                    "_working_set_tokens": working_set.total_tokens,
                    "_dropped_context": working_set.dropped_keys,
                },
            }
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
    ) -> str | None:
        """
        Determine if we should create a checkpoint.

        Returns:
            Checkpoint reason string if checkpoint needed, None otherwise.
        """
        # Checkpoint on error
        if not step_result.success and self.config.checkpoint_on_error:
            return "error"

        # Checkpoint on plan step completion
        if (
            step_result.state_updates.get("plan_step_completed")
            and self.config.checkpoint_on_plan_step
        ):
            return "plan_step"

        # Checkpoint on verification step
        if (
            step_result.step_type == StepType.VERIFY
            and self.config.checkpoint_on_verification
        ):
            return "verification"

        # Checkpoint on interval
        if state.current_step % self.config.checkpoint_interval_steps == 0:
            return "interval"

        # Checkpoint on budget thresholds
        if self.budget_tracker:
            current_ratio = self._get_budget_ratio()
            for threshold in self.config.checkpoint_budget_thresholds:
                if (
                    self._last_checkpoint_budget_ratio < threshold
                    and current_ratio >= threshold
                ):
                    self._last_checkpoint_budget_ratio = current_ratio
                    return "budget"

        return None

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

    # ------------------------------------------------------------------
    # Fast-path methods for intent routing
    # ------------------------------------------------------------------

    async def _direct_answer(
        self, goal: str, task_id: str | None
    ) -> AgentState:
        """Fast path: single LLM call, no tools."""
        from arcana.routing.executor import DirectExecutor

        executor = DirectExecutor()
        config = self._get_default_config()
        response = await executor.direct_answer(goal, self.gateway, config)
        answer = (response.content or "").replace("[DONE]", "").replace("[done]", "").strip()

        tokens_used = response.usage.total_tokens if response.usage else 0
        if response.usage and self.budget_tracker:
            self.budget_tracker.add_usage(response.usage)

        state = self._create_initial_state(goal, task_id)
        state = state.model_copy(
            update={
                "status": ExecutionStatus.COMPLETED,
                "current_step": 1,
                "working_memory": {"answer": answer},
                "tokens_used": tokens_used,
                "cost_usd": self.budget_tracker.cost_usd if self.budget_tracker else 0.0,
            }
        )
        return state

    async def _single_tool_answer(
        self,
        goal: str,
        classification: IntentClassification,
        task_id: str | None,
    ) -> AgentState:
        """Fast path: one tool call + one LLM summary."""
        from arcana.routing.executor import DirectExecutor

        if not self.tool_gateway:
            # No tool gateway available; fall through to normal loop
            # by creating initial state and returning via the full run path.
            return await self._run_full_loop(goal, task_id=task_id)

        executor = DirectExecutor()
        config = self._get_default_config()
        answer = await executor.single_tool_call(
            goal=goal,
            tool_name=classification.suggested_tools[0],
            tool_args={},  # executor will ask LLM to generate args
            gateway=self.gateway,
            config=config,
            tool_gateway=self.tool_gateway,
        )

        state = self._create_initial_state(goal, task_id)
        state = state.model_copy(
            update={
                "status": ExecutionStatus.COMPLETED,
                "current_step": 2,
                "working_memory": {"answer": answer},
            }
        )
        return state

    async def _run_full_loop(
        self,
        goal: str,
        *,
        task_id: str | None = None,
    ) -> AgentState:
        """Run the full agent loop, bypassing intent routing."""
        state = self._create_initial_state(goal, task_id)
        # Re-enter run() with initial_state set, which skips routing
        return await self.run(goal, initial_state=state, task_id=task_id)

    def _get_tool_names(self) -> list[str] | None:
        """Get available tool names from the tool gateway."""
        if self.tool_gateway:
            return self.tool_gateway.registry.list_tools()
        return None

    def _get_default_config(self) -> ModelConfig:
        """Get a ModelConfig for the default provider.

        Resolution order:
        1. Provider's default_model attribute
        2. Raise ValueError -- never guess a hardcoded model name
        """
        from arcana.contracts.llm import ModelConfig

        provider_name = self.gateway.default_provider or "deepseek"
        provider = self.gateway.get(provider_name)
        model_id: str | None = None
        if provider and hasattr(provider, "default_model"):
            dm = provider.default_model
            if isinstance(dm, str) and dm:
                model_id = dm
        if not model_id:
            msg = (
                f"No default model configured for provider '{provider_name}'. "
                "Pass model_config explicitly or register a provider with a default_model."
            )
            raise ValueError(msg)
        return ModelConfig(provider=provider_name, model_id=model_id)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

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
