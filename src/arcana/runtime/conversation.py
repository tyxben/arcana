"""ConversationAgent -- V2 execution model.

LLM-native conversation loop that replaces the Policy -> Step -> Reducer chain
with a single thin abstraction: LLM Turn -> Runtime Events -> State.

See specs/v2-architecture/conversation-loop.md for the full design.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from arcana.contracts.intent import IntentType
from arcana.contracts.llm import (
    ContentBlock,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    TokenUsage,
    ToolCallRequest,
)
from arcana.contracts.state import AgentState, ExecutionStatus
from arcana.contracts.streaming import StreamEvent, StreamEventType
from arcana.contracts.tool import ASK_USER_TOOL_NAME, ToolCall, ToolResult
from arcana.contracts.turn import TurnAssessment, TurnFacts

if TYPE_CHECKING:
    from arcana.contracts.channel import ExecutionChannel
    from arcana.gateway.budget import BudgetTracker
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.routing.classifier import IntentClassifier
    from arcana.runtime.state_manager import StateManager
    from arcana.tool_gateway.gateway import ToolGateway
    from arcana.tool_gateway.lazy_registry import LazyToolRegistry
    from arcana.trace.writer import TraceWriter

logger = logging.getLogger(__name__)

# Default system prompt — no artificial markers. The LLM stops when it's done.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's request directly and completely."
)

# Minimum characters for tool result truncation (safety floor).
_MIN_TOOL_RESULT_CHARS = 8000


class ConversationAgent:
    """V2 agent using LLM-native conversation as the execution model.

    The LLM drives the conversation. The runtime provides:
    - Tool execution (via ToolGateway)
    - Budget enforcement (via BudgetTracker)
    - Trace recording (via TraceWriter)
    - Completion detection (_assess_turn)
    - Error diagnosis

    The runtime does NOT:
    - Decide what strategy the LLM should use
    - Force a specific reasoning format
    - Require the LLM to output framework-specific JSON
    """

    def __init__(
        self,
        *,
        gateway: ModelGatewayRegistry,
        model_config: ModelConfig | None = None,
        tool_gateway: ToolGateway | None = None,
        budget_tracker: BudgetTracker | None = None,
        trace_writer: TraceWriter | None = None,
        intent_classifier: IntentClassifier | None = None,
        max_turns: int = 20,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        context_window: int = 128_000,
        memory_context: str | None = None,
        state_manager: StateManager | None = None,
        checkpoint_interval: int = 5,
        checkpoint_on_error: bool = True,
        checkpoint_budget_thresholds: list[float] | None = None,
        response_format_schema: dict[str, Any] | None = None,
        initial_user_content: list[ContentBlock] | None = None,
        input_handler: Callable[..., Any] | None = None,
        context_builder: Any | None = None,  # WorkingSetBuilder | None
        initial_messages: list[Message] | None = None,
        channel: ExecutionChannel | None = None,
        trace_include_prompt_snapshots: bool = False,
        cognitive_primitives: list[str] | None = None,
        pin_budget_fraction: float = 0.5,
    ) -> None:
        self.gateway = gateway
        self.model_config = model_config
        self.tool_gateway = tool_gateway
        self.budget_tracker = budget_tracker
        self.trace_writer = trace_writer
        self.intent_classifier = intent_classifier
        self.max_turns = max_turns
        self.system_prompt = system_prompt

        # Execution channel for Brain/Hands separation (optional, for future use)
        self._channel = channel

        # Opt-in trace verbosity: emit full prompt snapshots per LLM call
        self._trace_include_prompt_snapshots = trace_include_prompt_snapshots

        # Built-in ask_user tool handler
        from arcana.runtime.ask_user import AskUserHandler

        self._ask_user_handler = AskUserHandler(input_handler)

        # Pre-built initial messages (e.g. from ChatSession with history)
        self._initial_messages = initial_messages

        # Structured output schema (JSON Schema dict from Pydantic model)
        self._response_format_schema = response_format_schema

        # Multimodal: pre-built content blocks for the initial user message
        # (e.g. text + images). When set, _build_initial_messages uses this
        # instead of the plain goal string.
        self._initial_user_content: list[ContentBlock] | None = initial_user_content

        # Checkpoint configuration
        self._state_manager = state_manager
        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_on_error = checkpoint_on_error
        self._checkpoint_budget_thresholds = checkpoint_budget_thresholds or [0.5, 0.75, 0.9]
        self._last_checkpoint_budget_ratio = 0.0

        # Lazy tool registry — exposes only relevant tools to the LLM
        self._lazy_registry: LazyToolRegistry | None = None
        if tool_gateway is not None:
            from arcana.tool_gateway.lazy_registry import LazyToolRegistry as _LTR

            self._lazy_registry = _LTR(tool_gateway.registry)

        # Context management — use provided builder or create default
        if context_builder is not None:
            self._context_builder = context_builder
        else:
            from arcana.context.builder import WorkingSetBuilder
            from arcana.contracts.context import TokenBudget

            self._context_builder = WorkingSetBuilder(
                identity=system_prompt,
                token_budget=TokenBudget(total_window=context_window),
                goal=None,  # Set when run starts
                gateway=gateway,  # Pass gateway for LLM-based context compression
            )
        self._memory_context = memory_context

        # Cache ask_user schema token cost (static, never changes)
        self._ask_user_token_cost = self._estimate_tool_tokens(
            [self._ask_user_tool_schema()]
        )

        # Cognitive primitives (v0.7.0) — opt-in via RuntimeConfig.
        # The handler is always instantiated for API symmetry, but ``enabled``
        # controls whether any tool specs are injected or interception fires.
        from arcana.runtime.cognitive import CognitiveHandler

        self._cognitive_handler = CognitiveHandler(
            enabled=set(cognitive_primitives or []),
            pin_budget_fraction=pin_budget_fraction,
            total_token_window=self._context_builder.budget.total_window,
        )
        self._cognitive_token_cost = self._estimate_tool_tokens(
            self._cognitive_tool_schemas()
        )
        # Expose pins to the context builder when supported. Older builders
        # (e.g. test doubles) may not implement set_pin_state.
        if hasattr(self._context_builder, "set_pin_state"):
            self._context_builder.set_pin_state(self._cognitive_handler.pin_state)

        # Populated during a run; accessed by run() after astream() finishes.
        self._state: AgentState | None = None
        self._current_turn: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def channel(self) -> ExecutionChannel | None:
        """Execution channel for Brain/Hands separation (None = direct gateway calls)."""
        return self._channel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, goal: str) -> AgentState:
        """Run to completion by consuming astream().

        On cancellation (``asyncio.CancelledError``), the partially
        accumulated state is saved so that callers can still read
        ``tokens_used`` and ``cost_usd`` for budget tracking.
        """
        try:
            async for _event in self.astream(goal):
                pass  # All side-effects happen inside astream
        except asyncio.CancelledError:
            # astream may not have reached the end, so _state could be
            # unset.  Preserve whatever partial state we have.
            if self._state is None:
                # Build a minimal cancelled state so budget data survives.
                self._state = AgentState(
                    run_id="cancelled",
                    goal=goal,
                    status=ExecutionStatus.FAILED,
                    last_error="cancelled",
                )
            logger.info("ConversationAgent run cancelled for goal: %.80s", goal)
            raise
        assert self._state is not None, "astream must set _state before returning"
        return self._state

    async def astream(self, goal: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream execution events. This is the primary interface.

        Implements the 13-step turn contract from conversation-loop.md.
        """
        # ------------------------------------------------------------------
        # Phase 1: Optional intent routing
        # ------------------------------------------------------------------
        if self.intent_classifier and not self._response_format_schema:
            # Gather available tool names so the classifier can consider them
            available_tools: list[str] | None = None
            if self.tool_gateway and self.tool_gateway.registry:
                available_tools = self.tool_gateway.registry.list_tools()

            classification = await self.intent_classifier.classify(
                goal, available_tools=available_tools,
            )
            if classification.intent == IntentType.DIRECT_ANSWER:
                async for event in self._direct_answer(goal):
                    yield event
                return

        # ------------------------------------------------------------------
        # Phase 2: Initialise state and messages
        # ------------------------------------------------------------------
        run_id = str(uuid4())
        state = AgentState(
            run_id=run_id,
            goal=goal,
            max_steps=self.max_turns,
        )
        if self._initial_messages is not None:
            messages = list(self._initial_messages)  # Copy to avoid mutating caller's list
        else:
            messages = self._build_initial_messages(goal)
        self._context_builder.set_goal(goal)

        # Tool selection: lazy (subset) or eager (all)
        # ask_user is always injected regardless of path.
        active_tools: list[dict[str, Any]] | None
        if self._lazy_registry:
            self._lazy_registry.reset()
            self._lazy_registry.select_initial_tools(goal)
            active_tools = self._lazy_registry.to_openai_tools() or []
            active_tools.append(self._ask_user_tool_schema())
            active_tools.extend(self._cognitive_tool_schemas())
        else:
            active_tools = self._get_current_tools()
        if self._lazy_registry:
            tool_token_cost = (
                self._lazy_registry.tool_token_estimate
                + self._ask_user_token_cost
                + self._cognitive_token_cost
            )
        else:
            tool_token_cost = self._estimate_tool_tokens(active_tools)

        yield StreamEvent(
            event_type=StreamEventType.RUN_START,
            run_id=run_id,
            content=goal,
        )

        # ------------------------------------------------------------------
        # Phase 3: Conversation loop
        # ------------------------------------------------------------------
        prev_turn_step_id: str | None = None
        for _turn in range(self.max_turns):
            # ── Steps 1-6: happen EVERY turn ──

            # Track 1-indexed turn for cognitive primitives
            self._current_turn = _turn + 1

            # Pre-generate the step_id for this turn's TURN event so sibling
            # events (CONTEXT_DECISION, PROMPT_SNAPSHOT) and downstream
            # TOOL_CALL events can reference it as parent_step_id. This is
            # what makes ``arcana trace flow`` / ``arcana trace explain`` able
            # to stitch a turn's causes and effects.
            turn_step_id = str(uuid4())
            self._current_turn_step_id = turn_step_id
            self._prev_turn_step_id = prev_turn_step_id

            # Snapshot message count at turn start so recall can record
            # exactly what was added this turn.
            _turn_start_idx = len(messages)

            # 1. Budget check + consume iteration
            if self.budget_tracker:
                self.budget_tracker.check_budget()
                self.budget_tracker.consume_iteration()

            # 2. Context rebuild — delegate to WorkingSetBuilder
            #    Always include tools. Token optimization for tools belongs in
            #    LazyToolRegistry (dynamic tool selection), not here.
            #    Use async version for LLM-based compression when available.
            curated = await self._context_builder.abuild_conversation_context(
                messages,
                memory_context=self._memory_context if _turn == 0 else None,
                tool_token_estimate=tool_token_cost,
                turn=_turn,
            )
            # Write back so memory injection persists in messages for future turns
            if _turn == 0 and self._memory_context:
                messages = curated[:]

            # Store context report in state and yield CONTEXT_REPORT event
            if self._context_builder.last_report:
                ctx_report = self._context_builder.last_report
                # Update with lazy registry info
                if self._lazy_registry:
                    ctx_report = ctx_report.model_copy(update={
                        "tools_loaded": self._lazy_registry.loaded_count
                        if hasattr(self._lazy_registry, "loaded_count")
                        else len(self._lazy_registry.to_openai_tools() or []),
                        "tools_available": self._lazy_registry.total_count
                        if hasattr(self._lazy_registry, "total_count")
                        else (
                            len(self._lazy_registry._registry.list_tools())
                            if hasattr(self._lazy_registry, "_registry")
                            else 0
                        ),
                    })
                state = state.model_copy(update={"last_context_report": ctx_report})
                yield StreamEvent(
                    event_type=StreamEventType.CONTEXT_REPORT,
                    run_id=run_id,
                    step_id=str(state.current_step),
                    metadata=ctx_report.model_dump(),
                )

            # Trace the context decision (full structured evidence)
            self._emit_context_decision_event(state)

            # 3. LLM call
            request = LLMRequest(
                messages=curated,
                tools=active_tools,
                response_format=self._response_format_schema,
            )
            config = self._resolve_model_config()

            # Trace full prompt snapshot if opted in (off by default)
            self._emit_prompt_snapshot_event(state, request, config)

            response: LLMResponse
            if self._response_format_schema:
                # Structured output: use non-streaming for reliable usage
                # tracking and because we need complete JSON anyway
                response = await self.gateway.generate(
                    request=request, config=config,
                )
                # Still emit a chunk event for the complete text
                if response.content:
                    yield StreamEvent(
                        event_type=StreamEventType.LLM_CHUNK,
                        run_id=run_id,
                        step_id=str(state.current_step),
                        content=response.content,
                    )
            else:
                # Normal: try streaming, fall back to generate
                try:
                    from arcana.runtime.stream_accumulator import StreamAccumulator

                    acc = StreamAccumulator(model=config.model_id)

                    async for chunk in self.gateway.stream(
                        request=request, config=config,
                    ):
                        acc.feed(chunk)
                        if chunk.type == "text_delta" and chunk.text:
                            yield StreamEvent(
                                event_type=StreamEventType.LLM_CHUNK,
                                run_id=run_id,
                                step_id=str(state.current_step),
                                content=chunk.text,
                            )
                        elif chunk.type == "thinking_delta" and chunk.thinking:
                            yield StreamEvent(
                                event_type=StreamEventType.LLM_THINKING,
                                run_id=run_id,
                                step_id=str(state.current_step),
                                thinking=chunk.thinking,
                            )

                    response = acc.to_response()
                except (AttributeError, TypeError, NotImplementedError):
                    # Gateway doesn't support streaming — fall back to generate
                    response = await self.gateway.generate(
                        request=request, config=config,
                    )

            # Warn and estimate tokens if provider reported 0 usage
            if response.usage and response.usage.total_tokens == 0 and response.content:
                estimated_completion = len(response.content) // 4
                response = LLMResponse(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    usage=TokenUsage(
                        prompt_tokens=0,
                        completion_tokens=estimated_completion,
                        total_tokens=estimated_completion,
                    ),
                    model=response.model,
                    finish_reason=response.finish_reason,
                    raw_response=response.raw_response,
                    anthropic=response.anthropic,
                    gemini=response.gemini,
                    openai=response.openai,
                    ollama=response.ollama,
                )
                logger.warning(
                    "Provider reported 0 tokens; estimated %d from response length. "
                    "Cost tracking is approximate.",
                    estimated_completion,
                )

            # 4. Parse: response -> raw facts (NO interpretation)
            facts = self._parse_turn(response)

            # 5. Assess: runtime interprets facts (SEPARATE step)
            assessment = self._assess_turn(facts, state)

            # 6. Trace: record both
            self._trace_turn(facts, assessment, state)

            # ── Steps 7-10: conditional on facts + assessment ──

            yield StreamEvent(
                event_type=StreamEventType.STEP_COMPLETE,
                run_id=run_id,
                step_id=str(state.current_step),
                content=facts.assistant_text,
            )

            # 7a-10a. Tool calls
            if facts.tool_calls:
                logger.debug(
                    "LLM requested %d tool call(s): %s",
                    len(facts.tool_calls),
                    [tc.name for tc in facts.tool_calls],
                )
                # Expand lazy registry for any tools the LLM requested
                # (skip ask_user / cognitive primitives — they're built-in,
                # not in the registry)
                from arcana.runtime.cognitive import is_cognitive_tool

                if self._lazy_registry:
                    for tc in facts.tool_calls:
                        if tc.name != ASK_USER_TOOL_NAME and not is_cognitive_tool(tc.name):
                            self._lazy_registry.get_tool_on_demand(tc.name)
                    # Refresh active tools and token cost for next turn
                    active_tools = self._lazy_registry.to_openai_tools() or []
                    active_tools.append(self._ask_user_tool_schema())
                    active_tools.extend(self._cognitive_tool_schemas())
                    tool_token_cost = (
                        self._lazy_registry.tool_token_estimate
                        + self._ask_user_token_cost
                        + self._cognitive_token_cost
                    )

                # Yield TOOL_START events before execution
                import time as _time

                _tool_start_times: dict[str, float] = {}
                for tc in facts.tool_calls:
                    _tool_start_times[tc.id] = _time.monotonic()
                    try:
                        tc_args_parsed = json.loads(tc.arguments) if tc.arguments else {}
                    except json.JSONDecodeError:
                        tc_args_parsed = {}
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_START,
                        run_id=run_id,
                        step_id=str(state.current_step),
                        tool_name=tc.name,
                        tool_args=tc_args_parsed,
                    )

                results, ask_user_events = await self._execute_tools(
                    facts.tool_calls, run_id=run_id,
                )

                # Yield INPUT_NEEDED events from ask_user calls
                for ev in ask_user_events:
                    yield ev

                messages = self._add_tool_messages(messages, facts, results)

                # Yield TOOL_END events after execution
                for result in results:
                    duration_ms: int | None = None
                    start_t = _tool_start_times.get(result.tool_call_id or "")
                    if start_t is not None:
                        duration_ms = int((_time.monotonic() - start_t) * 1000)
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_END,
                        run_id=run_id,
                        step_id=str(state.current_step),
                        tool_name=result.name,
                        tool_result=result.output_str,
                        tool_duration_ms=duration_ms,
                    )

                for result in results:
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_RESULT,
                        run_id=run_id,
                        content=result.output_str,
                        tool_result_data=result.model_dump(),
                    )

                # 10a. Diagnose failures -> inject recovery context
                for result in results:
                    if not result.success:
                        diagnosis = self._diagnose_tool_failure(result)
                        messages.append(
                            Message(
                                role=MessageRole.USER,
                                content=diagnosis,
                            )
                        )
                        break  # one diagnosis per turn

            # 7b-9b. Completed
            elif assessment.completed:
                answer = (assessment.answer or facts.assistant_text or "").strip()
                state = state.model_copy(
                    update={
                        "status": ExecutionStatus.COMPLETED,
                        "working_memory": {
                            **state.working_memory,
                            "answer": answer,
                        },
                    }
                )

            # 7c-8c. Failed
            elif assessment.failed:
                state = state.model_copy(
                    update={
                        "status": ExecutionStatus.FAILED,
                        "last_error": assessment.completion_reason,
                    }
                )

            # 7d. Text only, not done -- intermediate reasoning
            elif facts.assistant_text:
                messages.append(
                    Message(role=MessageRole.ASSISTANT, content=facts.assistant_text)
                )

            # ── Steps 11-13: happen EVERY turn ──

            # 11. Increment turn
            state = state.increment_step()

            # 12. Track tokens
            if facts.usage:
                state = state.model_copy(
                    update={
                        "tokens_used": state.tokens_used + facts.usage.total_tokens,
                        "cost_usd": state.cost_usd + facts.usage.cost_estimate,
                    }
                )
                if self.budget_tracker:
                    self.budget_tracker.add_usage(facts.usage)

            # Yield TURN_END event
            yield StreamEvent(
                event_type=StreamEventType.TURN_END,
                run_id=run_id,
                step_id=str(state.current_step),
                turn_tokens=facts.usage.total_tokens if facts.usage else 0,
                turn_cost_usd=facts.usage.cost_estimate if facts.usage else 0.0,
            )

            # Record messages added this turn so `recall` can serve them.
            # When nothing was appended (terminal turn with completed text
            # only), include the assistant's final text as the turn's
            # single message so recall still works.
            turn_messages = [
                m.model_dump() for m in messages[_turn_start_idx:]
            ]
            if not turn_messages and facts.assistant_text:
                turn_messages = [
                    {
                        "role": "assistant",
                        "content": facts.assistant_text,
                    }
                ]
            self._cognitive_handler.record_turn(
                self._current_turn, turn_messages,
            )

            # Keep _state in sync so cancellation preserves budget data
            self._state = state

            # 13. Checkpoint — persist state for resume on crash/restart
            if self._state_manager:
                reason = self._should_checkpoint(state, assessment)
                if reason:
                    from arcana.contracts.trace import TraceContext as _TC

                    # Sync conversation messages into state for persistence
                    state = state.model_copy(
                        update={"messages": [
                            {"role": m.role.value if hasattr(m.role, "value") else m.role,
                             "content": m.content}
                            for m in messages
                        ]}
                    )
                    await self._state_manager.checkpoint(
                        state, _TC(run_id=state.run_id), reason=reason
                    )

            # Advance spine: next turn's TURN event links back to this one.
            prev_turn_step_id = turn_step_id

            # Exit if terminal
            if assessment.completed or assessment.failed:
                break

        # If loop exhausted without terminal state, mark failed
        if state.status not in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
            state = state.model_copy(
                update={
                    "status": ExecutionStatus.FAILED,
                    "last_error": f"max_turns ({self.max_turns}) exhausted",
                }
            )

        self._final_messages = messages
        self._state = state

        yield StreamEvent(
            event_type=StreamEventType.RUN_COMPLETE,
            run_id=run_id,
            content=state.working_memory.get("answer", ""),
            tokens_used=state.tokens_used,
            cost_usd=state.cost_usd,
        )

    @property
    def final_messages(self) -> list[Message]:
        """Messages at the end of the run, including all tool interactions."""
        return self._final_messages if hasattr(self, "_final_messages") else []

    # ------------------------------------------------------------------
    # Core two-step: parse + assess
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_turn(response: object) -> TurnFacts:
        """Parse provider response into raw facts. ONLY facts, NO interpretation.

        This function MUST NOT set completed/failed/answer.
        Pure, deterministic, provider-to-framework mapping.
        """
        from arcana.contracts.llm import LLMResponse

        if not isinstance(response, LLMResponse):  # pragma: no cover
            raise TypeError(f"Expected LLMResponse, got {type(response).__name__}")

        # Extract thinking content from provider extensions
        thinking: str | None = None
        provider_metadata: dict[str, object] = {}

        if response.anthropic:
            if response.anthropic.thinking_blocks:
                thinking = "\n".join(
                    tb.thinking for tb in response.anthropic.thinking_blocks
                )
            provider_metadata["anthropic"] = response.anthropic.model_dump()

        if response.gemini:
            if response.gemini.thinking_text:
                thinking = thinking or response.gemini.thinking_text
            provider_metadata["gemini"] = response.gemini.model_dump()

        return TurnFacts(
            assistant_text=response.content,
            tool_calls=response.tool_calls or [],
            usage=response.usage,
            finish_reason=response.finish_reason,
            thinking=thinking,
            provider_metadata=provider_metadata,
        )

    @staticmethod
    def _assess_turn(facts: TurnFacts, state: AgentState) -> TurnAssessment:
        """Runtime interprets the facts. Separate step, separate function.

        Trust the LLM's natural signals — no artificial markers needed.

        Rules, priority ordered:
        1. Tool calls present → not done (LLM wants to act)
        2. No text and no tools → failed
        3. finish_reason=stop + text → completed (LLM chose to stop)
        4. finish_reason=length → not done (LLM was cut off)
        """
        assessment = TurnAssessment()

        # Rule 1: tool calls present → LLM wants to use tools, not done
        if facts.tool_calls:
            return assessment

        # Rule 2: no output at all → something went wrong
        if not facts.assistant_text:
            assessment.failed = True
            assessment.completion_reason = "empty_response"
            return assessment

        text = facts.assistant_text.strip()

        # Rule 3: LLM stopped naturally with a response → completed.
        # The LLM chose to stop generating. It didn't request tools.
        # Trust its judgment — it said what it wanted to say.
        if facts.finish_reason == "stop" and text:
            assessment.completed = True
            assessment.answer = text
            assessment.completion_reason = "natural_stop"
            assessment.confidence = 0.85

            # Thinking-informed confidence adjustment
            if facts.thinking:
                thinking_lower = facts.thinking.lower()

                # Uncertainty signals → lower confidence
                uncertainty_patterns = [
                    "not sure", "uncertain", "might be wrong", "i'm guessing",
                    "hard to say", "unclear", "don't know", "not confident",
                    "不确定", "可能不对", "不太确定",
                ]
                if any(p in thinking_lower for p in uncertainty_patterns):
                    assessment.confidence *= 0.6

                # Verification intent → should NOT mark as complete yet
                verify_patterns = [
                    "need to verify", "should check", "let me confirm",
                    "i should validate", "need to double-check",
                    "需要验证", "应该确认",
                ]
                if any(p in thinking_lower for p in verify_patterns):
                    assessment.completed = False
                    assessment.completion_reason = "thinking_wants_verification"
                    return assessment

                # Incomplete information signals
                incomplete_patterns = [
                    "need more information", "missing data", "not enough context",
                    "i don't have", "would need to know",
                    "信息不足", "需要更多",
                ]
                if any(p in thinking_lower for p in incomplete_patterns):
                    assessment.confidence *= 0.5

            return assessment

        # Rule 4: finish_reason=length → LLM was cut off, not done
        # (Default: not done)
        return assessment

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    async def _execute_tools(
        self,
        tool_calls: list[ToolCallRequest],
        run_id: str = "",
    ) -> tuple[list[ToolResult], list[StreamEvent]]:
        """Execute tool calls via ToolGateway, intercepting ask_user.

        Returns a tuple of (results, extra_events) where extra_events
        contains any INPUT_NEEDED events that should be yielded.
        """
        results: list[ToolResult] = []
        extra_events: list[StreamEvent] = []

        # Log incoming tool calls
        for tc in tool_calls:
            logger.debug(
                "Tool call: %s(%s)",
                tc.name,
                tc.arguments[:200] if tc.arguments else "",
            )

        # Separate built-in tool calls from gateway calls
        from arcana.runtime.cognitive import is_cognitive_tool

        ask_user_calls: list[ToolCallRequest] = []
        cognitive_calls: list[ToolCallRequest] = []
        gateway_tool_calls: list[ToolCallRequest] = []

        for tc in tool_calls:
            if tc.name == ASK_USER_TOOL_NAME:
                ask_user_calls.append(tc)
            elif is_cognitive_tool(tc.name):
                cognitive_calls.append(tc)
            else:
                gateway_tool_calls.append(tc)

        # Handle ask_user calls directly (bypass ToolGateway)
        for tc in ask_user_calls:
            try:
                args = json.loads(tc.arguments) if tc.arguments else {}
            except json.JSONDecodeError:
                args = {}
            question = args.get("question", "")

            # Emit INPUT_NEEDED event before awaiting
            extra_events.append(StreamEvent(
                event_type=StreamEventType.INPUT_NEEDED,
                run_id=run_id,
                content=question,
            ))

            answer = await self._ask_user_handler.handle(question)
            results.append(ToolResult(
                tool_call_id=tc.id,
                name=ASK_USER_TOOL_NAME,
                success=True,
                output=answer,
            ))

        # Handle cognitive primitives (recall / pin / unpin) — bypass
        # ToolGateway. Errors are returned as structured tool results,
        # never as exceptions (Principle 5).
        for tc in cognitive_calls:
            result = self._handle_cognitive_tool_call(tc, run_id=run_id)
            results.append(result)

        # Handle regular tool calls via channel (preferred) or ToolGateway
        if gateway_tool_calls:
            if not self._channel and not self.tool_gateway:
                # No channel or tool gateway -- return synthetic errors
                for tc in gateway_tool_calls:
                    results.append(ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        success=False,
                        output=(
                            f"Tool '{tc.name}' cannot be executed: no tools are registered. "
                            f"Register tools with: Runtime(tools=[your_tool_function]) "
                            f"or @arcana.tool decorator."
                        ),
                    ))
            else:
                gw_calls: list[ToolCall] = []
                parent_step_id = getattr(self, "_current_turn_step_id", None)
                for tc in gateway_tool_calls:
                    try:
                        args = json.loads(tc.arguments) if tc.arguments else {}
                    except json.JSONDecodeError:
                        args = {"_raw": tc.arguments}
                    gw_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.name,
                            arguments=args,
                            run_id=run_id,
                            parent_step_id=parent_step_id,
                        )
                    )
                if self._channel:
                    gw_results = await self._channel.execute_many(gw_calls)
                else:
                    # Guarded by the `not self._channel and not self.tool_gateway`
                    # check above: if we reach this branch, tool_gateway is not None.
                    assert self.tool_gateway is not None
                    gw_results = await self.tool_gateway.call_many(gw_calls)
                results.extend(gw_results)

        # Log tool results
        for result in results:
            if result.success:
                logger.debug(
                    "Tool result: %s -> %s",
                    result.name,
                    result.output_str[:200] if result.output_str else "",
                )
            else:
                logger.debug(
                    "Tool failed: %s -> %s",
                    result.name,
                    result.output_str[:200] if result.output_str else "",
                )

        return results, extra_events

    def _handle_cognitive_tool_call(
        self,
        tc: ToolCallRequest,
        *,
        run_id: str,
    ) -> ToolResult:
        """Service a single recall/pin/unpin tool call.

        Dispatches to ``CognitiveHandler`` and emits a COGNITIVE_PRIMITIVE
        trace event with the request and result. Returns a ``ToolResult``
        whose ``output`` is the JSON of the structured response — the LLM
        can parse it and decide what to do next (Principle 5).
        """
        from arcana.contracts.cognitive import (
            PinRequest,
            RecallRequest,
            UnpinRequest,
        )
        from arcana.runtime.cognitive import (
            PIN_TOOL_NAME,
            RECALL_TOOL_NAME,
            UNPIN_TOOL_NAME,
        )

        try:
            args = json.loads(tc.arguments) if tc.arguments else {}
        except json.JSONDecodeError:
            args = {}

        result_payload: dict[str, Any]
        success = True
        try:
            if tc.name == RECALL_TOOL_NAME:
                recall_req = RecallRequest.model_validate(args)
                recall_result = self._cognitive_handler.handle_recall(recall_req)
                success = recall_result.found
                result_payload = recall_result.model_dump()
            elif tc.name == PIN_TOOL_NAME:
                pin_req = PinRequest.model_validate(args)
                pin_result = self._cognitive_handler.handle_pin(
                    pin_req, current_turn=self._current_turn,
                )
                success = pin_result.pinned
                result_payload = pin_result.model_dump()
                # When a new pin is accepted, let the context builder see it.
                if pin_result.pinned and hasattr(
                    self._context_builder, "set_pin_state",
                ):
                    self._context_builder.set_pin_state(
                        self._cognitive_handler.pin_state,
                    )
            elif tc.name == UNPIN_TOOL_NAME:
                unpin_req = UnpinRequest.model_validate(args)
                unpin_result = self._cognitive_handler.handle_unpin(unpin_req)
                success = unpin_result.unpinned
                result_payload = unpin_result.model_dump()
                if hasattr(self._context_builder, "set_pin_state"):
                    self._context_builder.set_pin_state(
                        self._cognitive_handler.pin_state,
                    )
            else:  # pragma: no cover — defensive
                result_payload = {
                    "error": "unknown_cognitive_primitive",
                    "name": tc.name,
                }
                success = False
        except Exception as e:  # noqa: BLE001 — invalid args → structured result
            result_payload = {
                "error": "invalid_arguments",
                "detail": str(e),
            }
            success = False

        self._emit_cognitive_primitive_event(
            run_id=run_id,
            primitive=tc.name,
            args=args,
            result=result_payload,
        )

        return ToolResult(
            tool_call_id=tc.id,
            name=tc.name,
            success=success,
            output=json.dumps(result_payload, ensure_ascii=False),
        )

    def _emit_cognitive_primitive_event(
        self,
        *,
        run_id: str,
        primitive: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Emit a COGNITIVE_PRIMITIVE trace event for observability."""
        if not self.trace_writer:
            return
        from arcana.contracts.trace import EventType, TraceEvent

        self.trace_writer.write(
            TraceEvent(
                run_id=run_id,
                parent_step_id=getattr(self, "_current_turn_step_id", None),
                event_type=EventType.COGNITIVE_PRIMITIVE,
                metadata={
                    "primitive": primitive,
                    "args": args,
                    "result": result,
                },
            )
        )

    def _add_tool_messages(
        self,
        messages: list[Message],
        facts: TurnFacts,
        results: list[ToolResult],
    ) -> list[Message]:
        """Append assistant tool_call message + tool result messages.

        Uses OpenAI-compatible format (tool_calls as separate field,
        tool results as role=TOOL messages with tool_call_id).
        This works with DeepSeek, OpenAI, Gemini-compat, and other
        OpenAI-format providers.
        """
        # Native tool conversation format:
        # 1. Assistant message with tool_calls (OpenAI format)
        # 2. Tool result messages with tool_call_id

        from arcana.contracts.llm import ToolCallRequest

        # Build assistant message with tool_calls
        tool_call_requests = [
            ToolCallRequest(id=tc.id, name=tc.name, arguments=tc.arguments)
            for tc in facts.tool_calls
        ]
        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=facts.assistant_text,
                tool_calls=tool_call_requests,
            )
        )

        # Dynamic truncation limit: context window size in chars (~1 char per token).
        # This is a safety net — context builder compression handles the rest next turn.
        max_chars = max(
            self._context_builder.budget.total_window,
            _MIN_TOOL_RESULT_CHARS,
        )

        # Tool result messages (one per result, truncated if oversized)
        for result in results:
            content = result.output_str
            if content and len(content) > max_chars:
                content = (
                    content[:max_chars]
                    + f"\n[truncated: {len(result.output_str)} chars total]"
                )
            messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=content,
                    tool_call_id=result.tool_call_id,
                )
            )

        return messages

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_initial_messages(self, goal: str) -> list[Message]:
        """Build the initial message list: system + user goal.

        When ``_initial_user_content`` is set (multimodal input), the user
        message uses the pre-built content blocks instead of the plain
        goal string.  This allows images and other content types to be
        included in the first user message.
        """
        msgs: list[Message] = []
        if self.system_prompt:
            msgs.append(Message(role=MessageRole.SYSTEM, content=self.system_prompt))
        # Use multimodal content blocks when available, otherwise plain text
        user_content: str | list[ContentBlock] = self._initial_user_content if self._initial_user_content else goal
        msgs.append(Message(role=MessageRole.USER, content=user_content))
        return msgs

    def _get_current_tools(self) -> list[dict[str, Any]] | None:
        """Get tool schemas in OpenAI format from ToolGateway.

        Always includes the built-in ask_user tool, even when no
        other tools are registered. Cognitive primitive tools
        (recall / pin / unpin) are injected only when enabled via
        ``RuntimeConfig.cognitive_primitives``.
        """
        tool_defs: list[dict[str, Any]] = []
        if self.tool_gateway:
            tool_defs = self.tool_gateway.registry.to_openai_tools()

        # Always inject ask_user
        tool_defs.append(self._ask_user_tool_schema())

        # Inject cognitive primitive tools when enabled
        tool_defs.extend(self._cognitive_tool_schemas())

        return tool_defs if tool_defs else None

    @staticmethod
    def _ask_user_tool_schema() -> dict[str, object]:
        """Return the ask_user tool definition in OpenAI function calling format."""
        from arcana.runtime.ask_user import ASK_USER_SPEC
        from arcana.tool_gateway.formatter import format_tool_for_llm

        return {
            "type": "function",
            "function": {
                "name": ASK_USER_SPEC.name,
                "description": format_tool_for_llm(ASK_USER_SPEC),
                "parameters": ASK_USER_SPEC.input_schema,
            },
        }

    def _cognitive_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool definitions for enabled cognitive primitives.

        When ``cognitive_primitives`` is empty (default), returns an empty
        list and no tools are injected into the LLM's tool list. If ``pin``
        is enabled, ``unpin`` is also exposed (they form a symmetric pair).
        """
        from arcana.runtime.cognitive import (
            PIN_SPEC,
            PIN_TOOL_NAME,
            RECALL_SPEC,
            RECALL_TOOL_NAME,
            UNPIN_SPEC,
        )
        from arcana.tool_gateway.formatter import format_tool_for_llm

        specs: list[Any] = []
        enabled = self._cognitive_handler.enabled
        if RECALL_TOOL_NAME in enabled:
            specs.append(RECALL_SPEC)
        if PIN_TOOL_NAME in enabled:
            specs.append(PIN_SPEC)
            specs.append(UNPIN_SPEC)  # unpin rides with pin

        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": format_tool_for_llm(s),
                    "parameters": s.input_schema,
                },
            }
            for s in specs
        ]


    @staticmethod
    def _estimate_tool_tokens(tools: list[dict[str, object]] | None) -> int:
        """Estimate tokens consumed by tool schemas in the request."""
        if not tools:
            return 0
        import json

        from arcana.context.builder import estimate_tokens
        # Tool schemas are sent as JSON; estimate their token cost
        return sum(estimate_tokens(json.dumps(t)) for t in tools)

    # ------------------------------------------------------------------
    # Model config resolution
    # ------------------------------------------------------------------

    def _resolve_model_config(self) -> ModelConfig:
        """Return the model config to use for LLM calls.

        Resolution order:
        1. Explicitly provided model_config (set at init)
        2. Provider's default_model attribute
        3. Raise ValueError -- never guess a hardcoded model name
        """
        if self.model_config:
            return self.model_config
        # Derive from gateway's default provider
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
    # Error diagnosis
    # ------------------------------------------------------------------

    def _diagnose_tool_failure(self, result: ToolResult) -> str:
        """Generate a diagnostic message for a failed tool call.

        Uses the structured diagnoser when a ToolError is available,
        falling back to a simple text message otherwise.
        """
        if result.error:
            from arcana.runtime.diagnosis.diagnoser import diagnose_tool_error

            diagnosis = diagnose_tool_error(
                tool_name=result.name,
                tool_error=result.error,
                available_tools=self._get_tool_names(),
            )
            return diagnosis.to_recovery_prompt()

        return (
            f"[System] Tool '{result.name}' failed with an unknown error. "
            f"Please adjust your approach or try an alternative."
        )

    def _get_tool_names(self) -> list[str]:
        """Return the names of all currently registered tools."""
        if not self.tool_gateway:
            return []
        return self.tool_gateway.registry.list_tools()

    # ------------------------------------------------------------------
    # Checkpoint / Resume
    # ------------------------------------------------------------------

    def _should_checkpoint(self, state: AgentState, assessment: TurnAssessment) -> str | None:
        """Decide whether to checkpoint. Returns reason string or None."""
        # On error
        if self._checkpoint_on_error and assessment.failed:
            return "error"

        # Interval — every N turns
        if (
            self._checkpoint_interval > 0
            and state.current_step > 0
            and state.current_step % self._checkpoint_interval == 0
        ):
            return "interval"

        # Budget threshold crossing
        if self.budget_tracker:
            total = self.budget_tracker.max_cost_usd or 0
            if total > 0:
                ratio = self.budget_tracker.cost_usd / total
                for threshold in self._checkpoint_budget_thresholds:
                    if self._last_checkpoint_budget_ratio < threshold <= ratio:
                        self._last_checkpoint_budget_ratio = ratio
                        return "budget"

        return None

    async def resume(
        self,
        run_id: str,
        *,
        max_turns: int | None = None,
    ) -> AgentState:
        """Resume execution from the latest checkpoint for a run.

        Loads the most recent StateSnapshot, restores conversation messages,
        and continues the conversation loop from where it left off.

        Args:
            run_id: Run ID to resume.
            max_turns: Override max turns (defaults to self.max_turns).

        Returns:
            Final AgentState after resumed execution.
        """
        if not self._state_manager:
            raise RuntimeError("Cannot resume without a state_manager.")

        snapshot = await self._state_manager.load_snapshot(run_id)
        if snapshot is None:
            raise ValueError(f"No checkpoint found for run '{run_id}'.")

        self._state_manager.verify_snapshot(snapshot)

        saved_state = snapshot.state
        goal = saved_state.goal or ""

        logger.info(
            "Resuming run '%s' from step %d (reason: %s)",
            run_id, saved_state.current_step, snapshot.checkpoint_reason,
        )

        # Restore messages from saved state
        messages: list[Message] = []
        for m in saved_state.messages:
            role_str = m.get("role", "user") if isinstance(m, dict) else "user"
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            role = MessageRole(role_str) if role_str in MessageRole.__members__.values() else MessageRole.USER
            messages.append(Message(role=role, content=content))

        if not messages:
            messages = self._build_initial_messages(goal)

        self._context_builder.set_goal(goal)
        self._last_checkpoint_budget_ratio = 0.0

        # Continue the conversation loop with remaining turns
        remaining = (max_turns or self.max_turns) - saved_state.current_step
        if remaining <= 0:
            return saved_state

        state = saved_state.model_copy(update={"status": ExecutionStatus.RUNNING})

        # Re-setup tools
        active_tools: list[dict[str, Any]] | None
        if self._lazy_registry:
            self._lazy_registry.reset()
            self._lazy_registry.select_initial_tools(goal)
            active_tools = self._lazy_registry.to_openai_tools() or []
            active_tools.append(self._ask_user_tool_schema())
            active_tools.extend(self._cognitive_tool_schemas())
        else:
            active_tools = self._get_current_tools()

        # Simplified resume loop — reuse the core turn logic
        for _turn in range(remaining):
            if self.budget_tracker:
                self.budget_tracker.check_budget()
                self.budget_tracker.consume_iteration()

            if self._lazy_registry:
                tool_token_cost = (
                    self._lazy_registry.tool_token_estimate
                    + self._ask_user_token_cost
                    + self._cognitive_token_cost
                )
            else:
                tool_token_cost = self._estimate_tool_tokens(active_tools)
            curated = self._context_builder.build_conversation_context(
                messages,
                tool_token_estimate=tool_token_cost,
                turn=saved_state.current_step + _turn,
            )

            request = LLMRequest(
                messages=curated,
                tools=active_tools,
                response_format=self._response_format_schema,
            )
            config = self._resolve_model_config()

            # Resume loop: emit prompt snapshot (context decision already traced
            # by the main loop prior to suspension, if applicable)
            self._emit_prompt_snapshot_event(state, request, config)

            response = await self.gateway.generate(request, config)
            facts = self._parse_turn(response)
            assessment = self._assess_turn(facts, state)

            # Append assistant message
            messages.append(Message(role=MessageRole.ASSISTANT, content=facts.assistant_text or ""))

            # Handle tool calls
            if facts.tool_calls:
                results, _resume_events = await self._execute_tools(facts.tool_calls)
                for tc_req in facts.tool_calls:
                    messages.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=f"[tool_call: {tc_req.name}]",
                    ))
                for result in results:
                    messages.append(Message(
                        role=MessageRole.TOOL,
                        content=result.output_str if hasattr(result, "output_str") else str(result.output),
                        tool_call_id=result.tool_call_id,
                    ))

            # Update state
            state = state.increment_step()
            if facts.usage:
                state = state.model_copy(update={
                    "tokens_used": state.tokens_used + facts.usage.total_tokens,
                    "cost_usd": state.cost_usd + facts.usage.cost_estimate,
                })

            if assessment.completed:
                answer = (assessment.answer or facts.assistant_text or "").strip()
                state = state.model_copy(update={
                    "status": ExecutionStatus.COMPLETED,
                    "working_memory": {**state.working_memory, "answer": answer},
                })
                break
            elif assessment.failed:
                state = state.model_copy(update={"status": ExecutionStatus.FAILED})
                break

        self._state = state
        return state

    # ------------------------------------------------------------------
    # Trace recording
    # ------------------------------------------------------------------

    def _trace_turn(
        self,
        facts: TurnFacts,
        assessment: TurnAssessment,
        state: AgentState,
    ) -> None:
        """Record turn facts and assessment to the trace writer."""
        if not self.trace_writer:
            return

        from arcana.contracts.trace import EventType, TraceEvent

        turn_step_id = getattr(self, "_current_turn_step_id", None)
        prev_turn_step_id = getattr(self, "_prev_turn_step_id", None)
        self.trace_writer.write(TraceEvent(
            run_id=state.run_id,
            step_id=turn_step_id or str(uuid4()),
            parent_step_id=prev_turn_step_id,
            event_type=EventType.TURN,
            metadata={
                "step": state.current_step,
                "facts": facts.model_dump(),
                "assessment": assessment.model_dump(),
            },
        ))

    def _emit_context_decision_event(self, state: AgentState) -> None:
        """Emit the CONTEXT_DECISION trace event with full structured evidence.

        Carries both the ContextDecision (per-message decisions + strategy)
        and the ContextReport (token accounting). This is what powers
        offline replay / explainability.
        """
        if not self.trace_writer:
            return
        decision = self._context_builder.last_decision
        if decision is None:
            return

        from arcana.contracts.trace import EventType, TraceEvent

        report = self._context_builder.last_report
        metadata: dict[str, Any] = {
            "turn": decision.turn,
            # New structure (v0.6.0)
            "context_decision": decision.model_dump(),
            # Old fields (backward compat for trace show --context)
            "explanation": decision.explanation,
        }
        if report is not None:
            metadata["context_report"] = report.model_dump()
            # Old fields (backward compat for trace show --context)
            metadata["compressed_count"] = report.compressed_count
            metadata["messages_in"] = report.messages_in
            metadata["messages_out"] = report.messages_out

        self.trace_writer.write(TraceEvent(
            run_id=state.run_id,
            parent_step_id=getattr(self, "_current_turn_step_id", None),
            event_type=EventType.CONTEXT_DECISION,
            metadata=metadata,
        ))

    def _emit_prompt_snapshot_event(
        self,
        state: AgentState,
        request: LLMRequest,
        config: ModelConfig,
    ) -> None:
        """Emit the PROMPT_SNAPSHOT trace event when opted in.

        Captures the exact messages/tools/model sent to the provider so
        a run can be offline-replayed via ``TraceReader.replay_prompt``.
        Gated by ``RuntimeConfig.trace_include_prompt_snapshots`` because
        prompt content may include PII / secrets and inflates trace size.
        """
        if not self._trace_include_prompt_snapshots:
            return
        if not self.trace_writer:
            return

        from arcana.contracts.trace import (
            BudgetSnapshot,
            EventType,
            PromptSnapshot,
            TraceEvent,
        )

        budget_snapshot: BudgetSnapshot | None = None
        if self.budget_tracker is not None:
            bt = self.budget_tracker
            budget_snapshot = BudgetSnapshot(
                max_tokens=bt.max_tokens,
                max_cost_usd=bt.max_cost_usd,
                max_time_ms=bt.max_time_ms,
                max_iterations=bt.max_iterations,
                tokens_used=bt.tokens_used,
                cost_usd=bt.cost_usd,
                time_ms=bt.elapsed_ms,
                iterations_used=bt.iterations_used,
            )

        snapshot = PromptSnapshot(
            turn=state.current_step,
            model=config.model_id,
            messages=[m.model_dump() for m in request.messages],
            tools=list(request.tools or []),
            response_format=request.response_format,
            budget_snapshot=budget_snapshot,
        )

        self.trace_writer.write(TraceEvent(
            run_id=state.run_id,
            parent_step_id=getattr(self, "_current_turn_step_id", None),
            event_type=EventType.PROMPT_SNAPSHOT,
            model=config.model_id,
            metadata={
                "turn": state.current_step,
                "prompt_snapshot": snapshot.model_dump(),
            },
        ))

    # ------------------------------------------------------------------
    # Intent routing fast path
    # ------------------------------------------------------------------

    async def _direct_answer(self, goal: str) -> AsyncGenerator[StreamEvent, None]:
        """Fast path: single LLM call, no tools, no loop."""
        from arcana.routing.executor import DirectExecutor

        run_id = str(uuid4())
        config = self._resolve_model_config()
        executor = DirectExecutor()
        response = await executor.direct_answer(
            goal, self.gateway, config, system_prompt=self.system_prompt,
        )

        # Warn and estimate tokens if provider reported 0 usage
        if response.usage and response.usage.total_tokens == 0 and response.content:
            estimated_completion = len(response.content) // 4
            response = LLMResponse(
                content=response.content,
                tool_calls=response.tool_calls,
                usage=TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=estimated_completion,
                    total_tokens=estimated_completion,
                ),
                model=response.model,
                finish_reason=response.finish_reason,
                raw_response=response.raw_response,
                anthropic=response.anthropic,
                gemini=response.gemini,
                openai=response.openai,
                ollama=response.ollama,
            )
            logger.warning(
                "Provider reported 0 tokens; estimated %d from response length. "
                "Cost tracking is approximate.",
                estimated_completion,
            )

        answer = (response.content or "").strip()

        # Track token usage
        tokens_used = 0
        cost_usd = 0.0
        if response.usage:
            tokens_used = response.usage.total_tokens
            if self.budget_tracker:
                self.budget_tracker.add_usage(response.usage)
                cost_usd = self.budget_tracker.cost_usd

        state = AgentState(
            run_id=run_id,
            goal=goal,
            status=ExecutionStatus.COMPLETED,
            current_step=1,
            working_memory={"answer": answer},
            tokens_used=tokens_used,
            cost_usd=cost_usd,
        )
        self._state = state

        yield StreamEvent(
            event_type=StreamEventType.STEP_COMPLETE,
            run_id=run_id,
            content=answer,
        )
        yield StreamEvent(
            event_type=StreamEventType.RUN_COMPLETE,
            run_id=run_id,
            content=answer,
        )


