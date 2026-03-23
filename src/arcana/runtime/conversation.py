"""ConversationAgent -- V2 execution model.

LLM-native conversation loop that replaces the Policy -> Step -> Reducer chain
with a single thin abstraction: LLM Turn -> Runtime Events -> State.

See specs/v2-architecture/conversation-loop.md for the full design.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import uuid4

from arcana.contracts.intent import IntentType
from arcana.contracts.llm import (
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
from arcana.contracts.tool import ToolCall, ToolResult
from arcana.contracts.turn import TurnAssessment, TurnFacts

if TYPE_CHECKING:
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
    ) -> None:
        self.gateway = gateway
        self.model_config = model_config
        self.tool_gateway = tool_gateway
        self.budget_tracker = budget_tracker
        self.trace_writer = trace_writer
        self.intent_classifier = intent_classifier
        self.max_turns = max_turns
        self.system_prompt = system_prompt

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

        # Context management — unified through WorkingSetBuilder
        from arcana.context.builder import WorkingSetBuilder
        from arcana.contracts.context import TokenBudget

        self._context_builder = WorkingSetBuilder(
            identity=system_prompt,
            token_budget=TokenBudget(total_window=context_window),
            goal=None,  # Set when run starts
        )
        self._memory_context = memory_context

        # Populated during a run; accessed by run() after astream() finishes.
        self._state: AgentState | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, goal: str) -> AgentState:
        """Run to completion by consuming astream()."""
        async for _event in self.astream(goal):
            pass  # All side-effects happen inside astream
        assert self._state is not None, "astream must set _state before returning"
        return self._state

    async def astream(self, goal: str) -> AsyncGenerator[StreamEvent, None]:
        """Stream execution events. This is the primary interface.

        Implements the 13-step turn contract from conversation-loop.md.
        """
        # ------------------------------------------------------------------
        # Phase 1: Optional intent routing
        # ------------------------------------------------------------------
        if self.intent_classifier:
            classification = await self.intent_classifier.classify(goal)
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
        messages = self._build_initial_messages(goal)
        self._context_builder.set_goal(goal)

        # Tool selection: lazy (subset) or eager (all)
        if self._lazy_registry:
            self._lazy_registry.reset()
            self._lazy_registry.select_initial_tools(goal)
            active_tools = self._lazy_registry.to_openai_tools() or None
        else:
            active_tools = self._get_current_tools()
        tool_token_cost = self._estimate_tool_tokens(active_tools)

        yield StreamEvent(
            event_type=StreamEventType.RUN_START,
            run_id=run_id,
            content=goal,
        )

        # ------------------------------------------------------------------
        # Phase 3: Conversation loop
        # ------------------------------------------------------------------
        for _turn in range(self.max_turns):
            # ── Steps 1-6: happen EVERY turn ──

            # 1. Budget check
            if self.budget_tracker:
                self.budget_tracker.check_budget()

            # 2. Context rebuild — delegate to WorkingSetBuilder
            #    Always include tools. Token optimization for tools belongs in
            #    LazyToolRegistry (dynamic tool selection), not here.
            curated = self._context_builder.build_conversation_context(
                messages,
                memory_context=self._memory_context if _turn == 0 else None,
                tool_token_estimate=tool_token_cost,
                turn=_turn,
            )
            # Write back so memory injection persists in messages for future turns
            if _turn == 0 and self._memory_context:
                messages = curated[:]

            # Trace the context decision
            if self.trace_writer and self._context_builder.last_decision:
                from arcana.contracts.trace import EventType, TraceEvent

                decision = self._context_builder.last_decision
                self.trace_writer.write(TraceEvent(
                    run_id=state.run_id,
                    event_type=EventType.CONTEXT_DECISION,
                    metadata={
                        "turn": decision.turn,
                        "budget_total": decision.budget_total,
                        "budget_used": decision.budget_used,
                        "budget_tools": decision.budget_tools,
                        "messages_in": decision.messages_in,
                        "messages_out": decision.messages_out,
                        "compressed_count": decision.compressed_count,
                        "memory_injected": decision.memory_injected,
                        "history_compressed": decision.history_compressed,
                        "explanation": decision.explanation,
                    },
                ))

            # 3. LLM call (streaming when gateway supports it)
            request = LLMRequest(
                messages=curated,
                tools=active_tools,
            )
            config = self._resolve_model_config()

            response: LLMResponse
            try:
                text_parts: list[str] = []
                tc_names: dict[str, str] = {}
                tc_args: dict[str, list[str]] = {}
                stream_usage: TokenUsage | None = None
                stream_finish = "stop"
                stream_model = config.model_id

                async for chunk in self.gateway.stream(
                    request=request, config=config,
                ):
                    if chunk.type == "text_delta" and chunk.text:
                        text_parts.append(chunk.text)
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
                    elif chunk.type == "tool_call_delta" and chunk.tool_call_id:
                        if chunk.tool_call_id not in tc_names:
                            tc_names[chunk.tool_call_id] = chunk.tool_name or ""
                        if chunk.tool_name:
                            tc_names[chunk.tool_call_id] = chunk.tool_name
                        if chunk.arguments_delta:
                            tc_args.setdefault(
                                chunk.tool_call_id, [],
                            ).append(chunk.arguments_delta)
                    elif chunk.type == "usage" and chunk.usage:
                        stream_usage = chunk.usage
                    elif chunk.type == "done":
                        if chunk.metadata:
                            stream_finish = chunk.metadata.get(
                                "finish_reason", stream_finish,
                            )
                            stream_model = chunk.metadata.get(
                                "model", stream_model,
                            )
                        if chunk.usage:
                            stream_usage = chunk.usage

                full_text = "".join(text_parts) if text_parts else None
                tool_calls_list = [
                    ToolCallRequest(
                        id=tc_id,
                        name=tc_names.get(tc_id, ""),
                        arguments="".join(tc_args.get(tc_id, [])),
                    )
                    for tc_id in tc_names
                ] or None
                response = LLMResponse(
                    content=full_text,
                    tool_calls=tool_calls_list,
                    usage=stream_usage or TokenUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    ),
                    model=stream_model,
                    finish_reason=stream_finish,
                )
            except (AttributeError, TypeError, NotImplementedError):
                # Gateway doesn't support streaming — fall back to generate
                response = await self.gateway.generate(
                    request=request, config=config,
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
                # Expand lazy registry for any tools the LLM requested
                if self._lazy_registry:
                    for tc in facts.tool_calls:
                        self._lazy_registry.get_tool_on_demand(tc.name)
                    # Refresh active tools and token cost for next turn
                    active_tools = self._lazy_registry.to_openai_tools() or None
                    tool_token_cost = self._estimate_tool_tokens(active_tools)

                results = await self._execute_tools(facts.tool_calls)
                messages = self._add_tool_messages(messages, facts, results)

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

        self._state = state

        yield StreamEvent(
            event_type=StreamEventType.RUN_COMPLETE,
            run_id=run_id,
            content=state.working_memory.get("answer", ""),
            tokens_used=state.tokens_used,
            cost_usd=state.cost_usd,
        )

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
    ) -> list[ToolResult]:
        """Execute tool calls via ToolGateway."""
        if not self.tool_gateway:
            # No tool gateway -- return synthetic errors
            return [
                ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    success=False,
                    output=(
                        f"Tool '{tc.name}' cannot be executed: no tools are registered. "
                        f"Register tools with: Runtime(tools=[your_tool_function]) "
                        f"or @arcana.tool decorator."
                    ),
                )
                for tc in tool_calls
            ]

        gateway_calls: list[ToolCall] = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.arguments) if tc.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": tc.arguments}
            gateway_calls.append(
                ToolCall(id=tc.id, name=tc.name, arguments=args)
            )

        return await self.tool_gateway.call_many(gateway_calls)

    @staticmethod
    def _add_tool_messages(
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

        # Tool result messages (one per result)
        for result in results:
            messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=result.output_str,
                    tool_call_id=result.tool_call_id,
                )
            )

        return messages

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_initial_messages(self, goal: str) -> list[Message]:
        """Build the initial message list: system + user goal."""
        msgs: list[Message] = []
        if self.system_prompt:
            msgs.append(Message(role=MessageRole.SYSTEM, content=self.system_prompt))
        msgs.append(Message(role=MessageRole.USER, content=goal))
        return msgs

    def _get_current_tools(self) -> list[dict[str, object]] | None:
        """Get tool schemas in OpenAI format from ToolGateway."""
        if not self.tool_gateway:
            return None
        tool_defs = self.tool_gateway.registry.to_openai_tools()
        return tool_defs if tool_defs else None

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
        if self._lazy_registry:
            self._lazy_registry.reset()
            self._lazy_registry.select_initial_tools(goal)
            active_tools = self._lazy_registry.to_openai_tools() or None
        else:
            active_tools = self._get_current_tools()

        # Simplified resume loop — reuse the core turn logic
        for _turn in range(remaining):
            if self.budget_tracker:
                self.budget_tracker.check_budget()

            tool_token_cost = self._estimate_tool_tokens(active_tools)
            curated = self._context_builder.build_conversation_context(
                messages,
                tool_token_estimate=tool_token_cost,
                turn=saved_state.current_step + _turn,
            )

            request = LLMRequest(messages=curated, tools=active_tools)
            config = self._resolve_model_config()

            response = await self.gateway.generate(request, config)
            facts = self._parse_turn(response)
            assessment = self._assess_turn(facts, state)

            # Append assistant message
            messages.append(Message(role=MessageRole.ASSISTANT, content=facts.assistant_text or ""))

            # Handle tool calls
            if facts.tool_calls:
                results = await self._execute_tools(facts.tool_calls)
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

        self.trace_writer.write(TraceEvent(
            run_id=state.run_id,
            event_type=EventType.TURN,
            metadata={
                "step": state.current_step,
                "facts": facts.model_dump(),
                "assessment": assessment.model_dump(),
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


