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
    Message,
    MessageRole,
    ModelConfig,
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
    from arcana.tool_gateway.gateway import ToolGateway
    from arcana.trace.writer import TraceWriter

logger = logging.getLogger(__name__)

# Sentinel markers the system prompt asks the LLM to use.
_COMPLETION_MARKERS = ("[done]", "[complete]", "[finished]")

# Default system prompt includes the [DONE] convention.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "When you have fully answered the user's request, end your response with [DONE]."
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
    ) -> None:
        self.gateway = gateway
        self.model_config = model_config
        self.tool_gateway = tool_gateway
        self.budget_tracker = budget_tracker
        self.trace_writer = trace_writer
        self.intent_classifier = intent_classifier
        self.max_turns = max_turns
        self.system_prompt = system_prompt

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
        tools = self._get_current_tools()

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

            # 2. Context rebuild (future: working_set integration)
            # Currently a no-op; messages list is the full context.

            # 3. LLM call
            request = LLMRequest(
                messages=messages,
                tools=tools if tools else None,
            )
            config = self._resolve_model_config()
            response = await self.gateway.generate(
                request=request,
                config=config,
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
                answer = assessment.answer or facts.assistant_text or ""
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

            # 13. Checkpoint (placeholder for future persistence)
            # if self._should_checkpoint(state):
            #     await self._checkpoint(state)

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

        Five rules, priority ordered:
        1. Tool calls present -> not done yet
        2. No text and no tools -> failed
        3. Explicit [DONE] marker -> completed
        4. Verifier (future)
        5. Heuristic: finish_reason=stop + non-first turn + text>100 chars
        """
        assessment = TurnAssessment()

        # Rule 1: tool calls present -> not done yet
        if facts.tool_calls:
            return assessment

        # Rule 2: no text and no tools -> something went wrong
        if not facts.assistant_text:
            assessment.failed = True
            assessment.completion_reason = "empty_response"
            return assessment

        text = facts.assistant_text.strip()

        # Rule 3: explicit finish markers in text
        text_lower = text.lower()
        if any(marker in text_lower for marker in _COMPLETION_MARKERS):
            assessment.completed = True
            assessment.answer = text
            assessment.completion_reason = "explicit_marker"
            assessment.confidence = 0.9
            return assessment

        # Rule 4: verifier (future)
        # if self.verifier: ...

        # Rule 5: heuristic -- substantial text + natural stop + not first turn
        if (
            facts.finish_reason == "stop"
            and state.current_step > 0
            and len(text) > 100
        ):
            assessment.completed = True
            assessment.answer = text
            assessment.completion_reason = "heuristic_natural_stop"
            assessment.confidence = 0.6
            return assessment

        # Default: not done, LLM is still thinking
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
                    output="Tool gateway not configured",
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

    @staticmethod
    def _diagnose_tool_failure(result: ToolResult) -> str:
        """Generate a diagnostic message for a failed tool call."""
        error_msg = result.error.message if result.error else "Unknown error"
        return (
            f"[System] Tool '{result.name}' failed: {error_msg}. "
            f"Please adjust your approach or try an alternative."
        )

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

        self.trace_writer.write_raw(
            state.run_id,
            {
                "event": "turn",
                "step": state.current_step,
                "facts": facts.model_dump(),
                "assessment": assessment.model_dump(),
            },
        )

    # ------------------------------------------------------------------
    # Intent routing fast path
    # ------------------------------------------------------------------

    async def _direct_answer(self, goal: str) -> AsyncGenerator[StreamEvent, None]:
        """Fast path: single LLM call, no tools, no loop."""
        from arcana.routing.executor import DirectExecutor

        run_id = str(uuid4())
        config = self._resolve_model_config()
        executor = DirectExecutor()
        answer = await executor.direct_answer(goal, self.gateway, config)

        state = AgentState(
            run_id=run_id,
            goal=goal,
            status=ExecutionStatus.COMPLETED,
            current_step=1,
            working_memory={"answer": answer},
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
