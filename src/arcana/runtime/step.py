"""StepExecutor - executes individual steps."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from arcana.contracts.llm import LLMRequest, LLMResponse, Message, MessageRole, ModelConfig
from arcana.contracts.runtime import PolicyDecision, StepResult, StepType
from arcana.contracts.tool import ToolCall
from arcana.runtime.exceptions import StepExecutionError
from arcana.runtime.validator import OutputValidator

if TYPE_CHECKING:
    from arcana.contracts.state import AgentState
    from arcana.contracts.trace import TraceContext
    from arcana.gateway.budget import BudgetTracker
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.runtime.verifiers.base import BaseVerifier
    from arcana.tool_gateway.gateway import ToolGateway
    from arcana.trace.writer import TraceWriter

logger = logging.getLogger(__name__)


class StepExecutor:
    """
    Executes individual agent steps.

    Handles:
    - LLM calls for reasoning
    - Tool execution (via future ToolGateway)
    - Result processing and observation
    """

    def __init__(
        self,
        *,
        gateway: ModelGatewayRegistry,
        tool_gateway: ToolGateway | None = None,
        verifier: BaseVerifier | None = None,
        trace_writer: TraceWriter | None = None,
        budget_tracker: BudgetTracker | None = None,
        model_config: ModelConfig | None = None,
        max_validation_retries: int = 2,
    ) -> None:
        """
        Initialize the step executor.

        Args:
            gateway: Model gateway for LLM calls
            tool_gateway: Optional tool gateway for tool execution
            verifier: Optional verifier for goal/plan verification
            trace_writer: Optional trace writer
            budget_tracker: Optional budget tracker
            model_config: Default model configuration
            max_validation_retries: Max retries for validation failures
        """
        self.gateway = gateway
        self.tool_gateway = tool_gateway
        self.verifier = verifier
        self.trace_writer = trace_writer
        self.budget_tracker = budget_tracker
        self.model_config = model_config or ModelConfig(
            provider="deepseek",
            model_id="deepseek-chat",
            temperature=0.0,
        )
        self.validator = OutputValidator(max_retry_attempts=max_validation_retries)

    async def execute(
        self,
        *,
        state: AgentState,
        decision: PolicyDecision,
        trace_ctx: TraceContext,
    ) -> StepResult:
        """
        Execute a step based on policy decision.

        Args:
            state: Current agent state
            decision: Policy decision about what to do
            trace_ctx: Trace context for logging

        Returns:
            StepResult with execution outcome
        """
        step_id = trace_ctx.new_step_id()

        try:
            if decision.action_type == "llm_call":
                return await self._execute_llm_call(
                    state, decision, step_id, trace_ctx
                )
            elif decision.action_type == "tool_call":
                return await self._execute_tool_calls(
                    state, decision, step_id, trace_ctx
                )
            elif decision.action_type == "complete":
                return StepResult(
                    step_type=StepType.VERIFY,
                    step_id=step_id,
                    success=True,
                    thought="Goal achieved",
                    state_updates={"goal_reached": True},
                )
            elif decision.action_type == "verify":
                return await self._execute_verify(
                    state, decision, step_id
                )
            elif decision.action_type == "fail":
                return StepResult(
                    step_type=StepType.VERIFY,
                    step_id=step_id,
                    success=False,
                    error=decision.stop_reason or "Policy decided to fail",
                    is_recoverable=False,
                )
            else:
                return StepResult(
                    step_type=StepType.THINK,
                    step_id=step_id,
                    success=False,
                    error=f"Unknown action type: {decision.action_type}",
                )

        except Exception as e:
            return StepResult(
                step_type=StepType.THINK,
                step_id=step_id,
                success=False,
                error=str(e),
                is_recoverable=self._is_recoverable_error(e),
            )

    async def _execute_llm_call(
        self,
        state: AgentState,
        decision: PolicyDecision,
        step_id: str,
        trace_ctx: TraceContext,
    ) -> StepResult:
        """Execute an LLM call step with optional validation."""
        # Build request
        messages = []
        for m in decision.messages:
            role = m.get("role", "user")
            if isinstance(role, str):
                role = MessageRole(role)
            messages.append(
                Message(
                    role=role,
                    content=m.get("content"),
                    name=m.get("name"),
                    tool_call_id=m.get("tool_call_id"),
                )
            )

        # Extract schema if provided in decision
        expected_schema = decision.metadata.get("expected_schema") if hasattr(decision, "metadata") else None
        validate_structured = decision.metadata.get("validate_structured", False) if hasattr(decision, "metadata") else False
        required_fields = decision.metadata.get("required_fields", ["thought", "action"]) if hasattr(decision, "metadata") else ["thought", "action"]

        # Try LLM call with retries for validation failures
        max_attempts = 1 + self.validator.max_retry_attempts
        last_error = None

        for attempt in range(max_attempts):
            request = LLMRequest(messages=messages)

            # Check budget before call
            if self.budget_tracker:
                self.budget_tracker.check_budget()

            # Make LLM call
            try:
                response = await self.gateway.generate(
                    request=request,
                    config=self.model_config,
                    trace_ctx=trace_ctx,
                )
            except Exception as e:
                raise StepExecutionError(
                    f"LLM call failed: {e}",
                    step_id=step_id,
                    recoverable=self._is_recoverable_error(e),
                ) from e

            # Update budget tracker
            if self.budget_tracker and response.usage:
                self.budget_tracker.add_usage(response.usage)

            # Handle AdaptivePolicy strategy responses before validation.
            # The strategy phase returns JSON with fields like "strategy",
            # "reasoning", "action" -- not the "Thought:/Action:" text format
            # that validate_structured expects. Intercept and handle early.
            is_strategy_phase = (
                decision.metadata.get("phase") == "strategy"
                if hasattr(decision, "metadata")
                else False
            )
            if is_strategy_phase:
                return self._handle_strategy_response(response, state, step_id)

            # Validate if schema expected
            if expected_schema:
                validation = self.validator.validate_json(response, expected_schema)
                if not validation.valid:
                    last_error = f"Validation failed: {', '.join(validation.errors)}"
                    # Retry with error feedback
                    if attempt < max_attempts - 1:
                        retry_prompt = self.validator.create_retry_prompt(
                            validation,
                            schema_description=f"Expected schema: {expected_schema.__name__}" if expected_schema else None
                        )
                        messages.append({"role": "assistant", "content": response.content or ""})
                        messages.append({"role": "user", "content": retry_prompt})
                        continue
                    # Last attempt failed
                    return StepResult(
                        step_type=StepType.THINK,
                        step_id=step_id,
                        success=False,
                        error=last_error,
                        is_recoverable=True,
                        llm_response=response,
                    )

            # Validate structured format if requested
            if validate_structured:
                validation = self.validator.validate_structured_format(response, required_fields)
                if not validation.valid:
                    last_error = f"Format validation failed: {', '.join(validation.errors)}"
                    if attempt < max_attempts - 1:
                        retry_prompt = self.validator.create_retry_prompt(
                            validation,
                            schema_description=f"Required fields: {', '.join(required_fields)}"
                        )
                        messages.append({"role": "assistant", "content": response.content or ""})
                        messages.append({"role": "user", "content": retry_prompt})
                        continue
                    # Last attempt failed
                    return StepResult(
                        step_type=StepType.THINK,
                        step_id=step_id,
                        success=False,
                        error=last_error,
                        is_recoverable=True,
                        llm_response=response,
                    )

            # Validation passed or no validation required
            break

        # Parse response for thought/action
        thought, action = self._parse_llm_response(response.content or "")

        # Check if action indicates completion
        if action and action.upper() == "FINISH":
            return StepResult(
                step_type=StepType.VERIFY,
                step_id=step_id,
                success=True,
                thought=thought,
                action=action,
                llm_response=response,
                state_updates={
                    "goal_reached": True,
                    "tokens_used": state.tokens_used + (response.usage.total_tokens if response.usage else 0),
                },
            )

        return StepResult(
            step_type=StepType.THINK,
            step_id=step_id,
            success=True,
            thought=thought,
            action=action,
            llm_response=response,
            state_updates={
                "tokens_used": state.tokens_used + (response.usage.total_tokens if response.usage else 0),
            },
        )

    async def _execute_tool_calls(
        self,
        state: AgentState,
        decision: PolicyDecision,
        step_id: str,
        trace_ctx: TraceContext,
    ) -> StepResult:
        """Execute tool calls via the ToolGateway."""
        if self.tool_gateway is None:
            return StepResult(
                step_type=StepType.ACT,
                step_id=step_id,
                success=False,
                error="ToolGateway not configured",
                is_recoverable=False,
            )

        tool_calls = []
        for tc_dict in decision.tool_calls:
            tool_calls.append(
                ToolCall(
                    id=tc_dict.get("id", str(uuid4())),
                    name=tc_dict["name"],
                    arguments=tc_dict.get("arguments", {}),
                    idempotency_key=tc_dict.get("idempotency_key"),
                    run_id=state.run_id,
                    step_id=step_id,
                )
            )

        results = await self.tool_gateway.call_many(tool_calls, trace_ctx=trace_ctx)

        all_success = all(r.success for r in results)
        observation = "\n".join(r.output_str for r in results)

        step_result = StepResult(
            step_type=StepType.ACT,
            step_id=step_id,
            success=all_success,
            tool_results=results,
            observation=observation,
            error=None if all_success else "One or more tool calls failed",
            is_recoverable=any(
                r.error and r.error.is_retryable
                for r in results
                if not r.success
            ),
        )

        # Diagnostic recovery: diagnose the first failed tool result so that
        # AdaptivePolicy can see the diagnosis on the next decide() call.
        for result in results:
            if not result.success and result.error:
                from arcana.runtime.diagnosis.diagnoser import diagnose_tool_error

                diagnosis = diagnose_tool_error(
                    tool_name=result.name,
                    tool_error=result.error,
                    available_tools=self._get_available_tool_names(),
                )
                step_result.state_updates["last_diagnosis"] = diagnosis.model_dump()
                step_result.state_updates["recovery_prompt"] = (
                    diagnosis.to_recovery_prompt()
                )
                break  # Diagnose only the first failure

        return step_result

    async def _execute_verify(
        self,
        state: AgentState,
        decision: PolicyDecision,
        step_id: str,
    ) -> StepResult:
        """Execute a verification step."""
        state_updates: dict[str, object] = {"goal_reached": True}

        if self.verifier is not None:
            # Load plan from decision metadata if available
            from arcana.contracts.plan import Plan

            plan: Plan | None = None
            plan_data = decision.metadata.get("plan")
            if plan_data is not None:
                try:
                    plan = Plan.model_validate(plan_data)
                except Exception:
                    plan = None

            verification = await self.verifier.verify(state, plan)
            state_updates["verification_result"] = verification.model_dump(mode="json")

            from arcana.contracts.plan import VerificationOutcome

            if verification.outcome == VerificationOutcome.FAILED:
                return StepResult(
                    step_type=StepType.VERIFY,
                    step_id=step_id,
                    success=False,
                    thought="Verification failed",
                    observation=f"Coverage: {verification.coverage:.0%}, Failed: {verification.failed_criteria}",
                    state_updates=state_updates,
                    is_recoverable=True,
                )

        return StepResult(
            step_type=StepType.VERIFY,
            step_id=step_id,
            success=True,
            thought="Verification complete",
            state_updates=state_updates,
        )

    def _handle_strategy_response(
        self,
        response: LLMResponse,
        state: AgentState,
        step_id: str,
    ) -> StepResult:
        """Handle AdaptivePolicy strategy JSON responses.

        Instead of validating for Thought:/Action: format, parse the
        strategy JSON and convert it into the appropriate StepResult.
        """
        from arcana.contracts.strategy import StrategyType
        from arcana.runtime.policies.adaptive import AdaptivePolicy

        content = response.content or ""
        token_delta = response.usage.total_tokens if response.usage else 0

        # Parse strategy decision from LLM output
        try:
            strategy_decision = AdaptivePolicy.parse_strategy_response(content)
        except ValueError:
            # LLM did not return valid strategy JSON -- treat as plain thought
            logger.warning("Could not parse strategy JSON; falling back to plain thought")
            return StepResult(
                step_type=StepType.THINK,
                step_id=step_id,
                success=True,
                thought=content,
                llm_response=response,
                state_updates={
                    "tokens_used": state.tokens_used + token_delta,
                },
            )

        # Load current adaptive state from working_memory
        adaptive_state = state.working_memory.get("adaptive_state", {})
        if isinstance(adaptive_state, dict):
            adaptive_state = dict(adaptive_state)  # shallow copy
            adaptive_state["current_strategy"] = strategy_decision.strategy.value
        else:
            # It might be an AdaptiveState pydantic model; convert to dict
            adaptive_state = {"current_strategy": strategy_decision.strategy.value}

        match strategy_decision.strategy:
            case StrategyType.DIRECT_ANSWER:
                return StepResult(
                    step_type=StepType.VERIFY,
                    step_id=step_id,
                    success=True,
                    thought=strategy_decision.reasoning,
                    action=f"direct_answer: {strategy_decision.action}",
                    llm_response=response,
                    state_updates={
                        "goal_reached": True,
                        "answer": strategy_decision.action,
                        "result": strategy_decision.action,
                        "adaptive_state": adaptive_state,
                        "tokens_used": state.tokens_used + token_delta,
                    },
                )

            case StrategyType.SINGLE_TOOL:
                return StepResult(
                    step_type=StepType.THINK,
                    step_id=step_id,
                    success=True,
                    thought=strategy_decision.reasoning,
                    action=f"tool_call: {strategy_decision.tool_name}({json.dumps(strategy_decision.tool_arguments or {})})",
                    llm_response=response,
                    state_updates={
                        "pending_tool_call": {
                            "name": strategy_decision.tool_name,
                            "arguments": strategy_decision.tool_arguments or {},
                        },
                        "adaptive_state": adaptive_state,
                        "tokens_used": state.tokens_used + token_delta,
                    },
                )

            case StrategyType.PARALLEL:
                return StepResult(
                    step_type=StepType.THINK,
                    step_id=step_id,
                    success=True,
                    thought=strategy_decision.reasoning,
                    action=f"parallel: {len(strategy_decision.parallel_actions or [])} tool calls",
                    llm_response=response,
                    state_updates={
                        "pending_parallel_calls": strategy_decision.parallel_actions or [],
                        "adaptive_state": adaptive_state,
                        "tokens_used": state.tokens_used + token_delta,
                    },
                )

            case StrategyType.PLAN_AND_EXECUTE:
                plan = strategy_decision.plan or []
                adaptive_state["plan_steps"] = plan
                adaptive_state["completed_plan_steps"] = []
                return StepResult(
                    step_type=StepType.PLAN,
                    step_id=step_id,
                    success=True,
                    thought=strategy_decision.reasoning,
                    action=f"plan: {len(plan)} steps",
                    llm_response=response,
                    state_updates={
                        "adaptive_state": adaptive_state,
                        "current_plan": plan,
                        "tokens_used": state.tokens_used + token_delta,
                    },
                )

            case StrategyType.PIVOT:
                adaptive_state["pivot_count"] = adaptive_state.get("pivot_count", 0) + 1
                return StepResult(
                    step_type=StepType.THINK,
                    step_id=step_id,
                    success=True,
                    thought=f"PIVOT: {strategy_decision.pivot_reason}",
                    action=f"New approach: {strategy_decision.pivot_new_approach}",
                    llm_response=response,
                    state_updates={
                        "adaptive_state": adaptive_state,
                        "tokens_used": state.tokens_used + token_delta,
                    },
                )

            case _:  # SEQUENTIAL or any other
                return StepResult(
                    step_type=StepType.THINK,
                    step_id=step_id,
                    success=True,
                    thought=strategy_decision.reasoning,
                    llm_response=response,
                    state_updates={
                        "adaptive_state": adaptive_state,
                        "tokens_used": state.tokens_used + token_delta,
                    },
                )

    def _parse_llm_response(
        self,
        content: str,
    ) -> tuple[str | None, str | None]:
        """
        Parse LLM response for thought and action.

        Expected format:
        Thought: <reasoning>
        Action: <action to take>
        """
        thought = None
        action = None

        lines = content.split("\n")
        current_section = None
        current_content: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("thought:"):
                # Save previous section if any
                if current_section == "thought" and current_content:
                    thought = " ".join(current_content)
                elif current_section == "action" and current_content:
                    action = " ".join(current_content)

                current_section = "thought"
                # Get content after "Thought:"
                remaining = stripped[8:].strip()
                current_content = [remaining] if remaining else []

            elif stripped.lower().startswith("action:"):
                # Save previous section
                if current_section == "thought" and current_content:
                    thought = " ".join(current_content)
                elif current_section == "action" and current_content:
                    action = " ".join(current_content)

                current_section = "action"
                remaining = stripped[7:].strip()
                current_content = [remaining] if remaining else []

            elif current_section and stripped:
                current_content.append(stripped)

        # Save last section
        if current_section == "thought" and current_content:
            thought = " ".join(current_content)
        elif current_section == "action" and current_content:
            action = " ".join(current_content)

        # If no structured format, treat entire content as thought
        if thought is None and action is None:
            thought = content.strip() if content.strip() else None

        return thought, action

    def _get_available_tool_names(self) -> list[str]:
        """Return tool names from the tool gateway registry."""
        if self.tool_gateway is not None:
            return self.tool_gateway.registry.list_tools()
        return []

    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable."""
        error_str = str(error).lower()

        # Rate limits and timeouts are recoverable
        if any(x in error_str for x in ["rate limit", "timeout", "503", "429"]):
            return True

        # Budget exceeded is not recoverable
        if "budget" in error_str:
            return False

        return True
