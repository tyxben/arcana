"""Replay mechanism for debugging and testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcana.contracts.llm import LLMResponse, TokenUsage
from arcana.contracts.trace import EventType, TraceEvent

if TYPE_CHECKING:
    from arcana.contracts.state import AgentState
    from arcana.trace.reader import TraceReader


class ReplayCache:
    """
    Cache for storing and retrieving replay data.

    Stores LLM responses and tool results indexed by request digest.
    """

    def __init__(self) -> None:
        """Initialize replay cache."""
        self._llm_responses: dict[str, dict[str, Any]] = {}
        self._tool_results: dict[str, dict[str, Any]] = {}

    def load_from_trace(self, reader: TraceReader, run_id: str) -> None:
        """
        Load cache from trace events.

        Args:
            reader: TraceReader instance
            run_id: Run ID to load from
        """
        events = reader.read_events(run_id)

        for event in events:
            if event.event_type == EventType.LLM_CALL:
                # Store LLM response indexed by request digest
                if event.llm_request_digest and event.llm_response_digest:
                    self._llm_responses[event.llm_request_digest] = {
                        "content": event.llm_response_content,
                        "model": event.model,
                        "usage": event.llm_usage,
                        "response_digest": event.llm_response_digest,
                    }

            elif event.event_type == EventType.TOOL_CALL:
                # Store tool result indexed by tool call digest
                if event.tool_call and event.tool_result:
                    call_digest = event.tool_call.get("idempotency_key")
                    if call_digest:
                        self._tool_results[call_digest] = event.tool_result

    def get_llm_response(self, request_digest: str) -> LLMResponse | None:
        """
        Get cached LLM response for a request.

        Args:
            request_digest: Request digest to look up

        Returns:
            Cached LLM response or None if not found
        """
        data = self._llm_responses.get(request_digest)
        if not data:
            return None

        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return LLMResponse(
            content=data.get("content"),
            model=data.get("model", "unknown"),
            usage=usage,
            finish_reason=data.get("finish_reason", "stop"),
        )

    def get_tool_result(self, call_digest: str) -> dict[str, Any] | None:
        """
        Get cached tool result.

        Args:
            call_digest: Tool call digest

        Returns:
            Cached result or None
        """
        return self._tool_results.get(call_digest)

    def has_llm_response(self, request_digest: str) -> bool:
        """Check if LLM response is cached."""
        return request_digest in self._llm_responses

    def has_tool_result(self, call_digest: str) -> bool:
        """Check if tool result is cached."""
        return call_digest in self._tool_results


class ReplayEngine:
    """
    Engine for replaying agent executions.

    Enables:
    - Debugging by reproducing exact execution paths
    - Testing with deterministic results
    - Analysis of failure scenarios
    """

    def __init__(
        self,
        reader: TraceReader,
        *,
        strict_mode: bool = True,
    ) -> None:
        """
        Initialize replay engine.

        Args:
            reader: TraceReader for loading events
            strict_mode: If True, fail on cache miss; if False, allow real calls
        """
        self.reader = reader
        self.strict_mode = strict_mode
        self.cache = ReplayCache()

    async def replay_run(
        self,
        run_id: str,
        *,
        from_step: int | None = None,
        to_step: int | None = None,
    ) -> AgentState:
        """
        Replay a previous run.

        Args:
            run_id: Run ID to replay
            from_step: Optional starting step (default: 0)
            to_step: Optional ending step (default: last)

        Returns:
            Final agent state after replay

        Raises:
            ReplayError: If replay fails (e.g., cache miss in strict mode)
        """
        # Load cache from trace
        self.cache.load_from_trace(self.reader, run_id)

        # Get all events for this run
        events = self.reader.read_events(run_id)

        # Filter by step range
        if from_step is not None or to_step is not None:
            events = [
                e
                for e in events
                if (from_step is None or e.step_id >= f"step_{from_step}")
                and (to_step is None or e.step_id <= f"step_{to_step}")
            ]

        # Reconstruct state by replaying events
        state = self._reconstruct_initial_state(events)

        # Apply events in sequence
        for event in events:
            state = self._apply_event(state, event)

        return state

    async def get_divergence_point(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> tuple[int, TraceEvent, TraceEvent] | None:
        """
        Find where two runs diverged.

        Useful for comparing a working run vs a failing run.

        Args:
            run_id_a: First run ID
            run_id_b: Second run ID

        Returns:
            (step_number, event_a, event_b) at divergence point, or None if identical
        """
        events_a = self.reader.read_events(run_id_a)
        events_b = self.reader.read_events(run_id_b)

        min_length = min(len(events_a), len(events_b))

        for i in range(min_length):
            event_a = events_a[i]
            event_b = events_b[i]

            # Compare state hashes
            if event_a.state_after_hash != event_b.state_after_hash:
                return (i, event_a, event_b)

            # Compare LLM responses
            if (
                event_a.llm_response_digest
                and event_b.llm_response_digest
                and event_a.llm_response_digest != event_b.llm_response_digest
            ):
                return (i, event_a, event_b)

        # Check if one is longer
        if len(events_a) != len(events_b):
            longer = events_a if len(events_a) > len(events_b) else events_b
            return (min_length, longer[min_length], longer[min_length])

        return None

    def _reconstruct_initial_state(
        self,
        events: list[TraceEvent],
    ) -> AgentState:
        """Reconstruct initial state from events."""
        from arcana.contracts.state import AgentState, ExecutionStatus

        # Get first event to extract metadata
        first_event = events[0] if events else None

        if not first_event:
            # No events, return empty state
            return AgentState(
                run_id="replay",
                goal="Unknown",
                status=ExecutionStatus.PENDING,
            )

        # Reconstruct from first event
        return AgentState(
            run_id=first_event.run_id,
            task_id=first_event.task_id,
            goal=first_event.metadata.get("goal", "Unknown")
            if first_event.metadata
            else "Unknown",
            status=ExecutionStatus.RUNNING,
        )

    def _apply_event(
        self,
        state: AgentState,
        event: TraceEvent,
    ) -> AgentState:
        """
        Apply a single event to state.

        Args:
            state: Current state
            event: Event to apply

        Returns:
            Updated state
        """
        # Update state based on event type
        if event.event_type == EventType.STATE_CHANGE:
            if event.metadata and "status" in event.metadata:
                state.status = event.metadata["status"]

        elif event.event_type == EventType.LLM_CALL:
            # Update token usage
            if event.llm_usage:
                state.tokens_used += event.llm_usage.get("total_tokens", 0)

        elif event.event_type == EventType.TOOL_CALL:
            # Track tool calls
            pass

        # Update step counter
        if event.step_id and event.step_id.startswith("step_"):
            try:
                step_num = int(event.step_id.split("_")[1])
                state.current_step = max(state.current_step, step_num)
            except (ValueError, IndexError):
                pass

        return state


class ReplayError(Exception):
    """Error during replay."""

    def __init__(
        self,
        message: str,
        run_id: str | None = None,
        step_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.step_id = step_id
