"""TraceReader - Read and query trace events from JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from arcana.contracts.trace import AgentRole, EventType, TraceEvent


class TraceReader:
    """
    Reads and queries trace events from JSONL files.

    Supports:
    - Reading all events for a run
    - Filtering by event type, role, time range
    - Reconstructing step sequences
    """

    def __init__(self, trace_dir: str | Path = "./traces"):
        """
        Initialize the trace reader.

        Args:
            trace_dir: Directory containing trace files
        """
        self.trace_dir = Path(trace_dir)

    def _get_trace_file(self, run_id: str) -> Path:
        """Get the trace file path for a run."""
        return self.trace_dir / f"{run_id}.jsonl"

    def exists(self, run_id: str) -> bool:
        """Check if a trace file exists for the given run."""
        return self._get_trace_file(run_id).exists()

    def read_events(self, run_id: str) -> list[TraceEvent]:
        """
        Read all trace events for a run.

        Args:
            run_id: The run ID

        Returns:
            List of TraceEvent objects, ordered by timestamp
        """
        trace_file = self._get_trace_file(run_id)
        if not trace_file.exists():
            return []

        events = []
        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        events.append(TraceEvent.model_validate(data))
                    except (json.JSONDecodeError, ValueError):
                        # Skip malformed lines
                        continue

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        return events

    def read_raw(self, run_id: str) -> list[dict[str, Any]]:
        """
        Read raw JSON data for a run.

        Args:
            run_id: The run ID

        Returns:
            List of raw dictionaries
        """
        trace_file = self._get_trace_file(run_id)
        if not trace_file.exists():
            return []

        events = []
        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return events

    def iter_events(self, run_id: str) -> Iterator[TraceEvent]:
        """
        Iterate over trace events for a run.

        Args:
            run_id: The run ID

        Yields:
            TraceEvent objects
        """
        trace_file = self._get_trace_file(run_id)
        if not trace_file.exists():
            return

        with open(trace_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        yield TraceEvent.model_validate(data)
                    except (json.JSONDecodeError, ValueError):
                        continue

    def filter_events(
        self,
        run_id: str,
        event_types: list[EventType] | None = None,
        roles: list[AgentRole] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[TraceEvent]:
        """
        Filter trace events by various criteria.

        Args:
            run_id: The run ID
            event_types: Filter by event types
            roles: Filter by agent roles
            start_time: Filter events after this time
            end_time: Filter events before this time

        Returns:
            Filtered list of TraceEvent objects
        """
        events = self.read_events(run_id)

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if roles:
            events = [e for e in events if e.role in roles]

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events

    def get_llm_calls(self, run_id: str) -> list[TraceEvent]:
        """Get all LLM call events for a run."""
        return self.filter_events(run_id, event_types=[EventType.LLM_CALL])

    def get_tool_calls(self, run_id: str) -> list[TraceEvent]:
        """Get all tool call events for a run."""
        return self.filter_events(run_id, event_types=[EventType.TOOL_CALL])

    def get_errors(self, run_id: str) -> list[TraceEvent]:
        """Get all error events for a run."""
        return self.filter_events(run_id, event_types=[EventType.ERROR])

    def get_step_sequence(self, run_id: str) -> list[str]:
        """
        Get the sequence of step IDs for a run.

        Args:
            run_id: The run ID

        Returns:
            Ordered list of step IDs
        """
        events = self.read_events(run_id)
        seen = set()
        sequence = []

        for event in events:
            if event.step_id not in seen:
                seen.add(event.step_id)
                sequence.append(event.step_id)

        return sequence

    def get_summary(self, run_id: str) -> dict[str, Any]:
        """
        Get a summary of a trace run.

        Args:
            run_id: The run ID

        Returns:
            Summary dictionary with statistics
        """
        events = self.read_events(run_id)

        if not events:
            return {"run_id": run_id, "exists": False}

        llm_calls = [e for e in events if e.event_type == EventType.LLM_CALL]
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        errors = [e for e in events if e.event_type == EventType.ERROR]

        # Find stop reason
        stop_event = next((e for e in reversed(events) if e.stop_reason), None)

        # Calculate unique steps in-place (avoid re-reading file)
        seen_steps: set[str] = set()
        for e in events:
            seen_steps.add(e.step_id)

        # Calculate total tokens and cost
        total_tokens = 0
        total_cost = 0.0
        for e in events:
            if e.budgets:
                total_tokens = max(total_tokens, e.budgets.tokens_used)
                total_cost = max(total_cost, e.budgets.cost_usd)

        return {
            "run_id": run_id,
            "exists": True,
            "total_events": len(events),
            "llm_calls": len(llm_calls),
            "tool_calls": len(tool_calls),
            "errors": len(errors),
            "unique_steps": len(seen_steps),
            "start_time": events[0].timestamp.isoformat() if events else None,
            "end_time": events[-1].timestamp.isoformat() if events else None,
            "stop_reason": stop_event.stop_reason.value if stop_event and stop_event.stop_reason else None,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
        }
