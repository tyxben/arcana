"""TraceReader - Read and query trace events from JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from arcana.contracts.context import ContextDecision, ContextReport
from arcana.contracts.trace import (
    AgentRole,
    BudgetSnapshot,
    EventType,
    PromptSnapshot,
    TraceEvent,
)


class PromptReplay(BaseModel):
    """Reconstructed prompt composition for a single LLM call.

    Joins the PROMPT_SNAPSHOT event (what was sent) and CONTEXT_DECISION
    event (why it was composed that way) for a given turn.
    """

    run_id: str
    turn: int
    prompt_snapshot: PromptSnapshot | None = None
    context_decision: ContextDecision | None = None
    context_report: ContextReport | None = None
    budget_snapshot: BudgetSnapshot | None = None


class BundleRunInfo(BaseModel):
    """One run's correlation + usage summary within a session bundle.

    Correlation fields are read from event ``metadata`` (stamped by the
    experimental subagent service): ``bundle_id`` groups runs, ``source_agent``
    names the subagent that produced the run, and ``delegated_by_run_id`` links
    a delegated run back to its parent.
    """

    run_id: str
    source_agent: str | None = None
    delegated_by_run_id: str | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_events: int = 0
    start_time: str | None = None
    end_time: str | None = None


class BundleSummary(BaseModel):
    """A session bundle: the set of runs sharing one ``bundle_id``."""

    bundle_id: str
    runs: list[BundleRunInfo] = []
    agents: list[str] = []
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def run_count(self) -> int:
        return len(self.runs)


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
                total_tokens += e.budgets.tokens_used
                total_cost += e.budgets.cost_usd

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

    # ------------------------------------------------------------------
    # Session bundles -- correlate multiple runs by ``bundle_id`` metadata
    # ------------------------------------------------------------------

    def _run_bundle_info(self, run_id: str) -> tuple[str | None, BundleRunInfo]:
        """Read a run's bundle correlation metadata + usage summary.

        Returns ``(bundle_id, BundleRunInfo)``. ``bundle_id`` is ``None`` when
        the run carries no bundle correlation (a plain, non-bundled run).
        """
        bundle_id: str | None = None
        source_agent: str | None = None
        delegated_by: str | None = None
        for event in self.iter_events(run_id):
            meta = event.metadata or {}
            if bundle_id is None and meta.get("bundle_id"):
                bundle_id = meta["bundle_id"]
            if source_agent is None and meta.get("source_agent"):
                source_agent = meta["source_agent"]
            if delegated_by is None and meta.get("delegated_by_run_id"):
                delegated_by = meta["delegated_by_run_id"]
            if bundle_id and source_agent and delegated_by:
                break

        summary = self.get_summary(run_id)
        info = BundleRunInfo(
            run_id=run_id,
            source_agent=source_agent,
            delegated_by_run_id=delegated_by,
            total_tokens=summary.get("total_tokens", 0),
            total_cost_usd=summary.get("total_cost_usd", 0.0),
            total_events=summary.get("total_events", 0),
            start_time=summary.get("start_time"),
            end_time=summary.get("end_time"),
        )
        return bundle_id, info

    def _summarize_bundle(
        self, bundle_id: str, runs: list[BundleRunInfo]
    ) -> BundleSummary:
        ordered = sorted(runs, key=lambda r: r.start_time or "")
        agents = sorted({r.source_agent for r in ordered if r.source_agent})
        return BundleSummary(
            bundle_id=bundle_id,
            runs=ordered,
            agents=agents,
            total_tokens=sum(r.total_tokens for r in ordered),
            total_cost_usd=sum(r.total_cost_usd for r in ordered),
        )

    def list_bundles(self) -> list[BundleSummary]:
        """Group every bundled run under ``trace_dir`` by its ``bundle_id``.

        Runs with no ``bundle_id`` metadata (plain single runs) are skipped.
        Bundles are returned most-recent-first by their earliest run.
        """
        if not self.trace_dir.exists():
            return []
        groups: dict[str, list[BundleRunInfo]] = {}
        for trace_file in self.trace_dir.glob("*.jsonl"):
            bundle_id, info = self._run_bundle_info(trace_file.stem)
            if bundle_id is None:
                continue
            groups.setdefault(bundle_id, []).append(info)
        summaries = [
            self._summarize_bundle(bid, runs) for bid, runs in groups.items()
        ]
        summaries.sort(
            key=lambda b: (b.runs[0].start_time or "") if b.runs else "",
            reverse=True,
        )
        return summaries

    def read_bundle(self, bundle_id: str) -> BundleSummary | None:
        """Return the bundle with ``bundle_id``, or ``None`` if not found."""
        if not self.trace_dir.exists():
            return None
        runs: list[BundleRunInfo] = []
        for trace_file in self.trace_dir.glob("*.jsonl"):
            found_id, info = self._run_bundle_info(trace_file.stem)
            if found_id == bundle_id:
                runs.append(info)
        if not runs:
            return None
        return self._summarize_bundle(bundle_id, runs)

    def list_turns(self, run_id: str) -> list[int]:
        """Return the turn numbers that have replay evidence in this run.

        Turns appear here if there is a CONTEXT_DECISION or PROMPT_SNAPSHOT
        event for them. The list is sorted ascending.
        """
        turns: set[int] = set()
        for event in self.iter_events(run_id):
            if event.event_type not in (
                EventType.CONTEXT_DECISION,
                EventType.PROMPT_SNAPSHOT,
            ):
                continue
            turn = event.metadata.get("turn")
            if turn is None:
                # Fall back to the decision dump, if present
                decision = event.metadata.get("context_decision")
                if isinstance(decision, dict):
                    turn = decision.get("turn")
            if isinstance(turn, int):
                turns.add(turn)
        return sorted(turns)

    def replay_prompt(self, run_id: str, turn: int) -> PromptReplay | None:
        """Reconstruct the prompt composition for a specific turn.

        Joins the PROMPT_SNAPSHOT event (what was sent to the LLM) and the
        CONTEXT_DECISION event (why it was composed that way). Returns None
        if neither event exists for the requested turn.

        When ``RuntimeConfig.trace_include_prompt_snapshots`` was disabled
        during the run, ``prompt_snapshot`` will be None but the context
        decision remains available.
        """
        snapshot: PromptSnapshot | None = None
        decision: ContextDecision | None = None
        report: ContextReport | None = None
        budget: BudgetSnapshot | None = None
        seen = False

        for event in self.iter_events(run_id):
            if event.event_type not in (
                EventType.CONTEXT_DECISION,
                EventType.PROMPT_SNAPSHOT,
            ):
                continue
            event_turn = event.metadata.get("turn")
            if event_turn is None:
                decision_dump = event.metadata.get("context_decision")
                if isinstance(decision_dump, dict):
                    event_turn = decision_dump.get("turn")
            if event_turn != turn:
                continue
            seen = True

            if event.event_type == EventType.PROMPT_SNAPSHOT:
                snap = event.metadata.get("prompt_snapshot")
                if isinstance(snap, dict):
                    try:
                        snapshot = PromptSnapshot.model_validate(snap)
                    except ValueError:
                        snapshot = None
                    if snapshot is not None and snapshot.budget_snapshot is not None:
                        budget = snapshot.budget_snapshot
            elif event.event_type == EventType.CONTEXT_DECISION:
                dec = event.metadata.get("context_decision")
                if isinstance(dec, dict):
                    try:
                        decision = ContextDecision.model_validate(dec)
                    except ValueError:
                        decision = None
                rep = event.metadata.get("context_report")
                if isinstance(rep, dict):
                    try:
                        report = ContextReport.model_validate(rep)
                    except ValueError:
                        report = None
            if event.budgets is not None and budget is None:
                budget = event.budgets

        if not seen:
            return None

        return PromptReplay(
            run_id=run_id,
            turn=turn,
            prompt_snapshot=snapshot,
            context_decision=decision,
            context_report=report,
            budget_snapshot=budget,
        )

    def collect_turn(
        self,
        run_id: str,
        turn: int,
    ) -> dict[str, Any]:
        """Collect every trace event belonging to a single turn.

        Returns a dict with keys:
          - ``turn_event``: the TURN event for this turn (or None)
          - ``context_decision``: CONTEXT_DECISION event (or None)
          - ``prompt_snapshot``: PROMPT_SNAPSHOT event (or None)
          - ``tool_calls``: list of TOOL_CALL events emitted by this turn
          - ``cognitive``: list of COGNITIVE_PRIMITIVE events for this turn
          - ``errors``: list of ERROR events whose parent is this turn
          - ``all``: chronological list of all events attached to this turn

        The join key is ``parent_step_id == turn_event.step_id``. Works even
        on pool-replay traces (pass the per-agent filtered events in via
        a TraceReader scoped to that run).
        """
        turn_event: TraceEvent | None = None
        for event in self.iter_events(run_id):
            if event.event_type != EventType.TURN:
                continue
            if event.metadata.get("step") == turn:
                turn_event = event
                break

        result: dict[str, Any] = {
            "turn_event": turn_event,
            "context_decision": None,
            "prompt_snapshot": None,
            "tool_calls": [],
            "cognitive": [],
            "errors": [],
            "all": [],
        }
        if turn_event is None:
            return result

        parent_id = turn_event.step_id
        for event in self.iter_events(run_id):
            if event.step_id == parent_id:
                result["all"].append(event)
                continue
            if event.parent_step_id != parent_id:
                continue
            result["all"].append(event)
            if event.event_type == EventType.CONTEXT_DECISION:
                result["context_decision"] = event
            elif event.event_type == EventType.PROMPT_SNAPSHOT:
                result["prompt_snapshot"] = event
            elif event.event_type == EventType.TOOL_CALL:
                result["tool_calls"].append(event)
            elif event.event_type == EventType.COGNITIVE_PRIMITIVE:
                result["cognitive"].append(event)
            elif event.event_type == EventType.ERROR:
                result["errors"].append(event)

        return result
