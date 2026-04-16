"""Tests for TraceReader.replay_prompt and list_turns.

Direct round-trip: write CONTEXT_DECISION + PROMPT_SNAPSHOT events via
TraceWriter, then reconstruct through TraceReader.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from arcana.contracts.context import (
    ContextDecision,
    ContextReport,
    MessageDecision,
)
from arcana.contracts.trace import (
    BudgetSnapshot,
    EventType,
    PromptSnapshot,
    TraceEvent,
)
from arcana.trace.reader import PromptReplay, TraceReader
from arcana.trace.writer import TraceWriter


@pytest.fixture()
def trace_dir(tmp_path):
    return tmp_path / "traces"


def _write_context_decision(writer: TraceWriter, run_id: str, turn: int) -> ContextDecision:
    decision = ContextDecision(
        turn=turn,
        strategy="tail_preserve",
        budget_total=1000,
        budget_used=800,
        messages_in=4,
        messages_out=3,
        compressed_count=1,
        history_compressed=True,
        decisions=[
            MessageDecision(
                index=0, role="system", outcome="kept",
                token_count_before=10, token_count_after=10,
                reason="tail_preserve_head",
            ),
            MessageDecision(
                index=1, role="user", outcome="compressed",
                fidelity="L2", relevance_score=0.3,
                token_count_before=100, token_count_after=20,
                reason="tail_preserve_middle_compressed",
            ),
            MessageDecision(
                index=2, role="assistant", outcome="kept",
                token_count_before=50, token_count_after=50,
                reason="tail_preserve_tail",
            ),
            MessageDecision(
                index=3, role="user", outcome="kept",
                token_count_before=10, token_count_after=10,
                reason="tail_preserve_tail",
            ),
        ],
    )
    report = ContextReport(
        turn=turn,
        strategy_used="tail_preserve",
        total_tokens=90,
        utilization=0.9,
    )
    writer.write(TraceEvent(
        run_id=run_id,
        event_type=EventType.CONTEXT_DECISION,
        metadata={
            "turn": turn,
            "context_decision": decision.model_dump(),
            "context_report": report.model_dump(),
        },
    ))
    return decision


def _write_prompt_snapshot(writer: TraceWriter, run_id: str, turn: int) -> PromptSnapshot:
    snapshot = PromptSnapshot(
        turn=turn,
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        tools=[{"name": "calc", "schema": {}}],
        response_format=None,
        budget_snapshot=BudgetSnapshot(
            max_tokens=10_000,
            tokens_used=123,
            max_cost_usd=1.0,
            cost_usd=0.01,
        ),
    )
    writer.write(TraceEvent(
        run_id=run_id,
        event_type=EventType.PROMPT_SNAPSHOT,
        model="deepseek-chat",
        metadata={
            "turn": turn,
            "prompt_snapshot": snapshot.model_dump(),
        },
    ))
    return snapshot


class TestListTurns:
    def test_empty_run(self, trace_dir):
        writer = TraceWriter(trace_dir=trace_dir)
        reader = TraceReader(trace_dir=trace_dir)
        run_id = str(uuid4())
        # Ensure file exists but is empty of replay events
        writer.write(TraceEvent(
            run_id=run_id, event_type=EventType.LLM_CALL, metadata={},
        ))
        assert reader.list_turns(run_id) == []

    def test_multiple_turns(self, trace_dir):
        writer = TraceWriter(trace_dir=trace_dir)
        reader = TraceReader(trace_dir=trace_dir)
        run_id = str(uuid4())
        for turn in [0, 2, 5]:
            _write_context_decision(writer, run_id, turn)
        assert reader.list_turns(run_id) == [0, 2, 5]

    def test_dedupes_across_both_event_types(self, trace_dir):
        writer = TraceWriter(trace_dir=trace_dir)
        reader = TraceReader(trace_dir=trace_dir)
        run_id = str(uuid4())
        _write_context_decision(writer, run_id, 3)
        _write_prompt_snapshot(writer, run_id, 3)
        assert reader.list_turns(run_id) == [3]


class TestReplayPrompt:
    def test_missing_turn_returns_none(self, trace_dir):
        writer = TraceWriter(trace_dir=trace_dir)
        reader = TraceReader(trace_dir=trace_dir)
        run_id = str(uuid4())
        _write_context_decision(writer, run_id, 0)
        assert reader.replay_prompt(run_id, turn=99) is None

    def test_decision_only_when_snapshot_absent(self, trace_dir):
        writer = TraceWriter(trace_dir=trace_dir)
        reader = TraceReader(trace_dir=trace_dir)
        run_id = str(uuid4())
        decision = _write_context_decision(writer, run_id, 0)

        replay = reader.replay_prompt(run_id, turn=0)
        assert replay is not None
        assert isinstance(replay, PromptReplay)
        assert replay.prompt_snapshot is None
        assert replay.context_decision is not None
        assert replay.context_decision.strategy == decision.strategy
        assert len(replay.context_decision.decisions) == 4
        # context_report was also attached
        assert replay.context_report is not None
        assert replay.context_report.strategy_used == "tail_preserve"

    def test_full_replay_round_trip(self, trace_dir):
        writer = TraceWriter(trace_dir=trace_dir)
        reader = TraceReader(trace_dir=trace_dir)
        run_id = str(uuid4())
        decision = _write_context_decision(writer, run_id, 1)
        snapshot = _write_prompt_snapshot(writer, run_id, 1)

        replay = reader.replay_prompt(run_id, turn=1)
        assert replay is not None
        # Snapshot round-trip
        assert replay.prompt_snapshot is not None
        assert replay.prompt_snapshot.model == snapshot.model
        assert replay.prompt_snapshot.messages == snapshot.messages
        assert replay.prompt_snapshot.tools == snapshot.tools
        # Decision round-trip
        assert replay.context_decision is not None
        assert replay.context_decision.strategy == decision.strategy
        # Budget snapshot pulled from PromptSnapshot
        assert replay.budget_snapshot is not None
        assert replay.budget_snapshot.tokens_used == 123

    def test_multiple_turns_isolated(self, trace_dir):
        writer = TraceWriter(trace_dir=trace_dir)
        reader = TraceReader(trace_dir=trace_dir)
        run_id = str(uuid4())
        _write_context_decision(writer, run_id, 0)
        _write_prompt_snapshot(writer, run_id, 0)
        _write_context_decision(writer, run_id, 1)
        _write_prompt_snapshot(writer, run_id, 1)

        r0 = reader.replay_prompt(run_id, 0)
        r1 = reader.replay_prompt(run_id, 1)
        assert r0 is not None and r1 is not None
        assert r0.turn == 0
        assert r1.turn == 1
        # Each reconstructs independently
        assert r0.prompt_snapshot is not None
        assert r1.prompt_snapshot is not None


class TestContextDecisionTraceSerialization:
    """Verify the full-model-dump CONTEXT_DECISION metadata can be
    recovered losslessly."""

    def test_round_trip_preserves_message_decisions(self, trace_dir):
        writer = TraceWriter(trace_dir=trace_dir)
        reader = TraceReader(trace_dir=trace_dir)
        run_id = str(uuid4())
        _write_context_decision(writer, run_id, 2)

        events = reader.filter_events(run_id, event_types=[EventType.CONTEXT_DECISION])
        assert len(events) == 1
        dump = events[0].metadata["context_decision"]
        recovered = ContextDecision.model_validate(dump)
        assert recovered.strategy == "tail_preserve"
        assert len(recovered.decisions) == 4
        assert recovered.decisions[1].fidelity == "L2"
        assert recovered.decisions[1].relevance_score == 0.3
