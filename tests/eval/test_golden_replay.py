"""Tests for golden-trace recording + asymmetric replay diff (F5, slice 6)."""

from __future__ import annotations

from arcana.contracts.eval import EvalCase, OutcomeCriterion
from arcana.contracts.trace import EventType, ToolCallRecord, TraceEvent
from arcana.eval.golden import GoldenStore, build_golden, replay_diff


def _ev(event_type, *, metadata=None, tool_call=None) -> TraceEvent:
    return TraceEvent(
        run_id="r", event_type=event_type,
        metadata=metadata or {}, tool_call=tool_call,
    )


def _case() -> EvalCase:
    return EvalCase(id="c1", goal="do the thing", expected_outcome=OutcomeCriterion.STATUS)


def _read_tool() -> TraceEvent:
    return _ev(EventType.TOOL_CALL,
               tool_call=ToolCallRecord(name="look", args_digest="d", side_effect="read"))


def _baseline_events() -> list[TraceEvent]:
    return [
        _ev(EventType.LLM_CALL),
        _read_tool(),
        _ev(EventType.GUARDRAIL_DECISION, metadata={"action": "allow", "boundary": "tool"}),
    ]


class TestReplayDiff:
    def test_identical_run_matches(self):
        case = _case()
        golden = build_golden(case, _baseline_events())
        diff = replay_diff(golden, _baseline_events(), case=case)
        assert diff.golden_status == "match"
        assert diff.is_regression is False
        assert diff.signal_regressions == []

    def test_guardrail_allow_to_block_regresses(self):
        case = _case()
        golden = build_golden(case, _baseline_events())
        worse = [
            _ev(EventType.LLM_CALL),
            _read_tool(),
            _ev(EventType.GUARDRAIL_DECISION, metadata={"action": "block", "boundary": "tool"}),
        ]
        diff = replay_diff(golden, worse, case=case)
        assert diff.is_regression is True
        assert diff.golden_status == "regressed"
        assert any("guardrail_blocks" in r for r in diff.signal_regressions)

    def test_new_write_tool_regresses(self):
        case = _case()
        golden = build_golden(case, _baseline_events())
        worse = _baseline_events() + [
            _ev(EventType.TOOL_CALL,
                tool_call=ToolCallRecord(name="rm", args_digest="d", side_effect="write")),
        ]
        diff = replay_diff(golden, worse, case=case)
        assert diff.is_regression is True
        assert any("write_tool_calls" in r for r in diff.signal_regressions)

    def test_provider_degradation_regresses(self):
        case = _case()
        golden = build_golden(case, _baseline_events())
        worse = [
            _ev(EventType.LLM_CALL, metadata={"degraded_capabilities": ["tool_calls"]}),
            _read_tool(),
            _ev(EventType.GUARDRAIL_DECISION, metadata={"action": "allow", "boundary": "tool"}),
        ]
        diff = replay_diff(golden, worse, case=case)
        assert diff.is_regression is True
        assert any("degraded" in r for r in diff.signal_regressions)

    def test_removing_a_denial_is_improvement_not_regression(self):
        case = _case()
        # golden had a permission denial; the new run has none -> safer.
        golden_events = _baseline_events() + [
            _ev(EventType.TOOL_CALL,
                metadata={"permission_decision": {"action": "deny"}},
                tool_call=ToolCallRecord(name="x", args_digest="d", side_effect="read")),
        ]
        golden = build_golden(case, golden_events)
        diff = replay_diff(golden, _baseline_events(), case=case)
        assert diff.is_regression is False
        assert diff.golden_status == "improved"

    def test_goal_change_marks_new_not_regression(self):
        case = _case()
        golden = build_golden(case, _baseline_events())
        other = EvalCase(id="c1", goal="a completely different goal",
                         expected_outcome=OutcomeCriterion.STATUS)
        diff = replay_diff(golden, _baseline_events(), case=other)
        assert diff.golden_status == "new"
        assert diff.is_regression is False


class TestGoldenStore:
    def test_record_and_load_roundtrip(self, tmp_path):
        store = GoldenStore(tmp_path)
        golden = build_golden(_case(), _baseline_events(), suite_name="suite")
        path = store.record(golden, force=True)  # explicit op -> live
        assert path.exists()
        loaded = store.load("suite", "c1")
        assert loaded is not None
        assert loaded.skeleton_digest == golden.skeleton_digest

    def test_unforced_record_lands_in_candidates(self, tmp_path):
        store = GoldenStore(tmp_path)
        golden = build_golden(_case(), _baseline_events(), suite_name="suite")
        # Without force, a record never becomes the trusted live baseline.
        path = store.record(golden)
        assert "_candidates" in str(path)
        assert store.load("suite", "c1") is None  # nothing trusted yet
        # force promotes it to live.
        forced = store.record(golden, force=True)
        assert "_candidates" not in str(forced)
        assert store.load("suite", "c1") is not None

    def test_load_rejects_tampered_golden(self, tmp_path):
        import json

        store = GoldenStore(tmp_path)
        golden = build_golden(_case(), _baseline_events(), suite_name="suite")
        path = store.record(golden, force=True)
        # Hand-edit the committed golden without re-recording its digest.
        data = json.loads(path.read_text())
        data["signals"]["permission_denials"] = 99  # relax a boundary silently
        path.write_text(json.dumps(data))
        import pytest

        with pytest.raises(ValueError, match="digest mismatch"):
            store.load("suite", "c1")

    def test_missing_returns_none(self, tmp_path):
        assert GoldenStore(tmp_path).load("suite", "nope") is None
