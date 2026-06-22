"""Tests for session-bundle correlation in TraceReader (Phase 4 slice).

A session bundle is the set of run JSONL files that share a ``bundle_id``
stamped in their event metadata (by the experimental subagent service).
``list_bundles`` / ``read_bundle`` group and aggregate them; the CLI
``arcana trace bundles`` / ``bundle <id>`` render the result.
"""

from __future__ import annotations

from arcana.contracts.trace import BudgetSnapshot, EventType, TraceEvent
from arcana.trace.reader import BundleSummary, TraceReader
from arcana.trace.writer import TraceWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_run(
    writer: TraceWriter,
    run_id: str,
    *,
    bundle_id: str | None = None,
    source_agent: str | None = None,
    delegated_by_run_id: str | None = None,
    tokens: int = 0,
    cost: float = 0.0,
    n_events: int = 2,
) -> None:
    """Write a small run trace with optional bundle correlation metadata."""
    meta: dict = {}
    if bundle_id is not None:
        meta["bundle_id"] = bundle_id
    if source_agent is not None:
        meta["source_agent"] = source_agent
    if delegated_by_run_id is not None:
        meta["delegated_by_run_id"] = delegated_by_run_id

    for i in range(n_events):
        writer.write(
            TraceEvent(
                run_id=run_id,
                event_type=EventType.LLM_CALL,
                metadata=dict(meta),
                budgets=BudgetSnapshot(tokens_used=tokens, cost_usd=cost)
                if i == n_events - 1
                else None,
            )
        )


# ---------------------------------------------------------------------------
# list_bundles
# ---------------------------------------------------------------------------


class TestListBundles:
    def test_groups_runs_by_bundle_id(self, tmp_path):
        w = TraceWriter(trace_dir=tmp_path)
        _write_run(w, "run-a", bundle_id="b1", source_agent="researcher")
        _write_run(w, "run-b", bundle_id="b1", source_agent="writer")
        _write_run(w, "run-c", bundle_id="b2", source_agent="solo")

        reader = TraceReader(trace_dir=tmp_path)
        bundles = {b.bundle_id: b for b in reader.list_bundles()}

        assert set(bundles) == {"b1", "b2"}
        assert bundles["b1"].run_count == 2
        assert bundles["b1"].agents == ["researcher", "writer"]
        assert bundles["b2"].run_count == 1

    def test_excludes_non_bundled_runs(self, tmp_path):
        w = TraceWriter(trace_dir=tmp_path)
        _write_run(w, "bundled", bundle_id="b1", source_agent="a")
        _write_run(w, "plain", bundle_id=None)  # no bundle metadata

        reader = TraceReader(trace_dir=tmp_path)
        bundles = reader.list_bundles()

        assert len(bundles) == 1
        assert bundles[0].bundle_id == "b1"

    def test_aggregates_tokens_and_cost(self, tmp_path):
        w = TraceWriter(trace_dir=tmp_path)
        _write_run(w, "r1", bundle_id="b1", source_agent="a", tokens=30, cost=0.01)
        _write_run(w, "r2", bundle_id="b1", source_agent="b", tokens=20, cost=0.02)

        reader = TraceReader(trace_dir=tmp_path)
        b = reader.list_bundles()[0]

        assert b.total_tokens == 50
        assert abs(b.total_cost_usd - 0.03) < 1e-9

    def test_empty_dir_returns_empty(self, tmp_path):
        reader = TraceReader(trace_dir=tmp_path / "missing")
        assert reader.list_bundles() == []


# ---------------------------------------------------------------------------
# read_bundle
# ---------------------------------------------------------------------------


class TestReadBundle:
    def test_returns_bundle_with_runs(self, tmp_path):
        w = TraceWriter(trace_dir=tmp_path)
        _write_run(w, "run-a", bundle_id="b1", source_agent="researcher")
        _write_run(w, "run-b", bundle_id="b1", source_agent="writer")

        reader = TraceReader(trace_dir=tmp_path)
        bundle = reader.read_bundle("b1")

        assert isinstance(bundle, BundleSummary)
        assert bundle.run_count == 2
        run_ids = {r.run_id for r in bundle.runs}
        assert run_ids == {"run-a", "run-b"}

    def test_preserves_delegation_parent_link(self, tmp_path):
        w = TraceWriter(trace_dir=tmp_path)
        _write_run(w, "parent", bundle_id="b1", source_agent="lead")
        _write_run(
            w,
            "child",
            bundle_id="b1",
            source_agent="helper",
            delegated_by_run_id="parent",
        )

        reader = TraceReader(trace_dir=tmp_path)
        bundle = reader.read_bundle("b1")
        by_id = {r.run_id: r for r in bundle.runs}

        assert by_id["child"].delegated_by_run_id == "parent"
        assert by_id["parent"].delegated_by_run_id is None

    def test_unknown_bundle_returns_none(self, tmp_path):
        w = TraceWriter(trace_dir=tmp_path)
        _write_run(w, "r1", bundle_id="b1", source_agent="a")

        reader = TraceReader(trace_dir=tmp_path)
        assert reader.read_bundle("nope") is None
