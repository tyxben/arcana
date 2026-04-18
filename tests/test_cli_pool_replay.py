"""CLI tests for v0.8.0 ``arcana trace pool-replay`` + --agent filter."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from arcana.cli.main import app

runner = CliRunner()


def _write_trace(path: Path, run_id: str, events: list[dict]) -> None:
    """Write a minimal JSONL trace file the CLI can read."""
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{run_id}.jsonl"
    with file.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _event(
    run_id: str,
    event_type: str,
    *,
    source_agent: str | None = None,
    extra_metadata: dict | None = None,
    model: str = "mock-model",
) -> dict:
    meta: dict = dict(extra_metadata or {})
    if source_agent is not None:
        meta["source_agent"] = source_agent
    return {
        "run_id": run_id,
        "task_id": None,
        "step_id": f"step-{source_agent or 'x'}-{event_type}",
        "timestamp": datetime.now(UTC).isoformat(),
        "role": "system",
        "event_type": event_type,
        "state_before_hash": None,
        "state_after_hash": None,
        "llm_request_digest": None,
        "llm_response_digest": None,
        "model": model,
        "llm_latency_ms": None,
        "tool_call": None,
        "budgets": None,
        "stop_reason": None,
        "stop_detail": None,
        "metadata": meta,
    }


# ── pool-replay summary view ────────────────────────────────────────────


class TestPoolReplaySummary:
    def test_summary_lists_each_pool_agent(self, tmp_path):
        run_id = "pool-run-1"
        events = [
            _event(run_id, "llm_call", source_agent="planner"),
            _event(run_id, "llm_call", source_agent="planner"),
            _event(run_id, "llm_call", source_agent="executor"),
            _event(run_id, "tool_call", source_agent="executor"),
        ]
        _write_trace(tmp_path, run_id, events)

        result = runner.invoke(
            app,
            ["trace", "pool-replay", run_id, "--dir", str(tmp_path)],
        )

        assert result.exit_code == 0, result.stdout
        assert "planner" in result.stdout
        assert "executor" in result.stdout
        # Each agent's event count is present
        assert "2" in result.stdout
        assert "pool-replay" in result.stdout  # usage hint for next step

    def test_summary_errors_when_no_pool_events(self, tmp_path):
        run_id = "legacy-run"
        # No source_agent metadata → legacy single-agent trace
        events = [_event(run_id, "llm_call")]
        _write_trace(tmp_path, run_id, events)

        result = runner.invoke(
            app,
            ["trace", "pool-replay", run_id, "--dir", str(tmp_path)],
        )

        assert result.exit_code == 1
        assert "No pool events" in result.stdout


# ── show --agent filter ─────────────────────────────────────────────────


class TestTraceShowAgentFilter:
    def test_show_filters_to_single_agent(self, tmp_path):
        run_id = "scoped"
        events = [
            _event(run_id, "llm_call", source_agent="a"),
            _event(run_id, "llm_call", source_agent="b"),
            _event(run_id, "tool_call", source_agent="a"),
        ]
        _write_trace(tmp_path, run_id, events)

        result = runner.invoke(
            app,
            ["trace", "show", run_id, "--dir", str(tmp_path), "--agent", "a"],
        )

        assert result.exit_code == 0, result.stdout
        # header confirms scoping
        assert "agent=a" in result.stdout
        # event count = 2 (a had 2 events)
        assert "Events: 2" in result.stdout

    def test_show_without_agent_includes_all(self, tmp_path):
        run_id = "unscoped"
        events = [
            _event(run_id, "llm_call", source_agent="a"),
            _event(run_id, "llm_call", source_agent="b"),
        ]
        _write_trace(tmp_path, run_id, events)

        result = runner.invoke(
            app,
            ["trace", "show", run_id, "--dir", str(tmp_path)],
        )

        assert result.exit_code == 0, result.stdout
        # Per-event agent tags appear when present
        assert "[a]" in result.stdout
        assert "[b]" in result.stdout

    def test_show_cognitive_shows_source_agent(self, tmp_path):
        run_id = "cog-run"
        events = [
            _event(
                run_id,
                "cognitive_primitive",
                source_agent="planner",
                extra_metadata={
                    "primitive": "recall",
                    "args": {"turn": 3},
                    "result": {"found": True, "messages": [{}, {}], "note": None},
                },
            ),
        ]
        _write_trace(tmp_path, run_id, events)

        result = runner.invoke(
            app,
            [
                "trace", "show", run_id, "--dir", str(tmp_path),
                "--cognitive",
            ],
        )

        assert result.exit_code == 0, result.stdout
        # agent tag + primitive details both present
        assert "[planner]" in result.stdout
        assert "recall" in result.stdout
        assert "turn=3" in result.stdout
