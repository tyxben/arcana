"""Tests for Trace Web UI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arcana.trace.web import create_app


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    """Create a temp trace directory with sample data."""
    traces = tmp_path / "traces"
    traces.mkdir()
    return traces


@pytest.fixture()
def sample_trace(trace_dir: Path) -> str:
    """Write a sample trace file and return the run ID."""
    run_id = "test-run-001"
    events = [
        {
            "run_id": run_id,
            "step_id": "step-1",
            "timestamp": "2025-01-15T10:00:00+00:00",
            "role": "system",
            "event_type": "llm_call",
            "model": "deepseek-chat",
            "budgets": {"tokens_used": 150, "cost_usd": 0.001},
        },
        {
            "run_id": run_id,
            "step_id": "step-2",
            "timestamp": "2025-01-15T10:00:01+00:00",
            "role": "executor",
            "event_type": "tool_call",
            "tool_call": {
                "name": "calculator",
                "args_digest": "abc123",
                "duration_ms": 45,
            },
            "budgets": {"tokens_used": 200, "cost_usd": 0.002},
        },
        {
            "run_id": run_id,
            "step_id": "step-3",
            "timestamp": "2025-01-15T10:00:02+00:00",
            "role": "system",
            "event_type": "llm_call",
            "model": "deepseek-chat",
            "stop_reason": "goal_reached",
            "budgets": {"tokens_used": 350, "cost_usd": 0.003},
        },
    ]
    trace_file = trace_dir / f"{run_id}.jsonl"
    with open(trace_file, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return run_id


@pytest.fixture()
def client(trace_dir: Path) -> TestClient:
    """Create a FastAPI test client pointing at the temp trace dir."""
    app = create_app(trace_dir=trace_dir)
    return TestClient(app)


class TestTraceWebIndex:
    def test_index_empty(self, client: TestClient, trace_dir: Path) -> None:
        """Index page with no traces shows empty message."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "No trace files found" in resp.text

    def test_index_no_dir(self, tmp_path: Path) -> None:
        """Index page with nonexistent dir shows empty message."""
        app = create_app(trace_dir=tmp_path / "nonexistent")
        c = TestClient(app)
        resp = c.get("/")
        assert resp.status_code == 200
        assert "No trace directory found" in resp.text

    def test_index_lists_traces(
        self, client: TestClient, sample_trace: str
    ) -> None:
        """Index page lists trace files with links."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert sample_trace in resp.text
        assert f"/trace/{sample_trace}" in resp.text
        # Should show summary stats
        assert "3" in resp.text  # 3 events
        assert "deepseek" not in resp.text or "LLM" in resp.text


class TestTraceWebDetail:
    def test_detail_missing(self, client: TestClient) -> None:
        """Detail page for missing trace shows 'not found'."""
        resp = client.get("/trace/nonexistent-run")
        assert resp.status_code == 200
        assert "not found" in resp.text.lower()

    def test_detail_shows_events(
        self, client: TestClient, sample_trace: str
    ) -> None:
        """Detail page shows event timeline."""
        resp = client.get(f"/trace/{sample_trace}")
        assert resp.status_code == 200
        # Should contain run ID
        assert sample_trace in resp.text
        # Should show event types
        assert "llm_call" in resp.text
        assert "tool_call" in resp.text
        # Should show model
        assert "deepseek-chat" in resp.text

    def test_detail_shows_summary_cards(
        self, client: TestClient, sample_trace: str
    ) -> None:
        """Detail page shows summary statistics."""
        resp = client.get(f"/trace/{sample_trace}")
        assert resp.status_code == 200
        # Events count
        assert "Events" in resp.text
        # LLM calls
        assert "LLM Calls" in resp.text
        # Tool calls
        assert "Tool Calls" in resp.text
        # Cost
        assert "$0.003" in resp.text or "0.003" in resp.text

    def test_detail_shows_tool_name(
        self, client: TestClient, sample_trace: str
    ) -> None:
        """Detail page shows tool call names."""
        resp = client.get(f"/trace/{sample_trace}")
        assert resp.status_code == 200
        assert "calculator" in resp.text

    def test_detail_shows_stop_reason(
        self, client: TestClient, sample_trace: str
    ) -> None:
        """Detail page shows stop reason."""
        resp = client.get(f"/trace/{sample_trace}")
        assert resp.status_code == 200
        assert "goal_reached" in resp.text

    def test_detail_back_link(
        self, client: TestClient, sample_trace: str
    ) -> None:
        """Detail page has a back link to index."""
        resp = client.get(f"/trace/{sample_trace}")
        assert resp.status_code == 200
        assert 'href="/"' in resp.text


class TestTraceWebMultipleRuns:
    def test_multiple_traces(self, client: TestClient, trace_dir: Path) -> None:
        """Index page handles multiple trace files."""
        for i in range(3):
            run_id = f"run-{i:03d}"
            trace_file = trace_dir / f"{run_id}.jsonl"
            event = {
                "run_id": run_id,
                "step_id": "s1",
                "timestamp": datetime.now(UTC).isoformat(),
                "role": "system",
                "event_type": "llm_call",
            }
            trace_file.write_text(json.dumps(event) + "\n")

        resp = client.get("/")
        assert resp.status_code == 200
        assert "run-000" in resp.text
        assert "run-001" in resp.text
        assert "run-002" in resp.text


class TestTraceWebErrorEvent:
    def test_error_event_display(self, client: TestClient, trace_dir: Path) -> None:
        """Error events are displayed with appropriate styling."""
        run_id = "error-run"
        events = [
            {
                "run_id": run_id,
                "step_id": "s1",
                "timestamp": datetime.now(UTC).isoformat(),
                "role": "system",
                "event_type": "error",
                "metadata": {"error": "Something went wrong"},
            },
        ]
        trace_file = trace_dir / f"{run_id}.jsonl"
        trace_file.write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        resp = client.get(f"/trace/{run_id}")
        assert resp.status_code == 200
        assert "error" in resp.text.lower()
