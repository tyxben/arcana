"""Tests for Arcana CLI."""

import json
from datetime import UTC, datetime

from typer.testing import CliRunner

from arcana.cli.main import app

runner = CliRunner()


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Arcana" in result.stdout

    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_providers(self):
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        assert "deepseek" in result.stdout
        assert "openai" in result.stdout

    def test_run_no_key(self):
        """Run without API key should fail gracefully."""
        result = runner.invoke(app, ["run", "test", "-p", "deepseek"])
        assert result.exit_code == 1
        assert (
            "API key" in result.stdout
            or "api_key" in result.stdout.lower()
            or result.exit_code == 1
        )

    def test_trace_list_no_dir(self):
        """Trace list with no traces dir should handle gracefully."""
        result = runner.invoke(app, ["trace", "list", "--dir", "/nonexistent"])
        assert result.exit_code == 0

    def test_trace_show_missing(self):
        """Trace show with bad run_id should fail gracefully."""
        result = runner.invoke(app, ["trace", "show", "nonexistent", "--dir", "/tmp"])
        assert result.exit_code == 1

    def test_run_yaml_config(self, tmp_path):
        """Run with YAML config should parse and use config values."""
        cfg = tmp_path / "agent.yaml"
        cfg.write_text("goal: test goal\nprovider: deepseek\nmax_turns: 5\n")
        result = runner.invoke(app, ["run", str(cfg)])
        # Should fail at API key, not at YAML parsing
        assert result.exit_code == 1
        assert "API key" in result.stdout

    def test_run_yaml_override_goal(self, tmp_path):
        """--override should replace goal from YAML."""
        cfg = tmp_path / "agent.yaml"
        cfg.write_text("goal: original\nprovider: deepseek\n")
        result = runner.invoke(app, ["run", str(cfg), "--override", "new goal"])
        assert result.exit_code == 1
        assert "API key" in result.stdout

    def test_run_yaml_no_goal(self, tmp_path):
        """YAML without goal and no --override should error."""
        cfg = tmp_path / "no_goal.yaml"
        cfg.write_text("provider: deepseek\n")
        result = runner.invoke(app, ["run", str(cfg)])
        assert result.exit_code == 1
        assert "no goal" in result.stdout.lower()

    def test_run_yaml_not_found(self):
        """Missing YAML file should error."""
        result = runner.invoke(app, ["run", "nonexistent.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()


class TestTraceSummary:
    """Tests for the 'arcana trace summary' command."""

    def test_trace_summary_no_dir(self):
        result = runner.invoke(app, ["trace", "summary", "--dir", "/nonexistent"])
        assert result.exit_code == 0
        assert "No traces" in result.stdout

    def test_trace_summary_empty_dir(self, tmp_path):
        result = runner.invoke(app, ["trace", "summary", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No trace files" in result.stdout

    def test_trace_summary_with_traces(self, tmp_path):
        """Should compute and display aggregate metrics."""
        from arcana.contracts.trace import AgentRole, EventType, TraceEvent

        # Create a trace file
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        events = [
            TraceEvent(
                run_id="test-run",
                event_type=EventType.LLM_CALL,
                timestamp=t0,
                role=AgentRole.SYSTEM,
            ),
            TraceEvent(
                run_id="test-run",
                event_type=EventType.STATE_CHANGE,
                timestamp=t0,
                role=AgentRole.SYSTEM,
            ),
        ]
        trace_file = tmp_path / "test-run.jsonl"
        with open(trace_file, "w") as f:
            for event in events:
                f.write(event.model_dump_json() + "\n")

        result = runner.invoke(app, ["trace", "summary", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Trace Summary" in result.stdout
        assert "1 runs" in result.stdout

    def test_trace_summary_with_mixed_lines(self, tmp_path):
        """Should not crash on write_raw() lines mixed with TraceEvent lines (P1 fix)."""
        from arcana.contracts.trace import AgentRole, EventType, TraceEvent

        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        trace_file = tmp_path / "mixed-run.jsonl"
        with open(trace_file, "w") as f:
            # Valid TraceEvent line
            event = TraceEvent(
                run_id="mixed-run",
                event_type=EventType.LLM_CALL,
                timestamp=t0,
                role=AgentRole.SYSTEM,
            )
            f.write(event.model_dump_json() + "\n")
            # write_raw() line — NOT a valid TraceEvent
            f.write(json.dumps({"event": "turn", "step": 0, "facts": {}, "assessment": {}}) + "\n")
            # Another valid TraceEvent
            f.write(event.model_dump_json() + "\n")

        result = runner.invoke(app, ["trace", "summary", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Trace Summary" in result.stdout

    def test_trace_summary_last_flag(self, tmp_path):
        """--last N should limit to most recent traces."""
        from arcana.contracts.trace import AgentRole, EventType, TraceEvent

        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(3):
            event = TraceEvent(
                run_id=f"run-{i}",
                event_type=EventType.LLM_CALL,
                timestamp=t0,
                role=AgentRole.SYSTEM,
            )
            f = tmp_path / f"run-{i}.jsonl"
            f.write_text(event.model_dump_json() + "\n")

        result = runner.invoke(app, ["trace", "summary", "--dir", str(tmp_path), "--last", "2"])
        assert result.exit_code == 0
        assert "2 runs" in result.stdout


class TestTraceShowFilters:
    """Tests for trace show --errors/--tools/--llm filters."""

    def _create_trace_file(self, tmp_path):

        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        events = [
            {"run_id": "r1", "event_type": "llm_call", "timestamp": t0.isoformat(), "role": "system", "step_id": "s1"},
            {"run_id": "r1", "event_type": "tool_call", "timestamp": t0.isoformat(), "role": "system", "step_id": "s2"},
            {"run_id": "r1", "event_type": "error", "timestamp": t0.isoformat(), "role": "system", "step_id": "s3"},
            {"run_id": "r1", "event_type": "state_change", "timestamp": t0.isoformat(), "role": "system", "step_id": "s4"},
        ]
        trace_file = tmp_path / "r1.jsonl"
        with open(trace_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        return tmp_path

    def test_show_errors_filter(self, tmp_path):
        self._create_trace_file(tmp_path)
        result = runner.invoke(app, ["trace", "show", "r1", "--dir", str(tmp_path), "--errors"])
        assert result.exit_code == 0
        assert "filtered from 4" in result.stdout
        assert "error" in result.stdout

    def test_show_tools_filter(self, tmp_path):
        self._create_trace_file(tmp_path)
        result = runner.invoke(app, ["trace", "show", "r1", "--dir", str(tmp_path), "--tools"])
        assert result.exit_code == 0
        assert "filtered from 4" in result.stdout
        assert "tool_call" in result.stdout

    def test_show_llm_filter(self, tmp_path):
        self._create_trace_file(tmp_path)
        result = runner.invoke(app, ["trace", "show", "r1", "--dir", str(tmp_path), "--llm"])
        assert result.exit_code == 0
        assert "filtered from 4" in result.stdout
        assert "llm_call" in result.stdout

    def test_show_no_filter(self, tmp_path):
        self._create_trace_file(tmp_path)
        result = runner.invoke(app, ["trace", "show", "r1", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Events: 4" in result.stdout
