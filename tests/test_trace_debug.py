"""Tests for v0.8.1 trace debugging improvements.

Covers:
- ``RuntimeConfig.dev_mode`` flag behavior (single-switch dev experience)
- ``TraceEvent.parent_step_id`` field (causal linking)
- ``TraceReader.collect_turn`` (bundle events for one turn)
- ``arcana trace explain`` CLI command
- ``arcana trace flow`` CLI command
- ``arcana trace show --errors --explain`` auto-unfold
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from arcana.cli.main import app
from arcana.contracts.tool import ToolCall
from arcana.contracts.trace import (
    EventType,
    ToolCallRecord,
    TraceEvent,
)
from arcana.runtime_core import RuntimeConfig
from arcana.trace.reader import TraceReader
from arcana.trace.writer import TraceWriter

runner = CliRunner()


# ---------------------------------------------------------------------------
# RuntimeConfig.dev_mode
# ---------------------------------------------------------------------------


class TestDevMode:
    def test_default_off(self) -> None:
        c = RuntimeConfig()
        assert c.dev_mode is False
        assert c.trace_include_prompt_snapshots is False

    def test_dev_mode_implies_snapshots(self) -> None:
        c = RuntimeConfig(dev_mode=True)
        assert c.dev_mode is True
        assert c.trace_include_prompt_snapshots is True

    def test_explicit_snapshot_preserved_when_dev_off(self) -> None:
        c = RuntimeConfig(trace_include_prompt_snapshots=True)
        assert c.trace_include_prompt_snapshots is True
        assert c.dev_mode is False

    def test_both_true_stays_true(self) -> None:
        c = RuntimeConfig(dev_mode=True, trace_include_prompt_snapshots=True)
        assert c.trace_include_prompt_snapshots is True


# ---------------------------------------------------------------------------
# TraceEvent.parent_step_id schema + contract
# ---------------------------------------------------------------------------


class TestParentStepIdField:
    def test_default_none(self) -> None:
        ev = TraceEvent(run_id="r1", event_type=EventType.TURN)
        assert ev.parent_step_id is None

    def test_accepts_value(self) -> None:
        ev = TraceEvent(
            run_id="r1",
            event_type=EventType.TOOL_CALL,
            parent_step_id="parent-xyz",
        )
        assert ev.parent_step_id == "parent-xyz"

    def test_roundtrip_via_json(self, tmp_path) -> None:
        w = TraceWriter(trace_dir=tmp_path)
        w.write(TraceEvent(
            run_id="r1",
            event_type=EventType.TURN,
            step_id="T1",
            parent_step_id="T0",
            metadata={"step": 1},
        ))
        r = TraceReader(trace_dir=tmp_path)
        events = r.read_events("r1")
        assert events[0].parent_step_id == "T0"

    def test_legacy_event_without_field(self, tmp_path) -> None:
        # Simulate a legacy trace file written before parent_step_id existed
        run_id = "legacy"
        (tmp_path / f"{run_id}.jsonl").write_text(
            json.dumps({
                "run_id": run_id,
                "event_type": "turn",
                "step_id": "legacy-step",
                "metadata": {"step": 1},
            }) + "\n"
        )
        r = TraceReader(trace_dir=tmp_path)
        events = r.read_events(run_id)
        assert len(events) == 1
        assert events[0].parent_step_id is None


class TestToolCallParentStepId:
    def test_default_none(self) -> None:
        tc = ToolCall(id="tc1", name="search", arguments={})
        assert tc.parent_step_id is None

    def test_accepts_value(self) -> None:
        tc = ToolCall(
            id="tc1", name="search", arguments={}, parent_step_id="turn-step-7",
        )
        assert tc.parent_step_id == "turn-step-7"


# ---------------------------------------------------------------------------
# TraceReader.collect_turn
# ---------------------------------------------------------------------------


def _seed_two_turn_run(trace_dir, run_id: str = "r1") -> None:
    """Helper: write a 2-turn run with tool call + error on turn 1."""
    w = TraceWriter(trace_dir=trace_dir)
    # Turn 1: LLM calls `search`, tool errors out
    w.write(TraceEvent(
        run_id=run_id, event_type=EventType.TURN, step_id="T1",
        model="test-model",
        metadata={
            "step": 1,
            "facts": {
                "assistant_text": "searching",
                "tool_calls": [{"name": "search", "arguments": "{}"}],
                "thinking": "need to look up X",
            },
            "assessment": {"completed": False, "confidence": 0.6},
        },
    ))
    w.write(TraceEvent(
        run_id=run_id, event_type=EventType.CONTEXT_DECISION,
        parent_step_id="T1",
        metadata={
            "turn": 1,
            "explanation": "all fit within budget",
            "context_report": {
                "messages_in": 3, "messages_out": 3, "compressed_count": 0,
            },
            "context_decision": {"decisions": []},
        },
    ))
    w.write(TraceEvent(
        run_id=run_id, event_type=EventType.TOOL_CALL,
        parent_step_id="T1",
        tool_call=ToolCallRecord(
            name="search", args_digest="abc",
            error="TimeoutError", duration_ms=1500, side_effect="read",
        ),
    ))
    w.write(TraceEvent(
        run_id=run_id, event_type=EventType.ERROR,
        parent_step_id="T1",
        metadata={"message": "Tool search timed out"},
    ))
    # Turn 2: LLM recovers, completes
    w.write(TraceEvent(
        run_id=run_id, event_type=EventType.TURN, step_id="T2",
        parent_step_id="T1", model="test-model",
        metadata={
            "step": 2,
            "facts": {"assistant_text": "here's my best guess", "tool_calls": []},
            "assessment": {"completed": True, "confidence": 0.8,
                           "completion_reason": "answered from memory"},
        },
    ))


class TestCollectTurn:
    def test_bundle_turn_1(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        r = TraceReader(trace_dir=tmp_path)
        b = r.collect_turn("r1", 1)
        assert b["turn_event"] is not None
        assert b["turn_event"].step_id == "T1"
        assert b["context_decision"] is not None
        assert len(b["tool_calls"]) == 1
        assert b["tool_calls"][0].tool_call.name == "search"
        assert len(b["errors"]) == 1

    def test_bundle_turn_2_has_no_children(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        r = TraceReader(trace_dir=tmp_path)
        b = r.collect_turn("r1", 2)
        assert b["turn_event"].parent_step_id == "T1"
        assert len(b["tool_calls"]) == 0

    def test_missing_turn_returns_empty(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        r = TraceReader(trace_dir=tmp_path)
        b = r.collect_turn("r1", 99)
        assert b["turn_event"] is None
        assert b["tool_calls"] == []


# ---------------------------------------------------------------------------
# CLI: arcana trace explain
# ---------------------------------------------------------------------------


class TestTraceExplainCLI:
    def test_explain_turn_1_renders(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app,
            ["trace", "explain", "r1", "--dir", str(tmp_path), "--turn", "1"],
        )
        assert result.exit_code == 0
        out = result.stdout
        assert "Turn 1" in out
        assert "r1" in out
        # Inputs section
        assert "Inputs" in out
        assert "messages: 3" in out
        # LLM output section
        assert "thinking:" in out
        assert "searching" in out
        assert "search" in out
        # Tool results
        assert "Tool results" in out
        assert "TimeoutError" in out
        # Runtime verdict
        assert "Runtime verdict" in out
        assert "completed: False" in out
        # Errors list
        assert "Errors" in out
        assert "timed out" in out

    def test_explain_turn_2_shows_completion(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app,
            ["trace", "explain", "r1", "--dir", str(tmp_path), "--turn", "2"],
        )
        assert result.exit_code == 0
        assert "Turn 2" in result.stdout
        assert "completed: True" in result.stdout
        assert "answered from memory" in result.stdout

    def test_explain_missing_turn(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app,
            ["trace", "explain", "r1", "--dir", str(tmp_path), "--turn", "99"],
        )
        assert result.exit_code == 1
        assert "No TURN event" in result.stdout

    def test_explain_requires_turn(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app, ["trace", "explain", "r1", "--dir", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "--turn N is required" in result.stdout

    def test_explain_missing_run(self, tmp_path) -> None:
        result = runner.invoke(
            app,
            ["trace", "explain", "nope", "--dir", str(tmp_path), "--turn", "1"],
        )
        assert result.exit_code == 1
        assert "Trace not found" in result.stdout

    def test_explain_json_mode(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app,
            ["trace", "explain", "r1", "--dir", str(tmp_path),
             "--turn", "1", "--json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["turn"] == 1
        assert payload["turn_event"]["step_id"] == "T1"
        assert len(payload["tool_calls"]) == 1

    def test_explain_degrades_without_snapshot(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app,
            ["trace", "explain", "r1", "--dir", str(tmp_path), "--turn", "1"],
        )
        assert result.exit_code == 0
        # No prompt_snapshot was written → shows hint
        assert "dev_mode=True" in result.stdout


# ---------------------------------------------------------------------------
# CLI: arcana trace flow
# ---------------------------------------------------------------------------


class TestTraceFlowCLI:
    def test_flow_renders_turns_and_tools(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app, ["trace", "flow", "r1", "--dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        out = result.stdout
        assert "Flow" in out
        assert "Turn 1" in out
        assert "Turn 2" in out
        assert "completed" in out
        # Tool appears as child of turn 1
        assert "search" in out
        # Failed tool is visually marked
        assert "✗" in out

    def test_flow_stop_reason(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app, ["trace", "flow", "r1", "--dir", str(tmp_path)],
        )
        assert "→ stop: completed" in result.stdout

    def test_flow_json(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app, ["trace", "flow", "r1", "--dir", str(tmp_path), "--json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert len(payload["turns"]) == 2
        assert payload["turns"][0]["turn"] == 1
        assert len(payload["turns"][0]["tool_calls"]) == 1
        assert payload["stop"] == "completed"

    def test_flow_missing_run(self, tmp_path) -> None:
        result = runner.invoke(
            app, ["trace", "flow", "nope", "--dir", str(tmp_path)],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# CLI: arcana trace show --errors --explain
# ---------------------------------------------------------------------------


class TestShowErrorsExplain:
    def test_errors_explain_unfolds_turn(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app,
            ["trace", "show", "r1", "--dir", str(tmp_path),
             "--errors", "--explain"],
        )
        assert result.exit_code == 0
        out = result.stdout
        # Normal error listing
        assert "error" in out.lower()
        # Auto-unfold banner
        assert "Error at turn 1" in out
        # Explain view content rendered
        assert "Tool results" in out
        assert "TimeoutError" in out

    def test_errors_without_explain_does_not_unfold(self, tmp_path) -> None:
        _seed_two_turn_run(tmp_path)
        result = runner.invoke(
            app,
            ["trace", "show", "r1", "--dir", str(tmp_path), "--errors"],
        )
        assert result.exit_code == 0
        assert "Error at turn" not in result.stdout
        assert "Tool results" not in result.stdout

    def test_errors_explain_deduplicates_same_turn(self, tmp_path) -> None:
        # Two error events on the same turn → only one explain unfold
        w = TraceWriter(trace_dir=tmp_path)
        run_id = "dupe"
        w.write(TraceEvent(
            run_id=run_id, event_type=EventType.TURN, step_id="T1",
            metadata={"step": 1, "facts": {}, "assessment": {}},
        ))
        w.write(TraceEvent(
            run_id=run_id, event_type=EventType.ERROR,
            parent_step_id="T1", metadata={"message": "err1"},
        ))
        w.write(TraceEvent(
            run_id=run_id, event_type=EventType.ERROR,
            parent_step_id="T1", metadata={"message": "err2"},
        ))
        result = runner.invoke(
            app,
            ["trace", "show", run_id, "--dir", str(tmp_path),
             "--errors", "--explain"],
        )
        assert result.exit_code == 0
        # Banner appears exactly once
        assert result.stdout.count("Error at turn 1") == 1


# ---------------------------------------------------------------------------
# ConversationAgent emit-method wiring — verifies sibling events share a parent
# ---------------------------------------------------------------------------


class TestConversationAgentWiring:
    """Exercise the emit methods with a fake turn_step_id set.

    Proves that ``_trace_turn`` / ``_emit_context_decision_event`` /
    ``_emit_prompt_snapshot_event`` honor ``_current_turn_step_id``.
    """

    def _make_agent(self, tmp_path, *, snapshots: bool = False):
        from arcana.contracts.turn import TurnAssessment, TurnFacts
        from arcana.gateway.registry import ModelGatewayRegistry
        from arcana.runtime.conversation import ConversationAgent

        gw = ModelGatewayRegistry()
        writer = TraceWriter(trace_dir=tmp_path)
        agent = ConversationAgent(
            gateway=gw,
            trace_writer=writer,
            trace_include_prompt_snapshots=snapshots,
        )
        return agent, writer, TurnFacts, TurnAssessment

    def test_trace_turn_uses_current_turn_step_id(self, tmp_path) -> None:
        from arcana.contracts.state import AgentState

        agent, writer, TurnFacts, TurnAssessment = self._make_agent(tmp_path)
        agent._current_turn_step_id = "T1"
        agent._prev_turn_step_id = "T0"

        state = AgentState(run_id="r1", goal="g", max_steps=5)
        agent._trace_turn(
            TurnFacts(assistant_text="hi", tool_calls=[]),
            TurnAssessment(completed=False, failed=False),
            state,
        )

        events = TraceReader(trace_dir=tmp_path).read_events("r1")
        turn_events = [e for e in events if e.event_type == EventType.TURN]
        assert len(turn_events) == 1
        assert turn_events[0].step_id == "T1"
        assert turn_events[0].parent_step_id == "T0"

    def test_context_decision_parents_to_current_turn(self, tmp_path) -> None:
        from arcana.contracts.context import ContextDecision
        from arcana.contracts.state import AgentState

        agent, writer, *_ = self._make_agent(tmp_path)
        agent._current_turn_step_id = "T42"
        agent._context_builder.last_decision = ContextDecision(
            turn=1, strategy="tail_preserve",
            budget_total=1000, budget_used=100,
            messages_in=2, messages_out=2,
        )
        agent._context_builder.last_report = None
        state = AgentState(run_id="r2", goal="g", max_steps=5)
        agent._emit_context_decision_event(state)

        events = TraceReader(trace_dir=tmp_path).read_events("r2")
        ctx = [e for e in events if e.event_type == EventType.CONTEXT_DECISION]
        assert len(ctx) == 1
        assert ctx[0].parent_step_id == "T42"

    def test_cognitive_event_parents_to_current_turn(self, tmp_path) -> None:
        agent, writer, *_ = self._make_agent(tmp_path)
        agent._current_turn_step_id = "T7"
        agent._emit_cognitive_primitive_event(
            run_id="r3", primitive="pin",
            args={"label": "x"},
            result={"pinned": True, "pin_id": "p1"},
        )

        events = TraceReader(trace_dir=tmp_path).read_events("r3")
        cog = [e for e in events if e.event_type == EventType.COGNITIVE_PRIMITIVE]
        assert len(cog) == 1
        assert cog[0].parent_step_id == "T7"


# ---------------------------------------------------------------------------
# ToolGateway propagates ToolCall.parent_step_id into the TOOL_CALL event
# ---------------------------------------------------------------------------


def test_tool_gateway_propagates_parent_step_id(tmp_path) -> None:
    """_log_tool_call copies ToolCall.parent_step_id onto the TOOL_CALL event."""
    from arcana.contracts.tool import SideEffect, ToolResult, ToolSpec
    from arcana.contracts.trace import TraceContext
    from arcana.tool_gateway.gateway import ToolGateway
    from arcana.tool_gateway.registry import ToolRegistry

    writer = TraceWriter(trace_dir=tmp_path)
    gw = ToolGateway(registry=ToolRegistry(), trace_writer=writer)

    spec = ToolSpec(
        name="echo",
        description="echo",
        input_schema={"type": "object"},
        side_effect=SideEffect.READ,
    )
    tc = ToolCall(
        id="tc1", name="echo", arguments={"msg": "hi"},
        run_id="gw-run", step_id="tool-step-1",
        parent_step_id="TURN-STEP-AAA",
    )
    tr_result = ToolResult(
        tool_call_id="tc1", name="echo", success=True, output="hi",
        duration_ms=10,
    )
    ctx = TraceContext(run_id="gw-run")
    gw._log_tool_call(tc, tr_result, spec, ctx)

    events = TraceReader(trace_dir=tmp_path).read_events("gw-run")
    tool_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
    assert len(tool_events) == 1
    assert tool_events[0].parent_step_id == "TURN-STEP-AAA"
    assert tool_events[0].tool_call.name == "echo"
