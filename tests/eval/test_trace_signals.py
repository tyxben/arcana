"""Tests for trace-derived eval signals (finding F5, slices 1-2)."""

from __future__ import annotations

from arcana.contracts.eval import (
    EvalReport,
    EvalResult,
    GateConfig,
    RegressionResult,
    TraceSignals,
)
from arcana.contracts.trace import EventType, ToolCallRecord, TraceEvent
from arcana.eval.signals import extract_signals, merge_signals

# ---------------------------------------------------------------------------
# Contracts backward-compat (slice 1)
# ---------------------------------------------------------------------------


class TestContractsBackwardCompat:
    def test_old_style_construction_still_validates(self):
        r = EvalResult(
            case_id="c", passed=True, actual_status="completed",
            steps=1, tokens_used=10, cost_usd=0.01, duration_ms=5,
        )
        assert r.signals is None
        assert r.golden_status == "skip"
        # default gate == legacy fields untouched, new gates off
        g = GateConfig()
        assert g.golden_replay == "off"
        assert g.forbid_provider_degradation is False
        assert g.max_permission_denials is None
        rr = RegressionResult(passed=True, current_pass_rate=1.0)
        assert rr.signal_violations == []
        assert rr.warnings == []
        EvalReport(suite_name="s", total=0, passed=0, failed=0, pass_rate=0.0)

    def test_signals_digest_deterministic_and_self_excluding(self):
        a = TraceSignals(permission_denials=2, tool_calls=5).with_digest()
        b = TraceSignals(permission_denials=2, tool_calls=5).with_digest()
        assert a.signals_digest == b.signals_digest != ""
        # tampering with the digest field does not change the recomputed digest
        tampered = a.model_copy(update={"signals_digest": "tampered"})
        assert tampered.compute_digest() == a.signals_digest
        # a different vector hashes differently
        c = TraceSignals(permission_denials=3).with_digest()
        assert c.signals_digest != a.signals_digest


# ---------------------------------------------------------------------------
# extract_signals (slice 2)
# ---------------------------------------------------------------------------


def _ev(event_type, *, metadata=None, tool_call=None) -> TraceEvent:
    return TraceEvent(
        run_id="r",
        event_type=event_type,
        metadata=metadata or {},
        tool_call=tool_call,
    )


class TestExtractSignals:
    def test_full_mixed_trace(self):
        events = [
            _ev(EventType.LLM_CALL),
            # A genuine executed write that errored (timeout) — counted as a
            # write + a tool error.
            _ev(
                EventType.TOOL_CALL,
                metadata={"provenance": {"origin": "mcp", "server_name": "fs"}},
                tool_call=ToolCallRecord(
                    name="writer", args_digest="d", side_effect="write",
                    error="timed out", error_category="timeout",
                ),
            ),
            # A permission-denied write — an AUTHORITY signal, NOT a tool error
            # and NOT a write side effect (it never executed).
            _ev(
                EventType.TOOL_CALL,
                metadata={"permission_decision": {"action": "deny"}},
                tool_call=ToolCallRecord(
                    name="danger", args_digest="d", side_effect="write",
                    error="UNAUTHORIZED", error_category="permission",
                ),
            ),
            _ev(EventType.GUARDRAIL_DECISION, metadata={"action": "block"}),
            _ev(EventType.CAPABILITY_ADMISSION, metadata={"decision": "downgraded"}),
            _ev(
                EventType.CONTEXT_DECISION,
                metadata={
                    "context_decision": {
                        "messages_in": 10,
                        "messages_out": 6,
                        "compressed_count": 2,
                        "decisions": [
                            {"outcome": "dropped", "fidelity": "L3"},
                            {"outcome": "compressed", "fidelity": "L2"},
                        ],
                    }
                },
            ),
        ]
        s = extract_signals(events)

        assert s.trace_available is True
        assert s.llm_calls == 1
        assert s.tool_calls == 2
        assert s.imported_capability_calls == 1
        # Orthogonality: the denied write is an authority signal only.
        assert s.permission_denials == 1
        assert s.write_tool_calls == 1  # only the executed write
        assert s.tool_error_categories == {"timeout": 1}  # denial NOT bucketed here
        assert s.guardrail_blocks == 1
        assert s.capability_downgrades == 1
        assert s.context_messages_in == 10
        assert s.context_messages_dropped == 1
        assert s.context_compressed_count == 2
        assert abs(s.context_loss_ratio - 0.1) < 1e-9
        assert s.context_fidelity_min == "L3"  # worst level reached
        assert s.signals_digest != ""

    def test_empty_trace_is_explicitly_unavailable(self):
        s = extract_signals([])
        assert s.trace_available is False
        assert s.permission_denials == 0
        assert s.signals_digest != ""  # still digested

    def test_permission_ask_counted_separately(self):
        s = extract_signals(
            [_ev(EventType.TOOL_CALL, metadata={"permission_decision": {"action": "ask"}},
                 tool_call=ToolCallRecord(name="t", args_digest="d"))]
        )
        assert s.permission_asks == 1
        assert s.permission_denials == 0

    def test_error_category_used_when_present(self):
        # Slice 3: a TOOL_CALL whose record carries error_category buckets there,
        # not under the legacy "unexpected" fallback.
        s = extract_signals(
            [_ev(EventType.TOOL_CALL, tool_call=ToolCallRecord(
                name="t", args_digest="d", error="boom", error_category="timeout"))]
        )
        assert s.tool_error_categories == {"timeout": 1}

    def test_provider_degraded_from_llm_call_marker(self):
        # Slice 4: LLM_CALL carrying degraded_capabilities sets the signal.
        s = extract_signals(
            [_ev(EventType.LLM_CALL, metadata={
                "provider": "x", "used_text_tools": True,
                "degraded_capabilities": ["tool_calls"]})]
        )
        assert s.provider_degraded is True
        assert s.degraded_capabilities == ["tool_calls"]

    def test_guardrail_warn_vs_block(self):
        s = extract_signals(
            [
                _ev(EventType.GUARDRAIL_DECISION, metadata={"action": "warn"}),
                _ev(EventType.GUARDRAIL_DECISION, metadata={"action": "redact"}),
                _ev(EventType.GUARDRAIL_DECISION, metadata={"action": "require_approval"}),
            ]
        )
        assert s.guardrail_warns == 1
        assert s.guardrail_blocks == 2  # redact + require_approval count as blocks


# ---------------------------------------------------------------------------
# merge_signals
# ---------------------------------------------------------------------------


class TestMergeSignals:
    def test_merge_sums_and_unions(self):
        a = TraceSignals(
            permission_denials=1, tool_error_categories={"timeout": 1},
            provider_degraded=True, degraded_capabilities=["tool_calls"],
            context_messages_in=10, context_messages_dropped=1,
            context_fidelity_min="L1", trace_available=True,
        )
        b = TraceSignals(
            permission_denials=2, tool_error_categories={"timeout": 1, "logic": 3},
            degraded_capabilities=["tool_calls"], context_messages_in=20,
            context_messages_dropped=4, context_fidelity_min="L3",
            trace_available=True,
        )
        m = merge_signals([a, b])

        assert m.permission_denials == 3
        assert m.tool_error_categories == {"timeout": 2, "logic": 3}
        assert m.provider_degraded is True
        assert m.degraded_capabilities == ["tool_calls"]  # unioned, no dup
        assert m.context_messages_in == 30
        assert m.context_messages_dropped == 5
        assert abs(m.context_loss_ratio - 5 / 30) < 1e-9
        assert m.context_fidelity_min == "L3"  # worst across cases
        assert m.trace_available is True

    def test_merge_empty(self):
        m = merge_signals([])
        assert m.trace_available is False
        assert m.permission_denials == 0
