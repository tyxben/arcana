"""Trace-derived eval signal extraction (finding F5).

``extract_signals`` turns a run's already-written trace events into a
:class:`~arcana.contracts.eval.TraceSignals` fitness vector. It is pure and
post-hoc — it never runs inside the agent turn loop (Principle 7 Corollary:
evals are evidence, not runtime governance). It is a sibling of
``MetricsCollector.summarize_run`` kept separate so the future EvidenceBundle
builder can call it over a sandbox run without importing the eval runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.eval import TraceSignals
from arcana.contracts.trace import EventType

if TYPE_CHECKING:
    from arcana.contracts.trace import TraceEvent

# Fidelity ladder: L0 = original (best) … L3 = dropped (worst). The "minimum
# fidelity" a run reached is the worst (highest-ranked) level any message hit.
_FIDELITY_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def extract_signals(events: list[TraceEvent]) -> TraceSignals:
    """Compute the trace-derived signal vector for one run.

    A single linear pass over the run's trace events. Deterministic. Returns a
    vector with ``trace_available`` reflecting whether there were any events
    (so an absent trace is never silently read as a clean run) and
    ``signals_digest`` filled in.
    """
    sig = TraceSignals(trace_available=bool(events))
    worst_fidelity_rank = -1

    for event in events:
        et = event.event_type
        meta = event.metadata or {}

        # Permission decisions ride on tool-call metadata today, but read off
        # any event for forward safety.
        permission = meta.get("permission_decision")
        if isinstance(permission, dict):
            action = permission.get("action")
            if action == "deny":
                sig.permission_denials += 1
            elif action == "ask":
                sig.permission_asks += 1

        if et == EventType.LLM_CALL:
            sig.llm_calls += 1
            degraded = meta.get("degraded_capabilities")
            if degraded:
                sig.provider_degraded = True
                for cap in degraded:
                    if cap not in sig.degraded_capabilities:
                        sig.degraded_capabilities.append(cap)

        elif et == EventType.TOOL_CALL:
            sig.tool_calls += 1
            if meta.get("provenance"):
                sig.imported_capability_calls += 1
            record = event.tool_call
            if record is not None:
                category = getattr(record, "error_category", None)
                # A pre-execution boundary refusal (permission deny/ask, or a
                # permission / confirmation-required error) is an AUTHORITY
                # signal, not a tool-execution failure. Count it only in the
                # authority dimensions (permission_denials, above; guardrail via
                # GUARDRAIL_DECISION) and keep it OUT of tool_error_categories
                # and write_tool_calls so the dimensions stay orthogonal and a
                # never-executed write is not counted as a write side effect.
                is_boundary_refusal = category in ("permission", "confirmation_required") or (
                    isinstance(permission, dict)
                    and permission.get("action") in ("deny", "ask")
                )
                if record.side_effect == "write" and not is_boundary_refusal:
                    sig.write_tool_calls += 1
                if record.error is not None and not is_boundary_refusal:
                    # error_category lands on the record in F5 slice 3; legacy
                    # traces bucket under "unexpected".
                    bucket = category or "unexpected"
                    sig.tool_error_categories[bucket] = (
                        sig.tool_error_categories.get(bucket, 0) + 1
                    )

        elif et == EventType.GUARDRAIL_DECISION:
            action = meta.get("action")
            if action in ("block", "redact", "require_approval"):
                sig.guardrail_blocks += 1
            elif action == "warn":
                sig.guardrail_warns += 1

        elif et == EventType.CAPABILITY_ADMISSION:
            if meta.get("decision") in ("downgraded", "refused"):
                sig.capability_downgrades += 1

        elif et == EventType.CONTEXT_DECISION:
            decision = meta.get("context_decision")
            if isinstance(decision, dict):
                sig.context_messages_in += int(decision.get("messages_in", 0) or 0)
                sig.context_messages_out += int(decision.get("messages_out", 0) or 0)
                sig.context_compressed_count += int(
                    decision.get("compressed_count", 0) or 0
                )
                for message_decision in decision.get("decisions", []) or []:
                    if message_decision.get("outcome") == "dropped":
                        sig.context_messages_dropped += 1
                    fidelity = message_decision.get("fidelity")
                    rank = _FIDELITY_RANK.get(fidelity, -1)
                    if rank > worst_fidelity_rank:
                        worst_fidelity_rank = rank
                        sig.context_fidelity_min = fidelity
            else:
                # Legacy flat keys (no nested decision dump).
                sig.context_messages_in += int(meta.get("messages_in", 0) or 0)
                sig.context_messages_out += int(meta.get("messages_out", 0) or 0)
                sig.context_compressed_count += int(
                    meta.get("compressed_count", 0) or 0
                )

    sig.context_loss_ratio = sig.context_messages_dropped / max(
        sig.context_messages_in, 1
    )
    return sig.with_digest()


def merge_signals(signals: list[TraceSignals]) -> TraceSignals:
    """Combine per-case signal vectors into a suite-level vector.

    Counters sum, category dicts merge (values summed), capability lists union,
    booleans OR, and ``context_fidelity_min`` takes the worst level seen.
    ``trace_available`` is True only if at least one case had a trace.
    """
    merged = TraceSignals()
    if not signals:
        return merged.with_digest()

    worst_fidelity_rank = -1
    for sig in signals:
        merged.permission_denials += sig.permission_denials
        merged.permission_asks += sig.permission_asks
        merged.guardrail_blocks += sig.guardrail_blocks
        merged.guardrail_warns += sig.guardrail_warns
        for category, count in sig.tool_error_categories.items():
            merged.tool_error_categories[category] = (
                merged.tool_error_categories.get(category, 0) + count
            )
        merged.provider_degraded = merged.provider_degraded or sig.provider_degraded
        for cap in sig.degraded_capabilities:
            if cap not in merged.degraded_capabilities:
                merged.degraded_capabilities.append(cap)
        merged.capability_downgrades += sig.capability_downgrades
        merged.write_tool_calls += sig.write_tool_calls
        merged.imported_capability_calls += sig.imported_capability_calls
        merged.context_messages_in += sig.context_messages_in
        merged.context_messages_out += sig.context_messages_out
        merged.context_messages_dropped += sig.context_messages_dropped
        merged.context_compressed_count += sig.context_compressed_count
        merged.llm_calls += sig.llm_calls
        merged.tool_calls += sig.tool_calls
        merged.trace_available = merged.trace_available or sig.trace_available
        rank = _FIDELITY_RANK.get(sig.context_fidelity_min or "", -1)
        if rank > worst_fidelity_rank:
            worst_fidelity_rank = rank
            merged.context_fidelity_min = sig.context_fidelity_min

    merged.context_loss_ratio = merged.context_messages_dropped / max(
        merged.context_messages_in, 1
    )
    return merged.with_digest()
