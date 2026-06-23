"""Golden-trace recording + asymmetric replay diff (finding F5, slice 6).

A golden trace is a committed reference run (redacted skeleton + signal
vector). Replaying a later run against it detects *behavioural* regressions
that pass-rate misses. The diff is deliberately ASYMMETRIC: a run that does
fewer denials / fewer errors / less context loss is an *improvement*, never a
regression — so the gate never punishes a genuinely safer change. Only moves
in the unsafe direction (a new write/permission/guardrail/degradation event, a
boundary counter going up, a new tool-error category, worse fidelity) count.

Goldens are recorded only by an explicit op (never as a side effect of a gated
run, which would let a bad change overwrite its own baseline) and are stored
as committed JSON so relaxing one is a reviewable git diff.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arcana.contracts.eval import GoldenDiff, GoldenTrace, TraceSignals
from arcana.contracts.trace import EventType
from arcana.eval.signals import extract_signals
from arcana.utils.hashing import canonical_hash

if TYPE_CHECKING:
    from arcana.contracts.eval import EvalCase
    from arcana.contracts.trace import TraceEvent

_FIDELITY_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def build_skeleton(events: list[TraceEvent]) -> list[dict[str, Any]]:
    """Redacted structural projection of a run — no prose, no args, no PII.

    Only boundary/structural fields are kept, so a prompt-wording change does
    not show up as a regression, but a behavioural change (a new write tool, a
    guardrail flip, a degradation) does.
    """
    skeleton: list[dict[str, Any]] = []
    for event in events:
        meta = event.metadata or {}
        proj: dict[str, Any] = {"event_type": event.event_type.value}

        if event.event_type == EventType.TOOL_CALL and event.tool_call is not None:
            proj["tool_name"] = event.tool_call.name
            proj["side_effect"] = event.tool_call.side_effect
            proj["has_error"] = event.tool_call.error is not None
            if event.tool_call.error_category:
                proj["error_category"] = event.tool_call.error_category

        permission = meta.get("permission_decision")
        if isinstance(permission, dict) and permission.get("action"):
            proj["permission_action"] = permission["action"]
        if meta.get("provenance"):
            proj["imported"] = True

        if event.event_type == EventType.GUARDRAIL_DECISION:
            proj["guardrail_action"] = meta.get("action")
            proj["guardrail_boundary"] = meta.get("boundary")
        if event.event_type == EventType.CAPABILITY_ADMISSION:
            proj["capability_decision"] = meta.get("decision")
        if meta.get("degraded_capabilities"):
            proj["degraded"] = sorted(meta["degraded_capabilities"])
        if event.stop_reason is not None:
            proj["stop_reason"] = event.stop_reason.value

        skeleton.append({k: v for k, v in proj.items() if v is not None})
    # Order-canonicalize so nondeterministic parallel-tool event order does not
    # produce a spurious structural change. The signal vector carries the
    # order-independent boundary counts; the skeleton is a structural multiset
    # fingerprint.
    skeleton.sort(key=canonical_hash)
    return skeleton


def build_golden(
    case: EvalCase,
    events: list[TraceEvent],
    *,
    suite_name: str = "default",
    arcana_version: str | None = None,
    recorded_at: str = "",
) -> GoldenTrace:
    """Construct a GoldenTrace from a run's case + trace events."""
    skeleton = build_skeleton(events)
    return GoldenTrace(
        case_id=case.id,
        suite_name=suite_name,
        recorded_at=recorded_at,
        arcana_version=arcana_version,
        goal_digest=canonical_hash(case.goal),
        signals=extract_signals(events),
        event_skeleton=skeleton,
        skeleton_digest=canonical_hash(skeleton),
    )


def _fidelity_rank(level: str | None) -> int:
    # None (no context decision) and L0 (original, kept verbatim) both mean "no
    # loss" — neither is a regression against the other.
    return _FIDELITY_RANK.get(level or "L0", 0)


def _worse_fidelity(current: str | None, golden: str | None) -> bool:
    return _fidelity_rank(current) > _fidelity_rank(golden)


def _signal_regressions(golden: TraceSignals, current: TraceSignals) -> list[str]:
    """Unsafe-direction signal moves only (asymmetric)."""
    out: list[str] = []
    if current.permission_denials > golden.permission_denials:
        out.append(
            f"permission_denials {golden.permission_denials}->{current.permission_denials}"
        )
    if current.guardrail_blocks > golden.guardrail_blocks:
        out.append(
            f"guardrail_blocks {golden.guardrail_blocks}->{current.guardrail_blocks}"
        )
    if current.provider_degraded and not golden.provider_degraded:
        out.append(f"provider degraded (was not): {sorted(current.degraded_capabilities)}")
    if current.capability_downgrades > golden.capability_downgrades:
        out.append(
            f"capability_downgrades {golden.capability_downgrades}->{current.capability_downgrades}"
        )
    if current.write_tool_calls > golden.write_tool_calls:
        out.append(
            f"write_tool_calls {golden.write_tool_calls}->{current.write_tool_calls}"
        )
    if current.imported_capability_calls > golden.imported_capability_calls:
        out.append(
            f"imported_capability_calls {golden.imported_capability_calls}->"
            f"{current.imported_capability_calls}"
        )
    if current.context_loss_ratio > golden.context_loss_ratio + 1e-9:
        out.append(
            f"context_loss_ratio {golden.context_loss_ratio:.3f}->"
            f"{current.context_loss_ratio:.3f}"
        )
    if _worse_fidelity(current.context_fidelity_min, golden.context_fidelity_min):
        out.append(
            f"context_fidelity_min {golden.context_fidelity_min}->"
            f"{current.context_fidelity_min}"
        )
    new_categories = set(current.tool_error_categories) - set(golden.tool_error_categories)
    if new_categories:
        out.append(f"new tool_error_categories: {sorted(new_categories)}")
    return out


def replay_diff(
    golden: GoldenTrace,
    events: list[TraceEvent],
    *,
    case: EvalCase | None = None,
) -> GoldenDiff:
    """Diff a run's events against a golden trace (asymmetric)."""
    current_signals = extract_signals(events)
    current_skeleton = build_skeleton(events)
    signal_regressions = _signal_regressions(golden.signals, current_signals)

    structural_changes: list[str] = []
    if canonical_hash(current_skeleton) != golden.skeleton_digest:
        structural_changes.append("structural skeleton changed")

    # The goal changed underneath the golden — the reference is stale, treat as
    # 'new' rather than asserting a regression against an unrelated baseline.
    # Still surface any signal regressions for information (a goal edit must not
    # silently mask a boundary weakening in the same run).
    if case is not None and canonical_hash(case.goal) != golden.goal_digest:
        return GoldenDiff(
            case_id=golden.case_id,
            golden_status="new",
            structural_changes=["goal changed since golden was recorded"],
            signal_regressions=signal_regressions,
            is_regression=False,
        )

    is_regression = bool(signal_regressions)
    if is_regression:
        status = "regressed"
    elif structural_changes or current_signals.signals_digest != golden.signals.signals_digest:
        # Changed but only in safe directions → an improvement, never a fail.
        status = "improved"
    else:
        status = "match"

    return GoldenDiff(
        case_id=golden.case_id,
        golden_status=status,
        structural_changes=structural_changes,
        signal_regressions=signal_regressions,
        is_regression=is_regression,
    )


class GoldenStore:
    """Loads/records golden traces as committed JSON files.

    Layout: ``{golden_dir}/{suite}/{case_id}.json``. New goldens (no existing
    reference) land in ``_candidates/`` and are never trusted live until
    promoted. Overwriting an existing golden requires ``force=True`` — relaxing
    a boundary is therefore a deliberate, reviewable git diff.
    """

    def __init__(self, golden_dir: str | Path) -> None:
        self.dir = Path(golden_dir)

    def _path(self, suite: str, case_id: str) -> Path:
        return self.dir / suite / f"{case_id}.json"

    def load(self, suite: str, case_id: str) -> GoldenTrace | None:
        """Load a golden, verifying its digests.

        A committed golden that was hand-edited without re-recording (its skeleton
        or signal vector changed but the stored digest did not) is a forgery that
        would silently relax a boundary — so the tamper-evidence is *enforced*
        here, not merely stored. A legitimate relaxation re-records (updating the
        digests) and passes. Raises ``ValueError`` on a digest mismatch.
        """
        path = self._path(suite, case_id)
        if not path.exists():
            return None
        golden = GoldenTrace.model_validate_json(path.read_text())
        if golden.signals.compute_digest() != golden.signals.signals_digest:
            raise ValueError(
                f"golden signals digest mismatch for {suite}/{case_id} "
                f"— hand-edited without re-recording (tampered)?"
            )
        if canonical_hash(golden.event_skeleton) != golden.skeleton_digest:
            raise ValueError(
                f"golden skeleton digest mismatch for {suite}/{case_id} "
                f"— hand-edited without re-recording (tampered)?"
            )
        return golden

    def record(self, golden: GoldenTrace, *, force: bool = False) -> Path:
        """Write a golden.

        Recording the trusted, live golden requires ``force=True`` — the
        deliberate, explicit op (``--record-golden`` / ``record_golden=True``).
        Without force the golden is written to ``_candidates/`` instead, so a
        bad or automated run can never silently establish itself as the trusted
        baseline. Relaxing a live golden is therefore always a forced,
        reviewable git diff.
        """
        if not force:
            candidate = (
                self.dir / "_candidates" / golden.suite_name / f"{golden.case_id}.json"
            )
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(
                json.dumps(golden.model_dump(), indent=2, sort_keys=True)
            )
            return candidate
        path = self._path(golden.suite_name, golden.case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(golden.model_dump(), indent=2, sort_keys=True))
        return path
