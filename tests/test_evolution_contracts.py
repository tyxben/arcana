"""Tests for the self-evolution contracts (Amendment 6 prereqs, slices 2-4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arcana.contracts.eval import EvalReport, TraceSignals
from arcana.contracts.evolution import (
    AcceptanceDecision,
    ApproverKind,
    AuthorityClass,
    EvalReportRef,
    EvidenceBundle,
    EvolutionProposal,
    EvolutionTarget,
    EvolutionTargetKind,
    MonitoringAnchor,
    PatchRef,
    PromotionRecord,
    RollbackPointer,
    SandboxVerificationRef,
    classify_authority,
)


def _signals(**kw) -> TraceSignals:
    return TraceSignals(trace_available=True, **kw).with_digest()


def _bundle(kind=EvolutionTargetKind.SKILL, signals=None) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="b1", target_kind=kind,
        trace_signals=signals if signals is not None else _signals(tool_calls=1),
    ).with_digest()


# ---------------------------------------------------------------------------
# classify_authority (slice 2)
# ---------------------------------------------------------------------------


class TestClassifyAuthority:
    @pytest.mark.parametrize("kind", [
        EvolutionTargetKind.PERMISSION_POLICY, EvolutionTargetKind.GUARDRAIL,
        EvolutionTargetKind.PROVIDER_CONFIG, EvolutionTargetKind.CREDENTIAL_CONFIG,
        EvolutionTargetKind.REMOTE_PROTOCOL_TRUST, EvolutionTargetKind.EXECUTION_BACKEND,
        EvolutionTargetKind.CORE_RUNTIME,
    ])
    def test_structural_trigger(self, kind):
        # Even with perfectly clean evidence, an authority surface is touching.
        assert classify_authority(kind, _signals()) == AuthorityClass.AUTHORITY_TOUCHING

    @pytest.mark.parametrize("field", [
        "permission_denials", "guardrail_blocks", "capability_downgrades",
        "write_tool_calls", "imported_capability_calls",
    ])
    def test_behavioral_trigger(self, field):
        # A low-authority target whose run crossed a boundary is laundering.
        assert classify_authority(
            EvolutionTargetKind.SKILL, _signals(**{field: 1})
        ) == AuthorityClass.AUTHORITY_TOUCHING

    def test_provider_degraded_triggers(self):
        assert classify_authority(
            EvolutionTargetKind.DOC, _signals(provider_degraded=True)
        ) == AuthorityClass.AUTHORITY_TOUCHING

    def test_clean_low_stays_low(self):
        assert classify_authority(
            EvolutionTargetKind.SKILL, _signals(tool_calls=3)
        ) == AuthorityClass.LOW


# ---------------------------------------------------------------------------
# EvidenceBundle (slice 2)
# ---------------------------------------------------------------------------


class TestEvidenceBundle:
    def test_digest_deterministic_and_self_excluding(self):
        a = _bundle()
        b = _bundle()
        assert a.bundle_digest == b.bundle_digest != ""
        tampered = a.model_copy(update={"bundle_digest": "x"})
        assert tampered.compute_digest() == a.bundle_digest

    def test_verify_clean_bundle(self):
        assert _bundle().verify_bundle() == []

    def test_rejects_fabricated_signals(self):
        b = _bundle()
        # hand-edit a signal without re-digesting the vector
        forged = b.model_copy(update={
            "trace_signals": b.trace_signals.model_copy(update={"permission_denials": 9})
        })
        assert any("trace_signals digest" in v for v in forged.verify_bundle())

    def test_rejects_empty_bundle(self):
        empty = EvidenceBundle(bundle_id="e", target_kind=EvolutionTargetKind.DOC).with_digest()
        assert empty.is_empty
        assert any("no evidence" in v for v in empty.verify_bundle())

    def test_rejects_mismatched_eval_report_digest(self):
        report = EvalReport(suite_name="s", total=1, passed=1, failed=0, pass_rate=1.0)
        b = EvidenceBundle(
            bundle_id="b", target_kind=EvolutionTargetKind.SKILL,
            trace_signals=_signals(tool_calls=1),
            eval_report_ref=EvalReportRef(report_digest="WRONGDIGEST"),
        ).with_digest()
        assert any("eval report digest" in v for v in b.verify_bundle(report=report))


# ---------------------------------------------------------------------------
# EvolutionProposal (slice 3)
# ---------------------------------------------------------------------------


def _proposal(authority=AuthorityClass.LOW, kind=EvolutionTargetKind.SKILL,
              evidence_digest="bd") -> EvolutionProposal:
    return EvolutionProposal(
        proposal_id="p1",
        target=EvolutionTarget(kind=kind, identifier="thing"),
        patch_ref=PatchRef(format="skill_md", patch_digest="pd"),
        evidence_digest=evidence_digest,
        authority_class=authority,
    ).with_digest()


def _bound(authority=AuthorityClass.LOW, kind=EvolutionTargetKind.SKILL, signals=None):
    """A (proposal, bundle) pair correctly bound by target + evidence_digest."""
    bundle = _bundle(kind=kind, signals=signals)
    proposal = _proposal(authority, kind, evidence_digest=bundle.bundle_digest)
    return proposal, bundle


class TestEvolutionProposal:
    def test_digest_stable_across_status_flip(self):
        p = _proposal()
        from arcana.contracts.evolution import ProposalStatus
        flipped = p.model_copy(update={"status": ProposalStatus.APPROVED})
        assert flipped.compute_digest() == p.proposal_digest

    def test_requires_human_approval_for_authority(self):
        assert _proposal(AuthorityClass.AUTHORITY_TOUCHING).requires_human_approval is True
        assert _proposal(AuthorityClass.LOW).requires_human_approval is False

    def test_under_claimed_authority_is_violation(self):
        # Proposal stored LOW but its OWN target is a GUARDRAIL (authority).
        p, bundle = _bound(AuthorityClass.LOW, EvolutionTargetKind.GUARDRAIL)
        assert any("under-claimed" in v for v in p.assert_authority_consistent(bundle))

    def test_laundering_skill_under_claimed(self):
        # SKILL target but the run did a write -> derived AUTHORITY_TOUCHING.
        p, bundle = _bound(AuthorityClass.LOW, EvolutionTargetKind.SKILL,
                           signals=_signals(write_tool_calls=1))
        assert any("under-claimed" in v for v in p.assert_authority_consistent(bundle))

    def test_lying_bundle_target_rejected(self):
        # A GUARDRAIL proposal cannot dodge review by attaching a SKILL bundle.
        bundle = _bundle(kind=EvolutionTargetKind.SKILL)
        p = _proposal(AuthorityClass.LOW, EvolutionTargetKind.GUARDRAIL,
                      evidence_digest=bundle.bundle_digest)
        violations = p.assert_authority_consistent(bundle)
        assert any("target_kind does not match" in v for v in violations)

    def test_unbound_evidence_digest_rejected(self):
        p, _ = _bound()
        other = _bundle(signals=_signals(tool_calls=99))  # different digest
        assert any("evidence_digest does not match" in v
                   for v in p.assert_authority_consistent(other))

    def test_consistent_low_proposal_clean(self):
        p, bundle = _bound(AuthorityClass.LOW, EvolutionTargetKind.SKILL)
        assert p.assert_authority_consistent(bundle) == []

    def test_evidence_sufficient(self):
        p, bundle = _bound()
        assert p.evidence_sufficient(bundle) is True
        empty = EvidenceBundle(bundle_id="e", target_kind=EvolutionTargetKind.DOC).with_digest()
        assert _proposal().evidence_sufficient(empty) is False


# ---------------------------------------------------------------------------
# PromotionRecord (slice 4)
# ---------------------------------------------------------------------------


def _verified_sandbox(digest="sd") -> SandboxVerificationRef:
    return SandboxVerificationRef(
        worktree_id="w", tests_passed=True, lint_passed=True,
        typecheck_passed=True, eval_gate_passed=True, trace_signals_digest=digest,
    )


def _record(**overrides) -> PromotionRecord:
    base = {
        "record_id": "r1", "proposal_digest": "pd",
        "target_kind": EvolutionTargetKind.SKILL,
        "authority_class": AuthorityClass.LOW,
        "decision": AcceptanceDecision.APPROVED,
        "approver_kind": ApproverKind.AUTOMATED_GATE,
        "approver_id": "ci", "sandbox": _verified_sandbox(),
        "rollback_pointer": RollbackPointer(kind="git_revert", pre_change_commit="abc"),
        "monitoring_anchor": MonitoringAnchor(watch_dims=["permission_denials"]),
    }
    base.update(overrides)
    return PromotionRecord(**base)


def _authority_record(**overrides) -> PromotionRecord:
    base = {
        "authority_class": AuthorityClass.AUTHORITY_TOUCHING,
        "target_kind": EvolutionTargetKind.GUARDRAIL,
        "approver_kind": ApproverKind.HUMAN, "approver_id": "alice",
        "proposed_by": "self_evolution_loop", "human_attestation": "sig:abc",
    }
    base.update(overrides)
    return _record(**base)


class TestPromotionRecord:
    def test_low_authority_automated_gate_valid(self):
        assert _record().assert_valid() == []

    def test_authority_approved_human_valid(self):
        # A complete authority approval: human approver, attestation, monitoring.
        assert _authority_record().assert_valid() == []

    def test_authority_auto_approved_is_unconstructible(self):
        with pytest.raises(ValidationError, match="HUMAN approver"):
            _record(authority_class=AuthorityClass.AUTHORITY_TOUCHING,
                    target_kind=EvolutionTargetKind.GUARDRAIL,
                    human_attestation="s", monitoring_anchor=MonitoringAnchor(
                        watch_dims=["x"]))

    def test_structural_floor_self_labelled_low_unconstructible(self):
        # PERMISSION_POLICY target self-labelled LOW + auto-approved must NOT
        # construct — the structural floor forces the authority checks.
        with pytest.raises(ValidationError):
            _record(target_kind=EvolutionTargetKind.PERMISSION_POLICY,
                    authority_class=AuthorityClass.LOW)

    def test_automation_marker_approver_unconstructible(self):
        with pytest.raises(ValidationError, match="automation marker"):
            _authority_record(approver_id="self_evolution_loop")

    def test_self_approval_unconstructible(self):
        with pytest.raises(ValidationError, match="self-approval"):
            _authority_record(approver_id="alice", proposed_by="alice")

    def test_missing_attestation_unconstructible(self):
        with pytest.raises(ValidationError, match="human_attestation"):
            _authority_record(human_attestation=None)

    def test_unverified_sandbox_unconstructible(self):
        with pytest.raises(ValidationError, match="verified sandbox"):
            _record(sandbox=SandboxVerificationRef(worktree_id="w", tests_passed=True))

    def test_missing_rollback_unconstructible(self):
        with pytest.raises(ValidationError, match="rollback"):
            _record(rollback_pointer=RollbackPointer(kind="git_revert"))

    def test_authority_without_monitoring_unconstructible(self):
        with pytest.raises(ValidationError, match="monitoring"):
            _authority_record(monitoring_anchor=MonitoringAnchor(watch_dims=[]))

    def test_non_approved_record_is_constructible(self):
        # A DEFERRED record is not acting, so the validator does not fire.
        r = _record(decision=AcceptanceDecision.DEFERRED,
                    target_kind=EvolutionTargetKind.PERMISSION_POLICY,
                    authority_class=AuthorityClass.LOW,
                    sandbox=SandboxVerificationRef())
        assert r.decision == AcceptanceDecision.DEFERRED

    def test_sandbox_binding_mismatch(self):
        r = _record(sandbox=_verified_sandbox(digest="other"))
        violations = r.assert_valid(expected_trace_signals_digest="cited")
        assert any("sandbox trace_signals_digest" in v for v in violations)

    def test_record_digest_survives_status_flip_and_rollback(self):
        r = _record().with_digest()
        rolled = r.mark_rolled_back("revert-sha")
        assert rolled.compute_digest() == r.record_digest  # identity preserved
        from arcana.contracts.evolution import PostMergeStatus
        assert rolled.post_merge_status == PostMergeStatus.ROLLED_BACK
        assert rolled.rollback_pointer.revert_commit == "revert-sha"
