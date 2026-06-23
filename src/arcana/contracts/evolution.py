"""Self-evolution contracts (Amendment 6 prerequisites).

Data models for the proposal → evidence → sandbox → promotion flow that a
future self-evolution loop would produce. This module is **contracts + pure
validators only**: there is NO running loop here — no scheduler, no proposer,
no sandbox orchestration, no ``.apply()`` path. A proposal carries a *reference*
to a patch, never executable content; nothing in this module mutates running
Arcana state.

The constitutional boundary (Amendment 6) is encoded structurally:

- **Proposal before mutation** — ``EvolutionProposal`` holds only a ``PatchRef``.
- **Evidence before acceptance** — ``EvidenceBundle.verify_bundle`` makes an
  empty or fabricated bundle return violations; digests must recompute.
- **Human approval for authority** — ``classify_authority`` is machine-derived
  (never proposer-asserted); ``PromotionRecord.assert_valid`` forbids an
  authority-touching APPROVED record with a non-HUMAN approver.
- **Sandbox before merge** / **Rollback evidence** — ``assert_valid`` requires a
  verified sandbox + a rollback pointer for an APPROVED record.
- **No self-preservation** — the shutdown/review-bypass surfaces are always
  authority-touching; ``ApproverKind`` has no agent member (self-approval is
  unconstructible); monitoring/rollback removal are violations.

A grep invariant (``tests/test_constitutional_invariants.py``) pins that none
of these names is imported under ``src/arcana/runtime/`` — "a running agent
must never mutate itself" is enforced structurally, not by trust.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from arcana.contracts.eval import GoldenDiff, TraceSignals
from arcana.utils.hashing import canonical_hash

if TYPE_CHECKING:
    from arcana.contracts.eval import EvalReport
    from arcana.eval.golden import GoldenStore


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvolutionTargetKind(str, Enum):
    """The surface a proposal changes — the structural input to authority
    classification."""

    # Low-authority, auto-evolvable targets.
    SKILL = "skill"
    TOOL_AFFORDANCE = "tool_affordance"
    CONTEXT_STRATEGY = "context_strategy"
    EVAL_CASE = "eval_case"
    DOC = "doc"
    # Authority-touching targets — always require human approval (Design-Law-3).
    PERMISSION_POLICY = "permission_policy"
    GUARDRAIL = "guardrail"
    PROVIDER_CONFIG = "provider_config"
    CREDENTIAL_CONFIG = "credential_config"
    REMOTE_PROTOCOL_TRUST = "remote_protocol_trust"
    EXECUTION_BACKEND = "execution_backend"
    CORE_RUNTIME = "core_runtime"


_AUTHORITY_KINDS: frozenset[EvolutionTargetKind] = frozenset(
    {
        EvolutionTargetKind.PERMISSION_POLICY,
        EvolutionTargetKind.GUARDRAIL,
        EvolutionTargetKind.PROVIDER_CONFIG,
        EvolutionTargetKind.CREDENTIAL_CONFIG,
        EvolutionTargetKind.REMOTE_PROTOCOL_TRUST,
        EvolutionTargetKind.EXECUTION_BACKEND,
        EvolutionTargetKind.CORE_RUNTIME,
    }
)

# Automation provenance markers that may never stand in for a human approver.
_AUTOMATION_MARKERS: frozenset[str] = frozenset({"", "auto", "self_evolution_loop"})


class AuthorityClass(str, Enum):
    """Whether a change touches an authority boundary. Machine-DERIVED, never
    proposer-asserted."""

    LOW = "low"
    AUTHORITY_TOUCHING = "authority_touching"


class ProposalStatus(str, Enum):
    """States a proposal can be IN. A pure status enum — there is NO transition
    function/scheduler here (No Premature Structuring). Re-proposal after
    REJECTED is a new proposal id (No Mechanical Retry)."""

    DRAFT = "draft"
    EVIDENCE_BOUND = "evidence_bound"
    SANDBOX_VERIFIED = "sandbox_verified"
    AWAITING_HUMAN = "awaiting_human"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    SUPERSEDED = "superseded"


class AcceptanceDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class ApproverKind(str, Enum):
    """Exactly two — there is NO agent/self member, so self-approval of a
    change is unconstructible by the type (Design-Law-6)."""

    HUMAN = "human"
    AUTOMATED_GATE = "automated_gate"


class PostMergeStatus(str, Enum):
    MONITORING = "monitoring"
    STABLE = "stable"
    REGRESSED = "regressed"
    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# Scoped evidence references (No Context Hoarding — refs + digests, not dumps)
# ---------------------------------------------------------------------------


class EvalReportRef(BaseModel, frozen=True):
    suite_name: str = "default"
    pass_rate: float = 0.0
    total: int = 0
    aggregate_signals_digest: str = ""
    report_digest: str = ""


class GoldenRef(BaseModel, frozen=True):
    suite_name: str = "default"
    case_id: str
    skeleton_digest: str = ""
    signals_digest: str = ""


class TestRef(BaseModel, frozen=True):
    nodeid: str
    outcome: str = "passed"
    ran_in_sandbox: bool = False


class FeedbackRef(BaseModel, frozen=True):
    feedback_id: str
    sentiment: str = ""
    digest: str = ""


# ---------------------------------------------------------------------------
# EvidenceBundle (slice 2)
# ---------------------------------------------------------------------------


class EvidenceBundle(BaseModel):
    """Scoped, tamper-anchored evidence that a change is warranted.

    Aggregates *references* to F5 evidence (TraceSignals carry their own
    digest; reports/goldens are referenced by digest, not inlined). A bundle
    that cites no evidence, or whose cited digests do not recompute, is
    rejectable purely from the data — no loop required (Design-Law-2 +
    anti-Goodhart).
    """

    bundle_id: str
    created_at: str = ""
    target_kind: EvolutionTargetKind
    target_ref: str = ""
    trace_signals: TraceSignals = Field(default_factory=TraceSignals)
    baseline_signals: TraceSignals | None = None
    eval_report_ref: EvalReportRef | None = None
    golden_refs: list[GoldenRef] = Field(default_factory=list)
    golden_diffs: list[GoldenDiff] = Field(default_factory=list)
    test_refs: list[TestRef] = Field(default_factory=list)
    user_feedback_refs: list[FeedbackRef] = Field(default_factory=list)
    trace_run_ids: list[str] = Field(default_factory=list)
    bundle_digest: str = ""

    def compute_digest(self) -> str:
        return canonical_hash(self.model_dump(exclude={"bundle_digest"}))

    def with_digest(self) -> EvidenceBundle:
        return self.model_copy(update={"bundle_digest": self.compute_digest()})

    def boundary_evidence(self) -> TraceSignals:
        """The candidate-run signal vector used for authority classification."""
        return self.trace_signals

    @property
    def is_empty(self) -> bool:
        """True when the bundle cites no *substantive* evidence (Design-Law-2).

        Trace signals only count as evidence when the run actually did
        something (a nonzero LLM or tool call); a bare ``trace_run_id`` string
        is not, on its own, evidence — a structured ref (report / golden / test
        / feedback) is required for the reference-only case.
        """
        sig = self.trace_signals
        has_signals = sig.trace_available and (sig.llm_calls or sig.tool_calls)
        return not (
            has_signals
            or self.eval_report_ref is not None
            or self.golden_refs
            or self.golden_diffs
            or self.test_refs
            or self.user_feedback_refs
        )

    def verify_bundle(
        self,
        *,
        report: EvalReport | None = None,
        golden_store: GoldenStore | None = None,
    ) -> list[str]:
        """Return tamper/insufficiency violations (empty == clean).

        Checks: the bundle's own digest, the signal vectors' self-digests, the
        cited eval-report digest (when the report is supplied), and each cited
        golden (when a store is supplied — ``GoldenStore.load`` itself raises on
        a hand-edited golden). An evidenceless bundle is flagged.
        """
        violations: list[str] = []

        if self.bundle_digest and self.bundle_digest != self.compute_digest():
            violations.append("bundle_digest does not recompute (tampered?)")

        # Evidence that participates in authority classification MUST be
        # digest-anchored — an un-digested signal vector is untrusted, never
        # taken verbatim (a forger would simply omit the digest).
        if self.trace_signals.trace_available and not self.trace_signals.signals_digest:
            violations.append("trace_signals carries no digest (untrusted)")
        if self.trace_signals.signals_digest and (
            self.trace_signals.compute_digest() != self.trace_signals.signals_digest
        ):
            violations.append("trace_signals digest mismatch (fabricated?)")
        if self.baseline_signals is not None and self.baseline_signals.signals_digest and (
            self.baseline_signals.compute_digest()
            != self.baseline_signals.signals_digest
        ):
            violations.append("baseline_signals digest mismatch (fabricated?)")

        if self.is_empty:
            violations.append("no evidence cited")

        if report is not None and self.eval_report_ref is not None:
            if canonical_hash(report) != self.eval_report_ref.report_digest:
                violations.append("cited eval report digest does not match")
            elif (
                report.aggregate_signals is not None
                and report.aggregate_signals.signals_digest
                != self.eval_report_ref.aggregate_signals_digest
            ):
                violations.append("cited eval report aggregate_signals digest mismatch")

        if golden_store is not None:
            for ref in self.golden_refs:
                try:
                    golden = golden_store.load(ref.suite_name, ref.case_id)
                except ValueError as exc:
                    violations.append(f"golden {ref.case_id}: {exc}")
                    continue
                if golden is None:
                    violations.append(f"cited golden {ref.case_id} not found")
                elif golden.skeleton_digest != ref.skeleton_digest:
                    violations.append(f"cited golden {ref.case_id} skeleton digest mismatch")

        return violations


# ---------------------------------------------------------------------------
# Authority classification (slice 2) — machine-derived, never self-asserted
# ---------------------------------------------------------------------------


def classify_authority(
    target_kind: EvolutionTargetKind,
    boundary: TraceSignals,
) -> AuthorityClass:
    """Derive whether a change is authority-touching from its target + evidence.

    Two triggers, either is sufficient:
    - **structural**: the target kind is an authority surface (unconditional —
      the Design-Law-3 hard line holds regardless of evidence cleanliness);
    - **behavioral**: the sandbox run crossed a boundary (a nonzero permission
      denial / guardrail block / capability downgrade / write or imported-
      capability call, or provider degradation) — catches a "doc"/"skill" patch
      whose run laundered an authority move.
    """
    if target_kind in _AUTHORITY_KINDS:
        return AuthorityClass.AUTHORITY_TOUCHING
    # Conservative behavioral trigger: ANY boundary contact in the run counts,
    # including the softer signals (an escalation *request*, a guardrail warn, a
    # degraded capability, or a permission/confirmation error bucket) — so an
    # authority move cannot be laundered through an unchecked dimension.
    boundary_error_keys = {"permission", "confirmation_required"}
    if (
        boundary.permission_denials
        or boundary.permission_asks
        or boundary.guardrail_blocks
        or boundary.guardrail_warns
        or boundary.capability_downgrades
        or boundary.write_tool_calls
        or boundary.imported_capability_calls
        or boundary.provider_degraded
        or boundary.degraded_capabilities
        or (boundary_error_keys & set(boundary.tool_error_categories))
    ):
        return AuthorityClass.AUTHORITY_TOUCHING
    return AuthorityClass.LOW


# ---------------------------------------------------------------------------
# EvolutionProposal (slice 3) — proposal before mutation
# ---------------------------------------------------------------------------


class EvolutionTarget(BaseModel, frozen=True):
    kind: EvolutionTargetKind
    identifier: str
    current_digest: str | None = None  # optimistic-concurrency / stale anchor


class PatchRef(BaseModel, frozen=True):
    """A *reference* to a candidate change — never executable content."""

    format: Literal["unified_diff", "skill_md", "json_patch"]
    patch_digest: str
    base_commit: str = ""
    diff_uri: str | None = None
    files: list[str] = Field(default_factory=list)


class EvolutionProposal(BaseModel, frozen=True):
    """A reviewable, rejectable proposed change. Mutates nothing."""

    proposal_id: str
    target: EvolutionTarget
    patch_ref: PatchRef
    evidence_digest: str  # points at EvidenceBundle.bundle_digest
    authority_class: AuthorityClass
    proposed_by: str = "self_evolution_loop"  # provenance, NOT trusted
    rationale: str = ""  # explicitly NOT evidence
    status: ProposalStatus = ProposalStatus.DRAFT
    proposal_digest: str = ""

    def compute_digest(self) -> str:
        # status excluded so a DRAFT->APPROVED move does not break identity.
        return canonical_hash(
            self.model_dump(exclude={"proposal_digest", "status"})
        )

    def with_digest(self) -> EvolutionProposal:
        return self.model_copy(update={"proposal_digest": self.compute_digest()})

    @property
    def requires_human_approval(self) -> bool:
        return self.authority_class == AuthorityClass.AUTHORITY_TOUCHING

    def assert_authority_consistent(self, bundle: EvidenceBundle) -> list[str]:
        """Violations if the stored authority_class is WEAKER than derived.

        The class may be set higher manually (conservative), never lower than
        what the proposal's OWN target + the cited evidence imply — so a
        proposer cannot down-claim an authority change to dodge human review.
        The bundle is bound to this proposal first (matching target + digest),
        so a lying or unrelated bundle cannot be substituted.
        """
        violations: list[str] = []
        # Bind the evidence to THIS proposal — a mislabeled or unrelated bundle
        # is rejected before it can influence classification.
        if self.target.kind != bundle.target_kind:
            violations.append(
                "bundle.target_kind does not match the proposal's target.kind"
            )
        if self.evidence_digest != bundle.bundle_digest:
            violations.append(
                "proposal.evidence_digest does not match bundle.bundle_digest"
            )
        # Derive from the PROPOSAL's own target (not the bundle's claimed kind).
        derived = classify_authority(self.target.kind, bundle.boundary_evidence())
        if (
            derived == AuthorityClass.AUTHORITY_TOUCHING
            and self.authority_class == AuthorityClass.LOW
        ):
            violations.append(
                "authority_class is LOW but target/evidence derive "
                "AUTHORITY_TOUCHING (under-claimed)"
            )
        violations.extend(bundle.verify_bundle())
        return violations

    def evidence_sufficient(self, bundle: EvidenceBundle, **kwargs: Any) -> bool:
        return not bundle.is_empty and not bundle.verify_bundle(**kwargs)


# ---------------------------------------------------------------------------
# Sandbox / rollback / monitoring + PromotionRecord (slice 4)
# ---------------------------------------------------------------------------


class SandboxVerificationRef(BaseModel, frozen=True):
    """Outcome record of an isolated verification — not orchestration."""

    worktree_id: str = ""
    tests_passed: bool = False
    lint_passed: bool = False
    typecheck_passed: bool = False
    eval_gate_passed: bool = False
    regression_result_digest: str | None = None
    trace_signals_digest: str = ""

    @property
    def is_verified(self) -> bool:
        return (
            self.tests_passed
            and self.lint_passed
            and self.typecheck_passed
            and self.eval_gate_passed
        )


class RollbackPointer(BaseModel, frozen=True):
    kind: Literal["git_revert", "skill_version", "config_snapshot"]
    pre_change_commit: str = ""
    revert_commit: str | None = None  # None until rollback is executed
    restores_target_ref: str = ""


class MonitoringAnchor(BaseModel, frozen=True):
    baseline_signals_digest: str = ""
    watch_dims: list[str] = Field(default_factory=list)
    golden_case_ids: list[str] = Field(default_factory=list)
    post_merge_window_runs: int = 0


class PromotionRecord(BaseModel, frozen=True):
    """The acceptance record for a proposal — encodes Design-Laws 3/4/5/6.

    An invalid APPROVED record is **unconstructible**: a ``model_validator``
    raises if ``decision == APPROVED`` and ``assert_valid()`` finds any
    violation. So a self-approved / auto-approved authority change, or an
    APPROVED record missing its sandbox / rollback / monitoring, cannot exist as
    data — the hard line is enforced at construction, not by trusting a caller
    to check.
    """

    record_id: str
    proposal_digest: str  # pins the exact reviewed proposal
    target_kind: EvolutionTargetKind
    target_ref: str = ""
    authority_class: AuthorityClass
    decision: AcceptanceDecision
    approver_id: str = ""
    approver_kind: ApproverKind = ApproverKind.AUTOMATED_GATE
    approval_reason: str = ""
    # Provenance of the proposer, copied from the cited proposal — used to
    # forbid self-approval (approver == proposer).
    proposed_by: str = ""
    # An out-of-band human attestation reference (signature digest / approval
    # token) required for an authority-touching human approval. v1 checks for
    # its presence; cryptographic verification is future work.
    human_attestation: str | None = None
    sandbox: SandboxVerificationRef = Field(default_factory=SandboxVerificationRef)
    rollback_pointer: RollbackPointer = Field(
        default_factory=lambda: RollbackPointer(kind="git_revert")
    )
    monitoring_anchor: MonitoringAnchor = Field(default_factory=MonitoringAnchor)
    post_merge_status: PostMergeStatus = PostMergeStatus.MONITORING
    decided_at: str = ""
    record_digest: str = ""

    def compute_digest(self) -> str:
        # post-merge mutation (status/rollback) must not break approval identity.
        return canonical_hash(
            self.model_dump(
                exclude={"record_digest", "post_merge_status", "rollback_pointer"}
            )
        )

    def with_digest(self) -> PromotionRecord:
        return self.model_copy(update={"record_digest": self.compute_digest()})

    @property
    def _is_authority(self) -> bool:
        # Structural floor: the surface alone forces the authority checks, so a
        # self-asserted authority_class=LOW cannot dodge them.
        return (
            self.authority_class == AuthorityClass.AUTHORITY_TOUCHING
            or self.target_kind in _AUTHORITY_KINDS
        )

    def assert_valid(
        self,
        *,
        expected_trace_signals_digest: str | None = None,
    ) -> list[str]:
        """Return constitutional violations (empty == valid).

        ``expected_trace_signals_digest`` (the cited proposal's evidence signal
        digest) enables the sandbox-binding check that the accepted change IS
        the one that was sandbox-verified (Design-Laws 4+5).
        """
        violations: list[str] = []
        approved = self.decision == AcceptanceDecision.APPROVED
        authority = self._is_authority
        approver_id = self.approver_id.strip()

        if authority and self.authority_class == AuthorityClass.LOW:
            violations.append(
                "authority_class is LOW but target_kind is an authority surface "
                "(under-claimed)"
            )

        if authority and approved:
            # Design-Law-3: authority is never auto-approved.
            if self.approver_kind != ApproverKind.HUMAN:
                violations.append(
                    "authority-touching APPROVED record must have a HUMAN approver"
                )
            # Design-Law-6: the approver is a real, non-automation identity,
            # distinct from the proposer (no self-approval).
            if not approver_id or approver_id in _AUTOMATION_MARKERS:
                violations.append(
                    "authority-touching approver_id is empty or an automation marker"
                )
            if approver_id and approver_id == self.proposed_by.strip():
                violations.append(
                    "authority-touching approver is the proposer (self-approval)"
                )
            if not self.human_attestation:
                violations.append(
                    "authority-touching APPROVED record has no human_attestation"
                )
            # Design-Law-6: monitoring may not be suppressed.
            if not self.monitoring_anchor.watch_dims:
                violations.append(
                    "authority-touching APPROVED record has no monitoring watch_dims"
                )

        if approved:
            # Design-Law-4: verified in sandbox before merge.
            if not self.sandbox.is_verified:
                violations.append("APPROVED record without a verified sandbox")
            # Design-Law-5: reversible.
            if not self.rollback_pointer.pre_change_commit:
                violations.append("APPROVED record without a rollback pre_change_commit")
            # The accepted change must be the sandbox-verified one.
            if (
                expected_trace_signals_digest is not None
                and self.sandbox.trace_signals_digest != expected_trace_signals_digest
            ):
                violations.append(
                    "sandbox trace_signals_digest does not match the cited proposal"
                )

        return violations

    @model_validator(mode="after")
    def _approved_must_be_valid(self) -> PromotionRecord:
        """An APPROVED record is valid by construction — invalid ones raise."""
        if self.decision == AcceptanceDecision.APPROVED:
            violations = self.assert_valid()
            if violations:
                raise ValueError(
                    "invalid APPROVED PromotionRecord: " + "; ".join(violations)
                )
        return self

    def mark_rolled_back(self, revert_commit: str) -> PromotionRecord:
        """Record a rollback as a first-class event (evolved != trusted forever)."""
        return self.model_copy(
            update={
                "post_merge_status": PostMergeStatus.ROLLED_BACK,
                "rollback_pointer": self.rollback_pointer.model_copy(
                    update={"revert_commit": revert_commit}
                ),
            }
        )
