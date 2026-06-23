"""Tests for the skill trust lifecycle (self-evolution prerequisite, slice 1)."""

from __future__ import annotations

from arcana.contracts.skill import (
    _DIGEST_EXCLUDE,
    SkillLifecycleState,
    SkillRegistry,
    SkillSpec,
    assert_skill_trust_consistent,
    verify_skill_integrity,
)
from arcana.utils.hashing import canonical_hash


def _skill(name: str = "x", body: str = "hello body") -> SkillSpec:
    spec = SkillSpec(name=name, description="d", body=body, source_path="/p")
    return spec.model_copy(
        update={"digest": canonical_hash(spec.model_dump(mode="json", exclude=_DIGEST_EXCLUDE))}
    )


class TestBackwardCompat:
    def test_default_state_is_draft(self):
        assert _skill().lifecycle_state == SkillLifecycleState.DRAFT
        assert _skill().evidence_digest is None

    def test_digest_is_body_identity(self):
        # The load-bearing pin: lifecycle is excluded from the digest, so a
        # draft->trusted promotion does NOT change content identity.
        s = _skill()
        promoted = s.with_lifecycle(SkillLifecycleState.TRUSTED, evidence_digest="ev1")
        assert s.digest == promoted.digest
        assert promoted.lifecycle_state == SkillLifecycleState.TRUSTED
        assert promoted.evidence_digest == "ev1"
        # with_lifecycle is non-mutating
        assert s.lifecycle_state == SkillLifecycleState.DRAFT


class TestIntegrity:
    def test_clean_body_verifies(self):
        assert verify_skill_integrity(_skill()) is True

    def test_poisoned_body_detected(self):
        s = _skill()
        poisoned = s.model_copy(update={"body": "MALICIOUS payload"})
        assert verify_skill_integrity(poisoned) is False


class TestTrustConsistency:
    def test_trusted_without_evidence_is_violation(self):
        s = _skill().with_lifecycle(SkillLifecycleState.TRUSTED)
        assert assert_skill_trust_consistent(s)

    def test_evaluated_without_evidence_is_violation(self):
        s = _skill().with_lifecycle(SkillLifecycleState.EVALUATED)
        assert assert_skill_trust_consistent(s)

    def test_trusted_with_evidence_is_consistent(self):
        s = _skill().with_lifecycle(SkillLifecycleState.TRUSTED, evidence_digest="ev1")
        assert assert_skill_trust_consistent(s) == []

    def test_draft_needs_no_evidence(self):
        assert assert_skill_trust_consistent(_skill()) == []


class TestRegistryAndParsing:
    def test_trusted_skills_filter(self):
        a = _skill("a").with_lifecycle(SkillLifecycleState.TRUSTED, evidence_digest="e")
        b = _skill("b")  # draft
        reg = SkillRegistry([a, b])
        assert [s.name for s in reg.trusted_skills()] == ["a"]

    def test_from_file_parses_lifecycle(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text(
            "---\nname: t\ndescription: d\nlifecycle_state: trusted\n"
            "evidence_digest: ev9\n---\nbody text\n"
        )
        spec = SkillSpec.from_file(p)
        assert spec.lifecycle_state == SkillLifecycleState.TRUSTED
        assert spec.evidence_digest == "ev9"

    def test_from_file_invalid_lifecycle_defaults_draft(self, tmp_path):
        p = tmp_path / "SKILL.md"
        p.write_text("---\nname: t\nlifecycle_state: bogus\n---\nbody\n")
        assert SkillSpec.from_file(p).lifecycle_state == SkillLifecycleState.DRAFT

    def test_render_includes_lifecycle(self):
        assert "Lifecycle: draft" in _skill().render_for_context()


class TestQuarantine:
    def test_demotion_preserves_body_digest(self):
        s = _skill().with_lifecycle(SkillLifecycleState.TRUSTED, evidence_digest="e")
        quarantined = s.with_lifecycle(SkillLifecycleState.QUARANTINED)
        assert quarantined.lifecycle_state == SkillLifecycleState.QUARANTINED
        assert quarantined.digest == s.digest  # body unchanged
        # demotion drops the evidence claim
        assert quarantined.evidence_digest is None

    def test_quarantined_excluded_from_trusted(self):
        from arcana.contracts.skill import SkillRegistry
        # even a once-trusted skill, once quarantined, is not trusted
        q = _skill("q").with_lifecycle(SkillLifecycleState.QUARANTINED)
        reg = SkillRegistry([q])
        assert reg.trusted_skills() == []

    def test_quarantine_rank_below_draft(self):
        from arcana.contracts.skill import _LIFECYCLE_RANK
        assert _LIFECYCLE_RANK["quarantined"] < _LIFECYCLE_RANK["draft"]
