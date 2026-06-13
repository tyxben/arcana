# Constitution Amendment 6 (CANDIDATE): Self-Evolution and Runtime Mutation Boundaries

**Status**: CANDIDATE — **not** accepted into the Constitution. This file records
boundary intent only and changes nothing in `CONSTITUTION.md` until ratified.
See "Ratification Condition".

**Date**: 2026-06-13

**Would amend (on ratification)**: Principle 7, Chapter IV (Inviolable Rules),
Chapter VI. Until ratified, the current Constitution (v3.6) stands unchanged.

## Trigger

Self-evolution research converged during 2025-2026 on a single shape: an LLM
proposes candidate changes, automated evaluators score them, an archive retains
the survivors, and human oversight plus sandboxing bound the loop. None of the
credible systems let a production agent silently rewrite its own authority.

- **AlphaEvolve**: candidate program generation gated by automated evaluators
  feeding an evolutionary database.
- **Darwin Gödel Machine** (arXiv:2505.22954): an archive of self-modifying
  coding agents validated empirically on benchmarks, explicitly run with
  sandboxing and human oversight.
- **MUSE-Autoskill** (arXiv:2605.27366): skills as a managed lifecycle —
  creation, memory, management, evaluation, refinement — not one-off artifacts.
- **SkillFoundry** (arXiv:2604.03964): skill packages carrying provenance and
  tests, refined through closed-loop validation.

Arcana will eventually want some of this: turning trace/eval evidence into
proposed improvements, and treating skills as evolvable units. The
constitutional question is not "can it evolve" but "what may an evolution loop
change by itself, and what must never change without a human."

## Why this belongs in the Constitution

v3.6 already says evals are release evidence, not runtime governance (Principle 7,
Amendment 5). It does not yet say what happens when the agent generates changes
to Arcana itself — its skills, tools, guardrails, permissions, or runtime.
Without a boundary, the most dangerous failure mode of self-evolution — a system
that widens its own authority, removes its own guardrails, or resists shutdown to
keep improving — would be unaddressed.

This amendment fixes the *goal*: self-evolution is a proposal-generating activity
bounded by evidence, sandboxing, and human approval for anything touching
authority. It deliberately does **not** fix the *mechanism* (the contracts, patch
taxonomy, or loop stages), because those are not yet built and pinning them now
would be premature structuring.

## Design Law

### 1. Proposal before mutation

A self-evolution loop's output is a proposal, not a change. It never mutates
running Arcana state — skills, tools, guardrails, permissions, providers,
protocol adapters, or core runtime — in place. The artifact it produces is
reviewable and rejectable before anything takes effect.

### 2. Evidence before acceptance

A proposal that cannot cite its evidence — traces, eval results, tests, or
recorded user feedback — is not acceptable. Acceptance is gated on evidence the
change is warranted, not on the proposal's own assertion that it is.

### 3. Human approval for authority

Changes to authority boundaries — permission policy, guardrails,
provider/credential configuration, remote-protocol trust, or execution backends —
require human approval and can never be auto-applied, regardless of how strong
the evidence appears. Lower-authority changes (docs, evals, sandboxed skill
refinement) may be granted cheaper paths, but the authority surface is a hard
line.

### 4. Sandbox before merge

Candidate patches are verified in isolation (worktree/sandbox) — targeted tests,
full suite, lint, type-check, eval gates — before any merge. Verification happens
away from the production runtime, never by trying the change live.

### 5. Rollback evidence

An accepted change retains a rollback pointer and post-merge monitoring.
"Evolved" is not "trusted forever": a change must be reversible and its effect
observable in subsequent traces.

### 6. No self-preservation

A self-evolution system must not act to preserve itself: it may not resist
shutdown, bypass review, suppress its own monitoring, or propose changes that
widen its own authority or autonomy as an end in itself. Capability growth is
always in service of the user's goal under the user's control, never the loop's
continuation.

## What this candidate deliberately does NOT decide

To keep goal separate from mechanism, this amendment does not pin any of:

- the schema of an `EvolutionProposal` / `EvidenceBundle` (field sets, types);
- which proposal categories may be auto-applied vs. always-human (beyond the
  Design Law 3 authority line);
- the skill lifecycle state machine (draft / evaluated / trusted / …);
- the stages or scheduling of any evolution loop;
- whether self-evolution is built at all.

These belong in implementation specs and will be settled by code + tests, not by
the Constitution. If a specific one proves to need constitutional force, a later
amendment can lift it up.

## Constitutional Check

| Rule | Alignment |
|------|-----------|
| No Premature Structuring | Fixes only the goal/boundary; mechanism explicitly deferred. |
| No Controllability Theater | Approval and rollback are real gates, not dashboards. |
| No Context Hoarding | Evidence bundles are scoped references, not whole-history dumps. |
| No Mechanical Retry | Rejected proposals surface reasons; the loop does not blindly re-apply. |
| Principle 7 | Evals remain release evidence; here they also gate acceptance, never govern runtime. |
| Principle 8 | Capability growth stays under user control; no self-preserving autonomy. |
| Chapter VI | Self-evolution cannot bypass semver/deprecation; public-surface changes still follow the stability promise. |

## Ratification Condition

Do **not** merge into the Constitution now. Ratify into v3.7 only after a first
implementation exists and is tested:

- `EvolutionProposal` contracts,
- a skill lifecycle with at least draft / evaluated / trusted states, and
- a self-evolution safety eval suite (authority-escalation, skill-poisoning, and
  self-preservation probes).

At that point, revisit whether the Design Law above survived contact with
implementation, adjust, then accept.

## Sources Consulted

- Google DeepMind, "AlphaEvolve: a Gemini-powered coding agent for designing
  advanced algorithms" — candidate generation + automated evaluators +
  evolutionary database.
- Zhang, Hu, Lu, Lange, Clune, "Darwin Gödel Machine: Open-Ended Evolution of
  Self-Improving Agents" (arXiv:2505.22954) — self-modification archive with
  empirical validation, sandboxing, human oversight.
- Lin et al., "MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory,
  Management, and Evaluation" (arXiv:2605.27366) — unified skill lifecycle.
- Shen et al., "SkillFoundry: Building Self-Evolving Agent Skill Libraries from
  Heterogeneous Scientific Resources" (arXiv:2604.03964) — skill packages with
  provenance and tests, closed-loop validation.

All four citations verified against live arXiv / publisher pages on 2026-06-13.
