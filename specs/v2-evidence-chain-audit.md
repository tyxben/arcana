# V2 Evidence / Budget / Degradation Chain Audit

**Status**: Active remediation tracker

**Date**: 2026-06-23

**Owns**: Closing the gap between Arcana's design slogans ("Trace Everything",
honest cost, no controllability theater) and what the V2 `ConversationAgent`
actually does, as a precondition for the self-evolution work in
`specs/constitution-amendment-6-self-evolution-boundaries.md`.

## Position

The architecture direction is sound (contracts-first, TurnFacts/TurnAssessment
separation, runtime-as-OS, tool side-effects, context decisions, trace replay).
The problem is not direction; it is that the **evidence chain, budget chain, and
degradation chain are not yet fully wired into the V2 engine**. Self-evolution
must be built on honest evidence, so these are fixed first.

A running agent must never mutate itself. The intended form (already scoped by
Amendment 6) is: trace/eval/user-feedback → `EvidenceBundle` → LLM-generated
`EvolutionProposal` → isolated sandbox runs test/lint/typecheck/eval gates →
reviewable patch → human approval for anything touching permissions,
guardrails, provider credentials, or execution backends.

## Findings (all verified against the code)

| # | Severity | Finding | Status |
|---|----------|---------|--------|
| F1 | High | V2 `ConversationAgent` did not pass a `trace_ctx` to `gateway.generate` / `tool_gateway.call_many`; provider `LLM_CALL` and gateway `TOOL_CALL` audit events are gated on `trace_ctx`, so they never fired on the live path. **Knock-on**: the Phase 1/3 per-call `permission_decision` / `GUARDRAIL_DECISION` metadata rode on the missing `TOOL_CALL` event, so that audit was real only in gateway-level unit tests. The shared `ToolGateway` (from `Runtime(tools=...)`) was also built without a `trace_writer`. | **DONE** |
| F3 | Med-High | `WorkingSetBuilder._compress_with_llm` calls `gateway.generate` for summarization, but that usage was discarded — not counted against budget and not traced. Long conversations systematically under-count cost. | **DONE** |
| F2 | High | Prompt-based tool fallback (`_generate_with_text_tools`) is a pseudo-capability: it asks the model to emit a JSON tool call in **text**, but the provider builds `LLMResponse.tool_calls` only from native `message.tool_calls`, and `_parse_turn` only reads `response.tool_calls`. A degraded provider shows tools but never executes them (No Controllability Theater). | **DONE** |
| F4 | Low-Med | `_execute_tools` buckets ask_user → cognitive → gateway and appends results in that order, so a turn mixing built-in and gateway tools returns `results` (and TOOL_END events) out of input order. The LLM conversation is `tool_call_id`-matched (answers are correct); the returned-order contract is the wart. `call_many` itself is already order-preserving. | **DONE** |
| F5 | Med | Eval gates on pass-rate / cost / tokens; `EvalResult` carries status/steps/tokens/cost. Self-evolution needs trace-derived signals (context loss, tool-error category, permission denial, provider degradation, prompt-snapshot replay diff) and golden-trace replay. | **DONE** |
| F6 | Med | (Surfaced during F2.) `sdk._signature_to_json_schema` maps a parameter's annotation through a `{type: json_type}` dict, but under `from __future__ import annotations` (the modern default) annotations are **strings** (`"int"`), so every typed param schemas as `"string"`. Tools defined in such a module reject correctly-typed int/float/bool/list/dict args at validation. Fix: resolve string annotations (`typing.get_type_hints`) before mapping. | **DONE** |

## Remediation order (by weight)

1. **F1 + F3 — evidence + budget chain (DONE).** One root cause: the V2 agent
   now builds a per-turn `TraceContext(run_id, parent_step_id=turn_step_id)` and
   threads it into `gateway.generate` / `gateway.stream`, `tool_gateway.call_many`
   (via `_execute_tools`), the direct-answer fast path (`DirectExecutor.direct_answer`
   gained a `trace_ctx` param), and the resume loop. The shared `ToolGateway` is
   now constructed with the runtime trace writer (init reordered: trace before
   tools). Compression threads `trace_ctx` into its summarization call and records
   usage via `WorkingSetBuilder.consume_compression_usage()`, which the agent
   folds into reported `state` cost + `budget_tracker` each turn.
   - Files: `runtime/conversation.py`, `context/builder.py`, `routing/executor.py`,
     `runtime_core.py`.
   - Tests: `tests/test_evidence_chain.py`, invariant
     `TestTraceEverythingOnLivePath`.
2. **F2 — make the prompt-based tool fallback real (DONE).** Module-level
   `parse_text_tool_calls(content)` in `openai_compatible.py` recovers the text
   JSON into `ToolCallRequest`s (fence-tolerant, string-aware balanced-brace
   scan, validates `tool_call`+`name`, normalizes double-encoded `arguments`,
   skips malformed). Wired into `generate()` (via a `use_text_tools` flag,
   promoting `finish_reason` to `tool_calls`) AND `stream()` (delegates to
   `generate()` for the degraded case so streaming has parity). `_parse_turn`
   stays a pure mapping.
   - Tests: `tests/test_text_tools_parser.py` (parser units + provider
     end-to-end + streaming parity + full degraded-agent loop that executes a
     real tool), invariant `TestDegradedToolCallsAreReal`.
3. **F4 — reassemble `results` in input order (DONE).** `_execute_tools`
   re-keys results by `tool_call_id` and re-emits in input order before
   returning. Tests: `tests/test_tool_result_ordering.py`.
4. **F5 — trace-derived eval metrics + golden trace replay (DONE).** A
   tamper-evident, orthogonal fitness vector (`TraceSignals`) extracted
   post-hoc from a run's trace (`eval/signals.py`), recorded on `EvalResult`
   and merged onto `EvalReport`; the `RegressionGate` gains opt-in hard-fail
   signal gates (default warn — evidence, not governance) and a grep invariant
   pins that it has no call site under `runtime/`. Two producer fixes close
   real blind spots: `ToolCallRecord.error_category` (new tool-error category
   is now detectable) and a provider-degradation trace marker on `LLM_CALL`
   (a silent downgrade was theater-adjacent — now a counted, gateable signal
   with its own constitutional invariant). Golden-trace replay
   (`eval/golden.py`) records a redacted, committed skeleton + signal vector
   and diffs asymmetrically — improvements never count as regressions, only
   unsafe-direction moves do; goldens are recorded only by an explicit op
   (never auto), so relaxing one is a reviewable git diff. Forward-composes
   into Amendment 6: `TraceSignals` is the EvidenceBundle atom and
   `RegressionGate.compare(candidate, baseline, goldens)` with boundary
   ceilings=0 + `golden_replay="strict"` IS the future acceptance gate — that
   config is expressible now, not wired. Decisions: eval contracts kept as
   reshapeable implementation surface (not stability §1.4); library default
   all-warn + a stricter "house" config for Arcana's own CI; goldens committed
   digest-only (zero PII). **Follow-up**: `arcana eval --record-golden` CLI
   (the current `arcana eval` command uses the separate `baseline.EvalGate`;
   exposing `EvalRunner` golden flags is a thin CLI refactor); prompt-snapshot
   digests for prompt-rewrite detection (opt-in, deferred). Adversarially
   reviewed (cloud workflow); fixed the real findings — authorization-failure
   denials are now counted (not laundered into `unexpected`), boundary refusals
   are excluded from `tool_error_categories`/`write_tool_calls` (orthogonality),
   the golden skeleton is canonical-sorted (determinism), `GoldenStore.load`
   enforces the digest tamper-check, the direct gate gained write/imported/
   downgrade/fidelity ceilings, and `require_trace` is per-case. **Deferred
   (lower-priority review items)**: surface-expansion signals
   (`capabilities_admitted` / protocol-source discovery — CAPABILITY_ADMISSION
   is MCP-setup-time, rarely in a single eval run's trace); same-category
   tool-error *count* explosions in golden (the new-category check is
   set-based); `tool_calls` denominator counting RAG-retrieval events.
   - Files: `contracts/eval.py`, `contracts/trace.py`, `eval/signals.py`,
     `eval/golden.py`, `eval/gate.py`, `eval/runner.py`,
     `gateway/providers/openai_compatible.py`, `tool_gateway/gateway.py`.
   - Tests: `tests/eval/test_trace_signals.py`, `tests/eval/test_signal_gate.py`,
     `tests/eval/test_golden_replay.py`, `tests/test_evidence_chain.py`;
     invariants `TestDegradedToolCallsAreReal` (degradation marker),
     `TestEvalIsGateNotGovernance` (no gate in runtime/).
5. **F6 — resolve string annotations in `_signature_to_json_schema`** so
   future-annotations modules get correctly-typed tool schemas.
6. **Self-evolution contracts (DONE — contracts only, no running loop).** The
   three Amendment-6 ratification prerequisites now exist + are tested:
   (a) `EvidenceBundle` / `EvolutionProposal` / `PromotionRecord` +
   `classify_authority` in `contracts/evolution.py`; (b) a skill trust lifecycle
   (`SkillLifecycleState` DRAFT/EVALUATED/TRUSTED/QUARANTINED on `SkillSpec`,
   body-identity digest, `verify_skill_integrity`); (c) a safety eval suite
   (authority-escalation / skill-poisoning / self-preservation probes +
   `SAFETY_GATE_CONFIG`). `TraceSignals` is the EvidenceBundle atom; the future
   acceptance gate is `RegressionGate(SAFETY_GATE_CONFIG).compare(...)` —
   expressible, not wired. Adversarially reviewed; fixed the critical holes —
   authority classification is derived from the proposal's OWN target (not an
   attacker-supplied bundle); the structural authority floor is re-derived at
   the promotion gate (a self-labelled-LOW authority surface cannot dodge it);
   an invalid APPROVED `PromotionRecord` is **unconstructible** (a model
   validator raises — self-approval / auto-approval / missing attestation /
   suppressed monitoring / removed rollback all fail at construction); the
   behavioral authority trigger covers the soft signals (asks/warns/degraded/
   permission-error buckets); evidence must be digest-anchored. The no-loop
   line is pinned by `TestEvolutionContractsNotInRuntime`. **Constitution NOT
   amended** — Amendment 6 ratifies into v3.7 only after this implementation
   proves out (its Ratification Condition). Deferred: cryptographic attestation
   verification (presence is checked); binding a skill's `evidence_digest` to a
   live bundle (the trust chain runs through `PromotionRecord`); the running
   loop itself (deliberately out of scope).
   - Files: `contracts/evolution.py`, `contracts/skill.py`, `contracts/__init__.py`.
   - Tests: `tests/test_evolution_contracts.py`, `tests/test_skill_lifecycle.py`,
     `tests/eval/test_self_evolution_safety.py`; invariant
     `TestEvolutionContractsNotInRuntime`.
