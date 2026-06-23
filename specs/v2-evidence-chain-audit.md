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
| F2 | High | Prompt-based tool fallback (`_generate_with_text_tools`) is a pseudo-capability: it asks the model to emit a JSON tool call in **text**, but the provider builds `LLMResponse.tool_calls` only from native `message.tool_calls`, and `_parse_turn` only reads `response.tool_calls`. A degraded provider shows tools but never executes them (No Controllability Theater). | pending |
| F4 | Low-Med | `_execute_tools` buckets ask_user → cognitive → gateway and appends results in that order, so a turn mixing built-in and gateway tools returns `results` (and TOOL_END events) out of input order. The LLM conversation is `tool_call_id`-matched (answers are correct); the returned-order contract is the wart. `call_many` itself is already order-preserving. | pending |
| F5 | Med | Eval gates on pass-rate / cost / tokens; `EvalResult` carries status/steps/tokens/cost. Self-evolution needs trace-derived signals (context loss, tool-error category, permission denial, provider degradation, prompt-snapshot replay diff) and golden-trace replay. | pending |

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
2. **F2 — fix or remove the prompt-based tool fallback.** Preferred: make it
   real — parse the text JSON back into `LLMResponse.tool_calls` in the provider
   so the agent executes it; cover with a test. If too fragile, remove it and
   surface an honest "provider does not support tools" error. No pseudo-capability.
3. **F4 — reassemble `results` in input order** at the end of `_execute_tools`.
4. **F5 — trace-derived eval metrics + golden trace replay.** Add the
   trace-level signals above to the eval outcome + gate.
5. **Self-evolution contracts** — `EvidenceBundle` / `EvolutionProposal` /
   `PromotionRecord` (Amendment 6). First auto-evolvable targets are
   low-authority only: skills, tool-affordance copy, context-strategy
   thresholds, eval cases, docs. Permissions and guardrails are never
   auto-applied.
