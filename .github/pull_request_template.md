<!--
Thanks for contributing. Please fill in the sections below.
The constitutional compliance checklist is mandatory — see CONSTITUTION.md.
-->

## Summary

<!-- One or two sentences. What changed and why. -->

## Constitutional Compliance

Arcana is governed by `CONSTITUTION.md`. Every change must answer to it.
Tick what applies and explain anything you're unsure about — it's fine to say
"N/A, this PR doesn't touch X."

### The Four Prohibitions

- [ ] **No Premature Structuring** — I did not impose a fixed Thought/Action/
  Observation loop, mandatory planning step, or other process template the LLM
  must walk through before solving the task.
- [ ] **No Controllability Theater** — I did not add validation, guardrails, or
  output gates whose only purpose is to look careful. Every check has a real
  failure mode it prevents.
- [ ] **No Context Hoarding** — I did not pile artifacts into the context
  window "just in case." New fields are scoped to a layer that can compress
  or drop them when budget is tight.
- [ ] **No Mechanical Retry** — Any new error path classifies the failure
  (`ToolErrorCategory`) and only retries TRANSPORT / TIMEOUT / RATE_LIMIT.
  Validation, permission, logic failures surface to the LLM immediately.

### The Division of Responsibility

- [ ] **LLM produces facts, runtime produces assessment.** I did not have the
  runtime decide what the LLM "meant" or whether its answer was good. I did
  not have the LLM judge its own completion / failure / confidence.
- [ ] **Runtime as OS, not form engine.** Any new runtime service is a
  capability the LLM may invoke, not a step it must traverse.
- [ ] **User is never forced to interact mid-execution.** If I added an
  ask-the-user pattern, it falls back gracefully when no handler is provided
  and never blocks the run.

### Token & Capability Surface

- [ ] **Token surface stays honest.** New schemas / system-prompt additions
  are sized appropriately, and prompt-caching boundaries (Anthropic
  `cache_control`, OpenAI `cached_tokens`) are not invalidated by my changes.
- [ ] **Tool dispatch respects side-effects.** If I touched the tool batch
  path, write tools still run sequentially and read tools may run
  concurrently — see `tests/test_constitutional_invariants.py`.
- [ ] **Cognitive primitives stay opt-in.** I did not auto-expose
  `recall` / `pin` / `unpin` to the LLM in any default code path.
- [ ] **Framework notes preserve provenance.** If I added memory,
  compressed summaries, context blocks, runtime diagnostics, or future
  cognitive alerts, they are clearly labeled by source and cannot be
  confused with user intent, system policy, or assistant conclusions.
- [ ] **No silent semantic downgrade.** If provider fallback, capability
  auto-degradation, lazy tool selection, prompt compression, or structured
  output fallback can weaken the caller-visible contract, that weakening is
  surfaced as structured feedback, result data, or auditable trace evidence.
- [ ] **Passive cognitive surfacing is armed first.** If I added passive
  monitoring for a cognitive primitive, it only fires after the primitive was
  enabled and the LLM explicitly armed it; the surfaced content is evidence,
  not a directive or verdict.

## Test Plan

<!--
- What you ran (e.g. `uv run pytest tests/test_X.py`)
- Anything that requires a real provider key or external service
- Manual verification steps if this is a UI/CLI change
-->

## Public Surface Impact

<!--
- Does this PR change a name listed as stable in
  `docs/guide/stability.md` / `specs/v1.0.0-stability.md` §1?
- If yes: signature change? rename? removal? behaviour change visible
  to user code?
- If breaking: the CHANGELOG entry MUST include a `Migration` section
  with before/after code snippets. See v0.9.0 entry for the format.
- If non-breaking (additive only): note "additive — minor bump
  candidate" so the next release roll-up captures it.
- If untouched: write "N/A — no public surface changed."
-->

## Notes for Reviewers

<!-- Optional. Tradeoffs you considered. Things you're unsure about. -->
