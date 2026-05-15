# Constitution Amendment 4 — Passive Cognitive Surfacing, Provenance, and Semantic Downgrade Honesty

**Status**: Accepted (2026-05-12) — landed in CONSTITUTION.md v3.5
**Date**: 2026-05-12
**Applies to**: `CONSTITUTION.md` — Principle 2 (Context as Working Set), Principle 6 (Runtime as OS), Principle 9 (Cognitive Primitives as Services), Chapter IV (Division of Responsibility), Chapter V (Contributor Compact)
**Origin**: Review of the `branch` / `anchor` / `hint` cognitive primitive roadmap after v1.0.0. The review found that `anchor` in particular introduces a new class of runtime behavior: the framework may surface a later status change after the LLM has explicitly armed a cognitive service. That is useful, but it needed a constitutional boundary before implementation.

---

## The tension

Principle 9 says cognitive primitives are services the LLM may invoke, and Chapter IV says the framework never decides when or how the LLM uses them. That was sufficient for `recall`, `pin`, and `unpin`: every primitive event corresponds directly to a tool call from the LLM.

The roadmap primitive `anchor` is different. The LLM may call `anchor(claim=...)`, and the runtime may later detect that new evidence possibly contradicts the anchored claim. If the runtime surfaces that status change, it is doing something after the original tool call.

Two bad readings are possible:

- **Too strict**: the framework may never surface anything unless the LLM asks again. Then `anchor` collapses into storage with metadata and loses the invalidation-surfacing behavior that makes it useful.
- **Too loose**: once a primitive is enabled, the framework may inject reminders, warnings, or corrections whenever it thinks they matter. That would let the runtime become an editor of the LLM's reasoning.

The amendment chooses a narrow middle: passive surfacing is allowed only after two opt-ins and only as labeled evidence.

---

## The amendment

### Edit 1 — Principle 2, context provenance

Add a corollary:

> **Context Provenance Is Mandatory:** Every context block must preserve where it came from and why it is present. User-authored content, assistant output, tool results, memory, compressed summaries, pinned content, framework notes, and provider/runtime diagnostics carry different authority. The working set may compress or drop content, but it must not blur authorship, hide whether content was summarized, or let a framework-authored note impersonate user intent, system policy, or an assistant conclusion.

Reason: future working sets increasingly mix user content, memory, compression summaries, pins, anchor alerts, and framework notes. The constitutional requirement is not just "include fewer tokens"; it is "do not launder authority while selecting tokens."

### Edit 2 — Principle 6, no silent semantic downgrade

Add a corollary:

> **No Silent Semantic Downgrade:** The runtime may adapt to provider capabilities, budget pressure, transport limits, and tool availability, but only silently when the meaning of the user's request and the caller's contract are preserved. If fallback, capability degradation, context compression, tool-surface reduction, or structured-output downgrade changes what can be expressed, validated, or guaranteed, the change must surface as structured feedback, trace evidence, or an explicit result field. Optimization can be transparent; semantic weakening cannot.

Reason: provider fallback, capability auto-degradation, lazy tool selection, and context compression are OS mechanics. They are valid only while they preserve the contract the caller reasonably believes they asked for.

### Edit 3 — Principle 9, passive surfacing carve-out

Add:

> Passive monitoring is permitted only after two opt-ins: the user enabled the primitive, and the LLM explicitly armed it by calling the primitive. For example, an `anchor` service may later surface a possible contradiction to the LLM because the LLM asked the runtime to watch that assumption. Such output must be labeled as framework-authored evidence, never as instruction, verdict, automatic correction, or hidden state mutation. The LLM decides whether the evidence matters.

Reason: this keeps `anchor` constitutional without giving the framework permission to nudge the LLM whenever it wants.

### Edit 4 — Principle 9, implementation status in the body

Add:

> Implementation status is not implied by this principle. As of v3.5, `recall`, `pin`, and `unpin` are implemented runtime services. `branch`, `anchor`, and `hint` remain roadmap primitives until their contracts and tests ship.

Reason: v3.2 clarified this only in revision history. That is too easy to miss because Principle 9 names all five primitives directly.

### Edit 5 — Chapter IV, new Inviolable Rules

Add:

> **Passive cognitive surfacing is evidence, not control.** After the LLM arms a cognitive primitive, the framework may surface passive status changes produced by that primitive (for example, a possible invalidation of an anchored assumption). It must label the surfaced content as framework-authored, explain the evidence that triggered it, and leave all interpretation and response to the LLM. It must not rewrite the LLM's conclusion, force a follow-up step, or treat silence as consent.

And:

> **The framework never silently weakens a caller-visible contract.** Provider fallback, capability downgrade, prompt compression, lazy tool selection, and cache optimization are allowed runtime mechanics. If they preserve semantics, they may stay invisible except in trace. If they weaken validation, remove a capability the LLM reasonably needed, alter output guarantees, or change the authority of context, they must be surfaced as structured feedback or auditable trace evidence.

Reason: the corollaries guide design; the Inviolable Rules give implementation review a hard stop.

---

## What this amendment permits

- An `anchor` primitive that surfaces a later `anchor_status` note after the LLM registered the anchor.
- Framework-authored context blocks such as `[framework note] anchor a_123 may be contradicted by turn 7: ...`, provided they are evidence-shaped and provenance-labeled.
- Provider capability auto-degradation when the outcome contract is unchanged, with trace evidence.
- Context compression and lazy tool selection that preserve provenance and leave enough trace evidence to debug why content or tools appeared.

## What this amendment forbids

- A runtime-injected reminder such as "you should check your anchors now" when the LLM did not arm an anchor.
- An anchor alert that says "your conclusion is wrong" rather than "this evidence may contradict claim X."
- Auto-updating, deleting, or rewriting an LLM-authored assumption because a heuristic fired.
- Silently downgrading structured output guarantees from schema validation to best-effort JSON if the caller receives no indication.
- Compressing a tool result into a summary that later appears indistinguishable from the original tool output.
- Hiding provider fallback when the fallback provider has materially different tool-calling or structured-output semantics.

---

## Implementation consequences

- `anchor` may reuse `EventType.COGNITIVE_PRIMITIVE` for `primitive="anchor_status"`, even though that event is framework-emitted rather than LLM-tool-call-emitted. The event must include the anchor id, the triggering evidence, and enough metadata to distinguish it from an LLM invocation.
- Context blocks introduced by framework services should carry explicit provenance in their content and, where contracts allow, in structured metadata / `ContextDecision` reasons.
- Future capability degradation code should distinguish semantic-preserving optimization from semantic-weakening fallback. The former may be trace-only; the latter must be visible to the caller or LLM.
- Constitutional invariant tests should grow when these features ship: provenance on framework notes, no silent structured-output downgrade, and passive cognitive surfacing only after explicit arming.

---

## Revision log entry

Appended to `CONSTITUTION.md` Revision History on acceptance (2026-05-12):

> **v3.5** (2026-05-12) — Clarify the boundary for advanced cognitive primitives and adaptive runtime behavior. Add mandatory context provenance under Principle 2, no silent semantic downgrade under Principle 6, and an Inviolable Rule allowing only opt-in passive cognitive surfacing as labeled evidence, not control. Move cognitive primitive implementation status into Principle 9 itself so roadmap names are not mistaken for shipped guarantees. See `specs/constitution-amendment-4-cognitive-passive-surfacing.md`.

Version bump: v3.4 → v3.5. This is a clarification plus two new Inviolable Rules, not a new principle. It narrows how future primitives may be implemented; it does not add new public API.
