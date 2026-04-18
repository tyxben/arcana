# Constitution Amendment 1 — Cognitive Primitives as Services

**Status**: Proposed (awaiting review)
**Date**: 2026-04-18
**Applies to**: `CONSTITUTION.md`
**Origin**: v0.7.0 planning discussion — "let me help the LLM release its remaining 20-30%"

---

## The Gap

The current constitution does excellent work on one axis: it **protects the LLM from being over-constrained**. The Four Prohibitions, Principle 4 (Strategy Leaps), Principle 6 (OS not Form Engine), Principle 7 (Outcome over Process) — all prevent the framework from turning into a cage.

But the current constitution is **silent on a second axis**: providing the LLM with services it can invoke to operate on its own reasoning state.

Chapter IV currently lists framework responsibilities:

- Providing capabilities (tools, models, memory, retrieval)
- Enforcing boundaries (budget, permissions, safety)
- Recording execution (trace, metrics, diagnostics)
- Organizing context (working set, compression, retrieval)
- Classifying errors (structured diagnostics)
- Enabling interaction (user-mid-execution communication)

Every item points **outward** — toward tools, toward user, toward storage. None point **inward** — toward the LLM's own reasoning process.

This creates a capability hole. The LLM works inside a context window, but has no way to:

- **Retrieve its own compressed history**: at turn 10 the LLM needs the full text of what it said at turn 3, but working set compression already reduced it to one line.
- **Protect conclusions from future compression**: the LLM just derived three critical results; without pinning, the working set may compress them on the next turn.
- **Branch reasoning without polluting context**: the LLM wants to test "if X, then Y" in a sandbox before committing to the line of argument.
- **Track assumptions that may get invalidated**: the LLM anchored several outputs on "user is on Python 3.11"; when that turns out wrong at turn 12, nothing flags the dependency chain.
- **Signal context preferences forward**: the LLM knows the next turn will need tool X's docs; currently it has no way to tell the working set.

These are **cognitive operations** — not external tools, not user-facing I/O. They are the LLM operating on its own working memory.

Workarounds today are prompt-level hacks: "please repeat critical conclusions" in the system prompt, or forcing the LLM to re-state assumptions each turn. These burn tokens to approximate what should be a first-class runtime service.

---

## The Principle

### Principle 9: Cognitive Primitives as Services

The runtime provides services not only for the world external to the LLM (tools, memory, trace), but also for the LLM's own reasoning state. The LLM may invoke these cognitive primitives to:

- **Recall** — retrieve earlier turn content bypassing working-set compression
- **Pin** — protect specified content from compression in future working sets
- **Branch** — open a sandboxed reasoning frame that can be committed or discarded
- **Anchor** — mark an assumption as provisional, so future invalidation can be surfaced
- **Hint** — signal preferences that the next working-set build should consider

These primitives are exposed as **intercepted tools** — the same mechanism as `ask_user`. The LLM sees them in its tool list, calls them by name; the runtime intercepts and services the call without going through `ToolGateway`.

**The LLM invokes these services by choice. The framework never compels their use, never prescribes when to use them.**

### Why this is not a form engine

A form engine says: "follow this structure." A service says: "this is available if you want it."

Cognitive primitives never appear in the framework's default path. The LLM's base behavior is unchanged — it reasons, calls tools, answers. If it never invokes a cognitive primitive, the runtime produces an identical result to a runtime without them.

The primitives expand the LLM's **optional capabilities**, they don't alter the required flow.

### Why this is not orchestration

An orchestration mechanism dictates *sequence* or *topology*. Cognitive primitives dictate neither. They operate on reasoning state (recall / pin / branch), not on execution flow.

Calling `recall(turn=3)` doesn't say "now do X." It answers a question ("what did I say at turn 3"). The LLM decides what that answer implies.

### Why this serves the LLM

Every primitive is a **capability the LLM currently lacks**. Without `recall`, the LLM cannot access its own compressed history. Without `pin`, the LLM cannot protect its conclusions. These are not conveniences — they are the difference between "LLM has perfect memory of what it said" and "LLM is at the mercy of the working-set compressor."

The fundamental question of the constitution — *"Is this helping the LLM or constraining it?"* — answers unambiguously: **helping**. Every primitive expands what the LLM can attempt, never what it must do.

---

## Amendment to Chapter IV

Add to "The Framework Is Responsible For":

> - **Providing cognitive primitives**: services for the LLM to operate on its own reasoning state (recall compressed history, pin critical content, branch reasoning, anchor assumptions, hint future context). These services are available for the LLM to invoke; the framework never compels their use.

Add to "The Inviolable Rules":

> **The framework never decides when or how the LLM uses cognitive primitives.** Offering a service is not prescribing its use. A cognitive primitive in the tool list is a door the LLM may open; it is not a corridor the LLM must walk.

---

## Constitutional check against existing rules

| Rule | Assessment |
|------|-----------|
| Prohibition 1 (No Premature Structuring) | ✅ Primitives don't structure execution; they only service explicit LLM requests |
| Prohibition 2 (No Controllability Theater) | ✅ Primitives improve outcomes (less context loss), not process visibility |
| Prohibition 3 (No Context Hoarding) | ✅ Primitives default off (opt-in via `RuntimeConfig`); tool schemas only loaded when enabled |
| Prohibition 4 (No Mechanical Retry) | ✅ No retry logic involved |
| Principle 1 (Direct by Default) | ✅ Direct-answer path unaffected; primitives are agent-loop extras |
| Principle 2 (Context as Working Set) | ✅ Extends working-set control; Identity/Task/Working/External layers unchanged |
| Principle 3 (Tools as Capabilities) | ✅ Each primitive is a declared capability with clear affordances |
| Principle 4 (Allow Strategy Leaps) | ✅ Gives LLM more ways to leap, never forces any |
| Principle 5 (Actionable Feedback) | ✅ Failure modes (e.g. `recall(turn=999)` → not found) produce structured errors |
| Principle 6 (Runtime as OS) | ✅ This IS the OS analogy deepened — primitives are like syscalls |
| Principle 7 (Outcome over Process) | ✅ Improves result quality via better memory management |
| Principle 8 (Agent Autonomy) | ✅ Each agent in a pool manages its own cognitive state independently |

---

## What this rules out

Principle 9 **does not** authorize:

1. **Automatic invocation** — the framework never calls a cognitive primitive on behalf of the LLM. All invocation is an explicit LLM tool call, recorded in trace.
2. **Prescribed usage patterns** — the framework never hints "consider calling recall here," never enforces "pin before branch" ordering, never evaluates whether the LLM used a primitive appropriately.

(Corollaries: no primitive-in-prompt injection, no cross-agent primitive sharing, no framework judgment of primitive effectiveness — these all follow from the two rules above.)

---

## Rollout strategy

This amendment is proposed alongside v0.7.0 Cognitive Primitives Release. Rollout order:

1. Amendment accepted → `CONSTITUTION.md` updated with Principle 9 + Chapter IV additions
2. v0.7.0 implements first two primitives (recall, pin) — MVP subset
3. v0.8.0 adds branch + anchor based on v0.7.0 feedback
4. v0.9.0 adds hint (last, requires working-set feedback channel)

Each primitive's introduction requires constitutional re-check against this amendment's "does not authorize" list.

---

## The Fundamental Test

For every cognitive primitive proposed after this amendment:

1. Does the LLM call it, or does the framework call it?
2. Does it expand what the LLM can do, or constrain how the LLM works?
3. Could a perfectly capable LLM produce correct results without ever using this primitive?

If all three answer correctly (LLM / expand / yes), it belongs. Otherwise refuse it.

---

*This amendment, if accepted, becomes part of the living constitution with the same weight as the original document.*
