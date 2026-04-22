# Constitution Amendment 2 — Collaboration Means, Not Guarantees

**Status**: Accepted — applied to `CONSTITUTION.md` v3.1 (2026-04-21)
**Date**: 2026-04-20 (drafted) / 2026-04-21 (accepted)
**Applies to**: `CONSTITUTION.md` — Principle 8 (Agent Autonomy in Collaboration)
**Origin**: v0.8.0 constitutional audit (2026-04-20). The audit found Principle 8's "can see what others have said" clause admits two incompatible readings; v0.8.0's implementation satisfies one and violates the other. Rather than pick a reading silently, we amend the clause to make the intent explicit.

---

## The Tension

Principle 8 today reads (emphasis added):

> The framework's role: ensure every agent gets its turn, stays within budget, and **can see what others have said**. The agents' role: decide what to say, when to agree, when to disagree, and when to declare the task complete.

Two incompatible readings of "can see":

**Strict reading.** "Ensure … can see" means the framework must guarantee that, when agent B takes its turn, agent A's prior output is present in B's prompt. Any collaboration path where B runs without A's output in its context violates Principle 8.

**Permissive reading.** "Ensure … can see" means the framework must make other agents' output *reachable* — through a communication mechanism the agents can invoke. The framework provides the channel; agents decide what to send, what to read, and when.

v0.8.0 ships the permissive reading: `AgentPool` provides `channel` (point-to-point + broadcast) and `shared` (key-value store); no agent's output is auto-injected into another agent's prompt. Under the strict reading, this is a Principle 8 violation. Under the permissive reading, it is the only coherent implementation.

---

## Why the strict reading cannot stand

The strict reading collapses on contact with the rest of the constitution.

**It manufactures a new Principle 8 violation.** To guarantee that agent B "sees" agent A's output, the framework must decide *which* output, *how much*, and *in what form*. Does B receive A's full transcript? Last message only? A framework-written summary? Every answer is a strategy decision — exactly what Principle 8 forbids the framework from making. The strict reading escapes one violation by committing a different one.

**It contradicts Principle 4 (Strategy Leaps) and Principle 6 (OS not Form Engine).** Auto-injecting one agent's output into another's context is a form of topology — the framework prescribing an information flow. Principle 4 says the framework must not prescribe strategy; Principle 6 says the framework is an operating system, not a prescriber of process shape.

**It forces the framework into the role of editor.** Context windows are finite. If the framework guarantees visibility, it must eventually decide what to keep and what to drop from agent B's view of agent A's history. That is an editorial judgment the LLM should own, not the runtime.

**It has no analogue elsewhere in the constitution.** Nothing else in the document is phrased as a guarantee that one agent's output reaches another agent's prompt. The rest of Principle 8 is about *providing infrastructure* — "coordination infrastructure," "communication channels," "budget allocation," "turn scheduling." Reading "can see" as a guarantee isolates one clause from the principle it belongs to.

---

## The amendment

**Before** (Principle 8, paragraph 3):

> The framework's role: ensure every agent gets its turn, stays within budget, and can see what others have said. The agents' role: decide what to say, when to agree, when to disagree, and when to declare the task complete.

**After**:

> The framework's role: ensure every agent gets its turn, stays within budget, and **is given the means to see what others have said** — name-addressed channels, shared context stores, or equivalent communication mechanisms the agents may invoke. The agents' role: decide what to say, who to say it to, what to read from whom, when to agree, when to disagree, and when to declare the task complete.

Two edits:

1. **"can see what others have said" → "is given the means to see what others have said"** — the framework's obligation is to provide the channel, not to guarantee the reception. This matches the rest of Principle 8, which is consistently about infrastructure.
2. **Agents' role gains "who to say it to, what to read from whom"** — making explicit that addressing and reading decisions belong to the agents, not the framework.

No other part of Principle 8 changes. The "no hierarchy by framework decree" paragraph and the planner/executor example remain exact.

---

## What this amendment is *not*

It is not permission to ship multi-agent features that leave agents mute. The amendment still requires the framework to provide *workable* means — the agents must be able to actually communicate, not just be told "you could communicate if you wrote your own transport." v0.8.0 satisfies this: `Channel` and `SharedContext` are first-class, documented, and available by default in every `AgentPool`.

It is not a retreat from Principle 8's intent. The intent — that agents, not the framework, own collaboration strategy — is reinforced. What changes is the framing: the framework's contribution is *enabling*, not *guaranteeing*.

It is not a license to invent new coordination topologies in the framework. Principle 4 (Strategy Leaps) and Principle 6 (OS not Form Engine) still apply. Future additions must remain infrastructure — new transport primitives, richer addressing, persistence — not policies about who talks to whom in what order.

---

## Practical consequences

**For v0.8.0 as shipped.** The audit finding resolves cleanly: `AgentPool`'s channel-plus-shared design is the canonical implementation of Principle 8, not a compromise against it. No code change is required.

**For future multi-agent work.** The amendment defines the shape of any future collaboration feature:

- ✅ Add a new channel type (typed broadcast, request/response, pub/sub) — infrastructure.
- ✅ Add persistence to `SharedContext` — infrastructure.
- ✅ Add an auditable message log for compliance — infrastructure.
- ❌ Auto-inject "the last thing the other agent said" into each agent's system prompt — topology.
- ❌ Framework-written summary of one agent's turn handed to the next — editorial judgment.
- ❌ Default round-robin "hand the baton" scheduler that carries state across agents — the `team(mode="shared")` pattern already marked for removal in v1.0.0.

**For the deprecation of `runtime.team()`.** The amendment reinforces the decision (tracked in `specs/v1.0.0-removals.md`) to remove `team()` without a compatibility shim. `team(mode="shared")` embeds a rounds counter and a fixed turn order — a framework-prescribed topology. Under the amended Principle 8, that implementation is not salvageable; it is the archetype of what the amendment rules out.

---

## Revision log entry

When accepted, append to `CONSTITUTION.md` Revision History:

> **v3.1** (2026-04-20) — Amend Principle 8 "can see what others have said" → "is given the means to see what others have said"; expand agents' role to include addressing and reading decisions. Clarifies that the framework's multi-agent obligation is to provide communication infrastructure, not to guarantee message reception. See `specs/constitution-amendment-2-collaboration-means.md`.

Version bump: v3.0 → v3.1 (semantic-versioning the constitution — this is a clarification, not a new principle).
