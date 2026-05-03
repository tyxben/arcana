# Constitution Amendment 3 — Multi-Agent OS: Transport Independence and Supervision Neutrality

**Status**: Accepted (2026-05-03) — landed in CONSTITUTION.md v3.4
**Date**: 2026-04-28 (drafted), 2026-05-03 (accepted)
**Applies to**: `CONSTITUTION.md` — Principle 8 (Agent Autonomy in Collaboration), Chapter IV (Division of Responsibility)
**Origin**: 2026-04-28 design review of Arcana's multi-agent surface against the "multi-agent era" thesis (cross-process agents, MCP / agent-to-agent protocols, supervision-tree pressure). The review found that Principle 8 as amended in v3.1 still admits readings that, taken silently, would fold either *transport semantics* or *supervision policy* into framework defaults — both of which contradict the rest of the constitution. Following the precedent set by Amendment 2, we amend rather than pick a reading silently.

---

## The Two Tensions

### Tension A — Transport assumption

Principle 8 today reads (emphasis added):

> The framework's role: ensure every agent gets its turn, stays within budget, and is given the means to see what others have said — **name-addressed channels, shared context stores, or equivalent communication mechanisms** the agents may invoke.

The clause is silent on whether those channels are in-process or cross-process. Today's implementation is in-process (`asyncio.Queue` inside `AgentPool`). MCP for tools and the emerging agent-to-agent (A2A) protocols make cross-process agents a near-term concern. Two readings:

**In-process reading.** The principle promises an in-process pool only. Cross-process collaboration is out of scope until the constitution says otherwise.

**Transport-agnostic reading.** "Communication mechanisms" abstracts over transport. In-process queues, stdio MCP, HTTP/gRPC RPC, and message-bus brokers are all valid implementations as long as the addressing semantics are preserved.

Shipping MCP / A2A integration without choosing explicitly would silently pick the transport-agnostic reading — exactly the failure mode Amendment 2 was written to prevent.

### Tension B — Supervision void

Principle 8 says nothing about what happens when an agent in a multi-agent session fails. Chapter IV ("Inviolable Rules") forbids the framework from deciding strategy, but supervision (cancel siblings on first failure, restart on transient error, isolate-and-continue) sits ambiguously between strategy and infrastructure. Erlang/OTP-style supervision trees are an attractive default, and that attraction is the danger:

- If the framework picks a default supervision policy, it is making a topology decision on behalf of users — a framework-decree pattern Principle 8 forbids.
- If the framework picks no default and lets exceptions vanish, it is failing the auditability obligation in Chapter IV ("Recording execution").

The current `AgentPool` exposes neither — failure handling is whatever the user happens to write in their orchestration code. This is technically permissive but constitutionally undefined.

---

## Why neither tension can be left to interpretation

**For transport (Tension A):** Cross-process collaboration introduces failure modes that don't exist in-process — partial delivery, network partitions, identity / authentication, serialization mismatches. Without a constitutional anchor, those failure modes will accrete ad-hoc in the implementation, and the LLM will eventually receive raw transport errors instead of structured feedback. That violates Principle 5. The framework must own *transport mechanics* (delivery, retry-of-the-byte-stream, dedup at the wire level) and must surface *transport-class failure* to the LLM as `ToolErrorCategory.TRANSPORT` style structured feedback — not as opaque exceptions and not as silent message loss.

**For supervision (Tension B):** Cancellation, restart, and failure-propagation are *coordination policy*. The same constitutional pattern that forbids the framework from deciding turn order applies: deciding what happens to agent B when agent A crashes is deciding the topology. The framework provides the mechanisms (task groups, cancel scopes, error propagation channels). The user — through orchestration code, not through framework defaults — picks the policy. A pool with no user-defined policy must fail in a predictable, audit-friendly way: errors propagate to the caller, no sibling is auto-cancelled, no agent is auto-restarted. *Predictably nothing* is a valid policy; *silently something* is not.

---

## The amendment

### Edit 1 — Principle 8, transport-agnostic clarification

**Before** (paragraph 3):

> The framework's role: ensure every agent gets its turn, stays within budget, and is given the means to see what others have said — name-addressed channels, shared context stores, or equivalent communication mechanisms the agents may invoke.

**After**:

> The framework's role: ensure every agent gets its turn, stays within budget, and is given the means to see what others have said — **addressable communication primitives** (name-addressed channels, shared context stores, or equivalent), **whose transport may be in-process, cross-process, or remote**. The framework owns transport mechanics (delivery, identity, serialization, wire-level retry); the LLM owns what to send, who to address, and how to react when a send returns a structured failure.

The substantive change: communication primitives are abstract; transport is named explicitly as a framework concern; transport-class failure is an LLM-visible event with structured feedback, not a swallowed exception or a raw error string.

### Edit 2 — Chapter IV, new Inviolable Rule on supervision

Append to *The Inviolable Rules*:

> **The framework never imposes a default supervision policy on multi-agent sessions.** Supervision — what happens to siblings when one agent fails, when to retry, when to escalate — is a coordination strategy decision that belongs to the user's orchestration code. The framework provides the mechanisms (task groups, cancel scopes, structured failure propagation) but ships no default policy. A pool that hits an unhandled failure with no user-defined policy fails *open*: the error propagates to the caller, siblings are not auto-cancelled, no agent is auto-restarted. Predictably nothing is a valid framework default; silently something is not.

No new principle. Both edits are clarifications that protect existing principles (Principle 8, Principle 5, Chapter IV "framework never decides strategy") under multi-agent-era pressure.

---

## What this amendment is *not*

It is not permission to ship A2A or MCP-as-agent transport in v1.0.0. The amendment opens the door; the implementation work is separate, must be designed under the same scrutiny as in-process channels, and must respect the v1.0.0 stable-surface gate. `arcana.contracts.mcp` exists today as a contract sketch; promoting it to a production transport requires its own spec.

It is not a retreat from Principle 8's intent. Cross-process agents do not get *more* framework authority than in-process agents. They get the same authority — including the same prohibition on framework-decreed hierarchy, framework-prescribed handoff schemas, and framework-injected context.

It is not a license to add a `pool.handoff(from, to, context=...)` API. Handoff is a strategy decision the LLM makes by sending a message with a recipient. Convenience APIs that bake in "what to copy from A to B" are framework-as-editor (already ruled out by Amendment 2). Handoff observability — *recognizing* a handoff pattern in the trace for debugging — is acceptable; *imposing* a handoff schema is not.

It is not a green light for default supervision. `pool.task_group()`, `pool.cancel_scope()`, `pool.fail_fast=True` may be added as **opt-in** primitives the user composes. None may be the default. The amendment exists precisely to keep that boundary visible.

---

## Practical consequences

**For Tension A (transport).**

- ✅ `Channel`'s in-process implementation continues unchanged; `Channel` is now understood as one implementation of an abstract `addressable communication primitive`, not the only one.
- ✅ Future cross-process transports (MCP-over-stdio, HTTP-RPC, message-bus adapters) ship behind the same `Channel`-level interface; addressing semantics are preserved.
- ✅ Wire-level transport failures surface to the LLM via `ToolErrorCategory.TRANSPORT` (same category already in use for tool gateway transport failures) — uniform structured feedback across local/remote.
- ❌ Auto-retrying a failed cross-process send N times before returning to the LLM. That is mechanical retry (Prohibition #4). One transport-level retry of the byte stream (TCP-style) is acceptable; replaying the whole agent-level message because the LLM's "intent" probably hasn't changed is not.
- ❌ Wrapping cross-process agents in a synthetic "remote agent" abstraction that hides their identity from the LLM. The LLM addresses the same way it does for in-process agents (by name); transport is invisible to the addressing layer but transport-class failure is visible to the LLM.

**For Tension B (supervision).**

- ✅ `AgentPool` documents its no-default-supervision contract: when a `ChatSession` inside the pool raises, the exception propagates to the awaiting caller. Siblings continue running until the caller cancels them.
- ✅ `pool.task_group()` and similar opt-in primitives may be added; the user wires them up explicitly.
- ✅ Trace records all agent failures with structured cause; per-agent stats (`pool.stats()`) expose cost/turn count/failure events per agent for debugging.
- ❌ A `Pool(fail_fast=True)` constructor *default*. It can exist as a flag, but the default must be predictable failure propagation, not auto-cancellation.
- ❌ Background reaper coroutines that restart "crashed" agents. Restart is a strategy decision — it lives in user code, not the framework.

**For trace and per-agent observability.**

- Trace already covers single-agent execution. Multi-agent trace must add: (a) channel send/receive as first-class events, (b) per-agent token / cost / turn-count slices, (c) handoff *recognition* (a passive trace heuristic, not a runtime hook). All of this is permitted under Chapter IV's "Recording execution" obligation and is reinforced — not blocked — by this amendment.

**For `arcana.contracts.trace.AgentRole`.**

- The enum (`PLANNER`, `EXECUTOR`, `CRITIC`, …) was introduced for the deprecated `team()` flow. Under Principle 8 + Amendment 2 + this amendment, agent identity is free-form (the agent's name in the pool); fixed roles are framework-prescribed topology. Pre-v1.0.0 cleanup must replace the enum with a free-form `agent_name: str` in trace events, or scope the enum to the (deprecated) `team()` code path so it doesn't leak into the v1.0.0 stable surface.

**For cross-agent context handoff.**

- Out of scope for this amendment. The follow-on (Amendment 4 candidate) is to extend Principle 9's cognitive-primitives model: an agent that wants to share its compressed working set with another agent invokes a primitive (e.g., `export_context(name=…)`); the receiving agent invokes `recall(name=…)`. Both decisions sit with the LLM, not the framework — the same constitutional pattern that gave us `recall` / `pin` in Amendment 1. This amendment does not introduce that primitive; it only ensures that the path to introducing it later is consistent with the multi-agent OS framing established here.

---

## Revision log entry

Appended to `CONSTITUTION.md` Revision History on acceptance (2026-05-03):

> **v3.4** (2026-05-03) — Amend Principle 8 to make communication primitives explicitly transport-agnostic (in-process, cross-process, or remote), with transport mechanics owned by the framework and transport-class failures surfaced to the LLM as structured feedback. Add a Chapter IV Inviolable Rule: the framework never imposes a default supervision policy on multi-agent sessions; pools fail open (errors propagate to caller, siblings not auto-cancelled, no auto-restart) absent user-defined policy. See `specs/constitution-amendment-3-multi-agent-os.md`.

Version bump: v3.3 → v3.4 (clarification + new Inviolable Rule, no new principle — semantic-versioning the constitution treats this as a minor bump, matching Amendment 2's v3.0 → v3.1 pattern). Note: the original draft proposed v3.3, but v3.3 was already used for Chapter VI Stability Commitments (commit `33643c9`, 2026-04-30) before this amendment was reviewed. The substantive content is unchanged; only the version target was retargeted from v3.3 to v3.4.

---

## Implementation follow-up (post-acceptance, separate PRs)

The amendment opens doors but does not implement them. Concrete work remaining:

- **`src/arcana/multi_agent/team.py::TeamOrchestrator` + `AgentRole` enum** (`src/arcana/contracts/trace.py`). The amendment's Practical Consequence in §103-105 flagged this for "pre-v1.0.0 cleanup," but v1.0.0 was cut (commit `bc15a51`, 2026-05-01) before this amendment was reviewed, and `Runtime.team()` (which the original cleanup language referred to) is a *different* surface from `TeamOrchestrator`. `TeamOrchestrator` is a Planner→Executor→Critic prescribed-topology orchestrator that violates Principle 8 + Amendment 2 + this amendment as a single piece. `arcana.multi_agent.*` is in the *internal — not stable* tier per `specs/v1.0.0-stability.md` §2, so removal does not require a major bump. Treat as a v1.x deprecation cycle: deprecate → next minor → remove.

- **Cross-process transport (MCP-as-agent / A2A).** The amendment opens the door; the implementation is its own spec. `arcana.contracts.mcp` exists today as a contract sketch. Promotion to a production transport must respect the addressable-by-name semantics and surface transport-class failures as `ToolErrorCategory.TRANSPORT`-shaped structured feedback.

- **Trace events for channel send/receive.** The amendment endorses (does not block) per-agent observability — channel send/receive as first-class trace events, per-agent token/cost/turn-count slices, handoff *recognition* (passive heuristic, not runtime hook). All permitted under Chapter IV "Recording execution"; none are required by this amendment.

- **`AgentRole` removal cascade.** `AgentRole` is consumed by `multi_agent/message_bus.py` (queue keying) and `multi_agent/team.py` (handoff schema). Replacement is `agent_name: str` everywhere. Deprecate the enum, ship a new `agent_name`-keyed message bus alongside, drop the old in the next minor.
