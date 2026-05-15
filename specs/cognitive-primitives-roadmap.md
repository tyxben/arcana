# Cognitive Primitives Roadmap — `branch` / `anchor` / `hint`

**Status**: Draft (2026-05-08)
**Owns**: the load-bearing design proposal for the three cognitive primitives that Constitution v3.5 §Principle 9 lists but has not yet shipped — `branch`, `anchor`, `hint`. Defines semantics, contracts, runtime integration, tracing, and constitutional analysis for each.
**Supersedes / cross-references**:
- `specs/v0.7.0-cognitive-primitives.md` — shipped MVP (`recall` / `pin` / `unpin`); this doc reuses its design invariants verbatim.
- `specs/constitution-amendment-1-cognitive-primitives.md` — the founding argument for Principle 9.
- `CONSTITUTION.md` v3.5 — Principle 9 + Chapter IV inviolable rules for cognitive primitives, including passive surfacing as labeled evidence after explicit arming.
- `specs/constitution-amendment-4-cognitive-passive-surfacing.md` — the constitutional carve-out that permits `anchor_status`-style passive surfacing without letting the runtime control the LLM's strategy.

This doc is a design proposal, not a release plan. Each primitive's mechanism is the load-bearing contribution; sequencing, surface freezes, and version targeting belong to a separate `vX.Y.Z` spec once a primitive is selected for implementation.

---

## 0. Reusing the v0.7.0 invariants

Every primitive in this doc inherits the five invariants from `specs/v0.7.0-cognitive-primitives.md` §"Design invariants":

1. **Intercepted-tool mechanism** — same pattern as `ask_user` / recall / pin. Dispatched by `ConversationAgent._execute_tools` before `ToolGateway`.
2. **Opt-in via `RuntimeConfig.cognitive_primitives`** — adding `"branch"` / `"anchor"` / `"hint"` to the list enables the corresponding tool schema in the LLM's tool list. Default empty.
3. **Full trace emission** — every invocation emits a `COGNITIVE_PRIMITIVE` event with `metadata = {"primitive", "args", "result"}`. The existing `EventType.COGNITIVE_PRIMITIVE` is reused; no new event types are required for branch/anchor/hint at the primitive-call level (specific sub-events called out per primitive below).
4. **Local to a session** — `AgentPool` does not share cognitive state. Each `ChatSession` owns its own state for every primitive.
5. **Never auto-invoked** — framework never calls any primitive on the LLM's behalf, never injects hint text into prompts, never evaluates whether the LLM used a primitive correctly.

Surface conventions inherited:
- Pydantic request/result types live in `arcana.contracts.cognitive`.
- Tool specs (`*_SPEC` constants) and a `*Handler`-extension live in `arcana.runtime.cognitive`.
- Errors are returned as **structured** `*Result` objects — never raised as exceptions (Principle 5).
- Naming: tool name is a verb (`branch`, `anchor`, `hint`), request type is `<Verb>Request`, result type is `<Verb>Result`.
- ID minting: opaque short handles (`b_<hex8>`, `a_<hex8>`, `h_<hex8>`) returned to the LLM for follow-up calls. Same shape as `pin_id = f"p_{uuid4().hex[:8]}"`.

---

## 1. `branch` — Hypothetical Reasoning Frame

### 1.1 Semantics

`branch` opens a **named, sandboxed reasoning frame** that the LLM can fill with exploratory turns without polluting main session state. The frame can later be `commit_branch`-ed (selected content folded into main context as a pinned summary) or `discard_branch`-ed (the frame and its trace evidence stay, but no content is folded into main). While a branch is open, all subsequent assistant turns and tool calls are tagged `branch_id=<id>` until the LLM explicitly closes the branch or a `branch_max_turns` ceiling is hit.

Problem it solves that recall/pin/unpin don't:
- **recall** answers "what did I say?" — read-only.
- **pin** answers "keep this verbatim" — write to working set, but only for material already produced.
- **branch** answers "let me try a hypothesis without it costing my main reasoning." That requires a *write-isolation boundary*: turns produced inside the branch are visible to the LLM during the branch and then either folded (commit) or set aside (discard). recall/pin/unpin have no isolation primitive — every assistant turn lands in the same conversation log.

Concretely:
- The LLM is mid-task, considers an alternative approach. Today: explore inline, contaminating main context with a possibly-wrong line of reasoning that compression then has to triage. With branch: open frame, explore, commit only the conclusion or discard wholesale.

This is the only one of the three roadmap primitives that introduces **execution-flow-visible state** (turns are tagged differently). That makes it the most constitutionally sensitive — see §1.5.

### 1.2 Surface

```python
# arcana.contracts.cognitive

class BranchOpenRequest(BaseModel):
    """Open a sandboxed reasoning frame."""
    label: str | None = None
    purpose: str | None = None  # LLM's own one-line rationale, traced
    max_turns: int | None = None  # caller-suggested ceiling; runtime caps it

class BranchOpenResult(BaseModel):
    """Structured result of branch open.

    On success: ``opened=True`` and ``branch_id`` is the opaque handle for
    follow-up calls. On refusal: ``opened=False`` plus ``reason`` (a parent
    branch is already open — branches are non-nesting in this MVP — or a
    cap was exceeded).
    """
    opened: bool
    branch_id: str | None = None
    label: str | None = None
    purpose: str | None = None
    parent_turn: int = 0  # main-session turn the branch was opened at

    # Populated on refusal
    reason: str | None = None
    active_branch_id: str | None = None
    suggestion: str | None = None


class BranchCommitRequest(BaseModel):
    """Fold a branch's salient content back into main as a pinned summary."""
    branch_id: str
    summary: str  # LLM authors this; framework does not summarize for it
    pin_summary: bool = True  # default: commit produces a pin


class BranchCommitResult(BaseModel):
    """Structured commit result. The LLM authored ``summary``; the framework
    only persists it. If ``pin_summary`` and the pin budget rejects it, the
    commit still succeeds (frame is closed, summary is appended to main as
    an unpinned message) and ``pin_refused`` carries the diagnostic — the
    LLM can then ``pin`` shorter content explicitly.
    """
    committed: bool
    branch_id: str
    summary_pin_id: str | None = None
    pin_refused: PinResult | None = None
    note: str | None = None


class BranchDiscardRequest(BaseModel):
    branch_id: str
    reason: str | None = None  # LLM's own rationale, traced


class BranchDiscardResult(BaseModel):
    discarded: bool
    branch_id: str
    note: str | None = None


class BranchEntry(BaseModel):
    """Session-local record of one branch frame.

    Like ``PinEntry``, this is mutable bookkeeping owned by ``BranchState``.
    Branch turns are stored separately from the main turn log so a discard
    does not have to rewrite history.
    """
    branch_id: str
    label: str | None
    purpose: str | None
    parent_turn: int
    opened_turn: int
    closed_turn: int | None = None
    status: Literal["open", "committed", "discarded"] = "open"
    branch_turn_log: list[list[dict[str, Any]]] = Field(default_factory=list)
    summary: str | None = None
    summary_pin_id: str | None = None


class BranchState(BaseModel):
    """Session-local registry of branches. Same shape as ``PinState``."""
    entries: list[BranchEntry] = Field(default_factory=list)
    active_branch_id: str | None = None  # at most one open branch (MVP)
```

Three tool names — `branch_open`, `branch_commit`, `branch_discard` — keeps each call unambiguous in the trace. `COGNITIVE_TOOL_NAMES` is extended to include all three. Open question on naming below.

### 1.3 Runtime integration

Affected files:
- `src/arcana/runtime/cognitive.py`: extend `CognitiveHandler` with `handle_branch_open` / `handle_branch_commit` / `handle_branch_discard`. Add `self.branch_state: BranchState`.
- `src/arcana/runtime/conversation.py`: `_execute_tools` already routes via `is_cognitive_tool`. Add the three branch tool names to that set. `_handle_cognitive_tool_call` gains three new branches.
- `src/arcana/runtime/conversation.py`: track an "active branch" pointer on the agent. When a branch is open, `record_turn` writes to `BranchEntry.branch_turn_log` *and* the main turn log (the LLM still sees its reasoning in the working set during the branch). On commit/discard, post-close turns go to main only. **Contract**: the working set during an open branch is built from `main_log + branch_log` so the LLM sees its in-progress reasoning. On discard, those branch turns become invisible to subsequent working-set builds — `WorkingSetBuilder.abuild_conversation_context` filters them out via a new `branch_filter_ids` parameter.
- `src/arcana/context/builder.py`: `abuild_conversation_context` accepts an optional `branch_filter_ids: set[str] | None` (default None = no filtering) and, when set, drops messages whose `_branch_id in branch_filter_ids`. The agent calls this with the discarded branch IDs. `Message._branch_id` is set on assistant/tool messages produced inside an open branch (mirrors how `_pinned` / `_pin_id` are attached).
- `src/arcana/contracts/runtime.py`: `RuntimeConfig.branch_max_turns: int = 8` — hard ceiling on a branch. Hitting it auto-closes the branch as `discarded` (with structured result the LLM sees on its next turn via tool result). This is a budget concern, not a strategy concern — Principle 6 OS, not Form Engine.

`commit_branch` produces an LLM-authored `summary` (the LLM provides the text in the request). The framework appends that summary as a system-role message to main and (if `pin_summary=True`) routes it through the existing `handle_pin` path so the same budget cap applies.

### 1.4 Tracing

Reuse `EventType.COGNITIVE_PRIMITIVE`. Each open/commit/discard call emits one event with `metadata`:

```python
# branch_open
{"primitive": "branch_open", "args": {...}, "result": {...},
 "branch": {"branch_id": "b_...", "parent_turn": 5}}

# branch_commit
{"primitive": "branch_commit", "args": {...}, "result": {...},
 "branch": {"branch_id": "b_...", "summary_pin_id": "p_...", "branch_turns": 3}}

# branch_discard
{"primitive": "branch_discard", "args": {...}, "result": {...},
 "branch": {"branch_id": "b_...", "branch_turns": 3}}
```

Each `LLM_REQUEST` / `LLM_RESPONSE` / `TOOL_CALL` produced inside a branch carries `metadata["branch_id"]` = the branch id. This lets `arcana trace show <run_id> --branch <id>` reconstruct branch turns post-hoc and the existing `pool-replay` tooling distinguish branch turns from main turns. The `TraceEvent` schema is unchanged — the tag lives in `metadata` (same precedent as `source_agent` from v0.8.0).

### 1.5 Constitutional analysis

**Goal**: give the LLM write-isolation for hypothetical reasoning, so an exploration that turns out wrong does not cost main-context tokens for the rest of the session.

**Mechanism candidates**:

❌ **Sub-`ChatSession` with full history fork** (the v0.7.0 spec sketch). Reject. Forking a session means duplicating provider state, separate budget tracking, separate pin state. That's coordination infrastructure ("how do you replay a sub-session?") which violates Principle 6 (Runtime as OS, not Form Engine) — building sub-conversation orchestration into the framework is exactly what AutoGen does. It also flirts with Prohibition 1 (No Premature Structuring) by codifying a particular branching topology (parent/child).

⚠️ **Auto-summarize on commit** (framework computes the merge content). Reject. Framework deciding what content survives a branch *is* the framework dictating reasoning state. Violates Principle 9's inviolable rule: "the framework never decides when or how the LLM uses cognitive primitives, never evaluates whether the LLM used them appropriately." A commit-time auto-summary is the framework picking one summary over another.

⚠️ **Nestable branches** (open inside open). Reject for MVP. Same family as auto-summary — once you nest, the runtime owes the LLM a tree of branch state, naming conventions for nested ids, and rules about commit propagation. That is structuring the LLM's reasoning, not servicing it. MVP is a single open frame at a time; the LLM that wants depth can `commit` then `open` again.

✅ **In-place tagging + LLM-authored commit summary**. Adopt. Branch turns sit in the same conversation log but carry `_branch_id`; the working-set builder optionally filters by ID. The LLM authors `summary` on commit. The framework holds plumbing (frame state, turn tagging, optional pinning of the summary) — the LLM holds strategy (when to branch, what content matters at commit, whether to pin the summary).

Per-prohibition / per-principle:

| Rule | Verdict | Reason |
|------|---------|--------|
| Prohibition 1 (No Premature Structuring) | ✅ | Branch is opt-in; framework imposes no required sequence; `branch_max_turns` is a cap, not a script. |
| Prohibition 2 (No Controllability Theater) | ✅ | Branch tags exist only in trace metadata, not as user-facing dashboards. The point is reasoning quality, not pretty diagrams. |
| Prohibition 3 (No Context Hoarding) | ✅ | Discarded branches' content is *removed* from future working-set builds via filter; commits produce one short pinned summary; budget cap (`pin_budget_fraction`) still enforced via reuse of pin path. |
| Prohibition 4 (No Mechanical Retry) | ✅ | No retry semantics; branch is exploratory not retry. |
| Principle 4 (Strategy Leaps) | ✅ — load-bearing | Branch *expands* leap options: lets the LLM try-then-revert without forcing N more turns. The framework never says "you should branch here." |
| Principle 6 (OS, not Form Engine) | ✅ | Three syscall-shaped tools (`open` / `commit` / `discard`); LLM authors all content; framework never composes prose. |
| Principle 9 + Chapter IV inviolable rule | ✅ | LLM invokes by choice; default-off; never auto-invoked. |

The mechanism passes the §"Fundamental Test" from amendment 1: LLM calls it / it expands what the LLM can do / a perfectly capable LLM could solve any task without it. ✅ on all three.

### 1.6 Open questions

- **Naming**: `branch_open` / `branch_commit` / `branch_discard` (three tools, unambiguous) vs `branch` with an `action` enum parameter (one tool, polymorphic). Three tools matches `pin` / `unpin` precedent (separate tool per write); one tool reduces tool-schema bloat. Lean toward three.
- **Branch turn numbering inside the branch**: do `recall` calls during a branch see branch-turn numbers (b1, b2, b3) or main turns (5, 6, 7)? Proposal: branch turns extend main numbering; the LLM can `recall(turn=N)` and get whatever sat at that turn whether main or branch. The trace metadata distinguishes them. But this means `RecallResult.messages` may contain branch-tagged content the LLM later discards — an explicit cross-primitive interaction worth user review.
- **Concurrent branches off the same parent**: forbidden in MVP (one open branch). Should the contract leave the door open for future relaxation, or hardcode "non-concurrent" in the contract docstring? Lean toward leaving the door open by accepting a future `parent_branch_id` field with default None.
- **Commit-while-branch-empty**: should `commit_branch` succeed on a branch with zero recorded turns? Lean yes (commits an empty summary if the LLM provides one); rejecting feels like the framework judging "did you actually do anything?"
- **Cost when budget is exhausted mid-branch**: the shared budget tracker may run out mid-branch. Currently runs propagate `BudgetError`; do we surface that as a structured tool result on the next branch turn, or let it bubble per current convention? Lean toward existing convention — branch is opt-in, doesn't deserve special budget semantics.

---

## 2. `anchor` — Provisional Assumption Tracking

### 2.1 Semantics

`anchor` records a **provisional assumption** the LLM is operating under. The LLM provides a claim string and an optional list of dependency tokens (turn refs, pin ids, free-text). The runtime stores the anchor in session-local state; on every subsequent assistant turn, the agent runs a **lightweight contradiction check** against new content. When a contradiction is detected, the framework injects a *structured anchor-status block* into the next working-set build — labeled clearly as "framework note: anchor `a_xyz` may be invalidated" — never as if the LLM said it. The LLM can then `unanchor` (the assumption is no longer load-bearing) or update the anchor.

Problem it solves that recall/pin/unpin/branch don't:
- **pin** keeps content visible. **anchor** tracks *contingent dependencies*: "this conclusion holds only if X." If X turns out false later, pin would still display the conclusion — the LLM has to re-derive every turn whether dependencies still hold. anchor is a passive dependency watcher.
- **recall** gives full text on demand; doesn't surface invalidation.
- **branch** is for explicit hypothetical exploration; anchor is for assumptions inside main reasoning.

This is the most constitutionally subtle of the three: it requires the framework to *do something on the LLM's behalf* (run a contradiction check). That risks Principle 4 (Strategy Leaps) and the inviolable rule. The mechanism below holds the line by making the contradiction check **non-prescriptive** — it surfaces a possible contradiction without instructing the LLM what to do.

### 2.2 Surface

```python
# arcana.contracts.cognitive

class AnchorRequest(BaseModel):
    """Record a provisional assumption."""
    claim: str
    depends_on: list[str] = Field(default_factory=list)  # free-text refs
    label: str | None = None


class AnchorResult(BaseModel):
    anchored: bool
    anchor_id: str | None = None
    label: str | None = None
    already_anchored: bool = False
    reason: str | None = None
    suggestion: str | None = None


class UnanchorRequest(BaseModel):
    anchor_id: str


class UnanchorResult(BaseModel):
    unanchored: bool
    anchor_id: str
    note: str | None = None


class AnchorStatus(BaseModel):
    """Per-anchor contradiction status, surfaced into working set as a
    framework note when ``possibly_contradicted=True``."""
    anchor_id: str
    label: str | None
    claim: str
    possibly_contradicted: bool = False
    contradiction_evidence: list[str] = Field(default_factory=list)
    detected_at_turn: int | None = None


class AnchorEntry(BaseModel):
    anchor_id: str
    claim: str
    claim_hash: str  # SHA-256 for idempotency, like pins
    depends_on: list[str]
    label: str | None
    created_turn: int
    status: AnchorStatus  # mutates as contradictions are detected


class AnchorState(BaseModel):
    entries: list[AnchorEntry] = Field(default_factory=list)
    # Methods: find_by_id, find_by_hash, add, remove, all_active(), with_status_changes_since(turn)
```

### 2.3 Runtime integration

Affected files:
- `src/arcana/runtime/cognitive.py`: `handle_anchor` / `handle_unanchor`; new method `evaluate_after_turn(turn, new_messages)` runs the contradiction check.
- `src/arcana/runtime/conversation.py`: after each turn's `record_turn`, call `cognitive_handler.evaluate_after_turn(...)`. Status changes are buffered for the next working-set build. A contradiction status change emits a `COGNITIVE_PRIMITIVE` trace event with `primitive="anchor_status"` (note: this is a *framework-emitted* primitive event, the only one in the cognitive family — see §2.5 constitutional analysis for why this is OK).
- `src/arcana/context/builder.py`: a new `set_anchor_state(state)` setter (parallel to `set_pin_state`). Active anchors with `possibly_contradicted=True` render as a system-role message clearly labeled `[framework note] anchor a_xyz: claim "...", possible contradiction at turn N: <evidence snippet>`. These blocks are not user-pinned, are excluded from compression, and are surfaced in `ContextDecision.decisions` with `reason="anchor_alert"`. Once the LLM `unanchor`s, the alert disappears from subsequent builds.

The contradiction check itself is the key design call. **MVP heuristic** is deliberately weak so the framework cannot be accused of judging the LLM's reasoning:

- Tokenize the claim into significant lemmas.
- For each new assistant message *and* each new tool result, compute lemma overlap with the claim.
- If overlap is high (≥ 0.5) AND the message contains a negation marker ("not", "no longer", "incorrect", "wrong", "actually", CJK equivalents), or contains a value that contradicts a numeric/string token in the claim — flag.
- The flag carries the evidence message index and a short snippet, never a verdict.

The check is per-turn, single-pass, no LLM call. Cost is negligible.

A **stronger** alternative is to use a small LLM call ("does this content contradict claim X?") — see open questions §2.6.

### 2.4 Tracing

- `anchor` and `unanchor` calls emit `COGNITIVE_PRIMITIVE` events with `primitive="anchor"` / `"unanchor"` (same shape as pin/unpin).
- A status change (going from `possibly_contradicted=False` → `True`) emits `COGNITIVE_PRIMITIVE` with `primitive="anchor_status"`, `metadata = {"anchor_id", "evidence", "detected_at_turn", "trigger_message_index"}`. **This is framework-emitted, not LLM-emitted.** That is a divergence from the v0.7.0 invariant "every primitive event corresponds to an LLM tool call." It is justified below; mark the divergence explicitly in the spec.
- `WorkingSetBuilder` rendering of an anchor alert emits no new event — it shows up in `CONTEXT_DECISION` like any other working-set block.

### 2.5 Constitutional analysis

**Goal**: give the LLM a way to track contingent assumptions so future invalidating evidence is surfaced rather than silently ignored.

**Mechanism candidates**:

❌ **Anchor as pure storage, no contradiction check**. Reject as too weak. The whole point is *invalidation surfacing* — without it, anchor is just a pin with metadata.

❌ **Strong contradiction check via an LLM call per turn**. Reject for MVP. Per-turn LLM contradiction check is (a) expensive on every turn (Prohibition 2 risk: looks rigorous, doesn't move outcomes if LLM is already paying attention), (b) adds a framework-side LLM call that itself becomes load-bearing — a runtime that runs an LLM check for the LLM is the runtime making strategy calls. Possibly acceptable as opt-in (`anchor_check_mode="llm"`), but not the default.

⚠️ **Inject the anchor list verbatim into every prompt** ("here are your active anchors, please re-check"). Reject. That violates the inviolable rule "the framework never hints at primitive use in prompts." The framework would be telling the LLM what to think about each turn.

✅ **Lemma-overlap + negation heuristic; surface as a labeled framework note when triggered**. Adopt. The check runs deterministically. The output is *evidence-shaped, not directive-shaped* — "claim X may be contradicted by message N at <snippet>" leaves the LLM to decide whether to update, unanchor, or ignore.

The framework-emitted event (`anchor_status`) is the load-bearing constitutional question. Justification:
- The check fires only against content the LLM itself produced (its own assistant messages and the tool results it requested). The framework is not introducing new information.
- The output is a labeled note ("[framework note] anchor X may be invalidated"), not a directive ("you should update X"). The LLM may ignore it.
- The user opted in by adding `"anchor"` to `cognitive_primitives` AND by the LLM calling `anchor()` to register an assumption. Two layers of opt-in.

Per-prohibition / per-principle:

| Rule | Verdict | Reason |
|------|---------|--------|
| Prohibition 1 (No Premature Structuring) | ✅ | No prescribed sequence; LLM decides when to anchor and how to react to a status change. |
| Prohibition 2 (No Controllability Theater) | ✅ | Anchor alerts only render when the heuristic fires; otherwise zero working-set cost. |
| Prohibition 3 (No Context Hoarding) | ✅ | Anchor entries stay outside the working set until contradiction triggers; the alert block is short and disappears on unanchor. |
| Prohibition 4 (No Mechanical Retry) | ✅ | Not a retry mechanism. |
| Principle 4 (Strategy Leaps) | ✅ — *with the heuristic, marginal* | Concern: the alert *nudges* strategy. Mitigation: the alert is evidence-shaped and labeled framework-note. The LLM remains free to ignore it; nothing in the runtime escalates if the alert is not addressed. |
| Principle 6 (OS, not Form Engine) | ✅ — load-bearing | Anchor is a syscall ("watch this and tell me if it breaks"), not a workflow ("you must verify anchors before proceeding"). |
| Principle 9 + Chapter IV inviolable rule | ⚠️ → ✅ | The framework *does* surface a note without an explicit LLM tool call. Constitutional only because (a) the user opted into "anchor", (b) the LLM authored the claim, (c) the note is evidence not directive. Document this explicitly. |

This primitive is the closest to the constitutional line of the three. The decision rule used in the v1.0.0-stability §3.5 analysis applies: separate goal from mechanism, accept any mechanism where the framework holds plumbing and the LLM holds strategy. Lemma overlap is plumbing; the LLM's response to the alert is strategy.

### 2.6 Open questions

- **Heuristic strength**: lemma overlap + negation is weak. False negatives are easy. Should the MVP ship with `anchor_check_mode="heuristic" | "llm" | "off"`, default `"heuristic"`, with an opt-in cheap LLM check available? Lean toward yes-but-future — MVP ships heuristic only, mode field reserved with a single accepted value.
- **Should the alert be a system-role message or a user-role message labeled "[framework note]"?** Anthropic and DeepSeek both reject back-to-back same-role; system-block injection mid-conversation may break some providers. Lean toward user-role with explicit framework-note prefix, mirroring how `[Run context]\n...` is injected as user content.
- **Anchor storage limits**: `MAX_ACTIVE_ANCHORS = 32`? Without a limit, a poorly-prompted LLM could anchor every claim. Pin has a budget cap; anchor needs a count cap. Lean: yes, cap at `RuntimeConfig.anchor_max_active = 32` with structured rejection.
- **Cross-anchor contradictions**: if anchor A claims "X = 3" and anchor B claims "X = 5", that itself is a contradiction. Detect or not? Lean: not in MVP. Detection would require the framework to compare anchors against each other, which is a step deeper than "watch new messages."
- **Scope of evidence surfaced**: snippet length? Anchor evidence carries up to 200 chars by default. Bikeshed.

---

## 3. `hint` — Working-Set Build Hint

### 3.1 Semantics

`hint` lets the LLM signal a **non-binding preference** for what the *next* working-set build should consider. The LLM passes a structured hint (e.g. `"include_tool_docs": ["shell", "http_get"]`, `"prioritize_topic": "migration plan"`, `"deprioritize_role": "user"`). The runtime stores hints in session-local state with a one-turn TTL by default. On the next `abuild_conversation_context` call, the builder reads the active hints and uses them to *adjust scoring weights* in `_relevance_score` / `_compress_with_relevance` — never to override budget caps, never to skip compression, never to dictate strategy. The hint is *advisory*: under tight budget, the builder can ignore it.

Problem it solves that recall/pin/unpin/branch/anchor don't:
- The LLM knows what the *next* turn needs that the working-set builder, which sees only goal keywords + role weights, cannot infer. Today: no channel from LLM to builder. With `hint`: a structured, traceable, expirable preference signal.
- Distinct from `pin`: pin is a binding "include this verbatim, period." Hint is "if you have budget, prefer this." Pin costs the pin budget; hint costs nothing (it's just scoring weights).

### 3.2 Surface

```python
# arcana.contracts.cognitive

class HintRequest(BaseModel):
    """Signal a preference for the next working-set build.

    All fields are optional and additive — multiple ``hint`` calls in one
    turn merge by union/sum. The runtime resolves each accepted hint type
    deterministically; unknown types are recorded but ignored.
    """
    include_tool_docs: list[str] = Field(default_factory=list)
    prioritize_topic: str | None = None
    deprioritize_topic: str | None = None
    prioritize_roles: list[str] = Field(default_factory=list)
    deprioritize_roles: list[str] = Field(default_factory=list)
    ttl_turns: int = 1  # how many *future* turns this hint applies to (default 1)
    note: str | None = None  # LLM rationale, traced


class HintResult(BaseModel):
    accepted: bool
    hint_id: str | None = None
    expires_after_turn: int | None = None
    accepted_keys: list[str] = Field(default_factory=list)
    ignored_keys: list[str] = Field(default_factory=list)  # unknown types
    reason: str | None = None  # populated on rejection (e.g. ttl out of bounds)


class HintEntry(BaseModel):
    hint_id: str
    request: HintRequest
    created_turn: int
    expires_after_turn: int


class HintState(BaseModel):
    entries: list[HintEntry] = Field(default_factory=list)
    # Methods: active_at(turn), add, gc(turn) — autogarbage by turn
```

There is intentionally no `unhint`. Hints expire by TTL. If the LLM wants to overwrite, it issues a fresh hint with the same keys. Rationale: revocation adds cognitive bookkeeping with no clear use case (the LLM can issue a counter-hint).

### 3.3 Runtime integration

Affected files:
- `src/arcana/runtime/cognitive.py`: `handle_hint(req, current_turn)` returns `HintResult`. Validation: `ttl_turns ∈ [1, RuntimeConfig.hint_max_ttl_turns]` (default 4); roles must be one of `{"user", "assistant", "tool", "system"}`.
- `src/arcana/runtime/conversation.py`: extend `_handle_cognitive_tool_call` for the new tool name. After accepting a hint, call `context_builder.set_hint_state(handler.hint_state)` (parallel to `set_pin_state`).
- `src/arcana/context/builder.py`:
  - New `set_hint_state(state)` setter on `WorkingSetBuilder`.
  - In `abuild_conversation_context` and the sync `build_conversation_context`, before relevance scoring, the builder calls `_active_hints(turn)` and applies adjustments:
    - `include_tool_docs`: passed up to the agent so it can select which tools to bind for the *next* request (the builder already accepts a `tool_token_estimate`; the agent reads `include_tool_docs` to bias `LazyToolRegistry` resolution). Hint-only — if the budget can't fit, normal lazy-tool logic takes over.
    - `prioritize_topic` / `deprioritize_topic`: extracted into keyword sets, added to / subtracted from `_goal_keywords` for this build only. Reset after the build.
    - `prioritize_roles` / `deprioritize_roles`: a per-role bonus/penalty added to the role-base score in `_relevance_score`.
  - `_relevance_score` adds the role bonus; `_extract_keywords` is unchanged but gets called against the topic strings to expand the keyword set transiently.
  - Hint application is recorded in `ContextDecision.decisions[*].reason` as `"hint_boosted"` / `"hint_demoted"` when a per-message decision was different from the no-hint baseline. (Implementation: compute baseline scores, compute hint-adjusted scores, mark only the messages whose fidelity level changed.)
  - After every build the builder calls `handler.hint_state.gc(turn)` so expired hints stop applying.
- `src/arcana/contracts/runtime.py`: `RuntimeConfig.hint_max_ttl_turns: int = 4`.

The hint is **purely advisory**: budget caps still bind, compression strategy is still resolved by `_resolve_strategy_name`, the builder never skips compression because of a hint. If a hint asks for `include_tool_docs=["X"]` but the tool budget is zero, the hint is ignored and `HintResult.accepted_keys` excludes the satisfied claim — but this is post-hoc; the immediate `HintResult` reports acceptance for valid syntax. (Open question on this — see §3.6.)

### 3.4 Tracing

- The `hint` call emits one `COGNITIVE_PRIMITIVE` event with `primitive="hint"`, `metadata = {"args", "result", "expires_after_turn"}`.
- Hint-driven adjustments in `WorkingSetBuilder` are surfaced via existing `ContextDecision.decisions[*].reason` strings (`"hint_boosted"` / `"hint_demoted"` / `"hint_tool_inclusion"`). No new trace event type.
- `ContextDecision` gains an additive field `applied_hint_ids: list[str] = Field(default_factory=list)` listing the hints that affected this build. (Additive minor-bump per §5 of the stability spec.)

### 3.5 Constitutional analysis

**Goal**: give the LLM a one-turn-forward preference channel to the working-set builder, so the LLM can express what it needs next without resorting to pinning (which is binding and costs budget).

**Mechanism candidates**:

❌ **Hint as binding directive** ("must include tool X next turn"). Reject. That overrides the budget enforcer's authority — the runtime owes its budget guarantee regardless of LLM preferences. Violates the "framework enforces boundaries" contract from Chapter IV. If the LLM wants binding, that is what `pin` is for.

❌ **Free-form hint string passed to the builder** ("the LLM said: please prioritize the migration plan"). Reject. Free-form means the runtime would have to interpret the hint, which is the runtime making strategy calls about what the LLM meant. Violates Principle 4 / the inviolable rule.

⚠️ **Open-ended dict of arbitrary keys**. Reject. Requires the runtime to grow handlers as new key types appear; gradient toward "hint became a DSL for context strategy" (Prohibition 1). Better: typed Pydantic model, fixed key set, additive grow-only.

✅ **Typed Pydantic request with a fixed key set, deterministic builder adjustments, advisory only**. Adopt. The contract is the schema. The runtime knows exactly what each accepted key maps to. Unknown keys are recorded and ignored (forward-compat).

The most subtle question for `hint` is: does an opt-in mechanism whose effects are only visible in compressed-message scoring count as "the LLM controlling its own context"? Or does it count as "the framework offering a knob the LLM happens to turn"?

Answer: the latter, *and that's fine*. The OS analogy from Principle 6 maps cleanly: a process can call `madvise(MADV_WILLNEED)` to advise the kernel about page access patterns. The kernel may or may not honor it. The process is not running the page replacement algorithm, but it is participating in it. `hint` is `madvise` for working sets.

Per-prohibition / per-principle:

| Rule | Verdict | Reason |
|------|---------|--------|
| Prohibition 1 (No Premature Structuring) | ✅ | Hint is opt-in, additive, default-empty; builder behavior unchanged when no hint is active. |
| Prohibition 2 (No Controllability Theater) | ✅ | Hint changes scoring weights; outcome is a different working set, not a different log. |
| Prohibition 3 (No Context Hoarding) | ✅ | Hints are session-local, TTL-expiring, no token cost in the working set itself. |
| Prohibition 4 (No Mechanical Retry) | ✅ | Not a retry mechanism. |
| Principle 4 (Strategy Leaps) | ✅ | Hint *is* the LLM leaping a strategy adjustment that the framework's scoring rules wouldn't otherwise reach. |
| Principle 5 (Actionable Feedback) | ✅ | `HintResult` distinguishes accepted from ignored keys; the LLM can adapt. |
| Principle 6 (OS, not Form Engine) | ✅ — load-bearing | Pure syscall shape: tell the kernel what you want, kernel decides what to do, you check the result. |
| Principle 9 + Chapter IV inviolable rule | ✅ | LLM invokes by choice; framework never auto-hints; framework never overrides budget for a hint. |

The mechanism passes the §"Fundamental Test": LLM calls / expands what LLM can do / a perfect LLM could solve any task without it.

### 3.6 Open questions

- **Reporting accepted-but-not-realized hints**: the hint API returns `HintResult.accepted=True` immediately at call time. But whether the *next* build actually used the hint is only knowable post-build. Should there be a follow-up structured surface (e.g. on the next `LLM_REQUEST`'s metadata, or on the next `ContextDecision`)? Lean toward `applied_hint_ids` on `ContextDecision` (see §3.4), and the LLM reads the decision via existing `last_decision` / `ContextReport` channels. No new tool call needed.
- **Keyset versioning**: the hint schema is fixed at v1. Adding `prioritize_pin_id: str | None` would be additive (minor bump). Removing or renaming a key is a major bump per the stability promise. Confirm scope of v1 keyset before shipping.
- **Conflict resolution**: simultaneous `prioritize_topic="A"` and `deprioritize_topic="A"` in two overlapping hints — adopt last-write-wins by `created_turn`? Or merge as zero adjustment? Lean: zero adjustment. Conflicting hints cancel; LLM responsible for not contradicting itself.
- **Hint-driven tool inclusion vs `LazyToolRegistry`**: the lazy registry already does goal-keyword tool selection. `include_tool_docs` partially overlaps. Merge logic: hint names are unioned with registry-selected names; budget cap still applies; over-budget tools are dropped per existing strategy. Possibly worth a separate sub-decision.
- **Should `hint` even ship?** Constitutionally cleanest of the three. But it solves a problem the user might not have if `pin` + lazy tool registry already covers 90% of cases. Validate against real workloads before committing — possibly defer to v0.10.x rather than v0.9.x.

---

## 4. Surface and stability impact

This doc proposes additions to the stable contracts surface. Each addition is **additive** (per `specs/v1.0.0-stability.md` §5), so each primitive's introduction is a minor bump.

### 4.1 New stable names per primitive

`branch`:
- `arcana.contracts.cognitive.BranchOpenRequest` / `BranchOpenResult`
- `arcana.contracts.cognitive.BranchCommitRequest` / `BranchCommitResult`
- `arcana.contracts.cognitive.BranchDiscardRequest` / `BranchDiscardResult`
- `arcana.contracts.cognitive.BranchEntry` / `BranchState`
- `RuntimeConfig.branch_max_turns: int = 8`

`anchor`:
- `arcana.contracts.cognitive.AnchorRequest` / `AnchorResult`
- `arcana.contracts.cognitive.UnanchorRequest` / `UnanchorResult`
- `arcana.contracts.cognitive.AnchorStatus` / `AnchorEntry` / `AnchorState`
- `RuntimeConfig.anchor_max_active: int = 32`

`hint`:
- `arcana.contracts.cognitive.HintRequest` / `HintResult`
- `arcana.contracts.cognitive.HintEntry` / `HintState`
- `RuntimeConfig.hint_max_ttl_turns: int = 4`
- `ContextDecision.applied_hint_ids: list[str]` (additive field)

### 4.2 No breaking changes

- `EventType.COGNITIVE_PRIMITIVE` is reused for all three primitives' events. No new enum members required. The `metadata.primitive` string discriminates (`"branch_open"`, `"branch_commit"`, `"branch_discard"`, `"anchor"`, `"unanchor"`, `"anchor_status"`, `"hint"`).
- `ContextBlock` does not need new fields. Hint adjustments live in scoring; anchor alerts use the existing block structure with a new `source` value (`"cognitive_anchor_alert"`).
- `MessageDecision.reason` strings grow (additive — `reason` is a free-form short string, not enum).

### 4.3 Implementation order proposal

This is a roadmap, not a release plan. But for sizing:

1. **`hint`** first — smallest blast radius, cleanest constitutional argument, lowest risk. Validates the typed-syscall pattern for builder feedback.
2. **`branch`** second — biggest user-visible feature, most likely to need iteration. Implementing it after `hint` proves out builder-state coupling (`set_hint_state` → `set_branch_state`).
3. **`anchor`** last — most constitutionally subtle (framework-emitted event), depends on heuristic tuning. Best built when the team has feedback from `branch` runs about what kinds of contradictions LLMs care about.

This ordering inverts the v0.7.0 spec's tier-2 list (which had branch / anchor / hint). Justification: weight-by-load-bearing, not weight-by-flashiness — hint is small and unblocks future builder feedback channels; branch is larger but well-bounded; anchor needs the most real-world signal to tune.

---

## 5. Cross-primitive interactions

| Primitive A | Primitive B | Interaction |
|------------|------------|-------------|
| `branch` | `pin` | A pin created inside an open branch is recorded in the *branch's* turn log. On commit, the pin survives. On discard, the pin is removed (call `unpin` on each). Document explicitly. |
| `branch` | `recall` | Inside an open branch, `recall(turn=N)` may return content tagged `_branch_id` for some N. The result is unfiltered — the LLM sees branch and main turns interleaved. Trace metadata distinguishes them. |
| `branch` | `anchor` | An anchor created inside a branch dies with the branch (commit *or* discard). On commit, the LLM can re-anchor via the summary if the assumption survives. |
| `branch` | `hint` | A hint with TTL > remaining branch turns persists past the branch close. Acceptable — hints affect *builds*, not *turns*. |
| `anchor` | `pin` | Anchoring pinned content is independently fine; the anchor watches the claim string, not the pin block. |
| `anchor` | `recall` | An anchor may reference a turn via free-text in `depends_on`; if the LLM `recall`s that turn and contradicts it, the anchor status updates next turn. |
| `hint` | `pin` | `hint.prioritize_pin_id` is **not** in the v1 keyset. Pins are already binding; hinting them is redundant. Open question §3.6 may revisit. |

---

## 6. Critical files for implementation (when each primitive is built)

- `src/arcana/contracts/cognitive.py`
- `src/arcana/runtime/cognitive.py`
- `src/arcana/runtime/conversation.py`
- `src/arcana/context/builder.py`
- `src/arcana/contracts/runtime.py`

---

## 7. Open questions across the three primitives

(Beyond per-primitive open questions in §1.6, §2.6, §3.6.)

1. **One spec or three?** This doc treats branch/anchor/hint as a single roadmap. If implementation is sequenced (§4.3), each primitive likely deserves its own focused `vX.Y.Z` spec (precedent: `v0.7.0-cognitive-primitives.md`). Lean: this doc as the umbrella, three smaller specs as work items kick off.
2. **Passive surfacing constitutional boundary**: resolved by Constitution v3.5 / Amendment 4. A framework-emitted `anchor_status` event is constitutional only after the user enabled `anchor` and the LLM explicitly armed the anchor, and only if the surfaced content is labeled framework-authored evidence rather than a directive or verdict.
3. **Multi-agent (`AgentPool`) story per primitive**: per v0.8.0, cognitive state is per-agent. Branch/anchor/hint should follow the same isolation. Confirm explicitly that `pool.add(..., cognitive_primitives=["branch"])` enables branch only for that agent and its branch state is invisible to siblings.
4. **CLI surface**: `arcana trace show <run_id> --cognitive` already exists for recall/pin/unpin. Extending it for branch (per-branch reconstruction), anchor (status timeline), and hint (per-build application list) is non-trivial. Worth scoping per-primitive, possibly batched in a v0.11.x trace CLI bump.
5. **Compatibility with `engine="adaptive"` (V1)**: V1 path doesn't use `WorkingSetBuilder.abuild_conversation_context` and doesn't have the cognitive handler integrated. Document that all three primitives are V2-only; calls under V1 should fall through to "tool not found" with a structured note.
6. **Demo / example coverage**: each primitive needs an `examples/demo_<name>.py` showing realistic use. Branch is straightforward (hypothetical exploration); anchor is harder to demo without crafted contradictions; hint may need a multi-domain workload to be visible.

---

## 8. Revision history

- **2026-05-08** — Draft 1. Scope: branch / anchor / hint design proposals, decomposed goal-vs-mechanism per v1.0.0-stability §3.5 style. Implementation-order recommendation: hint → branch → anchor.
- **2026-05-12** — Updated cross-references after Constitution v3.5 / Amendment 4 accepted the passive cognitive surfacing carve-out for `anchor_status`-style events.
