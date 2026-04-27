# Changelog

All notable changes to Arcana will be documented in this file.

## [Unreleased]

### Pre-v1.0.0 stability work

Tracked under `specs/v1.0.0-stability.md`. Each item below is non-
breaking and ships to users in a v0.9.x patch release as it lands.

#### Added — `ChatSession.seed_history()` (§3.1)

`ChatSession.seed_history(messages)` injects prior conversation
messages into a session at cold start — restoring a chatbot from
external storage without forcing user code through the private
`session._messages` attribute. Replaces the constitutional smell that
the existing `ChatSession.history` is read-only (dict copy) and there
was no public mutator: users were *required* to break encapsulation
to do something every chatbot needs.

- Accepts `Message` instances (canonical) or
  `{"role": str, "content": str}` dicts (convenience). Validates roles
  against `MessageRole`; rejects empty content and unknown types.
- System-role entries in the seed are **skipped** — the session's
  system prompt is owned by the constructor (`system_prompt=...`).
- Does not increment `turn_count` (those count turns *this* session
  executed; seed is pre-existing history).
- Calling twice extends — idempotency is the user's discipline.
- Allowed before *or* after `send()` — but the user owns the timing
  call; mid-stream seeding may collide with active compression state.
- Emits the new **`EventType.HISTORY_SEEDED`** trace event with
  `seed_count`, `role_counts`, `skipped_system`, `content_digest`
  (16-char canonical hash), and `message_count_after`. Auditability
  (Principle 5) is restored — the seed is not invisible.

#### Added — `EventType.HISTORY_SEEDED`

New `arcana.contracts.trace.EventType` member for session-lifecycle
audit. Backward compatible: existing event consumers ignore unknown
event types.

#### Added — `arcana.Message` / `arcana.MessageRole` top-level (§3.2)

`Message` and `MessageRole` are now re-exported at the top of the
`arcana` package, alongside `Runtime`, `Budget`, `ChatSession`, etc.
The canonical definitions stay in `arcana.contracts.llm`. The
top-level form is the **preferred** user import:

```python
# Preferred:
import arcana
msg = arcana.Message(role=arcana.MessageRole.USER, content="hi")

# Also supported (canonical):
from arcana.contracts.llm import Message, MessageRole

# Discouraged (works in CPython today, no stability promise):
from arcana.runtime.conversation import Message  # internal import
```

#### Cleanup — `arcana.runtime.conversation.__all__` (§3.2)

`arcana.runtime.conversation` now declares `__all__ = ["ConversationAgent"]`.
The module's docstring explicitly notes that `Message` / `MessageRole`
are imported there for internal use only and are not part of its
public surface. Explicit imports (`from arcana.runtime.conversation
import Message`) still work for backward compatibility but are no
longer advertised; `from arcana.runtime.conversation import *` will
now only yield `ConversationAgent`.

#### Added — `docs/guide/stability.md` (§3.3)

User-facing distillation of `specs/v1.0.0-stability.md` §1–2. Tells
users which Arcana imports are stability-promised and which are not,
includes the asymmetric cases (e.g. `arcana.contracts.*` is stable
under a sub-package; `arcana.runtime.conversation.Message` is *not*
stable even though it works today), summarises the post-v1.0.0
versioning rules, and explains the deprecation policy.

External feedback (Roboot, 2026-04) flagged that "what is stable" was
only knowable by reading source. This guide closes that loop. Linked
in `mkdocs.yml` nav under Guide → API Stability.

#### Fixed — `arcana.__version__` drift

`arcana.__version__` was last updated at v0.3.1 and silently drifted
through six releases while `pyproject.toml` was correctly bumped each
time. `import arcana; print(arcana.__version__)` now returns the
current version.

#### Fixed — `ChatSession.turn_count` / `max_history` were docs-only

`docs/guide/stability.md` and the `ChatSession.seed_history` docstring
both referenced `session.turn_count` and `session.max_history` as
public surface, but neither was a real attribute — only the private
`_turn_count` / `_max_history` existed. Discovered during a §3.3
post-merge audit (any new feature should round-trip its own docs).
Added both as read-only properties wrapping the existing internal
state. Additive, no user could have been affected.

#### Added — Provider tool-calling hints (§3.5a + §3.5b)

A user-controlled slot for "extra prompt scaffolding when this provider
is invoked with tools bound". Implements the constitutional middle
path between "framework auto-rewrites prompts per provider"
(violation: Principle 4 — framework deciding how to talk to the LLM)
and "users re-discover the same workarounds across every project"
(the original feedback's pain).

Decomposed into infrastructure (3.5a, code) and content (3.5b, docs):

##### 3.5a — `RuntimeConfig.tool_calling_hint{,s}` slot

- `RuntimeConfig.tool_calling_hint: str | None = None` — global default
- `RuntimeConfig.tool_calling_hints: dict[str, str] = {}` — per-provider
  override
- Resolution at request time: per-provider value wins over global. If
  neither resolves, no hint is injected.
- The hint is rendered as an **additional** system message, inserted
  after the user's authored system prompt(s) and before user/assistant
  turns. The user's original `system_prompt` is never mutated.
- Only rendered when `active_tools` is non-empty for the request — if
  the LLM is invoked without tools, the hint is a no-op.
- Fully captured in `PromptSnapshot` when trace snapshots are enabled
  (Principle 5: auditable). No new event type added — existing snapshot
  machinery covers it.
- Plumbed from `RuntimeConfig` through `Runtime` / `ChatSession` to
  `ConversationAgent` at all three construction sites
  (run / chat / chain).
- Default empty: zero behaviour change for existing users.

##### 3.5b — `docs/guide/providers.md` "Tool-Calling Hints" section

Per-provider observed quirks plus suggested hint text users can copy
into their `tool_calling_hints`. Updated as docs (not code), so
recommendation changes never become silent runtime behaviour changes.
Explicitly notes that GLM-4-flash benefits from a hint (the original
Roboot feedback case); OpenAI / Anthropic / DeepSeek / Gemini / Kimi
generally do not need one.

##### Constitutional rationale

- Principle 4 (Strategy Leaps): framework provides plumbing (the slot
  + the rendering rule); user owns content. No framework opinion on
  prompt strategy ships in code.
- Principle 5 (Auditability): the injection is visible in
  `PromptSnapshot` — fully traceable.
- Principle 6 (OS not Form Engine): a typed slot for an additional
  system block is OS-shaped, not Form-shaped (no prescribed
  reasoning loop, no compulsory hint).
- Prohibition 1 (No Premature Structuring): default-empty means no
  behaviour change for anyone who doesn't opt in. The slot was added
  in response to a real, observed need (Roboot integration, 2026-04),
  not a speculative one.
- The framework explicitly ships **no per-provider defaults** —
  default values would themselves be a position on prompt strategy
  and would drift silently across versions.

16 new tests in `tests/test_tool_calling_hint.py` cover: no-op cases
(no tools / no hint / wrong provider / empty string), injection
(global / per-provider / both / cross-provider fallback), insertion
order (after leading system blocks / at start when no leading system /
original prompt unchanged), the "no framework default" invariant
across six providers, and the `RuntimeConfig` plumbing path.

#### Added — `tests/test_stability_surface.py` (CI guard)

The same import-everything audit that caught `DiagnosticBrief` /
`ContextBudget` and `ChatSession.turn_count` is now a pytest module
with 14 parametrized cases. Every name listed in
`docs/guide/stability.md` round-trips through `hasattr` /
`importlib.import_module`. Includes:

- Top-level `arcana.*` (Runtime / Budget / Message / etc.)
- `arcana.__version__` is set and looks current
- `Runtime` methods (run / chat / chain / collaborate / ...)
- `ChatSession` public surface — **two-directional**: documented
  names must exist AND every public attribute must be documented
  (catches accidental public exposure of internals)
- Each `arcana.contracts.*` module's claimed name set
- §3.2 invariant: `arcana.runtime.conversation.__all__` excludes
  `Message` / `MessageRole`

If anyone removes a stable name or adds public surface without
updating the docs, the test fails before the PR can land.

#### Added — PR template "Public Surface Impact" section (§3.4)

Every PR now answers: does it touch a name on the stability list, and
if so is the change additive or breaking? Breaking changes require a
`Migration` section in the CHANGELOG entry per the v0.9.0 precedent.
Non-breaking additive changes are tagged "minor bump candidate" so
the release roll-up captures them. Encodes the practice that started
with v0.9.0's `ToolErrorCategory` migration recipe; no longer relies
on the contributor remembering.

## [0.9.0] - 2026-04-26 — "The Tool Boundary Release"

Two changes that together turn Prohibition 4 (No Mechanical Retry) and
Principle 6 (Runtime is an OS, not a Form Engine) from advisory into
runtime-enforced. The tool error contract no longer lets tools
self-report "is_retryable=True"; the gateway no longer schedules write
tools concurrently.

Both are breaking on user code that imported the old `ErrorType` enum or
read `ToolError.error_type`. Migration recipe is at the bottom of this
section.

### Changed — Tool error contract (BREAKING)
- **`arcana.contracts.tool.ErrorType` → `ToolErrorCategory`** — renamed
  and re-purposed. The old binary `RETRYABLE` / `NON_RETRYABLE` /
  `REQUIRES_HUMAN` axis conflated retry policy with error semantics;
  tools self-reported retry-eligibility and the gateway trusted them.
  The new categories are structural — `TRANSPORT`, `TIMEOUT`,
  `RATE_LIMIT` (the three retry-eligible) plus `VALIDATION`,
  `PERMISSION`, `LOGIC`, `CONFIRMATION_REQUIRED`, `UNEXPECTED`. Retry
  eligibility lives in a single frozenset in `arcana.contracts.tool`,
  not in tool code.
- **`ToolError.error_type` → `ToolError.category`** — field rename.
  `is_retryable` becomes a derived property keyed on the retry-eligible
  frozenset; tools cannot opt themselves into retry.
- **`ToolSpec.max_retries` default `3` → `2`** — one retry buys
  forgiveness for flap; more starts to mask real problems. Pass
  `max_retries=3` explicitly if you depended on the old default.

### Changed — Tool execution dispatch
- **`ToolGateway.call_many_concurrent` now batches by `SideEffect`** —
  read-side tools (`SideEffect.READ` / `SideEffect.PURE`) run
  concurrently as before. Write-side tools (`SideEffect.WRITE` /
  `SideEffect.NETWORK_WRITE`) serialize. The runtime owns dispatch
  semantics at the gateway boundary instead of asking tool authors not
  to race each other.

### Added — Constitutional invariant tests
- **`tests/test_constitutional_invariants.py`** — 13 tests covering the
  side-effect dispatch contract, `ask_user` non-blocking, cognitive-
  primitive opt-in, structured-output / tool coexistence, and the
  No-Mechanical-Retry contract. These are runtime-level enforcement of
  the Prohibition list; if any of them fail, the runtime is no longer
  constitutional.

### Added — Strict typing gate
- **`mypy --strict src/` is now a CI gate** — full strict mode passes.
  Run `uv run mypy src/` locally; CI blocks PRs that introduce `Any`
  leakage or untyped surfaces.

### Docs
- `docs/architecture.md` rewritten for V2 `ConversationAgent`. The V1-
  centric narrative is archived under `docs/legacy/`.
- `CONSTITUTION.md` v3.2 — corrects principle count (8 → 9 after v3.0
  added Principle 9) and tightens cognitive-primitive scope language.
- `.github/pull_request_template.md` — per-PR constitutional checklist.
- `docs/guide/api-tiers.md` — overview of the run / chat / chain /
  collaborate / batch tier.

### Migration
- `from arcana.contracts.tool import ErrorType` →
  `from arcana.contracts.tool import ToolErrorCategory`
- `tool_error.error_type` → `tool_error.category`
- `ToolError(error_type=ErrorType.RETRYABLE)` → pick a real category,
  most likely `ToolErrorCategory.TRANSPORT` or `ToolErrorCategory.TIMEOUT`.
- If you relied on `ToolSpec.max_retries=3` default, pass it explicitly.

## [0.8.2] - 2026-04-25 — "Bounded caches for long-running runtimes"

Two memory leaks for long-running runtimes. v0.8.1 caught the first
class of leak in `Channel`; v0.8.2 closes the same shape in `MessageBus`
and a separate leak in `ToolGateway`'s idempotency cache.

### Bounded `MessageBus` history + queue drain

`TeamOrchestrator` owns a single `MessageBus` instance that is reused
across every `run()` call. The orchestrator never calls `subscribe()`,
so published messages accumulated in per-role `asyncio.Queue`s forever
on top of the unbounded history. v0.8.2 bounds the history and drains
the queues at the end of every run. `HandoffResult.messages` is already
a detached `list(...)` copy taken before `reset()` fires, so callers
that retain the result see exactly what they saw before.

- **`MessageBus(history_limit=N)`** mirrors `Channel(history_limit=N)`.
  `None` (default) keeps unbounded history — matches pre-v0.8.2
  behaviour. `int >= 0` retains at most `N` past messages per session;
  `0` disables history retention entirely. Negative values raise
  `ValueError`. Implemented as a per-session
  `collections.deque(maxlen=...)`; `history()` still returns a plain
  `list` copy.
- **`MessageBus.reset()`** clears all history and drains every per-role
  queue via non-blocking `get_nowait()`. Required for owners that reuse
  a single bus across independent runs (e.g. `TeamOrchestrator`).
- **`TeamOrchestrator.run()` now calls `self._bus.reset()` in
  `finally`** — bus state no longer accumulates across runs.
- **`TeamOrchestrator(history_limit=N)`** — keyword-only; forwarded to
  the owned `MessageBus`.

Per-agent delivery queues in `arcana.multi_agent.channel` are
intentionally not bounded here — they are driven by the consumer's
`receive()` calls and an agent registered but never drained is a
consumer bug, not a retention bug.

### Bounded `ToolGateway` idempotency cache

`ToolGateway._idempotency_cache` grew unboundedly for the lifetime of
the owning `Runtime` and was never cleared on teardown. Any long-running
service that reuses a `Runtime` across `run()` calls with
`idempotency_key` (retries, dedupe, streaming pipelines) leaked memory
proportional to unique-key-count × `ToolResult.output` size — and
`ToolResult.output` is exactly the place large payloads land: stdout,
HTTP bodies, file contents.

- **`ToolGateway(idempotency_cache_limit=N)`** — keyword-only, defaults
  to `1024`. `None` keeps unbounded retention (the explicit opt-in for
  callers that need it). `int >= 0` caps at that size via LRU eviction;
  cache hits refresh MRU via `move_to_end`. `0` disables dedup entirely.
  Negative values raise `ValueError`. Backed by
  `collections.OrderedDict`.
- **`ToolGateway.close()` now clears `_idempotency_cache`** after
  `backend.cleanup()`, releasing all retained `ToolResult` references on
  `Runtime` teardown.

The `1024` default is the behaviour change: a caller with more than
1024 *live* keys will see LRU eviction where previously it saw unbounded
retention. Pass `idempotency_cache_limit=None` to restore the old
behaviour — but note that was always a leak in long-running processes.

## [0.8.1] - 2026-04-22 — "Trace You Can Actually Debug With"

Principle 5 (auditability) has always been Arcana's load-bearing promise.
v0.8.1 turns the trace from a dump of events into a first-class debugging
surface: one command per question, causal links between events, and a
single dev-mode switch that makes every turn fully replayable offline.

Also includes a previously-staged memory-leak fix for long-running pools
(bounded channel history).

### Added — Trace debugging

- **`arcana trace explain <run_id> --turn N`** — single-turn full story.
  One screen that joins *what went in* (curated messages, prompt
  snapshot, context decision, pinned items) with *what the LLM said*
  (thinking, assistant text, tool calls) and *what the runtime did with
  it* (tool results, `TurnAssessment`, error events). This is the
  "why did this turn do that?" command. `--json` for machine-readable
  output. Degrades gracefully when prompt snapshots are disabled.
- **`arcana trace flow <run_id>`** — ASCII DAG of the run.
  `Turn 1 → [tool_a ✓, tool_b ✗] → Turn 2 (completed)`. Follows
  `TraceEvent.parent_step_id` links to stitch the causal chain. Compact
  enough to eyeball in most terminals; `--json` for tooling.
- **`arcana trace show --errors --explain`** — the error triage shortcut.
  Lists error events as before, then auto-unfolds `trace explain` for
  the turn each error belongs to. Deduplicates turns that fired multiple
  errors.
- **`RuntimeConfig.dev_mode: bool = False`** — single switch that implies
  `trace_include_prompt_snapshots=True`. The idea: `dev_mode=True` in
  development gives you everything `explain` needs offline, without
  forcing ops-facing users to opt into PII-bearing snapshots. Explicit
  per-flag overrides still take precedence when already True.

### Added — Trace schema (backward compatible)
- **`TraceEvent.parent_step_id: str | None = None`** — causal link. For
  a single LLM turn, `CONTEXT_DECISION` / `PROMPT_SNAPSHOT` /
  `COGNITIVE_PRIMITIVE` / `TOOL_CALL` events all share the turn's
  `step_id` as their `parent_step_id`; the `TURN` event's
  `parent_step_id` points back to the previous turn (the spine). Legacy
  trace files (written before this release) parse unchanged — the field
  is optional and defaults to `None`.
- **`ToolCall.parent_step_id: str | None`** — threads the turn's
  `step_id` through to the `ToolGateway` so `TOOL_CALL` events can
  record it.
- **`TraceReader.collect_turn(run_id, turn)`** — bundles every event
  attached to one turn (turn event, context decision, prompt snapshot,
  tool calls, cognitive primitives, errors) via the parent link. The
  primitive behind `trace explain`; usable directly from Python.

### Added — Bounded channel history (memory leak fix)

Long-running `AgentPool`s retained every `Channel` message forever, which
turned the pool into a slow memory leak for daemon-style usage. v0.8.1
adds an opt-in bound.

- **`Channel(history_limit=N)`** in `arcana.multi_agent.channel`. ``None``
  (default) keeps unbounded history — pre-v0.8.1 behaviour. ``int >= 0``
  retains at most ``N`` past messages; ``0`` disables history retention
  entirely. Negative values raise ``ValueError``. Implemented as a
  ``collections.deque(maxlen=...)`` — ``Channel.history`` still returns a
  plain ``list`` copy, so readers are unaffected.
- **`AgentPool(channel_history_limit=N)`** and
  **`runtime.collaborate(channel_history_limit=N)`** — plumb the new knob
  through so users can set it at the entry point they actually use.

**Scope — what is *not* bounded:**
- Per-agent delivery queues (``asyncio.Queue`` per registered agent) are
  driven by the consumer's ``receive()`` calls. An agent that is
  registered but never receives will still grow its queue; that is a
  consumer problem, not a history-retention problem, and stays the
  user's responsibility.
- ``SharedContext`` is a user-written key-value store; its size is the
  user's to bound.

### Governance
- **Constitution v3.0 → v3.1** (2026-04-21) — Amend Principle 8: "can see
  what others have said" → "is given the means to see what others have
  said"; expand agents' role to include addressing and reading decisions.
  Clarifies that the framework's multi-agent obligation is to provide
  communication infrastructure, not to guarantee message reception.
  Resolves the v0.8.0 constitutional audit's only open finding; v0.8.0's
  `AgentPool` channel-plus-shared design is now the canonical
  implementation of Principle 8, not a compromise against it. No code
  change. See `specs/constitution-amendment-2-collaboration-means.md`.
- **v1.0.0 removals tracking** — `specs/v1.0.0-removals.md` records the
  policy (physical removal, no compatibility shims by default) and the
  first scheduled entry: `Runtime.team()` + `TeamSession` + `TeamMode`
  machinery. Rationale is tied to the amended Principle 8 — `team(mode=
  "shared")`'s rounds counter and fixed turn order are the exact topology
  the amendment rules out, so a shim would keep the violation alive.

## [0.8.0] - 2026-04-19 — "The Collaborative Cognition Release"

Multi-agent pools where each member is an independent cognitive instance.
Extends v0.7.0 primitives to pool settings without adding orchestration —
Principle 8 still holds, there is no graph DSL, no turn scheduler, no role
hierarchy. See the user guide at `docs/guide/multi-agent.md` and the spec
at `specs/v0.8.0-collaborative-cognition.md`.

### Added — Multi-agent infrastructure
- **`runtime.collaborate(budget?, cognitive_primitives?)`** — returns an
  `AgentPool`. Sync factory; the pool itself is an async context manager
  (`async with runtime.collaborate() as pool`). No `await` on the factory.
- **`AgentPool.add(name, *, system?, tools?, provider?, model?,
  max_history?, cognitive_primitives?)`** — registers a named
  `ChatSession` that shares the pool's `BudgetTracker`, `Channel`, and
  `SharedContext` but keeps its own prompt, tools, history, and
  cognitive state.
- **`AgentPool.channel`** — name-addressed `Channel` with point-to-point
  and broadcast delivery.
- **`AgentPool.shared`** — thread-safe `SharedContext` key-value store.
- **`AgentPool.agents`** — read-only snapshot of registered sessions.
- **`AgentPool` is an async context manager** — `__aexit__` releases
  sessions, drains the channel, and clears shared state. No orchestration
  actions (no auto-summaries, no strategy decisions).

### Added — Per-agent cognitive primitives
- **Per-agent `cognitive_primitives` override** on both
  `runtime.collaborate(...)` (pool default) and `pool.add(...)`
  (per-agent override). Resolution: per-agent explicit → pool default →
  `RuntimeConfig.cognitive_primitives`. Explicit `[]` opts an agent out
  even when a higher level opts in.
- **Isolated state per pool member** — each agent owns its own
  `CognitiveHandler`, `PinState`, and recall log. Pins made by agent A
  are never visible to agent B. `pin_budget_fraction` is evaluated
  against each agent's own context window.
- **Tool-name / primitive collision raises `ValueError`** at
  `pool.add(...)` time — a user tool named `recall`, `pin`, or `unpin`
  that collides with an active cognitive primitive is rejected instead
  of silently shadowed (Principle 5).

### Added — Pool-aware trace + CLI
- **`metadata["source_agent"]`** — every `TraceEvent` emitted during a
  pool run carries the originating agent's name in the existing
  `metadata` dict. The `TraceEvent` schema itself is unchanged, so
  v0.6.0/v0.7.0 trace consumers keep working.
- **`arcana trace pool-replay <run_id>`** — summary table listing each
  pool agent, their event count, and replayable turn list.
- **`arcana trace pool-replay <run_id> --agent <name> --turn <N>`** —
  per-agent prompt-composition replay scoped to one pool member.
- **`arcana trace show <run_id> --agent <name>`** and
  **`arcana trace replay <run_id> --agent <name> --turn N`** — agent
  scoping on the existing subcommands.
- **`arcana trace show`** annotates each event line with its
  `[source_agent]` tag when present; makes interleaved pool traces
  readable at a glance.

### Added — Contracts
- New `arcana.contracts.multi_agent.ChannelMessage` — immutable
  (`model_config = ConfigDict(frozen=True)`) so the single instance
  `Channel.send` fans out to all recipients cannot be mutated in place
  by one receiver at the others' expense. Use `model_copy(update=...)`
  to derive a modified message.
- `MessageType.CHAT` — added for default `ChannelMessage` classification.

### Changed — Deprecations
- **`runtime.team()` is deprecated** (emits `DeprecationWarning`). Use
  `runtime.collaborate()` instead. See the migration recipe at the bottom
  of `docs/guide/multi-agent.md`. Scheduled for removal in v1.0.0.

### Fixed — Pre-release bug fixes (from uncommitted v0.7.x pool work)
- **Bug: `Runtime.collaborate()` was `async def`** — the documented
  `async with runtime.collaborate() as pool` pattern failed with
  `TypeError: 'coroutine' object does not support the asynchronous
  context manager protocol`. Now a sync factory returning an
  `AgentPool` whose own `__aenter__/__aexit__` handle the context
  manager protocol (matches `runtime.chat()`).
- **Bug: `Channel.send` broadcast shared one mutable `ChannelMessage`**
  across all recipients plus `history`, so a mutation by any receiver
  bled across the others. `ChannelMessage` is now frozen; the shared
  instance is safe to fan out. Regression tests cover both bugs.

### Constitutional guard (explicitly NOT done)
- No graph DSL, no `StateGraph`-equivalent for multi-agent flows.
- No turn scheduler. Who talks when is user code (`async for` / `if` /
  `await`).
- No role hierarchy. Roles live in system prompts, not framework types.
- No auto stop conditions. Stop when user code decides to stop.
- No cross-agent cognitive inheritance. Pool agent A's pins never
  populate agent B's state; explicit `pool.shared.set(...)` remains the
  only intentional hand-off.

## [0.7.0] - 2026-04-18 — "The Cognitive Primitives Release"

Runtime services for the LLM's own reasoning state. The LLM can now invoke
two intercepted tools — `recall` and `pin` (with companion `unpin`) — to
work around the lossiness of working-set compression. See the user guide at
`docs/guide/cognitive-primitives.md` and Principle 9 in `CONSTITUTION.md`.

### Added — Cognitive primitives
- **`recall(turn, include?)`** — retrieve an earlier turn's messages at
  full fidelity, bypassing any working-set compression. Supports
  `include="all"` (default) / `"assistant_only"` / `"tool_calls"` filters.
  Delegates to the live conversation log, falls back to the trace reader.
  Out-of-range / invalid turns return structured `RecallResult` with
  `found=False` and an actionable `note` — never exceptions (Principle 5).
- **`pin(content, label?, until_turn?)`** — protect specific content from
  compression in future working sets. Returns a `pin_id` the LLM uses with
  `unpin`. Idempotent by SHA-256 of content (duplicate pin returns the
  existing id). Budget-capped at
  `RuntimeConfig.pin_budget_fraction * total_window` (default 50%) —
  over-cap requests are rejected with a structured diagnosis that includes
  current usage, requested size, cap, and a remediation suggestion. The
  framework never auto-unpins or truncates existing pins.
- **`unpin(pin_id)`** — remove an earlier pin; always returns a structured
  result whether or not the id existed.
- **Pinned blocks render inside the Working layer** as independent
  `ContextBlock(pinned=True)` entries, excluded from
  `_compress_with_relevance` / `_aggressive_truncate`, and surfaced in
  `ContextDecision.decisions` with `outcome="kept"` and `reason="pinned"`.
  Principle 2's four-layer structure (Identity/Task/Working/External) is
  unchanged — no new layer.
- **`RuntimeConfig.cognitive_primitives: list[str] = []`** and
  **`RuntimeConfig.pin_budget_fraction: float = 0.5`** — opt-in per
  runtime; empty default means no behavioural change.
- **`EventType.COGNITIVE_PRIMITIVE`** — every primitive invocation emits
  a trace event with `{primitive, args, result}` metadata.

### Added — Contracts
- New module `arcana.contracts.cognitive` — `RecallRequest/Result`,
  `PinRequest/Result`, `UnpinRequest/Result`, `PinEntry`, `PinState`.
- `ContextBlock.pinned: bool = False` — per-block flag.

### Added — Runtime
- `arcana.runtime.cognitive.CognitiveHandler` — session-local handler
  that owns `PinState` and services interception, wired into
  `ConversationAgent._execute_tools` via the same mechanism as `ask_user`.
- `WorkingSetBuilder.set_pin_state(pin_state)` — attaches the session's
  pin state so active pins are rendered as independent messages in every
  working set build.

### Added — CLI
- `arcana trace show <run_id> --cognitive` — filter to
  `COGNITIVE_PRIMITIVE` events with human-readable formatting per primitive.
- `arcana trace show <run_id> --context` — pinned entries are now flagged
  with a `[PIN]` prefix in the per-message decisions view.
- `arcana trace replay <run_id> --turn N` — appends an *Active pins at
  turn N* section reconstructed from the run's cognitive events.

### Added — Documentation
- New user guide: `docs/guide/cognitive-primitives.md`.
- `CONSTITUTION.md` v3.0 — Principle 9 (Cognitive Primitives as Services)
  and two Chapter IV entries.

### Constitutional guard (explicitly NOT done)
- Framework does not call a primitive on the LLM's behalf; every
  invocation is an explicit LLM tool call with a `tool_call_id` and trace
  record.
- No system-prompt hints such as *"consider using recall here"*.
- No auto-unpin, no pin truncation, no eviction on budget pressure — the
  LLM decides how to free budget when rejected.
- Default-off: empty `cognitive_primitives` list means zero behavioural
  change over v0.6.0.

### Stats
- 1368 tests passing (+31 new): `test_cognitive_recall.py` (11) +
  `test_cognitive_pin.py` (19) + `test_context_decision_evidence.py` (+1
  pinned-block case).

## [0.6.0] - 2026-04-17 — "The Explainability Release"

### Added — Context Explainability
- **`MessageDecision` contract**: Structured per-message evidence for every context composition. Records index / role / outcome (kept / compressed / dropped / summarized) / fidelity level (L0–L3) / relevance score / token counts before/after / reason. One entry per input message.
- **`ContextDecision.decisions`**: Authoritative list of `MessageDecision` replacing the free-text-only explanation. Covers all 5 strategy paths (passthrough, tail_preserve head/tail/middle, aggressive_truncate, LLM summarize, no-summary-budget).
- **Stale tool result pruning visibility**: `_prune_stale_tool_results` now returns `pruning_info` mapping pruned indices to original token counts. Phase 0 pruning is visible in `decisions` with `reason="stale_tool_result"` (or merged with the downstream strategy reason).
- **`CONTEXT_DECISION` trace event**: metadata now carries the full `ContextDecision.model_dump()` and `ContextReport.model_dump()` — consumers can losslessly reconstruct either.

### Added — Prompt Snapshots & Replay
- **`PromptSnapshot` contract**: Captures the exact `LLMRequest` (messages, tools, model, response_format, budget snapshot) sent to the provider for a single turn.
- **`EventType.PROMPT_SNAPSHOT`**: Emitted before each `gateway.generate()` / `gateway.stream()` when opted in.
- **`RuntimeConfig.trace_include_prompt_snapshots: bool = False`**: Opt-in flag. Default off to avoid PII/secret leakage and trace bloat.
- **`TraceReader.list_turns(run_id)`**: Enumerate turn numbers that have replay evidence.
- **`TraceReader.replay_prompt(run_id, turn)`**: Reconstruct `PromptReplay` for a single turn (prompt snapshot + context decision + context report + budget snapshot).
- **`arcana trace replay <run_id> --turn N`**: CLI subcommand for human-readable or JSON replay output. Supports `--prompt-only` / `--decision-only` / `--json` modes.

### Constitutional guard (explicitly NOT done)
- Framework does not inject `MessageDecision` or relevance scores into the LLM prompt itself (would violate Prohibition 1 / P4)
- No automatic recovery, retry, or tool expansion based on decisions (decisions are retrospective evidence, not an input channel)
- Existing strategy selection / compression thresholds / fidelity logic unchanged (pure transparency layer)
- Prompt snapshots default off (anti context/trace hoarding, anti PII leakage)

### Stats
- 1337 tests passing (+13 new): `test_context_decision_evidence.py` (5) + `test_trace_replay.py` (8)

## [0.5.0] - 2026-04-12 — "The Resilience Release"

> Note: v0.5.0 was authored in CHANGELOG and shipped as part of the v0.6.0 release cycle but never received its own git tag. The 2026-04-12 date is the commit date of `feat: v0.5.0 resilience improvements` (e208799). Entries below describe what landed under the v0.5.0 banner.

### Added — Runtime OS Reliability
- **Phase 0 tool result pruning**: zero-cost compression stage before strategy-level compression. Old tool results (outside `tool_result_staleness_turns * 3` recent messages) replaced with summary placeholders. Error/failure results never pruned.
- **Iteration budget sharing**: `BudgetTracker.max_iterations` / `iterations_used`; `Budget.max_iterations` propagates to shared tracker; team agents share a global iteration cap.
- **MCP dynamic tool discovery**: `MCPConnection` listens for `notifications/tools/list_changed`; `MCPToolProvider` refreshes the registry on change.

### Constitutional review decisions (from specs/v050-upgrade.md)
- **Rejected**: Parallel tool conflict detection — tool authors should self-protect (over-engineering, framework overreach)
- **Rejected**: Diagnosis → recovery loop — framework crossing into LLM strategy territory (violates Principle 6 boundary)
- **Rejected**: ChatSession persistence — application-layer concern, not Runtime OS

## [0.4.0] - 2026-04-11

### Added — Execution Isolation Architecture
- **`ExecutionBackend` protocol**: Pluggable abstraction for WHERE tools execute. Decouples tool logic from execution environment. Ships with `InProcessBackend` (default, zero overhead). Framework extension point for subprocess/container/remote backends
- **`ExecutionChannel` protocol**: Pluggable abstraction for HOW the agent loop communicates with tool execution. Enables future physical separation of Brain (reasoning) and Hands (tool execution). Ships with `LocalChannel` (wraps ToolGateway, zero overhead)
- **`ToolGateway.close()`**: Lifecycle method that invokes `backend.cleanup()`, ensuring non-default backends (socket, container, etc.) release resources properly
- **`Runtime.close()` chains to `ToolGateway.close()`**: Full resource cleanup cascade from Runtime → ToolGateway → ExecutionBackend
- **`ConversationAgent` channel routing**: `_execute_tools()` prefers `ExecutionChannel` when provided, falls back to `ToolGateway` otherwise. `ask_user` always bypasses the channel

### Stats
- All 1227 tests passing, 0 failures (+25 new tests)

## [0.3.3] - 2026-04-06

### Fixed — Provider Compatibility
- **Intent router bypasses structured output**: When `response_format` is set, the intent router no longer short-circuits to `direct_answer` (which didn't pass the format to the LLM). Structured output now always goes through the full ConversationAgent loop
- **Intent router ignores available tools**: `classify()` now receives `available_tools` from the tool registry, so "What is X? Use the calc tool" correctly routes to the agent loop instead of direct_answer
- **Structured output code fence stripping**: Providers that return JSON wrapped in markdown code fences (` ```json ... ``` `) are now auto-stripped before parsing. Fixes GLM and MiniMax structured output
- **Structured output schema prompt strengthened**: `json_object` fallback mode now includes exact field names, a concrete example, and "do not rename or omit" instruction. Fixes Kimi/GLM/MiniMax returning wrong field names
- **MiniMax auto-degraded to prompt-based tools**: MiniMax rejects native `tool_calls` with 400; `ProviderProfile` auto-degrades on first failure, subsequent calls use prompt-based fallback seamlessly

### Verified Providers
Real API verification for all accessible providers:
- **DeepSeek**: direct answer ✓, tool calling ✓, structured output ✓
- **OpenAI**: direct answer ✓, tool calling ✓, structured output ✓
- **Anthropic**: direct answer ✓, structured output ✓ (previously verified)
- **Kimi (Moonshot)**: direct answer ✓, tool calling ✓, structured output ✓ ← NEW
- **GLM (Zhipu)**: direct answer ✓, tool calling ✓, structured output ✓ ← NEW
- **MiniMax**: direct answer ✓, tool calling ✓ (auto-degraded), structured output ✓ ← NEW
- Gemini: blocked by region restriction (API key valid)
- Ollama: not tested (requires local deployment)

### Stats
- All 1202 tests passing, 0 failures (+18 new tests)

## [0.3.2] - 2026-04-06

### Changed — Architecture
- **ChatSession delegates to ConversationAgent**: `ChatSession.send()` now runs the full `ConversationAgent` turn loop internally, gaining all V2 features automatically (ask_user, lazy tools, diagnostics, fidelity compression, thinking assessment, streaming events). Removes ~300 lines of duplicated LLM/tool dispatch logic
- **Memory injection moved to first user message**: Memory context is now injected into the first user message instead of the system prompt, keeping the system prompt stable for provider prompt caching

### Added — Context Intelligence
- **Fidelity-graded compression**: Context compression now uses 4 fidelity levels instead of binary keep/compress:
  - **L0** (score ≥ 0.7): Original message preserved verbatim
  - **L1** (score ≥ 0.4): Condensed to key content
  - **L2** (score < 0.4): Single summary line
  - **L3**: Dropped entirely
  - `ContextReport.fidelity_distribution` tracks the distribution per turn
- **`StreamAccumulator`**: New utility class (`runtime/stream_accumulator.py`) for assembling streaming chunks into a complete `LLMResponse` — single state-management point for text, thinking, tool calls, and usage
- **`LazyToolRegistry.tool_token_estimate`**: Cached token estimate for current working set tools, auto-invalidated on expansion/reset
- **`Message.token_count` caching**: Token estimation now uses cached property on `Message` instead of re-computing from content text each call

### Fixed
- **Recall tool budgeting**: Fixed budget accounting for recall tool invocations
- **Zero-budget page table eviction**: Fixed edge case where zero remaining budget caused incorrect eviction behavior in context compression

### Removed
- **Virtual memory subsystem**: Added in Context OS commit, removed after evaluation — the fidelity spectrum approach achieves the same goals with less complexity

### Stats
- All 1184 tests passing, 0 failures

## [0.3.1] - 2026-04-05

### Added — Provider Compatibility
- **`ProviderProfile`**: Unified capability system per provider. Tracks `tool_calls`, `json_schema`, `json_mode`, `streaming`, `stream_options`. Known providers get pre-configured profiles; custom providers get conservative defaults
- **Auto-degradation**: When a provider returns 400 for tool_calls, the profile is updated automatically — subsequent calls skip native tools and use prompt-based fallback. Only fails once per capability
- **Custom provider registration**: `providers={"siliconflow": {"api_key": "...", "base_url": "...", "model": "...", "tool_calls": False}}` — any OpenAI-compatible API with explicit capability overrides
- **`ChatSession.send(message, images=[...])`**: Multimodal messages in chat sessions
- **`runtime.create_chat_session()`**: Returns ChatSession directly without requiring `async with`, for use across HTTP requests
- **`arcana.RuntimeConfig`**: Now exported from the top-level package

### Stats
- All 1173 tests passing, 0 failures

## [0.3.0] - 2026-04-04 — "The Context Release"

### Added — Context Transparency

- **`ContextReport`**: Every LLM call now produces a detailed report of how the context window was composed. Shows token allocation across layers (identity, task, tools, history, memory), compression metrics, and window utilization. Available on `RunResult.context_report` and `ChatResponse.context_report`
- **`ContextStrategy`**: Adaptive compression strategy system replaces one-size-fits-all compression. Four tiers:
  - **passthrough** (< 50% utilization): No compression, zero overhead
  - **tail_preserve** (50-75%): Compress middle history, keep recent turns verbatim
  - **llm_summarize** (75-90%): Use cheap LLM call for semantic summarization
  - **aggressive_truncate** (> 90%): Keep only system + last 2 turns
  - Configurable via `Runtime(context_strategy=ContextStrategy(...))` or shorthand `"off"` / `"always_compress"`
- **Structured stream events**: `runtime.stream()` and `ChatSession.stream()` now emit:
  - `TOOL_START` — tool name and arguments before execution
  - `TOOL_END` — tool result and duration after execution
  - `TURN_END` — token count and cost at end of each turn
  - `CONTEXT_REPORT` — full context composition report per turn
- **`StreamEventType` exported**: `arcana.StreamEventType` for `match` statements on stream events

### Stats
- Tests: 1142 → 1173 (+31 new tests for context features)
- All 1173 tests passing, 0 failures

## [0.2.2] - 2026-04-04

### Fixed — Core Reliability
- **asyncio.Lock replaces threading.Lock**: `Runtime._totals_lock` was a `threading.Lock` blocking the event loop in async code. Now uses `asyncio.Lock` for proper async concurrency
- **Tool gateway idempotency race**: Fixed TOCTOU race where two concurrent calls with the same idempotency key could both execute. Lock now covers the entire check→execute→cache window
- **Budget boundary off-by-one**: `BudgetTracker.check_budget()` used `>=` (triggers at exact limit), now uses `>` (allows using exactly the allocated budget). Same fix applied to `BudgetScope`
- **Provider close() isolation**: `ModelGatewayRegistry.close()` now catches exceptions per-provider — one failing provider no longer blocks cleanup of others
- **MCP reconnect serialization**: Added `asyncio.Lock` to `MCPConnection._reconnect()` preventing concurrent reconnect attempts from corrupting transport state
- **MCP disconnect_all resilience**: Individual server disconnect failures no longer abort the cleanup loop
- **Graph checkpointer blocking I/O**: `GraphCheckpointer.save()/load()/delete()` were fake-async (blocking file I/O). Now uses `asyncio.to_thread()` + atomic write (temp file + rename) to prevent corruption on crash
- **Trace reader token/cost accounting**: `TraceReader.summarize()` used `max()` instead of `+=` for tokens/cost, reporting peak values instead of totals
- **SSE line terminator**: MCP Streamable HTTP transport now handles `\r\n` and `\r` per SSE spec, not just `\n`
- **Silent hook/callback failures**: Bare `except: pass` in agent hooks and `on_parse_error` callback now logs to `logger.debug` for debuggability

### Added
- **`Runtime` as async context manager**: `async with Runtime(...) as rt:` ensures `close()` is called, preventing HTTP connection leaks
- **`BudgetTracker.can_afford(estimated_tokens, estimated_cost)`**: Now checks cost budget in addition to token budget

### Removed — Dead Code Cleanup
- **`orchestrator/`**: Entire module (Orchestrator, TaskScheduler, TaskGraph, ExecutorPool) — never used by runtime
- **`gateway/router.py`**: ModelRouter — never imported
- **`gateway/capabilities.py`**: CapabilityRegistry — never queried
- **`streaming/sse.py`**: SSE formatter — never called
- **`runtime/replay.py`**: ReplayEngine — never wired up
- **`tool_gateway/adapters/langchain.py`**: LangChain bridge — never loaded
- **`storage/postgres.py`**, **`storage/chroma.py`**: Production storage backends removed. Arcana provides the `StorageBackend`/`VectorStore` interfaces; users implement for their infrastructure
- Removed `chromadb` dev dependency

### Stats
- Tests: 1234 → 1142 (removed 92 tests for deleted dead code)
- All 1142 tests passing, 0 failures
- mypy strict: 8 errors (all pre-existing)

## [0.2.1] - 2026-03-28

### Fixed — Production High Availability
- **Provider connection leak**: `Runtime.close()` now cascades to all provider HTTP clients (AsyncOpenAI, AsyncAnthropic). Previously only closed MCP connections, leaking connection pools in long-running apps
- **Budget race condition**: `Runtime._total_tokens_used` and `_total_cost_usd` now protected by `threading.Lock`. Concurrent `run()` calls no longer corrupt cumulative budget counters
- **timeout_ms actually wired**: `ModelConfig.timeout_ms` now passed to provider SDK `create()` calls as per-request timeout. Previously the config existed but was silently ignored (SDK defaulted to 600s)
- **Cancellation safety**: `asyncio.CancelledError` and `KeyboardInterrupt` in `Runtime.run()` and `ConversationAgent` now record partial budget and leave state consistent before re-raising

### Added — Developer Experience
- **`arcana init`**: CLI scaffold command generates `main.py` + `.env.example` + `agent.yaml` for 30-second quickstart
- **`Runtime.on()` / `Runtime.off()`**: Event hook API for runtime lifecycle events (`run_start`, `run_end`, `error`). Supports sync and async callbacks, chainable
- **`ChatSession(max_history=N)`**: Sliding window on message history to prevent OOM in long conversations. System messages always preserved. `runtime.chat(max_history=100)`
- **LangChain adapter test suite**: 18 tests covering spec extraction, execution, error handling, protocol compliance
- **SECURITY.md**: Honest security model documentation — what Arcana secures and what it doesn't
- **CI coverage reporting**: pytest-cov + Codecov upload in GitHub Actions
- **Dynamic README badges**: PyPI version, CI status, coverage — no more stale static badges

### Changed
- Example 13 rewritten to use `runtime.chat()` / `ChatSession.send()` instead of manual LLM message management

### Stats
- Tests: 1164 → 1234 (+70 new tests)
- All 1234 tests passing, 0 failures

## [0.2.0] - 2026-03-27

### Fixed — Structured Output Reliability
- **`result.parsed` always returns `BaseModel | None`**: Fixed bug where `parsed` could be a raw `dict` when provider degrades to `json_object` mode. Now handles dict inputs, validates `on_parse_error` callback returns, and guarantees type consistency
- **Anthropic structured output**: `AnthropicProvider` now supports `response_format` — injects JSON schema into system prompt (same fallback strategy as DeepSeek/Ollama/Kimi). Works with and without tools

### Added — Batch API & Budget Granularity
- **`Runtime.run_batch(tasks, concurrency=5)`**: Run multiple independent tasks concurrently with `asyncio.Semaphore`. Individual failures don't crash the batch. Returns `BatchResult` with results, succeeded/failed counts, total tokens/cost
- **Provider-level `batch_generate()`**: `OpenAICompatibleProvider.batch_generate(requests, config, concurrency=5)` for concurrent LLM calls. Registry-level fallback when provider doesn't implement batch
- **`ChainStep.budget`**: Per-step budget in `chain()` pipelines. Each step can have its own budget cap, always capped by chain-level remaining budget. Steps without budget share the chain pool

### Stats
- Tests: 1164, all passing

## [0.1.0-beta.8] - 2026-03-27

### Added — Team Dual Mode
- **`runtime.team(mode="shared"|"session")`**: Two collaboration modes. `"shared"` (default) — all agents share one conversation history. `"session"` — each agent has an independent context; other agents' messages arrive as user messages

### Stats
- Tests: 1135, all passing

## [0.1.0-beta.7] - 2026-03-27

### Fixed — Provider Compatibility
- **Cost estimation**: `TokenUsage.cost_estimate` now uses realistic mid-range pricing ($0.15/M input, $0.60/M output) instead of placeholder values
- **Zero-token warning**: When a provider reports 0 tokens, the runtime estimates from response length and logs a warning instead of silently tracking $0
- **Structured output + json_schema auto-downgrade**: Providers that don't support `json_schema` response format (DeepSeek, Ollama, Kimi, GLM, MiniMax) automatically fall back to `json_object` with schema instructions injected into system prompt
- **Provider model config**: `providers` dict now accepts `{"provider": {"api_key": "...", "model": "..."}}` to override default model per provider
- **Tool call logging**: Debug-level logs for all tool calls and results (name, arguments, output)

## [0.1.0-beta.6] - 2026-03-26

### Added — Pipeline & Budget Control
- **Parallel chain branches**: `runtime.chain()` now accepts nested lists for parallel execution — `[ChainStep, [ChainStep, ChainStep], ChainStep]` runs the inner list concurrently with `asyncio.gather`
- **Per-run provider/model selection**: `runtime.run(provider="openai", model="gpt-4o")` overrides default provider/model for a single run. Also available on `runtime.stream()` and `ChainStep`
- **Budget scoping**: `async with runtime.budget_scope(max_cost_usd=0.50) as scoped:` isolates budget for a subset of runs
- **`on_parse_error` callback**: `runtime.run(response_format=MyModel, on_parse_error=fix_fn)` — fires on `json.JSONDecodeError` or `pydantic.ValidationError`, NOT on provider-level format rejection
- **`result.parsed` field**: `RunResult.parsed` holds the validated Pydantic model (separate from `result.output` for backward compatibility)
- **`Tool` class**: Non-decorator tool registration — `Tool(fn=my_func, when_to_use="...")` for when `@arcana.tool` is not practical

### Changed
- `ChainStep` now supports `provider`, `model`, and `on_parse_error` fields
- Tools and structured output coexist — agent uses tools during reasoning and returns structured output on the final turn
- `BudgetScope` exported from `arcana` package

## [0.1.0-beta.5] - 2026-03-25

### Fixed
- 8 user-reported issues: SDK `system` and `context` parameters, fallback chain logging, budget tracking across runs, `runtime.fallback_order` property, provider `get_fallback_chain()` method, Tool wrapper support in registry

### Added
- **`arcana.run(system=..., context=...)`**: System prompt and context injection available at SDK level
- **`runtime.budget_remaining_usd`** / **`runtime.tokens_used`**: Runtime-level cumulative budget tracking properties
- **Auto fallback chain**: Multiple providers automatically form a fallback chain based on registration order

## [0.1.0-beta.4] - 2026-03-24

### Fixed
- 14 mypy strict errors regressed after beta.3
- `ChatSession.send()` now uses `generate()` instead of `stream()` for reliable usage tracking
- CI lint errors (unused imports, import sorting, bare except)

### Added
- Automated PyPI publish workflow (CI)
- Integration verification tests for b7 features

## [0.1.0-beta.3] - 2026-03-24

### Added — LLM Capability Amplification
- **Parallel Tool Execution**: Multiple tool calls in a single turn now run concurrently via `asyncio.gather`, with order-preserving results and individual failure isolation
- **Prompt Caching**: Anthropic system prompt + tool schemas automatically tagged with `cache_control`; OpenAI `cached_tokens` tracked. Up to 90% input token savings on multi-turn runs
- **Thinking-Informed Assessment**: `_assess_turn` now analyzes extended thinking blocks for uncertainty, verification intent, and incomplete information signals. Adjusts confidence and completion accordingly
- **Structured Output**: `arcana.run(response_format=MyModel)` returns validated Pydantic instances. Provider-level `json_schema` mode for OpenAI-compatible APIs
- **Multimodal Input**: `arcana.run(images=[...])` accepts URLs, file paths, and data URIs. OpenAI ↔ Anthropic content block format auto-conversion
- **LLM-Driven Context Compression**: `WorkingSetBuilder` can use a cheap LLM to produce semantic summaries instead of keyword-based truncation. Async `abuild_conversation_context()` with graceful fallback

### Added — Interactive Capabilities
- **`ask_user` Built-in Tool**: LLM can ask clarifying questions mid-execution. Intercepted at runtime level (bypasses ToolGateway). Sync/async `input_handler` callback. Graceful fallback when no handler provided
- **`runtime.chat()`**: Multi-turn conversational sessions with persistent history, shared budget, context compression, and streaming support. `ChatSession.send()` / `ChatSession.stream()`
- **CLI `arcana chat`**: Interactive terminal chat with Rich formatting, per-turn token/cost stats, budget enforcement
- **Examples 13-14**: Interactive chat and ask_user demonstrations

### Changed — Constitution v2
- **Principle 2** expanded: context is modality-agnostic (text, images, structured data)
- **Principle 4** corollary: thinking is signal, not contract — runtime may listen but never constrain
- **Principle 8** added: agent autonomy in collaboration — framework provides coordination, never hierarchy
- **Chapter IV** expanded: User role defined (intent, information, judgment). Two new inviolable rules: user never forced to interact; LLM asks but never blocks
- **Contributor Compact**: Questions 8-9 added (agent autonomy, user optionality)

### Added — Documentation
- `docs/guide/quickstart.md` — Installation → Deployment guide
- `docs/guide/configuration.md` — Full configuration reference (16 sections)
- `docs/guide/providers.md` — 8 provider setup guides with fallback chains
- `docs/guide/api.md` — Public API reference (881 lines)

### Stats
- Tests: 878 → 1045 (+167 new tests)
- All 1045 tests passing, 0 failures
- 9 new features, 4 documentation files

## [0.1.0-beta.1] - 2026-03-18

### Added
- **Runtime + Session**: Long-lived resource container, create once use everywhere
- **Runtime.team()**: Multi-agent collaboration (constitutional — Runtime provides comm, agents decide strategy)
- **Runtime.stream()**: Async generator for streaming
- **Runtime.graph()**: StateGraph factory
- **Memory v2**: Relevance-based retrieval (keyword + recency + importance + token budget)
- **MCP Client**: stdio transport, MCPToolProvider → ToolGateway bridge
- **CLI**: `arcana run/trace/providers/version`
- **ConversationAgent (V2)**: TurnFacts/TurnAssessment separation, 51% token savings
- **108 new tests**: All user-facing modules now covered (713 total)
- **Actionable error messages**: 8 files improved
- **Intent Router**: Default on in ConversationAgent
- **Diagnostic Recovery**: Structured diagnosis in V2

### Changed
- `arcana.run()` delegates to Runtime, accepts `api_key` param
- Default engine is V2 ConversationAgent
- Hardcoded model IDs removed — user explicit > provider default > error
- README → "Agent Runtime for Production"

### Fixed
- AdaptivePolicy execution closure
- Memory injection through direct_answer fast path
- Tool results use native OpenAI format
- single_tool argument generation

## [0.1.0-alpha.2] - 2026-03-18

### Changed
- `arcana.run()` now accepts `api_key` parameter — no .env file needed
- Default engine switched to ConversationAgent (V2)
- `max_steps` renamed to `max_turns` in `arcana.run()`
- `engine="conversation"` (default) or `engine="adaptive"` (V1)

### Fixed
- SDK no longer forces environment variables for API keys
- OpenAI and Anthropic providers now work in `arcana.run()`
- Clear error message when no API key provided

## [0.1.0-alpha.1] - 2026-03-18

### Added

#### V2 Execution Engine
- **ConversationAgent**: LLM-native execution model with TurnFacts/TurnAssessment separation
- **TurnFacts**: Raw provider output, zero interpretation
- **TurnAssessment**: Runtime completion/failure judgment, separate from facts
- 13-step turn contract with 7 invariants
- Streaming via `ConversationAgent.astream()`

#### V2 Architecture
- **CONSTITUTION.md**: 7 design principles, 4 prohibitions
- **Intent Router**: Rule-based + LLM + Hybrid classifiers, 4 execution paths
- **Adaptive Policy**: 6 strategy types (direct_answer, single_tool, sequential, parallel, plan_and_execute, pivot)
- **Lazy Tool Loading**: Keyword-based tool selection, affordance fields on ToolSpec
- **Diagnostic Recovery**: 7 error categories, structured feedback, RecoveryTracker
- **Multi-Model Routing**: 5 model roles, complexity-based selection
- **Working Set Builder**: 4-layer context management (identity/task/working/external)

#### Provider Infrastructure
- **BaseProvider Protocol**: Replaces ABC, maps to Rust trait
- **AnthropicProvider**: Native Claude support with extended thinking
- **Chinese Providers**: Kimi (Moonshot), GLM (Zhipu), MiniMax factory functions
- **Capability Registry**: 22 capabilities across 8 providers
- **Error Hierarchy**: RateLimitError, AuthenticationError, ModelNotFoundError, ContentFilterError, ContextLengthError
- **StreamChunk**: Unified streaming data model

#### SDK
- `arcana.run()`: Zero-config entry point
- `@arcana.tool`: Decorator with affordance fields (when_to_use, what_to_expect, failure_meaning)
- `RunResult`: Structured result with output, steps, tokens, cost
- Budget tracking wired into SDK

#### Evaluation
- EvalMetrics: first_attempt_success, goal_achievement_rate, cost_per_success
- RuleJudge + LLMJudge + HybridJudge
- EvalRunnerV2 with suite reporting

#### Code Quality
- 605 tests, 0 failures
- Verified with real APIs: DeepSeek, OpenAI (gpt-4o-mini), Anthropic (claude-sonnet-4)
- AgentState immutable pattern
- ToolProvider ABC → Protocol
- asyncio.Lock replaces threading.Lock

### V1 (Preserved)
- Agent + AdaptivePolicy + StepExecutor + Reducer pipeline
- ReAct and PlanExecute policies
- Graph engine (StateGraph, CompiledGraph)
- Multi-agent orchestration (TeamOrchestrator)
- JSONL Trace system
- Budget tracking
- Checkpoint/Resume
