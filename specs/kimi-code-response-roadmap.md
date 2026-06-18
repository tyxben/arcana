# Kimi Code Response Roadmap

**Status**: Draft implementation plan

**Date**: 2026-05-27

**Owns**: Arcana's concrete response to Kimi Code-style product pressure:
coding-agent ergonomics, skills, hooks, MCP permissions, subagents, and
session trace UX, without violating the Arcana Constitution.

## Position

Kimi Code validates the product direction: serious agents need local workflow
integration, permissioned tools, reusable skills, lifecycle hooks, isolated
subtasks, MCP, and inspectable session records.

Arcana should absorb those capabilities as OS services, not as a fixed coding
workflow. The framework must keep its core law:

- no default main-agent/subagent topology
- no hidden planner inside guardrails or hooks
- no trust by protocol membership alone
- no context hoarding through always-on skills or full remote catalogs
- no silent weakening of tool, output, budget, or provenance contracts

## Related Planning Artifacts

This file is the **canonical, code-linked roadmap** (each item carries a
`done`/pending marker and points at the source + test files). Two companion
artifacts orbit it:

- `specs/arcana-agent-architecture-evolution-plan.html` — an **external-trends
  design note** (OpenAI Agents / MCP / A2A / AG-UI / LangGraph / AgentCore /
  AlphaEvolve / DGM), not a tracker. Its Phase 1–4/6 items are re-skins of the
  phases below; treat this roadmap, not the HTML, as the source of truth for what
  is actually built.
- `specs/constitution-amendment-6-self-evolution-boundaries.md` — the **one
  genuinely net-new pillar** from that note (self-evolution), captured as a
  *candidate* constitutional boundary (principles only, mechanism deferred). No
  self-evolution code is in scope for this roadmap until those contracts/evals
  land.

## Agent Team Findings

Four focused reviews were run against the current repo before this roadmap was
finalized.

| Track | Conclusion |
|-------|------------|
| Runtime / gateway audit | Arcana already has the right boundary shape (`ToolGateway`, `SideEffect`, `ToolErrorCategory`, `ExecutionChannel`), but the current runtime has four honesty bugs: per-run tools are ignored, stream usage is not accumulated, parallel chain budgets are not shared, and `max_history=0` keeps all history. |
| Contracts / protocol design | The next layer should be contracts-first: `CapabilityProvenance`, `PermissionPolicy`, `GuardrailDecision`, `SkillSpec`, and traceable protocol import decisions. MCP should be one adapter into that model, not a special trust path. |
| Subagents / trace UX | Kimi-like subagents should be an optional facade over `AgentPool` and `ChatSession`, with explicit `ask()` / `as_tools()` calls. No default main agent, scheduler, or supervisor policy. Session bundles should correlate multiple run JSONL files through metadata. |
| Safety / tests | Release order should be: P0 runtime fixes, trace provenance, permission policy, protocol bridge, skills, hooks, then subagent service. Each step needs invariant tests proving policy denial, approval, provenance, and guardrail behavior are visible and non-bypassable. |

## Release Train

### Phase 0: Trust the Existing Surface

Goal: make the current stable API honest before adding new product layers.

Required fixes:

1. `Runtime.run(..., tools=...)` and `Runtime.session(..., tools=...)` must
   actually merge per-run tools into the active tool gateway.
2. `Runtime.stream(...)` must account for token and cost usage in runtime totals.
3. `Runtime.chain(...)` must enforce chain-level budgets across parallel branch
   groups.
4. `ChatSession(max_history=0)` must retain zero non-system messages.

Primary files:

- `src/arcana/runtime_core.py`
- `tests/test_runtime_core.py`
- `tests/test_chat_session.py`

Exit criteria:

- targeted regression tests fail before the fix and pass after
- no public API shape change
- no new abstractions introduced to hide the bugfixes

### Phase 1: Capability Boundary Layer

Goal: build the shared substrate for MCP, shell/write approvals, hooks, and
remote tools.

Work items:

1. `PermissionPolicy`
   - allow/deny rules
   - wildcard tool patterns
   - scopes: call, session, project, global
   - match fields: tool name, capability, origin, server name, side-effect class
   - side-effect defaults
   - denial reason surfaced as structured feedback
   - approval decisions traced

2. `ProtocolBridge`
   - normalize MCP/connectors/remote tools into Arcana `ToolSpec` providers
   - preserve origin, authority, auth context, side-effect class, approval policy,
     provenance, timeout, and protocol error class
   - refuse or conservatively downgrade remote capabilities missing required
     metadata
   - re-apply policy on dynamic discovery updates before exposing new tools

3. Lifecycle hooks / guardrails
   - typed hook events at runtime boundaries
   - `PreToolUse` can block by returning structured feedback
   - observer hooks fail open
   - blocking hooks are never hidden planners
   - guardrail failures are never allowed to silently weaken permission policy

Primary files:

- `src/arcana/contracts/tool.py`
- `src/arcana/contracts/trace.py`
- `src/arcana/contracts/capability.py` or
  `src/arcana/contracts/permission.py`
- `src/arcana/tool_gateway/gateway.py`
- `src/arcana/mcp/protocol.py`
- `src/arcana/mcp/setup.py`
- `src/arcana/mcp/tool_provider.py`
- `src/arcana/runtime/hooks/base.py`

Exit criteria:

- MCP tools and local tools go through the same visible policy path
- prompt-injected tool output is treated as untrusted tool output, not instruction
- project-level stdio MCP requires explicit trust or conservative refusal

### Phase 2: Skills as Lazy Capabilities

Goal: provide reusable workflow knowledge without turning context into a
warehouse.

Work items:

1. `SkillSpec`
   - name, description, when_to_use, arguments, source path, trust scope,
     token estimate, invocation mode
2. `SkillRegistry`
   - project, user, extra, built-in scopes
   - deterministic shadowing rules
   - provenance for every loaded skill
3. Working-set integration
   - inject only selected skill summaries or bodies
   - record selection reasons in `ContextDecision`
   - slash-command/manual invocation can force a skill, still with provenance

Primary files:

- `src/arcana/contracts/skill.py`
- `src/arcana/context/builder.py`
- `src/arcana/runtime/conversation.py`
- `tests/test_skills.py`

Exit criteria:

- skills do not always enter the prompt
- skill body cannot impersonate system/user/developer authority
- skill selection is auditable

### Phase 3: Optional Subtask Isolation

Goal: provide Kimi-like subtask isolation without making main/subagent topology a
framework default.

**Status: core + correlation + permission/approval done**
(`arcana.experimental.subagents`). Work items 1 (explicit subagent API), 2
(isolated context + `SubagentResult`), 3 (full — per-subtask budget caps,
narrowed/inherited `PermissionPolicy`, and approval before high-authority
delegation), and the trace-correlation half of item 4 are implemented and
tested. The facade lives under `arcana.experimental` (not on the stable
`Runtime` surface) while its semantics incubate — open question #3 is resolved
by reusing `ChatSession` per ask rather than a purpose-built frame.
**Still deferred**: item 4's `bundle_id` is stamped but `delegated_by_run_id`
is caller-threaded (auto-capture across arbitrary parent agents deferred), and
graduation from `arcana.experimental` to the stable surface.

Work items:

1. Explicit subagent API
   - `runtime.subagents(...)` returns a `SubagentService`
   - `SubagentService.add(...)` wraps the existing `AgentPool.add(...)`
   - `SubagentService.ask(name, task)` is user-directed orchestration
   - `SubagentService.as_tool(name)` exposes delegation only when the caller
     chooses to hand it to an LLM
2. Isolated context
   - subtask sees task packet and granted tools only
   - subtask output returns `SubagentResult(agent, content, run_id, tokens,
     cost, trace_refs)`
   - no recursive subtask spawning in v1
3. Budget and permission
   - per-subtask budget cap
   - inherited or narrowed permission policy
   - approval request before high-authority delegation
4. Trace correlation
   - keep the existing one-run-per-JSONL trace shape
   - add metadata such as `session_id`, `bundle_id`, `source_agent`,
     `delegated_by_run_id`, and `delegated_by_step_id`
   - do not add stable fields to `TraceEvent` until the metadata shape proves
     itself

Exit criteria:

- no built-in planner/executor topology
- no auto-cancel/retry supervision policy
- traces can reconstruct parent/subtask relationship

Primary files:

- `src/arcana/runtime_core.py`
- `src/arcana/multi_agent/agent_pool.py`
- `src/arcana/contracts/multi_agent.py`
- `src/arcana/trace/reader.py`
- `tests/test_subagent_service.py`
- `tests/test_trace_pool.py`

### Phase 4: Session Trace UX and CLI

Goal: make Arcana runs inspectable and resumable enough for real coding-agent
workflows.

Work items:

1. Session bundle
   - `arcana trace bundle list`
   - `arcana trace bundle show <bundle_id>`
   - `arcana trace bundle export <bundle_id> --out bundle.zip`
   - include turns, tool calls, context decisions, approvals, prompt snapshots
     when enabled, subtask events, and provenance
2. CLI surfaces
   - `arcana mcp add/list/remove/auth`
   - `arcana skills list/show`
   - `arcana chat` as a minimal interactive loop
3. Human-readable trace
   - timeline by turn
   - filter by agent/subtask/branch/tool
   - redact credentials by default
4. Wire trace
   - start with normalized provider request/response evidence
   - raw HTTP is opt-in only and redacted by default
   - V2 `ConversationAgent` must pass trace context into the gateway before
     provider-level wire evidence is considered complete

Exit criteria:

- debugging a failed agent run does not require reading raw JSONL
- exported bundles warn when they may contain secrets or private tool output

Primary files:

- `src/arcana/cli/main.py`
- `src/arcana/trace/reader.py`
- `src/arcana/trace/writer.py`
- `src/arcana/gateway/providers/openai_compatible.py`
- `tests/test_trace_replay.py`
- `tests/test_trace_debug.py`

## Non-Goals

- No mandatory coding-agent TUI in the runtime package.
- No default YOLO mode.
- No automatic main-agent/subagent hierarchy.
- No automatic MCP project config execution without trust.
- No always-on project skill injection.
- No benchmark score as a substitute for runtime policy.

## First Iteration Slice

The first implementation slice should be deliberately small:

1. Fix the Phase 0 bugs. **(done)**
2. Add `PermissionPolicy` contracts and wire them only into `ToolGateway.call`.
   **(done)**
3. Add tests for allow, deny, confirmation-required, callback failure, and
   traced denial. **(done)**
4. Fix MCP error classification so JSON-RPC validation, permission, and logic
   failures do not all become retryable transport failures. **(done)**
5. Extend MCP tool registration with origin/provenance metadata if it can be
   done without breaking `ToolSpec`; otherwise add a wrapper metadata structure
   and keep `ToolSpec` stable. **(done)** — resolved to the on-`ToolSpec` path:
   adding an optional `provenance: ToolProvenance | None` field is a sanctioned
   minor bump (`specs/v1.0.0-stability.md` §1.4, "field addition with default"),
   so no wrapper was needed. `mcp_tool_to_arcana_spec` stamps
   `ToolProvenance(origin="mcp", server_name=...)`; `from_tool_spec` reads it as
   the single source of truth; the gateway emits it to trace metadata. This
   activates the previously-dead `origins` / `server_names` policy dimensions.

This gives Arcana the safety substrate needed before accepting more remote tools,
skills, hooks, or subtask delegation.

Two adjacent gaps were surfaced while landing item 5 and are **deliberately
carved out** of this slice (tracked as P1 rows below), so the recorded-provenance
behavior is not mistaken for more than it is:

- ~~The dynamic-discovery `on_tools_changed` → `ToolRegistry` re-registration
  bridge does not exist (`setup_mcp_tools` installs no handler), so dynamically
  discovered/removed MCP tools never reach the registry.~~ **The bridge has
  since landed**: `setup_mcp_tools` installs an `on_tools_changed` handler that
  removes stale MCP providers for the server, re-invokes the admission gate,
  re-stamps provenance, and emits fresh `CAPABILITY_ADMISSION` trace events
  with capability digests.
- ~~This slice **records** provenance and lets policy act on it; it does **not**
  implement the Phase 1 item-2 *exposure gate*.~~ **The exposure gate has since
  landed** (conservative side-effect downgrade + provenance-integrity refuse,
  traced as `CAPABILITY_ADMISSION`); see the backlog row below. Only the
  dynamic-discovery bridge remains carved out.

## Implementation Backlog

| Priority | Item | Files | Tests |
|----------|------|-------|-------|
| P0 | ~~Per-run tools actually merge with runtime tools, including `ChainStep.tools`.~~ **done** | `runtime_core.py` | `tests/test_runtime_core.py` |
| P0 | ~~Runtime stream usage accumulates tokens/cost even on exceptions or early generator close.~~ **done** | `runtime_core.py`, `runtime/conversation.py` | `tests/test_runtime_core.py` |
| P0 | ~~Parallel chain budget has a defined shared-budget behavior.~~ **done** (parallel groups split remaining chain budget across branches before applying per-step caps) | `runtime_core.py` | `tests/test_runtime_core.py` |
| P0 | ~~`ChatSession(max_history=0)` keeps zero non-system messages.~~ **done** | `runtime_core.py` | `tests/test_chat_session.py` |
| P1 | ~~Add `PermissionPolicy` / approval contracts and wire into `ToolGateway.call`.~~ **done** | `contracts/*`, `tool_gateway/gateway.py` | `tests/test_permission_policy.py`, `tests/test_tool_gateway.py` |
| P1 | ~~Normalize MCP provenance and fix MCP error classification.~~ **done** (origin/server via `ToolProvenance` on `ToolSpec`; error classification via `mcp_error_to_tool_error`). | `mcp/protocol.py`, `mcp/tool_provider.py` | `tests/test_mcp.py` |
| P1 | Trace metadata for permission **(done — `permission_decision` + `provenance` keys)**, protocol-discovery decisions **(done — `PROTOCOL_DISCOVERY` + `ProtocolDiscoveryRecord` for MCP initial discovery, refresh, ignore, and failure)**, and tool-call guardrail decisions **(done — `GUARDRAIL_DECISION` + `GuardrailDecisionRecord` emitted at ToolGateway boundary)**. | `contracts/trace.py`, `mcp/setup.py`, `tool_gateway/gateway.py`, `trace/*` | `tests/test_constitutional_invariants.py`, `tests/test_mcp.py`, `tests/test_mcp_dynamic_discovery.py`, `tests/test_tool_gateway.py` |
| P1 | ~~**Dynamic-discovery bridge**: install an `on_tools_changed` handler in `setup_mcp_tools` that re-registers (unregisters stale + registers fresh, re-hashes, re-authorizes, re-traces, re-stamps provenance) into the same `ToolRegistry` on `tools/list_changed`.~~ **done** (`setup_mcp_tools` now wires `MCPClient(on_tools_changed=...)`, removes stale providers by MCP provenance, re-registers fresh tools through `register_mcp_tools`, and traces `capability_digest` for each admission). | `mcp/setup.py` | `tests/test_mcp_dynamic_discovery.py` |
| P1 | ~~**Provenance exposure gate**: refuse or conservatively downgrade imported capabilities missing required metadata before exposing them to the LLM (Phase 1 item 2).~~ **done** (admission gate in `register_mcp_tools`). Scoped honestly to what the contracts can express: **side-effect** — a tool with no authoritative `readOnlyHint` is exposed as WRITE + confirmation (the old name-keyword heuristic, a silent v3.5 downgrade, was deleted); **provenance** — a spec claiming a remote origin without a `server_name` is refused; both decisions emit a `CAPABILITY_ADMISSION` trace event. **approval policy** deliberately NOT gated — it is not modeled per-tool (only global `PermissionPolicy` + `requires_confirmation`), so requiring it would be premature structuring; deferred to ProtocolBridge. | `mcp/setup.py`, `mcp/protocol.py`, `contracts/mcp.py`, `contracts/trace.py` | `tests/test_constitutional_invariants.py` (`TestImportedCapabilityExposureGate`), `tests/test_mcp.py` |
| P2 | ~~Add Skills v1: `SKILL.md` discovery, digest, explicit/lazy loading, context provenance.~~ **done** for runtime/context surface (`SkillSpec`, `SkillRegistry`, `RuntimeConfig.skill_paths`, `Runtime.run(skills=...)`, `arcana.run(skill_paths=..., skills=...)`, auto/explicit/slash selection, `ContextDecision.skill_selections`, labeled non-authoritative skill context). CLI helpers remain a later tooling slice. | `contracts/skill.py`, `context/builder.py`, `runtime_core.py`, `sdk.py` | `tests/test_skills.py` |
| P2 | Add lifecycle hooks v1; block only at explicit boundaries. **Two halves done.** (a) **Blocking** = tool-call guardrails (`Runtime(guardrails=...)`, `ToolGuardrailRequest`, `GuardrailDecision`, block/warn/redact/require-approval in `ToolGateway`). (b) **Observing** = typed V2 lifecycle hooks (`turn_start`/`turn_end`/`tool_start`/`tool_end` with frozen `contracts/lifecycle.py` payloads, plus `run_start`/`run_end`/`error` extended to the chat path) emitted by `ConversationAgent` through the existing `Runtime.on(...)` `_EventBus`, now **fail-open** (a raising observer is logged and ignored). Observers have **no return channel** — they cannot block or rewrite the run, which is what keeps them from becoming hidden planners (v3.6). The direct-answer fast path emits one logical completed turn. Constitutional invariant `TestLifecycleHooksAreObservers` pins observer-only + fail-open. **Deferred**: lifecycle hooks for the V1 `Agent`/`RuntimeHook` protocol stay V1-only; no new context/budget-boundary events (already trace-covered). | `contracts/lifecycle.py`, `contracts/guardrail.py`, `runtime/conversation.py`, `runtime_core.py`, `tool_gateway/gateway.py` | `tests/test_lifecycle_hooks.py`, `tests/test_constitutional_invariants.py`, `tests/test_tool_gateway.py` |
| P3 | **Optional subagent facade + delegation tools** — **done (core + correlation + permission/approval)**, incubated under `arcana.experimental.subagents` (`subagents()` factory / `SubagentService` / `SubagentResult`) rather than on the stable `Runtime` surface while semantics harden (open question #3). Landed: isolated single-shot `ask()` (fresh session per call — no shared history), no-recursion enforcement via a `contextvars` delegation flag, opt-in `as_tool()` delegation (conservative `write` side-effect; refuses recursively; `requires_approval` marks the spec confirmation-required so the parent gateway gates before the subagent runs), per-subtask `Budget` caps (fresh per-ask) vs. service-level shared budget, trace correlation (`source_agent` / `bundle_id` / optional `delegated_by_run_id` via the generalized `_PoolTaggedTraceWriter`), and **per-subagent `PermissionPolicy`** (service-level default, overridden — not merged — per subagent) + **service-level `approval_handler`** wired into the subagent session gateway (DENY non-bypassable, ASK/write gate behind approval, no handler ⇒ `CONFIRMATION_REQUIRED`). Constitutional invariant `TestSubagentServiceImposesNoTopology` pins no-topology, no-recursion, and authority confinement. **Deferred**: auto-capture of `delegated_by_run_id` across arbitrary parent agents (caller-threaded); graduation from `arcana.experimental` to the stable surface. | `experimental/subagents.py`, `runtime_core.py` | `tests/test_subagent_service.py`, `tests/test_constitutional_invariants.py`, `tests/test_trace_pool.py` |
| P3 | Add session bundle and normalized wire trace CLI. | `cli/main.py`, `trace/*`, `gateway/*` | `tests/test_trace_replay.py`, `tests/test_trace_debug.py` |

## Test Matrix

The roadmap is not complete unless these behaviors are covered:

- Tool output and remote-agent messages remain labeled evidence; they cannot
  become system, developer, or user instructions.
- Permission denial, missing approval, and approval denial do not execute tools
  and surface non-retryable structured feedback.
- Remote capabilities without an authoritative side-effect declaration are
  exposed conservatively (WRITE + confirmation), and those claiming a remote
  origin with no server identity are refused outright — both before reaching
  the LLM, both traced. (**done** — `TestImportedCapabilityExposureGate`.)
  Per-tool approval-policy gating is deferred (not modeled; ProtocolBridge).
- Dynamic MCP tool changes are re-hashed, re-authorized, and traced before
  becoming visible. (**done** — `TestMCPSetupDynamicRegistryBridge`.)
- Guardrails can block, redact, warn, or request approval; they cannot rewrite
  the goal, choose a new plan, or insert a hidden workflow.
- Skills do not imply shell, write, MCP, or remote-agent authority. (**done**
  for Skills v1 runtime/context surface — selected skill bodies are labeled as
  reusable knowledge, not system/developer/user instructions.)
- Subagent calls inherit narrowed permissions, get separate budget accounting,
  and preserve parent/child trace correlation.
- Session bundles can replay the relevant turns, tools, approvals, policy
  decisions, provenance, and budget overrun evidence.

## Open Questions

1. Should `PermissionPolicy` live in `contracts/tool.py`, a new
   `contracts/permission.py`, or `runtime/security.py`?
2. Should skills be a public stable API in v1.x, or incubate under
   `arcana.experimental.skills` until their selection semantics harden?
3. Should subtask isolation reuse `ChatSession` internally or have a smaller
   purpose-built execution frame?
4. Should `ProtocolBridge` be MCP-first, or define a generic remote capability
   interface before revisiting MCP?
