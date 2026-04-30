# API Stability

This page tells you which Arcana APIs you can rely on across releases
and which ones may change without notice. It is the user-facing
distillation of `specs/v1.0.0-stability.md`.

The short version:

| If you import from… | Then it is… |
|---|---|
| `arcana` (top level) | **Stable** |
| `arcana.contracts.*` | **Stable** |
| `arcana.runtime.*`, `arcana.gateway.*`, `arcana.tool_gateway.*`, `arcana.context.*`, `arcana.routing.*`, `arcana.eval.*`, `arcana.trace.*`, `arcana.streaming.*`, `arcana.multi_agent.*` | **Internal — no stability promise** |
| Anything starting with `_` | **Internal** — period |

If you are not sure, check the explicit list below or open a stability
question in the issue tracker.

---

## Stable surfaces

These names ship a stability promise: rename, removal, or
signature-breaking change is a major version bump (e.g. `1.x` → `2.0`),
not a minor.

### Top-level package — `arcana.*`

```python
import arcana

arcana.run                # one-shot task
arcana.Runtime            # resource container
arcana.Budget             # budget configuration
arcana.AgentConfig        # agent configuration
arcana.RuntimeConfig      # runtime configuration
arcana.ChatSession        # multi-turn session
arcana.ChatResponse       # chat response model
arcana.ChainStep          # pipeline step
arcana.ChainResult        # pipeline result
arcana.BatchResult        # batch result wrapper
arcana.AgentPool          # multi-agent pool
arcana.Message            # conversation message (re-export)
arcana.MessageRole        # role enum (re-export)
arcana.tool               # @arcana.tool decorator
arcana.Tool               # tool wrapper class
arcana.RunResult          # run() return type
arcana.StreamEvent        # streaming event
arcana.StreamEventType    # streaming event type enum
arcana.StateGraph         # graph engine
```

### Runtime methods

```python
runtime = arcana.Runtime(...)

await runtime.run(...)
await runtime.run_batch(...)
async with runtime.chat() as session: ...
await runtime.chain([ChainStep(...), ...])
async with runtime.collaborate() as pool: ...
runtime.session(...)
await runtime.close()
```

`runtime.team()` was removed in v1.0.0; use `runtime.collaborate()`
instead. The migration recipe lives in
[Multi-Agent Collaboration → Migration from `runtime.team()`](multi-agent.md#migration-from-runtimeteam).

### ChatSession surface

```python
async with runtime.chat() as session:
    await session.send(...)
    session.stream(...)

    session.history             # read-only dict copy
    session.seed_history(...)   # cold-start history injection (v0.9.x+)
    session.total_cost_usd
    session.total_tokens
    session.turn_count
    session.message_count
    session.session_id
    session.max_history
```

### Contracts — `arcana.contracts.*`

The data layer is part of the stable surface. Field renames are major
bumps; new optional fields with defaults are minor bumps.

| Module | Stable names |
|---|---|
| `arcana.contracts.tool` | `ToolSpec`, `ToolCall`, `ToolResult`, `ToolError`, `ToolErrorCategory`, `SideEffect`, `ASK_USER_TOOL_NAME` |
| `arcana.contracts.llm` | `Message`, `MessageRole`, `LLMRequest`, `LLMResponse`, `ContentBlock`, `ModelConfig` |
| `arcana.contracts.turn` | `TurnFacts`, `TurnAssessment` |
| `arcana.contracts.context` | `ContextBlock`, `ContextDecision`, `MessageDecision`, `ContextReport`, `ContextStrategy`, `ContextLayer`, `TokenBudget`, `WorkingSet`, `StepContext` |
| `arcana.contracts.diagnosis` | `ErrorDiagnosis`, `ErrorCategory`, `ErrorLayer`, `RecoveryStrategy` |
| `arcana.contracts.streaming` | `StreamEvent`, `StreamEventType` |
| `arcana.contracts.runtime` | `RuntimeConfig` |
| `arcana.contracts.cognitive` | `RecallRequest`, `RecallResult`, `PinRequest`, `PinResult`, `UnpinRequest`, `UnpinResult`, `PinEntry`, `PinState` |
| `arcana.contracts.trace` | `TraceEvent`, `EventType`, `BudgetSnapshot`, `PromptSnapshot`, `ToolCallRecord` |

### Tool authoring

```python
from arcana import tool
from arcana.contracts.tool import SideEffect, ToolErrorCategory

@tool(side_effect=SideEffect.READ)
def search(query: str) -> str:
    ...
```

`@arcana.tool` decorator surface, `ToolSpec` field set, `SideEffect`
enum values, and `ToolErrorCategory` enum values + retry-eligibility
rule are stable.

### CLI

```
arcana run ...
arcana init ...
arcana trace show ...
arcana trace replay ...
arcana trace explain ...
arcana trace flow ...
arcana trace pool-replay ...
```

Invocation form, exit codes, and primary flag set are stable. Output
formatting is best-effort — script against `--json` outputs, not
human-readable text, if you need machine-parseable results.

---

## Internal — do not import

These imports work today but will break across releases without
deprecation warnings. User code that reaches into them assumes the
maintenance burden when internals change.

```python
# All of these are internal — DO NOT depend on them in user code:
from arcana.runtime.conversation import ConversationAgent  # internal engine
from arcana.runtime.agent import Agent                     # V1 legacy
from arcana.gateway.openai_compat import ...               # provider plumbing
from arcana.tool_gateway.gateway import ToolGateway        # tool plumbing
from arcana.context.builder import WorkingSetBuilder       # context plumbing
from arcana.streaming.accumulator import StreamAccumulator # streaming impl
from arcana.multi_agent.channel import Channel             # collab impl
```

If you find yourself reaching into one of these, please file an issue —
that is a signal the public surface is missing something.

---

## Asymmetric cases

A few places trip people up:

### `arcana.contracts.*` is stable, but `arcana.contracts` itself is a sub-package

The convention "anything under a sub-package is internal" does **not**
apply to `arcana.contracts`. The contracts layer is the data-model
stable surface. Treat it the same way you treat `arcana.*`.

### `arcana.runtime.conversation.Message` works, but is **not** stable

`Message` and `MessageRole` happen to be reachable through
`arcana.runtime.conversation` because that module imports them
internally. This works in CPython today and may keep working for some
time, but it carries no stability promise. The internal `from ... import`
line that exposes them can move at any time.

```python
# ✓ Stable:
from arcana import Message, MessageRole
from arcana.contracts.llm import Message, MessageRole

# ✗ NOT stable — happens to work, no promise:
from arcana.runtime.conversation import Message
```

### CLI output formatting

The CLI's *invocation form* (subcommands, flags, exit codes) is stable.
The exact text it prints is not. If you need to consume CLI output
programmatically, use the `--json` flag where available; the JSON
schema is more stable than the human-readable rendering.

---

## Versioning policy (effective from v1.0.0)

- **Major (X.0.0)**: any break to the surfaces listed above. Field
  rename, signature change, removal, behaviour change visible to user
  code.
- **Minor (0.X.0)**: additive only. New public method, new contract
  field with default, new provider, new event type.
- **Patch (0.0.X)**: bug fix, internal refactor invisible to public
  surface.

Pre-v1.0.0, Arcana uses a relaxed semver where minor bumps may include
breaking changes accompanied by a `DeprecationWarning` for at least one
prior release. From v1.0.0 onward the rules above apply strictly.

---

## Deprecation policy

When a stable name is going to be removed in a future major bump:

1. **One minor release minimum** ships with a `DeprecationWarning`
   before removal. Pre-v1.0.0 this is one minor; post-v1.0.0 this is
   one minor on the current major.
2. The CHANGELOG entry that adds the deprecation includes a
   `Migration` section with before/after code snippets.
3. The successor name and its replacement path are live in the same
   release that adds the deprecation — never "use the future API once
   we build it."
4. Pending removals are tracked in `specs/vX.0.0-removals.md`.

If you encounter a `DeprecationWarning` from Arcana, the migration
recipe is in the CHANGELOG entry that introduced the deprecation. The
warning itself includes a one-line pointer.

---

## Reporting a stability bug

If you find a name on the stable list that breaks across a minor or
patch release, that is a bug. Please file an issue with:

- The name (e.g. `arcana.Runtime.chat`)
- The two versions (e.g. `0.9.0` → `0.9.1`)
- A minimal reproduction

Conversely, if you find yourself depending on something *not* on the
stable list, also file an issue — that is a signal we are missing a
public surface, not that you should keep depending on the internal one.

---

## Source of truth

This page is generated from `specs/v1.0.0-stability.md`, which carries
the full constitutional reasoning and the implementation checklist. If
the two ever disagree, the spec wins; please file a documentation issue
so we can re-sync this page.
