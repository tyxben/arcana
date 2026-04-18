# Cognitive Primitives

Most agent frameworks treat the LLM as a passenger in its own context window.
The working-set builder decides what to keep, what to compress, what to drop;
the LLM sees the result and reasons from there. When the builder's judgment
fails the LLM — an earlier conclusion compressed to a stub, a load-bearing
fact dropped from the tail — the LLM has no recourse.

Arcana v0.7.0 introduces **cognitive primitives**: runtime services the LLM
can invoke to operate on its own reasoning state.

The MVP ships two primitives:

- **`recall`** — retrieve the original content of an earlier turn, bypassing
  working-set compression.
- **`pin`** (with companion `unpin`) — protect specified content from
  compression in future working sets.

These are intercepted tools — same mechanism as `ask_user`. The LLM sees
them in its tool list, calls them by name; the runtime services the call
directly, bypassing `ToolGateway`.

---

## Why this exists

Context management in a long-running agent is a series of lossy decisions.
v0.6.0 made those decisions **visible**. v0.7.0 makes them **operable**.

Consider two failure modes working-set compression creates:

### 1. The LLM can't read its own history

Turn 3 contains a detailed plan. Turn 8 runs into the compression budget; the
builder demotes turn 3's assistant message from L0 to L3 ("[plan]"). Turn 10
the LLM wants to verify a specific step — it cannot.

Without `recall`, the LLM has three bad options:

- Guess at the original content (hallucination risk)
- Ask the user to repeat (friction, and the user may not remember either)
- Proceed without the information (silent degradation)

With `recall(turn=3)`, the LLM pulls the original message back at full
fidelity for one turn, reasons with it, and moves on.

### 2. The LLM can't protect its own conclusions

At turn 5, the LLM derives three critical facts about the user's system.
Standard fidelity scoring at turn 7 may not rank them highly; compression
kicks in; the facts get compressed or dropped. At turn 10 the LLM is
reasoning with lossy stubs of its own conclusions.

With `pin(content=...)`, the LLM tells the runtime: **these lines are
load-bearing — keep them at full fidelity until I say otherwise.**

---

## `recall` — retrospective probe

### Tool shape

```python
# What the LLM sees
recall(turn: int, include: "all" | "assistant_only" | "tool_calls" = "all")

# What the LLM gets back (structured)
{
    "turn": 3,
    "found": true,
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ],
    "note": null
}
```

### Failure modes — always structured

Per Principle 5, errors are never exceptions. They are actionable tool
results the LLM can reason about:

| Situation | Result |
|-----------|--------|
| Turn out of range | `{found: false, note: "turn out of range: 999, max=5"}` |
| `turn <= 0` | `{found: false, note: "turn must be 1 or greater ..."}` |
| No trace, no in-memory history yet | `{found: false, note: "trace not available ..."}` |

The LLM decides what to do — fall back to working-set content, ask the user,
or accept the gap.

### Filtering with `include`

- `"all"` — every message that was part of that turn (default)
- `"assistant_only"` — just assistant messages; useful when you want your
  own prior conclusion without user noise
- `"tool_calls"` — tool-call assistant messages and their tool results; useful
  for replaying how a prior observation was obtained

---

## `pin` — protected content

### Tool shape

```python
# What the LLM sees
pin(content: str, label: str | None = None, until_turn: int | None = None)
unpin(pin_id: str)

# pin() returns
{
    "pinned": true,
    "pin_id": "p_a1b2c3d4",
    "label": "three load-bearing facts",
    "until_turn": null,
    "already_pinned": false
}
```

### What gets pinned

Pin does **not** match against existing messages. It inserts the pinned
content as a **new, independent block** inside the Working layer, even if
similar text exists elsewhere in the conversation. This is the only reliable
semantic — string matching on live messages would break under paraphrasing.

Principle 2's four-layer structure (Identity / Task / Working / External)
is unchanged. Pin is not a new layer; it is a per-block flag inside
Working.

### Idempotency

Pinning the same content twice is a no-op. Duplicates are detected by
SHA-256 of the content string; the second call returns the existing
`pin_id` with `already_pinned=True`. The original label wins.

### The hard budget cap

Pinned content gets a **hard budget cap**:

```
pinned_tokens <= total_window * pin_budget_fraction   (default 0.5)
```

If the cap would be exceeded, the pin call is rejected with a
structured diagnosis:

```python
{
    "pinned": false,
    "reason": "pin_budget_exceeded",
    "current_pin_tokens": 3200,
    "requested_tokens": 800,
    "cap": 4000,
    "suggestion": "unpin older content (see active pins ...) or pin a shorter excerpt."
}
```

**The framework never auto-unpins. Existing pins are never truncated.** The
LLM decides: unpin something older, shrink the new content, or proceed
without pinning. The framework offers the service; it does not make the
decision.

### `until_turn` — automatic expiry

Pass `until_turn=N` for a scoped pin (e.g., "keep this intact until I finish
the current plan at turn 12"). After turn N, the pin is no longer rendered
into the working set, but it remains in the trace for later replay.

Without `until_turn`, pins live for the session's lifetime (or until the LLM
calls `unpin`).

---

## Enabling primitives

Primitives are opt-in per runtime. An empty list — the default — produces
**zero behavioral change** over v0.6.0.

```python
import arcana

runtime = arcana.Runtime(
    providers={"deepseek": "sk-..."},
    config=arcana.RuntimeConfig(
        cognitive_primitives=["recall", "pin"],
        pin_budget_fraction=0.5,  # default
    ),
)
```

Accepted primitive names: `"recall"`, `"pin"`. Enabling `"pin"` also
exposes `unpin` (they form a symmetric pair). Tool schemas are injected
into the LLM's tool list only for the primitives you opt in to — unused
primitives never bloat context (anti Prohibition 3).

---

## Trace integration

Every primitive invocation emits a `COGNITIVE_PRIMITIVE` trace event:

```json
{
    "event_type": "cognitive_primitive",
    "metadata": {
        "primitive": "pin",
        "args": {"content": "...", "label": "key facts"},
        "result": {
            "pinned": true,
            "pin_id": "p_a1b2c3d4",
            "label": "key facts",
            ...
        }
    }
}
```

### CLI filters

```bash
# Show only cognitive primitive events for a run
arcana trace show <run_id> --cognitive

# Show context decisions with pinned entries flagged as [PIN]
arcana trace show <run_id> --context

# Replay a specific turn; active pins at that turn are listed after
# the prompt snapshot.
arcana trace replay <run_id> --turn 7
```

Example `--cognitive` output:

```
 15. cognitive_primitive   2026-04-18T12:34  recall    turn=3 → 2 messages
 23. cognitive_primitive   2026-04-18T12:35  pin       pin_id=p_a1b2c3d4 label='key facts'
 41. cognitive_primitive   2026-04-18T12:37  unpin     pin_id=p_a1b2c3d4 ok
```

`--context` now flags pinned blocks in the per-message decisions:

```
[PIN] [3] system 86 tokens (kept at L0)
```

`trace replay --turn N` adds an active-pin section after the prompt
snapshot, showing which pins were in effect at that turn.

---

## Constitutional boundaries

The primitives are deliberately constrained.

**What the framework does:**

- Provides the tool specs when opted in
- Intercepts tool calls by name and services them
- Emits structured trace events for every invocation
- Enforces the pin budget cap
- Renders active pins into future working sets

**What the framework does not do:**

- Call a primitive on the LLM's behalf. Every invocation is an explicit
  LLM tool call with a `tool_call_id` and trace record.
- Inject system-prompt hints like *"consider using recall here"* — that
  would be a corridor, not a door.
- Evaluate whether the LLM used primitives appropriately.
- Auto-unpin, truncate pins, or evict pins when budget pressure rises.

This is Principle 9 applied literally: cognitive primitives are services
the LLM may invoke at its discretion. Offering a service is not
prescribing its use.

---

## Relationship with v0.6.0

`recall` (v0.7.0) and `TraceReader.replay_prompt` (v0.6.0) solve related
problems from opposite sides:

| | `recall` (v0.7.0) | `replay_prompt` (v0.6.0) |
|---|-----|-----|
| Who calls it | LLM (at runtime, via tool) | User (offline, via CLI or Python) |
| What it returns | Clean messages for LLM to read | Full `PromptReplay` with decision + snapshot |
| When it runs | Mid-conversation | After the fact |

Both share the same underlying trace infrastructure. `recall` delegates to
the live conversation history first and falls back to `TraceReader` — no
parallel trace-parsing code.

---

## Worked example

```python
import arcana

runtime = arcana.Runtime(
    providers={"deepseek": os.environ["DEEPSEEK_API_KEY"]},
    config=arcana.RuntimeConfig(
        cognitive_primitives=["recall", "pin"],
    ),
)

async with runtime.chat() as c:
    # Turn 1: the LLM derives three key facts
    await c.send("Analyse this codebase and list three load-bearing "
                 "assumptions we'll rely on for the refactor.")

    # The LLM calls pin() here at its discretion with the three facts.

    # Many turns of detailed work...
    for question in long_question_list:
        await c.send(question)

    # Turn N: return to the original plan
    await c.send("Check each load-bearing assumption against what we "
                 "learned so far. If any are contradicted, flag them.")

    # The LLM may call recall(1) to re-read the original turn, or read
    # the pinned block from the current working set.
```

The runtime gets out of the way. The LLM decides whether to recall, whether
to pin, and when to unpin.
