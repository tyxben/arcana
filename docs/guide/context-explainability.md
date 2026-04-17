# Context Explainability

Every LLM call Arcana makes is preceded by a composition decision: which of the
prior messages stay verbatim, which get compressed, which get dropped. Most
frameworks hide that decision. Arcana records it as structured evidence you can
read, diff, and replay offline.

This guide covers the two layers of that evidence:

- **Per-message decisions** — why each input message was kept / compressed / dropped / summarized
- **Prompt snapshots** — the exact `messages` / `tools` / `model` sent to the provider, captured for offline replay

---

## Why this exists

When an agent gives an unexpected answer, the first question is *what did the
model actually see?* Without structured evidence you are left with:

- `llm_request_digest` — a 16-char hash. Tells you nothing about content.
- A free-text `explanation` like "compressed 4 messages" — no indices, no scores, no before/after tokens.
- Re-running the agent with identical inputs — often non-deterministic.

Arcana v0.6.0 closes this gap. Every `ContextDecision` carries a
`MessageDecision` per input message, and (opt-in) the full `LLMRequest` can be
persisted to trace for offline replay.

This is observability, not intervention: the framework **does not** inject these
decisions back into the prompt, **does not** auto-retry based on them, **does
not** change strategy selection. Decisions are retrospective evidence.

---

## Per-message decisions

Every `ContextDecision` now carries a `decisions: list[MessageDecision]`. One
entry per input message, in input order.

### `MessageDecision` fields

```python
class MessageDecision(BaseModel):
    index: int                      # position in original messages
    role: str                       # user / assistant / tool / system
    outcome: Literal["kept", "compressed", "dropped", "summarized"]
    fidelity: Literal["L0", "L1", "L2", "L3"] | None
    relevance_score: float | None
    token_count_before: int
    token_count_after: int           # 0 when dropped
    reason: str                      # machine-readable reason string
```

### Outcomes

| outcome | meaning |
|---------|---------|
| `kept` | Passed through unchanged (head / tail preservation, passthrough strategy) |
| `compressed` | Truncated to a lower fidelity level (L0 → L3) |
| `dropped` | Removed entirely (low relevance, aggressive truncation, no summary budget) |
| `summarized` | Folded into a single summary message by an LLM call |

### Fidelity levels

For compression, the fidelity level tells you how aggressively the content was
truncated:

- `L0` — original content
- `L1` — light truncation (head + tail preserved)
- `L2` — heavy truncation (head only)
- `L3` — replaced with a one-line placeholder

### Reasons

The `reason` field is machine-readable. Typical values:

- `passthrough` — no compression applied
- `tail_preserve_head` / `tail_preserve_tail` — kept as part of the tail window
- `tail_preserve_middle_compressed` — compressed middle message
- `tail_preserve_middle_dropped` — dropped due to budget
- `tail_preserve_no_budget_for_middle` — middle has no summary budget left
- `aggressive_truncate_kept` / `aggressive_truncate_drop` — aggressive strategy
- `llm_summarized_into_single` — rolled into LLM-generated summary
- `stale_tool_result` — Phase 0 pruning hit a stale tool output

### Accessing decisions programmatically

```python
import arcana
from arcana.trace.reader import TraceReader

runtime = arcana.Runtime(providers={"deepseek": "sk-xxx"})

async with runtime.chat() as chat:
    await chat.send("Hello")
    await chat.send("Tell me a long story...")
    r = await chat.send("What was the first thing I said?")

    # Inspect the last context decision
    reader = TraceReader(trace_dir="./traces")
    replay = reader.replay_prompt(run_id=chat.run_id, turn=3)

    for d in replay.context_decision.decisions:
        print(f"[{d.index}] {d.role:10} {d.outcome:12} "
              f"{d.token_count_before}→{d.token_count_after} "
              f"reason={d.reason}")
```

Example output:

```
[0] system     kept         32→32   reason=passthrough
[1] user       kept         8→8     reason=tail_preserve_head
[2] assistant  compressed   240→60  reason=tail_preserve_middle_compressed
[3] user       kept         12→12   reason=tail_preserve_tail
[4] assistant  kept         18→18   reason=tail_preserve_tail
```

---

## Prompt snapshots (opt-in)

Per-message decisions tell you *why* a prompt was built a certain way. Prompt
snapshots tell you *what* prompt was actually sent.

### Enabling

Off by default — prompts can carry PII / secrets and inflate trace size.

```python
runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    config=arcana.RuntimeConfig(
        trace_include_prompt_snapshots=True,
    ),
)
```

When enabled, every `gateway.generate()` / `gateway.stream()` call emits a
`PROMPT_SNAPSHOT` trace event with the complete `LLMRequest`:

```python
class PromptSnapshot(BaseModel):
    turn: int
    model: str
    messages: list[dict]             # full Message.model_dump()
    tools: list[dict]                # provider-ready tool schemas
    response_format: dict | None
    budget_snapshot: BudgetSnapshot | None
```

### When to enable

- **Debugging a specific regression** — turn on for the repro, off otherwise
- **Eval harnesses** — capture golden prompts, compare across model/config changes
- **Internal-only runs** — when PII leakage is not a concern

### When not to enable

- Production with user-generated content (PII risk)
- High-traffic services (trace file growth)
- Any environment where trace files leave trust boundary

---

## Replaying a turn

The `TraceReader` joins `CONTEXT_DECISION` + `PROMPT_SNAPSHOT` events by turn
number into a single `PromptReplay`.

### CLI

```bash
# List turns that have replay evidence
arcana trace replay <run_id>

# Full replay of a specific turn
arcana trace replay <run_id> --turn 3

# Decision only (no prompt content — safe to share)
arcana trace replay <run_id> --turn 3 --decision-only

# Prompt only
arcana trace replay <run_id> --turn 3 --prompt-only

# Raw JSON
arcana trace replay <run_id> --turn 3 --json
```

Default human-readable output includes a per-message table:

```
Turn 3  strategy=tail_preserve  utilization=0.42

  idx  role        outcome       tokens       reason                            fidelity  score
    0  system      kept          32→32        passthrough
    1  user        kept          8→8          tail_preserve_head
    2  assistant   compressed    240→60       tail_preserve_middle_compressed   L2        0.31
    3  user        kept          12→12        tail_preserve_tail
    4  assistant   kept          18→18        tail_preserve_tail

Prompt preview (5 messages, 130 tokens):
  [system]    You are a helpful assistant. Answer clearly and concisely...
  [user]      Hello
  [assistant] [compressed L2] I can help with a variety of topics. Some...
  [user]      Tell me a long story...
  [assistant] [The story content continues with detailed descriptions of...
```

### Programmatic

```python
from arcana.trace.reader import TraceReader

reader = TraceReader(trace_dir="./traces")

turns = reader.list_turns(run_id="abc123")
# -> [1, 2, 3, 4, 5]

replay = reader.replay_prompt(run_id="abc123", turn=3)
# -> PromptReplay with:
#    - prompt_snapshot      (None if flag was off)
#    - context_decision     (always, includes .decisions)
#    - context_report       (token accounting)
#    - budget_snapshot
```

---

## What this gives you

- **Post-mortem a bad answer**: pull the turn, see every message's outcome, find the one that got compressed too aggressively.
- **Eval deltas**: run the same dataset with `tail_preserve` vs `aggressive_truncate`, compare decisions side-by-side.
- **Audit trails**: decisions are structured and serializable — dump to your storage, query with any tool.
- **CI regression tests**: assert that a given input produces expected outcomes (e.g. *the system prompt is always kept at L0*).

---

## Constitutional boundaries

Arcana's philosophy draws a line between **observability** and **intervention**.
Context Explainability is strictly the former:

| Does | Does not |
|------|----------|
| Record every decision as structured evidence | Inject decisions back into the LLM prompt |
| Expose decisions through trace + reader + CLI | Auto-retry based on decisions |
| Let you assert / test against decisions | Change compression thresholds based on "feedback" |
| Snapshot full requests when opted in | Enable snapshots by default (anti-PII / anti-hoarding) |

The LLM remains the sole strategist. The framework's job is to make what
happened legible.

---

## See also

- [Quick Start](quickstart.md)
- [Configuration](configuration.md)
- [Architecture](../architecture.md)
