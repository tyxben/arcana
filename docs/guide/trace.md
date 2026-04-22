# Trace — debugging what actually happened

Every Arcana run writes a JSONL trace file to `./traces/<run_id>.jsonl`.
Unlike a log, a trace is structured evidence: each LLM turn carries its
own context decision, prompt snapshot (opt-in), assistant output,
tool-call outcomes, and runtime verdict, joined by explicit causal links.

This page shows the debug loop: one question → one command.

## TL;DR — the four commands

| Question | Command |
|---|---|
| What happened in turn N? | `arcana trace explain <run_id> --turn N` |
| What's the shape of this run? | `arcana trace flow <run_id>` |
| What went wrong and where? | `arcana trace show <run_id> --errors --explain` |
| List runs I have on disk | `arcana trace list` |

Pair any of them with `--dir /some/path` to point at a non-default trace
directory, and `--json` (on `explain` / `flow`) to pipe into tooling.

## Turn on dev mode

During development, turn on full capture — the trace records the exact
prompt sent to each LLM call so `explain` can replay offline:

```python
runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    tools=[...],
    dev_mode=True,       # implies trace_include_prompt_snapshots=True
    trace_dir="./traces",
)
```

`dev_mode` is `False` by default because prompt snapshots can contain
PII / secrets and inflate trace files. Leave it off in production; rely
on `trace_include_prompt_snapshots=True` only where legal & safe.

## `arcana trace explain` — the full story of one turn

```bash
$ arcana trace explain 7a3b-… --turn 2
Turn 2 — run 7a3b-… · model deepseek-chat
step_id=9f01-…

Inputs
  messages: 8 → 6  (2 compressed) · tokens: 2450 → 1820
  [PIN 3] user 120 tokens (kept at L0)
  prompt: 6 messages, 4 tools available
    [-3] user: what's the current weather in Tokyo
    [-2] assistant: I need to look that up
    [-1] tool: {"temp_c": 18, "condition": "cloudy"}

LLM output
  thinking: The tool returned valid weather data, I can summarize now
  text: It's 18°C and cloudy in Tokyo.
  tool_calls: 0

Tool results (0)

Runtime verdict
  completed: True   failed: False
  confidence: 0.92
  completion_reason: answer provided with tool-verified data
```

Everything you need to judge that single turn is in one place: the
inputs (curated messages + context decision), the raw LLM output, tool
outcomes, and the runtime's verdict. No chasing through a log.

Without `dev_mode=True`, the "prompt: ..." preview falls back to a hint
— `explain` still works, it just doesn't have the raw prompt text.

## `arcana trace flow` — the causal spine

```bash
$ arcana trace flow 7a3b-…
Flow — run 7a3b-… · 3 turns

  Turn 1  deepseek-chat
    ├─ ✓ web_search 812ms
    │
  Turn 2  deepseek-chat
    ├─ ✓ get_weather 340ms
    │
  Turn 3 (completed)  deepseek-chat
  → stop: completed
```

Follow the parent_step_id links to see exactly which tools each turn
fired, in order. When a tool call failed it shows `✗`, and the turn that
completed the run is tagged.

## `arcana trace show --errors --explain` — triage

When something broke, this is your first stop:

```bash
$ arcana trace show 7a3b-… --errors --explain
Trace: 7a3b-…
Events: 1 (filtered from 47)

    1. error                2026-04-22T10:25:44 None

━━ Error at turn 2 ━━
Turn 2 — run 7a3b-… · model deepseek-chat
…full explain view for the failing turn…
```

Each error event is resolved to its turn (via `parent_step_id`) and the
full `explain` view for that turn is printed inline. No error bouncing
between grep windows.

## What's in a trace

| Event type | Written by | Links to |
|---|---|---|
| `turn` | ConversationAgent | previous `turn` (spine) |
| `context_decision` | WorkingSetBuilder | current `turn` |
| `prompt_snapshot` *(opt-in)* | ConversationAgent | current `turn` |
| `tool_call` | ToolGateway | triggering `turn` |
| `cognitive_primitive` | CognitiveHandler | current `turn` |
| `llm_call` | OpenAI-compat provider | — |
| `error` | runtime | offending `turn` (when applicable) |

The `parent_step_id` field on every event is what makes `flow` and
`explain` possible without heuristics.

## Using the reader from Python

For custom analysis or CI gates:

```python
from arcana.trace.reader import TraceReader

reader = TraceReader(trace_dir="./traces")
bundle = reader.collect_turn(run_id, turn=2)

print(bundle["turn_event"].metadata["assessment"])
for tc in bundle["tool_calls"]:
    print(tc.tool_call.name, tc.tool_call.error)
```

`collect_turn` returns a dict with keys: `turn_event`, `context_decision`,
`prompt_snapshot`, `tool_calls`, `cognitive`, `errors`, `all`.

## Pool traces

For multi-agent pools (`runtime.collaborate()`), every event carries a
`metadata["source_agent"]` tag. Scope any of the above with `--agent`:

```bash
arcana trace explain <run_id> --agent researcher --turn 3
arcana trace show <run_id> --agent critic --errors --explain
```

See `docs/guide/multi-agent.md` for the full pool model.
