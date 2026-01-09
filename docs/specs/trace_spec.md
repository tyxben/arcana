# Trace Specification

> Version: 1.0.0
> Status: Active

## Overview

The Trace system provides auditable, reproducible event logging for all Agent operations. Every LLM call, tool execution, and state change is recorded in structured JSONL format.

## Design Principles

1. **Auditability**: Every action must be traceable
2. **Reproducibility**: Traces enable replay of execution
3. **Minimal overhead**: Logging should not impact performance
4. **Schema consistency**: All events follow strict schemas

## Event Schema

### TraceEvent

```python
class TraceEvent:
    # Identifiers
    run_id: str           # Unique run identifier (UUID)
    task_id: str | None   # Optional task grouping
    step_id: str          # Unique step identifier (UUID)
    timestamp: datetime   # UTC timestamp

    # Classification
    role: AgentRole       # system | planner | executor | critic
    event_type: EventType # llm_call | tool_call | state_change | error

    # State digests (Canonical JSON SHA-256, first 16 chars)
    state_before_hash: str | None
    state_after_hash: str | None

    # LLM-related
    llm_request_digest: str | None
    llm_response_digest: str | None
    model: str | None

    # Tool-related
    tool_call: ToolCallRecord | None

    # Budget tracking
    budgets: BudgetSnapshot | None

    # Stop information
    stop_reason: StopReason | None
    stop_detail: str | None

    # Additional context
    metadata: dict
```

### Event Types

| Type | Description |
|------|-------------|
| `llm_call` | LLM request/response |
| `tool_call` | Tool execution |
| `state_change` | Agent state modification |
| `error` | Error occurrence |
| `checkpoint` | State checkpoint for resume |
| `plan` | Plan generation/update |
| `verify` | Verification step |

### Stop Reasons

| Reason | Description |
|--------|-------------|
| `goal_reached` | Task completed successfully |
| `max_steps` | Maximum step limit reached |
| `max_time` | Time budget exhausted |
| `max_cost` | Cost budget exhausted |
| `max_tokens` | Token budget exhausted |
| `no_progress` | Consecutive steps without progress |
| `error` | Unrecoverable error |
| `user_cancelled` | User cancelled execution |
| `tool_blocked` | Tool call blocked by policy |

## Hash/Digest Specification

All digests use **Canonical JSON** hashing:

1. **Serialization**:
   - Keys sorted alphabetically
   - No extra whitespace: `separators=(',', ':')`
   - UTF-8 encoding
   - Floats normalized to 6 decimal places

2. **Hash Algorithm**: SHA-256, truncated to first 16 characters

3. **Implementation**:
```python
def canonical_hash(obj: Any, length: int = 16) -> str:
    json_str = json.dumps(
        normalize(obj),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    )
    return hashlib.sha256(json_str.encode()).hexdigest()[:length]
```

## File Format

- **Format**: JSONL (JSON Lines)
- **Encoding**: UTF-8
- **File naming**: `{run_id}.jsonl`
- **Location**: Configurable via `TRACE_DIR` environment variable

### Example

```jsonl
{"run_id":"abc123","step_id":"step001","timestamp":"2024-01-15T10:30:00Z","event_type":"llm_call","model":"gemini-2.0-flash","llm_request_digest":"a1b2c3d4e5f6g7h8"}
{"run_id":"abc123","step_id":"step002","timestamp":"2024-01-15T10:30:01Z","event_type":"tool_call","tool_call":{"name":"search","args_digest":"x1y2z3"}}
```

## API

### TraceWriter

```python
writer = TraceWriter(trace_dir="./traces", enabled=True)

# Write event
writer.write(event)

# Create context
ctx = writer.create_context(task_id="task-001")

# List runs
runs = writer.list_runs()
```

### TraceReader

```python
reader = TraceReader(trace_dir="./traces")

# Read all events
events = reader.read_events(run_id)

# Filter events
llm_calls = reader.filter_events(
    run_id,
    event_types=[EventType.LLM_CALL],
)

# Get summary
summary = reader.get_summary(run_id)
```

## Best Practices

1. **Always create a TraceContext** before starting a run
2. **Log all LLM calls** with request/response digests
3. **Include budget snapshots** for cost tracking
4. **Set stop_reason** when execution ends
5. **Use step_id** to group related events
