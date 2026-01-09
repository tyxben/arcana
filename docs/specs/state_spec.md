# State Specification

> Version: 1.0.0
> Status: Active

## Overview

The State system manages agent execution state, enabling checkpointing, resume, and replay capabilities. State is tracked at multiple levels: run, task, and step.

## Design Principles

1. **Immutable snapshots**: State changes create new snapshots
2. **Hash verification**: All snapshots are hash-verified
3. **Resumable**: Any checkpoint can be used to resume
4. **Minimal**: Only essential state is persisted

## State Schema

### AgentState

```python
class AgentState:
    # Identifiers
    run_id: str
    task_id: str | None

    # Execution tracking
    status: ExecutionStatus    # pending | running | paused | completed | failed | cancelled
    current_step: int
    max_steps: int

    # Goal and progress
    goal: str | None
    current_plan: list[str]
    completed_steps: list[str]

    # Working memory
    working_memory: dict[str, Any]

    # Conversation history
    messages: list[dict]

    # Budget tracking
    tokens_used: int
    cost_usd: float
    start_time: datetime | None
    elapsed_ms: int

    # Error tracking
    last_error: str | None
    consecutive_errors: int
    consecutive_no_progress: int
```

### Execution Status

| Status | Description |
|--------|-------------|
| `pending` | Awaiting execution |
| `running` | Currently executing |
| `paused` | Paused (can resume) |
| `completed` | Successfully finished |
| `failed` | Failed with error |
| `cancelled` | User cancelled |

## Checkpointing

### StateSnapshot

```python
class StateSnapshot:
    run_id: str
    step_id: str
    timestamp: datetime

    # Integrity
    state_hash: str           # SHA-256 of serialized state

    # Data
    state: AgentState

    # Metadata
    checkpoint_reason: str | None
    is_resumable: bool
```

### Checkpoint Triggers

1. **Step completion**: After each successful step
2. **Error recovery**: Before retry attempts
3. **Budget threshold**: At 50%, 75%, 90% budget
4. **Time interval**: Every N minutes (configurable)
5. **Manual**: Explicit checkpoint request

## State Transitions

```
┌─────────┐     start()      ┌─────────┐
│ pending │─────────────────▶│ running │
└─────────┘                  └────┬────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              pause()│       complete()│    fail()│
                    │             │             │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │  paused   │ │ completed │ │  failed   │
              └─────┬─────┘ └───────────┘ └───────────┘
                    │
              resume()
                    │
              ┌─────▼─────┐
              │  running  │
              └───────────┘
```

## Working Memory

Working memory stores intermediate results and context:

```python
state.working_memory = {
    "search_results": [...],
    "extracted_facts": [...],
    "user_preferences": {...},
}
```

### Memory Best Practices

1. **Use descriptive keys**: `search_results` not `sr`
2. **Keep it minimal**: Only store what's needed
3. **Version sensitive data**: Include version/timestamp
4. **Clear on completion**: Clean up temporary data

## Resume Protocol

```python
# Load checkpoint
snapshot = load_snapshot(run_id, step_id)

# Verify integrity
assert canonical_hash(snapshot.state) == snapshot.state_hash

# Resume execution
agent.resume(snapshot.state)
```

## Replay Protocol

Replay recreates execution from trace events:

```python
# Load trace
events = trace_reader.read_events(run_id)

# Replay with cached LLM responses
for event in events:
    if event.event_type == EventType.LLM_CALL:
        # Use cached response from trace
        response = cache.get(event.llm_request_digest)
    elif event.event_type == EventType.TOOL_CALL:
        # Re-execute or use cached result
        ...
```

## Progress Detection

The system detects lack of progress:

```python
# No-progress conditions
if state.consecutive_no_progress >= 3:
    stop_reason = StopReason.NO_PROGRESS
```

Triggers for no-progress increment:
- Same output as previous step
- No new information gained
- Repeated tool calls with same arguments

## Hash Specification

State hashes use canonical JSON (see trace_spec.md):

```python
def compute_state_hash(state: AgentState) -> str:
    # Exclude volatile fields
    serializable = state.model_dump(
        exclude={"start_time", "elapsed_ms"}
    )
    return canonical_hash(serializable)
```

## Example Usage

```python
# Create initial state
state = AgentState(
    run_id="run-001",
    goal="Research and summarize topic X",
    max_steps=50,
)

# Update state
state.current_step += 1
state.working_memory["research"] = results

# Create checkpoint
snapshot = StateSnapshot(
    run_id=state.run_id,
    step_id="step-005",
    state_hash=compute_state_hash(state),
    state=state,
    checkpoint_reason="step_complete",
)
```

## Best Practices

1. **Checkpoint frequently**: Low overhead, high value
2. **Verify hashes on load**: Catch corruption early
3. **Clean old checkpoints**: Implement retention policy
4. **Test resume paths**: Verify recovery works
5. **Log state transitions**: Aid debugging
