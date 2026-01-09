# Tool Specification

> Version: 1.0.0
> Status: Active

## Overview

The Tool system provides a controlled, auditable interface for Agent interactions with external systems. All tool calls go through the Tool Gateway which enforces permissions, validation, and retry policies.

## Design Principles

1. **Permission-first**: Every tool requires explicit capability grants
2. **Idempotency**: Write operations must be idempotent
3. **Auditability**: All calls are logged with full context
4. **Fail-safe**: Errors are categorized for appropriate handling

## Tool Specification Schema

### ToolSpec

```python
class ToolSpec:
    name: str                    # Unique tool identifier
    description: str             # Human-readable description
    input_schema: dict           # JSON Schema for input
    output_schema: dict | None   # JSON Schema for output

    # Capabilities and constraints
    side_effect: SideEffect      # read | write | none
    requires_confirmation: bool  # Needs human approval
    capabilities: list[str]      # Required capabilities

    # Retry configuration
    max_retries: int            # Default: 3
    retry_delay_ms: int         # Default: 1000
    timeout_ms: int             # Default: 30000
```

### Side Effects

| Type | Description | Protection |
|------|-------------|------------|
| `none` | No external side effects | None required |
| `read` | Read-only operation | Rate limiting |
| `write` | Modifies external state | Confirmation + idempotency |

## Error Classification

### ErrorType

| Type | Description | Action |
|------|-------------|--------|
| `retryable` | Temporary failure | Auto-retry with backoff |
| `non_retryable` | Permanent failure | Fail immediately |
| `requires_human` | Needs human decision | Escalate to HITL |

### Error Response

```python
class ToolError:
    error_type: ErrorType
    message: str
    code: str | None
    details: dict
```

## Tool Call Flow

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│   Agent     │────▶│ Tool Gateway │────▶│   Tool     │
└─────────────┘     └──────────────┘     └────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────▼─────┐ ┌─────▼─────┐
              │   Authz   │ │  Validate │
              └───────────┘ └───────────┘
                    │             │
              ┌─────▼─────┐ ┌─────▼─────┐
              │   Audit   │ │   Retry   │
              └───────────┘ └───────────┘
```

## Gateway Contract

### ToolCall

```python
class ToolCall:
    id: str                    # Unique call identifier
    name: str                  # Tool name
    arguments: dict            # Tool arguments
    idempotency_key: str | None # For write operations

    # Context
    run_id: str | None
    step_id: str | None
```

### ToolResult

```python
class ToolResult:
    tool_call_id: str
    name: str
    success: bool
    output: Any | None
    error: ToolError | None

    # Metadata
    duration_ms: int | None
    retry_count: int
```

## Idempotency

Write operations MUST include an `idempotency_key`:

```python
call = ToolCall(
    id="call-001",
    name="create_file",
    arguments={"path": "/tmp/test.txt", "content": "hello"},
    idempotency_key="create-file-abc123",
)
```

The gateway tracks idempotency keys and returns cached results for duplicate calls.

## Capability System

Tools require capabilities that must be granted to the agent:

```python
# Tool definition
file_write_tool = ToolSpec(
    name="file_write",
    capabilities=["fs:write", "fs:read"],
    ...
)

# Agent capability grant
agent_capabilities = ["fs:read"]  # No write permission

# Result: Tool call rejected
```

### Standard Capabilities

| Capability | Description |
|------------|-------------|
| `fs:read` | Read file system |
| `fs:write` | Write file system |
| `net:http` | HTTP requests |
| `net:ws` | WebSocket connections |
| `db:read` | Database reads |
| `db:write` | Database writes |
| `exec:shell` | Shell execution |

## Audit Trail

Every tool call generates a trace event:

```python
class ToolCallRecord:
    name: str
    args_digest: str          # Canonical hash of arguments
    idempotency_key: str | None
    result_digest: str | None
    error: str | None
    duration_ms: int | None
    side_effect: str | None
```

## Example Tool Definition

```python
search_tool = ToolSpec(
    name="web_search",
    description="Search the web for information",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "max_results": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
    output_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "url": {"type": "string"},
                "snippet": {"type": "string"},
            },
        },
    },
    side_effect=SideEffect.READ,
    capabilities=["net:http"],
    timeout_ms=10000,
)
```

## Best Practices

1. **Always validate input** against the schema before execution
2. **Use idempotency keys** for all write operations
3. **Set appropriate timeouts** based on expected execution time
4. **Log errors with context** for debugging
5. **Implement graceful degradation** for non-critical tools
