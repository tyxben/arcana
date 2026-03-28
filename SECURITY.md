# Security

## Security Model

Arcana is an agent runtime that executes LLM-directed tool calls. This document describes the security boundaries and what is — and is not — protected.

Arcana is designed as a single-tenant, developer-facing framework. It provides operational safety (budget limits, tool authorization, audit logging) but does not attempt to be a security sandbox for untrusted code or untrusted users.

## What Arcana Secures

### Tool Authorization

Every tool declares a `ToolSpec` with:
- `side_effect`: `READ`, `WRITE`, or `NONE` — classifies the tool's impact
- `requires_confirmation`: boolean flag for human-in-the-loop gating
- `capabilities`: list of required capability strings

The `ToolGateway` enforces these before execution:
1. **Capability check** — the agent's granted capabilities are compared against the tool's requirements. Missing capabilities produce an `UNAUTHORIZED` error. The attempt is logged to the trace.
2. **Confirmation gate** — tools with `side_effect=WRITE` or `requires_confirmation=True` are blocked until a confirmation callback approves execution. If no callback is registered, a `CONFIRMATION_REQUIRED` error is returned.
3. **Argument validation** — tool arguments are validated against the tool's `input_schema` (JSON Schema) before execution.

### Budget Enforcement

Hard limits on cost (USD) and token usage prevent runaway spending:
- `Budget(max_cost_usd=..., max_tokens=...)` sets per-runtime limits
- `BudgetTracker` checks limits before every LLM call and raises `BudgetExceededError` when exhausted
- `runtime.budget_scope()` creates isolated sub-budgets that deduct from both the scope and the runtime total
- `ChatSession` shares a single budget tracker across all turns
- `ChainStep.budget` allows per-step budget caps within a pipeline

Budget exhaustion stops execution gracefully — it does not kill the process.

### Audit Trail

Every LLM call, tool call, and state change is logged to JSONL trace files:
- One file per run: `{trace_dir}/{run_id}.jsonl`
- Events include `LLM_CALL`, `TOOL_CALL`, `STATE_CHANGE`, `ERROR`, and more
- Tool call records include `args_digest` and `result_digest` — SHA-256 hashes of canonical JSON, truncated to 16 characters
- `arcana trace` CLI for inspecting trace files
- Events are appended atomically with file locking

### Input Validation

All data flows through Pydantic models (contracts-first design):
- Tool arguments are validated against JSON Schema before execution
- LLM requests and responses are typed via `LLMRequest`/`LLMResponse` contracts
- Invalid arguments produce structured `ToolError` responses with `NON_RETRYABLE` classification

### Retry Safety

Tools can declare `idempotency_key` on calls. The `ToolGateway` caches results by idempotency key, preventing duplicate execution of the same operation during retries.

## What Arcana Does NOT Secure

### No Multi-Tenant Isolation

Arcana is a single-process, single-tenant runtime. There is no request-level isolation between concurrent users, no memory separation between sessions, and no per-user capability scoping. It is not suitable for shared multi-user deployments without an additional isolation layer (e.g., separate processes per tenant behind an API gateway).

### No Encryption at Rest

- Trace files are plaintext JSONL on disk
- API keys are passed as strings or read from environment variables
- No built-in secrets management, key rotation, or vault integration
- Digests in trace files are truncated SHA-256 hashes of content, not encrypted content

### No Network Sandboxing

- Tools can make arbitrary network calls — there is no egress filtering or domain allowlist
- MCP server connections are established to whatever endpoints the configuration specifies
- Tool authors are responsible for their own network security

### No Prompt Injection Defense

Arcana does not detect or prevent prompt injection attacks. The Constitution governs framework architecture, not LLM output content. The `ask_user` tool allows the LLM to request user input mid-execution, but there is no sanitization of that input before it reaches the LLM. Applications should implement input validation and output filtering at their own layer.

### No Filesystem Sandboxing

Trace files are written to a configurable directory with no chroot or filesystem isolation. Tools that interact with the filesystem operate with the same permissions as the host process.

## Reporting Vulnerabilities

If you discover a security vulnerability, please open a private security advisory on the GitHub repository. Do not file a public issue for security-sensitive bugs.

## Recommendations for Production

1. **Run behind an API gateway** with authentication and rate limiting
2. **Use budget limits** as a cost safety net — set `max_cost_usd` and `max_tokens` conservatively
3. **Review tool `side_effect` declarations** — ensure destructive tools are marked `WRITE` with `requires_confirmation=True`
4. **Register a confirmation callback** on the `ToolGateway` for write operations
5. **Enable trace logging** and monitor for anomalies (unexpected tool calls, budget spikes)
6. **Keep API keys in environment variables**, not hardcoded in source
7. **Scope capabilities narrowly** — grant only the capabilities each agent actually needs
8. **Use `budget_scope()`** to isolate cost exposure for sub-tasks
9. **Run one runtime per tenant** if serving multiple users
