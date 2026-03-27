# API Reference

This document covers the complete public API of the `arcana` package -- everything
exported from `import arcana`. Internal modules, private helpers, and contract
sub-types are intentionally omitted.

Version: 0.2.0

---

## Table of Contents

- [Quick-Start Functions](#quick-start-functions)
  - [arcana.run()](#arcanarun)
  - [arcana.tool()](#arcanatool)
- [Core Classes](#core-classes)
  - [Runtime](#runtime)
  - [Session](#session)
  - [Budget](#budget)
- [Result Types](#result-types)
  - [RunResult (sdk)](#runresult-sdk)
  - [RunResult (runtime)](#runresult-runtime)
  - [TeamResult](#teamresult)
  - [BatchResult](#batchresult)
- [Configuration](#configuration)
  - [AgentConfig](#agentconfig)
  - [ChainStep](#chainstep)
  - [MCPServerConfig](#mcpserverconfig)
- [Graph Engine](#graph-engine)
  - [StateGraph](#stategraph)
  - [START / END](#start--end)
  - [CompiledGraph](#compiledgraph)
- [Streaming](#streaming)
  - [StreamEvent](#streamevent)

---

## Quick-Start Functions

### `arcana.run()`

The simplest entry point. Creates a temporary `Runtime`, runs a single task, and
returns the result. Best for scripts, demos, and one-off calls.

```python
async def run(
    goal: str,
    *,
    tools: list[Callable] | None = None,
    provider: str = "deepseek",
    model: str | None = None,
    api_key: str | None = None,
    max_turns: int = 20,
    max_cost_usd: float = 1.0,
    auto_route: bool = True,
    engine: str = "conversation",
    stream: bool = False,
) -> RunResult
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `goal` | `str` | required | What you want the agent to accomplish. |
| `tools` | `list[Callable] \| None` | `None` | List of `@arcana.tool` decorated functions. |
| `provider` | `str` | `"deepseek"` | LLM provider name (`"deepseek"`, `"openai"`, `"anthropic"`, `"gemini"`, `"kimi"`, `"glm"`, `"minimax"`, `"ollama"`). |
| `model` | `str \| None` | `None` | Model ID. Auto-selected from provider default when `None`. |
| `api_key` | `str \| None` | `None` | API key for the provider. Falls back to `<PROVIDER>_API_KEY` env var. |
| `max_turns` | `int` | `20` | Maximum number of agent turns before stopping. |
| `max_cost_usd` | `float` | `1.0` | Maximum spend in USD for this run. |
| `auto_route` | `bool` | `True` | Enable intent routing (direct answer vs. agent loop). |
| `engine` | `str` | `"conversation"` | Execution engine: `"conversation"` (V2, recommended) or `"adaptive"` (V1). |
| `stream` | `bool` | `False` | Reserved for future use. |

**Returns:** [`RunResult`](#runresult-sdk) with output text and execution metadata.

**Example**

```python
import arcana

# Simplest usage
result = await arcana.run("What is 2+2?", api_key="sk-xxx")
print(result.output)

# With tools
@arcana.tool(when_to_use="For math calculations")
def calc(expression: str) -> str:
    return str(eval(expression))

result = await arcana.run("What is 15 * 37 + 89?", tools=[calc], api_key="sk-xxx")

# With OpenAI
result = await arcana.run(
    "Summarize quantum computing",
    provider="openai",
    api_key="sk-proj-xxx",
)
```

---

### `arcana.tool()`

Decorator that registers a function as an Arcana tool. The decorated function
retains its normal behavior but gains metadata that `Runtime` reads at
registration time.

```python
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    when_to_use: str | None = None,
    what_to_expect: str | None = None,
    failure_meaning: str | None = None,
    side_effect: str = "read",
    requires_confirmation: bool = False,
) -> Callable
```

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str \| None` | `None` | Override tool name. Defaults to the function name. |
| `description` | `str \| None` | `None` | Override tool description. Defaults to the function docstring. |
| `when_to_use` | `str \| None` | `None` | Guidance for the LLM on when this tool is appropriate. |
| `what_to_expect` | `str \| None` | `None` | Describes what the tool returns so the LLM knows how to interpret results. |
| `failure_meaning` | `str \| None` | `None` | Explains what a failure means so the LLM can decide whether to retry. |
| `side_effect` | `str` | `"read"` | Side effect category: `"read"`, `"write"`, `"network"`, `"filesystem"`. |
| `requires_confirmation` | `bool` | `False` | If `True`, the runtime asks for user confirmation before execution. |

**Returns:** The original function, with `_arcana_tool_spec` attached.

**Notes:**
- Input schema is auto-inferred from the function signature (type hints map to JSON Schema types).
- Both sync and async functions are supported.
- Pass the decorated function directly in the `tools=` list to `arcana.run()` or `Runtime()`.

**Example**

```python
import arcana

@arcana.tool(
    when_to_use="When you need to search the web for current information",
    what_to_expect="Returns a list of search result snippets",
    side_effect="network",
)
async def web_search(query: str) -> str:
    results = await search_api(query)
    return "\n".join(results)

@arcana.tool(
    name="calculator",
    when_to_use="For any mathematical computation",
    requires_confirmation=False,
)
def calc(expression: str) -> str:
    return str(eval(expression))
```

---

## Core Classes

### `Runtime`

The central object in Arcana. Create once at application startup; reuse across
requests. Holds long-lived resources: provider connections, tool registry, trace
backend, budget policy, and memory store.

```python
class Runtime:
    def __init__(
        self,
        *,
        providers: dict[str, str] | None = None,
        tools: list[Callable] | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        budget: Budget | None = None,
        trace: bool = False,
        memory: bool = False,
        memory_budget_tokens: int = 800,
        config: RuntimeConfig | None = None,
        namespace: str | None = None,
    ) -> None
```

**Constructor Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `providers` | `dict[str, str] \| None` | `None` | Provider name to API key mapping, e.g. `{"deepseek": "sk-xxx", "openai": "sk-proj-xxx"}`. Empty string reads from env var. |
| `tools` | `list[Callable] \| None` | `None` | List of `@arcana.tool` decorated functions to register. |
| `mcp_servers` | `list[MCPServerConfig] \| None` | `None` | MCP server configurations for external tool servers. |
| `budget` | `Budget \| None` | `None` | Default budget policy. Defaults to `Budget(max_cost_usd=10.0)`. |
| `trace` | `bool` | `False` | Enable JSONL trace logging. |
| `memory` | `bool` | `False` | Enable cross-run memory (fact extraction and retrieval). |
| `memory_budget_tokens` | `int` | `800` | Token budget for memory context injection. |
| `config` | `RuntimeConfig \| None` | `None` | Advanced runtime configuration (default provider, model, max turns, system prompt). |
| `namespace` | `str \| None` | `None` | Tenant isolation namespace. When set, memory and trace are partitioned. |

> **Import note:** `RuntimeConfig` is **not** exported from `arcana`. Import it directly:
> ```python
> from arcana.runtime_core import RuntimeConfig
> ```
> Do not use `arcana.RuntimeConfig` -- it does not exist.

#### `Runtime.run()`

Run a task to completion.

```python
async def run(
    self,
    goal: str,
    *,
    engine: str = "conversation",
    max_turns: int | None = None,
    budget: Budget | None = None,
    tools: list[Callable] | None = None,
) -> RunResult
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `goal` | `str` | required | What to accomplish. |
| `engine` | `str` | `"conversation"` | `"conversation"` (V2) or `"adaptive"` (V1). |
| `max_turns` | `int \| None` | `None` | Override default max turns for this run. |
| `budget` | `Budget \| None` | `None` | Override default budget for this run. |
| `tools` | `list[Callable] \| None` | `None` | Additional tools for this run only (merged with runtime tools). |

**Returns:** [`RunResult`](#runresult-runtime)

#### `Runtime.stream()`

Stream agent execution events in real time.

```python
async def stream(
    self,
    goal: str,
    *,
    engine: str = "conversation",
    max_turns: int | None = None,
) -> AsyncGenerator[StreamEvent, None]
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `goal` | `str` | required | What to accomplish. |
| `engine` | `str` | `"conversation"` | Only `"conversation"` is supported for streaming. |
| `max_turns` | `int \| None` | `None` | Override default max turns. |

**Yields:** [`StreamEvent`](#streamevent) objects.

**Example**

```python
async for event in runtime.stream("Analyze this data"):
    print(event.event_type, event.content)
```

#### `Runtime.session()`

Create a session for manual control over execution. Used as an async context
manager.

```python
@asynccontextmanager
async def session(
    self,
    *,
    engine: str = "conversation",
    max_turns: int | None = None,
    budget: Budget | None = None,
    tools: list[Callable] | None = None,
) -> AsyncGenerator[Session, None]
```

**Returns:** [`Session`](#session) (via `async with`).

**Example**

```python
async with runtime.session() as s:
    result = await s.run("Do something")
    print(s.state)
    print(s.budget.to_snapshot())
```

#### `Runtime.team()`

Run a team of agents on a shared goal. Each agent gets its own system prompt and
takes turns in a shared conversation. The runtime manages resource isolation,
communication, budget, and trace.

```python
async def team(
    self,
    goal: str,
    agents: list[AgentConfig],
    *,
    max_rounds: int = 3,
    budget: Budget | None = None,
) -> TeamResult
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `goal` | `str` | required | The shared objective. |
| `agents` | `list[AgentConfig]` | required | Agent configurations (name, prompt, optional model/provider overrides). |
| `max_rounds` | `int` | `3` | Maximum conversation rounds (each agent speaks once per round). |
| `budget` | `Budget \| None` | `None` | Budget for the entire team run. |

**Returns:** [`TeamResult`](#teamresult)

#### `Runtime.chain()`

Run a sequential pipeline of steps with automatic context passing. Each step's
output flows as context to the next. Parallel branches (nested lists) run
concurrently.

```python
async def chain(
    self,
    steps: list[ChainStep | list[ChainStep]],
    *,
    budget: Budget | None = None,
) -> ChainResult
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `steps` | `list[ChainStep \| list[ChainStep]]` | required | Steps to execute. Nested lists run in parallel. |
| `budget` | `Budget \| None` | `None` | Budget for the entire chain. |

**Returns:** `ChainResult` with per-step outputs accessible via `result.steps["name"]`.

**Example**

```python
result = await runtime.chain([
    arcana.ChainStep(name="research", goal="Find key facts about quantum computing"),
    [  # parallel branch
        arcana.ChainStep(name="summary", goal="Write a concise summary"),
        arcana.ChainStep(name="critique", goal="Identify gaps and biases",
                         budget=arcana.Budget(max_cost_usd=0.50)),
    ],
    arcana.ChainStep(name="final", goal="Integrate summary and critique into a report"),
])
print(result.steps["final"])
```

#### `Runtime.run_batch()`

Run multiple independent tasks concurrently. Individual failures do not crash
the batch -- the corresponding `RunResult` will have `success=False`.

```python
async def run_batch(
    self,
    tasks: list[dict[str, Any]],
    *,
    concurrency: int = 5,
) -> BatchResult
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tasks` | `list[dict[str, Any]]` | required | List of task dicts. Each must have a `"goal"` key. Optional keys match `run()` parameters (`tools`, `system`, `provider`, `model`, `response_format`, etc.). |
| `concurrency` | `int` | `5` | Maximum number of concurrent runs. |

**Returns:** [`BatchResult`](#batchresult)

**Example**

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]

results = await runtime.run_batch([
    {"goal": "Summarize article 1", "response_format": Summary},
    {"goal": "Summarize article 2", "response_format": Summary},
    {"goal": "Summarize article 3", "response_format": Summary},
], concurrency=10)

print(f"{results.succeeded}/{len(results.results)} succeeded")
print(f"Total cost: ${results.total_cost_usd:.4f}")
```

#### `Runtime.graph()`

Create a `StateGraph` connected to this runtime's resources.

```python
def graph(self, state_schema: type | None = None) -> StateGraph
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `state_schema` | `type \| None` | `None` | Pydantic model class for graph state. |

**Returns:** [`StateGraph`](#stategraph) ready for node/edge configuration.

**Example**

```python
graph = runtime.graph(state_schema=MyState)
graph.add_node("search", search_fn)
graph.add_edge(START, "search")
app = graph.compile()
result = await app.ainvoke(initial_state)
```

#### `Runtime.make_llm_node()`

Create an `LLMNode` pre-wired with this runtime's gateway and model config.

```python
def make_llm_node(self, *, system_prompt: str | None = None) -> LLMNode
```

#### `Runtime.make_tool_node()`

Create a `ToolNode` pre-wired with this runtime's tool gateway.

```python
def make_tool_node(self) -> ToolNode
```

Raises `ValueError` if no tools are registered.

#### `Runtime.connect_mcp()`

Connect to configured MCP servers and register their tools.

```python
async def connect_mcp(self) -> list[str]
```

**Returns:** List of registered MCP tool names.

#### `Runtime.close()`

Clean up runtime resources (MCP connections, etc.).

```python
async def close(self) -> None
```

#### Runtime Properties

| Property | Type | Description |
|----------|------|-------------|
| `providers` | `list[str]` | List of registered provider names. |
| `tools` | `list[str]` | List of registered tool names. |
| `namespace` | `str \| None` | The namespace for tenant isolation, or `None`. |
| `memory` | `Any` | Access the memory store (if enabled). |

---

### `Session`

Per-run execution context. Created by `Runtime.session()`. Holds run-scoped
resources: run ID, per-run budget tracker, and execution state.

```python
class Session:
    def __init__(
        self,
        runtime: Runtime,
        engine: str = "conversation",
        max_turns: int = 20,
        budget: Budget | None = None,
        extra_tools: list[Callable] | None = None,
        memory_context: str = "",
    ) -> None
```

You typically do not construct `Session` directly. Use `Runtime.session()` instead.

#### `Session.run()`

Run a task within this session.

```python
async def run(self, goal: str) -> RunResult
```

#### Session Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `run_id` | `str` | Unique identifier for this run (UUID). |
| `state` | `AgentState \| None` | The execution state after `run()` completes. |
| `budget` | `BudgetTracker` | Per-run budget tracker. Call `budget.to_snapshot()` for current usage. |

---

### `Budget`

Budget configuration. Passed to `Runtime()` or `Runtime.run()` to cap spending.

```python
class Budget(BaseModel):
    max_cost_usd: float = 10.0
    max_tokens: int = 500_000
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_cost_usd` | `float` | `10.0` | Maximum spend in USD before the runtime stops. |
| `max_tokens` | `int` | `500_000` | Maximum total tokens (input + output) before the runtime stops. |

**Example**

```python
import arcana

# Conservative budget for a demo
budget = arcana.Budget(max_cost_usd=0.50, max_tokens=100_000)

runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    budget=budget,
)
```

---

## Result Types

> **Two `RunResult` classes -- read this first.**
>
> Arcana has two separate `RunResult` classes with identical fields but different
> origins. This is intentional: the SDK and runtime layers are decoupled.
>
> | Class | Import path | Returned by |
> |-------|-------------|-------------|
> | `arcana.RunResult` | `import arcana; arcana.RunResult` | `arcana.run()` (SDK convenience function) |
> | `arcana.RuntimeResult` | `import arcana; arcana.RuntimeResult` | `Runtime.run()`, `Session.run()` |
>
> Both have the same fields (`output`, `parsed`, `success`, `steps`, `tokens_used`,
> `cost_usd`, `run_id`), but they are **different classes** -- `isinstance`
> checks will not match across them.
>
> **Rule of thumb:** If you are type-checking results from `Runtime` or `Session`,
> use `arcana.RuntimeResult`. If you are type-checking results from `arcana.run()`,
> use `arcana.RunResult`.

### `RunResult` (sdk)

Returned by `arcana.run()`. A simplified result with execution metadata.

Import path: `arcana.RunResult` (this is the SDK version).

```python
class RunResult(BaseModel):
    output: Any = None
    parsed: Any = None
    success: bool = False
    steps: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    run_id: str = ""
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output` | `Any` | `None` | The agent's final output (usually a string, or the parsed model when `response_format` succeeds). |
| `parsed` | `BaseModel \| None` | `None` | Validated Pydantic instance when `response_format` is set and parsing succeeds. Always `BaseModel \| None` -- never a raw dict. |
| `success` | `bool` | `False` | Whether the task completed successfully. |
| `steps` | `int` | `0` | Number of execution steps taken. |
| `tokens_used` | `int` | `0` | Total tokens consumed (input + output). |
| `cost_usd` | `float` | `0.0` | Estimated cost in USD. |
| `run_id` | `str` | `""` | Unique run identifier for trace correlation. |

---

### `RunResult` (runtime)

Returned by `Runtime.run()` and `Session.run()`. Identical fields to the SDK
version. Defined in `arcana.runtime_core`.

Import path: `arcana.RuntimeResult` (aliased from `arcana.runtime_core.RunResult`).

```python
class RunResult(BaseModel):
    output: Any = None
    parsed: Any = None
    success: bool = False
    steps: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    run_id: str = ""
```

Fields are identical to [RunResult (sdk)](#runresult-sdk).

---

### `TeamResult`

Returned by `Runtime.team()`. Contains the team conversation output and metadata.

```python
class TeamResult(BaseModel):
    output: Any = None
    success: bool = False
    rounds: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    agent_outputs: dict[str, str] = {}
    conversation_log: list[dict[str, Any]] = []
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output` | `Any` | `None` | Final output from the last speaking agent. |
| `success` | `bool` | `False` | `True` if any agent emitted `[DONE]`, `False` if max rounds reached. |
| `rounds` | `int` | `0` | Number of conversation rounds completed. |
| `total_tokens` | `int` | `0` | Total tokens consumed across all agents. |
| `total_cost_usd` | `float` | `0.0` | Estimated total cost in USD. |
| `agent_outputs` | `dict[str, str]` | `{}` | Map of agent name to their last output. |
| `conversation_log` | `list[dict]` | `[]` | Full conversation history. Each entry has `round`, `agent`, `content`, `tokens`. |

**Example**

```python
import arcana

runtime = arcana.Runtime(providers={"deepseek": "sk-xxx"})

result = await runtime.team(
    goal="Design a REST API for a todo app",
    agents=[
        arcana.AgentConfig(
            name="architect",
            prompt="You are a senior API architect. Focus on clean REST design.",
        ),
        arcana.AgentConfig(
            name="reviewer",
            prompt="You are a code reviewer. Find issues and suggest improvements.",
        ),
    ],
    max_rounds=3,
)

for entry in result.conversation_log:
    print(f"[Round {entry['round']}] {entry['agent']}: {entry['content'][:100]}")
```

---

### `BatchResult`

Returned by `Runtime.run_batch()`. Aggregates results from all tasks in the batch.

```python
class BatchResult(BaseModel):
    results: list[RunResult] = []
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    succeeded: int = 0
    failed: int = 0
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `results` | `list[RunResult]` | `[]` | All results, preserving input order. Failed tasks have `success=False`. |
| `total_tokens` | `int` | `0` | Total tokens consumed across all tasks. |
| `total_cost_usd` | `float` | `0.0` | Estimated total cost in USD. |
| `succeeded` | `int` | `0` | Number of tasks that completed successfully. |
| `failed` | `int` | `0` | Number of tasks that failed. |

---

## Configuration

### `AgentConfig`

Configuration for a single agent in a team. Passed in a list to `Runtime.team()`.

```python
class AgentConfig(BaseModel):
    name: str
    prompt: str
    model: str | None = None
    provider: str | None = None
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Unique name for this agent (appears in conversation log). |
| `prompt` | `str` | required | System prompt defining this agent's role and personality. |
| `model` | `str \| None` | `None` | Override the runtime's default model for this agent. |
| `provider` | `str \| None` | `None` | Override the runtime's default provider for this agent. |

---

### `ChainStep`

Configuration for a single step in a `Runtime.chain()` pipeline.

```python
class ChainStep(BaseModel):
    name: str
    goal: str
    system: str | None = None
    response_format: type[BaseModel] | None = None
    tools: list[Callable] | None = None
    provider: str | None = None
    model: str | None = None
    on_parse_error: Callable[[str, Exception], BaseModel | None] | None = None
    budget: Budget | None = None
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Unique name for this step (used as key in `ChainResult.steps`). |
| `goal` | `str` | required | Prompt for this step. Previous step output is injected as context automatically. |
| `system` | `str \| None` | `None` | System prompt override for this step. |
| `response_format` | `type[BaseModel] \| None` | `None` | Pydantic model for structured output. |
| `tools` | `list[Callable] \| None` | `None` | Additional tools for this step only. |
| `provider` | `str \| None` | `None` | Override provider for this step. |
| `model` | `str \| None` | `None` | Override model for this step. |
| `on_parse_error` | `Callable \| None` | `None` | Callback for structured output parse failures. |
| `budget` | `Budget \| None` | `None` | Per-step budget cap. Capped by chain-level remaining budget. Steps without `budget` share the chain pool. |

---

### `MCPServerConfig`

Configuration for connecting to an MCP (Model Context Protocol) server. Pass a
list of these to `Runtime(mcp_servers=[...])`.

```python
class MCPServerConfig(BaseModel):
    name: str
    transport: MCPTransportType = MCPTransportType.STDIO
    # stdio transport
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    # HTTP transport
    url: str | None = None
    headers: dict[str, str] = {}
    # Common
    timeout_ms: int = 30000
    reconnect_attempts: int = 3
    reconnect_delay_ms: int = 1000
    capability_prefix: str | None = None
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Human-readable name for this MCP server. |
| `transport` | `MCPTransportType` | `STDIO` | Transport type: `"stdio"`, `"sse"`, or `"streamable_http"`. |
| `command` | `str \| None` | `None` | Command to launch the server (stdio transport). |
| `args` | `list[str]` | `[]` | Command-line arguments (stdio transport). |
| `env` | `dict[str, str]` | `{}` | Environment variables for the server process (stdio transport). |
| `url` | `str \| None` | `None` | Server URL (SSE / HTTP transport). |
| `headers` | `dict[str, str]` | `{}` | HTTP headers (SSE / HTTP transport). |
| `timeout_ms` | `int` | `30000` | Connection timeout in milliseconds. |
| `reconnect_attempts` | `int` | `3` | Number of reconnection attempts on failure. |
| `reconnect_delay_ms` | `int` | `1000` | Delay between reconnection attempts in milliseconds. |
| `capability_prefix` | `str \| None` | `None` | Prefix for tool names to avoid collisions across servers. |

**Example**

```python
import arcana

runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    mcp_servers=[
        arcana.MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ),
        arcana.MCPServerConfig(
            name="remote-tools",
            transport="sse",
            url="http://localhost:8080/sse",
        ),
    ],
)

# MCP tools are auto-connected on first run, or connect manually:
tool_names = await runtime.connect_mcp()
print(f"Registered MCP tools: {tool_names}")
```

---

## Graph Engine

The graph engine is for advanced orchestration with explicit control flow. For
most tasks, `Runtime.run()` (which uses the conversation engine internally) is
the right entry point.

### `StateGraph`

Declarative graph builder. Add nodes (functions) and edges (control flow), then
compile into an executable graph.

```python
class StateGraph:
    def __init__(self, state_schema: type[BaseModel] | None = None) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `state_schema` | `type[BaseModel] \| None` | `None` | Pydantic model class defining the graph state shape. |

#### `StateGraph.add_node()`

```python
def add_node(
    self,
    name: str,
    fn: Callable[..., Any],
    *,
    node_type: NodeType = NodeType.FUNCTION,
    metadata: dict[str, Any] | None = None,
) -> StateGraph
```

Add a node to the graph. The function `fn` receives the current state and
returns an updated state (or partial state dict).

**Returns:** `self` (for chaining).

#### `StateGraph.add_edge()`

```python
def add_edge(self, source: str, target: str) -> StateGraph
```

Add a direct edge between two nodes. Use `START` and `END` constants for
entry/exit points.

**Returns:** `self` (for chaining).

#### `StateGraph.add_conditional_edges()`

```python
def add_conditional_edges(
    self,
    source: str,
    path_fn: Callable[..., str],
    path_map: dict[str, str] | None = None,
) -> StateGraph
```

Add conditional routing from a source node. The `path_fn` receives the current
state and returns a string key. If `path_map` is provided, the key is mapped to
a target node name; otherwise the key is used directly as the target.

**Returns:** `self` (for chaining).

#### `StateGraph.compile()`

```python
def compile(
    self,
    *,
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    name: str = "default",
) -> CompiledGraph
```

Compile the graph into an executable `CompiledGraph`. Validates that an entry
point exists, all edge references are valid, and at least one path reaches `END`.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `checkpointer` | `Any \| None` | `None` | Checkpoint backend for interrupt/resume. |
| `interrupt_before` | `list[str] \| None` | `None` | Node names to interrupt before executing. |
| `interrupt_after` | `list[str] \| None` | `None` | Node names to interrupt after executing. |
| `name` | `str` | `"default"` | Name for this compiled graph. |

**Returns:** `CompiledGraph` -- call `await app.ainvoke(state)` to execute.

#### `StateGraph.set_entry_point()` / `StateGraph.set_finish_point()`

```python
def set_entry_point(self, name: str) -> StateGraph
def set_finish_point(self, name: str) -> StateGraph
```

Alternative to `add_edge(START, name)` and `add_edge(name, END)`.

**Example**

```python
import arcana
from arcana import START, END

graph = arcana.StateGraph(state_schema=MyState)

graph.add_node("search", search_fn)
graph.add_node("summarize", summarize_fn)
graph.add_node("decide", decide_fn)

graph.add_edge(START, "search")
graph.add_conditional_edges("search", route_fn, {
    "needs_more": "search",
    "ready": "summarize",
})
graph.add_edge("summarize", END)

app = graph.compile()
result = await app.ainvoke({"query": "quantum computing trends"})
```

---

### `START` / `END`

Sentinel constants for graph entry and exit points.

```python
from arcana import START, END

# Values (for reference, do not hardcode these):
# START = "__start__"
# END   = "__end__"
```

Use these with `StateGraph.add_edge()`:

```python
graph.add_edge(START, "first_node")
graph.add_edge("last_node", END)
```

---

### `CompiledGraph`

The return type of `StateGraph.compile()`. You never instantiate this directly --
always obtain it via `graph.compile()`.

```python
from arcana.graph.compiled_graph import CompiledGraph
```

#### `CompiledGraph.ainvoke()`

Execute the graph to completion and return the final state.

```python
async def ainvoke(
    self,
    input: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `dict[str, Any]` | required | Initial state for the graph (must match the state schema). |
| `config` | `dict[str, Any] \| None` | `None` | Optional runtime configuration passed through to node functions. |

**Returns:** `dict[str, Any]` -- the final graph state after all nodes have executed.

**Example**

```python
app = graph.compile()
result = await app.ainvoke({"query": "quantum computing trends"})
print(result["summary"])
```

#### `CompiledGraph.astream()`

Execute the graph with streaming output, yielding state or updates after each node.

```python
async def astream(
    self,
    input: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    mode: str = "values",
) -> AsyncGenerator[dict[str, Any], None]
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input` | `dict[str, Any]` | required | Initial state for the graph. |
| `config` | `dict[str, Any] \| None` | `None` | Optional runtime configuration. |
| `mode` | `str` | `"values"` | Streaming mode: `"values"` (full state after each node), `"updates"` (`{"node": name, "output": {...}}` after each node), or `"messages"` (new messages added at each step). |

**Yields:** `dict[str, Any]` -- state snapshots or update dicts depending on `mode`.

**Example**

```python
app = graph.compile()
async for state in app.astream({"query": "search me"}, mode="updates"):
    print(f"Node: {state['node']}, Output: {state['output']}")
```

#### `CompiledGraph.aresume()`

Resume execution from a checkpoint after a human-in-the-loop interrupt.

```python
async def aresume(
    self,
    checkpoint_id: str,
    command: Command | None = None,
) -> dict[str, Any]
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `checkpoint_id` | `str` | required | The checkpoint ID from a `GraphInterrupt` exception. |
| `command` | `Command \| None` | `None` | Optional command to influence resumed execution. |

**Returns:** `dict[str, Any]` -- the final graph state after resumed execution completes.

**Raises:** `RuntimeError` if the graph was compiled without a checkpointer.
`ValueError` if the checkpoint ID is not found.

The `Command` class (from `arcana.graph.interrupt`) has three optional fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `resume` | `Any` | `None` | Value passed to the interrupted node on resume. |
| `update` | `dict[str, Any] \| None` | `None` | State updates to apply before resuming. |
| `goto` | `str \| None` | `None` | Jump to a specific node instead of resuming at the interrupt point. |

**Example**

```python
from arcana.graph.interrupt import Command, GraphInterrupt

app = graph.compile(
    checkpointer=my_checkpointer,
    interrupt_before=["human_review"],
)

try:
    result = await app.ainvoke({"query": "draft a proposal"})
except GraphInterrupt as e:
    # Human reviews and approves
    result = await app.aresume(
        e.checkpoint_id,
        command=Command(resume="approved", update={"feedback": "Looks good"}),
    )
```

#### CompiledGraph Properties

| Property | Type | Description |
|----------|------|-------------|
| `config` | `GraphConfig` | The graph configuration (name, interrupt settings). |
| `nodes` | `dict[str, GraphNodeSpec]` | Copy of the registered node specifications. |

---

## Streaming

### `StreamEvent`

Event emitted during streaming execution via `Runtime.stream()`.

```python
class StreamEvent(BaseModel):
    event_type: StreamEventType
    timestamp: datetime
    run_id: str
    step_id: str | None = None

    # Content
    content: str | None = None
    thinking: str | None = None
    node_name: str | None = None

    # Structured data
    step_result_data: dict[str, Any] | None = None
    tool_result_data: dict[str, Any] | None = None
    state_delta: dict[str, Any] | None = None

    # Metrics
    tokens_used: int | None = None
    cost_usd: float | None = None
    budget_remaining_pct: float | None = None

    # Error
    error: str | None = None
    error_type: str | None = None

    # Metadata
    metadata: dict[str, Any] = {}
```

**Key Fields**

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | `StreamEventType` | The event kind (see table below). |
| `timestamp` | `datetime` | When the event was created (UTC). |
| `run_id` | `str` | Run identifier for correlation. |
| `content` | `str \| None` | LLM text content (for `LLM_CHUNK` / `LLM_COMPLETE`). |
| `thinking` | `str \| None` | LLM thinking/reasoning text (for `LLM_THINKING`). |
| `tokens_used` | `int \| None` | Cumulative token usage at this point. |
| `cost_usd` | `float \| None` | Cumulative cost at this point. |
| `error` | `str \| None` | Error message (for `ERROR` events). |

**StreamEventType Values**

> **Import note:** `StreamEventType` is **not** exported from `arcana`. If you need
> to reference it directly, import it from the contracts module:
> ```python
> from arcana.contracts.streaming import StreamEventType
> ```
> Do not use `arcana.StreamEventType` -- it does not exist.
>
> As a simpler alternative, compare against `.value` strings instead of importing
> the enum:
> ```python
> if event.event_type.value == "llm_chunk":
>     ...
> ```

| Value | Description |
|-------|-------------|
| `RUN_START` | Execution has started. |
| `RUN_COMPLETE` | Execution has finished. |
| `STEP_START` | A new agent step/turn is starting. |
| `STEP_COMPLETE` | An agent step/turn has finished. |
| `LLM_CHUNK` | A chunk of streaming LLM output. |
| `LLM_COMPLETE` | The full LLM response for this turn is ready. |
| `LLM_THINKING` | LLM reasoning/thinking text (chain-of-thought). |
| `TOOL_CALL_START` | A tool call is about to execute. |
| `TOOL_RESULT` | A tool call has returned. |
| `STATE_UPDATE` | Agent state has changed. |
| `CHECKPOINT` | A checkpoint was saved. |
| `ERROR` | An error occurred. |
| `NODE_START` | A graph node has started executing. |
| `NODE_COMPLETE` | A graph node has finished executing. |

**Example**

```python
import arcana

runtime = arcana.Runtime(providers={"deepseek": "sk-xxx"})

async for event in runtime.stream("Explain quantum entanglement"):
    if event.event_type.value == "llm_chunk":
        print(event.content, end="", flush=True)
    elif event.event_type.value == "tool_call_start":
        print(f"\n[Calling tool: {event.metadata.get('tool_name')}]")
    elif event.event_type.value == "error":
        print(f"\nError: {event.error}")
```
