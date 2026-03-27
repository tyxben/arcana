# Quick Start to Production

This guide takes you from installation to a production-ready agent service.
Every code example uses the real Arcana API and can be run as-is.

---

## Installation

Arcana requires Python 3.11+. Install with pip or uv:

```bash
# Core (includes DeepSeek, OpenAI, Kimi, GLM, MiniMax, Ollama)
pip install arcana-agent

# With Anthropic (Claude) support
pip install arcana-agent[anthropic]

# With Google Gemini support
pip install arcana-agent[gemini]

# All LLM providers
pip install arcana-agent[all-providers]

# Trace Web UI (FastAPI-based inspector)
pip install arcana-agent[ui]
```

Or with uv:

```bash
uv add arcana-agent
uv add arcana-agent[all-providers]
```

Set your API key as an environment variable (or pass it directly in code):

```bash
export DEEPSEEK_API_KEY=sk-xxx
```

---

## Your First Agent

The simplest way to run an agent is `arcana.run()`. It creates a temporary
Runtime, routes the intent, calls the LLM, and returns a result -- all in
one call.

```python
import asyncio
import arcana

async def main():
    result = await arcana.run(
        "What are the three laws of thermodynamics?",
        provider="deepseek",
        api_key="sk-xxx",
    )
    print(result.output)
    print(f"Success: {result.success}")
    print(f"Run ID: {result.run_id}")
    print(f"Steps: {result.steps} | Tokens: {result.tokens_used} | Cost: ${result.cost_usd:.4f}")

asyncio.run(main())
```

**What happens under the hood:**

1. Arcana creates a temporary `Runtime` with the specified provider.
2. An intent classifier decides the approach: simple questions get a direct
   answer (one LLM call), complex tasks enter a multi-turn agent loop.
3. The `ConversationAgent` (V2 engine) manages turns, budget, and tool calls.
4. The result is returned as a `RunResult` with output, token count, and cost.

`arcana.run()` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `goal` | (required) | What you want the agent to accomplish |
| `provider` | `"deepseek"` | LLM provider name |
| `model` | `None` | Model ID (auto-selected from provider default if None) |
| `api_key` | `None` | API key; falls back to env var `{PROVIDER}_API_KEY` |
| `tools` | `None` | List of `@arcana.tool` decorated functions |
| `max_turns` | `20` | Maximum execution turns |
| `max_cost_usd` | `1.0` | Budget cap in USD |
| `auto_route` | `True` | Enable intent routing (bypasses agent loop for simple queries) |
| `response_format` | `None` | Pydantic `BaseModel` class for structured output |
| `stream` | `False` | Reserved for future streaming support |
| `engine` | `"conversation"` | `"conversation"` (V2) or `"adaptive"` (V1) |

---

## Adding Tools

Tools give your agent the ability to act. Decorate any function with
`@arcana.tool` and pass it to `run()` or `Runtime`.

The decorator accepts **affordances** -- metadata that tells the LLM when
and how to use the tool:

```python
import arcana

@arcana.tool(
    when_to_use="When you need to perform mathematical calculations",
    what_to_expect="Returns the exact numeric result as a string",
    failure_meaning="The expression was malformed or contained undefined operations",
)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    # WARNING: eval() is unsafe for production -- use a math parser instead
    return str(eval(expression))
```

### Affordance fields

| Field | Purpose |
|-------|---------|
| `when_to_use` | Tells the LLM the conditions under which this tool is appropriate |
| `what_to_expect` | Describes the shape and content of a successful response |
| `failure_meaning` | Explains what a failure result indicates, helping the LLM recover |
| `side_effect` | `"read"` (default) or `"write"` -- declares whether the tool mutates state |
| `requires_confirmation` | `True` if the tool needs user approval before execution |

### Using tools with `arcana.run()`

```python
import asyncio
import arcana

@arcana.tool(
    when_to_use="When you need to search for information",
    what_to_expect="Returns a text snippet with relevant information",
)
async def web_search(query: str) -> str:
    # Replace with a real search API call
    return f"Top result for '{query}': Arcana is a runtime-first agent framework."

async def main():
    result = await arcana.run(
        "What is Arcana?",
        tools=[web_search],
        api_key="sk-xxx",
    )
    print(result.output)

asyncio.run(main())
```

Both sync and async tool functions are supported. Arcana infers the JSON
schema for tool parameters from the function signature automatically.

---

## Structured Output

Pass a Pydantic model as `response_format` to get typed results instead of
free-form text. Tools remain available -- the LLM can still call tools
during execution and returns the structured object at the end.

```python
import asyncio
from pydantic import BaseModel
import arcana

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

async def main():
    result = await arcana.run(
        "Summarize this article: ...",
        response_format=Summary,
        api_key="sk-xxx",
    )

    # result.parsed is always Summary | None, never a raw dict
    if result.parsed:
        print(result.parsed.title)
        for point in result.parsed.key_points:
            print(f"  - {point}")

asyncio.run(main())
```

`result.parsed` contains the validated Pydantic instance, or `None` if
parsing failed. The raw text is still available in `result.output`.

For custom error recovery, pass `on_parse_error`:

```python
result = await arcana.run(
    "Summarize this article: ...",
    response_format=Summary,
    on_parse_error=lambda raw, err: Summary(
        title="Parse failed", key_points=[raw[:200]], sentiment="unknown"
    ),
)
```

---

## Using Runtime

`arcana.run()` is convenient for scripts, but production services should
use `Runtime` directly. A Runtime is created once at startup and reused
across requests. It holds long-lived resources: provider connections, tool
registry, budget policy, and trace backend.

```python
import asyncio
import arcana

async def main():
    runtime = arcana.Runtime(
        providers={"deepseek": "sk-xxx", "openai": "sk-proj-xxx"},
        tools=[calculator],          # registered once, available to every run
        budget=arcana.Budget(max_cost_usd=10.0, max_tokens=500_000),
        trace=True,                  # enable JSONL trace logging
    )

    # Run 1
    result = await runtime.run("What is 25 * 4 + 13 * 7?")
    print(result.output)

    # Run 2 -- same runtime, same tools, same budget policy
    result = await runtime.run("Explain quantum entanglement briefly")
    print(result.output)

asyncio.run(main())
```

### Runtime constructor

| Parameter | Default | Description |
|-----------|---------|-------------|
| `providers` | `None` | Dict of `{"name": "api_key"}`. Empty string reads from env var. |
| `tools` | `None` | List of `@arcana.tool` decorated functions |
| `mcp_servers` | `None` | List of `MCPServerConfig` for MCP tool servers |
| `budget` | `Budget(max_cost_usd=10.0, max_tokens=500_000)` | Default budget policy for all runs |
| `trace` | `False` | Enable JSONL trace logging |
| `memory` | `False` | Enable cross-run memory |
| `config` | `RuntimeConfig()` | Advanced config (default provider, model, trace dir). See note below. |
| `namespace` | `None` | Tenant isolation key for memory and trace partitioning |

> **Note:** `RuntimeConfig` is not part of the top-level public API. Import it directly:
>
> ```python
> from arcana.runtime_core import RuntimeConfig
> ```

### Per-run overrides

Every call to `runtime.run()` can override defaults:

```python
result = await runtime.run(
    "Complex analysis task",
    engine="conversation",          # or "adaptive" for V1
    max_turns=30,                   # override default 20
    budget=arcana.Budget(max_cost_usd=5.0),  # override budget for this run
    tools=[extra_tool],             # additional tools for this run only
)
```

### Sessions for manual control

For fine-grained access to run state, use a session:

```python
async with runtime.session(max_turns=10) as s:
    result = await s.run("Analyze the dataset")
    print(f"Run ID: {s.run_id}")
    print(f"State:  {s.state}")
    print(f"Budget: {s.budget.to_snapshot()}")
```

### Streaming

Stream execution events in real time:

```python
async for event in runtime.stream("Summarize this article"):
    print(event.event_type, event.content)
```

Streaming is supported with the `conversation` engine only.

### Pipeline

`runtime.chain()` runs a sequence of steps, automatically passing each
step's output as context to the next. Each step can have its own budget
cap, and the chain has an overall budget.

```python
import asyncio
import arcana

async def main():
    runtime = arcana.Runtime(providers={"deepseek": "sk-xxx"})
    result = await runtime.chain([
        arcana.ChainStep(
            name="research",
            goal="Research the topic: AI agents",
            budget=arcana.Budget(max_cost_usd=0.50),
        ),
        arcana.ChainStep(
            name="write",
            goal="Write a blog post based on the research",
            budget=arcana.Budget(max_cost_usd=0.30),
        ),
    ], budget=arcana.Budget(max_cost_usd=1.00))

    print(result.steps["write"])  # Final output

asyncio.run(main())
```

The chain-level budget acts as a hard ceiling. If a step exhausts its own
budget, execution moves to the next step. If the chain budget is exceeded,
the entire chain stops.

### Batch Processing

`runtime.run_batch()` runs many tasks concurrently with controlled
parallelism. Each task is a dict of `run()` keyword arguments.

```python
import asyncio
import arcana
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

async def main():
    runtime = arcana.Runtime(providers={"deepseek": "sk-xxx"})

    articles = ["Article 1 text...", "Article 2 text...", "Article 3 text..."]
    tasks = [
        {"goal": f"Summarize: {article}", "response_format": Summary}
        for article in articles
    ]
    batch = await runtime.run_batch(tasks, concurrency=10)
    print(f"{batch.succeeded}/{len(batch.results)} succeeded, cost: ${batch.total_cost_usd:.4f}")

asyncio.run(main())
```

Failed tasks do not block the batch. Inspect individual results via
`batch.results` -- each entry has `.success`, `.output`, and `.error`.

### Inspecting the runtime

```python
print(runtime.providers)   # ["deepseek", "openai"]
print(runtime.tools)       # ["calculator", "web_search"]
```

---

## Multi-Agent

`runtime.team()` runs multiple agents on a shared goal. Each agent has its
own system prompt and takes turns in a shared conversation. The runtime
manages communication, budget, and trace -- it does not decide strategy or
assign roles.

```python
import asyncio
import arcana

async def main():
    runtime = arcana.Runtime(
        providers={"deepseek": "sk-xxx"},
        budget=arcana.Budget(max_cost_usd=1.0),
    )

    result = await runtime.team(
        goal="Design a REST API for a bookmark manager app.",
        agents=[
            arcana.AgentConfig(
                name="architect",
                prompt="You are a senior API architect. Design clean, RESTful endpoints.",
            ),
            arcana.AgentConfig(
                name="reviewer",
                prompt="You are a security-focused API reviewer. "
                       "If the design is solid, approve with [DONE].",
            ),
        ],
        max_rounds=3,
    )

    print(f"Success: {result.success}")
    print(f"Rounds: {result.rounds}, Cost: ${result.total_cost_usd:.4f}")

    for entry in result.conversation_log:
        print(f"--- Round {entry['round']} [{entry['agent']}] ---")
        print(entry["content"][:300])

asyncio.run(main())
```

Each agent can have its own `model` and `provider` override via
`AgentConfig`. The conversation ends when any agent includes `[DONE]` in
its output, or when `max_rounds` is reached.

---

## Graph Orchestration

For workflows that need deterministic step ordering, branching, or
human-in-the-loop, use `runtime.graph()` to create a `StateGraph`.

Most tasks should use `runtime.run()`. Reach for graphs only when you
need explicit control flow.

```python
import asyncio
from typing import Annotated, Any
from pydantic import BaseModel, Field
from arcana.graph import START, END, StateGraph, append_reducer

class PipelineState(BaseModel):
    messages: Annotated[list, append_reducer] = Field(default_factory=list)
    decision: str = ""

async def analyze(state: dict[str, Any]) -> dict[str, Any]:
    user_msg = state.get("messages", [])[-1]["content"]
    return {
        "decision": "escalate" if "urgent" in user_msg.lower() else "respond",
        "messages": [{"role": "assistant", "content": f"Analyzed: {user_msg}"}],
    }

async def escalate(state: dict[str, Any]) -> dict[str, Any]:
    return {"messages": [{"role": "assistant", "content": "Escalated to human."}]}

async def respond(state: dict[str, Any]) -> dict[str, Any]:
    return {"messages": [{"role": "assistant", "content": "Here is your answer."}]}

def route(state: dict[str, Any]) -> str:
    return "escalate" if state.get("decision") == "escalate" else "respond"

async def main():
    graph = StateGraph(state_schema=PipelineState)
    graph.add_node("analyze", analyze)
    graph.add_node("escalate", escalate)
    graph.add_node("respond", respond)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges("analyze", route, {
        "escalate": "escalate",
        "respond": "respond",
    })
    graph.add_edge("escalate", END)
    graph.add_edge("respond", END)

    app = graph.compile()

    result = await app.ainvoke({
        "messages": [{"role": "user", "content": "URGENT: system down"}],
    })
    print(result["messages"][-1]["content"])

asyncio.run(main())
```

Key concepts:

- **StateGraph** -- directed graph with typed state (Pydantic model).
- **Nodes** -- async functions that receive and return state dicts.
- **Edges** -- `add_edge(a, b)` for fixed routing, `add_conditional_edges()`
  for branching.
- **`START` / `END`** -- special sentinel nodes marking entry and exit.
- **`append_reducer`** -- annotate list fields to accumulate values across
  nodes instead of overwriting.
- **`compile()`** -- returns a runnable app with `ainvoke()` and `astream()`.

> If you are already using `Runtime`, you can use `runtime.graph(state_schema=...)` as a
> shortcut that returns a `StateGraph` pre-connected to the runtime's gateway and tool gateway.

A prebuilt `create_react_agent` factory is available for the standard
agent-calls-tools loop:

```python
from arcana.graph.prebuilt.react_agent import create_react_agent

# Note: _gateway and _tool_gateway are internal APIs; prefer runtime.make_llm_node()
# and runtime.make_tool_node() for graph integration
react = create_react_agent(
    gateway=runtime._gateway,
    tool_gateway=runtime._tool_gateway,
    model_config=model_config,
    system_prompt="You are a helpful assistant.",
)
result = await react.ainvoke({
    "messages": [{"role": "user", "content": "Search for Arcana"}],
})
```

---

## Deployment Patterns

### FastAPI integration

Create the Runtime once at module scope. Every request reuses the same
provider connections and tool registry.

```python
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import arcana

runtime = arcana.Runtime(
    providers={"deepseek": os.environ.get("DEEPSEEK_API_KEY", "")},
    budget=arcana.Budget(max_cost_usd=0.5),
    trace=True,
)

app = FastAPI(title="Agent API")

class AgentRequest(BaseModel):
    goal: str
    max_turns: int = 10

class AgentResponse(BaseModel):
    output: str
    success: bool
    steps: int
    tokens: int
    cost_usd: float

@app.post("/agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    try:
        result = await runtime.run(request.goal, max_turns=request.max_turns)
        return AgentResponse(
            output=str(result.output),
            success=result.success,
            steps=result.steps,
            tokens=result.tokens_used,
            cost_usd=result.cost_usd,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/health")
async def health():
    return {"status": "ok", "providers": runtime.providers, "tools": runtime.tools}
```

Run it:

```bash
export DEEPSEEK_API_KEY=sk-xxx
uvicorn myapp:app --host 0.0.0.0 --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/agent \
     -H "Content-Type: application/json" \
     -d '{"goal": "What is Python?"}'
```

### Environment variables

Arcana resolves API keys from environment variables when the key string is
empty. The convention is `{PROVIDER}_API_KEY`:

| Provider | Environment variable |
|----------|---------------------|
| DeepSeek | `DEEPSEEK_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Kimi | `KIMI_API_KEY` |
| GLM | `GLM_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |

Pass an empty string to read from the environment:

```python
runtime = arcana.Runtime(
    providers={
        "deepseek": "",   # reads DEEPSEEK_API_KEY
        "openai": "",     # reads OPENAI_API_KEY
    },
)
```

Or pass the key directly -- no `.env` file required:

```python
runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
)
```

### Trace storage

Enable trace to get a JSONL audit log of every LLM call, tool invocation,
and runtime decision:

```python
from arcana.runtime_core import RuntimeConfig

runtime = arcana.Runtime(
    providers={"deepseek": ""},
    trace=True,
    config=RuntimeConfig(trace_dir="./my-traces"),
)
```

Traces are written to the `trace_dir` directory (default `./traces`).
Each run produces a JSONL file that can be inspected with the CLI:

```bash
arcana trace serve    # opens the visual trace inspector
```

### MCP tool servers

Connect to external tool servers using the Model Context Protocol:

```python
runtime = arcana.Runtime(
    providers={"deepseek": ""},
    mcp_servers=[
        arcana.MCPServerConfig(name="my-tools", command="npx my-mcp-server"),
    ],
)

# Tools from MCP servers are auto-registered on first run
tool_names = await runtime.connect_mcp()
print(tool_names)
```

### Cleanup

If you use MCP servers or other long-lived connections, close the runtime
when shutting down:

```python
await runtime.close()
```

---

## Next Steps

- **[Architecture](../architecture.md)** -- full system design, layer
  structure, and design principles.
- **[Examples](../../examples/)** -- runnable examples covering every
  feature, from hello world to graph orchestration.
- **[Providers Guide](../guide/providers.md)** -- detailed provider
  configuration and supported models.
- **[GitHub](https://github.com/tyxben/arcana)** -- source code, issues,
  and changelog.
