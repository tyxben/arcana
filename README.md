# Arcana

**Agent runtime that lets LLMs think, not just execute.**

[![Version](https://img.shields.io/badge/version-0.1.0b7-blue)](https://pypi.org/project/arcana-agent/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-1045%20passing-brightgreen)]()

---

## The Problem

Most agent frameworks treat LLMs as unreliable workers that need a manager watching every step. They force rigid formats -- ReAct loops, JSON command schemas, mechanical retries -- and dump entire tool catalogs into every prompt. The result is high ceremony, low capability. The framework spends its complexity constraining the LLM instead of releasing it.

## The Arcana Approach

Arcana is an operating system for LLM agents, not a pipeline. The LLM decides strategy; the runtime provides services -- budget enforcement, tool dispatch, trace recording, context management. The framework never interprets LLM output: raw facts (`TurnFacts`) and runtime assessment (`TurnAssessment`) are kept visibly separate. Eight design principles and four prohibitions, codified in a [Constitution](./CONSTITUTION.md), govern every line of code.

---

## Quick Start

```bash
pip install arcana-agent
```

```python
import arcana

result = await arcana.run("Summarize this article", api_key="sk-xxx")
print(result.output)
print(f"Cost: ${result.cost_usd:.4f} | Tokens: {result.tokens_used}")
```

---

## Core Features

### Tools with Affordances

Tools declare *when* and *why*, not just *how*. The LLM reasons about whether to call a tool, not just how.

```python
@arcana.tool(
    when_to_use="When you need to do math calculations",
    what_to_expect="Returns the numeric result as a string",
    failure_meaning="The expression was malformed",
)
def calc(expression: str) -> str:
    return str(eval(expression))

result = await arcana.run("What is 15 * 37 + 89?", tools=[calc], api_key="sk-xxx")
```

### Runtime

Create once at startup, use across your entire application. Holds providers, tools, budget, and trace as long-lived resources.

```python
runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx", "openai": "sk-proj-xxx"},
    tools=[calc, web_search],
    budget=arcana.Budget(max_cost_usd=5.0),
    trace=True,
)

result = await runtime.run("Analyze recent trends in quantum computing")
```

### Interactive Chat

Multi-turn sessions with persistent history, shared budget, and context compression.

```python
async with runtime.chat() as c:
    r = await c.send("What are the main themes in this dataset?")
    r = await c.send("Expand on the second theme")
    print(c.total_cost_usd)
```

### Ask User

The LLM can ask clarifying questions mid-execution. If no handler is provided, it proceeds with best judgment -- interaction is a capability, not a dependency.

```python
result = await runtime.run(
    "Book a restaurant for dinner",
    input_handler=lambda q: input(f"Agent asks: {q}\n> "),
)
```

### Structured Output

Return validated Pydantic instances instead of raw text.

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

result = await arcana.run(
    "Summarize this article",
    response_format=Summary,
    api_key="sk-xxx",
)
print(result.output.title)       # str
print(result.output.key_points)  # list[str]
```

### Multimodal

Pass images alongside text. URLs, local file paths, and data URIs all work.

```python
result = await arcana.run(
    "Describe what you see in this image",
    images=["https://example.com/photo.jpg"],
    provider="openai",
    api_key="sk-proj-xxx",
)
```

### Multi-Agent Teams

Runtime provides communication and budget. Agents decide strategy -- no forced hierarchy.

```python
result = await runtime.team(
    "Design a landing page for an AI product",
    agents=[
        arcana.AgentConfig(name="designer", prompt="You are a senior UX designer."),
        arcana.AgentConfig(name="copywriter", prompt="You are a conversion copywriter."),
        arcana.AgentConfig(name="critic", prompt="You find weaknesses and suggest improvements."),
    ],
    max_rounds=3,
)
```

### Graph Orchestration

For workflows that need explicit state machines. Available when you need it, never forced.

```python
from arcana import StateGraph, START, END

graph = runtime.graph(state_schema=MyState)
graph.add_node("research", research_fn)
graph.add_node("write", write_fn)
graph.add_edge(START, "research")
graph.add_edge("research", "write")
graph.add_edge("write", END)

app = graph.compile()
result = await app.ainvoke(initial_state)
```

---

## What Makes Arcana Different

| | LangChain | CrewAI | AutoGPT | **Arcana** |
|---|---|---|---|---|
| **LLM autonomy** | Framework-driven chains | Role-locked agents | Fully autonomous, no guardrails | LLM decides strategy within runtime boundaries |
| **Token efficiency** | Full context every call | Full prompt per agent | Unbounded context growth | Working-set discipline -- only what this step needs |
| **Thinking signals** | Ignored | Ignored | Ignored | Runtime listens to thinking for confidence, never constrains |
| **Tool management** | All tools in every prompt | Per-agent tool sets | All tools always | Dynamic per-turn exposure with affordances |
| **User interaction** | Not built in | Not built in | Not built in | `ask_user` built-in, graceful fallback if no handler |
| **Default path** | Chain/graph required | Crew required | Agent loop always | Direct answer when possible, agent loop when needed |

---

## Providers

DeepSeek | OpenAI | Anthropic | Google Gemini | Kimi (Moonshot) | GLM (Zhipu) | MiniMax | Ollama

All providers use a single OpenAI-compatible adapter. Adding a new provider is one function call.

---

## Documentation

| Guide | Description |
|---|---|
| [Quick Start](./docs/guide/quickstart.md) | Installation through deployment |
| [Configuration](./docs/guide/configuration.md) | Full configuration reference |
| [Providers](./docs/guide/providers.md) | Provider setup and fallback chains |
| [API Reference](./docs/guide/api.md) | Public API documentation |
| [Architecture](./docs/architecture.md) | System design and internals |
| [Constitution](./CONSTITUTION.md) | Design principles and prohibitions |
| [Examples](./examples/) | Runnable code examples |
| [Changelog](./CHANGELOG.md) | Release history |

---

## Installation

```bash
pip install arcana-agent                   # Core (DeepSeek, OpenAI)
pip install arcana-agent[anthropic]        # + Claude support
pip install arcana-agent[gemini]           # + Gemini support
pip install arcana-agent[all-providers]    # All providers
pip install arcana-agent[ui]              # + Trace Web UI
```

Or with uv:

```bash
uv add arcana-agent
uv add arcana-agent --extra all-providers
```

Requires Python 3.11+.

---

## License

MIT
