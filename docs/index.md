# Arcana

**Agent runtime that lets LLMs think, not just execute.**

[![PyPI](https://img.shields.io/pypi/v/arcana-agent)](https://pypi.org/project/arcana-agent/)
[![Python](https://img.shields.io/pypi/pyversions/arcana-agent)](https://pypi.org/project/arcana-agent/)
[![License](https://img.shields.io/github/license/tyxben/arcana)](https://github.com/tyxben/arcana/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/tyxben/arcana/ci.yml?label=tests)](https://github.com/tyxben/arcana/actions)
[![Coverage](https://img.shields.io/codecov/c/github/tyxben/arcana)](https://codecov.io/gh/tyxben/arcana)

Give your LLM a runtime with budget, tools, and trace -- then let it decide how to solve the problem.

[Get Started](guide/quickstart.md){ .md-button .md-button--primary }
[GitHub](https://github.com/tyxben/arcana){ .md-button }

```python
import asyncio
import arcana

async def main():
    result = await arcana.run("Summarize the key benefits of Rust", api_key="sk-xxx")
    print(result.output)
    print(f"Cost: ${result.cost_usd:.4f} | Tokens: {result.tokens_used}")

asyncio.run(main())
```

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
print(result.parsed.title)       # str
print(result.parsed.key_points)  # list[str]
```

!!! info
    `result.parsed` is always `BaseModel | None` -- never a raw dict. It is `None` when no `response_format` is set or when parsing fails. `result.output` contains the same parsed model when successful, or the raw text when parsing fails.

---

## More Capabilities

Arcana also supports multimodal input, ask-user interaction, sequential pipelines with parallel branches, batch processing, multi-agent teams, budget scoping, and graph orchestration. See the [Quick Start guide](guide/quickstart.md) for details on each.

---

## What Makes Arcana Different

| | Traditional Frameworks | **Arcana** |
|---|---|---|
| **LLM autonomy** | Framework-driven chains or role-locked agents | LLM decides strategy within runtime boundaries |
| **Token efficiency** | Full context every call | Working-set discipline -- only what this step needs |
| **Tool management** | All tools in every prompt | Dynamic per-turn exposure with affordances |
| **User interaction** | Not built in | `ask_user` built-in, graceful fallback if no handler |
| **Default path** | Chain/graph/crew required | Direct answer when possible, agent loop when needed |

---

## Providers

DeepSeek | OpenAI | Anthropic | Google Gemini | Kimi (Moonshot) | GLM (Zhipu) | MiniMax | Ollama

All providers use a single OpenAI-compatible adapter. Adding a new provider is one function call.

---

## Documentation

| Guide | Description |
|---|---|
| [Quick Start](guide/quickstart.md) | Installation through deployment |
| [Configuration](guide/configuration.md) | Full configuration reference |
| [Providers](guide/providers.md) | Provider setup and fallback chains |
| [API Reference](api/index.md) | Public API documentation |
| [Architecture](architecture.md) | System design and internals |
| [Constitution](constitution.md) | Design principles and prohibitions |
| [Changelog](changelog.md) | Release history |

---

## Why Arcana

Most agent frameworks treat LLMs as unreliable workers that need a manager watching every step. They force rigid formats -- ReAct loops, JSON command schemas, mechanical retries -- and dump entire tool catalogs into every prompt. The result is high ceremony, low capability. The framework spends its complexity constraining the LLM instead of releasing it.

Arcana takes the opposite approach: it is an operating system for LLM agents, not a pipeline. The LLM decides strategy; the runtime provides services -- budget enforcement, tool dispatch, trace recording, context management. The framework never interprets LLM output: raw facts (`TurnFacts`) and runtime assessment (`TurnAssessment`) are kept visibly separate. Eight design principles and four prohibitions, codified in a [Constitution](constitution.md), govern every line of code.

---

## License

MIT
