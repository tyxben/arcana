# Arcana

**Agent Runtime for Production** — Budget, tools, trace, and error recovery. Create once, use everywhere.

面向生产环境的 Agent 运行时 — 预算管控、工具权限、全链路追踪、错误诊断。创建一次，处处复用。

---

## Why Arcana

- **Runtime, not framework** — Arcana doesn't tell your LLM how to think. It provides budget, tools, trace, and error recovery as managed services. Your agent decides the strategy.

- **Create once, use everywhere** — `Runtime` holds provider connections, tool registry, and budget policy. Reuse across requests, workers, and services.

- **Production defaults** — Every run has budget limits, tool authorization, full trace logging, and structured error diagnosis. No configuration needed.

---

## Quick Start

```python
import arcana

# Create runtime (once, at startup)
runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    budget=arcana.Budget(max_cost_usd=5.0),
    trace=True,
)

# Run tasks
result = await runtime.run("Analyze this data")
print(result.output)
print(f"Cost: ${result.cost_usd:.4f}, Steps: {result.steps}")
```

---

## Add Tools

```python
@arcana.tool(when_to_use="For math calculations")
def calc(expression: str) -> str:
    return str(eval(expression))

runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    tools=[calc],
)

result = await runtime.run("What is 15 * 37 + 89?")
# Agent calls calc("15 * 37 + 89") → "644"
```

---

## Integrate with Your Service

```python
from fastapi import FastAPI
import arcana
import os

app = FastAPI()
runtime = arcana.Runtime(
    providers={"deepseek": os.environ["DEEPSEEK_API_KEY"]},
    budget=arcana.Budget(max_cost_usd=1.0),
    trace=True,
)

@app.post("/agent")
async def agent_endpoint(goal: str):
    result = await runtime.run(goal)
    return {"answer": result.output, "cost": result.cost_usd}
```

---

## What You Get for Free

| Capability | Description | Default |
|-----------|-------------|---------|
| Budget Control | Token and cost limits per run | On |
| Tool Authorization | Tools need explicit capabilities | On |
| Full Trace | Every LLM call and tool invocation logged | Opt-in |
| Error Diagnosis | Structured feedback on failures | On |
| Multi-Provider | DeepSeek, OpenAI, Anthropic, Kimi, GLM, MiniMax, Ollama | Any |
| Native Tool Use | LLM calls tools directly, no wrapper | On |

---

## Three Layers

```
Layer 1: SDK                    ← arcana.run() / Runtime.run()
Layer 2: Tool Runtime           ← ToolGateway + MCP (coming soon)
Layer 3: Advanced Orchestration ← Graph / Multi-Agent (when needed)
```

Most users stay on Layer 1. Layer 2 activates when you add tools. Layer 3 is for complex workflows.

---

## Installation

```bash
pip install arcana-agent
pip install arcana-agent[anthropic]        # + Claude support
pip install arcana-agent[all-providers]    # everything
```

Or from source:

```bash
git clone https://github.com/anthropic/arcana.git
cd arcana
uv sync --all-extras
```

---

## Supported Providers

| Provider | Models | Status |
|----------|--------|--------|
| **DeepSeek** | DeepSeek-Chat, DeepSeek-Reasoner | Verified |
| **OpenAI** | GPT-4o, GPT-4o-mini | Verified |
| **Anthropic** | Claude Opus, Sonnet, Haiku | Verified |
| **Kimi** (月之暗面) | Moonshot | Supported |
| **GLM** (智谱) | GLM-4 | Supported |
| **MiniMax** | abab6.5 | Supported |
| **Google** | Gemini Pro, Flash | Supported |
| **Ollama** | Any local model | Supported |

All cloud providers use the OpenAI-compatible adapter. Adding a new provider is one function call.

---

## The Constitution

> *"The framework provides capabilities, manages risk, and records execution. The LLM understands goals, forms strategies, and adapts. Never reverse this."*

Arcana is governed by a [Constitution](./CONSTITUTION.md) — four prohibitions (no premature structuring, no controllability theater, no context hoarding, no mechanical retry) and seven design principles. Every design decision answers to this document.

---

## Comparison

| | Arcana | LangChain | OpenAI SDK |
|--|--------|-----------|-----------|
| Type | Runtime | Framework | SDK |
| Budget control | Built-in | No | No |
| Tool auth/audit | Built-in | No | No |
| Trace logging | Built-in | Paid (LangSmith) | No |
| Error diagnosis | Structured | Retry | No |
| Create once, reuse | Runtime | Per-chain | Per-call |
| LLM decides strategy | Yes | Framework decides | Yes |

---

## Roadmap

- [x] Multi-provider gateway (DeepSeek, OpenAI, Anthropic, + 5 more)
- [x] Budget enforcement and full trace logging
- [x] Structured error diagnosis
- [x] Native tool use with authorization
- [ ] `Runtime` top-level API (in progress)
- [ ] MCP protocol support
- [ ] CLI tool — `arcana run agent.yaml`
- [ ] Trace Web UI (local LangSmith alternative)
- [ ] Rust core rewrite

---

## Contributing

Before submitting a PR, self-check against the Constitution:

1. Does it honor the fast path (direct by default)?
2. Does it add to context only what's needed?
3. Can the LLM reason about when to use it?
4. Does it expand what problems the LLM can solve, not constrain how?
5. Does failure produce something the LLM can act on?
6. Is it a service the LLM can call, not a step it's forced through?
7. Does it improve result quality, not just process visibility?

Full details in [CONSTITUTION.md](./CONSTITUTION.md).

```bash
git clone https://github.com/anthropic/arcana.git
cd arcana
uv sync --all-extras

uv run ruff check .         # Lint
uv run mypy src/            # Type check
uv run pytest               # Test
uv run pytest --cov=arcana  # Coverage
```

---

## License

MIT
