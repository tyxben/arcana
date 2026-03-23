# Arcana

**Runtime-first agent framework for context-governed, production-oriented LLM execution.**

```bash
pip install arcana-agent
```

Arcana is not a toolkit for assembling agents. It is a runtime that manages every LLM call — context, tools, budget, trace — so your agent code stays small and your production stays auditable.

---

## Why Arcana

- **Context management** — Every LLM call, Arcana can explain why it composed the prompt that way. Context blocks are assembled, budgeted, and tracked, not concatenated.
- **Tool governance** — Dynamic tool exposure per turn, per-tool auth and validation, side-effect declarations, full audit trail. Tools are runtime-managed resources, not loose functions.
- **Runtime diagnosis** — Structured trace of every decision: which tools were called, what the LLM returned, why the runtime continued or stopped. Not just logs.
- **Production defaults** — Budget limits, provider fallback, retry, checkpoint, and cost tracking are built in. You opt out of safety, not into it.

---

## Quick Start

```python
import arcana

runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    budget=arcana.Budget(max_cost_usd=1.0),
)

result = await runtime.run("Summarize the key trends in quantum computing")
print(result.output)
print(f"Cost: ${result.cost_usd:.4f} | Tokens: {result.tokens_used}")
```

Add tools:

```python
@arcana.tool(when_to_use="For math calculations")
def calc(expression: str) -> str:
    return str(eval(expression))

runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    tools=[calc],
)
result = await runtime.run("What is 15 * 37 + 89?")
```

Or one-shot for scripts:

```python
result = await arcana.run("Hello", api_key="sk-xxx")
```

---

## What Arcana is NOT

- **Not LangChain.** Arcana is not chasing ecosystem breadth. There is one runtime, one tool protocol, one trace format. Depth over surface area.
- **Not graph-first.** Graph orchestration exists as an advanced capability, but the default path is a single runtime loop. You reach for graphs when you need them, not because the framework demands it.
- **Not an MCP framework.** MCP is a supported transport for tools (stdio + Streamable HTTP). It is a capability of the tool gateway, not the product.

---

## Core Concepts

| Concept | What it does |
|---------|-------------|
| **Runtime** | Holds providers, tools, budget, and trace config. Created once, reused across requests. |
| **Context Management** | Assembles context blocks with token budgets and auto-compression. Every prompt is explainable. |
| **Tool Gateway** | Registers tools with schemas, auth, validation, and audit. Governs what the LLM can call per turn. |
| **Trace** | JSONL audit log of every LLM call, tool invocation, and runtime decision. Replayable. |
| **Budget** | Token and cost limits enforced per run. The runtime stops before you overspend. |

---

## Supported Providers

| Provider | Status |
|----------|--------|
| DeepSeek | Verified |
| OpenAI | Verified |
| Anthropic | Verified |
| Kimi (Moonshot) | Supported |
| GLM (Zhipu) | Supported |
| MiniMax | Supported |
| Google Gemini | Supported |
| Ollama | Supported |

All providers use a single OpenAI-compatible adapter. Adding a new provider is one function call.

---

## Installation

```bash
pip install arcana-agent                   # Core
pip install arcana-agent[anthropic]        # + Claude support
pip install arcana-agent[all-providers]    # All providers
pip install arcana-agent[ui]              # + Trace Web UI
```

---

## CLI

```bash
arcana run "What is 2+2?"
arcana run agent.yaml
arcana run agent.yaml --provider openai --max-cost 0.50

arcana trace serve                         # Visual trace inspector
arcana eval run eval_suite.yaml            # Run evaluation suite
```

---

## Links

- [Architecture](./docs/architecture.md)
- [Examples](./examples/)
- [Constitution](./CONSTITUTION.md)
- [Changelog](./CHANGELOG.md)

## License

MIT
