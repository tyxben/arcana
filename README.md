# Arcana

**Agent Runtime for Production** — Budget, tools, trace, and error recovery. Create once, use everywhere.

---

## Quick Start

```bash
pip install arcana-agent
```

```python
import arcana

runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    budget=arcana.Budget(max_cost_usd=5.0),
)

result = await runtime.run("Analyze this data")
print(result.output)
print(f"Cost: ${result.cost_usd:.4f}, Tokens: {result.tokens_used}")
```

Or from CLI:

```bash
export DEEPSEEK_API_KEY=sk-xxx
arcana run "What is 2+2?"
arcana run agent.yaml
```

---

## Why Arcana

- **Runtime, not framework** — Arcana doesn't tell your LLM how to think. It provides budget, tools, trace, and error recovery as managed services.
- **Create once, use everywhere** — `Runtime` holds provider connections, tool registry, and budget policy. Reuse across requests, workers, and services.
- **Production defaults** — Every run has budget limits, tool authorization, full trace logging, and structured error diagnosis.

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

## Connect MCP Tools

```python
from arcana.contracts.mcp import MCPServerConfig

runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    mcp_servers=[
        MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        ),
    ],
)
# MCP tools auto-discovered and available to the agent
```

Supports stdio and Streamable HTTP transports.

---

## Streaming

```python
async for event in runtime.stream("Write a poem"):
    if event.event_type.value == "llm_chunk":
        print(event.content, end="", flush=True)
```

Real token-level streaming from any OpenAI-compatible provider.

---

## Integrate with Your Service

```python
from fastapi import FastAPI
import arcana, os

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

| Capability | Description |
|-----------|-------------|
| Budget Control | Token and cost limits per run |
| Tool Authorization | Tools need explicit capabilities |
| Full Trace | Every LLM call and tool invocation logged |
| Error Diagnosis | Structured feedback on failures |
| Context Management | Auto-compression for long conversations |
| Token Streaming | Real-time token output |
| Multi-Provider | DeepSeek, OpenAI, Anthropic, Kimi, GLM, MiniMax, Gemini, Ollama |
| MCP Support | stdio + Streamable HTTP transports |
| YAML Config | `arcana run agent.yaml` |
| Trace Web UI | `arcana trace serve` for visual debugging |

---

## Architecture

```
Runtime Default Path (built-in):
  SDK           — arcana.run() / Runtime.run() / Runtime.stream()
  Tool Runtime  — ToolGateway + MCP
  Context       — WorkingSetBuilder (auto-compression, token budget)
  Memory        — RunMemoryStore (lightweight cross-run recall)
  Engine        — ConversationAgent (LLM-native, default)

Advanced Platform Capabilities (composable):
  Graph Engine    — StateGraph + LLMNode/ToolNode + Interrupt/Resume
  Multi-Agent     — runtime.team()
  Advanced Memory — MemoryManager + Governance
```

Most users stay on the default path. Advanced capabilities activate when needed.

---

## Supported Providers

| Provider | Status |
|----------|--------|
| **DeepSeek** | Verified |
| **OpenAI** | Verified |
| **Anthropic** | Verified |
| **Kimi** (Moonshot) | Supported |
| **GLM** (Zhipu) | Supported |
| **MiniMax** | Supported |
| **Google Gemini** | Supported |
| **Ollama** | Supported |

All cloud providers use the OpenAI-compatible adapter. Adding a new provider is one function call.

---

## Installation

```bash
pip install arcana-agent                     # Core
pip install arcana-agent[anthropic]          # + Claude
pip install arcana-agent[all-providers]      # All providers
pip install arcana-agent[ui]                 # + Trace Web UI
```

---

## The Constitution

> *"The framework provides capabilities, manages risk, and records execution. The LLM understands goals, forms strategies, and adapts. Never reverse this."*

See [CONSTITUTION.md](./CONSTITUTION.md).

---

## License

MIT
