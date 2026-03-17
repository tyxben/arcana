# Arcana

**An LLM-native agent framework that treats models as strategy cores, not workflow nodes.**

以 LLM 为策略核心的 Agent 框架 -- 不把模型当流水线工人，而是让它自己决定怎么解决问题。

---

## Why Arcana? / 为什么选 Arcana？

- **Default direct, escalate to agent (默认直答，必要时才 agent)** -- Most requests don't need a loop. The Intent Router sends simple questions straight to the LLM. One call, one answer, done.

- **LLM picks the strategy (LLM 自己选策略)** -- No rigid ReAct chains. The Adaptive Policy asks the model "what do you want to do next?" instead of telling it. It can answer, call tools, run parallel actions, pivot, or stop -- every step.

- **Lazy tool loading (工具懒加载)** -- 10 tools registered, only 3 injected. Tools are selected by relevance to the current goal. Context window is working memory, not a warehouse.

- **Diagnosis, not retry (诊断式恢复，不是机械重试)** -- When a tool fails, the model receives a structured diagnostic brief: what failed, why, and what to try differently. Not a raw traceback. Not "retry 3 times and pray."

- **Budget + full trace (预算管控 + 全链路追踪)** -- Every run has a cost ceiling. Every LLM call, tool invocation, and strategy decision is recorded. You know exactly what happened and what it cost.

---

## Quick Start

```python
import arcana

@arcana.tool(when_to_use="When you need to calculate")
def calc(expression: str) -> str:
    return str(eval(expression))

result = await arcana.run("What is 15 * 37 + 89?", tools=[calc])
print(result.output)  # "644"
```

Five lines. Intent routing, adaptive policy, budget guardrails, and trace logging are all on by default.

---

## Installation

```bash
pip install arcana-agent

# With specific providers:
pip install arcana-agent[anthropic]
pip install arcana-agent[all-providers]
```

Or from source:

```bash
git clone https://github.com/anthropic/arcana.git
cd arcana
pip install -e ".[dev]"
```

---

## Core Concepts / 核心概念

### Intent Router (意图路由)

A lightweight classifier that decides the cheapest correct path for each request. Factual questions get a direct LLM call. Single-tool tasks skip the loop. Only genuinely complex tasks enter the full agent loop. Rule-based by default, with LLM fallback for ambiguous cases.

### Adaptive Policy (自适应策略)

One policy that replaces rigid ReAct and PlanExecute patterns. Each step, the LLM receives current state and chooses from six strategies: `DIRECT_ANSWER`, `SINGLE_TOOL`, `SEQUENTIAL`, `PARALLEL`, `PLAN_AND_EXECUTE`, or `PIVOT`. Strategy can change at any step -- a sequential task can pivot when the approach is wrong, a plan can short-circuit when the answer appears early.

### Lazy Tool Loading (工具懒加载)

Tools declare affordances -- `when_to_use`, `what_to_expect`, `failure_meaning` -- not just JSON schemas. At runtime, only the 3-5 most relevant tools are injected into the context. If the LLM needs a capability that isn't exposed, the runtime searches the full registry and expands dynamically.

### Diagnostic Recovery (诊断式恢复)

Errors are classified (`transient`, `validation`, `permission`, `logic`, `resource`) and packaged into a `DiagnosticBrief` with root cause, recovery options, and priority suggestion. The LLM decides: retry with different arguments, switch tools, narrow scope, or accept failure.

### Multi-Model Routing (多模型路由)

Different tasks in a single run can use different models. Intent classification goes to a cheap/fast model. Strategy decisions go to the strongest model. Context compression goes to a small model. One agent, multiple models, automatic routing.

### Working Set Context (工作集上下文)

Context is organized in four layers: Identity (always present), Task (goal + constraints), Working (current step's data), and External (everything else, loaded on demand). Only the Working layer changes per step. Every token earns its place.

---

## Architecture / 架构

```
Request → Intent Router → Direct Answer (fast path, 1 LLM call)
                        → Agent Loop (Adaptive Policy)
                             ↕
                        Runtime OS
                        ├── Budget Guardian
                        ├── Trace Recorder
                        ├── Tool Gateway (auth, validation, audit)
                        ├── Lazy Tool Registry
                        ├── Diagnostic Recovery
                        └── Model Router
```

The runtime is an operating system, not a pipeline. It provides services -- budget enforcement, trace recording, tool dispatch, error diagnostics -- but never dictates strategy. The LLM calls on them as needed.

---

## Comparison / 框架对比

| Feature | Arcana | LangChain | CrewAI | OpenAI Agents SDK |
|---------|--------|-----------|--------|-------------------|
| Default direct answer (默认直答) | ✅ | ❌ | ❌ | ❌ |
| LLM picks strategy (LLM 选策略) | ✅ | ❌ | ❌ | ❌ |
| Lazy tool loading (工具懒加载) | ✅ | ❌ | ❌ | ❌ |
| Diagnostic recovery (诊断式恢复) | ✅ | ❌ | ❌ | ❌ |
| Budget control (预算管控) | ✅ | ⚠️ | ❌ | ❌ |
| Multi-model routing (多模型路由) | ✅ | ⚠️ | ❌ | ❌ |
| MCP protocol support | 🔜 | ❌ | ❌ | ❌ |
| Rust core rewrite planned | ✅ | ❌ | ❌ | ❌ |

**We believe** most agent frameworks over-engineer the loop and under-engineer the judgment. Arcana inverts this: minimal framework ceremony, maximum model autonomy, strict resource control.

---

## Supported Providers / 支持的模型

| Provider | Models | Status |
|----------|--------|--------|
| **Anthropic** | Claude Opus, Sonnet, Haiku | ✅ |
| **Google** | Gemini Pro, Flash | ✅ |
| **OpenAI** | GPT-4o, GPT-4o-mini | ✅ |
| **DeepSeek** | DeepSeek-Chat, DeepSeek-Reasoner | ✅ |
| **Kimi** (月之暗面) | Moonshot | ✅ |
| **GLM** (智谱) | GLM-4 | ✅ |
| **MiniMax** | abab6.5 | ✅ |
| **Ollama** | Any local model | ✅ |

All cloud providers use the OpenAI-compatible adapter. Adding a new provider is one function call.

---

## The Arcana Constitution / 宪法

> *"LLMs are not workflow nodes. They are reasoning engines. The framework's job is to provide capabilities, manage risk, and record execution -- not to think for the model."*

Arcana is governed by a [Constitution](./CONSTITUTION.md) with four prohibitions and seven principles:

- **No Premature Structuring** -- Don't nail down steps before the LLM assesses the problem.
- **No Controllability Theater** -- Process elegance is not result quality.
- **No Context Hoarding** -- Every token must earn its place.
- **No Mechanical Retry** -- Error recovery is a diagnostic act.

Every design decision answers to this document.

---

## Contributing / 贡献

Before submitting a PR, self-check against the seven principles:

1. Does it honor the fast path (direct by default)?
2. Does it add to context only what's needed (working set discipline)?
3. Can the LLM reason about when to use it (capability, not interface)?
4. Does it expand what problems the LLM can solve, not constrain how (strategy freedom)?
5. Does failure produce something the LLM can act on (actionable feedback)?
6. Is it a service the LLM can call, not a step it's forced through (OS, not workflow)?
7. Does it improve result quality, not just process visibility (outcome-oriented)?

Full details in [CONSTITUTION.md](./CONSTITUTION.md).

```bash
# Development setup
git clone https://github.com/anthropic/arcana.git
cd arcana
uv sync --all-extras

# Run checks
uv run ruff check .         # Lint
uv run mypy src/            # Type check (strict)
uv run pytest               # Test
uv run pytest --cov=arcana  # Test with coverage
```

---

## Roadmap / 路线图

- ✅ **V2 Core** -- Intent Router, Adaptive Policy, Lazy Tools, Diagnostic Recovery
- ✅ **Multi-Model Routing** -- Role-based model selection per call type
- ✅ **Working Set Context** -- 4-layer context management
- 🔜 **MCP Protocol Support** -- Model Context Protocol integration
- 🔜 **CLI Tool** -- `arcana run agent.yaml`
- 🔜 **Trace Web UI** -- Local trace viewer (LangSmith alternative)
- 🔜 **Rust Core Rewrite** -- Zero-cost abstractions, same API shape

---

## License

MIT
