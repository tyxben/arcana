# Examples

Runnable examples covering every Arcana feature, from a single LLM call to multi-agent collaboration and graph orchestration.

All examples live in the
[`examples/`](https://github.com/tyxben/arcana/tree/main/examples)
directory of the repository.

## Prerequisites

```bash
pip install arcana-agent[all-providers]

# Set at least one provider key
export DEEPSEEK_API_KEY=sk-xxx
```

---

## Getting Started

| Example | Description |
|---------|-------------|
| [`01_hello.py`][01] | **Hello World** -- the simplest agent: one LLM call, one response. Supports `--provider openai`. |
| [`02_with_tools.py`][02] | **Tools via SDK** -- define a tool with `@arcana.tool` and call `arcana.run()`. |
| [`18_provider_switching.py`][18] | **Provider Switching** -- DeepSeek, OpenAI, Anthropic usage with both `arcana.run()` and `Runtime`. |
| [`06_conversation_agent.py`][06] | **ConversationAgent** -- V2 engine with direct answer, tool usage, and streaming events. |
| [`07_full_demo.py`][07] | **Full Demo** -- end-to-end integration: intent routing, adaptive policy, tools, and the full SDK API. |

[01]: https://github.com/tyxben/arcana/blob/main/examples/01_hello.py
[02]: https://github.com/tyxben/arcana/blob/main/examples/02_with_tools.py
[18]: https://github.com/tyxben/arcana/blob/main/examples/18_provider_switching.py
[06]: https://github.com/tyxben/arcana/blob/main/examples/06_conversation_agent.py
[07]: https://github.com/tyxben/arcana/blob/main/examples/07_full_demo.py

### Quick taste -- Tools with the SDK

```python
import asyncio
import arcana

@arcana.tool(
    when_to_use="When you need to calculate math expressions",
    what_to_expect="Returns the numeric result",
)
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

async def main():
    result = await arcana.run(
        "What is (15 * 37) + (89 * 2)?",
        tools=[calculator],
        provider="deepseek",
        api_key="sk-xxx",
        max_turns=5,
    )
    print(f"Answer: {result.output}")
    print(f"Steps: {result.steps}, Tokens: {result.tokens_used}")

asyncio.run(main())
```

---

## Tools & Features

| Example | Description |
|---------|-------------|
| [`05_budget_control.py`][05] | **Budget Control** -- agent stops when budget is exhausted; never runs away. |
| [`10_tool_with_runtime.py`][10] | **Custom Tools with Runtime** -- register tools once on Runtime; every run gets authorization, validation, and audit logging. |
| [`13_interactive_chat.py`][13] | **Interactive Chat** -- multi-turn conversation with `runtime.chat()`, persistent history, shared budget, and tools. |
| [`14_ask_user.py`][14] | **Ask User** -- LLM asks clarifying questions mid-execution via the built-in `ask_user` tool. Graceful fallback when no handler is provided. |
| [`08_fastapi_integration.py`][08] | **FastAPI Integration** -- embed Arcana Runtime in a production web service; create once at startup, reuse across requests. |
| [`17_context_benchmark.py`][17] | **Context Benchmark** -- demonstrates Arcana's token-efficient context compression vs naive full-context. No API key required. |

[05]: https://github.com/tyxben/arcana/blob/main/examples/05_budget_control.py
[10]: https://github.com/tyxben/arcana/blob/main/examples/10_tool_with_runtime.py
[13]: https://github.com/tyxben/arcana/blob/main/examples/13_interactive_chat.py
[14]: https://github.com/tyxben/arcana/blob/main/examples/14_ask_user.py
[08]: https://github.com/tyxben/arcana/blob/main/examples/08_fastapi_integration.py
[17]: https://github.com/tyxben/arcana/blob/main/examples/17_context_benchmark.py

### Quick taste -- Multi-turn chat session

```python
import asyncio
import arcana

async def main():
    runtime = arcana.Runtime(
        providers={"deepseek": "sk-xxx"},
        tools=[calculator],
        budget=arcana.Budget(max_cost_usd=1.0),
    )

    async with runtime.chat(
        system_prompt="You are a helpful math tutor.",
    ) as session:
        response = await session.send("What is 123 * 456?")
        print(response.content)

        response = await session.send("Now divide that by 3")
        print(response.content)

        print(f"Total cost: ${session.total_cost_usd:.4f}")

    await runtime.close()

asyncio.run(main())
```

---

## Advanced Patterns

| Example | Description |
|---------|-------------|
| [`03_adaptive_policy.py`][03] | **Adaptive Policy** -- V1 engine where the agent chooses its own strategy: direct answer, tool call, plan, or pivot. |
| [`04_intent_router.py`][04] | **Intent Router** -- simple questions get direct answers (1 LLM call); complex tasks enter the agent loop. |
| [`09_multi_agent.py`][09] | **Multi-Agent Collaboration** -- two agents collaborate (one designs, one reviews) with shared budget and trace. |
| [`11_advanced_memory.py`][11] | **Advanced Memory** -- composable `MemoryManager` for governed multi-tier memory outside the Runtime. |
| [`12_graph_orchestration.py`][12] | **Graph Orchestration** -- explicit nodes, edges, reducers, and interrupt/resume for deterministic workflows. Includes custom graph and prebuilt ReAct patterns. |
| [`15_code_review_assistant.py`][15] | **Code Review Assistant** -- practical demo combining chat + tools + ask_user for interactive code review. |
| [`16_research_assistant.py`][16] | **Research Assistant** -- multi-phase pipeline: tool-driven research, team analysis, and structured output. |

[03]: https://github.com/tyxben/arcana/blob/main/examples/03_adaptive_policy.py
[04]: https://github.com/tyxben/arcana/blob/main/examples/04_intent_router.py
[09]: https://github.com/tyxben/arcana/blob/main/examples/09_multi_agent.py
[11]: https://github.com/tyxben/arcana/blob/main/examples/11_advanced_memory.py
[12]: https://github.com/tyxben/arcana/blob/main/examples/12_graph_orchestration.py
[15]: https://github.com/tyxben/arcana/blob/main/examples/15_code_review_assistant.py
[16]: https://github.com/tyxben/arcana/blob/main/examples/16_research_assistant.py

---

## Running Examples

All examples follow the same pattern:

```bash
# Set your API key
export DEEPSEEK_API_KEY=sk-xxx

# Run any example
uv run python examples/01_hello.py
```

Most examples use DeepSeek by default. To use a different provider, set the
corresponding environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
and change the `provider` parameter in the example code.

The context benchmark (`17_context_benchmark.py`) runs entirely with simulated
data and requires no API key.
