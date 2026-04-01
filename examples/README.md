# Arcana Examples

## Quick Start
- `01_hello.py` -- Single LLM call (supports `--provider openai`)
- `02_with_tools.py` -- Tools via SDK
- `18_provider_switching.py` -- DeepSeek / OpenAI / Anthropic switching

## Runtime (Recommended)
- `08_fastapi_integration.py` -- Web service integration
- `09_multi_agent.py` -- Multi-agent collaboration
- `10_tool_with_runtime.py` -- Runtime with custom tools

## Interactive
- `13_interactive_chat.py` -- Multi-turn chat with context and tools
- `14_ask_user.py` -- LLM asks clarifying questions via ask_user tool

## Advanced
- `03_adaptive_policy.py` -- V1 Adaptive Policy engine
- `04_intent_router.py` -- Intent classification
- `05_budget_control.py` -- Budget enforcement
- `06_conversation_agent.py` -- V2 ConversationAgent
- `07_full_demo.py` -- All features demo
- `11_advanced_memory.py` -- Cross-run memory
- `12_graph_orchestration.py` -- StateGraph workflows
- `15_code_review_assistant.py` -- Code review with chat + tools + ask_user
- `16_research_assistant.py` -- Multi-phase pipeline
- `17_context_benchmark.py` -- Context compression (no API key needed)

## Running
All examples need an API key (except `17_context_benchmark.py`):
```bash
export DEEPSEEK_API_KEY=sk-xxx
uv run python examples/01_hello.py

# Use OpenAI instead
export OPENAI_API_KEY=sk-proj-xxx
uv run python examples/01_hello.py --provider openai
```
