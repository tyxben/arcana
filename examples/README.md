# Arcana Examples

## Quick Start
- `01_hello.py` -- Single LLM call
- `02_with_tools.py` -- Tools via SDK

## Runtime (Recommended)
- `08_fastapi_integration.py` -- Web service integration
- `09_multi_agent.py` -- Multi-agent collaboration
- `10_tool_with_runtime.py` -- Runtime with custom tools

## Advanced
- `03_adaptive_policy.py` -- V1 Adaptive Policy engine
- `04_intent_router.py` -- Intent classification
- `05_budget_control.py` -- Budget enforcement
- `06_conversation_agent.py` -- V2 ConversationAgent
- `07_full_demo.py` -- All features demo

## Running
All examples need an API key:
```bash
export DEEPSEEK_API_KEY=sk-xxx
uv run python examples/08_fastapi_integration.py
```
