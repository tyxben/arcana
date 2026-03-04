# Examples

## Prerequisites

1. Install the project:
   ```bash
   cd /path/to/arcana
   pip install -e "."
   ```

2. Copy and configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (DeepSeek, Gemini, etc.)
   ```

## Available Examples

### Trace System Demo

Demonstrates the JSONL trace system, canonical hashing, and multi-provider LLM calls with tracing:

```bash
python examples/demo_trace.py
```

This demo runs without API keys for the trace and hashing portions. The LLM call section requires `DEEPSEEK_API_KEY` and/or `GEMINI_API_KEY` in `.env`.

### Agent Runtime Demo

Demonstrates the full agent runtime with Policy-Step-Reducer pattern, including:
- Basic agent execution
- Budget tracking and trace logging
- Progress detection (stuck agent)
- Checkpointing and resume

```bash
python examples/demo_runtime.py
```

**Note**: Demos 1-2 require `DEEPSEEK_API_KEY` in `.env`. Demos 3-4 use mock gateways and run without API keys.
