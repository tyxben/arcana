# Arcana

A controllable, reproducible, and evaluable Agent Platform.

## Features

### Core Platform

- **Model Gateway**: Pluggable multi-provider LLM support (Gemini, DeepSeek, OpenAI, etc.)
- **Trace System**: JSONL-based event logging for full auditability
- **Contracts**: Pydantic-based strict typing for all data models
- **Budget Management**: Token and cost tracking with limits

### Agent Runtime (NEW ✨)

- **Execution Engine**: Policy-Step-Reducer pattern with state machine control
- **Progress Detection**: Automatic detection of loops and stuck states
- **Checkpointing**: Resume from any step with state snapshots
- **Replay System**: Debug by replaying execution with cached responses
- **Schema Validation**: LLM output validation with automatic retries
- **Error Handling**: Classified error types with retry strategies

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Run tests
uv run pytest

# Run demos
uv run python examples/demo_trace.py     # Trace system demo
uv run python examples/demo_runtime.py   # Agent runtime demo
```

## Project Structure

```
arcana/
├── src/arcana/
│   ├── contracts/    # Pydantic data models
│   ├── trace/        # JSONL trace writer/reader
│   ├── gateway/      # Model Gateway + providers
│   ├── runtime/      # Agent Runtime (execution engine)
│   │   ├── agent.py           # Main orchestrator
│   │   ├── step.py            # Step executor
│   │   ├── state_manager.py  # State & checkpointing
│   │   ├── progress.py        # Progress detector
│   │   ├── validator.py       # Schema validator
│   │   ├── replay.py          # Replay engine
│   │   ├── error_handler.py   # Error handling
│   │   ├── policies/          # Decision policies
│   │   └── reducers/          # State reducers
│   └── utils/        # Hashing, config utilities
├── tests/            # pytest test suite
├── examples/         # Usage examples
└── docs/             # Documentation
```

## Usage Example

```python
import asyncio
from arcana.runtime.agent import Agent
from arcana.runtime.policies.react import ReActPolicy
from arcana.runtime.reducers.default import DefaultReducer
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.gateway.providers import create_deepseek_provider
from arcana.contracts.runtime import RuntimeConfig

async def main():
    # Setup gateway
    gateway = ModelGatewayRegistry()
    gateway.register("deepseek", create_deepseek_provider(api_key="your-key"))
    gateway.set_default("deepseek")

    # Create agent
    agent = Agent(
        policy=ReActPolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=10),
    )

    # Run task
    state = await agent.run("Calculate the 10th Fibonacci number")
    print(f"Status: {state.status}")
    print(f"Steps: {state.current_step}")

asyncio.run(main())
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run linter
uv run ruff check .

# Run type checker
uv run mypy src/

# Run tests with coverage
uv run pytest --cov=arcana
```

## License

MIT
