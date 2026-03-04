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

## Module Maturity

| Module | Status | Description |
|--------|--------|-------------|
| Contracts | ✅ Stable | Pydantic data models for all subsystems |
| Trace | ✅ Stable | JSONL event logging, reader/writer |
| Model Gateway | ✅ Stable | Multi-provider LLM routing with fallback |
| Budget Tracker | ✅ Stable | Token/cost/time enforcement |
| Agent Runtime | ✅ Stable | Policy-Step-Reducer execution engine |
| Progress Detection | ✅ Stable | Loop and stuck state detection |
| Checkpoint/Resume | ✅ Stable | State snapshots and recovery |
| Replay Engine | ✅ Stable | Debug via cached response replay |
| Schema Validation | ✅ Stable | LLM output validation with retries |
| Error Handling | ✅ Stable | Classified errors with retry strategies |
| Tool Gateway | 🔧 Beta | Authorization, validation, audit pipeline |
| RAG | 🔧 Beta | Embedding, retrieval, reranking |
| Memory | 🔧 Beta | Working, long-term, episodic memory |
| Orchestrator | 🔧 Beta | Task DAG scheduling, concurrent execution |
| Multi-Agent | 🔧 Beta | Team collaboration (Planner-Executor-Critic) |
| Observability | 🧪 Alpha | Metrics collection and hooks |
| Eval Harness | 🧪 Alpha | Evaluation runner and quality gates |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Run linting
make lint

# Run type checking
make typecheck

# Run tests
make test

# Run demos
python examples/demo_trace.py     # Trace system demo
python examples/demo_runtime.py   # Agent runtime demo
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
pip install -e ".[dev]"

# Run all checks
make all          # lint + typecheck + test

# Individual commands
make lint         # ruff check
make format       # ruff format + fix
make typecheck    # mypy strict
make test         # pytest
make test-cov     # pytest with coverage
make clean        # remove build artifacts
```

## License

MIT
