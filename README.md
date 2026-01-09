# Arcana

A controllable, reproducible, and evaluable Agent Platform.

## Features

- **Model Gateway**: Pluggable multi-provider LLM support (Gemini, DeepSeek, OpenAI, etc.)
- **Trace System**: JSONL-based event logging for full auditability
- **Contracts**: Pydantic-based strict typing for all data models
- **Budget Management**: Token and cost tracking with limits

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Run tests
uv run pytest

# Run demo
uv run python examples/demo_trace.py
```

## Project Structure

```
arcana/
├── src/arcana/
│   ├── contracts/    # Pydantic data models
│   ├── trace/        # JSONL trace writer/reader
│   ├── gateway/      # Model Gateway + providers
│   └── utils/        # Hashing, config utilities
├── tests/            # pytest test suite
├── examples/         # Usage examples
└── docs/specs/       # Specification documents
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
