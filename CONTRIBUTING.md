# Contributing to Arcana

## The Arcana Constitution

Every contribution must align with the [CONSTITUTION.md](CONSTITUTION.md). Before submitting a PR, check:

1. Does this help the LLM or restrict it?
2. Does it add a translation layer the LLM doesn't need?
3. Does it keep the context window clean?
4. Does it allow strategy leaps?
5. Does it provide actionable feedback on errors?
6. Is the runtime serving the LLM or controlling it?
7. Does it evaluate results or process?

If a change fails any of these checks, it needs redesign.

## Development Setup

```bash
# Clone
git clone https://github.com/tysama/arcana-agent.git
cd arcana-agent

# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Type check
uv run mypy src/
```

## Project Structure

```
src/arcana/
├── contracts/      # Pydantic data models (the foundation)
├── runtime/        # Execution engines
│   ├── conversation.py  # V2: ConversationAgent (recommended)
│   ├── agent.py         # V1: Agent + Policy + Reducer
│   ├── policies/        # Adaptive, ReAct, PlanExecute
│   ├── diagnosis/       # Error diagnosis + recovery
│   └── reducers/        # State reducers (V1)
├── routing/        # Intent classification + fast paths
├── gateway/        # LLM providers + model routing
├── tool_gateway/   # Tool execution pipeline
├── context/        # Working set builder
├── streaming/      # SSE adapter
├── trace/          # JSONL event logging
├── memory/         # Working + long-term + episodic
├── eval/           # Evaluation framework
├── graph/          # Graph execution engine
├── sdk.py          # Public API (arcana.run, @arcana.tool)
└── ...
```

## Key Principles

### TurnFacts vs TurnAssessment
Provider facts and runtime interpretation must NEVER be mixed. `_parse_turn()` only writes facts. `_assess_turn()` only writes assessment. If you see assessment logic in `_parse_turn()`, that's a bug.

### Backward Compatibility
V1 (Agent + Policy) and V2 (ConversationAgent) coexist. Don't break V1 to improve V2.

### Testing
- All changes must pass `uv run pytest`
- New features need tests
- Integration tests with real LLMs go in `examples/`
- Mock tests go in `tests/`

## Pull Request Process

1. Fork and create a feature branch
2. Write tests first
3. Implement
4. Run the 7-question self-check above
5. `uv run pytest && uv run ruff check .`
6. Submit PR with clear description

## Code Style

- Python 3.11+
- Pydantic v2 for all data models
- `from __future__ import annotations` in every file
- Type hints everywhere
- ruff for linting and formatting
