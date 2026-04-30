# Contributing to Arcana

## The Arcana Constitution

Every contribution must align with [`CONSTITUTION.md`](CONSTITUTION.md): nine
principles, four prohibitions, and a clear division of responsibility between
the LLM, the runtime, and the user. Read it before your first PR.

The constitutional compliance checklist lives in
[`.github/pull_request_template.md`](.github/pull_request_template.md) — the
template is opened automatically when you create a PR. Fill it out honestly;
it is the source of truth, not this file.

For quick orientation, every change should answer "yes" to the same set of
questions:

- Does it help the LLM, or restrict it?
- Does it keep the context window honest (no hoarding, no surface bloat)?
- Does it leave strategy decisions to the LLM, with the runtime providing
  services rather than dictating steps?
- Does it provide actionable feedback on errors instead of mechanical retry?
- Does it judge outcomes, not process?

If a change fails any of these, it needs redesign — not a workaround.

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

## Stability & Versioning

From v1.0.0 onward, the names listed in
[`specs/v1.0.0-stability.md`](specs/v1.0.0-stability.md) §1 are the **stable
public surface**. Any rename, removal, or signature-breaking change there is
a major version bump — never a minor or patch. Internal modules listed in §2
carry no stability promise.

If your PR touches the stable surface:

- A break must first ship a `DeprecationWarning` for at least one minor
  release. The successor name and migration recipe must be live in the same
  release that adds the deprecation.
- Update [`specs/v1.0.0-removals.md`](specs/v1.0.0-removals.md) (or the
  successor `vX.0.0-removals.md`) so the next-major removal list stays
  current.
- The CHANGELOG entry must include a `Migration` section.

The user-facing summary lives at
[`docs/guide/stability.md`](docs/guide/stability.md). The constitutional
framing is [Chapter VI of `CONSTITUTION.md`](CONSTITUTION.md#chapter-vi-stability-commitments).
If you are not sure whether a name is stable, check `docs/guide/stability.md`
or ask in the PR.

## Pull Request Process

1. Fork and create a feature branch
2. Write tests first — including a constitutional invariant test in
   `tests/test_constitutional_invariants.py` if you're touching a load-bearing
   contract (tool dispatch, retry policy, ask_user, cognitive primitives,
   structured output, etc.)
3. Implement
4. `uv run pytest && uv run ruff check . && uv run mypy src/`
5. Submit PR — fill in the constitutional checklist in the auto-opened
   template

## Code Style

- Python 3.11+
- Pydantic v2 for all data models
- `from __future__ import annotations` in every file
- Type hints everywhere
- ruff for linting and formatting
