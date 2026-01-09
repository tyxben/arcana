# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install all dependencies (including dev)
uv sync --all-extras

# Run tests
uv run pytest

# Run single test file
uv run pytest tests/test_trace.py

# Run tests with coverage
uv run pytest --cov=arcana

# Lint
uv run ruff check .

# Type check (strict mode)
uv run mypy src/

# Run demo
uv run python examples/demo_trace.py
```

## Architecture Overview

Arcana is an Agent Platform following a "contracts-first" design. All modules define Pydantic schemas before implementation, enabling future language migrations (Go/Rust) without changing upper-layer logic.

### Layer Structure

```
┌─────────────────────────────────────────────────┐
│              Application (Agents)                │
├─────────────────────────────────────────────────┤
│  Model Gateway │ Tool Gateway │ Memory │ Orch   │  ← Platform Services
├─────────────────────────────────────────────────┤
│           Contracts (Pydantic Schemas)           │  ← Data Models
├─────────────────────────────────────────────────┤
│              Trace (JSONL Audit Log)             │  ← Observability
└─────────────────────────────────────────────────┘
```

### Core Modules

**contracts/** - All data models (trace, tool, state, llm):
- `TraceEvent`: Audit log entry with run_id, step_id, event_type, digests
- `ToolSpec`/`ToolCall`/`ToolResult`: Tool execution contracts
- `AgentState`: Execution state with budget tracking
- `LLMRequest`/`LLMResponse`: Unified LLM interface

**trace/** - JSONL-based event logging:
- `TraceWriter`: Writes events to `{run_id}.jsonl` files
- `TraceReader`: Query, filter, summarize trace events

**gateway/** - Model Gateway with provider abstraction:
- `ModelGateway`: Abstract base for LLM providers
- `ModelGatewayRegistry`: Multi-provider routing with fallback chains
- `OpenAICompatibleProvider`: Universal adapter for OpenAI-format APIs (DeepSeek, Gemini, Ollama)
- `BudgetTracker`: Token/cost/time limit enforcement

### Key Design Patterns

1. **Canonical Hashing**: All digests use SHA-256 on sorted JSON, truncated to 16 chars. See `utils/hashing.py`.

2. **Provider Abstraction**: Single `OpenAICompatibleProvider` handles any OpenAI-format API by changing base_url.

3. **Trace Everything**: Every LLM call auto-logs to trace with request/response digests.

## Configuration

Copy `.env.example` to `.env` and set API keys:
- `DEEPSEEK_API_KEY`
- `GEMINI_API_KEY`

## Project Roadmap Reference

See `../step.md` for the 14-week implementation plan. Current status: Week 1-2 complete (Contracts, Trace, Model Gateway).

Next milestones:
- Week 3-4: Agent Runtime (state machine execution)
- Week 5: Tool Gateway (authz, idempotency, audit)
- Week 6: RAG (indexing, retrieval, citations)

## Learning Resources

See `docs/KNOWLEDGE.md` for comprehensive knowledge points including:
- Core concepts and design principles
- Module deep-dives with interview-ready explanations
- High-frequency interview questions with answer frameworks
- Extended learning resources
