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

# Run integration tests (requires API keys in .env)
uv run pytest tests/integration/ -v

# Run demo
uv run python examples/demo_trace.py
```

## Architecture Overview

Arcana is an Agent Platform following a "contracts-first" design. All modules define Pydantic schemas before implementation, enabling future language migrations (Go/Rust) without changing upper-layer logic.

### V2 Architecture (current)

The V2 engine centers on `ConversationAgent` — no Policy, no Reducer, just LLM turns. Each turn produces `TurnFacts` (raw LLM output) and `TurnAssessment` (runtime interpretation), kept visibly separate.

```
Request → Intent Router (routing/)
           → Direct Answer (1 LLM call)
           → ConversationAgent (runtime/conversation.py)
                LLM Turn → TurnFacts → TurnAssessment → State
                Runtime OS: Budget | Trace | Tools | Diagnostics

V1 path (still compatible):
           → Agent + AdaptivePolicy (runtime/agent.py)
```

### Layer Structure

```
┌─────────────────────────────────────────────────┐
│              Application (Agents)                │
│   runtime/conversation.py  (V2 ConversationAgent)│
│   runtime/agent.py         (V1 Agent)            │
├─────────────────────────────────────────────────┤
│  Routing │ Gateway │ Tool Gateway │ Context      │  ← Platform Services
│  Eval    │ Memory  │ Streaming    │ Diagnosis    │
├─────────────────────────────────────────────────┤
│           Contracts (Pydantic Schemas)           │  ← Data Models
│   turn, routing, context, diagnosis, llm, tool   │
├─────────────────────────────────────────────────┤
│              Trace (JSONL Audit Log)             │  ← Observability
└─────────────────────────────────────────────────┘
```

### Core Modules

**contracts/** - All data models:
- `turn.py`: `TurnFacts`, `TurnAssessment` — the V2 separation principle
- `routing.py`: `RoutingDecision`, `IntentCategory` — intent classification
- `context.py`: `ContextBlock`, `ContextBudget` — working set context
- `diagnosis.py`: `DiagnosticBrief`, `ErrorCategory` — structured error recovery
- `llm.py`: `LLMRequest`, `LLMResponse`, `ModelConfig` — unified LLM interface
- `tool.py`: `ToolSpec`, `ToolCall`, `ToolResult` — tool execution contracts
- `state.py`: `AgentState` — execution state with budget tracking
- `runtime.py`: Runtime configuration and turn state
- `eval.py`: Evaluation framework contracts
- `streaming.py`: Streaming response contracts
- `multi_agent.py`: Multi-agent coordination contracts
- `plan.py`, `strategy.py`, `intent.py`, `graph.py`, `orchestrator.py`, `memory.py`, `rag.py`

**runtime/** - Agent execution engines:
- `conversation.py`: `ConversationAgent` — the V2 engine (recommended)
- `agent.py`: V1 agent with AdaptivePolicy
- `state_manager.py`: State lifecycle management
- `step.py`: Single-step execution
- `error_handler.py`: Runtime error handling
- `validator.py`: Input/output validation
- `factory.py`: Agent factory
- `replay.py`: Trace replay for debugging
- `progress.py`: Progress tracking

**routing/** - Intent classification and routing:
- `classifier.py`: Rule-based + LLM fallback intent classifier
- `executor.py`: Route execution (direct answer vs agent loop)

**context/** - Working set context management:
- `builder.py`: Context block assembly and budget enforcement

**gateway/** - Model Gateway with provider abstraction:
- `ModelGatewayRegistry`: Multi-provider routing with fallback chains
- `OpenAICompatibleProvider`: Universal adapter for OpenAI-format APIs
- Provider factories: DeepSeek, OpenAI, Anthropic, Kimi, GLM, MiniMax, Gemini, Ollama

**tool_gateway/** - Tool execution with auth, validation, audit

**eval/** - Evaluation framework for agent quality

**trace/** - JSONL-based event logging

**multi_agent/** - Multi-agent coordination

**streaming/** - Streaming response support

### Key Design Patterns

1. **TurnFacts / TurnAssessment Separation**: The LLM produces facts (what it said, what tools it called). The runtime produces assessment (should we continue, is budget exceeded, did a tool fail). Never mix the two — the LLM does not judge its own output; the runtime does not generate content.

2. **Canonical Hashing**: All digests use SHA-256 on sorted JSON, truncated to 16 chars. See `utils/hashing.py`.

3. **Provider Abstraction**: Single `OpenAICompatibleProvider` handles any OpenAI-format API by changing base_url. Adding a new provider is one function call.

4. **Trace Everything**: Every LLM call auto-logs to trace with request/response digests.

5. **Contracts First**: All data flows through Pydantic models defined in `contracts/`. Implementation never invents ad-hoc dicts.

## Configuration

Copy `.env.example` to `.env` and set API keys:
- `DEEPSEEK_API_KEY` (primary, verified)
- `OPENAI_API_KEY` (verified)
- `ANTHROPIC_API_KEY` (verified)
- `GEMINI_API_KEY`, `KIMI_API_KEY`, `GLM_API_KEY`, `MINIMAX_API_KEY` (supported)

## Verified Providers

These providers have been tested with real API keys:
- DeepSeek (deepseek-chat) — direct answer + tools + multi-step
- OpenAI (gpt-4o-mini) — direct answer + tools
- Anthropic (claude-sonnet-4) — direct answer

## Project Status

Current: v0.1.0-alpha.1 — V2 ConversationAgent engine working with real APIs.

## Learning Resources

See `docs/architecture.md` for the full system architecture.
Legacy v1 learning docs are archived in `docs/legacy/`.
