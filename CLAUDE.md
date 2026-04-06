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

# Run integration tests (requires API keys via env vars or .env)
uv run pytest tests/integration/ -v

# Run demo
uv run python examples/demo_trace.py

# Build docs
uv sync --extra docs && uv run mkdocs build

# Serve docs locally
uv run mkdocs serve
```

## Architecture Overview

Arcana is an Agent Platform following a "contracts-first" design. All modules define Pydantic schemas before implementation, enabling future language migrations (Go/Rust) without changing upper-layer logic.

### V2 Architecture (current)

The V2 engine centers on `ConversationAgent` -- no Policy, no Reducer, just LLM turns. Each turn produces `TurnFacts` (raw LLM output) and `TurnAssessment` (runtime interpretation, including thinking-informed confidence adjustment), kept visibly separate.

Key capabilities beyond basic turn loop:
- **Parallel tool execution**: multiple tool calls in one turn run concurrently via `asyncio.gather` with order-preserving results
- **`ask_user` built-in tool**: LLM can ask clarifying questions mid-execution; intercepted at runtime level (bypasses ToolGateway); graceful fallback when no handler provided
- **Prompt caching**: transparent, provider-level -- Anthropic `cache_control` tags on system/tools, OpenAI `cached_tokens` tracked
- **Thinking-informed assessment**: `_assess_turn` analyzes extended thinking blocks for uncertainty/verification/incomplete signals, adjusts confidence
- **Structured output**: `response_format` passes a Pydantic model's JSON Schema to the provider; tools remain available and coexist with structured output
- **Multimodal input**: `images` parameter accepts URLs, file paths, data URIs; auto-converts between OpenAI and Anthropic content block formats
- **Fidelity-graded context compression**: `WorkingSetBuilder.abuild_conversation_context()` compresses history using 4 fidelity levels (L0 original → L3 dropped) based on relevance scoring; falls back to LLM summarization or aggressive truncation
- **Multi-turn chat**: `runtime.chat()` returns a `ChatSession` that delegates to `ConversationAgent` internally, gaining all V2 features (ask_user, lazy tools, diagnostics, fidelity compression, thinking assessment)
- **Sequential pipeline**: `runtime.chain()` runs a list of `ChainStep`s sequentially, automatically passing each step's output as context to the next
- **Context passing**: `runtime.run(context=...)` injects additional context (dict or string) into the agent's goal as a `<context>` block

```
Request -> Intent Router (routing/)
            -> Direct Answer (1 LLM call)
            -> ConversationAgent (runtime/conversation.py)
                 LLM Turn -> TurnFacts -> TurnAssessment -> State
                 Runtime OS: Budget | Trace | Tools | Diagnostics | ask_user

Multi-turn: runtime.chat() -> ChatSession -> send() / stream()
Team:       runtime.team(mode="shared"|"session") -> shared conversation or independent sessions
Pipeline:   runtime.chain([ChainStep, ...]) -> sequential run() with auto context

V1 path (still compatible):
            -> Agent + AdaptivePolicy (runtime/agent.py)
```

### Layer Structure

```
+-------------------------------------------------+
|              Application (Agents)                |
|   runtime/conversation.py  (V2 ConversationAgent)|
|   runtime/agent.py         (V1 Agent)            |
|-------------------------------------------------|
|  Routing | Gateway | Tool Gateway | Context      |  <- Platform Services
|  Eval    | Memory  | Streaming    | Diagnosis    |
|-------------------------------------------------|
|           Contracts (Pydantic Schemas)           |  <- Data Models
|   turn, routing, context, diagnosis, llm, tool   |
|-------------------------------------------------|
|              Trace (JSONL Audit Log)             |  <- Observability
+-------------------------------------------------+
```

### Core Modules

**contracts/** - All data models:
- `turn.py`: `TurnFacts`, `TurnAssessment` -- the V2 separation principle
- `routing.py`: `RoutingDecision`, `IntentCategory` -- intent classification
- `context.py`: `ContextBlock`, `ContextBudget`, `ContextDecision` -- working set context
- `diagnosis.py`: `DiagnosticBrief`, `ErrorCategory` -- structured error recovery
- `llm.py`: `LLMRequest`, `LLMResponse`, `ModelConfig`, `ContentBlock` -- unified LLM interface
- `tool.py`: `ToolSpec`, `ToolCall`, `ToolResult`, `ASK_USER_TOOL_NAME` -- tool execution contracts
- `state.py`: `AgentState` -- execution state with budget tracking
- `runtime.py`: Runtime configuration and turn state
- `eval.py`: Evaluation framework contracts
- `streaming.py`: Streaming response contracts (`StreamEvent`, `StreamEventType`)
- `multi_agent.py`: Multi-agent coordination contracts
- `plan.py`, `strategy.py`, `intent.py`, `graph.py`, `orchestrator.py`, `memory.py`, `rag.py`

**runtime/** - Agent execution engines:
- `conversation.py`: `ConversationAgent` -- V2 engine with parallel tools, thinking assessment, ask_user interception
- `ask_user.py`: `AskUserHandler`, `ASK_USER_SPEC` -- built-in tool for LLM-to-user clarification
- `agent.py`: V1 agent with AdaptivePolicy
- `state_manager.py`: State lifecycle management
- `step.py`: Single-step execution
- `error_handler.py`: Runtime error handling
- `validator.py`: Input/output validation
- `factory.py`: Agent factory
- `replay.py`: Trace replay for debugging
- `progress.py`: Progress tracking

**runtime_core.py** - `Runtime`, `Session`, `ChatSession`, `ChatResponse`, `Budget`, `AgentConfig`, `ChainStep`, `ChainResult`, `BatchResult`:
- `Runtime`: Long-lived resource container (providers, tools, budget, trace)
- `Runtime.run()`: Single-shot task execution
- `Runtime.chat()`: Multi-turn conversational sessions (`ChatSession.send()` / `ChatSession.stream()`)
- `Runtime.chain()`: Sequential pipeline with automatic context passing between steps
- `Runtime.run_batch()`: Concurrent task execution, returns list of `BatchResult`
- `Runtime.session()`: Manual session control for advanced usage
- `ChainStep.budget`: Optional per-step budget override (scopes cost/turn limits to individual steps)

**routing/** - Intent classification and routing:
- `classifier.py`: Rule-based + LLM fallback intent classifier
- `executor.py`: Route execution (direct answer vs agent loop)

**context/** - Working set context management:
- `builder.py`: `WorkingSetBuilder` -- context block assembly, budget enforcement, fidelity-graded compression (L0-L3 spectrum) (`abuild_conversation_context`)

**gateway/** - Model Gateway with provider abstraction:
- `ModelGatewayRegistry`: Multi-provider routing with fallback chains
- `OpenAICompatibleProvider`: Universal adapter for OpenAI-format APIs, with prompt caching support
- Provider factories: DeepSeek, OpenAI, Anthropic, Kimi, GLM, MiniMax, Gemini, Ollama
- `batch_generate()`: Available on both individual providers and `ModelGatewayRegistry` -- concurrent LLM calls via `asyncio.Semaphore`

**tool_gateway/** - Tool execution with auth, validation, audit:
- `gateway.py`: `ToolGateway` with `call_many_concurrent()` for parallel execution via `asyncio.gather`

**eval/** - Evaluation framework for agent quality

**trace/** - JSONL-based event logging

**multi_agent/** - Multi-agent coordination

**streaming/** - Streaming response support

### Key Design Patterns

1. **TurnFacts / TurnAssessment Separation**: The LLM produces facts (what it said, what tools it called). The runtime produces assessment (should we continue, is budget exceeded, did a tool fail). Never mix the two -- the LLM does not judge its own output; the runtime does not generate content.

2. **Canonical Hashing**: All digests use SHA-256 on sorted JSON, truncated to 16 chars. See `utils/hashing.py`.

3. **Provider Abstraction**: Single `OpenAICompatibleProvider` handles any OpenAI-format API by changing base_url. Adding a new provider is one function call.

4. **Trace Everything**: Every LLM call auto-logs to trace with request/response digests.

5. **Contracts First**: All data flows through Pydantic models defined in `contracts/`. Implementation never invents ad-hoc dicts.

6. **Ask User as Capability**: `ask_user` is a tool the LLM can call, not a forced interaction step. Intercepted at runtime level (bypasses ToolGateway). When no `input_handler` is provided, the LLM gets a fallback message and proceeds with best judgment. The user is never forced to interact.

7. **Thinking as Signal**: The runtime listens to extended thinking blocks (Anthropic, Gemini) for uncertainty, verification intent, and incomplete information signals. It adjusts confidence and completion decisions accordingly. It never constrains what the LLM thinks.

8. **Prompt Caching**: Transparent, provider-level optimization. Anthropic system prompts and tool schemas get `cache_control` tags automatically. OpenAI `cached_tokens` are tracked in usage. No application-level changes needed.

## Configuration

Pass `api_key` directly to `arcana.run()` -- no `.env` file required:

```python
result = await arcana.run("Hello", api_key="sk-xxx")
```

For multi-turn sessions, use `runtime.chat()`:

```python
runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    tools=[my_tool],
    budget=arcana.Budget(max_cost_usd=5.0),
)
async with runtime.chat() as c:
    r = await c.send("Hello")
    r = await c.send("Tell me more")
    print(c.total_cost_usd)
```

Environment variables are still supported as a fallback:
- `DEEPSEEK_API_KEY` (primary, verified)
- `OPENAI_API_KEY` (verified)
- `ANTHROPIC_API_KEY` (verified)
- `GEMINI_API_KEY`, `KIMI_API_KEY`, `GLM_API_KEY`, `MINIMAX_API_KEY` (supported)

## Verified Providers

These providers have been tested with real API keys:
- DeepSeek (deepseek-chat) -- direct answer + tools + structured output
- OpenAI (gpt-4o-mini) -- direct answer + tools + structured output
- Anthropic (claude-sonnet-4) -- direct answer + structured output
- Kimi/Moonshot (moonshot-v1-8k) -- direct answer + tools + structured output
- GLM/Zhipu (glm-4-flash) -- direct answer + tools + structured output
- MiniMax (abab6.5s-chat) -- direct answer + tools (auto-degraded) + structured output

## SDK Interface (`sdk.py`)

Key parameters for `arcana.run()`:
- `api_key`: API key string (preferred over env vars)
- `provider`: Provider name (`"deepseek"`, `"openai"`, `"anthropic"`, etc.)
- `engine`: `"conversation"` (default, V2 ConversationAgent) or `"adaptive"` (V1)
- `max_turns`: Maximum number of agent turns (default: 20)
- `tools`: List of tools decorated with `@arcana.tool`
- `response_format`: Pydantic `BaseModel` class for structured output (tools remain available)
- `images`: List of image URLs, file paths, or data URIs for multimodal input
- `input_handler`: Sync or async callback for the `ask_user` built-in tool (None = graceful fallback)
- `system`: System prompt for the run (overrides `RuntimeConfig.system_prompt`)
- `context`: Additional context (dict or string) injected into the goal
- `on_parse_error`: Callback `(raw_string, error) -> BaseModel | None` for structured output parse failures. **Scope**: fires only on `json.JSONDecodeError` or `pydantic.ValidationError` (LLM returned text that doesn't match the schema). Does NOT fire on provider-level format rejection (`ProviderError`) -- those are handled by provider capability detection / auto-downgrade.

Runtime methods also include:
- `runtime.run_batch(tasks, ...)`: Execute multiple tasks concurrently. Returns list of `BatchResult` (each wraps success/failure per task).
- `BatchResult` is exported from the SDK (`arcana.BatchResult`).

## Constitution

`CONSTITUTION.md` -- v2, 8 principles (was 7). Defines the four prohibitions, the division of responsibility (LLM vs runtime vs user), and the contributor compact. Key additions in v2: Principle 8 (agent autonomy in collaboration), expanded Chapter IV (user role, user optionality rules).

## Project Status

Current: v0.3.3 -- 1202 tests passing. Features: parallel tools, prompt caching, thinking assessment, structured output (coexists with tools, on_parse_error callback, parsed always BaseModel|None), multimodal input, fidelity-graded context compression (L0-L3 spectrum), ask_user, multi-turn chat (ChatSession delegates to ConversationAgent), pipeline with parallel branches (chain), context passing, per-run provider/model selection, budget scoping (chain-level + step-level), batch API (run_batch + provider batch_generate), Anthropic structured output, system prompt on run(), Runtime event hooks (on/off), arcana init CLI scaffold, ChatSession max_history, provider connection lifecycle, cancellation safety, ProviderProfile with auto-degradation, custom provider registration, StreamAccumulator, LazyToolRegistry token caching.

## Learning Resources

See `docs/architecture.md` for the full system architecture.
See `docs/guide/` for quickstart, configuration, providers, and API reference.
Legacy v1 learning docs are archived in `docs/legacy/`.
