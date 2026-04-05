# Changelog

All notable changes to Arcana will be documented in this file.

## [0.3.1] - 2026-04-05

### Added — Provider Compatibility
- **`ProviderProfile`**: Unified capability system per provider. Tracks `tool_calls`, `json_schema`, `json_mode`, `streaming`, `stream_options`. Known providers get pre-configured profiles; custom providers get conservative defaults
- **Auto-degradation**: When a provider returns 400 for tool_calls, the profile is updated automatically — subsequent calls skip native tools and use prompt-based fallback. Only fails once per capability
- **Custom provider registration**: `providers={"siliconflow": {"api_key": "...", "base_url": "...", "model": "...", "tool_calls": False}}` — any OpenAI-compatible API with explicit capability overrides
- **`ChatSession.send(message, images=[...])`**: Multimodal messages in chat sessions
- **`runtime.create_chat_session()`**: Returns ChatSession directly without requiring `async with`, for use across HTTP requests
- **`arcana.RuntimeConfig`**: Now exported from the top-level package

### Stats
- All 1173 tests passing, 0 failures

## [0.3.0] - 2026-04-04 — "The Context Release"

### Added — Context Transparency

- **`ContextReport`**: Every LLM call now produces a detailed report of how the context window was composed. Shows token allocation across layers (identity, task, tools, history, memory), compression metrics, and window utilization. Available on `RunResult.context_report` and `ChatResponse.context_report`
- **`ContextStrategy`**: Adaptive compression strategy system replaces one-size-fits-all compression. Four tiers:
  - **passthrough** (< 50% utilization): No compression, zero overhead
  - **tail_preserve** (50-75%): Compress middle history, keep recent turns verbatim
  - **llm_summarize** (75-90%): Use cheap LLM call for semantic summarization
  - **aggressive_truncate** (> 90%): Keep only system + last 2 turns
  - Configurable via `Runtime(context_strategy=ContextStrategy(...))` or shorthand `"off"` / `"always_compress"`
- **Structured stream events**: `runtime.stream()` and `ChatSession.stream()` now emit:
  - `TOOL_START` — tool name and arguments before execution
  - `TOOL_END` — tool result and duration after execution
  - `TURN_END` — token count and cost at end of each turn
  - `CONTEXT_REPORT` — full context composition report per turn
- **`StreamEventType` exported**: `arcana.StreamEventType` for `match` statements on stream events

### Stats
- Tests: 1142 → 1173 (+31 new tests for context features)
- All 1173 tests passing, 0 failures

## [0.2.2] - 2026-04-04

### Fixed — Core Reliability
- **asyncio.Lock replaces threading.Lock**: `Runtime._totals_lock` was a `threading.Lock` blocking the event loop in async code. Now uses `asyncio.Lock` for proper async concurrency
- **Tool gateway idempotency race**: Fixed TOCTOU race where two concurrent calls with the same idempotency key could both execute. Lock now covers the entire check→execute→cache window
- **Budget boundary off-by-one**: `BudgetTracker.check_budget()` used `>=` (triggers at exact limit), now uses `>` (allows using exactly the allocated budget). Same fix applied to `BudgetScope`
- **Provider close() isolation**: `ModelGatewayRegistry.close()` now catches exceptions per-provider — one failing provider no longer blocks cleanup of others
- **MCP reconnect serialization**: Added `asyncio.Lock` to `MCPConnection._reconnect()` preventing concurrent reconnect attempts from corrupting transport state
- **MCP disconnect_all resilience**: Individual server disconnect failures no longer abort the cleanup loop
- **Graph checkpointer blocking I/O**: `GraphCheckpointer.save()/load()/delete()` were fake-async (blocking file I/O). Now uses `asyncio.to_thread()` + atomic write (temp file + rename) to prevent corruption on crash
- **Trace reader token/cost accounting**: `TraceReader.summarize()` used `max()` instead of `+=` for tokens/cost, reporting peak values instead of totals
- **SSE line terminator**: MCP Streamable HTTP transport now handles `\r\n` and `\r` per SSE spec, not just `\n`
- **Silent hook/callback failures**: Bare `except: pass` in agent hooks and `on_parse_error` callback now logs to `logger.debug` for debuggability

### Added
- **`Runtime` as async context manager**: `async with Runtime(...) as rt:` ensures `close()` is called, preventing HTTP connection leaks
- **`BudgetTracker.can_afford(estimated_tokens, estimated_cost)`**: Now checks cost budget in addition to token budget

### Removed — Dead Code Cleanup
- **`orchestrator/`**: Entire module (Orchestrator, TaskScheduler, TaskGraph, ExecutorPool) — never used by runtime
- **`gateway/router.py`**: ModelRouter — never imported
- **`gateway/capabilities.py`**: CapabilityRegistry — never queried
- **`streaming/sse.py`**: SSE formatter — never called
- **`runtime/replay.py`**: ReplayEngine — never wired up
- **`tool_gateway/adapters/langchain.py`**: LangChain bridge — never loaded
- **`storage/postgres.py`**, **`storage/chroma.py`**: Production storage backends removed. Arcana provides the `StorageBackend`/`VectorStore` interfaces; users implement for their infrastructure
- Removed `chromadb` dev dependency

### Stats
- Tests: 1234 → 1142 (removed 92 tests for deleted dead code)
- All 1142 tests passing, 0 failures
- mypy strict: 8 errors (all pre-existing)

## [0.2.1] - 2026-03-28

### Fixed — Production High Availability
- **Provider connection leak**: `Runtime.close()` now cascades to all provider HTTP clients (AsyncOpenAI, AsyncAnthropic). Previously only closed MCP connections, leaking connection pools in long-running apps
- **Budget race condition**: `Runtime._total_tokens_used` and `_total_cost_usd` now protected by `threading.Lock`. Concurrent `run()` calls no longer corrupt cumulative budget counters
- **timeout_ms actually wired**: `ModelConfig.timeout_ms` now passed to provider SDK `create()` calls as per-request timeout. Previously the config existed but was silently ignored (SDK defaulted to 600s)
- **Cancellation safety**: `asyncio.CancelledError` and `KeyboardInterrupt` in `Runtime.run()` and `ConversationAgent` now record partial budget and leave state consistent before re-raising

### Added — Developer Experience
- **`arcana init`**: CLI scaffold command generates `main.py` + `.env.example` + `agent.yaml` for 30-second quickstart
- **`Runtime.on()` / `Runtime.off()`**: Event hook API for runtime lifecycle events (`run_start`, `run_end`, `error`). Supports sync and async callbacks, chainable
- **`ChatSession(max_history=N)`**: Sliding window on message history to prevent OOM in long conversations. System messages always preserved. `runtime.chat(max_history=100)`
- **LangChain adapter test suite**: 18 tests covering spec extraction, execution, error handling, protocol compliance
- **SECURITY.md**: Honest security model documentation — what Arcana secures and what it doesn't
- **CI coverage reporting**: pytest-cov + Codecov upload in GitHub Actions
- **Dynamic README badges**: PyPI version, CI status, coverage — no more stale static badges

### Changed
- Example 13 rewritten to use `runtime.chat()` / `ChatSession.send()` instead of manual LLM message management

### Stats
- Tests: 1164 → 1234 (+70 new tests)
- All 1234 tests passing, 0 failures

## [0.2.0] - 2026-03-27

### Fixed — Structured Output Reliability
- **`result.parsed` always returns `BaseModel | None`**: Fixed bug where `parsed` could be a raw `dict` when provider degrades to `json_object` mode. Now handles dict inputs, validates `on_parse_error` callback returns, and guarantees type consistency
- **Anthropic structured output**: `AnthropicProvider` now supports `response_format` — injects JSON schema into system prompt (same fallback strategy as DeepSeek/Ollama/Kimi). Works with and without tools

### Added — Batch API & Budget Granularity
- **`Runtime.run_batch(tasks, concurrency=5)`**: Run multiple independent tasks concurrently with `asyncio.Semaphore`. Individual failures don't crash the batch. Returns `BatchResult` with results, succeeded/failed counts, total tokens/cost
- **Provider-level `batch_generate()`**: `OpenAICompatibleProvider.batch_generate(requests, config, concurrency=5)` for concurrent LLM calls. Registry-level fallback when provider doesn't implement batch
- **`ChainStep.budget`**: Per-step budget in `chain()` pipelines. Each step can have its own budget cap, always capped by chain-level remaining budget. Steps without budget share the chain pool

### Stats
- Tests: 1164, all passing

## [0.1.0-beta.8] - 2026-03-27

### Added — Team Dual Mode
- **`runtime.team(mode="shared"|"session")`**: Two collaboration modes. `"shared"` (default) — all agents share one conversation history. `"session"` — each agent has an independent context; other agents' messages arrive as user messages

### Stats
- Tests: 1135, all passing

## [0.1.0-beta.7] - 2026-03-27

### Fixed — Provider Compatibility
- **Cost estimation**: `TokenUsage.cost_estimate` now uses realistic mid-range pricing ($0.15/M input, $0.60/M output) instead of placeholder values
- **Zero-token warning**: When a provider reports 0 tokens, the runtime estimates from response length and logs a warning instead of silently tracking $0
- **Structured output + json_schema auto-downgrade**: Providers that don't support `json_schema` response format (DeepSeek, Ollama, Kimi, GLM, MiniMax) automatically fall back to `json_object` with schema instructions injected into system prompt
- **Provider model config**: `providers` dict now accepts `{"provider": {"api_key": "...", "model": "..."}}` to override default model per provider
- **Tool call logging**: Debug-level logs for all tool calls and results (name, arguments, output)

## [0.1.0-beta.6] - 2026-03-26

### Added — Pipeline & Budget Control
- **Parallel chain branches**: `runtime.chain()` now accepts nested lists for parallel execution — `[ChainStep, [ChainStep, ChainStep], ChainStep]` runs the inner list concurrently with `asyncio.gather`
- **Per-run provider/model selection**: `runtime.run(provider="openai", model="gpt-4o")` overrides default provider/model for a single run. Also available on `runtime.stream()` and `ChainStep`
- **Budget scoping**: `async with runtime.budget_scope(max_cost_usd=0.50) as scoped:` isolates budget for a subset of runs
- **`on_parse_error` callback**: `runtime.run(response_format=MyModel, on_parse_error=fix_fn)` — fires on `json.JSONDecodeError` or `pydantic.ValidationError`, NOT on provider-level format rejection
- **`result.parsed` field**: `RunResult.parsed` holds the validated Pydantic model (separate from `result.output` for backward compatibility)
- **`Tool` class**: Non-decorator tool registration — `Tool(fn=my_func, when_to_use="...")` for when `@arcana.tool` is not practical

### Changed
- `ChainStep` now supports `provider`, `model`, and `on_parse_error` fields
- Tools and structured output coexist — agent uses tools during reasoning and returns structured output on the final turn
- `BudgetScope` exported from `arcana` package

## [0.1.0-beta.5] - 2026-03-25

### Fixed
- 8 user-reported issues: SDK `system` and `context` parameters, fallback chain logging, budget tracking across runs, `runtime.fallback_order` property, provider `get_fallback_chain()` method, Tool wrapper support in registry

### Added
- **`arcana.run(system=..., context=...)`**: System prompt and context injection available at SDK level
- **`runtime.budget_remaining_usd`** / **`runtime.tokens_used`**: Runtime-level cumulative budget tracking properties
- **Auto fallback chain**: Multiple providers automatically form a fallback chain based on registration order

## [0.1.0-beta.4] - 2026-03-24

### Fixed
- 14 mypy strict errors regressed after beta.3
- `ChatSession.send()` now uses `generate()` instead of `stream()` for reliable usage tracking
- CI lint errors (unused imports, import sorting, bare except)

### Added
- Automated PyPI publish workflow (CI)
- Integration verification tests for b7 features

## [0.1.0-beta.3] - 2026-03-24

### Added — LLM Capability Amplification
- **Parallel Tool Execution**: Multiple tool calls in a single turn now run concurrently via `asyncio.gather`, with order-preserving results and individual failure isolation
- **Prompt Caching**: Anthropic system prompt + tool schemas automatically tagged with `cache_control`; OpenAI `cached_tokens` tracked. Up to 90% input token savings on multi-turn runs
- **Thinking-Informed Assessment**: `_assess_turn` now analyzes extended thinking blocks for uncertainty, verification intent, and incomplete information signals. Adjusts confidence and completion accordingly
- **Structured Output**: `arcana.run(response_format=MyModel)` returns validated Pydantic instances. Provider-level `json_schema` mode for OpenAI-compatible APIs
- **Multimodal Input**: `arcana.run(images=[...])` accepts URLs, file paths, and data URIs. OpenAI ↔ Anthropic content block format auto-conversion
- **LLM-Driven Context Compression**: `WorkingSetBuilder` can use a cheap LLM to produce semantic summaries instead of keyword-based truncation. Async `abuild_conversation_context()` with graceful fallback

### Added — Interactive Capabilities
- **`ask_user` Built-in Tool**: LLM can ask clarifying questions mid-execution. Intercepted at runtime level (bypasses ToolGateway). Sync/async `input_handler` callback. Graceful fallback when no handler provided
- **`runtime.chat()`**: Multi-turn conversational sessions with persistent history, shared budget, context compression, and streaming support. `ChatSession.send()` / `ChatSession.stream()`
- **CLI `arcana chat`**: Interactive terminal chat with Rich formatting, per-turn token/cost stats, budget enforcement
- **Examples 13-14**: Interactive chat and ask_user demonstrations

### Changed — Constitution v2
- **Principle 2** expanded: context is modality-agnostic (text, images, structured data)
- **Principle 4** corollary: thinking is signal, not contract — runtime may listen but never constrain
- **Principle 8** added: agent autonomy in collaboration — framework provides coordination, never hierarchy
- **Chapter IV** expanded: User role defined (intent, information, judgment). Two new inviolable rules: user never forced to interact; LLM asks but never blocks
- **Contributor Compact**: Questions 8-9 added (agent autonomy, user optionality)

### Added — Documentation
- `docs/guide/quickstart.md` — Installation → Deployment guide
- `docs/guide/configuration.md` — Full configuration reference (16 sections)
- `docs/guide/providers.md` — 8 provider setup guides with fallback chains
- `docs/guide/api.md` — Public API reference (881 lines)

### Stats
- Tests: 878 → 1045 (+167 new tests)
- All 1045 tests passing, 0 failures
- 9 new features, 4 documentation files

## [0.1.0-beta.1] - 2026-03-18

### Added
- **Runtime + Session**: Long-lived resource container, create once use everywhere
- **Runtime.team()**: Multi-agent collaboration (constitutional — Runtime provides comm, agents decide strategy)
- **Runtime.stream()**: Async generator for streaming
- **Runtime.graph()**: StateGraph factory
- **Memory v2**: Relevance-based retrieval (keyword + recency + importance + token budget)
- **MCP Client**: stdio transport, MCPToolProvider → ToolGateway bridge
- **CLI**: `arcana run/trace/providers/version`
- **ConversationAgent (V2)**: TurnFacts/TurnAssessment separation, 51% token savings
- **108 new tests**: All user-facing modules now covered (713 total)
- **Actionable error messages**: 8 files improved
- **Intent Router**: Default on in ConversationAgent
- **Diagnostic Recovery**: Structured diagnosis in V2

### Changed
- `arcana.run()` delegates to Runtime, accepts `api_key` param
- Default engine is V2 ConversationAgent
- Hardcoded model IDs removed — user explicit > provider default > error
- README → "Agent Runtime for Production"

### Fixed
- AdaptivePolicy execution closure
- Memory injection through direct_answer fast path
- Tool results use native OpenAI format
- single_tool argument generation

## [0.1.0-alpha.2] - 2026-03-18

### Changed
- `arcana.run()` now accepts `api_key` parameter — no .env file needed
- Default engine switched to ConversationAgent (V2)
- `max_steps` renamed to `max_turns` in `arcana.run()`
- `engine="conversation"` (default) or `engine="adaptive"` (V1)

### Fixed
- SDK no longer forces environment variables for API keys
- OpenAI and Anthropic providers now work in `arcana.run()`
- Clear error message when no API key provided

## [0.1.0-alpha.1] - 2026-03-18

### Added

#### V2 Execution Engine
- **ConversationAgent**: LLM-native execution model with TurnFacts/TurnAssessment separation
- **TurnFacts**: Raw provider output, zero interpretation
- **TurnAssessment**: Runtime completion/failure judgment, separate from facts
- 13-step turn contract with 7 invariants
- Streaming via `ConversationAgent.astream()`

#### V2 Architecture
- **CONSTITUTION.md**: 7 design principles, 4 prohibitions
- **Intent Router**: Rule-based + LLM + Hybrid classifiers, 4 execution paths
- **Adaptive Policy**: 6 strategy types (direct_answer, single_tool, sequential, parallel, plan_and_execute, pivot)
- **Lazy Tool Loading**: Keyword-based tool selection, affordance fields on ToolSpec
- **Diagnostic Recovery**: 7 error categories, structured feedback, RecoveryTracker
- **Multi-Model Routing**: 5 model roles, complexity-based selection
- **Working Set Builder**: 4-layer context management (identity/task/working/external)

#### Provider Infrastructure
- **BaseProvider Protocol**: Replaces ABC, maps to Rust trait
- **AnthropicProvider**: Native Claude support with extended thinking
- **Chinese Providers**: Kimi (Moonshot), GLM (Zhipu), MiniMax factory functions
- **Capability Registry**: 22 capabilities across 8 providers
- **Error Hierarchy**: RateLimitError, AuthenticationError, ModelNotFoundError, ContentFilterError, ContextLengthError
- **StreamChunk**: Unified streaming data model

#### SDK
- `arcana.run()`: Zero-config entry point
- `@arcana.tool`: Decorator with affordance fields (when_to_use, what_to_expect, failure_meaning)
- `RunResult`: Structured result with output, steps, tokens, cost
- Budget tracking wired into SDK

#### Evaluation
- EvalMetrics: first_attempt_success, goal_achievement_rate, cost_per_success
- RuleJudge + LLMJudge + HybridJudge
- EvalRunnerV2 with suite reporting

#### Code Quality
- 605 tests, 0 failures
- Verified with real APIs: DeepSeek, OpenAI (gpt-4o-mini), Anthropic (claude-sonnet-4)
- AgentState immutable pattern
- ToolProvider ABC → Protocol
- asyncio.Lock replaces threading.Lock

### V1 (Preserved)
- Agent + AdaptivePolicy + StepExecutor + Reducer pipeline
- ReAct and PlanExecute policies
- Graph engine (StateGraph, CompiledGraph)
- Multi-agent orchestration (TeamOrchestrator)
- JSONL Trace system
- Budget tracking
- Checkpoint/Resume
