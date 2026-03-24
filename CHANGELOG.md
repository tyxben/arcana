# Changelog

All notable changes to Arcana will be documented in this file.

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
