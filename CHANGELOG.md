# Changelog

All notable changes to Arcana will be documented in this file.

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
