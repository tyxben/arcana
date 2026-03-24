# Configuration Reference

This document is the complete reference for every configuration option in Arcana. All field names, types, and defaults are extracted directly from the source code.

---

## Environment Variables

Arcana reads environment variables as a fallback when API keys are not passed directly. The `load_config()` utility in `arcana.utils.config` reads these from the process environment or a `.env` file.

> **Note:** The environment variables in this section (prefixed with `DEFAULT_`) are only read by the file-based configuration system (`load_config()`). They are NOT automatically consumed by `arcana.run()` or `Runtime()`. The `arcana.run()` function has its own defaults (e.g., `provider="deepseek"`). API key variables (`*_API_KEY`) ARE automatically read by `Runtime` as a fallback when no explicit key is provided.

### Provider API Keys

| Variable | Provider | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek | API key for DeepSeek |
| `OPENAI_API_KEY` | OpenAI | API key for OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic | API key for Anthropic Claude |
| `GEMINI_API_KEY` | Gemini | API key for Google Gemini |
| `KIMI_API_KEY` | Kimi (Moonshot) | API key for Kimi |
| `GLM_API_KEY` | GLM (Zhipu AI) | API key for GLM |
| `MINIMAX_API_KEY` | MiniMax | API key for MiniMax |

When you pass `providers={"deepseek": ""}` to `Runtime()` with an empty string, the runtime resolves the key from the environment variable `DEEPSEEK_API_KEY` (pattern: `{PROVIDER_NAME}_API_KEY`).

### Provider Base URLs

| Variable | Default | Description |
|---|---|---|
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | Gemini API endpoint |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API endpoint |

### Model Defaults

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_PROVIDER` | `"gemini"` | Default LLM provider name |
| `DEFAULT_MODEL` | `"gemini-2.0-flash"` | Default model ID |
| `DEFAULT_TEMPERATURE` | `0.0` | Default sampling temperature |
| `DEFAULT_MAX_TOKENS` | `4096` | Default max output tokens |
| `DEFAULT_TIMEOUT_MS` | `30000` | Default request timeout in milliseconds |

### Budget Defaults

| Variable | Default | Description |
|---|---|---|
| `MAX_TOKENS_PER_RUN` | `100000` | Maximum tokens per run |
| `MAX_COST_PER_RUN_USD` | `1.0` | Maximum cost per run in USD |

### Trace Settings

| Variable | Default | Description |
|---|---|---|
| `TRACE_ENABLED` | `true` | Enable or disable trace output |
| `TRACE_DIR` | `"./traces"` | Directory for trace JSONL files |

---

## arcana.run() Parameters

The `arcana.run()` function is the simplest entry point. It creates a temporary `Runtime` internally, runs the task, and returns a result.

```python
result = await arcana.run(
    "Research quantum computing trends",
    provider="deepseek",
    api_key="sk-xxx",
    tools=[my_search],
    max_turns=10,
    max_cost_usd=2.0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `goal` | `str` | *(required)* | What you want the agent to accomplish |
| `tools` | `list[Callable] \| None` | `None` | List of `@arcana.tool` decorated functions |
| `provider` | `str` | `"deepseek"` | LLM provider name |
| `model` | `str \| None` | `None` | Model ID. Auto-selected from provider default if `None` |
| `api_key` | `str \| None` | `None` | API key for the provider. Falls back to environment variable if `None` |
| `max_turns` | `int` | `20` | Maximum execution turns |
| `max_cost_usd` | `float` | `1.0` | Maximum cost in USD |
| `auto_route` | `bool` | `True` | Enable intent routing (reserved) |
| `engine` | `str` | `"conversation"` | Execution engine: `"conversation"` (V2) or `"adaptive"` (V1) |
| `stream` | `bool` | `False` | Enable streaming output (reserved for future use) |

### RunResult

The return value of `arcana.run()`.

| Field | Type | Default | Description |
|---|---|---|---|
| `output` | `Any` | `None` | The agent's final output |
| `success` | `bool` | `False` | Whether the task completed successfully |
| `steps` | `int` | `0` | Number of execution steps taken |
| `tokens_used` | `int` | `0` | Total tokens consumed |
| `cost_usd` | `float` | `0.0` | Total cost in USD |
| `run_id` | `str` | `""` | Unique run identifier |

---

## Runtime Configuration

The `Runtime` class is the long-lived entry point for production use. Create it once at application startup.

```python
runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx", "openai": "sk-proj-xxx"},
    tools=[my_search, my_calculator],
    budget=arcana.Budget(max_cost_usd=10.0),
    trace=True,
    memory=True,
    namespace="tenant-42",
)
```

### Runtime() Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `providers` | `dict[str, str] \| None` | `None` | Map of provider name to API key (e.g., `{"deepseek": "sk-xxx"}`) |
| `tools` | `list[Callable] \| None` | `None` | List of `@arcana.tool` decorated functions |
| `mcp_servers` | `list[MCPServerConfig] \| None` | `None` | MCP server configurations for external tool sources |
| `budget` | `Budget \| None` | `None` | Default budget policy. Uses `Budget()` defaults if `None` |
| `trace` | `bool` | `False` | Enable JSONL trace output |
| `memory` | `bool` | `False` | Enable cross-run memory |
| `memory_budget_tokens` | `int` | `800` | Token budget for memory context injection |
| `config` | `RuntimeConfig \| None` | `None` | Advanced runtime configuration. Uses `RuntimeConfig()` defaults if `None` |
| `namespace` | `str \| None` | `None` | Namespace for tenant isolation. Memory and trace are partitioned per namespace |

### Supported Providers

| Provider Name | Default Model | Base URL | Notes |
|---|---|---|---|
| `"deepseek"` | `deepseek-chat` | `https://api.deepseek.com` | Primary verified provider |
| `"openai"` | `gpt-4o-mini` | `https://api.openai.com/v1` | Verified |
| `"anthropic"` | *(must specify model)* | Anthropic Messages API | Verified. Uses native SDK, not OpenAI-compatible |
| `"gemini"` | `gemini-2.0-flash` | `https://generativelanguage.googleapis.com/v1beta/openai` | Via OpenAI-compatible endpoint |
| `"kimi"` | `moonshot-v1-8k` | `https://api.moonshot.cn/v1` | Moonshot AI |
| `"glm"` | `glm-4-flash` | `https://open.bigmodel.cn/api/paas/v4` | Zhipu AI |
| `"minimax"` | `abab6.5s-chat` | `https://api.minimax.chat/v1` | |
| `"ollama"` | `llama3.2` | `http://localhost:11434/v1` | Local. No API key required |

### runtime.run()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `goal` | `str` | *(required)* | What to accomplish |
| `engine` | `str` | `"conversation"` | `"conversation"` (V2) or `"adaptive"` (V1) |
| `max_turns` | `int \| None` | `None` | Override default max turns from `RuntimeConfig` |
| `budget` | `Budget \| None` | `None` | Override default budget for this run |
| `tools` | `list[Callable] \| None` | `None` | Additional tools for this run only |

### runtime.stream()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `goal` | `str` | *(required)* | What to accomplish |
| `engine` | `str` | `"conversation"` | Only `"conversation"` is supported for streaming |
| `max_turns` | `int \| None` | `None` | Override default max turns |

### runtime.team()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `goal` | `str` | *(required)* | The shared objective |
| `agents` | `list[AgentConfig]` | *(required)* | List of agent configurations |
| `max_rounds` | `int` | `3` | Maximum conversation rounds (each agent speaks once per round) |
| `budget` | `Budget \| None` | `None` | Budget for the entire team run |

### runtime.session()

Returns an async context manager yielding a `Session`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `engine` | `str` | `"conversation"` | Execution engine |
| `max_turns` | `int \| None` | `None` | Override default max turns |
| `budget` | `Budget \| None` | `None` | Override default budget |
| `tools` | `list[Callable] \| None` | `None` | Additional tools for this session |

---

## RuntimeConfig

Advanced runtime-level settings. Passed via `config=` to `Runtime()`.

```python
from arcana.runtime_core import RuntimeConfig

config = RuntimeConfig(
    default_provider="openai",
    default_model="gpt-4o",
    max_turns=50,
    trace_dir="./my-traces",
    system_prompt="You are a helpful research assistant.",
)
runtime = arcana.Runtime(providers={...}, config=config)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `default_provider` | `str` | `"deepseek"` | Default LLM provider name |
| `default_model` | `str \| None` | `None` | Default model ID. If `None`, uses the provider's built-in default |
| `max_turns` | `int` | `20` | Default maximum turns per run |
| `trace_dir` | `str` | `"./traces"` | Directory for trace output files |
| `system_prompt` | `str \| None` | `None` | Default system prompt for agents |

---

## Budget

Budget constraints for a `Runtime` or individual run. Defined in `arcana.runtime_core`.

```python
budget = arcana.Budget(max_cost_usd=5.0, max_tokens=200_000)
runtime = arcana.Runtime(providers={...}, budget=budget)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `max_cost_usd` | `float` | `10.0` | Maximum cost in USD |
| `max_tokens` | `int` | `500_000` | Maximum total tokens |

There is also a lower-level `Budget` in `arcana.contracts.llm` used for individual LLM requests:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int \| None` | `None` | Maximum tokens for this request |
| `max_cost_usd` | `float \| None` | `None` | Maximum cost for this request |
| `max_time_ms` | `int \| None` | `None` | Maximum time in milliseconds |

---

## AgentConfig

Configuration for a single agent in a `runtime.team()` call.

```python
agents = [
    arcana.AgentConfig(
        name="researcher",
        prompt="You are an expert researcher. Find and cite sources.",
        model="deepseek-chat",
        provider="deepseek",
    ),
    arcana.AgentConfig(
        name="critic",
        prompt="You are a critical reviewer. Check claims for accuracy.",
    ),
]
result = await runtime.team("Analyze quantum computing trends", agents=agents)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Agent name (visible in team conversation) |
| `prompt` | `str` | *(required)* | System prompt defining this agent's role and personality |
| `model` | `str \| None` | `None` | Override model for this agent |
| `provider` | `str \| None` | `None` | Override provider for this agent |

---

## ModelConfig

LLM model configuration. Defined in `arcana.contracts.llm`.

```python
from arcana.contracts.llm import ModelConfig

config = ModelConfig(
    provider="openai",
    model_id="gpt-4o",
    temperature=0.7,
    max_tokens=8192,
    timeout_ms=60000,
    extra_params={"top_p": 0.9},
)
```

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `provider` | `str` | *(required)* | | Provider name (validated at registry level) |
| `model_id` | `str` | *(required)* | | Model identifier (e.g., `"deepseek-chat"`, `"gpt-4o"`) |
| `temperature` | `float` | `0.0` | `>= 0.0`, `<= 2.0` | Sampling temperature |
| `seed` | `int \| None` | `None` | | Random seed for reproducibility |
| `max_tokens` | `int` | `4096` | `> 0` | Maximum output tokens |
| `timeout_ms` | `int` | `30000` | `> 0` | Request timeout in milliseconds |
| `extra_params` | `dict[str, Any]` | `{}` | | Provider-specific parameters passed through to the API |

---

## Provider-Specific Request Extensions

These are optional fields on `LLMRequest` that enable provider-specific features. They are only relevant when building custom agents with direct LLM calls.

### AnthropicRequestExt

| Field | Type | Default | Description |
|---|---|---|---|
| `system` | `str \| None` | `None` | System prompt (Anthropic uses a dedicated field) |
| `thinking` | `ThinkingConfig \| None` | `None` | Extended thinking configuration |
| `prompt_caching` | `bool \| None` | `None` | Enable prompt caching |

### GeminiRequestExt

| Field | Type | Default | Description |
|---|---|---|---|
| `grounding` | `GroundingConfig \| None` | `None` | Google Search grounding configuration |
| `code_execution` | `bool \| None` | `None` | Enable server-side code execution |
| `safety_settings` | `list[SafetySetting] \| None` | `None` | Safety filter thresholds |
| `thinking` | `ThinkingConfig \| None` | `None` | Extended thinking configuration |
| `cached_content` | `str \| None` | `None` | Cached content resource name |

### OpenAIRequestExt

| Field | Type | Default | Description |
|---|---|---|---|
| `json_schema` | `dict[str, Any] \| None` | `None` | JSON Schema for structured output |
| `parallel_tool_calls` | `bool \| None` | `None` | Allow parallel tool calls |
| `logprobs` | `bool \| None` | `None` | Return log probabilities |
| `top_logprobs` | `int \| None` | `None` | Number of top log probabilities per token |
| `prediction` | `dict[str, Any] \| None` | `None` | Predicted output for speculative decoding |

### OllamaRequestExt

| Field | Type | Default | Description |
|---|---|---|---|
| `keep_alive` | `str \| None` | `None` | How long to keep the model loaded (e.g., `"5m"`) |
| `num_ctx` | `int \| None` | `None` | Context window size |
| `num_gpu` | `int \| None` | `None` | Number of GPU layers |
| `raw_mode` | `bool \| None` | `None` | Skip prompt templating |
| `options` | `dict[str, Any] \| None` | `None` | Additional Ollama model options |

### ThinkingConfig

Used by Anthropic and Gemini for extended thinking / chain-of-thought.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Enable extended thinking |
| `budget_tokens` | `int \| None` | `None` | Token budget for thinking |

### GroundingConfig

Used by Gemini for Google Search grounding.

| Field | Type | Default | Description |
|---|---|---|---|
| `google_search` | `bool` | `False` | Enable Google Search grounding |
| `dynamic_retrieval_threshold` | `float \| None` | `None` | Threshold for dynamic retrieval |

---

## TokenBudget (Context Management)

Controls how the context window is allocated for each LLM call. Defined in `arcana.contracts.context`.

```python
from arcana.contracts.context import TokenBudget

budget = TokenBudget(
    total_window=128000,
    identity_tokens=200,
    task_tokens=300,
    response_reserve=4096,
    tool_budget=8000,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `total_window` | `int` | `128000` | Total context window size in tokens |
| `identity_tokens` | `int` | `200` | Tokens reserved for the identity/system layer |
| `task_tokens` | `int` | `300` | Tokens reserved for the task description layer |
| `response_reserve` | `int` | `4096` | Tokens reserved for the model's response |
| `tool_budget` | `int \| None` | `None` | Hard cap on tokens for tool definitions. `None` means no cap |
| `history_budget` | `int \| None` | `None` | Hard cap on tokens for conversation history. `None` means no cap |
| `memory_budget` | `int \| None` | `None` | Hard cap on tokens for memory context. `None` means no cap |

The computed property `working_budget` returns the available tokens for working content:

```
working_budget = total_window - identity_tokens - task_tokens - response_reserve
```

---

## ToolSpec

Specification for a tool that the agent can invoke. Defined in `arcana.contracts.tool`. Normally created via the `@arcana.tool` decorator, but can be constructed directly for advanced use.

### @arcana.tool Decorator

```python
@arcana.tool(
    name="web_search",
    description="Search the web for current information",
    when_to_use="When you need up-to-date information not in your training data",
    what_to_expect="Returns search results that may need filtering for relevance",
    failure_meaning="The search service may be unavailable; try rephrasing the query",
    side_effect="read",
    requires_confirmation=False,
)
async def web_search(query: str, max_results: int = 5) -> str:
    ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | `None` | Tool name. Defaults to the function name |
| `description` | `str \| None` | `None` | Tool description. Defaults to the function docstring |
| `when_to_use` | `str \| None` | `None` | Guidance for the LLM on when to call this tool |
| `what_to_expect` | `str \| None` | `None` | What the LLM should expect from the output |
| `failure_meaning` | `str \| None` | `None` | What a failure from this tool means |
| `side_effect` | `str` | `"read"` | Side effect type: `"read"`, `"write"`, or `"none"` |
| `requires_confirmation` | `bool` | `False` | Whether to require user confirmation before execution |

### ToolSpec Fields (Full Model)

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Tool name |
| `description` | `str` | *(required)* | Tool description |
| `input_schema` | `dict[str, Any]` | *(required)* | JSON Schema for input parameters |
| `output_schema` | `dict[str, Any] \| None` | `None` | JSON Schema for output |
| `side_effect` | `SideEffect` | `SideEffect.READ` | `"read"`, `"write"`, or `"none"` |
| `requires_confirmation` | `bool` | `False` | Require user confirmation |
| `capabilities` | `list[str]` | `[]` | Capability tags |
| `max_retries` | `int` | `3` | Maximum retry attempts on failure |
| `retry_delay_ms` | `int` | `1000` | Delay between retries in milliseconds |
| `timeout_ms` | `int` | `30000` | Execution timeout in milliseconds |
| `when_to_use` | `str \| None` | `None` | LLM affordance: when to use this tool |
| `what_to_expect` | `str \| None` | `None` | LLM affordance: expected output |
| `failure_meaning` | `str \| None` | `None` | LLM affordance: what failure means |
| `success_next_step` | `str \| None` | `None` | LLM affordance: suggested next step on success |
| `category` | `str \| None` | `None` | Tool category (e.g., `"search"`, `"file"`, `"code"`, `"web"`, `"data"`, `"shell"`) |
| `related_tools` | `list[str]` | `[]` | Names of related tools |

---

## MCPServerConfig

Configuration for connecting to an MCP (Model Context Protocol) server. Defined in `arcana.contracts.mcp`.

```python
import arcana
from arcana.contracts.mcp import MCPServerConfig, MCPTransportType

runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
    mcp_servers=[
        # stdio transport (subprocess)
        MCPServerConfig(
            name="filesystem",
            transport=MCPTransportType.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            timeout_ms=10000,
        ),
        # HTTP transport
        MCPServerConfig(
            name="remote-tools",
            transport=MCPTransportType.STREAMABLE_HTTP,
            url="https://mcp.example.com/sse",
            headers={"Authorization": "Bearer xxx"},
        ),
    ],
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Server name (used for identification) |
| `transport` | `MCPTransportType` | `MCPTransportType.STDIO` | Transport type: `"stdio"`, `"sse"`, or `"streamable_http"` |
| `command` | `str \| None` | `None` | Command to launch the server (stdio transport) |
| `args` | `list[str]` | `[]` | Command arguments (stdio transport) |
| `env` | `dict[str, str]` | `{}` | Environment variables for the subprocess (stdio transport) |
| `url` | `str \| None` | `None` | Server URL (HTTP transports) |
| `headers` | `dict[str, str]` | `{}` | HTTP headers (HTTP transports) |
| `timeout_ms` | `int` | `30000` | Connection/request timeout in milliseconds |
| `reconnect_attempts` | `int` | `3` | Number of reconnection attempts |
| `reconnect_delay_ms` | `int` | `1000` | Delay between reconnection attempts in milliseconds |
| `capability_prefix` | `str \| None` | `None` | Prefix added to tool names from this server |

---

## Trace Configuration

Tracing is enabled by passing `trace=True` to `Runtime()`. Trace events are written as JSONL (one JSON object per line) to files named `{run_id}.jsonl` in the trace directory.

### TraceWriter

| Parameter | Type | Default | Description |
|---|---|---|---|
| `trace_dir` | `str \| Path` | `"./traces"` | Directory to store trace files |
| `enabled` | `bool` | `True` | Whether tracing is active |
| `namespace` | `str \| None` | `None` | Namespace for tenant isolation. When set, files go to `{trace_dir}/{namespace}/` |

### TraceConfig (from utils/config.py)

Used by `load_config()` for file-based configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | Enable trace output |
| `directory` | `Path` | `Path("./traces")` | Trace output directory |

---

## ModelGatewayRegistry

The gateway registry manages multiple LLM providers with retry and fallback support.

```python
from arcana.gateway.registry import ModelGatewayRegistry

gateway = ModelGatewayRegistry(max_retries=3, retry_base_delay_ms=1000)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_retries` | `int` | `2` | Max retry attempts per provider before falling back (`0` = no retries) |
| `retry_base_delay_ms` | `int` | `500` | Base delay for exponential backoff in milliseconds |

Retry delays use exponential backoff: `delay = retry_base_delay_ms * 2^(attempt - 1)`. If the provider returns a `retry_after_ms` hint, the larger of the two values is used.

---

## BudgetTracker

> **Note:** `BudgetTracker` is a `@dataclass`, not a Pydantic `BaseModel`.

Runtime budget enforcement. Created automatically by `Runtime`; documented here for advanced use.

```python
from arcana.gateway.budget import BudgetTracker

tracker = BudgetTracker(
    max_tokens=500_000,
    max_cost_usd=10.0,
    max_time_ms=300_000,  # 5 minutes
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `max_tokens` | `int \| None` | `None` | Maximum token limit |
| `max_cost_usd` | `float \| None` | `None` | Maximum cost limit in USD |
| `max_time_ms` | `int \| None` | `None` | Maximum time limit in milliseconds |
| `tokens_used` | `int` | `0` | Tokens consumed so far |
| `cost_usd` | `float` | `0.0` | Cost consumed so far |
| `start_time_ms` | `int` | Current time in milliseconds | Timestamp (ms) when the tracker was created |

The tracker raises `BudgetExceededError` when any limit is hit. Check remaining budget with the `tokens_remaining`, `cost_remaining`, and `time_remaining_ms` properties.

---

## V1 Engine RuntimeConfig

The V1 adaptive engine uses a separate `RuntimeConfig` defined in `arcana.contracts.runtime`. This is only relevant when using `engine="adaptive"`.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_steps` | `int` | `100` | Maximum execution steps |
| `max_consecutive_errors` | `int` | `3` | Stop after this many consecutive errors |
| `max_consecutive_no_progress` | `int` | `3` | Stop after this many turns with no progress |
| `checkpoint_interval_steps` | `int` | `5` | Steps between automatic checkpoints |
| `checkpoint_on_error` | `bool` | `True` | Create checkpoint on error |
| `checkpoint_budget_thresholds` | `list[float]` | `[0.5, 0.75, 0.9]` | Budget usage fractions that trigger checkpoints |
| `checkpoint_on_plan_step` | `bool` | `True` | Checkpoint after each plan step |
| `checkpoint_on_verification` | `bool` | `True` | Checkpoint after verification steps |
| `step_retry_count` | `int` | `2` | Number of retries per failed step |
| `step_retry_delay_ms` | `int` | `1000` | Delay between step retries in milliseconds |
| `progress_window_size` | `int` | `5` | Number of recent steps to evaluate for progress |
| `similarity_threshold` | `float` | `0.95` | Similarity threshold for detecting loops (no-progress detection) |

---

## ArcanaConfig (File-Based Configuration)

The `load_config()` function in `arcana.utils.config` loads a full configuration from environment variables and `.env` files. This is useful for applications that prefer declarative configuration over programmatic setup.

```python
from arcana.utils.config import load_config

config = load_config(env_file=".env.production")
print(config.default_model.provider)   # "gemini"
print(config.budget.max_cost_per_run_usd)  # 1.0
```

### ArcanaConfig Structure

```
ArcanaConfig
  gemini: ModelProviderConfig
  deepseek: ModelProviderConfig
  openai: ModelProviderConfig
  anthropic: ModelProviderConfig
  trace: TraceConfig
  default_model: DefaultModelConfig
  budget: BudgetConfig
```

### ModelProviderConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | Provider API key |
| `base_url` | `str \| None` | `None` | Provider base URL |

### DefaultModelConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | `"gemini"` | Default provider name |
| `model_id` | `str` | `"gemini-2.0-flash"` | Default model ID |
| `temperature` | `float` | `0.0` | Default temperature |
| `max_tokens` | `int` | `4096` | Default max output tokens |
| `timeout_ms` | `int` | `30000` | Default timeout in milliseconds |

### BudgetConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `max_tokens_per_run` | `int` | `100000` | Maximum tokens per run |
| `max_cost_per_run_usd` | `float` | `1.0` | Maximum cost per run in USD |
