# Provider Guide

Arcana supports multiple LLM providers through a unified gateway architecture.
This guide covers every built-in provider: how to install, configure, and use it.

---

## Table of Contents

- [How the Provider System Works](#how-the-provider-system-works)
- [DeepSeek](#deepseek)
- [OpenAI](#openai)
- [Anthropic](#anthropic)
- [Google Gemini](#google-gemini)
- [Chinese Providers](#chinese-providers)
  - [Kimi / Moonshot](#kimi--moonshot)
  - [GLM / Zhipu AI](#glm--zhipu-ai)
  - [MiniMax](#minimax)
- [Ollama (Local)](#ollama-local)
- [Fallback Chains](#fallback-chains)
- [Custom Provider](#custom-provider)

---

## How the Provider System Works

Arcana's provider architecture has three layers:

1. **`OpenAICompatibleProvider`** -- A single implementation that works with any
   LLM API following the OpenAI chat completions format. Most providers (DeepSeek,
   OpenAI, Gemini, Ollama, Kimi, GLM, MiniMax) use this as their base. Only
   Anthropic has a separate native implementation to support Anthropic-specific
   features like extended thinking and prompt caching.

2. **`ModelGatewayRegistry`** -- Manages multiple providers, routes requests to the
   correct one, and handles retry with exponential backoff plus fallback chains
   when a provider fails.

3. **`CapabilityRegistry`** -- A static + runtime registry that tracks what each
   provider supports (streaming, tool use, multimodal input, extended thinking,
   etc.), enabling capability-aware provider selection.

When you pass `providers={"deepseek": "sk-xxx"}` to `Runtime()`, Arcana
looks up a factory function for that name, creates the provider instance, and
registers it in the gateway. The first provider listed (or the one matching
`RuntimeConfig.default_provider`) becomes the default.

There are two ways to use providers:

- **`arcana.run()` (quick)** -- Creates a temporary Runtime. Good for scripts and
  one-off tasks.
- **`arcana.Runtime()` (production)** -- Create once at app startup, reuse across
  requests.

### Structured Output Support

All providers support `response_format`, but the mechanism varies:

| Strategy | Providers | How it works |
|----------|-----------|--------------|
| Native `json_schema` | OpenAI, Gemini | Provider API enforces the schema natively |
| Fallback `json_object` | DeepSeek, Ollama, Kimi, GLM, MiniMax | `response_format={"type": "json_object"}` + schema in system prompt |
| System prompt injection | Anthropic | JSON schema injected into system prompt (no native API support) |

### Batch Generation

Every provider exposes `batch_generate(requests, config, concurrency=5)` for
concurrent LLM calls. It reuses the existing `generate()` logic under the hood,
gating concurrency with an `asyncio.Semaphore`. This is also available on the
registry level via `ModelGatewayRegistry.batch_generate()`.

```python
from arcana.contracts.llm import LLMRequest, ModelConfig

responses = await registry.batch_generate(
    requests=[LLMRequest(messages=[...]), LLMRequest(messages=[...])],
    config=ModelConfig(model="deepseek-chat"),
    concurrency=5,
)
```

---

## DeepSeek

DeepSeek is the default provider. It is OpenAI-compatible and requires only the
`openai` Python package, which is a core dependency of Arcana.

### Install

No extra install needed -- `openai` is included in Arcana's core dependencies.

```bash
pip install arcana-agent
```

### Configure

| Setting | Value |
|---------|-------|
| Environment variable | `DEEPSEEK_API_KEY` |
| Base URL | `https://api.deepseek.com` |
| Default model | `deepseek-chat` |

### Example

```python
import arcana

# Quick run -- api_key passed directly (preferred)
result = await arcana.run(
    "Explain quantum entanglement",
    provider="deepseek",
    api_key="sk-xxx",
)
print(result.output)

# Production -- Runtime with explicit provider
runtime = arcana.Runtime(
    providers={"deepseek": "sk-xxx"},
)
result = await runtime.run("Explain quantum entanglement")
```

If `api_key` is omitted or empty, Arcana reads from the `DEEPSEEK_API_KEY`
environment variable.

### Available Models

| Model | Description |
|-------|-------------|
| `deepseek-chat` | General-purpose chat (default) |
| `deepseek-coder` | Code-focused |
| `deepseek-reasoner` | Deep thinking / chain-of-thought |

### Capabilities

Chat, streaming, tool use, structured output, deep thinking.

### Notes

- DeepSeek is the default provider (`RuntimeConfig.default_provider = "deepseek"`).
  You can change this in RuntimeConfig.
- The `deepseek-reasoner` model supports the `deep_thinking` capability for
  chain-of-thought reasoning tasks.

---

## OpenAI

OpenAI uses the `OpenAICompatibleProvider` base class pointed at the official
OpenAI API endpoint.

### Install

No extra install needed -- `openai` is a core dependency.

```bash
pip install arcana-agent
```

### Configure

| Setting | Value |
|---------|-------|
| Environment variable | `OPENAI_API_KEY` |
| Base URL | `https://api.openai.com/v1` |
| Default model | `gpt-4o-mini` |

### Example

```python
import arcana

# Quick run
result = await arcana.run(
    "Write a haiku about programming",
    provider="openai",
    api_key="sk-proj-xxx",
)

# Production
runtime = arcana.Runtime(
    providers={"openai": "sk-proj-xxx"},
    config=arcana.RuntimeConfig(default_provider="openai"),
)
result = await runtime.run("Write a haiku about programming")
```

### Available Models

Any model available on your OpenAI account. Common choices:

| Model | Description |
|-------|-------------|
| `gpt-4o-mini` | Fast, cost-effective (default) |
| `gpt-4o` | Most capable GPT-4o |
| `gpt-4.1` | Latest GPT-4 flagship |
| `gpt-4.1-mini` | Balanced speed/capability |
| `gpt-4.1-nano` | Ultra-fast, lowest cost |
| `o3` | Full reasoning model |
| `o3-mini` | Small reasoning model |
| `o4-mini` | Latest small reasoning model |

To use a specific model, pass it via `model=` in `arcana.run()` or set
`default_model` in `RuntimeConfig`.

### Capabilities

Chat, streaming, tool use, multimodal input, structured output, JSON schema
output, parallel tool calls, logprobs, predicted output.

---

## Anthropic

Anthropic (Claude) has a **native provider implementation** that talks directly
to the Anthropic Messages API. This is the only provider that does not use
`OpenAICompatibleProvider`, because it needs first-class support for
Anthropic-specific features like extended thinking, prompt caching, computer
use, and PDF input.

### Install

The Anthropic SDK is an **optional dependency**. Install it explicitly:

```bash
pip install arcana-agent[anthropic]

# or install directly:
pip install anthropic>=0.42
```

### Configure

| Setting | Value |
|---------|-------|
| Environment variable | `ANTHROPIC_API_KEY` |
| API | Anthropic Messages API (native, not OpenAI-compatible) |
| Default Model | No default (must specify) |

> **Important:** Unlike other providers, Anthropic has no default model. You MUST
> specify a model explicitly -- either via the `model` parameter in `arcana.run()`
> or via `default_model` in `RuntimeConfig`. Omitting it will raise an error.

### Example

```python
import arcana

# Quick run
result = await arcana.run(
    "Analyze this code for bugs",
    provider="anthropic",
    api_key="sk-ant-xxx",
    model="claude-sonnet-4-20250514",  # required -- Anthropic has no default model
)

# Production
from arcana.runtime_core import RuntimeConfig

runtime = arcana.Runtime(
    providers={"anthropic": "sk-ant-xxx"},
    config=RuntimeConfig(
        default_provider="anthropic",
        default_model="claude-sonnet-4-20250514",  # required -- Anthropic has no default model
    ),
)
result = await runtime.run("Analyze this code for bugs")
```

### Extended Thinking

Anthropic models support extended thinking (chain-of-thought reasoning visible
to the caller). This is configured through the `LLMRequest.anthropic` extension
field:

```python
from arcana.contracts.llm import (
    AnthropicRequestExt,
    LLMRequest,
    ModelConfig,
    ThinkingConfig,
)

request = LLMRequest(
    messages=[...],
    anthropic=AnthropicRequestExt(
        thinking=ThinkingConfig(enabled=True, budget_tokens=4096),
    ),
)
```

When extended thinking is enabled, temperature is automatically omitted from the
API call (Anthropic requires this). Thinking blocks are returned in
`response.anthropic.thinking_blocks`.

### Available Models

| Model | Description |
|-------|-------------|
| `claude-opus-4-20250514` | Most capable |
| `claude-sonnet-4-20250514` | Balanced speed/capability |
| `claude-haiku-4-20250414` | Fast and lightweight |

### Capabilities

Chat, streaming, tool use, multimodal input, structured output, extended
thinking, prompt caching, computer use, PDF input.

### Notes

- Tool definitions are automatically converted from OpenAI format to Anthropic
  format. You use the same `@arcana.tool` decorator regardless of provider.
- System messages are extracted and passed via the Anthropic `system` parameter
  (not as a message), matching what the Anthropic API expects.
- Error mapping is comprehensive: rate limit (429), auth (401), not found (404),
  content filter, context length, and overloaded (529) errors are all mapped to
  the appropriate `ProviderError` subclass.

### Structured Output (v0.2.0)

As of v0.2.0, Anthropic supports `response_format` for structured output.
Because the Anthropic Messages API has no native `response_format` parameter,
Arcana injects the JSON schema into the system prompt and instructs the model to
respond with conforming JSON. This is the same fallback strategy used by
DeepSeek, Ollama, Kimi, GLM, and MiniMax. Structured output works with and
without tools -- the two capabilities coexist.

---

## Google Gemini

Gemini uses Google's OpenAI-compatible endpoint, so it runs on the standard
`OpenAICompatibleProvider`.

### Install

The `openai` package (core dependency) is all you need for the OpenAI-compatible
endpoint. The optional `google-genai` package is listed under extras but is not
required for basic Gemini usage through the compatibility layer:

```bash
pip install arcana-agent

# Optional: install the native Google SDK if needed elsewhere
pip install arcana-agent[gemini]
```

### Configure

| Setting | Value |
|---------|-------|
| Environment variable | `GEMINI_API_KEY` |
| Base URL | `https://generativelanguage.googleapis.com/v1beta/openai` |
| Default model | `gemini-2.0-flash` |

### Example

```python
import arcana

# Quick run
result = await arcana.run(
    "Summarize the history of the internet",
    provider="gemini",
    api_key="AIza-xxx",
)

# Production
runtime = arcana.Runtime(
    providers={"gemini": "AIza-xxx"},
    config=arcana.RuntimeConfig(default_provider="gemini"),
)
result = await runtime.run("Summarize the history of the internet")
```

### Available Models

| Model | Description |
|-------|-------------|
| `gemini-2.0-flash` | Fast multimodal (default) |
| `gemini-2.0-flash-lite` | Ultra lightweight |
| `gemini-1.5-flash` | Previous generation fast |
| `gemini-1.5-flash-8b` | 8B parameter variant |
| `gemini-1.5-pro` | Previous generation pro |

### Capabilities

Chat, streaming, tool use, multimodal input, structured output, grounding,
code execution, safety settings, cached content.

---

## Chinese Providers

Arcana has built-in support for Chinese LLM providers. All three use the
`OpenAICompatibleProvider` base with pre-configured endpoints.

### Kimi / Moonshot

| Setting | Value |
|---------|-------|
| Environment variable | `KIMI_API_KEY` |
| Base URL | `https://api.moonshot.cn/v1` |
| Default model | `moonshot-v1-8k` |

```python
runtime = arcana.Runtime(
    providers={"kimi": "sk-xxx"},
    config=arcana.RuntimeConfig(default_provider="kimi"),
)
result = await runtime.run("Summarize this long document")
```

**Available models:** `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`

**Capabilities:** Chat, streaming, tool use, long context, web search.

### GLM / Zhipu AI

| Setting | Value |
|---------|-------|
| Environment variable | `GLM_API_KEY` |
| Base URL | `https://open.bigmodel.cn/api/paas/v4` |
| Default model | `glm-4-flash` |

```python
runtime = arcana.Runtime(
    providers={"glm": "xxx.xxx"},
    config=arcana.RuntimeConfig(default_provider="glm"),
)
result = await runtime.run("Analyze this image")
```

**Available models:** `glm-4`, `glm-4-flash`, `glm-4v`, `glm-4-long`

**Capabilities:** Chat, streaming, tool use, multimodal input, web search,
code interpreter.

### MiniMax

| Setting | Value |
|---------|-------|
| Environment variable | `MINIMAX_API_KEY` |
| Base URL | `https://api.minimax.chat/v1` |
| Default model | `abab6.5s-chat` |

```python
runtime = arcana.Runtime(
    providers={"minimax": "xxx"},
    config=arcana.RuntimeConfig(default_provider="minimax"),
)
result = await runtime.run("Generate a story")
```

**Available models:** `abab6.5s-chat`, `abab6.5-chat`, `abab5.5-chat`

**Capabilities:** Chat, streaming, tool use, long context, text-to-audio.

---

## Ollama (Local)

Ollama runs models locally on your machine. It exposes an OpenAI-compatible API,
so Arcana uses `OpenAICompatibleProvider` with no real API key required.

### Install

1. Install Ollama from [ollama.com](https://ollama.com).
2. Pull a model:

```bash
ollama pull llama3.2
```

3. Ollama runs on `http://localhost:11434` by default.

No extra Python packages are needed beyond Arcana's core dependencies.

### Configure

| Setting | Value |
|---------|-------|
| Environment variable | None required |
| Base URL | `http://localhost:11434/v1` |
| Default model | `llama3.2` |
| API key | `"ollama"` (placeholder, not validated) |

### Example

```python
import arcana

# Production -- Ollama needs no API key
runtime = arcana.Runtime(
    providers={"ollama": "ollama"},
    config=arcana.RuntimeConfig(default_provider="ollama"),
)
result = await runtime.run("Write a poem")
```

With `arcana.run()`:

```python
result = await arcana.run(
    "Write a poem",
    provider="ollama",
    api_key="ollama",
)
```

### Available Models

Whatever you have pulled locally. Common choices:

| Model | Size |
|-------|------|
| `llama3.2` | 3B (default) |
| `llama3.1` | 8B / 70B |
| `mistral` | 7B |
| `codellama` | 7B / 13B / 34B |
| `phi3` | 3.8B |

### Capabilities

Chat, streaming, tool use, local execution, model management, raw generate.

### Notes

- Make sure Ollama is running (`ollama serve`) before starting your Arcana
  application.
- To use a different model, set `default_model` in `RuntimeConfig` or override
  the model in your `ModelConfig`.

---

## Fallback Chains

The `ModelGatewayRegistry` supports fallback chains: when the primary provider
fails with a retryable error (rate limit, timeout, server error), the registry
automatically retries with exponential backoff and then falls back to backup
providers.

### How It Works

1. Request goes to the primary provider.
2. If a **retryable** error occurs (429, 502, 503, 504, connection error, timeout),
   the registry retries up to `max_retries` times with exponential backoff
   (base delay 500ms, doubling each attempt).
3. If all retries are exhausted, the request moves to the next provider in the
   fallback chain.
4. Each fallback provider also gets `max_retries` retry attempts.
5. **Non-retryable** errors (auth failure, model not found, content filter,
   context length exceeded) are raised immediately without retry or fallback.

### Configuration

```python
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.gateway.providers.openai_compatible import (
    create_deepseek_provider,
    create_gemini_provider,
)
from arcana.gateway.providers.anthropic import AnthropicProvider

# Create registry with custom retry settings
registry = ModelGatewayRegistry(
    max_retries=2,           # default: 2 retries before fallback
    retry_base_delay_ms=500, # default: 500ms base delay
)

# Register providers
registry.register("deepseek", create_deepseek_provider("sk-xxx"))
registry.register("openai", OpenAICompatibleProvider(
    provider_name="openai",
    api_key="sk-proj-xxx",
    base_url="https://api.openai.com/v1",
    default_model="gpt-4o-mini",
))
registry.register("gemini", create_gemini_provider("AIza-xxx"))

# Set fallback chain: deepseek -> openai -> gemini
registry.set_fallback_chain("deepseek", ["openai", "gemini"])

# Set default provider
registry.set_default("deepseek")
```

### Streaming Fallback

Fallback works for streaming too, with one important constraint: fallback only
occurs if **no chunks have been yielded yet**. Once the first chunk is sent to
the caller, a mid-stream error propagates directly (because partial output has
already been delivered and switching providers mid-response would produce
incoherent results).

---

## Custom Provider

Since most modern LLM APIs follow the OpenAI chat completions format, adding a
new provider is straightforward. There are two approaches.

### Option 1: Factory Function (Recommended)

Create a factory function that returns an `OpenAICompatibleProvider` with your
provider's settings:

```python
from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider


def create_my_provider(
    api_key: str,
    base_url: str = "https://api.myprovider.com/v1",
) -> OpenAICompatibleProvider:
    """Create a provider for MyProvider's API."""
    return OpenAICompatibleProvider(
        provider_name="myprovider",
        api_key=api_key,
        base_url=base_url,
        default_model="my-model-v1",
        supported_models=["my-model-v1", "my-model-v2"],
    )
```

Then register it with the gateway:

```python
from arcana.gateway.registry import ModelGatewayRegistry

registry = ModelGatewayRegistry()
registry.register("myprovider", create_my_provider("my-api-key"))
registry.set_default("myprovider")
```

### Option 2: Subclass

For providers that need customized behavior, subclass `OpenAICompatibleProvider`:

```python
from arcana.gateway.providers.openai_compatible import OpenAICompatibleProvider


class MyProvider(OpenAICompatibleProvider):
    DEFAULT_BASE_URL = "https://api.myprovider.com/v1"
    DEFAULT_MODEL = "my-model-v1"
    SUPPORTED_MODELS = ["my-model-v1", "my-model-v2"]

    def __init__(self, api_key: str, base_url: str | None = None):
        super().__init__(
            provider_name="myprovider",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            default_model=self.DEFAULT_MODEL,
            supported_models=self.SUPPORTED_MODELS,
        )
```

### Option 3: BaseProvider Protocol (Non-OpenAI APIs)

If the LLM API does not follow the OpenAI format at all, implement the
`BaseProvider` protocol directly:

```python
from arcana.gateway.base import BaseProvider
from arcana.contracts.llm import LLMRequest, LLMResponse, ModelConfig, StreamChunk
from arcana.contracts.trace import TraceContext
from collections.abc import AsyncIterator


class MyNativeProvider:
    """Implements the BaseProvider protocol."""

    @property
    def provider_name(self) -> str:
        return "myprovider"

    @property
    def supported_models(self) -> list[str]:
        return ["my-model-v1"]

    async def generate(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> LLMResponse:
        # Your API call logic here
        ...

    async def stream(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> AsyncIterator[StreamChunk]:
        # Your streaming logic here
        ...

    async def health_check(self) -> bool:
        # Return True if the API is reachable
        ...
```

### Extra Headers

`OpenAICompatibleProvider` accepts an `extra_headers` parameter for APIs that
require custom HTTP headers:

```python
provider = OpenAICompatibleProvider(
    provider_name="myprovider",
    api_key="xxx",
    base_url="https://api.myprovider.com/v1",
    default_model="my-model",
    extra_headers={"X-Custom-Header": "value"},
)
```

### Registering Capabilities

If you want your custom provider to participate in capability-based selection,
register its capabilities:

```python
from arcana.gateway.capabilities import Capability, CapabilityRegistry

cap_registry = CapabilityRegistry()
cap_registry.register("myprovider", frozenset({
    Capability.CHAT,
    Capability.STREAMING,
    Capability.TOOL_USE,
}))

# Now you can query
cap_registry.supports("myprovider", Capability.TOOL_USE)  # True
cap_registry.best_provider_for(
    required={Capability.CHAT, Capability.TOOL_USE},
    preferred=["myprovider", "deepseek"],
)
```
