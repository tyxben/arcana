# 第二章：Model Gateway 与 Provider 系统

> 本章深入剖析 Arcana 的模型网关层——如何用一套统一接口对接多家 LLM 供应商，同时实现预算控制、故障转移和全链路追踪。

---

## 目录

1. [Model Gateway 概述](#1-model-gateway-概述)
2. [抽象基类设计](#2-抽象基类设计)
3. [Provider 适配器模式](#3-provider-适配器模式)
4. [注册表与路由](#4-注册表与路由)
5. [预算控制](#5-预算控制)
6. [Trace 集成](#6-trace-集成)
7. [设计模式总结](#7-设计模式总结)
8. [生产注意事项](#8-生产注意事项)
9. [本章小结](#9-本章小结)

---

## 1. Model Gateway 概述

### 为什么需要网关层？

在构建 AI Agent 系统时，直接调用 LLM API 会带来三个核心问题：

**问题一：供应商锁定（Vendor Lock-in）**

如果业务逻辑直接依赖某家供应商的 SDK（如 `openai.ChatCompletion.create()`），切换供应商时需要修改所有调用点。Arcana 的网关层通过统一的 `generate()` 接口，让上层代码完全不感知底层供应商差异。

**问题二：成本失控**

LLM 调用按 token 计费，一个有 bug 的循环可能在几分钟内烧掉数百美元。网关层内置了 `BudgetTracker`，在每次调用前后检查 token、费用和时间的预算限制。

**问题三：单点故障**

依赖单一供应商意味着当该供应商宕机或限流时，整个系统不可用。网关层的注册表（Registry）支持配置故障转移链（fallback chain），自动在多家供应商之间切换。

### 架构总览

```
                    ┌──────────────────┐
                    │  Agent / Runtime  │
                    └────────┬─────────┘
                             │ generate(request, config)
                    ┌────────▼─────────┐
                    │ ModelGatewayRegistry │  ← 路由 + 故障转移
                    └────────┬─────────┘
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ DeepSeek     │ │ Gemini       │ │ Ollama       │
    │ Provider     │ │ Provider     │ │ Provider     │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           ▼                ▼                ▼
    OpenAICompatibleProvider (共享实现)
           │
    ┌──────▼───────┐
    │ AsyncOpenAI  │  ← openai SDK
    │   Client     │
    └──────────────┘
```

**关键洞察**：由于现代 LLM 供应商几乎都提供了 OpenAI 兼容的 API 端点，Arcana 只需要**一个**核心实现类 `OpenAICompatibleProvider`，通过不同的 `base_url` 和 `api_key` 即可对接所有供应商。

---

## 2. 抽象基类设计

### ModelGateway ABC

文件：`src/arcana/gateway/base.py`

`ModelGateway` 是所有 Provider 的抽象基类，定义了三个核心契约：

```python
# src/arcana/gateway/base.py:13-56
class ModelGateway(ABC):
    """Abstract base class for LLM provider implementations."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of this provider."""
        ...

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """Get list of model IDs supported by this provider."""
        ...

    @abstractmethod
    async def generate(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        ...

    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible."""
        return True
```

#### 设计决策解析

**为什么 `generate()` 是异步的？**

LLM 调用本质上是网络 I/O 密集型操作，一次调用通常耗时数秒。使用 `async` 可以让 Agent 在等待 LLM 响应时执行其他工作（如并行调用多个 tool）。

**为什么将 `request` 和 `config` 分开？**

`LLMRequest` 包含消息、工具定义等**业务数据**，而 `ModelConfig` 包含 provider、model_id、temperature 等**运行时配置**。分离二者的好处是：同一个请求可以轻松发送给不同的 provider/model，只需更换 config 即可——这正是故障转移所需要的。

**为什么 `trace_ctx` 是可选的？**

并非所有场景都需要追踪（如单元测试、本地实验），将其设为可选避免了不必要的依赖。

### 异常体系

```python
# src/arcana/gateway/base.py:68-89
class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        retryable: bool = False,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class BudgetExceededError(Exception):
    """Exception raised when budget limits are exceeded."""

    def __init__(self, message: str, budget_type: str):
        super().__init__(message)
        self.budget_type = budget_type
```

**`retryable` 标志是核心设计**。`ProviderError` 携带了 `retryable` 属性，让注册表在路由层可以智能决策：

- `retryable=True`（如 429 限流、503 过载）→ 尝试 fallback chain
- `retryable=False`（如 400 参数错误、401 认证失败）→ 直接抛出，不重试

这避免了对不可恢复错误做无意义的重试，同时对临时性故障自动降级。

---

## 3. Provider 适配器模式

### OpenAICompatibleProvider：一个类支撑多家供应商

文件：`src/arcana/gateway/providers/openai_compatible.py`

这是整个 Gateway 层最核心的类。现代 LLM 供应商（DeepSeek、Gemini、Ollama、vLLM、LiteLLM、Azure OpenAI、Together AI、Groq 等）几乎都支持 OpenAI 的 Chat Completions API 格式，因此一个实现即可通吃。

```python
# src/arcana/gateway/providers/openai_compatible.py:49-81
class OpenAICompatibleProvider(ModelGateway):
    """
    Universal provider for OpenAI-compatible APIs.

    This single implementation can be used for any LLM API that follows
    the OpenAI chat completions format. Just provide different base_url
    and api_key for each provider.
    """
```

#### 构造函数：灵活的初始化

```python
# src/arcana/gateway/providers/openai_compatible.py:83-120
def __init__(
    self,
    provider_name: str,
    api_key: str,
    base_url: str,
    default_model: str | None = None,
    supported_models: list[str] | None = None,
    trace_writer: TraceWriter | None = None,
    extra_headers: dict[str, str] | None = None,
):
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai is not installed. Install with: pip install openai"
        )

    self._provider_name = provider_name
    self._default_model = default_model
    self._supported_models = supported_models or []
    self.trace_writer = trace_writer

    # Create AsyncOpenAI client
    self.client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=extra_headers,
    )
```

**关键设计点**：

1. **依赖检测**（`OPENAI_AVAILABLE`）：openai SDK 是可选依赖，未安装时给出清晰的错误提示，而非隐晦的 `AttributeError`
2. **`extra_headers`**：某些供应商需要额外的请求头（如 API 版本标识），此参数提供了扩展性
3. **复用 `AsyncOpenAI` 客户端**：不重新发明轮子，直接利用 openai SDK 成熟的连接池、重试机制和类型安全

#### 消息格式转换

```python
# src/arcana/gateway/providers/openai_compatible.py:134-150
def _convert_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
    """Convert Arcana messages to OpenAI format."""
    result = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump()

        # Handle role conversion
        role = msg.get("role", "user")
        if hasattr(role, "value"):  # Enum
            role = role.value

        result.append({
            "role": role,
            "content": msg.get("content", ""),
        })
    return result
```

这个方法将 Arcana 内部的 `Message` Pydantic 模型（使用 `MessageRole` 枚举）转换为 OpenAI API 需要的纯字典格式。注意对 Enum 的 `.value` 提取——这是 Pydantic v2 模型与 OpenAI API 之间的桥梁。

#### generate() 核心流程

`generate()` 方法是整个调用链的核心，可以分为五个阶段：

**阶段一：请求构建**

```python
# src/arcana/gateway/providers/openai_compatible.py:159-191
messages = self._convert_messages(request.messages)

# Build request parameters
params: dict[str, Any] = {
    "model": config.model_id,
    "messages": messages,
    "temperature": config.temperature,
    "max_tokens": config.max_tokens,
}

# Add seed if specified (for reproducibility)
if config.seed is not None:
    params["seed"] = config.seed

# Add response format if schema specified
if request.response_schema:
    params["response_format"] = {"type": "json_object"}

# Add tools if specified
if request.tools:
    params["tools"] = request.tools

# Add any extra params from config
if config.extra_params:
    params.update(config.extra_params)
```

注意 `extra_params` 的设计——它允许传入供应商特有的参数（如 DeepSeek 的 `top_k`、Gemini 的 `safety_settings`），而无需修改核心接口。这是**开闭原则**的体现：对扩展开放，对修改关闭。

**阶段二：API 调用与错误分类**

```python
# src/arcana/gateway/providers/openai_compatible.py:194-219
response = await self.client.chat.completions.create(**params)

# 错误处理将不同类型的异常映射到 ProviderError
except RateLimitError as e:
    raise ProviderError(str(e), provider=self._provider_name,
                        retryable=True, status_code=429) from e
except (APIConnectionError, APITimeoutError) as e:
    raise ProviderError(str(e), provider=self._provider_name,
                        retryable=True) from e
except Exception as e:
    error_msg = str(e)
    retryable = any(code in error_msg for code in ["503", "529", "502", "504"])
    raise ProviderError(error_msg, provider=self._provider_name,
                        retryable=retryable) from e
```

错误分类逻辑：
- **429（RateLimitError）**：限流，可重试 → fallback
- **连接/超时错误**：网络问题，可重试 → fallback
- **503/502/504**：服务端过载，可重试 → fallback
- **其他错误**：不可重试，直接上抛

**阶段三：响应解析**

```python
# src/arcana/gateway/providers/openai_compatible.py:222-251
choice = response.choices[0]
content = choice.message.content

# Parse tool calls if any
tool_calls = None
if choice.message.tool_calls:
    tool_calls = [
        ToolCallRequest(
            id=tc.id,
            name=tc.function.name,
            arguments=tc.function.arguments,
        )
        for tc in choice.message.tool_calls
    ]

# Get usage
usage = TokenUsage(
    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
    completion_tokens=response.usage.completion_tokens if response.usage else 0,
    total_tokens=response.usage.total_tokens if response.usage else 0,
)

llm_response = LLMResponse(
    content=content,
    tool_calls=tool_calls,
    usage=usage,
    model=response.model,
    finish_reason=choice.finish_reason or "stop",
)
```

注意 tool calls 的解析——将 OpenAI SDK 的 `tool_calls` 对象转换为 Arcana 自己的 `ToolCallRequest`，保持内部模型与外部 SDK 的解耦。

### 工厂函数：预配置的供应商快捷方式

```python
# src/arcana/gateway/providers/openai_compatible.py:293-342
def create_deepseek_provider(
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    trace_writer: TraceWriter | None = None,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider_name="deepseek",
        api_key=api_key,
        base_url=base_url,
        default_model="deepseek-chat",
        supported_models=["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        trace_writer=trace_writer,
    )


def create_gemini_provider(
    api_key: str,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
    trace_writer: TraceWriter | None = None,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider_name="gemini",
        api_key=api_key,
        base_url=base_url,
        default_model="gemini-2.0-flash",
        supported_models=["gemini-2.0-flash", "gemini-2.0-flash-lite",
                          "gemini-1.5-flash", "gemini-1.5-pro"],
        trace_writer=trace_writer,
    )


def create_ollama_provider(
    base_url: str = "http://localhost:11434/v1",
    default_model: str = "llama3.2",
    trace_writer: TraceWriter | None = None,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider_name="ollama",
        api_key="ollama",  # Ollama doesn't require a real API key
        base_url=base_url,
        default_model=default_model,
        trace_writer=trace_writer,
    )
```

工厂函数封装了每家供应商的默认配置（URL、模型列表等），用户只需提供 API key 即可使用。注意 Ollama 的 `api_key="ollama"` ——本地部署不需要真实密钥，但 OpenAI SDK 要求此字段非空。

### 具体 Provider 子类：薄包装器

DeepSeek 和 Gemini 各有一个子类，但它们极其精简——仅提供类常量和简化的构造函数：

```python
# src/arcana/gateway/providers/deepseek.py:19-58
class DeepSeekProvider(OpenAICompatibleProvider):
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"
    SUPPORTED_MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]

    def __init__(self, api_key: str, base_url: str | None = None,
                 trace_writer: TraceWriter | None = None):
        super().__init__(
            provider_name="deepseek",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            default_model=self.DEFAULT_MODEL,
            supported_models=self.SUPPORTED_MODELS,
            trace_writer=trace_writer,
        )
```

```python
# src/arcana/gateway/providers/gemini.py:19-60
class GeminiProvider(OpenAICompatibleProvider):
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    DEFAULT_MODEL = "gemini-2.0-flash"
    SUPPORTED_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite",
                        "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

    def __init__(self, api_key: str, base_url: str | None = None,
                 trace_writer: TraceWriter | None = None):
        super().__init__(
            provider_name="gemini",
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            default_model=self.DEFAULT_MODEL,
            supported_models=self.SUPPORTED_MODELS,
            trace_writer=trace_writer,
        )
```

**为什么既有工厂函数又有子类？**

两种方式各有适用场景：
- **工厂函数**（`create_deepseek_provider()`）：适合快速使用，返回 `OpenAICompatibleProvider` 实例，无需额外导入
- **子类**（`DeepSeekProvider`）：适合需要 `isinstance` 检查或后续扩展供应商特有行为的场景

这种双轨设计在 Python 生态中很常见（如 `pathlib.Path` 既有工厂方法也有子类）。

---

## 4. 注册表与路由

文件：`src/arcana/gateway/registry.py`

`ModelGatewayRegistry` 是多供应商管理的核心，负责三个职责：Provider 注册、请求路由、故障转移。

### 核心数据结构

```python
# src/arcana/gateway/registry.py:15-29
class ModelGatewayRegistry:
    """Registry for managing multiple LLM providers."""

    def __init__(self) -> None:
        self._providers: dict[str, ModelGateway] = {}
        self._fallback_chains: dict[str, list[str]] = {}
        self._default_provider: str | None = None
```

三个内部状态：
- `_providers`：名称 → Provider 实例的映射表
- `_fallback_chains`：主 Provider → 备选 Provider 列表
- `_default_provider`：当请求未指定 provider 时的默认选项

### 注册与配置

```python
# 注册 Provider
registry = ModelGatewayRegistry()
registry.register("gemini", gemini_provider)
registry.register("deepseek", deepseek_provider)
registry.register("ollama", ollama_provider)

# 设置默认 Provider
registry.set_default("gemini")

# 配置故障转移链
registry.set_fallback_chain("gemini", ["deepseek", "ollama"])
registry.set_fallback_chain("deepseek", ["gemini"])
```

### 路由与故障转移逻辑

`generate()` 方法实现了完整的路由和故障转移策略：

```python
# src/arcana/gateway/registry.py:98-159
async def generate(
    self,
    request: LLMRequest,
    config: ModelConfig,
    trace_ctx: TraceContext | None = None,
    use_fallback: bool = True,
) -> LLMResponse:
    provider_name = config.provider
    provider = self._providers.get(provider_name)

    # Fall back to default provider if specified provider not found
    if provider is None and self._default_provider:
        provider_name = self._default_provider
        provider = self._providers.get(provider_name)

    if provider is None:
        raise KeyError(f"Provider '{config.provider}' is not registered")

    # Try primary provider
    try:
        return await provider.generate(request, config, trace_ctx)
    except ProviderError as e:
        if not use_fallback or not e.retryable:
            raise

        # Try fallback chain
        fallbacks = self._fallback_chains.get(provider_name, [])
        last_error = e

        for fallback_name in fallbacks:
            fallback = self._providers.get(fallback_name)
            if fallback is None:
                continue

            # Create new config for fallback
            fallback_config = config.model_copy(
                update={"provider": fallback_name}
            )

            try:
                return await fallback.generate(
                    request, fallback_config, trace_ctx
                )
            except ProviderError as fallback_error:
                last_error = fallback_error
                if not fallback_error.retryable:
                    raise

        # All providers failed
        raise last_error from last_error
```

#### 路由决策流程图

```
请求到达 (config.provider = "gemini")
    │
    ▼
provider 已注册？ ─── 否 ──→ 使用 default_provider
    │                              │
    是                      default 也为空？ → KeyError
    │
    ▼
调用 provider.generate()
    │
    ├── 成功 → 返回 LLMResponse
    │
    └── ProviderError
           │
           ├── retryable=False → 直接抛出
           │
           └── retryable=True + use_fallback=True
                   │
                   ▼
              遍历 fallback_chains["gemini"]
                   │
                   ├── fallback[0] "deepseek" → 成功？返回
                   │                         → 失败？继续
                   │
                   ├── fallback[1] "ollama"  → 成功？返回
                   │                         → 失败？继续
                   │
                   └── 全部失败 → 抛出最后一个错误
```

**关键细节**：`config.model_copy(update={"provider": fallback_name})`——使用 Pydantic v2 的 `model_copy()` 创建配置副本，仅修改 provider 字段。这保证了原始 config 不被污染，同时让 fallback provider 知道自己在处理一个转发请求。

### 健康检查

```python
# src/arcana/gateway/registry.py:161-174
async def health_check_all(self) -> dict[str, bool]:
    """Check health of all registered providers."""
    results = {}
    for name, provider in self._providers.items():
        try:
            results[name] = await provider.health_check()
        except Exception:
            results[name] = False
    return results
```

返回所有 Provider 的健康状态字典，可用于监控面板或自动化运维。

---

## 5. 预算控制

文件：`src/arcana/gateway/budget.py`

### BudgetTracker：三维预算守卫

`BudgetTracker` 跟踪三个维度的资源消耗：

```python
# src/arcana/gateway/budget.py:14-33
@dataclass
class BudgetTracker:
    """Tracks token usage, cost, and time against budget limits.
    Thread-safe budget tracking with real-time limit enforcement."""

    # Limits
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_time_ms: int | None = None

    # Consumed
    tokens_used: int = field(default=0)
    cost_usd: float = field(default=0.0)
    start_time_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    # Thread safety
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
```

| 维度 | 限制字段 | 消耗字段 | 用途 |
|------|---------|---------|------|
| Token | `max_tokens` | `tokens_used` | 防止 token 消耗失控 |
| 费用 | `max_cost_usd` | `cost_usd` | 直接控制美元成本 |
| 时间 | `max_time_ms` | `elapsed_ms` | 防止长时间运行 |

### 线程安全设计

注意 `_lock: threading.Lock` 的使用——所有修改状态的方法都通过锁保护：

```python
# src/arcana/gateway/budget.py:73-97
def check_budget(self) -> None:
    """Check if any budget limit has been exceeded."""
    with self._lock:
        if self.max_tokens and self.tokens_used >= self.max_tokens:
            raise BudgetExceededError(
                f"Token budget exceeded: {self.tokens_used}/{self.max_tokens}",
                budget_type="tokens",
            )
        if self.max_cost_usd and self.cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(
                f"Cost budget exceeded: ${self.cost_usd:.4f}/${self.max_cost_usd:.4f}",
                budget_type="cost",
            )
        if self.max_time_ms and self.elapsed_ms >= self.max_time_ms:
            raise BudgetExceededError(
                f"Time budget exceeded: {self.elapsed_ms}ms/{self.max_time_ms}ms",
                budget_type="time",
            )
```

**为什么需要线程安全？**

虽然 Python 有 GIL，但 `BudgetTracker` 可能在多个异步任务中被共享——例如一个 Agent 的多个并行 tool 调用都需要检查预算。`threading.Lock` 确保 `check_budget()` 和 `add_usage()` 的原子性。

### 用量累加

```python
# src/arcana/gateway/budget.py:99-109
def add_usage(self, usage: TokenUsage, cost: float | None = None) -> None:
    """Add token usage and optionally cost."""
    with self._lock:
        self.tokens_used += usage.total_tokens
        self.cost_usd += cost if cost is not None else usage.cost_estimate
```

`cost` 参数可选：如果供应商返回了精确费用就用精确值，否则使用 `TokenUsage.cost_estimate` 的估算值。

### 预测性检查

```python
# src/arcana/gateway/budget.py:111-123
def can_afford(self, estimated_tokens: int) -> bool:
    """Check if the budget can afford an estimated token count."""
    if self.max_tokens and (self.tokens_used + estimated_tokens) > self.max_tokens:
        return False
    return True
```

`can_afford()` 是**预测性**检查——在发起 LLM 调用之前估算是否还有足够预算。这比调用后才发现超预算要好得多，因为一次 LLM 调用可能消耗数千 token。

### 快照导出

```python
# src/arcana/gateway/budget.py:125-134
def to_snapshot(self) -> BudgetSnapshot:
    """Create a snapshot of current budget state."""
    return BudgetSnapshot(
        max_tokens=self.max_tokens,
        max_cost_usd=self.max_cost_usd,
        max_time_ms=self.max_time_ms,
        tokens_used=self.tokens_used,
        cost_usd=self.cost_usd,
        time_ms=self.elapsed_ms,
    )
```

`to_snapshot()` 创建一个不可变的 Pydantic 模型快照，可以安全地写入 Trace 或返回给上层。

### 从 Budget 契约创建

```python
# src/arcana/gateway/budget.py:35-45
@classmethod
def from_budget(cls, budget: Budget | None) -> BudgetTracker:
    """Create a tracker from a Budget object."""
    if budget is None:
        return cls()

    return cls(
        max_tokens=budget.max_tokens,
        max_cost_usd=budget.max_cost_usd,
        max_time_ms=budget.max_time_ms,
    )
```

当 `budget=None` 时返回一个无限制的 tracker——这是**空对象模式**的应用，避免了上层代码到处做 `if tracker is not None` 的检查。

---

## 6. Trace 集成

### LLM 调用的自动追踪

每次 `generate()` 调用都会自动记录到 Trace 系统。这发生在 `OpenAICompatibleProvider.generate()` 的末尾：

```python
# src/arcana/gateway/providers/openai_compatible.py:254-270
if self.trace_writer and trace_ctx:
    response_digest = canonical_hash({
        "content": content,
        "usage": usage.model_dump(),
    })

    event = TraceEvent(
        run_id=trace_ctx.run_id,
        task_id=trace_ctx.task_id,
        step_id=trace_ctx.new_step_id(),
        timestamp=datetime.now(UTC),
        event_type=EventType.LLM_CALL,
        llm_request_digest=request_digest,
        llm_response_digest=response_digest,
        model=response.model,
    )
    self.trace_writer.write(event)
```

#### request_digest 与 response_digest

```python
# src/arcana/gateway/providers/openai_compatible.py:163-166
request_digest = canonical_hash({
    "messages": messages,
    "config": config.model_dump(),
})
```

Arcana 不在 Trace 中存储完整的请求/响应内容（那可能非常大），而是存储**规范化哈希摘要**（canonical hash）。这有两个好处：

1. **节省存储**：一个 SHA-256 摘要只有 16 字符，而一次 LLM 调用的完整内容可能数千字符
2. **可审计性**：相同的输入始终产生相同的摘要，方便检测重复调用或验证缓存

摘要使用 `utils/hashing.py` 中的 `canonical_hash()` 函数生成——对输入做排序后的 JSON 序列化，再取 SHA-256 的前 16 字符。

#### 追踪事件包含的信息

每个 `TraceEvent` 记录了：
- `run_id` / `task_id` / `step_id`：完整的调用链路标识
- `timestamp`：UTC 时间戳
- `event_type`：固定为 `EventType.LLM_CALL`
- `llm_request_digest`：请求摘要
- `llm_response_digest`：响应摘要
- `model`：实际使用的模型名

这使得事后可以完整还原 Agent 的每一步决策过程。

---

## 7. 设计模式总结

Arcana 的 Gateway 层运用了多个经典设计模式，它们的组合产生了优雅而强大的架构：

### 策略模式（Strategy Pattern）

`ModelGateway` ABC 定义了统一接口，每个 Provider 是一个可互换的策略。Registry 在运行时选择具体策略。

```
ModelGateway (Strategy)
├── OpenAICompatibleProvider (ConcreteStrategy)
│   ├── DeepSeekProvider
│   └── GeminiProvider
```

**价值**：上层代码（Agent Runtime）只依赖 `ModelGateway` 接口，完全不感知具体实现。

### 适配器模式（Adapter Pattern）

`OpenAICompatibleProvider` 将各家供应商的 API 适配到 Arcana 内部的 `LLMRequest`/`LLMResponse` 契约。`_convert_messages()` 就是典型的适配逻辑。

```
Arcana 内部模型                         外部 API
LLMRequest ──→ _convert_messages() ──→ OpenAI 格式
LLMResponse ←── 响应解析 ←────────── API 返回
```

**价值**：内部数据模型的变更不影响外部 API 调用，反之亦然。

### 注册表模式（Registry Pattern）

`ModelGatewayRegistry` 维护了 Provider 名称到实例的映射，支持动态注册/注销。

```python
registry.register("gemini", provider)    # 注册
registry.get("gemini")                   # 查找
registry.unregister("gemini")            # 注销
registry.list_providers()                # 列举
```

**价值**：Provider 的生命周期管理集中在一处，支持运行时动态变更。

### 工厂模式（Factory Pattern）

`create_deepseek_provider()`、`create_gemini_provider()`、`create_ollama_provider()` 是工厂函数，封装了 Provider 的复杂构造逻辑。

**价值**：用户无需记住每家供应商的 base_url、默认模型等细节。

### 责任链模式（Chain of Responsibility）

fallback chain 实现了责任链——请求依次传递给链上的 Provider，直到有一个成功处理。

```
请求 → Gemini → (失败) → DeepSeek → (失败) → Ollama → 响应
```

**价值**：故障转移逻辑对调用者透明，新增 fallback 只需修改配置。

### 空对象模式（Null Object Pattern）

`BudgetTracker.from_budget(None)` 返回一个无限制的 tracker，而非 `None`。

**价值**：避免遍地的空值检查，简化调用方代码。

---

## 8. 生产注意事项

### 限流与退避（Rate Limiting & Backoff）

当前实现在遇到 429 错误时直接切换到 fallback provider。在生产环境中，建议增加以下增强：

```python
# 建议的指数退避策略
import asyncio

async def generate_with_retry(provider, request, config, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await provider.generate(request, config)
        except ProviderError as e:
            if not e.retryable or attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait_time)
```

### 熔断器（Circuit Breaker）

频繁失败的 Provider 应该被暂时"熔断"，避免浪费时间和请求配额：

```python
# 概念性的熔断器设计
class CircuitBreaker:
    """连续失败 N 次后暂停 M 秒"""
    failure_threshold: int = 5
    recovery_timeout_ms: int = 30000
    state: Literal["closed", "open", "half_open"] = "closed"
```

当前 Arcana 未内置熔断器，但架构支持在 `Registry.generate()` 中轻松添加。

### 成本优化策略

1. **分级模型选择**：简单任务用便宜模型（如 `gemini-2.0-flash-lite`），复杂任务用强力模型（如 `gemini-1.5-pro`）
2. **Token 预估**：利用 `BudgetTracker.can_afford()` 在调用前预估成本
3. **缓存**：对相同输入的请求做缓存（`request_digest` 天然支持缓存键）
4. **Prompt 压缩**：在 token 预算紧张时自动截断历史消息

### 供应商多样性

避免所有 fallback 都依赖同一家基础设施提供商。推荐配置：

```python
# 好的 fallback 配置——跨云、跨地域
registry.set_fallback_chain("gemini", ["deepseek", "ollama"])

# 差的 fallback 配置——都在同一家云
registry.set_fallback_chain("openai", ["azure_openai"])
```

### 安全性

1. **API Key 管理**：绝不在代码中硬编码 API key，使用环境变量或密钥管理服务
2. **请求日志脱敏**：Trace 中只存储摘要（digest），不存储原始 prompt——这是合理的安全设计
3. **输入验证**：`ModelConfig` 使用 Pydantic 的 `Field` 约束（如 `temperature: float = Field(ge=0.0, le=2.0)`）防止非法参数

### 监控建议

利用已有的追踪基础设施：
- 监控每家 Provider 的**成功率**和**延迟分布**
- 设置 `cost_usd` 的阈值告警
- 跟踪 fallback 触发频率——频繁触发说明主 Provider 不稳定
- 利用 `health_check_all()` 做定期探活

---

## 9. 本章小结

### 核心收获

1. **一个类通吃多家供应商**：`OpenAICompatibleProvider` 利用 OpenAI 兼容 API 这一事实标准，一个实现对接 8+ 家供应商。这是"少即是多"的典范——代码量极小，但覆盖面极广。

2. **契约先行的威力**：`LLMRequest`/`LLMResponse`/`ModelConfig` 这些 Pydantic 模型在 `contracts/llm.py` 中定义，Gateway 层只是它们的消费者。这意味着未来可以用 Go/Rust 重写 Gateway 而不影响上层 Agent。

3. **防御性设计无处不在**：
   - `BudgetTracker` 防止成本失控
   - `ProviderError.retryable` 智能区分可重试/不可重试错误
   - fallback chain 保障服务可用性
   - Trace digest 保障可审计性

4. **扩展友好**：添加新供应商只需写一个薄包装器子类或调用工厂函数，无需修改任何已有代码。

### 与上下章的关系

- **上一章（Contracts + Trace）**：定义了 `LLMRequest`、`LLMResponse`、`TraceEvent` 等数据模型，本章的 Gateway 是这些契约的第一个核心消费者
- **下一章（Agent Runtime）**：Agent 运行时将通过 `ModelGatewayRegistry.generate()` 调用 LLM，Gateway 层为 Agent 提供了可靠、可控、可观测的 LLM 访问能力

---

> **下一章预告**：第三章将进入 Agent Runtime——状态机驱动的 Agent 执行引擎。我们将看到 Agent 如何利用 Gateway 层进行 LLM 调用、执行 Tool、管理状态，以及如何通过 Policy 和 Reducer 实现灵活的行为控制。
