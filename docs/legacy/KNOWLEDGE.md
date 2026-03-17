# Agent 平台工程知识体系

> 面向高端 Agent 工程师的知识点整理，含面试要点与扩展延伸

---

## 目录

1. [核心概念](#1-核心概念)
2. [Trace 系统](#2-trace-系统审计与可追溯)
3. [Model Gateway](#3-model-gateway模型网关)（含 Structured Output、Streaming、Context 管理）
4. [Tool Gateway](#4-tool-gateway工具网关)
5. [State 管理](#5-state-管理状态机)
6. [RAG 系统](#6-rag-系统)
7. [Multi-Agent](#7-multi-agent-协作)
8. [生产治理](#8-生产治理)
9. [面试高频题](#9-面试高频题)

---

## 1. 核心概念

### 1.1 "先平台后框架" 原则

```
❌ 错误做法：直接用 LangChain 写 demo
✅ 正确做法：先建平台（Gateway、Trace、治理），框架只是加速器
```

**为什么？**
- 框架更新快，被 lock-in 风险高
- 生产需要的能力（审计、预算、权限）框架不提供
- 出问题时需要深入底层排查

### 1.2 Contract-First（契约优先）

```python
# 先定义接口，再写实现
class LLMRequest(BaseModel):
    messages: list[Message]
    response_schema: dict | None = None
    tools: list[dict] | None = None
    budget: Budget | None = None
```

**好处**：
| 优势 | 说明 |
|------|------|
| 可替换 | 换 Provider 只改实现，不改上层 |
| 可测试 | Mock 只需符合 Schema |
| 可迁移 | Go/Rust 重写只需实现相同 Schema |
| 可验证 | Pydantic 自动校验 |

### 1.3 核心模块依赖图

```
Week 1: Contracts + Trace（基础）
    ↓
Week 2: Model Gateway（模型接入）
    ↓
Week 3-4: Agent Runtime（状态机执行）
    ↓
Week 5: Tool Gateway（工具权限）
    ↓
Week 6: RAG（知识检索）
    ↓
Week 7: Memory（记忆系统）
    ↓
Week 8: Plan-Execute（规划执行）
    ↓
Week 9-10: Orchestrator（任务调度）
    ↓
Week 11-12: Observability + Multi-Agent
```

---

## 2. Trace 系统（审计与可追溯）

### 2.1 为什么 Trace 是第一优先级？

**面试要点**：如果一个 Agent 出错了，你怎么排查？

```
没有 Trace：
- 只能看日志猜测
- 无法复现问题
- 不知道花了多少钱

有 Trace：
- 按 run_id 回放每一步
- 看到完整的 LLM 输入输出
- 知道在哪一步、为什么停止
- 成本、Token 消耗一目了然
```

### 2.2 TraceEvent 核心字段

```python
class TraceEvent:
    # 标识符
    run_id: str      # 本次运行 ID
    task_id: str     # 任务 ID（可选）
    step_id: str     # 步骤 ID
    timestamp: datetime

    # 分类
    role: AgentRole           # system | planner | executor | critic
    event_type: EventType     # llm_call | tool_call | state_change | error

    # 状态摘要（关键！）
    state_before_hash: str    # 执行前状态 hash
    state_after_hash: str     # 执行后状态 hash

    # LLM 调用
    llm_request_digest: str   # 请求 hash
    llm_response_digest: str  # 响应 hash
    model: str

    # 工具调用
    tool_call: ToolCallRecord

    # 预算
    budgets: BudgetSnapshot

    # 停止原因
    stop_reason: StopReason
```

### 2.3 Canonical Hash 规范

**面试要点**：为什么不直接存原始内容？为什么要 Canonical？

```python
def canonical_hash(obj: Any, length: int = 16) -> str:
    """
    规范化 JSON 哈希
    - sort_keys=True：key 按字典序
    - separators=(',', ':')：无多余空格
    - 浮点数归一化到 6 位小数
    - SHA-256 截断前 16 字符
    """
    normalized = normalize_floats(obj)
    json_str = json.dumps(
        normalized,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    )
    return hashlib.sha256(json_str.encode()).hexdigest()[:length]
```

**为什么 16 字符？**
- 16 字符 hex = 64 bit = 2^64 种可能
- 对于 Trace 场景（百万级记录）碰撞概率可忽略
- 可读性好，适合日志和调试
- **注意**：生日攻击下约 2^32 次尝试有 50% 碰撞概率，如需安全敏感场景（如签名验证），建议使用 32 字符（128 bit）或完整 64 字符

### 2.4 Stop Reasons（停止原因枚举）

| 原因 | 说明 | 处理策略 |
|------|------|----------|
| `goal_reached` | 目标完成 | 正常结束 |
| `max_steps` | 达到步数上限 | 可能需要调大或优化 |
| `max_time` | 超时 | 检查是否卡住 |
| `max_cost` | 超预算 | 触发降级策略 |
| `max_tokens` | Token 耗尽 | 触发降级策略 |
| `no_progress` | 连续无进展 | 检测死循环 |
| `error` | 不可恢复错误 | 需要人工介入 |
| `tool_blocked` | 工具被拦截 | 权限问题 |

**面试要点**：如何检测 no_progress？
- 连续 N 步输出相同
- 重复调用相同工具 + 相同参数
- state_after_hash 连续相同

---

## 3. Model Gateway（模型网关）

### 3.1 为什么需要统一网关？

```
直接调用：
openai.chat.completions.create(...)
deepseek.chat.completions.create(...)
gemini.generate_content(...)

问题：
- 接口不统一
- 切换模型要改代码
- 无法统一计费
- 无法做 fallback
```

```
通过 Gateway：
gateway.generate(request, config)

好处：
- 统一接口
- 可插拔 Provider
- 统一预算控制
- 自动 fallback
- 统一 trace
```

### 3.2 Provider 抽象

```python
class ModelGateway(ABC):
    @abstractmethod
    async def generate(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None,
    ) -> LLMResponse:
        ...
```

**关键设计**：用 OpenAI 兼容格式统一多厂商

```python
# 一套代码，不同配置
deepseek = OpenAICompatibleProvider(
    provider_name="deepseek",
    base_url="https://api.deepseek.com",
    api_key="sk-xxx",
)

gemini = OpenAICompatibleProvider(
    provider_name="gemini",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key="AIza-xxx",
)
```

### 3.3 Fallback 链

```python
registry = ModelGatewayRegistry()
registry.register("deepseek", deepseek_provider)
registry.register("gemini", gemini_provider)
registry.set_fallback_chain("deepseek", ["gemini"])

# deepseek 挂了自动切 gemini
response = await registry.generate(request, config, use_fallback=True)
```

**面试要点**：什么时候触发 fallback？
- `retryable=True` 的错误（429 限流、503 服务不可用）
- 非 retryable 错误不 fallback（400 参数错误）

### 3.4 预算追踪

```python
class BudgetTracker:
    max_tokens: int | None
    max_cost_usd: float | None
    max_time_ms: int | None

    def check_budget(self) -> None:
        """超限时抛出 BudgetExceededError"""
        if self.tokens_used >= self.max_tokens:
            raise BudgetExceededError("Token budget exceeded")
```

**面试要点**：预算超限后怎么处理？
1. 立即停止（保守）
2. 触发降级（用更便宜的模型）
3. 通知用户确认

### 3.5 LLM 调用确定性

**面试要点**：如何让 LLM 调用可复现？

```python
# 开发/测试环境
config = ModelConfig(
    model_id="deepseek-chat",
    temperature=0.0,  # 关闭随机性
    seed=42,          # 固定种子（如果模型支持）
)

# 生产环境
# 1. 记录 temperature/seed 到 trace
# 2. replay 时读取并复用

# Cache key 计算
cache_key = canonical_hash({
    "model": config.model_id,
    "messages": request.messages,
    "tools": request.tools,
    "temperature": config.temperature,
    "seed": config.seed,
})
```

### 3.6 Structured Output（结构化输出）

**面试要点**：如何确保 LLM 返回有效 JSON？

```python
# 方案 1：response_format（仅保证 JSON 格式）
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={"type": "json_object"},  # 仅保证是 JSON
)

# 方案 2：Structured Outputs（OpenAI 独有，强制符合 Schema）
from pydantic import BaseModel

class SearchResult(BaseModel):
    query: str
    results: list[str]
    confidence: float

response = await client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=SearchResult,  # 强制符合 Schema
)
result = response.choices[0].message.parsed  # 类型安全
```

**各厂商支持情况**：
| 能力 | OpenAI | Anthropic | DeepSeek | Gemini |
|------|--------|-----------|----------|--------|
| JSON Mode | ✅ | ✅ (tool_use) | ✅ | ⚠️ |
| Schema 强制 | ✅ Structured Outputs | ❌ | ❌ | ❌ |

**兜底策略**：
```python
async def parse_json_response(response: str, schema: type[BaseModel]) -> BaseModel:
    """尝试解析 JSON，失败时重试"""
    try:
        data = json.loads(response)
        return schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        # 可选：让 LLM 修正
        # 或：返回默认值 / 抛出错误
        raise ParseError(f"Failed to parse response: {e}")
```

### 3.7 Streaming（流式响应）

**面试要点**：为什么要用流式？

```
非流式：用户等待 5-30 秒看到完整响应（体验差）
流式：立即开始看到文字输出（体验好）
```

**流式处理示例**：
```python
async def stream_response(request: LLMRequest, config: ModelConfig):
    """流式生成，逐 chunk 返回"""
    stream = await client.chat.completions.create(
        model=config.model_id,
        messages=request.messages,
        stream=True,
    )

    full_content = ""
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
            yield content  # 逐步返回给前端

    # 流式结束后记录 Trace
    # 注意：流式响应通常不返回 usage，需要自行估算或额外请求
```

**流式 + Trace 的挑战**：
```python
# 问题：流式响应不返回 token 统计
# 解决方案：
# 1. 事后估算（不精确）
# 2. 部分厂商支持 stream_options.include_usage（OpenAI）
# 3. 用 tiktoken 本地计算（仅限 OpenAI 模型）

stream = await client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    stream=True,
    stream_options={"include_usage": True},  # OpenAI 支持
)
```

### 3.8 Context Window 管理

**面试要点**：消息太长超过上下文窗口怎么办？

```python
class ContextManager:
    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens
        self.reserve_for_response = 4096  # 预留给响应

    def truncate_messages(
        self,
        messages: list[Message],
        strategy: str = "sliding_window"
    ) -> list[Message]:
        """截断消息以适应上下文窗口"""
        available = self.max_tokens - self.reserve_for_response

        if strategy == "sliding_window":
            # 保留 system + 最近 N 条消息
            return self._sliding_window(messages, available)
        elif strategy == "summarize":
            # 压缩历史为摘要
            return self._summarize_history(messages, available)
        elif strategy == "smart_trim":
            # 智能裁剪：保留重要消息
            return self._smart_trim(messages, available)

    def _sliding_window(self, messages: list[Message], limit: int) -> list[Message]:
        """滑动窗口：保留 system 消息 + 最近的对话"""
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        # 从最新往回取，直到达到 token 限制
        result = []
        current_tokens = sum(estimate_tokens(m.content) for m in system_msgs)

        for msg in reversed(other_msgs):
            msg_tokens = estimate_tokens(msg.content)
            if current_tokens + msg_tokens > limit:
                break
            result.insert(0, msg)
            current_tokens += msg_tokens

        return system_msgs + result
```

**各模型上下文窗口**（2024-2025）：
| 模型 | 上下文窗口 | 备注 |
|------|-----------|------|
| GPT-4o | 128K | 输出最多 16K |
| Claude 3.5 | 200K | 输出最多 8K |
| DeepSeek V3 | 64K | |
| Gemini 1.5 Pro | 2M | 超长上下文 |

---

## 4. Tool Gateway（工具网关）

### 4.1 为什么工具需要网关？

**安全问题**：
- 注入攻击：Agent 被诱导执行危险命令
- 越权调用：调用没有权限的工具
- 重复执行：写操作被重复调用

### 4.2 工具契约

```python
class ToolSpec:
    name: str
    description: str
    input_schema: dict      # JSON Schema
    output_schema: dict

    # 关键字段
    side_effect: SideEffect    # none | read | write
    requires_confirmation: bool
    capabilities: list[str]    # 所需权限
    idempotency_key: str       # 幂等键（写操作必须）
```

### 4.3 Side Effect 分类

| 类型 | 说明 | 保护措施 |
|------|------|----------|
| `none` | 无副作用（计算类） | 无需保护 |
| `read` | 只读（查询类） | 限流 |
| `write` | 写操作（修改类） | 确认 + 幂等 |

**面试要点**：为什么 write 操作需要幂等？

```
场景：Agent 调用 create_file 创建文件
问题：网络超时，不知道是否成功
重试：再调一次 create_file

没有幂等：创建两个文件
有幂等：通过 idempotency_key 识别重复，返回缓存结果
```

### 4.4 Capability 权限系统

```python
# 工具定义
file_write_tool = ToolSpec(
    name="file_write",
    capabilities=["fs:write", "fs:read"],
)

# Agent 被授权的能力
agent_capabilities = ["fs:read"]  # 只有读权限

# 调用时检查
if not all(cap in agent_capabilities for cap in tool.capabilities):
    raise AuthorizationError("Missing capability: fs:write")
```

**标准能力列表**：
| 能力 | 说明 |
|------|------|
| `fs:read` | 文件系统读 |
| `fs:write` | 文件系统写 |
| `net:http` | HTTP 请求 |
| `db:read` | 数据库读 |
| `db:write` | 数据库写 |
| `exec:shell` | Shell 执行 |

### 4.5 错误分类与重试

```python
class ErrorType(Enum):
    RETRYABLE = "retryable"         # 临时错误，自动重试
    NON_RETRYABLE = "non_retryable" # 永久错误，直接失败
    REQUIRES_HUMAN = "requires_human" # 需人工介入
```

**重试策略**：
```python
for attempt in range(max_retries):
    try:
        result = await tool.execute(args)
        return result
    except ToolError as e:
        if e.error_type == ErrorType.NON_RETRYABLE:
            raise  # 不重试
        if e.error_type == ErrorType.REQUIRES_HUMAN:
            return await escalate_to_human(e)  # 升级
        await asyncio.sleep(retry_delay * (2 ** attempt))  # 指数退避
```

---

## 5. State 管理（状态机）

### 5.1 Agent 状态模型

```python
class AgentState:
    # 标识
    run_id: str
    task_id: str | None

    # 执行状态
    status: ExecutionStatus  # pending | running | paused | completed | failed
    current_step: int
    max_steps: int

    # 目标与计划
    goal: str
    current_plan: list[str]
    completed_steps: list[str]

    # 工作记忆
    working_memory: dict[str, Any]

    # 对话历史
    messages: list[dict]

    # 预算消耗
    tokens_used: int
    cost_usd: float
    elapsed_ms: int

    # 错误追踪
    consecutive_errors: int
    consecutive_no_progress: int
```

### 5.2 状态转换图

```
    ┌─────────┐
    │ pending │
    └────┬────┘
         │ start()
    ┌────▼────┐
    │ running │◀──────┐
    └────┬────┘       │
         │            │ resume()
    ┌────┴────┬───────┴──┐
    │         │          │
    ▼         ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│complete│ │ failed │ │ paused │
└────────┘ └────────┘ └────────┘
```

### 5.3 Checkpoint 机制

**面试要点**：如何实现断点续跑？

```python
class StateSnapshot:
    run_id: str
    step_id: str
    timestamp: datetime
    state_hash: str      # 状态完整性校验
    state: AgentState    # 完整状态
    is_resumable: bool
```

**Checkpoint 触发时机**：
1. 每步完成后
2. 错误恢复前
3. 预算到达阈值（50%, 75%, 90%）
4. 定时（每 N 分钟）

### 5.4 Resume 与 Replay

```python
# Resume：从 checkpoint 继续执行
snapshot = load_snapshot(run_id, step_id)
assert canonical_hash(snapshot.state) == snapshot.state_hash  # 完整性校验
agent.resume(snapshot.state)

# Replay：重放历史执行（用于调试）
events = trace_reader.read_events(run_id)
for event in events:
    if event.event_type == EventType.LLM_CALL:
        # 用缓存的响应，不真正调用 LLM
        response = cache.get(event.llm_request_digest)
```

**面试要点**：Resume 和 Replay 的区别？
| 对比 | Resume | Replay |
|------|--------|--------|
| 目的 | 继续执行 | 调试/复现 |
| LLM 调用 | 真实调用 | 用缓存 |
| 工具调用 | 真实执行 | 可选真实/缓存 |
| 结果 | 可能不同 | 确定性复现 |

---

## 6. RAG 系统

### 6.1 RAG 核心流程

```
Query → Rewrite → Retrieve → Rerank → Context → Generate → Verify
  ↓        ↓         ↓          ↓         ↓         ↓         ↓
用户问题  优化查询  向量检索   重排序   组装上下文  LLM生成   验证引用
```

### 6.2 Document 与 Chunk

```python
class Document:
    id: str
    source: str           # 来源（URL/文件路径）
    timestamp: datetime
    tags: list[str]
    content: str

class Chunk:
    id: str
    document_id: str
    content: str
    start_offset: int
    end_offset: int
    embedding: list[float]
```

**分块策略**：
| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `fixed` | 固定 token 数 | 通用文本 |
| `semantic` | 按句子边界 | 文章 |
| `paragraph` | 按段落 | 结构化文档 |
| `recursive` | 层级分割 | 代码、长文档 |

### 6.3 Retrieval 与 Rerank

```python
class RetrievalResult:
    doc_id: str
    chunk_id: str
    score: float         # 0-1 相关度分数
    snippet: str
    citation: Citation   # 引用信息
```

**Rerank 方案**（延迟为参考值，实际因硬件/网络而异）：
| 方案 | 特点 | 延迟（参考） |
|------|------|-------------|
| `bge-reranker-base` | 本地部署，约 400MB | ~50ms (GPU) / ~200ms (CPU) |
| `cohere-rerank` | 云 API，高质量 | ~100-300ms（取决于网络） |
| `bm25` | 无模型依赖，纯算法 | ~5ms |
| `cross-encoder` | 精度高但慢 | ~100-500ms |

### 6.4 Citation（引用）是一等公民

**面试要点**：为什么引用很重要？

```
没有引用：
- 用户：法国首都是哪？
- Agent：巴黎
- 问题：怎么验证？可能是幻觉

有引用：
- Agent：巴黎 [1]
- [1] 来源：Wikipedia - France, 检索时间：2024-01-15
- 好处：可追溯、可验证、可信赖
```

```python
class RAGAnswer:
    content: str
    citations: list[Citation]      # 必须有引用
    retrieved_chunks: list[str]    # 用了哪些 chunk

class CitationVerifier:
    def verify(self, answer: RAGAnswer) -> VerificationResult:
        # 1. 检查所有声明是否有引用
        # 2. 验证引用确实存在于检索结果中
        # 3. 检查引用相关性
        # 4. 标记无支撑的声明
```

### 6.5 RAG 评测指标

**检索指标**：
| 指标 | 说明 |
|------|------|
| `recall@k` | top-k 中包含相关文档的比例 |
| `precision@k` | top-k 中相关文档的精确率 |
| `mrr` | 平均倒数排名 |
| `ndcg` | 归一化折损累积增益 |

**回答指标**：
| 指标 | 说明 |
|------|------|
| `citation_coverage` | 声明有引用的比例 |
| `evidence_relevance` | 引用与声明的相关性 |
| `hallucination_rate` | 无支撑声明的比例 |
| `answer_faithfulness` | 回答与来源的一致性 |

---

## 7. Multi-Agent 协作

### 7.1 角色分工

| 角色 | 职责 | 输出 |
|------|------|------|
| **Planner** | 分解任务、制定计划 | 结构化计划 |
| **Executor** | 执行具体步骤 | 执行结果 |
| **Critic** | 验证结果、提出修正 | 验证报告 |

### 7.2 协作协议

```python
# Planner 输出
class Plan:
    steps: list[PlanStep]
    dependencies: dict[str, list[str]]
    budget_allocation: dict[str, float]
    acceptance_criteria: list[str]

# Executor 输出
class ExecutionResult:
    step_id: str
    success: bool
    output: Any
    artifacts: list[Artifact]

# Critic 输出
class VerificationReport:
    passed: bool
    issues: list[Issue]
    recommendations: list[str]
    should_replan: bool
```

### 7.3 仲裁机制

**面试要点**：多 Agent 意见冲突怎么办？

1. **投票**：多数决定
2. **优先级**：Critic 可以否决 Executor
3. **升级**：无法解决时触发 HITL（Human-In-The-Loop）

```python
if critic_report.should_replan:
    if replan_count < max_replans:
        return await planner.replan(critic_report.issues)
    else:
        return await escalate_to_human(critic_report)
```

---

## 8. 生产治理

### 8.1 配额与背压

```python
class QuotaManager:
    def check_quota(self, user_id: str, resource: str) -> bool:
        """检查用户配额"""
        current = self.get_usage(user_id, resource)
        limit = self.get_limit(user_id, resource)
        return current < limit

    def apply_backpressure(self, queue_depth: int) -> Action:
        """根据队列深度决定策略"""
        if queue_depth > HIGH_THRESHOLD:
            return Action.REJECT
        elif queue_depth > MEDIUM_THRESHOLD:
            return Action.DELAY
        else:
            return Action.ACCEPT
```

### 8.2 降级策略

| 触发条件 | 降级策略 |
|----------|----------|
| 主模型不可用 | 切换到备用模型 |
| 预算即将耗尽 | 用更便宜的模型 |
| 延迟过高 | 减少检索数量 |
| 系统过载 | 拒绝新请求 |

### 8.3 Eval Harness（评测系统）

```python
class EvalHarness:
    def run_regression(self, test_suite: str) -> EvalReport:
        """回归测试"""
        results = []
        for case in self.load_cases(test_suite):
            result = await self.run_case(case)
            results.append(result)
        return self.aggregate(results)

    def check_gate(self, report: EvalReport) -> bool:
        """发布门禁"""
        return (
            report.success_rate >= 0.95 and
            report.p95_latency_ms <= 5000 and
            report.hallucination_rate <= 0.05
        )
```

### 8.4 对抗测试集

| 测试类型 | 目的 |
|----------|------|
| 注入测试 | 诱导 Agent 执行危险操作 |
| 越权测试 | 尝试调用无权限工具 |
| 循环测试 | 诱导无限循环 |
| 超时测试 | 模拟工具长时间无响应 |
| 检索误导 | 提供错误的检索结果 |

---

## 9. 面试高频题

### Q1: Agent 出错了怎么排查？

**回答框架**：
1. 通过 `run_id` 找到对应的 trace 文件
2. 查看 `stop_reason` 确定停止原因
3. 按时间顺序回溯每个 step
4. 对比 `state_before_hash` 和 `state_after_hash` 找到异常变化
5. 如果是 LLM 问题，用 `llm_request_digest` 定位具体请求
6. 如果需要复现，使用 replay 机制

### Q2: 如何防止 Agent 死循环？

**回答框架**：
```python
# 多重防护
1. max_steps 硬限制
2. max_time 超时限制
3. max_cost 成本限制
4. no_progress 检测：
   - 连续 N 步 state_hash 相同
   - 重复相同的 tool call
   - 输出内容重复率过高
```

### Q3: 如何做模型路由和降级？

**回答框架**：
```
1. 正常路由：
   - 按任务类型选模型（复杂任务用大模型）
   - 按成本预算选模型

2. 自动降级：
   - 主模型超时/错误 → fallback 链
   - 预算不足 → 切换便宜模型
   - 使用 two-stage：先小模型评估，必要时再用大模型

3. 实现：
   - ModelGatewayRegistry 管理多 Provider
   - set_fallback_chain 设置降级顺序
   - retryable 标记决定是否 fallback
```

### Q4: 为什么工具需要幂等性？

**回答框架**：
```
场景：网络不稳定，调用工具后不确定是否成功

没有幂等：
- 重试可能导致重复执行（创建两个文件、发两封邮件）

有幂等：
- 通过 idempotency_key 识别重复请求
- 重复请求返回缓存结果
- 实现：在 Gateway 层维护 key→result 映射
```

### Q5: RAG 如何保证回答可信？

**回答框架**：
```
1. 强制引用：
   - 每个声明必须有 citation
   - CitationVerifier 验证引用有效性

2. 可追溯：
   - RAG Trace 记录检索过程
   - 保存 query、命中 chunks、scores

3. 评测：
   - citation_coverage：声明覆盖率
   - hallucination_rate：幻觉率
   - 发布门禁卡控指标

4. 降级：
   - 检索为空时明确告知"无法回答"
   - 低置信度时触发人工确认
```

### Q6: 如何设计多 Agent 协作？

**回答框架**：
```
1. 角色分工：
   - Planner：分解任务
   - Executor：执行步骤
   - Critic：验证结果

2. 协作协议：
   - 输入输出 schema 明确定义
   - 验收条件可量化
   - 失败升级路径清晰

3. 仲裁机制：
   - Critic 可阻断危险操作
   - 冲突时投票或升级 HITL

4. 资源共享：
   - Memory 读写隔离
   - 预算分配策略
```

### Q7: 生产环境如何保证稳定性？

**回答框架**：
```
1. 可观测性：
   - 全链路 Trace
   - 结构化日志
   - 关键指标监控（成功率、延迟、成本）

2. 治理能力：
   - 配额管理
   - 背压策略
   - 降级开关

3. 质量门禁：
   - 回归测试
   - 对抗集
   - 发布前评测

4. 故障恢复：
   - Checkpoint/Resume
   - Replay 复现
   - 自动告警
```

---

## 延伸学习资源

### 论文
- ReAct: Synergizing Reasoning and Acting in Language Models
- Toolformer: Language Models Can Teach Themselves to Use Tools
- Self-RAG: Learning to Retrieve, Generate, and Critique

### 开源项目
- LangChain / LangGraph（学习编排思想）
- DSPy（学习优化思想）
- AutoGPT（学习自主 Agent 设计）

### 技术博客
- Anthropic Claude 文档（最佳实践）
- OpenAI Cookbook（实战示例）
- Pinecone / Weaviate 博客（向量检索）

---

## 10. 技术栈选型与原因

### 10.1 核心依赖选型

| 技术 | 选择 | 原因 | 替代方案对比 |
|------|------|------|--------------|
| **数据校验** | Pydantic v2 | 性能比 v1 快 5-50 倍；原生支持 JSON Schema；类型提示友好 | dataclasses（无校验）、attrs（生态弱） |
| **HTTP 客户端** | httpx | 原生 async；API 与 requests 兼容；HTTP/2 支持 | requests（无 async）、aiohttp（API 复杂） |
| **LLM SDK** | openai | 事实标准；多厂商兼容；维护活跃 | 直接 HTTP（繁琐）、litellm（多一层抽象） |
| **环境变量** | python-dotenv | 简单可靠；开发环境友好 | environs（过度封装） |
| **包管理** | uv | 比 pip 快 10-100 倍（官方 benchmark）；lockfile 原生支持 | poetry（慢）、pip（无 lock） |
| **Lint** | ruff | 比 flake8+isort+black 快 10-100 倍（官方 benchmark）；单一工具 | pylint（慢）、black+flake8（多工具） |
| **类型检查** | mypy strict | 严格模式抓更多 bug；生态成熟 | pyright（更快，VS Code 默认） |

### 10.2 为什么用 OpenAI SDK 而不是直接 HTTP？

```python
# ❌ 直接 HTTP（繁琐）
async with httpx.AsyncClient() as client:
    response = await client.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "deepseek-chat", "messages": [...]}
    )
    data = response.json()
    # 还要处理错误码、重试、streaming...

# ✅ OpenAI SDK（统一）
client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
response = await client.chat.completions.create(
    model="deepseek-chat",
    messages=[...]
)
# SDK 自动处理认证、错误、类型提示
```

**关键洞察**：大多数 LLM 厂商都兼容 OpenAI 格式，所以用一套 SDK + 换 `base_url` 就能接入多厂商。

### 10.3 为什么 Pydantic v2 而不是 v1？

```python
# v2 新特性
from pydantic import BaseModel, field_validator, model_validator

class LLMRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.7

    # v2 新语法：field_validator
    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: float) -> float:
        if not 0 <= v <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return v

    # v2 新语法：model_validator
    @model_validator(mode="after")
    def check_messages_not_empty(self) -> "LLMRequest":
        if not self.messages:
            raise ValueError("messages cannot be empty")
        return self
```

**性能对比**（官方 benchmark，实际因场景而异）：
- 模型实例化：快 5-50 倍
- JSON 序列化：快 2-3 倍
- 内存占用：减少约 30%

---

## 11. 常见问题与踩坑记录

### 11.1 异步编程常见坑

#### 坑 1：忘记 await
```python
# ❌ 错误：忘记 await，返回 coroutine 对象
response = client.chat.completions.create(...)  # 这是 coroutine！
print(response.content)  # AttributeError

# ✅ 正确
response = await client.chat.completions.create(...)
```

#### 坑 2：在同步函数中调用异步
```python
# ❌ 错误：直接在同步上下文调用
def sync_function():
    result = await async_call()  # SyntaxError

# ✅ 方案 1：改成 async
async def async_function():
    result = await async_call()

# ✅ 方案 2：用 asyncio.run（入口点）
import asyncio
result = asyncio.run(async_call())
```

#### 坑 3：并发控制
```python
# ❌ 错误：无限并发可能触发限流
tasks = [call_llm(msg) for msg in messages]  # 1000 个并发请求！
results = await asyncio.gather(*tasks)

# ✅ 正确：用 Semaphore 限制并发
semaphore = asyncio.Semaphore(10)  # 最多 10 并发

async def limited_call(msg):
    async with semaphore:
        return await call_llm(msg)

tasks = [limited_call(msg) for msg in messages]
results = await asyncio.gather(*tasks)
```

### 11.2 JSON 序列化边界 Case

#### 坑 1：浮点数精度不一致
```python
# 问题：同一个数在不同环境可能序列化不同
import json
data = {"score": 0.1 + 0.2}  # 0.30000000000000004

# ❌ 不同机器可能得到不同 hash
hash1 = hashlib.sha256(json.dumps(data).encode()).hexdigest()

# ✅ 解决：浮点数归一化
def normalize_floats(obj, precision=6):
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: normalize_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_floats(item, precision) for item in obj]
    return obj
```

#### 坑 2：Key 顺序不一致
```python
# Python 3.7+ dict 保持插入顺序，但不同来源的数据顺序可能不同
data1 = {"b": 2, "a": 1}
data2 = {"a": 1, "b": 2}
json.dumps(data1) != json.dumps(data2)  # True！

# ✅ 解决：sort_keys=True
json.dumps(data1, sort_keys=True) == json.dumps(data2, sort_keys=True)  # True
```

### 11.3 API 兼容性差异

#### 各厂商 OpenAI 兼容度差异

| 功能 | OpenAI | DeepSeek | Gemini | Ollama |
|------|--------|----------|--------|--------|
| `seed` 参数 | ✅ | ✅ | ✅ (via generation_config) | ✅ |
| `response_format: json_object` | ✅ | ✅ | ⚠️ 部分模型 | ⚠️ 部分模型 |
| `tools` (function calling) | ✅ | ✅ | ✅ | ⚠️ 取决于模型 |
| `stream` | ✅ | ✅ | ✅ | ✅ |
| Token 统计 | ✅ 精确 | ✅ 精确 | ⚠️ 有时缺失 | ⚠️ 有时缺失 |
| Structured Outputs (JSON Schema) | ✅ | ❌ | ❌ | ❌ |

> **注**：兼容性随版本更新变化，建议以各厂商最新文档为准。

#### 处理策略
```python
# 检测 seed 是否支持
async def generate_with_fallback(self, config: ModelConfig):
    params = {"model": config.model_id, "messages": messages}

    # seed 可能不被支持
    if config.seed is not None and self.supports_seed:
        params["seed"] = config.seed

    try:
        return await self.client.chat.completions.create(**params)
    except Exception as e:
        if "seed" in str(e).lower():
            # 降级：不用 seed
            params.pop("seed", None)
            return await self.client.chat.completions.create(**params)
        raise
```

### 11.4 Token 计算问题

#### 问题：不同模型 tokenizer 不同
```python
# GPT-4 用 cl100k_base
# DeepSeek 用自己的 tokenizer
# Claude 用自己的 tokenizer
# 估算 token 数会有显著偏差

# ✅ 最佳实践：用响应中的实际值
response = await client.chat.completions.create(...)
actual_tokens = response.usage.total_tokens  # 以 API 返回为准

# 预估只用于粗略判断（误差可能 ±50%）
def estimate_tokens(text: str) -> int:
    """
    粗略估算，仅用于预判是否超限，不可用于精确计算。

    经验值（差异很大）：
    - 英文：约 0.75 token/word 或 4 chars/token
    - 中文：约 1.5-2.5 token/字（不同模型差异大）
    """
    # 简单估算：取中间值
    return len(text) // 2

# ✅ 更准确的方式：使用 tiktoken（仅限 OpenAI 模型）
# import tiktoken
# encoding = tiktoken.encoding_for_model("gpt-4")
# tokens = len(encoding.encode(text))
```

> **注意**：Token 估算仅用于粗略预判，计费和限制检查必须以 API 返回的 `usage` 为准。

---

## 12. 调试技巧与最佳实践

### 12.1 Trace 文件分析

#### 查看某次运行的所有步骤
```bash
# 查看 trace 文件
cat traces/{run_id}.jsonl | jq '.'

# 过滤 LLM 调用
cat traces/{run_id}.jsonl | jq 'select(.event_type == "llm_call")'

# 查看 stop_reason
cat traces/{run_id}.jsonl | jq 'select(.stop_reason != null)'

# 统计 token 消耗（需要解析 budgets 字段）
cat traces/{run_id}.jsonl | jq '.budgets.tokens_used' | tail -1
```

#### Python 分析脚本
```python
from arcana.trace.reader import TraceReader

reader = TraceReader("traces")

# 获取运行摘要
summary = reader.get_summary(run_id)
print(f"总事件: {summary['total_events']}")
print(f"LLM 调用: {summary['llm_calls']}")
print(f"工具调用: {summary['tool_calls']}")

# 按类型过滤事件
errors = list(reader.filter_events(run_id, [EventType.ERROR]))
for err in errors:
    print(f"Step {err.step_id}: {err.error}")
```

### 12.2 异步调试技巧

#### 使用 asyncio debug 模式
```python
# 开启 debug 模式，会警告未 await 的 coroutine
import asyncio
asyncio.get_event_loop().set_debug(True)

# 或者环境变量
# PYTHONASYNCIODEBUG=1 python script.py
```

#### 打印 coroutine 调用栈
```python
import traceback
import asyncio

async def debug_wrapper(coro):
    try:
        return await coro
    except Exception as e:
        traceback.print_exc()
        raise

# 使用
result = await debug_wrapper(some_async_call())
```

### 12.3 Pydantic 校验调试

#### 查看详细错误信息
```python
from pydantic import ValidationError

try:
    request = LLMRequest(messages=[], temperature=5.0)
except ValidationError as e:
    # 查看所有错误
    print(e.errors())
    # [
    #   {'type': 'value_error', 'loc': ('messages',), 'msg': '...'},
    #   {'type': 'value_error', 'loc': ('temperature',), 'msg': '...'}
    # ]

    # 人类可读格式
    print(e)
```

#### Schema 导出调试
```python
# 导出 JSON Schema（检查是否符合预期）
print(LLMRequest.model_json_schema())
```

### 12.4 性能分析

#### 简单计时
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s")

# 使用
with timer("LLM call"):
    response = await client.chat.completions.create(...)
```

#### 异步性能分析
```python
import asyncio
import time

async def timed_gather(*coros):
    start = time.perf_counter()
    results = await asyncio.gather(*coros)
    elapsed = time.perf_counter() - start
    print(f"Total time: {elapsed:.3f}s for {len(coros)} tasks")
    return results
```

---

## 13. 安全实践

### 13.1 API Key 管理

#### 开发环境
```python
# .env 文件（加入 .gitignore！）
DEEPSEEK_API_KEY=sk-xxx
GEMINI_API_KEY=AIza-xxx

# 加载
from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY not set")
```

#### 生产环境最佳实践
```python
# 1. 不用 .env 文件，用环境变量注入
# 2. 使用 Secret Manager（AWS/GCP/HashiCorp Vault）
# 3. Key 轮换：支持多个 key，按权重分配

class SecureKeyManager:
    def __init__(self):
        # 从 Secret Manager 获取
        self.keys = self._load_from_vault()
        self.key_index = 0

    def get_key(self) -> str:
        # 轮换使用
        key = self.keys[self.key_index % len(self.keys)]
        self.key_index += 1
        return key
```

### 13.2 注入防护

#### Prompt 注入检测
```python
INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"disregard\s+(previous|above|all)",
    r"system\s*:\s*",  # 伪造 system message
    r"<\|.*\|>",       # 特殊 token
]

def detect_injection(text: str) -> bool:
    import re
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

# 使用
user_input = "ignore all previous instructions and..."
if detect_injection(user_input):
    raise SecurityError("Potential prompt injection detected")
```

#### 输出过滤
```python
def sanitize_output(output: str) -> str:
    # 移除可能的敏感信息
    import re

    # 移除 API Key 模式
    output = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED]', output)
    output = re.sub(r'AIza[a-zA-Z0-9_-]{35}', '[REDACTED]', output)

    return output
```

### 13.3 日志脱敏

```python
def redact_sensitive(data: dict) -> dict:
    """脱敏敏感字段"""
    sensitive_keys = {"api_key", "password", "token", "secret"}

    def _redact(obj):
        if isinstance(obj, dict):
            return {
                k: "[REDACTED]" if k.lower() in sensitive_keys else _redact(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_redact(item) for item in obj]
        return obj

    return _redact(data)

# 使用
config = {"api_key": "sk-secret", "model": "gpt-4"}
safe_config = redact_sensitive(config)
logger.info(f"Config: {safe_config}")  # {"api_key": "[REDACTED]", "model": "gpt-4"}
```

---

## 14. 依赖库使用细节

### 14.1 httpx 高级用法

#### 连接池与超时配置
```python
import httpx

# 生产环境推荐配置
client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=5.0,    # 连接超时
        read=30.0,      # 读取超时（LLM 响应可能慢）
        write=10.0,     # 写入超时
        pool=5.0,       # 连接池获取超时
    ),
    limits=httpx.Limits(
        max_keepalive_connections=20,  # 保持连接数
        max_connections=100,           # 最大连接数
    ),
)
```

#### 重试机制
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
async def call_with_retry(client, url, data):
    response = await client.post(url, json=data)
    response.raise_for_status()
    return response.json()
```

### 14.2 Pydantic v2 高级模式

#### 自定义序列化
```python
from pydantic import BaseModel, field_serializer
from datetime import datetime

class TraceEvent(BaseModel):
    timestamp: datetime

    # 自定义序列化为 ISO 格式
    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        return v.isoformat()
```

#### 泛型模型
```python
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T")

class Response(BaseModel, Generic[T]):
    success: bool
    data: T | None
    error: str | None

# 使用
class UserData(BaseModel):
    name: str
    age: int

response = Response[UserData](success=True, data=UserData(name="test", age=20), error=None)
```

### 14.3 pytest-asyncio 配置

```python
# conftest.py
import pytest

# 自动为所有 async 测试函数应用 asyncio
pytest_plugins = ["pytest_asyncio"]

# 或者在 pyproject.toml
# [tool.pytest.ini_options]
# asyncio_mode = "auto"

# 测试示例
async def test_llm_call():
    provider = create_deepseek_provider(api_key="test")
    response = await provider.generate(request, config)
    assert response.content is not None
```

---

## 15. 进阶学习路径

### 15.1 按周推进的学习重点

| 周次 | 学习重点 | 实践目标 |
|------|----------|----------|
| Week 1-2 | Pydantic、异步编程、JSONL | 完成 Trace + Gateway |
| Week 3-4 | 状态机模式、错误处理 | 完成 Agent Runtime |
| Week 5 | 权限模型、幂等性设计 | 完成 Tool Gateway |
| Week 6 | 向量数据库、Embedding | 完成 RAG v1 |
| Week 7 | 记忆系统、一致性 | 完成 Memory |
| Week 8 | 规划算法、验证机制 | 完成 Plan-Execute |
| Week 9-10 | 分布式系统、队列 | 完成 Orchestrator |
| Week 11-12 | 可观测性、评测体系 | 完成 Eval Harness |
| Week 13-14 | 系统集成、性能调优 | 完成 Demo |

### 15.2 推荐阅读顺序

**入门**：
1. Pydantic v2 官方文档（2 小时）
2. Python asyncio 官方教程（3 小时）
3. httpx 文档（1 小时）

**进阶**：
1. 《Designing Data-Intensive Applications》第 5-9 章（一致性、分布式）
2. OpenAI API 文档 + Cookbook
3. LangChain/LangGraph 源码

**深入**：
1. ReAct / Toolformer / Self-RAG 论文
2. Anthropic Claude 技术博客
3. 生产级 Agent 系统案例（如 Devin、Claude Computer Use）
