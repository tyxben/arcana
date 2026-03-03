# Week 3-4: Agent Runtime 学习要点

> 完成时间：Week 3-4
> 核心目标：掌握 Agent 执行引擎的设计与实现

---

## 目录

1. [核心概念理解](#1-核心概念理解)
2. [Policy-Step-Reducer 模式](#2-policy-step-reducer-模式)
3. [Schema Validation 深度解析](#3-schema-validation-深度解析)
4. [Replay 机制设计](#4-replay-机制设计)
5. [错误分类与处理策略](#5-错误分类与处理策略)
6. [面试高频问题](#6-面试高频问题)
7. [实战经验总结](#7-实战经验总结)

---

## 1. 核心概念理解

### 1.1 什么是 Agent Runtime？

**定义**：Agent Runtime 是 Agent 系统的执行引擎，负责：
- 管理 Agent 的生命周期
- 协调各组件交互
- 维护执行状态
- 处理异常情况

**类比理解**：
```
Agent Runtime 之于 Agent = 操作系统 之于应用程序
```

| 操作系统 | Agent Runtime |
|---------|---------------|
| 进程调度 | Step 执行 |
| 内存管理 | State 管理 |
| 文件系统 | Checkpoint 系统 |
| 异常处理 | Error Handler |
| 系统调用 | Tool Gateway |

### 1.2 为什么需要独立的 Runtime？

**反例：没有 Runtime 的 Agent**
```python
# ❌ 糟糕的实现
while not done:
    response = llm.generate(prompt)
    result = some_tool.execute(response)
    done = check_done(result)
```

**问题**：
- 无法知道为什么停止
- 出错无法复现
- 无法限制资源消耗
- 无法中断和恢复

**正确：使用 Runtime**
```python
# ✅ 良好的实现
agent = Agent(
    policy=ReActPolicy(),
    reducer=DefaultReducer(),
    gateway=gateway,
    config=RuntimeConfig(max_steps=10),
)
state = await agent.run("Task description")
# 可追溯、可控制、可恢复
```

### 1.3 Runtime 的核心职责

```
┌─────────────────────────────────────────┐
│          Agent Runtime                  │
├─────────────────────────────────────────┤
│  1. 执行编排 (Orchestration)            │
│     - 协调 Policy、Executor、Reducer     │
│     - 管理执行流程                       │
│                                         │
│  2. 状态管理 (State Management)         │
│     - 维护 AgentState                   │
│     - 状态转换验证                       │
│     - Checkpoint 创建与恢复              │
│                                         │
│  3. 资源控制 (Resource Control)         │
│     - Token/Cost/Time 预算              │
│     - 限流与背压                         │
│                                         │
│  4. 安全边界 (Safety Boundary)          │
│     - 进度检测（防死循环）               │
│     - 错误处理与恢复                     │
│     - Stop Conditions                   │
└─────────────────────────────────────────┘
```

---

## 2. Policy-Step-Reducer 模式

### 2.1 设计思想

**来源**：借鉴 Redux 的单向数据流

```
┌──────────┐      ┌────────────┐      ┌──────────┐
│  Policy  │─────▶│   Step     │─────▶│ Reducer  │
│ (决策层)  │      │  Executor  │      │ (更新层)  │
│          │      │  (执行层)   │      │          │
└──────────┘      └────────────┘      └──────────┘
     ▲                                       │
     │              ┌──────────┐            │
     └──────────────│  State   │◀───────────┘
                    │ (状态中心) │
                    └──────────┘
```

### 2.2 各层职责

#### Policy（决策层）

**职责**：根据当前状态，决定下一步做什么

```python
class BasePolicy(ABC):
    @abstractmethod
    async def decide(self, state: AgentState) -> PolicyDecision:
        """决定下一步行动"""
        ...
```

**示例：ReAct Policy**
```python
class ReActPolicy(BasePolicy):
    async def decide(self, state: AgentState) -> PolicyDecision:
        # 构建 prompt
        prompt = f"""
        Goal: {state.goal}
        History: {format_history(state.completed_steps)}

        Think step by step:
        Thought: <your reasoning>
        Action: <what to do next>
        """

        return PolicyDecision(
            action_type="llm_call",
            messages=[{"role": "user", "content": prompt}],
        )
```

**面试要点**：
- Q: 为什么要把决策逻辑抽象成 Policy？
- A: **关注点分离**。Policy 只关心"做什么"，不关心"怎么做"。这样可以：
  1. 轻松切换不同的决策策略（ReAct / Plan-and-Execute / CoT）
  2. 独立测试决策逻辑
  3. 复用执行和状态管理逻辑

#### StepExecutor（执行层）

**职责**：执行 Policy 的决策，返回结果

```python
class StepExecutor:
    async def execute(
        self,
        state: AgentState,
        decision: PolicyDecision
    ) -> StepResult:
        if decision.action_type == "llm_call":
            return await self._execute_llm_call(...)
        elif decision.action_type == "tool_call":
            return await self._execute_tool_calls(...)
```

**关键点**：
- 不做决策，只执行
- 处理 LLM 调用、工具调用
- 返回标准化的 `StepResult`

#### Reducer（更新层）

**职责**：根据 StepResult 更新 State

```python
class DefaultReducer(BaseReducer):
    async def reduce(
        self,
        state: AgentState,
        step_result: StepResult
    ) -> AgentState:
        # 1. 更新完成步骤
        state.completed_steps.append(summarize(step_result))

        # 2. 更新工作记忆
        state.working_memory.update(step_result.memory_updates)

        # 3. 更新错误追踪
        if step_result.success:
            state.consecutive_errors = 0
        else:
            state.consecutive_errors += 1

        return state
```

**面试要点**：
- Q: 为什么需要 Reducer？直接在 Executor 里更新 State 不行吗？
- A: **纯函数原则**。Reducer 是纯函数：
  ```
  (State, StepResult) => NewState
  ```
  好处：
  1. 可测试：输入输出明确
  2. 可预测：无副作用
  3. 可追溯：State 变化有明确来源
  4. 可时间旅行：配合 Checkpoint 可以回退

### 2.3 执行流程

```python
# Agent.run() 主循环
while True:
    # 1. Policy 决策
    decision = await policy.decide(state)

    # 2. StepExecutor 执行
    step_result = await step_executor.execute(state, decision)

    # 3. Reducer 更新状态
    state = await reducer.reduce(state, step_result)

    # 4. 检查停止条件
    if should_stop(state):
        break
```

**面试要点**：
- Q: 这个模式和 React/Redux 有什么相似之处？
- A: 都是单向数据流：
  ```
  React:  Action -> Reducer -> State -> View
  Agent:  Policy -> Executor -> Reducer -> State
  ```

---

## 3. Schema Validation 深度解析

### 3.1 为什么需要 Schema Validation？

**问题场景**：
```python
# LLM 输出不可靠
response = llm.generate("Return a JSON with name and age")
# 可能返回：
# "Here's the JSON: {name: 'Alice', age: 30}"  # ❌ 不是纯 JSON
# "```json\n{...}\n```"                        # ❌ 在代码块里
# '{"name": "Bob"}'                            # ❌ 缺少 age 字段
```

**解决方案**：自动验证和重试

### 3.2 设计思路

**核心类**：
```python
class OutputValidator:
    def validate_json(
        self,
        response: LLMResponse,
        schema: type[BaseModel]
    ) -> ValidationResult:
        """验证 JSON 并校验 schema"""

    def validate_structured_format(
        self,
        response: LLMResponse,
        required_fields: list[str]
    ) -> ValidationResult:
        """验证结构化文本格式"""

    def create_retry_prompt(
        self,
        validation_result: ValidationResult
    ) -> str:
        """生成重试 prompt"""
```

### 3.3 实现细节

#### 3.3.1 JSON 提取策略

```python
def _extract_json_from_markdown(self, content: str) -> dict | None:
    """从 markdown 代码块中提取 JSON"""
    patterns = [
        r"```json\s*\n(.*?)\n```",      # ```json ... ```
        r"```\s*\n(\{.*?\})\n```",      # ``` {...} ```
        r"\{[^{}]*(?:\{[^{}]*\})*\}",   # 裸 JSON 对象
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except JSONDecodeError:
                continue
```

**面试要点**：
- Q: 为什么要支持从代码块提取？
- A: LLM 有时会在 JSON 外包裹说明文字或代码块。宽容解析提高成功率。

#### 3.3.2 Retry 机制

```python
async def _execute_llm_call_with_validation(self):
    for attempt in range(max_attempts):
        response = await llm.generate(messages)

        # 验证
        validation = validator.validate_json(response, schema)

        if validation.valid:
            return response  # 成功

        # 失败，生成 retry prompt
        if attempt < max_attempts - 1:
            retry_prompt = validator.create_retry_prompt(validation)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": retry_prompt})
            continue  # 重试

        # 最后一次失败
        return StepResult(success=False, error=validation.errors)
```

**关键点**：
1. **保留上下文**：把失败的响应加入 messages
2. **反馈错误**：告诉 LLM 哪里错了
3. **指导修正**：提示正确格式

**示例 retry prompt**：
```
Your previous response was invalid. Please try again.

Errors found:
- Missing required field: age
- Invalid value for field name: expected string, got number

Your previous response:
{"name": 123}

Expected format:
{
  "name": "string",
  "age": "number"
}

Please provide a valid response.
```

### 3.4 最佳实践

**1. 宽容解析，严格验证**
```python
# 宽容：尝试多种提取方式
data = extract_json(response) or extract_from_markdown(response)

# 严格：Pydantic 校验
validated = MySchema.model_validate(data)
```

**2. 渐进式重试**
```python
# 第 1 次：直接提示
# 第 2 次：详细错误 + 示例
# 第 3 次：放弃或降级
```

**3. 记录验证失败**
```python
if not validation.valid:
    trace_event = TraceEvent(
        event_type=EventType.VALIDATION_FAILED,
        metadata={
            "errors": validation.errors,
            "raw_content": response.content,
        }
    )
```

---

## 4. Replay 机制设计

### 4.1 核心概念

**Replay 是什么？**
- 重放历史执行，精确复现每一步
- 用于调试、测试、对比分析

**为什么需要 Replay？**

| 场景 | 没有 Replay | 有 Replay |
|------|------------|----------|
| Bug 调试 | "我也不知道为什么，运行 100 次可能复现 1 次" | "replay 这个 run_id，100% 复现" |
| 回归测试 | "改了代码，不知道会不会影响其他场景" | "replay 历史所有 case，看有没有退化" |
| A/B 对比 | "换了 prompt，不知道效果" | "同样输入，对比两次 run 的差异" |

### 4.2 架构设计

```
┌──────────────────────────────────────────┐
│          ReplayEngine                    │
├──────────────────────────────────────────┤
│                                          │
│  ┌────────────────────────────────┐     │
│  │      ReplayCache               │     │
│  ├────────────────────────────────┤     │
│  │ - llm_responses: dict          │     │
│  │ - tool_results: dict           │     │
│  │                                │     │
│  │ load_from_trace(run_id)        │     │
│  │ get_llm_response(digest)       │     │
│  │ get_tool_result(digest)        │     │
│  └────────────────────────────────┘     │
│                                          │
│  replay_run(run_id, from_step, to_step) │
│  get_divergence_point(run_a, run_b)     │
│                                          │
└──────────────────────────────────────────┘
```

### 4.3 实现细节

#### 4.3.1 Cache 加载

```python
class ReplayCache:
    def load_from_trace(self, reader: TraceReader, run_id: str):
        events = reader.read_events(run_id)

        for event in events:
            if event.event_type == EventType.LLM_CALL:
                # 按请求 digest 索引响应
                self._llm_responses[event.llm_request_digest] = {
                    "content": event.llm_response_content,
                    "usage": event.llm_usage,
                    "model": event.model,
                }

            elif event.event_type == EventType.TOOL_CALL:
                # 按幂等键索引结果
                key = event.tool_call["idempotency_key"]
                self._tool_results[key] = event.tool_result
```

**关键点**：
- 使用 **digest/hash 作为索引**，而非顺序
- 支持 **非线性 replay**（跳步、分支）

#### 4.3.2 确定性保证

**问题**：如何保证 replay 和原始执行一致？

**策略**：
1. **LLM 调用**：使用 `temperature=0` + `seed`
2. **工具调用**：使用幂等键去重
3. **状态哈希**：验证每步状态一致性

```python
async def replay_run(self, run_id: str) -> AgentState:
    # 1. 加载 cache
    self.cache.load_from_trace(self.reader, run_id)

    # 2. 重放事件
    events = self.reader.read_events(run_id)
    state = self._reconstruct_initial_state(events)

    for event in events:
        # 3. 从 cache 获取响应（而非真正调用 LLM）
        if event.event_type == EventType.LLM_CALL:
            cached_response = self.cache.get_llm_response(
                event.llm_request_digest
            )
            # 使用缓存响应继续执行

        # 4. 验证状态一致性
        if event.state_after_hash:
            actual_hash = canonical_hash(state)
            assert actual_hash == event.state_after_hash, \
                "State diverged!"

    return state
```

### 4.4 分歧点检测

**用途**：对比两次运行，找到第一次不同的地方

```python
async def get_divergence_point(
    self,
    run_id_a: str,
    run_id_b: str
) -> tuple[int, TraceEvent, TraceEvent] | None:
    events_a = self.reader.read_events(run_id_a)
    events_b = self.reader.read_events(run_id_b)

    for i, (event_a, event_b) in enumerate(zip(events_a, events_b)):
        # 对比状态 hash
        if event_a.state_after_hash != event_b.state_after_hash:
            return (i, event_a, event_b)

        # 对比 LLM 响应
        if event_a.llm_response_digest != event_b.llm_response_digest:
            return (i, event_a, event_b)

    return None
```

**应用场景**：
- 测试 prompt 改动的影响
- 对比不同模型的输出
- 定位引入 bug 的代码变更

---

## 5. 错误分类与处理策略

### 5.1 错误分类体系

**ErrorType（错误类型）**：
```python
class ErrorType(str, Enum):
    # 可重试
    RETRYABLE = "retryable"          # 限流、超时
    VALIDATION = "validation"         # Schema 校验失败
    PARTIAL_FAILURE = "partial_failure"  # 部分成功

    # 不可重试
    PERMANENT = "permanent"           # 逻辑错误
    BUDGET_EXCEEDED = "budget_exceeded"  # 预算耗尽
    AUTHORIZATION = "authorization"   # 权限不足

    # 需要升级
    REQUIRES_HUMAN = "requires_human"  # 需要人工介入
    SAFETY_VIOLATION = "safety_violation"  # 安全违规
```

**ErrorSeverity（严重程度）**：
```python
class ErrorSeverity(str, Enum):
    LOW = "low"          # 轻微，系统继续
    MEDIUM = "medium"    # 中等，性能下降
    HIGH = "high"        # 严重，功能受损
    CRITICAL = "critical"  # 致命，无法继续
```

### 5.2 错误分类逻辑

```python
def _classify_error(self, error: Exception) -> RuntimeError:
    error_str = str(error).lower()

    # 限流 -> RETRYABLE + MEDIUM
    if any(x in error_str for x in ["rate limit", "429"]):
        return RuntimeError(
            str(error),
            error_type=ErrorType.RETRYABLE,
            severity=ErrorSeverity.MEDIUM,
        )

    # 预算 -> BUDGET_EXCEEDED + HIGH
    if "budget" in error_str:
        return RuntimeError(
            str(error),
            error_type=ErrorType.BUDGET_EXCEEDED,
            severity=ErrorSeverity.HIGH,
        )

    # 默认 -> PERMANENT + MEDIUM
    return RuntimeError(
        str(error),
        error_type=ErrorType.PERMANENT,
        severity=ErrorSeverity.MEDIUM,
    )
```

### 5.3 处理策略

**根据错误类型决定动作**：

| ErrorType | 处理策略 |
|-----------|---------|
| RETRYABLE | 自动重试（指数退避） |
| VALIDATION | 重试 + 提示修正 |
| PARTIAL_FAILURE | 记录日志，继续执行 |
| PERMANENT | 停止执行，返回错误 |
| BUDGET_EXCEEDED | 触发降级策略 |
| AUTHORIZATION | 停止执行，记录审计 |
| REQUIRES_HUMAN | 暂停，等待人工确认 |
| SAFETY_VIOLATION | 立即停止，记录告警 |

### 5.4 Retry 策略

**指数退避**：
```python
class RetryStrategy:
    def get_delay(self, attempt: int) -> float:
        delay_ms = min(
            self.initial_delay_ms * (2 ** attempt),
            self.max_delay_ms
        )

        # 加入 jitter（随机抖动）
        if self.jitter:
            jitter_range = delay_ms * 0.25
            delay_ms += random.uniform(-jitter_range, jitter_range)

        return delay_ms / 1000

# 示例：
# attempt 0: 1000ms (1s)
# attempt 1: 2000ms (2s)
# attempt 2: 4000ms (4s)
# attempt 3: 8000ms (8s, 加 ±25% jitter)
```

**为什么需要 jitter？**
- 防止"惊群效应"：多个 agent 同时重试导致服务器压力激增
- 分散请求时间，提高成功率

---

## 6. 面试高频问题

### Q1: 解释一下 Policy-Step-Reducer 模式

**回答框架**：
```
1. 背景：借鉴 Redux 的单向数据流思想
2. 三层职责：
   - Policy: 根据状态决定做什么（决策层）
   - StepExecutor: 执行决策，调用 LLM/工具（执行层）
   - Reducer: 根据结果更新状态（更新层）
3. 好处：
   - 关注点分离：每层职责单一
   - 可测试性：纯函数，输入输出明确
   - 可扩展性：轻松替换 Policy
   - 可维护性：数据流清晰可追溯
```

### Q2: Schema Validation 为什么要自动重试？

**回答框架**：
```
1. 问题：LLM 输出不稳定
   - 格式问题：JSON 在代码块里
   - Schema 问题：缺少字段
   - 概率问题：同样 prompt，10次有1次出错

2. 重试策略：
   - 保留上下文：把失败响应加入对话
   - 反馈错误：明确告诉 LLM 哪里错了
   - 指导修正：提示正确格式

3. 收益：
   - 成功率提升：从 90% 到 99%+
   - 用户体验：自动修正，无需人工介入
   - 成本可控：最多重试 2-3 次
```

### Q3: Replay 机制如何保证确定性？

**回答框架**：
```
1. 挑战：LLM 调用本质是随机的
2. 解决方案：
   - LLM: temperature=0 + seed 固定
   - Cache: 按 request digest 索引响应
   - 工具: 幂等键去重
   - 验证: 每步对比 state hash
3. 应用场景：
   - 调试：100% 复现 bug
   - 测试：回归所有历史 case
   - 对比：找到两次运行的分歧点
```

### Q4: 错误分类的意义是什么？

**回答框架**：
```
1. 背景：不是所有错误都该重试
2. 分类维度：
   - 类型：可重试 vs 永久性
   - 严重性：LOW/MEDIUM/HIGH/CRITICAL
3. 处理策略：
   - RETRYABLE: 自动重试（指数退避）
   - BUDGET_EXCEEDED: 触发降级
   - REQUIRES_HUMAN: 暂停等待人工
4. 收益：
   - 避免无效重试浪费资源
   - 快速失败，不浪费时间
   - 清晰的错误上报和告警
```

### Q5: ProgressDetector 如何检测死循环？

**回答框架**：
```
1. 三种检测机制：
   a. 重复步骤：连续N步 hash 相同
   b. 循环模式：A-B-A-B 重复
   c. 输出相似：连续输出文本几乎一样

2. 实现细节：
   - 滑动窗口：保留最近 N 步
   - Hash 去重：对比 step 的 digest
   - 相似度：unique_outputs / total_outputs

3. Stop 策略：
   - 连续 3 步无进展 -> 停止
   - 记录 stop_reason = NO_PROGRESS
   - 返回失败状态，便于分析
```

---

## 7. 实战经验总结

### 7.1 设计经验

**1. 契约优先 (Contract-First)**
```python
# ✅ 先定义接口
class BasePolicy(ABC):
    @abstractmethod
    async def decide(self, state) -> PolicyDecision:
        ...

# ✅ 再写实现
class ReActPolicy(BasePolicy):
    async def decide(self, state) -> PolicyDecision:
        # 实现细节
```

**好处**：
- 接口稳定，实现可替换
- 便于 Mock 和测试
- 多人协作边界清晰

**2. 状态即数据 (State as Data)**
```python
# ✅ State 是纯数据结构
@dataclass
class AgentState:
    run_id: str
    current_step: int
    completed_steps: list[str]
    # ... 只有数据，没有方法

# ❌ 不要把逻辑混入 State
class AgentState:
    def execute_next_step(self):  # ❌ 违反原则
        ...
```

**好处**：
- 序列化简单（Checkpoint）
- 易于测试和调试
- 便于 Replay 和时间旅行

**3. 单向数据流 (Unidirectional Data Flow)**
```
Policy -> Executor -> Reducer -> State
  ↑                                 │
  └─────────────────────────────────┘
```

**好处**：
- 数据流清晰可追踪
- 没有循环依赖
- 易于理解和维护

### 7.2 实现技巧

**1. 分步验证**
```python
# ✅ 逐步验证，早发现早失败
async def _execute_llm_call(self):
    # 1. 验证预算
    if self.budget_tracker:
        self.budget_tracker.check_budget()  # 可能抛异常

    # 2. 调用 LLM
    response = await self.gateway.generate(request)

    # 3. 验证 Schema
    if expected_schema:
        validation = self.validator.validate_json(response, schema)
        if not validation.valid:
            return StepResult(success=False, error=...)

    # 4. 返回结果
    return StepResult(success=True, ...)
```

**2. 防御式编程**
```python
# ✅ 处理所有边界情况
def canonical_hash(obj: Any) -> str:
    if obj is None:
        return "null"

    if isinstance(obj, float):
        # 归一化浮点数精度
        obj = round(obj, 6)

    # ... 更多边界情况
```

**3. 日志与 Trace**
```python
# ✅ 关键路径全部记录
if validation.valid:
    logger.info("Validation passed", extra={
        "run_id": run_id,
        "step_id": step_id,
    })
else:
    logger.warning("Validation failed", extra={
        "errors": validation.errors,
        "raw_content": validation.raw_content[:200],
    })
    trace_writer.write(TraceEvent(
        event_type=EventType.VALIDATION_FAILED,
        ...
    ))
```

### 7.3 性能优化

**1. 避免重复计算**
```python
# ❌ 每次都计算 hash
for event in events:
    hash = canonical_hash(event.state)

# ✅ 缓存 hash
state_hash = canonical_hash(state)
for event in events:
    event.state_hash = state_hash
```

**2. 异步批量操作**
```python
# ❌ 串行写入
for event in events:
    await trace_writer.write(event)

# ✅ 批量写入
await trace_writer.write_batch(events)
```

**3. 懒加载**
```python
# ✅ 只在需要时才初始化
@property
def progress_detector(self) -> ProgressDetector:
    if self._progress_detector is None:
        self._progress_detector = ProgressDetector(...)
    return self._progress_detector
```

### 7.4 常见坑与解决

**坑 1：LLM 输出格式不稳定**
```python
# 问题：有时返回 JSON，有时返回文本 + JSON
response = '这是结果：{"name": "Alice"}'

# 解决：宽容解析
def extract_json(text: str) -> dict | None:
    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass

    # 2. 提取代码块
    if "```" in text:
        return extract_from_markdown(text)

    # 3. 正则提取
    return extract_by_regex(text)
```

**坑 2：State hash 不一致**
```python
# 问题：同样的 State，hash 不同
# 原因：浮点数精度、key 顺序、时间戳

# 解决：
def canonical_hash(obj):
    # 1. 排除时间戳等易变字段
    obj = exclude(obj, ["start_time", "elapsed_ms"])

    # 2. 浮点数归一化
    obj = normalize_floats(obj, precision=6)

    # 3. JSON 规范化（key 排序）
    json_str = json.dumps(obj, sort_keys=True, separators=(',', ':'))

    return hashlib.sha256(json_str.encode()).hexdigest()[:16]
```

**坑 3：Replay 时 cache miss**
```python
# 问题：replay 找不到缓存的响应
# 原因：request digest 计算不一致

# 解决：统一 digest 计算
def compute_request_digest(request: LLMRequest) -> str:
    # 确保字段顺序、格式一致
    canonical = {
        "model": request.model,
        "messages": [msg.model_dump() for msg in request.messages],
        "temperature": request.temperature,
        "seed": request.seed,
    }
    return canonical_hash(canonical)
```

---

## 8. 下一步学习

### Week 5 预告：Tool Gateway

即将学习：
- 工具权限模型（Capability-based）
- 参数校验与注入防护
- 幂等性设计
- HITL（Human-In-The-Loop）
- 审计日志

### 扩展阅读

**论文**：
- ReAct: Synergizing Reasoning and Acting in Language Models
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**开源项目**：
- LangGraph: State machines for agents
- AutoGPT: Autonomous agent framework
- DSPy: Programming with LLMs

**技术博客**：
- Anthropic: Building reliable agents
- OpenAI: Function calling best practices

---

## 总结

**本周核心收获**：
1. ✅ 理解 Agent Runtime 的架构设计
2. ✅ 掌握 Policy-Step-Reducer 模式
3. ✅ 学会 Schema Validation 和重试策略
4. ✅ 理解 Replay 机制的设计与实现
5. ✅ 建立完整的错误分类体系

**关键能力**：
- 架构设计能力：分层、解耦、单向数据流
- 工程能力：错误处理、重试、日志、测试
- 调试能力：Trace、Replay、分歧点检测
- 面试准备：高频问题有备无患

**下周目标**：
- 开始 Week 5: Tool Gateway
- 继续保持学习节奏
- 注重理论与实践结合
