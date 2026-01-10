# Agent Runtime 知识点详解

## 目录

1. [核心概念](#1-核心概念)
2. [架构设计](#2-架构设计)
3. [Policy-Step-Reducer 模式](#3-policy-step-reducer-模式)
4. [状态管理](#4-状态管理)
5. [检查点与恢复](#5-检查点与恢复)
6. [进度检测](#6-进度检测)
7. [执行控制](#7-执行控制)
8. [面试高频问题](#8-面试高频问题)
9. [代码细节考点](#9-代码细节考点)
10. [扩展阅读](#10-扩展阅读)

---

## 1. 核心概念

### 1.1 什么是 Agent Runtime？

**定义**: Agent Runtime 是 Agent 系统的执行引擎，负责：
- 管理 Agent 的生命周期（初始化、运行、暂停、恢复、终止）
- 协调各组件（LLM、工具、记忆）的交互
- 维护执行状态和上下文
- 处理错误和异常情况

**类比理解**:
```
Agent Runtime 之于 Agent = JVM 之于 Java 程序 = Node.js 之于 JavaScript
```

### 1.2 为什么需要独立的 Runtime？

| 问题 | Runtime 如何解决 |
|------|------------------|
| 执行可控性 | 统一的状态机管理，明确的状态转换 |
| 可观测性 | 所有操作通过 Runtime，便于追踪和审计 |
| 可恢复性 | 检查点机制，支持中断后恢复 |
| 可测试性 | 组件解耦，便于单元测试和模拟 |
| 资源管理 | 统一的预算控制和限流 |

### 1.3 核心职责

```
┌─────────────────────────────────────────────────────────┐
│                     Agent Runtime                        │
├─────────────────────────────────────────────────────────┤
│  1. 执行编排 (Orchestration)                             │
│     - 协调 Policy、Executor、Reducer 的执行流程           │
│                                                          │
│  2. 状态管理 (State Management)                          │
│     - 维护 AgentState                                    │
│     - 管理状态转换                                        │
│     - 检查点创建与恢复                                    │
│                                                          │
│  3. 资源控制 (Resource Control)                          │
│     - Token 预算                                         │
│     - 成本限制                                           │
│     - 时间限制                                           │
│                                                          │
│  4. 安全边界 (Safety Boundary)                           │
│     - 进度检测（防止死循环）                              │
│     - 错误处理与恢复                                      │
│     - 停止条件判断                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 架构设计

### 2.1 整体架构

```
                    ┌──────────────────────┐
                    │       Agent          │
                    │   (Orchestrator)     │
                    └──────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     Policy      │  │  StepExecutor   │  │    Reducer      │
│  (决策层)       │  │   (执行层)      │  │   (更新层)      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
         │                     ▼                     │
         │           ┌─────────────────┐             │
         │           │  ModelGateway   │             │
         │           │  (LLM 调用)     │             │
         │           └─────────────────┘             │
         │                                           │
         └──────────────────┬────────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   AgentState    │
                   │   (状态中心)    │
                   └─────────────────┘
```

### 2.2 组件职责

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **Agent** | 执行编排、生命周期管理 | goal, initial_state | final_state |
| **Policy** | 决策下一步动作 | AgentState | PolicyDecision |
| **StepExecutor** | 执行具体操作 | PolicyDecision | StepResult |
| **Reducer** | 更新状态 | AgentState + StepResult | AgentState |
| **StateManager** | 状态转换、检查点 | AgentState | StateSnapshot |
| **ProgressDetector** | 检测执行进度 | StepResult | is_making_progress |

### 2.3 设计原则

**1. 单一职责原则 (SRP)**
```python
# 每个组件只做一件事
Policy      → 只负责决策
Executor    → 只负责执行
Reducer     → 只负责状态更新
```

**2. 依赖倒置原则 (DIP)**
```python
# Agent 依赖抽象，不依赖具体实现
class Agent:
    def __init__(
        self,
        policy: BasePolicy,       # 抽象类
        reducer: BaseReducer,     # 抽象类
        gateway: ModelGatewayRegistry,  # 抽象接口
    ): ...
```

**3. 开闭原则 (OCP)**
```python
# 扩展新 Policy 无需修改 Agent
class CustomPolicy(BasePolicy):
    async def decide(self, state: AgentState) -> PolicyDecision:
        # 自定义决策逻辑
        ...
```

---

## 3. Policy-Step-Reducer 模式

### 3.1 模式来源

该模式借鉴自：
- **Redux**: Action → Reducer → State
- **Elm Architecture**: Model → Update → View
- **ReAct 论文**: Reasoning + Acting

### 3.2 执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         执行循环                                 │
└─────────────────────────────────────────────────────────────────┘

    AgentState
        │
        ▼
┌───────────────┐
│    Policy     │  ← 输入: 当前状态
│   .decide()   │  → 输出: PolicyDecision (下一步做什么)
└───────┬───────┘
        │
        ▼
    PolicyDecision
        │
        ▼
┌───────────────┐
│ StepExecutor  │  ← 输入: 决策
│  .execute()   │  → 输出: StepResult (执行结果)
└───────┬───────┘
        │
        ▼
    StepResult
        │
        ▼
┌───────────────┐
│   Reducer     │  ← 输入: 旧状态 + 执行结果
│   .reduce()   │  → 输出: 新状态
└───────┬───────┘
        │
        ▼
    AgentState (更新后)
        │
        └──────────→ 循环继续...
```

### 3.3 数据流详解

**Step 1: Policy 决策**
```python
class PolicyDecision(BaseModel):
    action_type: str  # "llm_call", "tool_call", "complete", "fail"
    messages: list[dict]  # LLM 调用时的消息
    tool_calls: list[dict]  # 工具调用时的参数
    stop_reason: str | None  # 完成/失败原因
    reasoning: str | None  # 决策理由（用于调试）
```

**Step 2: Executor 执行**
```python
class StepResult(BaseModel):
    step_type: StepType  # THINK, ACT, OBSERVE, PLAN, VERIFY
    step_id: str
    success: bool

    # 输出
    thought: str | None  # LLM 的思考
    action: str | None   # 决定的动作
    observation: str | None  # 观察结果

    # LLM 响应
    llm_response: LLMResponse | None

    # 状态变更
    state_updates: dict[str, Any]  # 直接更新字段
    memory_updates: dict[str, Any]  # 工作记忆更新

    # 错误信息
    error: str | None
    is_recoverable: bool
```

**Step 3: Reducer 更新**
```python
async def reduce(self, state: AgentState, step_result: StepResult) -> AgentState:
    # 1. 记录完成的步骤
    state.completed_steps.append(summarize(step_result))

    # 2. 应用状态更新
    for key, value in step_result.state_updates.items():
        setattr(state, key, value)

    # 3. 更新工作记忆
    state.working_memory.update(step_result.memory_updates)

    # 4. 跟踪错误
    if step_result.success:
        state.consecutive_errors = 0
    else:
        state.consecutive_errors += 1

    return state
```

### 3.4 为什么用这个模式？

| 优点 | 解释 |
|------|------|
| **可预测性** | 状态变化都通过 Reducer，易于追踪 |
| **可测试性** | 每个组件可独立测试 |
| **可扩展性** | 新增 Policy 或 Reducer 不影响其他组件 |
| **可调试性** | 清晰的数据流，便于定位问题 |
| **时间旅行调试** | 可以回放任意步骤的状态 |

### 3.5 面试考点：与 Redux 的对比

| 方面 | Redux | Agent Runtime |
|------|-------|---------------|
| **Action** | 同步对象 | PolicyDecision（异步生成）|
| **Reducer** | 纯函数，同步 | 异步，可有副作用 |
| **中间件** | Redux Middleware | Hooks 系统 |
| **状态** | 不可变 | Pydantic Model（可变但可序列化）|
| **触发** | dispatch(action) | 自动循环 |

---

## 4. 状态管理

### 4.1 AgentState 设计

```python
class AgentState(BaseModel):
    # ===== 标识符 =====
    run_id: str           # 执行运行 ID
    task_id: str | None   # 任务分组 ID

    # ===== 执行跟踪 =====
    status: ExecutionStatus  # PENDING/RUNNING/PAUSED/COMPLETED/FAILED/CANCELLED
    current_step: int = 0    # 当前步骤计数
    max_steps: int = 100     # 最大步骤限制

    # ===== 目标与进度 =====
    goal: str | None                  # 执行目标
    current_plan: list[str] = []      # 当前计划
    completed_steps: list[str] = []   # 已完成步骤

    # ===== 工作记忆 =====
    working_memory: dict[str, Any] = {}  # 中间结果存储

    # ===== 对话历史 =====
    messages: list[dict] = []  # LLM 对话上下文

    # ===== 预算跟踪 =====
    tokens_used: int = 0       # 已用 Token
    cost_usd: float = 0.0      # 已用成本
    start_time: datetime       # 开始时间
    elapsed_ms: int = 0        # 已用时间

    # ===== 错误跟踪 =====
    last_error: str | None = None     # 最后错误
    consecutive_errors: int = 0       # 连续错误数
    consecutive_no_progress: int = 0  # 连续无进展数
```

### 4.2 状态转换图

```
                         start()
    ┌─────────┐ ─────────────────────► ┌─────────┐
    │ PENDING │                         │ RUNNING │
    └─────────┘                         └────┬────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          │                  │                  │
                     pause()            complete()           fail()
                          │                  │                  │
                          ▼                  ▼                  ▼
                    ┌─────────┐       ┌───────────┐       ┌────────┐
                    │ PAUSED  │       │ COMPLETED │       │ FAILED │
                    └────┬────┘       └───────────┘       └────────┘
                         │
                    resume()
                         │
                         └──────────► RUNNING

    任意状态 ────────cancel()──────────► CANCELLED
```

### 4.3 状态转换验证

```python
VALID_TRANSITIONS = {
    ExecutionStatus.PENDING: {ExecutionStatus.RUNNING},
    ExecutionStatus.RUNNING: {
        ExecutionStatus.PAUSED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
    },
    ExecutionStatus.PAUSED: {
        ExecutionStatus.RUNNING,
        ExecutionStatus.CANCELLED,
    },
    ExecutionStatus.COMPLETED: set(),  # 终态
    ExecutionStatus.FAILED: set(),     # 终态
    ExecutionStatus.CANCELLED: set(),  # 终态
}

def transition(state: AgentState, new_status: ExecutionStatus) -> AgentState:
    if new_status not in VALID_TRANSITIONS[state.status]:
        raise StateTransitionError(state.status, new_status)
    state.status = new_status
    return state
```

### 4.4 面试考点：为什么状态要这样设计？

**Q: 为什么 working_memory 用 dict 而不是固定字段？**

A: 灵活性与通用性：
- 不同任务需要存储不同类型的中间结果
- 避免 AgentState 变成"上帝对象"
- 允许 Policy 自定义存储格式

**Q: 为什么要分 completed_steps 和 messages？**

A: 不同用途：
- `completed_steps`: 人类可读的步骤摘要，用于上下文压缩
- `messages`: LLM 原始对话，用于保持对话连贯性

**Q: consecutive_errors 和 consecutive_no_progress 的区别？**

A:
- `consecutive_errors`: 捕获的异常数，说明执行出错
- `consecutive_no_progress`: 正常执行但没有新产出，说明可能陷入循环

---

## 5. 检查点与恢复

### 5.1 检查点设计

```python
class StateSnapshot(BaseModel):
    run_id: str           # 所属运行
    step_id: str          # 步骤标识
    timestamp: datetime   # 创建时间

    state_hash: str       # 状态哈希（完整性校验）
    state: AgentState     # 完整状态

    checkpoint_reason: str | None  # 检查点原因
    is_resumable: bool = True      # 是否可恢复
```

### 5.2 检查点触发时机

```python
def _should_checkpoint(state, step_result) -> bool:
    # 1. 错误时检查点（便于调试）
    if not step_result.success and config.checkpoint_on_error:
        return True

    # 2. 固定间隔检查点
    if state.current_step % config.checkpoint_interval_steps == 0:
        return True

    # 3. 预算阈值检查点（50%, 75%, 90%）
    budget_ratio = get_budget_ratio()
    for threshold in [0.5, 0.75, 0.9]:
        if crossed_threshold(budget_ratio, threshold):
            return True

    return False
```

### 5.3 哈希计算

```python
def compute_state_hash(state: AgentState) -> str:
    # 排除易变字段
    serializable = state.model_dump(exclude={"start_time", "elapsed_ms"})

    # 规范化 JSON（排序 key，无空格）
    canonical = json.dumps(serializable, sort_keys=True, separators=(',', ':'))

    # SHA-256 哈希，截断 16 字符
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

**为什么排除 start_time 和 elapsed_ms？**
- 这些是"易变"字段，每次序列化值都不同
- 包含它们会导致相同逻辑状态的哈希不同
- 影响检查点的去重和验证

### 5.4 恢复流程

```python
async def resume(snapshot: StateSnapshot) -> AgentState:
    # 1. 验证哈希
    actual_hash = compute_state_hash(snapshot.state)
    if actual_hash != snapshot.state_hash:
        raise HashVerificationError(snapshot.state_hash, actual_hash)

    # 2. 重置进度检测器
    progress_detector.reset()

    # 3. 从快照状态继续运行
    return await run(
        goal=snapshot.state.goal,
        initial_state=snapshot.state,
    )
```

### 5.5 Resume vs Replay

| 方面 | Resume（恢复） | Replay（重放） |
|------|----------------|----------------|
| **LLM 调用** | 真实调用，可能返回不同结果 | 使用缓存响应，结果确定 |
| **工具调用** | 真实执行 | 使用缓存结果或模拟 |
| **用途** | 从中断点继续 | 调试、分析、复现 |
| **结果** | 可能不同 | 完全相同 |

### 5.6 面试考点

**Q: 检查点存储在哪里？如何选择存储方案？**

A: 当前实现使用 JSONL 文件，生产环境可考虑：

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| JSONL 文件 | 简单、无依赖 | 不支持并发、无查询 | 开发/测试 |
| SQLite | 单文件、支持查询 | 并发受限 | 单机生产 |
| PostgreSQL | 完整功能、高并发 | 运维复杂 | 分布式生产 |
| Redis | 高性能、TTL | 持久化需配置 | 短期检查点 |
| S3 | 无限存储、便宜 | 延迟高 | 归档、长期存储 |

**Q: 如何处理检查点过多的问题？**

A: 检查点保留策略：
```python
class RetentionPolicy:
    keep_last_n: int = 10           # 保留最近 N 个
    keep_on_error: bool = True      # 错误检查点总是保留
    ttl_hours: int = 24             # 24 小时后过期
    keep_milestones: bool = True    # 保留里程碑（50%, 完成）
```

---

## 6. 进度检测

### 6.1 为什么需要进度检测？

LLM Agent 常见的失败模式：

```
# 1. 重复循环
Step 1: Search for "python tutorial"
Step 2: Search for "python tutorial"  ← 完全相同
Step 3: Search for "python tutorial"

# 2. 无效循环
Step 1: I need to find information
Step 2: Let me search for that
Step 3: I should look this up       ← 不断"思考"但不行动

# 3. 振荡
Step 1: Add item to cart
Step 2: Remove item from cart
Step 3: Add item to cart            ← A-B-A-B 模式
Step 4: Remove item from cart
```

### 6.2 检测策略

```python
class ProgressDetector:
    def __init__(self, window_size: int = 5, similarity_threshold: float = 0.95):
        self._step_hashes: deque[str] = deque(maxlen=window_size)
        self._recent_outputs: deque[str] = deque(maxlen=window_size)
        self._action_sequence: deque[str] = deque(maxlen=window_size * 2)

    def is_making_progress(self) -> bool:
        # 检测 1: 重复步骤
        if self._has_duplicate_steps():
            return False

        # 检测 2: 循环模式
        if self._has_cyclic_pattern():
            return False

        # 检测 3: 输出相似度
        if self._outputs_too_similar():
            return False

        return True
```

### 6.3 重复检测

```python
def _has_duplicate_steps(self) -> bool:
    """检测是否有重复的步骤"""
    if len(self._step_hashes) < 2:
        return False

    last_hash = self._step_hashes[-1]
    # 计算最后一个步骤在窗口中出现的次数
    duplicates = sum(1 for h in list(self._step_hashes)[:-1] if h == last_hash)

    # 超过 2 次认为卡住了
    return duplicates >= 2
```

### 6.4 循环检测

```python
def _has_cyclic_pattern(self) -> bool:
    """检测 A-B-A-B 这样的循环模式"""
    actions = list(self._action_sequence)

    # 检测不同长度的循环
    for cycle_length in range(2, len(actions) // 2 + 1):
        if self._is_repeating_cycle(actions, cycle_length):
            return True

    return False

def _is_repeating_cycle(self, sequence: list, cycle_length: int) -> bool:
    """检测是否存在长度为 cycle_length 的重复循环"""
    recent = sequence[-cycle_length * 2:]
    first_half = recent[:cycle_length]
    second_half = recent[cycle_length:]
    return first_half == second_half
```

### 6.5 相似度检测

```python
def _outputs_too_similar(self) -> bool:
    """检测输出是否过于相似"""
    outputs = list(self._recent_outputs)
    unique_outputs = set(outputs)

    # 相似度 = 1 - 唯一输出数/总输出数
    similarity = 1 - (len(unique_outputs) / len(outputs))

    return similarity >= self.similarity_threshold  # 默认 0.95
```

### 6.6 面试考点

**Q: 为什么用哈希而不是直接比较内容？**

A:
1. **性能**: 哈希比较 O(1)，内容比较 O(n)
2. **内存**: 只存储 16 字符哈希，不存储完整内容
3. **隐私**: 不暴露实际内容，便于日志

**Q: 如何改进相似度检测？**

A: 当前实现是简单的精确匹配，可改进：
```python
# 1. 使用编辑距离
def levenshtein_similarity(s1, s2):
    distance = levenshtein_distance(s1, s2)
    return 1 - distance / max(len(s1), len(s2))

# 2. 使用嵌入向量余弦相似度
def embedding_similarity(s1, s2, encoder):
    v1 = encoder.encode(s1)
    v2 = encoder.encode(s2)
    return cosine_similarity(v1, v2)

# 3. 语义相似度（LLM 判断）
def semantic_similarity(s1, s2, llm):
    prompt = f"判断这两个输出是否表达相同意思：\n1: {s1}\n2: {s2}"
    return llm.judge(prompt)
```

**Q: window_size 如何选择？**

A: 权衡考量：
- **太小** (如 2-3): 可能误报，正常的探索行为被判为无进展
- **太大** (如 20+): 检测延迟，浪费资源后才发现循环
- **推荐** (5-10): 平衡敏感度和容忍度

---

## 7. 执行控制

### 7.1 停止条件

```python
class StopReason(str, Enum):
    GOAL_REACHED = "goal_reached"      # 目标达成
    MAX_STEPS = "max_steps"            # 步数超限
    MAX_TIME = "max_time"              # 时间超限
    MAX_COST = "max_cost"              # 成本超限
    MAX_TOKENS = "max_tokens"          # Token 超限
    NO_PROGRESS = "no_progress"        # 无进展
    ERROR = "error"                    # 不可恢复错误
    USER_CANCELLED = "user_cancelled"  # 用户取消
    TOOL_BLOCKED = "tool_blocked"      # 工具被阻止
```

### 7.2 停止条件检查

```python
def _check_stop_conditions(state: AgentState) -> StopReason | None:
    # 1. 步数限制
    if state.current_step >= state.max_steps:
        return StopReason.MAX_STEPS

    # 2. 进度检测
    if state.consecutive_no_progress >= config.max_consecutive_no_progress:
        return StopReason.NO_PROGRESS

    # 3. 预算检查
    if budget_tracker:
        try:
            budget_tracker.check_budget()
        except BudgetExceededError as e:
            if "token" in str(e).lower():
                return StopReason.MAX_TOKENS
            if "cost" in str(e).lower():
                return StopReason.MAX_COST
            if "time" in str(e).lower():
                return StopReason.MAX_TIME

    # 4. 连续错误
    if state.consecutive_errors >= config.max_consecutive_errors:
        return StopReason.ERROR

    return None  # 继续执行
```

### 7.3 停止处理

```python
async def _handle_stop(state: AgentState, reason: StopReason) -> AgentState:
    # 根据停止原因决定最终状态
    if reason == StopReason.GOAL_REACHED:
        state = state_manager.transition(state, ExecutionStatus.COMPLETED)
    else:
        state = state_manager.transition(state, ExecutionStatus.FAILED)

    # 记录停止事件
    trace_writer.write(TraceEvent(
        run_id=state.run_id,
        event_type=EventType.STATE_CHANGE,
        stop_reason=reason,
        state_after_hash=canonical_hash(state),
    ))

    return state
```

### 7.4 目标完成检测

```python
# 方式 1: Policy 显式声明
class PolicyDecision:
    action_type: str  # "complete" 表示目标完成

# 方式 2: 特殊 Action
# LLM 输出 "Action: FINISH" 时
if action and action.upper() == "FINISH":
    return StepResult(state_updates={"goal_reached": True})

# 方式 3: 外部验证（未实现）
# 使用单独的 Verifier 组件判断
```

### 7.5 面试考点

**Q: 如何处理 LLM 不遵循格式的情况？**

A: 多层防护：
```python
# 1. 重试机制
for attempt in range(max_retries):
    response = await llm.generate(request)
    thought, action = parse_response(response.content)
    if thought or action:  # 解析成功
        break
    # 重试时可以添加更明确的指令

# 2. 降级处理
if thought is None and action is None:
    # 将整个响应作为 thought
    thought = response.content

# 3. 格式强制（使用 structured output）
response = await llm.generate(
    request,
    response_format={"type": "json_object"},
    schema=ReActResponseSchema,
)
```

**Q: MAX_STEPS 设置多少合适？**

A: 取决于任务复杂度：

| 任务类型 | 推荐 MAX_STEPS | 原因 |
|----------|----------------|------|
| 简单问答 | 5-10 | 几轮对话足够 |
| 代码生成 | 20-30 | 需要思考+写+检查 |
| 研究任务 | 50-100 | 需要多轮搜索和整合 |
| 开放探索 | 100+ | 需要大量尝试 |

---

## 8. 面试高频问题

### 8.1 基础概念

**Q1: 什么是 Agent Runtime？它解决什么问题？**

A: Agent Runtime 是 Agent 系统的执行引擎，主要解决：
1. **执行可控性**: 统一的状态机管理
2. **可观测性**: 所有操作可追踪
3. **可恢复性**: 检查点支持中断恢复
4. **资源管理**: 预算控制和限流

**Q2: Policy-Step-Reducer 模式的优势是什么？**

A:
1. **可预测性**: 状态变化集中管理
2. **可测试性**: 组件独立测试
3. **可扩展性**: 新增组件不影响现有代码
4. **可调试性**: 清晰的数据流

**Q3: 如何防止 Agent 陷入死循环？**

A: 多重保护：
1. `max_steps` 限制总步数
2. `consecutive_no_progress` 检测无进展
3. `ProgressDetector` 检测重复和循环
4. 预算限制（Token、成本、时间）

### 8.2 设计决策

**Q4: 为什么选择 JSONL 而不是数据库存储检查点？**

A:
- **优点**: 简单、无依赖、人类可读、便于调试
- **缺点**: 无查询、无并发控制
- **适用**: 开发和测试环境
- **生产建议**: 可切换到 PostgreSQL 或 Redis

**Q5: 为什么 Reducer 是异步的？**

A:
1. 可能需要持久化状态到数据库
2. 可能需要发送通知或触发 webhook
3. 保持与其他异步组件的一致性
4. 便于未来扩展

**Q6: Hook 系统的设计目的是什么？**

A: 提供扩展点，不修改核心代码：
```python
class MetricsHook:
    async def on_step_complete(self, state, step_result, trace_ctx):
        metrics.record_step_duration(step_result.duration_ms)
        metrics.record_tokens(step_result.llm_response.usage.total_tokens)

class NotificationHook:
    async def on_run_end(self, state, trace_ctx):
        if state.status == ExecutionStatus.FAILED:
            await slack.send_alert(f"Agent {state.run_id} failed")
```

### 8.3 实现细节

**Q7: 哈希计算为什么要排除某些字段？**

A: 排除"易变"字段：
- `start_time`: 每次运行都不同
- `elapsed_ms`: 实时变化
- 包含它们会导致：
  - 相同逻辑状态哈希不同
  - 检查点去重失效
  - 恢复时验证失败

**Q8: 如何处理 LLM 响应解析失败？**

A: 分层处理：
```python
# 1. 尝试结构化解析
thought, action = parse_structured(response)

# 2. 降级到全文作为 thought
if not thought and not action:
    thought = response.content

# 3. 记录解析失败事件
trace_writer.write(TraceEvent(
    event_type=EventType.ERROR,
    metadata={"parse_failure": True, "raw_content": response.content},
))
```

**Q9: consecutive_errors 和 consecutive_no_progress 有什么区别？**

A:
| 指标 | consecutive_errors | consecutive_no_progress |
|------|-------------------|------------------------|
| 触发条件 | 捕获到异常 | 正常执行但检测到无进展 |
| 典型原因 | API 错误、超时、格式错误 | 循环、重复、相似输出 |
| 处理方式 | 可能需要重试或切换 Provider | 可能需要调整 Prompt 或策略 |
| 对应 StopReason | ERROR | NO_PROGRESS |

### 8.4 场景设计

**Q10: 如何实现 Agent 的暂停和恢复？**

A:
```python
# 暂停
state = state_manager.transition(state, ExecutionStatus.PAUSED)
snapshot = await state_manager.checkpoint(state, trace_ctx, reason="user_pause")

# 恢复
snapshot = await state_manager.load_snapshot(run_id)
state_manager.verify_snapshot(snapshot)
state = state_manager.transition(snapshot.state, ExecutionStatus.RUNNING)
# 继续执行循环...
```

**Q11: 如何支持多 Agent 协作？**

A: 设计思路：
```python
# 1. 共享状态
shared_memory = SharedMemory()  # Redis/DB

# 2. Agent 间通信
class Orchestrator:
    agents: dict[str, Agent]

    async def delegate(self, task: str, agent_name: str):
        agent = self.agents[agent_name]
        result = await agent.run(task)
        shared_memory.update(result)

# 3. 任务队列
task_queue = TaskQueue()
while task := await task_queue.pop():
    agent = select_agent(task)
    await agent.run(task)
```

**Q12: 如何实现 Agent 的 A/B 测试？**

A:
```python
# 1. 实验配置
experiments = {
    "control": RuntimeConfig(max_steps=50),
    "treatment": RuntimeConfig(max_steps=100, checkpoint_interval_steps=10),
}

# 2. 分流
experiment_group = hash(user_id) % 2
config = experiments["control" if experiment_group == 0 else "treatment"]

# 3. 指标收集
metrics.record("experiment", {
    "group": experiment_group,
    "run_id": run_id,
    "success": state.status == ExecutionStatus.COMPLETED,
    "steps": state.current_step,
    "tokens": state.tokens_used,
})
```

---

## 9. 代码细节考点

### 9.1 类型安全

```python
# 使用 TYPE_CHECKING 避免循环导入
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcana.gateway.registry import ModelGatewayRegistry

class Agent:
    def __init__(self, gateway: ModelGatewayRegistry): ...
```

### 9.2 懒加载模式

```python
class Agent:
    def __init__(self):
        self._step_executor: StepExecutor | None = None

    @property
    def step_executor(self) -> StepExecutor:
        if self._step_executor is None:
            from arcana.runtime.step import StepExecutor
            self._step_executor = StepExecutor(...)
        return self._step_executor
```

**优点**:
- 避免循环导入
- 延迟初始化，减少启动时间
- 可注入测试替身

### 9.3 协议类 vs 抽象类

```python
# Protocol: 结构化类型（鸭子类型）
@runtime_checkable
class RuntimeHook(Protocol):
    async def on_run_start(self, state, trace_ctx) -> None: ...

# ABC: 名义类型（显式继承）
class BasePolicy(ABC):
    @abstractmethod
    async def decide(self, state) -> PolicyDecision: ...
```

**选择建议**:
- 核心组件（Policy, Reducer）用 ABC，强制实现
- 可选扩展（Hook）用 Protocol，更灵活

### 9.4 Pydantic 配置

```python
class ModelConfig(BaseModel):
    # 禁用命名空间保护（允许 model_id 字段）
    model_config = {"protected_namespaces": ()}

    model_id: str  # 不会警告
```

### 9.5 异常层次

```python
class RuntimeError(Exception):
    """基础异常"""
    def __init__(self, message: str, recoverable: bool = False):
        self.message = message
        self.recoverable = recoverable

class StateTransitionError(RuntimeError):
    """状态转换错误 - 不可恢复"""

class StepExecutionError(RuntimeError):
    """步骤执行错误 - 可能可恢复"""
```

---

## 10. 扩展阅读

### 10.1 论文

1. **ReAct**: [Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
   - Policy-Step-Reducer 的理论基础

2. **Reflexion**: [Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
   - 自我反思和错误纠正

3. **Tree of Thoughts**: [Deliberate Problem Solving with LLMs](https://arxiv.org/abs/2305.10601)
   - 多路径探索策略

### 10.2 开源项目

1. **LangChain**: [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
   - Agent 框架参考

2. **AutoGPT**: [github.com/Significant-Gravitas/AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
   - 自主 Agent 实现

3. **CrewAI**: [github.com/joaomdmoura/crewAI](https://github.com/joaomdmoura/crewAI)
   - 多 Agent 协作

### 10.3 设计模式

1. **State Pattern**: 状态机实现
2. **Strategy Pattern**: Policy 策略选择
3. **Observer Pattern**: Hook 系统
4. **Command Pattern**: PolicyDecision
5. **Memento Pattern**: StateSnapshot

### 10.4 进一步学习

- [ ] 实现 Plan-Execute-Verify 模式
- [ ] 添加向量记忆和检索
- [ ] 实现多 Agent 协作
- [ ] 添加人机协作（Human-in-the-loop）
- [ ] 实现评估和基准测试框架

---

## 总结

Agent Runtime 是构建可靠 Agent 系统的核心基础设施。掌握以下关键点：

1. **架构**: Policy-Step-Reducer 分层设计
2. **状态**: AgentState 是执行的核心数据结构
3. **控制**: 多重停止条件保证安全边界
4. **恢复**: 检查点支持中断和调试
5. **监控**: Trace 和 Hook 提供可观测性

面试时重点展示：
- 设计决策的权衡考量
- 对边界情况的处理
- 代码的可测试性和可扩展性
