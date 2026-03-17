# Part 1: 架构总览与 Contracts 层

> Arcana Agent Framework 深度教程 - 第一部分

---

## 目录

1. [框架总览](#1-框架总览)
2. [架构分层](#2-架构分层)
3. [Contracts 层详解](#3-contracts-层详解)
4. [Trace 系统](#4-trace-系统)
5. [Canonical Hashing](#5-canonical-hashing)
6. [配置系统](#6-配置系统)
7. [本章小结](#7-本章小结)

---

## 1. 框架总览

### 1.1 什么是 Arcana

Arcana 是一个 **contracts-first**（合约优先）的 Agent 平台框架。它的核心理念可以用三个词概括：

- **可控** — 通过预算（Budget）、策略（Policy）、护栏（Guard）实现对 Agent 行为的精确控制
- **可复现** — 通过 Canonical Hashing 和 Trace 日志，任意一次运行都能被精确重放和审计
- **可评测** — 内置 Eval 合约，支持回归门控（Regression Gate），保证每次迭代不退化

### 1.2 为什么选择 Contracts-First

传统 Agent 框架往往先写实现再补接口，导致数据结构散落各处、难以跨语言迁移。Arcana 反其道而行：**先定义所有 Pydantic Schema，再写实现**。这带来几个关键优势：

1. **跨语言迁移** — 合约即规范，未来迁移到 Go/Rust 只需按 Schema 重新实现，上层逻辑不变
2. **团队协作** — 前后端、不同模块的开发者只需对齐合约，可以完全并行开发
3. **测试先行** — Schema 定义好后，测试用例可以提前编写，实现代码一完成就能验证
4. **文档即代码** — Pydantic 模型自带类型检查和字段约束，本身就是最精确的文档

### 1.3 设计哲学

Arcana 遵循以下设计原则：

- **每个数据流过的 "关节" 都有明确的 Schema** — 不存在裸 dict 在模块间传递
- **所有副作用可追踪** — 工具调用标注 side_effect，LLM 请求/响应记录 digest
- **预算是一等公民** — 从 LLM 调用到任务编排，每一层都有独立的预算约束
- **幂等性内建** — ToolCall 支持 idempotency_key，避免重复执行

---

## 2. 架构分层

### 2.1 整体架构图

```
┌───────────────────────────────────────────────────────────────┐
│                    Application Layer                          │
│               (Agents / User Applications)                    │
├───────────────────────────────────────────────────────────────┤
│                   Platform Services Layer                     │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────────────┐   │
│  │  Model   │ │   Tool   │ │ Memory │ │   Orchestrator   │   │
│  │ Gateway  │ │ Gateway  │ │ System │ │   / Multi-Agent  │   │
│  └────┬─────┘ └────┬─────┘ └───┬────┘ └────────┬─────────┘   │
├───────┼─────────────┼──────────┼───────────────┼─────────────┤
│       │    Contracts Layer (Pydantic Schemas)   │             │
│  ┌────┴───┐  ┌──────┴──┐  ┌───┴────┐  ┌───────┴──────────┐  │
│  │llm.py  │  │tool.py  │  │state.py│  │plan.py / rag.py  │  │
│  │        │  │         │  │        │  │memory.py/eval.py  │  │
│  └────────┘  └─────────┘  └────────┘  │orchestrator.py    │  │
│                                       │multi_agent.py     │  │
│                                       │graph.py           │  │
│                                       └───────────────────┘  │
├───────────────────────────────────────────────────────────────┤
│                Infrastructure Layer                           │
│  ┌─────────────────────┐  ┌──────────────────────────────┐   │
│  │  Trace (JSONL Audit) │  │  Utils (Hashing / Config)   │   │
│  │  writer.py/reader.py│  │  hashing.py / config.py      │   │
│  └─────────────────────┘  └──────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

依赖方向严格**自上而下**，下层模块绝不依赖上层：

```
trace.py ← (无外部依赖，是最底层合约)
   ↑
llm.py, tool.py, state.py ← (基础合约，互不依赖)
   ↑
runtime.py ← (依赖 llm.py, tool.py)
   ↑
plan.py, graph.py ← (依赖基础合约)
   ↑
rag.py, memory.py ← (独立的领域合约)
   ↑
multi_agent.py ← (依赖 trace.py 的 AgentRole)
   ↑
orchestrator.py ← (任务编排，依赖基础合约)
   ↑
eval.py ← (评测层，依赖基础合约)
```

关键依赖示例：`runtime.py` 导入了 `llm.py` 和 `tool.py`：

```python
# src/arcana/contracts/runtime.py:L10-L11
from arcana.contracts.llm import LLMResponse
from arcana.contracts.tool import ToolResult
```

`multi_agent.py` 依赖 `trace.py` 的 `AgentRole`：

```python
# src/arcana/contracts/multi_agent.py:L12
from arcana.contracts.trace import AgentRole
```

### 2.3 公共导出接口

`__init__.py` 扮演 "门面" 角色，精选导出核心类型，让使用者无需记忆具体文件路径：

```python
# src/arcana/contracts/__init__.py:L3-L25
from arcana.contracts.llm import (
    Budget, LLMRequest, LLMResponse, Message, ModelConfig, TokenUsage,
)
from arcana.contracts.runtime import (
    PolicyDecision, RuntimeConfig, StepResult, StepType,
)
from arcana.contracts.state import AgentState, ExecutionStatus, StateSnapshot
from arcana.contracts.tool import ToolCall, ToolResult, ToolSpec
from arcana.contracts.trace import (
    BudgetSnapshot, StopReason, ToolCallRecord, TraceContext, TraceEvent,
)
```

注意：`plan.py`、`rag.py`、`memory.py` 等高级合约**没有**在 `__init__.py` 中导出，说明它们属于"按需引用"的领域模块，不属于核心公共 API。这是一个有意的分层设计。

---

## 3. Contracts 层详解

### 3.1 trace.py — 追踪合约

**文件路径**: `src/arcana/contracts/trace.py`

这是整个框架最底层的合约文件，定义了审计追踪所需的全部数据模型。

#### 核心枚举

**StopReason** — Agent 停止运行的原因分类：

```python
# src/arcana/contracts/trace.py:L13-L24
class StopReason(str, Enum):
    GOAL_REACHED = "goal_reached"
    MAX_STEPS = "max_steps"
    MAX_TIME = "max_time"
    MAX_COST = "max_cost"
    MAX_TOKENS = "max_tokens"
    NO_PROGRESS = "no_progress"
    ERROR = "error"
    USER_CANCELLED = "user_cancelled"
    TOOL_BLOCKED = "tool_blocked"
```

设计意图：停止原因被显式枚举而非用字符串，保证了在 trace 分析时可以精确过滤和统计。`NO_PROGRESS` 和 `TOOL_BLOCKED` 是 Agent 特有的停止原因，反映了框架对"死循环"和"权限阻塞"场景的预判。

**AgentRole** — Agent 角色分类：

```python
# src/arcana/contracts/trace.py:L27-L33
class AgentRole(str, Enum):
    SYSTEM = "system"
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
```

设计意图：这四个角色对应经典的 Plan-Execute-Critique 三阶段 Agent 架构。`SYSTEM` 用于框架内部事件。角色定义在 trace.py 而非 multi_agent.py，因为**每个 trace 事件都需要标注角色**。

**EventType** — 事件类型，涵盖三大类：

```python
# src/arcana/contracts/trace.py:L36-L58
class EventType(str, Enum):
    # 核心事件
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    STATE_CHANGE = "state_change"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    PLAN = "plan"
    VERIFY = "verify"
    MEMORY_WRITE = "memory_write"

    # 编排器事件
    TASK_SUBMIT = "task_submit"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAIL = "task_fail"

    # 图引擎事件
    GRAPH_NODE_START = "graph_node_start"
    GRAPH_NODE_COMPLETE = "graph_node_complete"
    GRAPH_TRANSITION = "graph_transition"
    GRAPH_INTERRUPT = "graph_interrupt"
```

设计意图：事件类型按子系统分组——核心事件（8 种）、编排器事件（4 种）、图引擎事件（4 种）。这种分组让 trace 文件在不同运行模式下保持一致的结构。

#### 核心数据模型

**BudgetSnapshot** — 预算快照，记录某一时刻的资源消耗状态：

```python
# src/arcana/contracts/trace.py:L61-L89
class BudgetSnapshot(BaseModel):
    # 限制
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_time_ms: int | None = None

    # 已消耗
    tokens_used: int = 0
    cost_usd: float = 0.0
    time_ms: int = 0

    @property
    def budget_exhausted(self) -> bool:
        if self.max_tokens and self.tokens_used >= self.max_tokens:
            return True
        if self.max_cost_usd and self.cost_usd >= self.max_cost_usd:
            return True
        if self.max_time_ms and self.time_ms >= self.max_time_ms:
            return True
        return False
```

设计意图：预算有三个维度（token/费用/时间），任一维度超限即视为耗尽。`None` 表示该维度不限制。这是典型的"多维度约束取并集"模式。

生产注意点：`budget_exhausted` 使用的是 `and` 短路逻辑——如果 `max_tokens` 为 `0`（falsy），条件不会触发。这意味着**设置为 0 等同于不设限**，而非"零预算"。

**ToolCallRecord** — 工具调用的追踪记录：

```python
# src/arcana/contracts/trace.py:L92-L101
class ToolCallRecord(BaseModel):
    name: str
    args_digest: str          # 参数的 Canonical Hash
    idempotency_key: str | None = None
    result_digest: str | None = None
    error: str | None = None
    duration_ms: int | None = None
    side_effect: str | None = None  # "read" or "write"
```

设计意图：存储的是参数和结果的 **digest**（摘要哈希）而非原始数据。这有两个好处：(1) trace 文件体积可控，不会因为大型工具输出而膨胀；(2) 通过比对 digest 可以检测参数/结果是否一致，支持幂等性验证。

**TraceContext** — 追踪上下文，贯穿一次 run 的标识信息：

```python
# src/arcana/contracts/trace.py:L104-L113
class TraceContext(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str | None = None
    parent_step_id: str | None = None

    def new_step_id(self) -> str:
        return str(uuid4())
```

设计意图：`run_id` 是一次执行的全局标识，`task_id` 用于多任务编排，`parent_step_id` 支持步骤嵌套。这三级标识构成了 trace 的"坐标系统"。

**TraceEvent** — 追踪事件，是 trace 系统的核心实体：

```python
# src/arcana/contracts/trace.py:L116-L153
class TraceEvent(BaseModel):
    # 标识
    run_id: str
    task_id: str | None = None
    step_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 分类
    role: AgentRole = AgentRole.SYSTEM
    event_type: EventType

    # 状态摘要（Canonical JSON SHA-256 截断至 16 字符）
    state_before_hash: str | None = None
    state_after_hash: str | None = None

    # LLM 相关
    llm_request_digest: str | None = None
    llm_response_digest: str | None = None
    model: str | None = None

    # 工具相关
    tool_call: ToolCallRecord | None = None

    # 预算
    budgets: BudgetSnapshot | None = None

    # 停止信息
    stop_reason: StopReason | None = None
    stop_detail: str | None = None

    # 附加上下文
    metadata: dict[str, Any] = Field(default_factory=dict)
```

设计意图：TraceEvent 是一个**宽表结构** — 同一个模型通过不同的 `event_type` 和字段组合表达不同语义的事件。相比为每种事件定义独立模型，宽表的好处是 JSONL 格式统一，便于序列化和查询。缺点是某些事件只用到部分字段，其余为 None。

关键设计：`state_before_hash` 和 `state_after_hash` 是**状态指纹**，用于检测状态变更。如果两个连续事件的 `state_after_hash` 相同，说明该步骤没有产生有效状态变更——这正是 `NO_PROGRESS` 检测的基础。

---

### 3.2 llm.py — LLM 合约

**文件路径**: `src/arcana/contracts/llm.py`

定义了与大语言模型交互的全部数据结构。

#### 核心数据模型

**ModelConfig** — 模型配置：

```python
# src/arcana/contracts/llm.py:L51-L64
class ModelConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: Literal["gemini", "deepseek", "openai", "anthropic", "ollama"]
    model_id: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    max_tokens: int = Field(default=4096, gt=0)
    timeout_ms: int = Field(default=30000, gt=0)
    extra_params: dict[str, Any] = Field(default_factory=dict)
```

设计意图：
- `provider` 使用 `Literal` 枚举，编译期即可检测非法 provider
- `temperature` 默认 `0.0`，保证可复现性（这是框架"可复现"理念的体现）
- `seed` 支持随机种子，进一步增强可复现性
- `extra_params` 作为扩展点，支持 provider 特有参数而不污染核心 Schema

生产注意点：`model_config = {"protected_namespaces": ()}` 是因为 Pydantic V2 默认保护 `model_` 前缀的字段名，而 `model_id` 会触发警告。这行配置禁用了该保护。

**Message** — 对话消息：

```python
# src/arcana/contracts/llm.py:L20-L26
class Message(BaseModel):
    role: MessageRole  # system / user / assistant / tool
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
```

设计意图：这是 OpenAI Chat Completion 格式的精简映射。`content` 可以为 None（当 assistant 消息只包含 tool_calls 时）。`tool_call_id` 用于关联工具调用与返回结果。

**TokenUsage** — Token 使用统计：

```python
# src/arcana/contracts/llm.py:L29-L40
class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost_estimate(self) -> float:
        return (self.prompt_tokens * 0.001 + self.completion_tokens * 0.002) / 1000
```

生产注意点：`cost_estimate` 使用的是硬编码的价格估算，仅作为占位符。生产环境中应根据具体 provider 和 model 动态计算。

**Budget** — 预算约束（用于单次 LLM 调用）：

```python
# src/arcana/contracts/llm.py:L43-L48
class Budget(BaseModel):
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_time_ms: int | None = None
```

注意 `Budget` 与 `BudgetSnapshot` 的区别：`Budget` 是约束定义（"最多用多少"），`BudgetSnapshot` 是运行时快照（"已经用了多少/还剩多少"）。

**LLMRequest / LLMResponse** — 请求/响应对：

```python
# src/arcana/contracts/llm.py:L75-L97
class LLMRequest(BaseModel):
    messages: list[Message]
    response_schema: dict[str, Any] | None = None  # JSON Schema for structured output
    tools: list[dict[str, Any]] | None = None
    budget: Budget | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class LLMResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    content: str | None = None
    tool_calls: list[ToolCallRequest] | None = None
    usage: TokenUsage
    model: str
    finish_reason: str
    raw_response: dict[str, Any] | None = None
```

设计意图：
- `response_schema` 支持 Structured Output（结构化输出），让 LLM 返回符合 JSON Schema 的响应
- `raw_response` 保留原始响应数据，用于调试和 trace 分析
- `finish_reason` 直接透传 provider 返回值（"stop"、"tool_calls" 等）

---

### 3.3 tool.py — 工具合约

**文件路径**: `src/arcana/contracts/tool.py`

定义了工具网关（Tool Gateway）的核心数据结构。

#### 核心枚举

```python
# src/arcana/contracts/tool.py:L11-L24
class SideEffect(str, Enum):
    READ = "read"
    WRITE = "write"
    NONE = "none"

class ErrorType(str, Enum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    REQUIRES_HUMAN = "requires_human"
```

设计意图：`SideEffect` 将工具按副作用分类，这对安全策略至关重要 — 一个 READ 工具可以自动执行，而 WRITE 工具可能需要确认。`ErrorType` 的三分法（可重试/不可重试/需人工）驱动了上层的重试策略。

#### 核心数据模型

**ToolSpec** — 工具规格说明：

```python
# src/arcana/contracts/tool.py:L27-L43
class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]   # JSON Schema
    output_schema: dict[str, Any] | None = None

    side_effect: SideEffect = SideEffect.READ
    requires_confirmation: bool = False
    capabilities: list[str] = Field(default_factory=list)

    max_retries: int = 3
    retry_delay_ms: int = 1000
    timeout_ms: int = 30000
```

设计意图：
- `input_schema` / `output_schema` 使用 JSON Schema，与 LLM 的 function calling 格式对齐
- `requires_confirmation` 支持人机协作场景（Agent 调用危险工具前需要人工确认）
- 重试配置内建在 ToolSpec 而非运行时，让每个工具可以独立配置重试策略

**ToolCall** — 工具调用请求：

```python
# src/arcana/contracts/tool.py:L46-L56
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]
    idempotency_key: str | None = None
    run_id: str | None = None
    step_id: str | None = None
```

设计意图：`idempotency_key` 是幂等性的关键 — 相同 key 的调用保证只执行一次。`run_id` 和 `step_id` 用于关联 trace 上下文。注意 `arguments` 是 `dict` 而非 JSON string（对比 `ToolCallRequest.arguments` 是 `str`），因为到达 Tool Gateway 时参数已经被解析。

**ToolResult** — 工具执行结果：

```python
# src/arcana/contracts/tool.py:L72-L96
class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    success: bool
    output: Any | None = None
    error: ToolError | None = None
    duration_ms: int | None = None
    retry_count: int = 0

    @property
    def output_str(self) -> str:
        if self.error:
            return f"Error: {self.error.message}"
        if self.output is None:
            return "Success (no output)"
        if isinstance(self.output, str):
            return self.output
        import json
        return json.dumps(self.output, ensure_ascii=False)
```

设计意图：`output_str` 属性提供了一个统一的字符串化接口，让上层（特别是 LLM 消息构建）不需要处理各种 output 类型。`retry_count` 记录实际重试次数，用于后续的性能分析。

---

### 3.4 state.py — 状态合约

**文件路径**: `src/arcana/contracts/state.py`

定义了 Agent 运行时的状态管理数据结构。

#### 核心数据模型

**AgentState** — Agent 当前运行状态：

```python
# src/arcana/contracts/state.py:L23-L69
class AgentState(BaseModel):
    # 标识
    run_id: str
    task_id: str | None = None

    # 执行跟踪
    status: ExecutionStatus = ExecutionStatus.PENDING
    current_step: int = 0
    max_steps: int = 100

    # 目标和进度
    goal: str | None = None
    current_plan: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)

    # 工作记忆（键值存储）
    working_memory: dict[str, Any] = Field(default_factory=dict)

    # 对话历史
    messages: list[dict[str, Any]] = Field(default_factory=list)

    # 预算跟踪
    tokens_used: int = 0
    cost_usd: float = 0.0
    start_time: datetime | None = None
    elapsed_ms: int = 0

    # 错误跟踪
    last_error: str | None = None
    consecutive_errors: int = 0
    consecutive_no_progress: int = 0
```

设计意图：AgentState 是一个**单一状态对象**，包含了 Agent 运行所需的全部上下文。这种设计让状态管理变得显式和可序列化 — 随时可以将状态快照保存下来，用于恢复或分析。

关键字段解析：
- `working_memory` — 运行期间的临时键值存储，类似于人的"工作记忆"
- `consecutive_errors` / `consecutive_no_progress` — 连续错误/无进展计数器，驱动自动停止逻辑
- `current_plan` / `completed_steps` — 计划和完成步骤的简单列表形式

生产注意点：`messages` 使用 `list[dict[str, Any]]` 而非 `list[Message]`，这是为了兼容不同来源的消息格式（例如 raw provider responses）。

**StateSnapshot** — 状态快照，用于检查点：

```python
# src/arcana/contracts/state.py:L72-L92
class StateSnapshot(BaseModel):
    run_id: str
    step_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    state_hash: str           # 完整性校验哈希
    state: AgentState         # 完整状态数据

    checkpoint_reason: str = ""  # "interval", "error", "plan_step", "verification", "budget"
    plan_progress: dict[str, Any] = Field(default_factory=dict)
    is_resumable: bool = True
```

设计意图：`state_hash` 用于完整性校验（通过 canonical_hash 计算），确保快照在存储/传输过程中没有被篡改。`checkpoint_reason` 记录触发检查点的原因，便于分析哪些场景最频繁触发快照。`is_resumable` 标记该快照是否可用于恢复执行。

---

### 3.5 runtime.py — 运行时合约

**文件路径**: `src/arcana/contracts/runtime.py`

定义了 Agent 运行时引擎的数据结构。

#### 核心枚举

**StepType** — 执行步骤类型：

```python
# src/arcana/contracts/runtime.py:L14-L21
class StepType(str, Enum):
    THINK = "think"      # LLM 推理
    ACT = "act"          # 工具执行
    OBSERVE = "observe"  # 处理结果
    PLAN = "plan"        # 规划步骤
    VERIFY = "verify"    # 验证步骤
```

设计意图：这是经典 ReAct（Reasoning + Acting）模式的扩展。原始 ReAct 只有 Think-Act-Observe 三步，Arcana 加入了 Plan 和 Verify，形成 **Plan-Think-Act-Observe-Verify** 的完整循环。

#### 核心数据模型

**StepResult** — 单步执行结果：

```python
# src/arcana/contracts/runtime.py:L24-L48
class StepResult(BaseModel):
    step_type: StepType
    step_id: str
    success: bool

    thought: str | None = None
    action: str | None = None
    observation: str | None = None

    llm_response: LLMResponse | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)

    state_updates: dict[str, Any] = Field(default_factory=dict)
    memory_updates: dict[str, Any] = Field(default_factory=dict)

    error: str | None = None
    is_recoverable: bool = True
```

设计意图：StepResult 统一了所有步骤类型的结果。`thought`/`action`/`observation` 三个字段对应 ReAct 的三个阶段，但不是每个步骤都会用到全部字段。`is_recoverable` 让上层决策器知道是否值得重试。

**PolicyDecision** — 策略决策，决定下一步做什么：

```python
# src/arcana/contracts/runtime.py:L51-L68
class PolicyDecision(BaseModel):
    action_type: str  # "llm_call", "tool_call", "complete", "fail"

    prompt_template: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)

    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    stop_reason: str | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

设计意图：PolicyDecision 是**策略模式（Strategy Pattern）** 的数据载体。Agent 的行为不是硬编码的，而是由 Policy 对象根据当前状态生成 PolicyDecision，再由 Runtime 执行。`reasoning` 字段记录策略的决策理由，用于调试和审计。

**RuntimeConfig** — 运行时配置：

```python
# src/arcana/contracts/runtime.py:L71-L94
class RuntimeConfig(BaseModel):
    max_steps: int = 100
    max_consecutive_errors: int = 3
    max_consecutive_no_progress: int = 3

    checkpoint_interval_steps: int = 5
    checkpoint_on_error: bool = True
    checkpoint_budget_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 0.9]
    )
    checkpoint_on_plan_step: bool = True
    checkpoint_on_verification: bool = True

    step_retry_count: int = 2
    step_retry_delay_ms: int = 1000

    progress_window_size: int = 5
    similarity_threshold: float = 0.95
```

设计意图：
- `checkpoint_budget_thresholds: [0.5, 0.75, 0.9]` — 在预算消耗到 50%、75%、90% 时自动创建检查点，这是一种**渐进式安全网**策略
- `progress_window_size` + `similarity_threshold` — 通过比较最近 N 步的状态相似度检测"无进展"（死循环），相似度超过 95% 视为无进展
- `max_consecutive_errors: 3` — 连续 3 次错误即停止，避免无谓的 API 消耗

---

### 3.6 plan.py — 计划合约

**文件路径**: `src/arcana/contracts/plan.py`

定义了结构化规划（Plan-and-Execute）的数据模型。

#### 核心数据模型

**PlanStep** — 计划中的单个步骤：

```python
# src/arcana/contracts/plan.py:L21-L30
class PlanStep(BaseModel):
    id: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)  # 前置步骤 ID
    status: PlanStepStatus = PlanStepStatus.PENDING
    result: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

设计意图：每个步骤都有 `acceptance_criteria`（验收标准）和 `dependencies`（依赖关系）。这不是简单的线性任务列表，而是一个**DAG（有向无环图）**——步骤可以声明对其他步骤的依赖，只有依赖完成才能开始。

**Plan** — 结构化执行计划：

```python
# src/arcana/contracts/plan.py:L33-L49
class Plan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    goal: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)  # 全局验收标准

    def next_step(self) -> PlanStep | None:
        completed_ids = {s.id for s in self.steps if s.status == PlanStepStatus.COMPLETED}
        for step in self.steps:
            if step.status != PlanStepStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in step.dependencies):
                return step
        return None
```

设计意图：`next_step()` 实现了一个简单的 DAG 调度器 — 遍历所有 PENDING 步骤，找到依赖全部满足的第一个。这比手动管理执行顺序更安全，因为依赖关系由 LLM 在规划阶段生成，执行引擎自动遵守。

Plan 还提供了丰富的进度查询属性：

```python
# src/arcana/contracts/plan.py:L60-L82
@property
def is_complete(self) -> bool:
    return all(
        s.status in (PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED)
        for s in self.steps
    )

@property
def progress_ratio(self) -> float:
    if not self.steps:
        return 0.0
    done = sum(
        1 for s in self.steps
        if s.status in (PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED)
    )
    return done / len(self.steps)
```

**GoalVerificationResult** — 目标验证结果：

```python
# src/arcana/contracts/plan.py:L93-L101
class GoalVerificationResult(BaseModel):
    outcome: VerificationOutcome  # PASSED / FAILED / PARTIAL
    criteria_results: dict[str, bool] = Field(default_factory=dict)
    coverage: float = 0.0
    failed_criteria: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)  # 重新规划建议
```

设计意图：验证不是简单的"通过/失败"，而是有三态（PASSED/FAILED/PARTIAL）。`suggestions` 字段让验证器可以给出重新规划的建议，实现 Plan-Execute-Verify-Replan 的闭环。

---

### 3.7 graph.py — 图引擎合约

**文件路径**: `src/arcana/contracts/graph.py`

定义了图执行引擎（Graph Engine）的数据模型，类似 LangGraph 的设计。

#### 核心数据模型

```python
# src/arcana/contracts/graph.py:L11-L17
class NodeType(str, Enum):
    FUNCTION = "function"
    LLM = "llm"
    TOOL = "tool"
    SUBGRAPH = "subgraph"
```

设计意图：四种节点类型覆盖了图引擎的主要场景。`SUBGRAPH` 支持图的嵌套组合，这是构建复杂工作流的关键。

**GraphNodeSpec / GraphEdgeSpec** — 图的节点和边：

```python
# src/arcana/contracts/graph.py:L20-L42
class GraphNodeSpec(BaseModel):
    id: str
    name: str
    node_type: NodeType = NodeType.FUNCTION
    metadata: dict[str, Any] = Field(default_factory=dict)

class GraphEdgeSpec(BaseModel):
    source: str
    target: str  # "__end__" 表示终止
    edge_type: EdgeType = EdgeType.DIRECT

class ConditionalEdgeSpec(BaseModel):
    source: str
    path_map: dict[str, str] = Field(default_factory=dict)
    # path_fn 单独存储（不可序列化）
```

设计意图：`ConditionalEdgeSpec` 将路径映射（可序列化的 `path_map`）和路径函数（不可序列化的 `path_fn`）分离。这样图的结构可以被序列化为 JSON，而执行逻辑在运行时注入。`"__end__"` 是一个哨兵值，表示终止节点。

**GraphConfig** — 图配置，支持断点调试：

```python
# src/arcana/contracts/graph.py:L52-L58
class GraphConfig(BaseModel):
    name: str = "default"
    entry_point: str = ""
    interrupt_before: list[str] = Field(default_factory=list)
    interrupt_after: list[str] = Field(default_factory=list)
```

设计意图：`interrupt_before` / `interrupt_after` 支持在指定节点前后暂停执行，这对于人机协作（human-in-the-loop）至关重要 — 可以在关键节点前让人类审核。

**GraphExecutionState** — 图执行状态：

```python
# src/arcana/contracts/graph.py:L61-L68
class GraphExecutionState(BaseModel):
    current_node: str | None = None
    visited_nodes: list[str] = Field(default_factory=list)
    node_outputs: dict[str, Any] = Field(default_factory=dict)
    is_interrupted: bool = False
    interrupt_node: str | None = None
```

---

### 3.8 rag.py — RAG 合约

**文件路径**: `src/arcana/contracts/rag.py`

定义了检索增强生成（Retrieval-Augmented Generation）的完整数据模型。

#### 核心数据模型

**Document / Chunk / Citation** — 文档、分块、引用三层结构：

```python
# src/arcana/contracts/rag.py:L21-L61
class Document(BaseModel):
    id: str
    source: str           # 来源（URL、文件路径等）
    content: str
    timestamp: datetime
    tags: list[str]
    content_type: str = "text/plain"

class Chunk(BaseModel):
    id: str
    document_id: str      # 关联到源文档
    content: str
    start_offset: int = 0
    end_offset: int = 0
    embedding: list[float] | None = None
    embedding_model: str | None = None

class Citation(BaseModel):
    source: str
    chunk_id: str | None = None
    document_id: str | None = None
    snippet: str = ""     # 相关文本摘录
    score: float = 0.0
    retrieved_at: datetime
```

设计意图：三层结构清晰地分离了关注点 — Document 是原始资料，Chunk 是向量化的最小单元，Citation 是答案引用的来源证据。`Chunk.embedding` 直接存储向量，避免了需要另外的 embedding 存储。

**RetrievalQuery / RetrievalResult / RetrievalResponse** — 检索链路：

```python
# src/arcana/contracts/rag.py:L64-L91
class RetrievalQuery(BaseModel):
    query: str
    top_k: int = 10
    filters: dict[str, Any] | None = None
    rerank: bool = True
    min_score: float = 0.0

class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    content: str
    citation: Citation     # 每个结果自带引用信息

class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    total_candidates: int = 0  # Rerank 前的候选数量
```

设计意图：`rerank: bool = True` 默认开启重排序，体现了"质量优先"的理念。`total_candidates` 记录重排序前的候选数量，用于评估检索召回率。

**RAGAnswer** — 带引用的 RAG 回答：

```python
# src/arcana/contracts/rag.py:L111-L121
class RAGAnswer(BaseModel):
    content: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    retrieved_chunks: list[str] = Field(default_factory=list)

    query_digest: str | None = None
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)
```

设计意图：`query_digest` 使用 canonical_hash 存储查询摘要，支持缓存和去重。`retrieval_trace` 记录检索过程的详细信息，便于调试。

**VerificationResult** — 引用验证：

```python
# src/arcana/contracts/rag.py:L124-L131
class VerificationResult(BaseModel):
    valid: bool
    coverage: float = 0.0          # 有引用支持的声明比例
    unsupported_claims: list[str]  # 无引用支持的声明
    weak_citations: list[str]      # 弱引用
```

设计意图：这是防止 RAG 幻觉的关键合约 — 通过验证每个声明是否有引用支持，量化输出的可信度。

---

### 3.9 memory.py — 记忆合约

**文件路径**: `src/arcana/contracts/memory.py`

定义了 Agent 记忆系统的三种记忆类型。

#### 核心设计：三种记忆类型

```python
# src/arcana/contracts/memory.py:L12-L17
class MemoryType(str, Enum):
    WORKING = "working"      # 短期、运行范围的键值存储
    LONG_TERM = "long_term"  # 持久化、向量索引的事实
    EPISODIC = "episodic"    # 事件轨迹日志
```

设计意图：这三种记忆类型对应认知科学中的分类：
- **工作记忆** — 类似人的短期记忆，仅在当前运行有效
- **长期记忆** — 持久化的事实知识，跨运行保留
- **情景记忆** — 过往经历的记录，用于从历史中学习

#### 核心数据模型

**MemoryEntry** — 记忆条目：

```python
# src/arcana/contracts/memory.py:L20-L48
class MemoryEntry(BaseModel):
    id: str
    memory_type: MemoryType
    key: str
    content: str

    # 治理元数据
    confidence: float = 1.0        # 0.0 到 1.0
    source: str = ""               # "tool", "llm", "user", "step_result" 等
    source_run_id: str | None = None
    source_step_id: str | None = None

    # 生命周期
    created_at: datetime
    updated_at: datetime

    # 撤销（软删除，保留历史）
    revoked: bool = False
    revoked_at: datetime | None = None
    revoked_reason: str | None = None

    # 完整性
    content_hash: str = ""
```

设计意图：
- `confidence` — 记忆的可信度评分，低置信度的记忆可以被过滤或降权
- `source` / `source_run_id` / `source_step_id` — 完整的溯源链，知道每条记忆是从哪个步骤产生的
- **软删除而非硬删除** — `revoked` 标记为废弃但不删除，保留完整的记忆演变历史
- `content_hash` — 内容哈希，用于去重和完整性校验

**MemoryQuery** — 记忆查询，支持多种检索方式：

```python
# src/arcana/contracts/memory.py:L51-L61
class MemoryQuery(BaseModel):
    query: str = ""                     # 语义搜索（长期记忆）
    key: str | None = None              # 精确键查找（工作记忆）
    memory_type: MemoryType | None = None
    tags: list[str] = Field(default_factory=list)
    run_id: str | None = None
    top_k: int = 10
    min_confidence: float = 0.0
    include_revoked: bool = False       # 默认隐藏已撤销的条目
```

设计意图：同一个查询模型支持两种检索模式 — 精确键查找（工作记忆场景）和语义搜索（长期记忆场景）。`include_revoked = False` 默认隐藏已撤销条目，但保留查看历史的能力。

**MemoryConfig** — 记忆系统配置：

```python
# src/arcana/contracts/memory.py:L95-L109
class MemoryConfig(BaseModel):
    min_write_confidence: float = 0.5     # 低于此值拒绝写入
    warn_confidence_threshold: float = 0.7 # 低于此值发出警告
    working_namespace_prefix: str = "wm"
    embedding_model: str = "text-embedding-ada-002"
    max_episodic_results: int = 50
```

设计意图：双阈值（`min_write_confidence` 和 `warn_confidence_threshold`）实现了分级管控 — 低置信度直接拒绝，中等置信度允许写入但发出警告，高置信度正常写入。这防止了 Agent 将不确定的推断写入记忆污染知识库。

---

### 3.10 multi_agent.py — 多 Agent 协作合约

**文件路径**: `src/arcana/contracts/multi_agent.py`

定义了多 Agent 协作的消息传递和会话管理。

#### 核心数据模型

**AgentMessage** — Agent 间消息：

```python
# src/arcana/contracts/multi_agent.py:L25-L35
class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    sender_role: AgentRole
    recipient_role: AgentRole
    message_type: MessageType  # plan / result / feedback / handoff / escalate
    content: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
```

设计意图：消息按类型分为五种 — 计划(plan)、结果(result)、反馈(feedback)、交接(handoff)、升级(escalate)。`sender_role` 和 `recipient_role` 使用 trace.py 中的 `AgentRole`，保持了角色定义的单一来源。

**CollaborationSession** — 协作会话：

```python
# src/arcana/contracts/multi_agent.py:L38-L52
class CollaborationSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    roles: list[AgentRole] = Field(
        default_factory=lambda: [
            AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.CRITIC,
        ]
    )
    max_rounds: int = 5
    shared_memory_ns: str = ""
    status: str = "active"
```

设计意图：默认角色配置 Planner-Executor-Critic 体现了经典的三方协作模式。`max_rounds` 防止无限对话。`shared_memory_ns` 提供共享记忆命名空间，让 Agent 之间可以通过记忆系统交换信息。

---

### 3.11 orchestrator.py — 编排器合约

**文件路径**: `src/arcana/contracts/orchestrator.py`

定义了任务调度和编排的数据结构。

#### 核心数据模型

**Task** — 编排器中的工作单元：

```python
# src/arcana/contracts/orchestrator.py:L40-L57
class Task(BaseModel):
    id: str
    goal: str
    dependencies: list[str] = Field(default_factory=list)
    priority: int = 0             # 数值越大越紧急
    budget: TaskBudget | None = None
    deadline: datetime | None = None
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    attempt: int = 0
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
```

设计意图：Task 与 PlanStep 的区别在于 Task 是**编排层的工作单元**，具有更丰富的调度属性 — 优先级、截止时间、重试策略、独立预算。`dependencies` 同样构成 DAG，但调度逻辑更复杂（需要考虑并发、优先级等）。

**RetryPolicy** — 重试策略：

```python
# src/arcana/contracts/orchestrator.py:L23-L29
class RetryPolicy(BaseModel):
    max_retries: int = 0
    delay_ms: int = 1000
    backoff_multiplier: float = 2.0
    max_delay_ms: int = 30000
```

设计意图：指数退避（exponential backoff）是生产环境重试的最佳实践。`max_delay_ms` 设置上限防止退避时间过长。默认 `max_retries = 0` 表示不重试，需要显式启用。

**OrchestratorConfig** — 编排器配置：

```python
# src/arcana/contracts/orchestrator.py:L73-L81
class OrchestratorConfig(BaseModel):
    max_concurrent_tasks: int = 4
    global_max_tokens: int | None = None
    global_max_cost_usd: float | None = None
    global_max_time_ms: int | None = None
    default_retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    scheduling_interval_ms: int = 100
```

设计意图：`max_concurrent_tasks = 4` 限制并发任务数，防止资源争抢。全局预算（`global_max_tokens` 等）在所有任务间共享，是 Agent 系统成本控制的最高层护栏。

---

### 3.12 eval.py — 评测合约

**文件路径**: `src/arcana/contracts/eval.py`

定义了评测体系（Eval Harness）和回归门控（Regression Gate）的数据结构。

#### 核心数据模型

**EvalCase** — 单个评测用例：

```python
# src/arcana/contracts/eval.py:L22-L31
class EvalCase(BaseModel, frozen=True):
    id: str
    goal: str
    expected_outcome: OutcomeCriterion
    expected_value: Any = None
    max_steps: int = 50
    timeout_ms: int = 30_000
    metadata: dict[str, Any] = Field(default_factory=dict)
```

设计意图：`frozen=True` 使 EvalCase **不可变**，这对评测至关重要 — 测试用例一旦定义就不应被修改，保证评测结果的可比性。`OutcomeCriterion` 支持多种评测维度（状态、停止原因、步数、费用等）。

**EvalReport** — 评测报告：

```python
# src/arcana/contracts/eval.py:L48-L59
class EvalReport(BaseModel):
    suite_name: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[EvalResult] = Field(default_factory=list)
    aggregate_tokens: int = 0
    aggregate_cost_usd: float = 0.0
    aggregate_duration_ms: int = 0
```

**GateConfig / RegressionResult** — 回归门控：

```python
# src/arcana/contracts/eval.py:L62-L78
class GateConfig(BaseModel, frozen=True):
    min_pass_rate: float = 0.9          # 最低通过率 90%
    max_regression_pct: float = 0.05    # 最大回归幅度 5%
    max_avg_cost_usd: float | None = None
    max_avg_tokens: int | None = None

class RegressionResult(BaseModel):
    passed: bool
    current_pass_rate: float
    baseline_pass_rate: float | None = None
    regression_pct: float | None = None
    gate_violations: list[str] = Field(default_factory=list)
```

设计意图：回归门控是 CI/CD 中的质量关卡。默认配置要求通过率不低于 90%，且相比基线的回退不超过 5%。`gate_violations` 列出所有违反的条件，便于定位问题。这让 Agent 系统的迭代有了量化保障 — 每次修改都不会让整体表现退化超过可接受范围。

---

## 4. Trace 系统

### 4.1 概述

Trace 系统是 Arcana 的"飞行记录仪"，基于 JSONL（JSON Lines）格式，记录 Agent 运行过程中的每一个事件。每次运行生成一个独立的 trace 文件：`{trace_dir}/{run_id}.jsonl`。

### 4.2 TraceWriter — 写入器

**文件路径**: `src/arcana/trace/writer.py`

```python
# src/arcana/trace/writer.py:L14-L20
class TraceWriter:
    """
    Writes trace events to JSONL files.
    Each run gets its own trace file: {trace_dir}/{run_id}.jsonl
    Events are appended atomically with file locking.
    """
```

#### 初始化与目录管理

```python
# src/arcana/trace/writer.py:L22-L39
def __init__(self, trace_dir: str | Path = "./traces", enabled: bool = True):
    self.trace_dir = Path(trace_dir)
    self.enabled = enabled
    self._lock = threading.Lock()

    if self.enabled:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
```

设计意图：
- `enabled` 参数支持全局关闭 trace，零开销
- `threading.Lock()` 保证多线程环境下的写入原子性
- 自动创建 trace 目录

#### 核心写入方法

```python
# src/arcana/trace/writer.py:L45-L63
def write(self, event: TraceEvent) -> None:
    if not self.enabled:
        return

    trace_file = self._get_trace_file(event.run_id)
    event_json = event.model_dump_json()

    with self._lock:
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(event_json + "\n")
```

设计意图：每个事件序列化为一行 JSON，追加写入文件。使用 `model_dump_json()` 而非手动 `json.dumps()`，利用 Pydantic V2 的高性能序列化。锁的粒度是整个 write 操作，保证一行 JSON 不会被其他线程打断。

生产注意点：当前使用的是 Python 线程锁，仅保证**进程内**的线程安全。如果多个进程同时写入同一个 trace 文件（如分布式部署），需要升级为文件锁（`fcntl.flock`）或使用消息队列。

#### Raw 写入

```python
# src/arcana/trace/writer.py:L65-L86
def write_raw(self, run_id: str, data: dict[str, Any]) -> None:
    if not self.enabled:
        return

    trace_file = self._get_trace_file(run_id)

    if "timestamp" not in data:
        data["timestamp"] = datetime.now(UTC).isoformat()

    json_line = json.dumps(data, ensure_ascii=False, default=str)

    with self._lock:
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")
```

设计意图：`write_raw` 允许写入非 TraceEvent 格式的数据，用于扩展性场景（如自定义事件、provider 原始响应等）。自动补充 timestamp 确保所有行都有时间戳。

### 4.3 TraceReader — 读取器

**文件路径**: `src/arcana/trace/reader.py`

#### 事件读取

```python
# src/arcana/trace/reader.py:L41-L69
def read_events(self, run_id: str) -> list[TraceEvent]:
    trace_file = self._get_trace_file(run_id)
    if not trace_file.exists():
        return []

    events = []
    with open(trace_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    events.append(TraceEvent.model_validate(data))
                except (json.JSONDecodeError, ValueError):
                    continue  # 跳过格式错误的行

    events.sort(key=lambda e: e.timestamp)
    return events
```

设计意图：
- 容错处理 — 格式错误的行被静默跳过（`continue`），保证部分损坏的 trace 文件仍可读取
- 按时间戳排序 — 即使写入顺序因并发而乱序，读取结果始终按时间有序

#### 流式迭代

```python
# src/arcana/trace/reader.py:L97-L119
def iter_events(self, run_id: str) -> Iterator[TraceEvent]:
    trace_file = self._get_trace_file(run_id)
    if not trace_file.exists():
        return

    with open(trace_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    yield TraceEvent.model_validate(data)
                except (json.JSONDecodeError, ValueError):
                    continue
```

设计意图：`iter_events` 使用 generator（yield），不将所有事件加载到内存。对于大型 trace 文件（百万级事件），这是必须的。

#### 多维过滤

```python
# src/arcana/trace/reader.py:L121-L156
def filter_events(
    self,
    run_id: str,
    event_types: list[EventType] | None = None,
    roles: list[AgentRole] | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> list[TraceEvent]:
    events = self.read_events(run_id)

    if event_types:
        events = [e for e in events if e.event_type in event_types]
    if roles:
        events = [e for e in events if e.role in roles]
    if start_time:
        events = [e for e in events if e.timestamp >= start_time]
    if end_time:
        events = [e for e in events if e.timestamp <= end_time]

    return events
```

设计意图：链式过滤器，每个维度可独立启用。这是一种简洁的"可组合过滤"模式。

#### 运行摘要

```python
# src/arcana/trace/reader.py:L191-L239
def get_summary(self, run_id: str) -> dict[str, Any]:
    events = self.read_events(run_id)
    if not events:
        return {"run_id": run_id, "exists": False}

    llm_calls = [e for e in events if e.event_type == EventType.LLM_CALL]
    tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
    errors = [e for e in events if e.event_type == EventType.ERROR]

    # 查找停止原因
    stop_event = next((e for e in reversed(events) if e.stop_reason), None)

    # 统计 token 和费用（取最大值，因为是累计值）
    total_tokens = 0
    total_cost = 0.0
    for e in events:
        if e.budgets:
            total_tokens = max(total_tokens, e.budgets.tokens_used)
            total_cost = max(total_cost, e.budgets.cost_usd)

    return {
        "run_id": run_id,
        "exists": True,
        "total_events": len(events),
        "llm_calls": len(llm_calls),
        "tool_calls": len(tool_calls),
        "errors": len(errors),
        "unique_steps": len(seen_steps),
        "start_time": events[0].timestamp.isoformat(),
        "end_time": events[-1].timestamp.isoformat(),
        "stop_reason": stop_event.stop_reason.value if stop_event and stop_event.stop_reason else None,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
    }
```

设计意图：摘要方法提供了一次运行的"仪表盘"视图 — LLM 调用次数、工具调用次数、错误数、总 token、总费用、停止原因等。`total_tokens` 取 `max` 而非 `sum`，因为 `BudgetSnapshot.tokens_used` 是累计值而非增量值。

---

## 5. Canonical Hashing

**文件路径**: `src/arcana/utils/hashing.py`

### 5.1 为什么需要 Canonical JSON

JSON 规范不要求键的顺序。同一个对象，`{"a":1,"b":2}` 和 `{"b":2,"a":1}` 语义等价但字符串不同，导致 SHA-256 哈希不同。Canonical JSON 通过排序键和统一格式解决这个问题，保证**相同数据始终产生相同哈希**。

### 5.2 值规范化

```python
# src/arcana/utils/hashing.py:L13-L40
def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if value != value:  # NaN 检测
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return round(value, 6)
    if isinstance(value, Decimal):
        return float(round(value, 6))
    if isinstance(value, BaseModel):
        return _normalize_value(value.model_dump())
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return str(value)
```

关键设计决策：
1. **bool 在 int 之前检查** — Python 中 `isinstance(True, int)` 返回 `True`，如果不先检查 bool，`True` 会被当作 `1` 处理
2. **浮点数取 6 位小数** — 消除浮点精度差异（如 `0.1 + 0.2 = 0.30000000000000004`）
3. **NaN/Infinity 转为字符串** — JSON 规范不支持这些特殊值，转为字符串是安全的做法
4. **Pydantic BaseModel 自动展开** — 通过 `model_dump()` 转为 dict 再递归处理
5. **dict 键排序** — `sorted(value.items())` 保证键顺序一致

### 5.3 哈希生成与验证

```python
# src/arcana/utils/hashing.py:L43-L76
def canonical_json(obj: Any) -> str:
    normalized = _normalize_value(obj)
    return json.dumps(
        normalized,
        sort_keys=True,       # 双重保险的键排序
        separators=(",", ":"),  # 最紧凑格式，无空格
        ensure_ascii=False,    # 支持 Unicode
    )

def canonical_hash(obj: Any, length: int = 16) -> str:
    json_str = canonical_json(obj)
    hash_bytes = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    return hash_bytes[:length]

def verify_hash(obj: Any, expected_hash: str, length: int = 16) -> bool:
    actual_hash = canonical_hash(obj, length)
    return actual_hash == expected_hash
```

设计意图：
- 截断到 16 字符（64 位）— 对于 trace 场景足够唯一，同时大幅减少存储空间
- `separators=(",", ":")` 使用最紧凑的 JSON 格式，消除空格差异
- `sort_keys=True` 与 `_normalize_value` 中的排序形成双重保险

生产注意点：16 字符的 hex 提供 64 位的碰撞空间，对于单次运行内的对象（通常不超过百万级）几乎不可能碰撞。但如果用于全局唯一性（如跨系统 ID），建议增加到 32 字符。

---

## 6. 配置系统

**文件路径**: `src/arcana/utils/config.py`

### 6.1 配置模型

```python
# src/arcana/utils/config.py:L44-L56
class ArcanaConfig(BaseModel):
    gemini: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    deepseek: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    openai: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    anthropic: ModelProviderConfig = Field(default_factory=ModelProviderConfig)

    trace: TraceConfig = Field(default_factory=TraceConfig)
    default_model: DefaultModelConfig = Field(default_factory=DefaultModelConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
```

设计意图：所有配置集中在一个 Pydantic 模型中，支持类型检查和默认值。Provider 配置是扁平的（每个 provider 一个字段），而非嵌套的字典，让 IDE 自动补全和类型检查更友好。

### 6.2 环境变量加载

```python
# src/arcana/utils/config.py:L59-L126
def load_config(env_file: str | Path | None = None) -> ArcanaConfig:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    return ArcanaConfig(
        gemini=ModelProviderConfig(
            api_key=get_env("GEMINI_API_KEY"),
            base_url=get_env("GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai"),
        ),
        # ... 其他 provider 配置
    )
```

设计意图：使用 `dotenv` 加载环境变量，支持 `.env` 文件。每个环境变量都有合理的默认值。

### 6.3 单例模式

```python
# src/arcana/utils/config.py:L129-L144
_config: ArcanaConfig | None = None

def get_config() -> ArcanaConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config

def reset_config() -> None:
    global _config
    _config = None
```

设计意图：懒加载单例 — 首次调用时加载配置，之后返回缓存实例。`reset_config()` 用于测试中重置配置状态。

生产注意点：单例模式在多线程环境下存在竞态条件（两个线程同时判断 `_config is None`）。如果框架在多线程中使用，建议加锁或使用 `threading.local`。

---

## 7. 本章小结

### 7.1 核心设计模式回顾

| 模式 | 应用场景 | 示例 |
|------|----------|------|
| Contracts-First | 全局 | 所有模块先定义 Pydantic Schema |
| 宽表事件 | Trace | TraceEvent 用一个模型表达多种事件 |
| DAG 调度 | Plan / Orchestrator | PlanStep.dependencies, Task.dependencies |
| 三级标识 | 追踪 | run_id > task_id > step_id |
| 摘要而非原文 | Trace | args_digest / result_digest / state_hash |
| 软删除 | Memory | revoked 标记而非物理删除 |
| 多维约束取并集 | Budget | token/cost/time 任一超限即停 |
| 双阈值管控 | Memory | min_write_confidence / warn_confidence_threshold |
| 指数退避 | Orchestrator | RetryPolicy.backoff_multiplier |

### 7.2 关键要点

1. **Contracts 是系统的骨架** — 13 个合约文件定义了 60+ 个数据模型，覆盖了 Agent 系统的所有数据流
2. **依赖严格分层** — trace.py 在最底层，eval.py 在最上层，中间层按功能域划分
3. **可复现性贯穿始终** — temperature=0.0、seed、canonical hash、trace digest 多重保障
4. **预算是一等公民** — 从单次 LLM 调用（Budget）到运行级（BudgetSnapshot）到任务级（TaskBudget）到全局（OrchestratorConfig），四级预算控制
5. **Trace 是审计基石** — JSONL 格式简单高效，写入原子安全，读取支持流式和过滤

### 7.3 下一步

Part 2 将深入 Model Gateway 层，了解 Arcana 如何通过 OpenAI 兼容协议统一接入多家 LLM Provider，以及预算追踪器（BudgetTracker）如何在运行时执行预算约束。
