# Part 3: Runtime 引擎 -- Agent 执行的心脏

> 本章深入剖析 Arcana 的 Runtime 模块。Runtime 是整个 Agent 框架的核心执行引擎,它将 LLM 调用、工具执行、状态管理和策略决策编排成一个完整的自治循环。理解 Runtime 就是理解 "Agent 如何思考和行动"。

---

## 目录

1. [Runtime 引擎概述](#1-runtime-引擎概述)
2. [Agent 主循环](#2-agent-主循环)
3. [策略模式 (Policy)](#3-策略模式-policy)
4. [步骤执行器 (StepExecutor)](#4-步骤执行器-stepexecutor)
5. [状态归约器 (Reducer)](#5-状态归约器-reducer)
6. [状态管理器 (StateManager)](#6-状态管理器-statemanager)
7. [进度检测 (ProgressDetector)](#7-进度检测-progressdetector)
8. [输出验证 (Validator)](#8-输出验证-validator)
9. [Replay 引擎](#9-replay-引擎)
10. [错误处理](#10-错误处理)
11. [Hook 系统](#11-hook-系统)
12. [验证器 (Verifier)](#12-验证器-verifier)
13. [数据流图](#13-数据流图)
14. [生产注意事项](#14-生产注意事项)
15. [本章小结](#15-本章小结)

---

## 1. Runtime 引擎概述

### 什么是 Runtime?

Runtime 是 Arcana 的**状态机执行引擎**。它不是简单地调用一次 LLM 然后返回结果,而是构建了一个完整的 **"感知-决策-执行-更新"** 循环,让 Agent 能够自主地朝目标迈进。

从模块入口 (`src/arcana/runtime/__init__.py`) 可以看到 Runtime 的核心组件:

```python
# src/arcana/runtime/__init__.py:9-26
from arcana.runtime.agent import Agent
from arcana.runtime.step import StepExecutor
from arcana.runtime.state_manager import StateManager
from arcana.runtime.progress import ProgressDetector
from arcana.runtime.policies import BasePolicy, ReActPolicy
from arcana.runtime.reducers import BaseReducer, DefaultReducer
from arcana.runtime.hooks import RuntimeHook
```

### 状态机执行模型

Runtime 将 Agent 的执行建模为一个**有限状态机**。每个 Agent 在其生命周期中经历以下状态转换:

```
                  ┌──────────────────────────────────┐
                  │         状态转换图                 │
                  └──────────────────────────────────┘

    ┌─────────┐     run()     ┌─────────┐
    │ PENDING │──────────────>│ RUNNING │
    └─────────┘               └────┬────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    v              v              v
              ┌──────────┐  ┌───────────┐  ┌───────────┐
              │  PAUSED  │  │ COMPLETED │  │  FAILED   │
              └─────┬────┘  └───────────┘  └───────────┘
                    │                             ^
                    │  resume()                   │
                    └──────> RUNNING ─────────────┘
                                │
                                v
                          ┌───────────┐
                          │ CANCELLED │
                          └───────────┘
```

状态转换规则定义在 `state_manager.py:21-33`:

```python
# src/arcana/runtime/state_manager.py:21-33
VALID_TRANSITIONS: dict[ExecutionStatus, set[ExecutionStatus]] = {
    ExecutionStatus.PENDING: {ExecutionStatus.RUNNING},
    ExecutionStatus.RUNNING: {
        ExecutionStatus.PAUSED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
    },
    ExecutionStatus.PAUSED: {ExecutionStatus.RUNNING, ExecutionStatus.CANCELLED},
    ExecutionStatus.COMPLETED: set(),   # 终态,不可转换
    ExecutionStatus.FAILED: set(),      # 终态,不可转换
    ExecutionStatus.CANCELLED: set(),   # 终态,不可转换
}
```

**为什么采用状态机模型?**

1. **确定性**: 每个状态只能转换到预定义的下一组状态,防止非法跳转
2. **可恢复性**: PAUSED 状态允许 Agent 暂停后恢复,支持长时间运行的任务
3. **审计友好**: 状态转换都被追踪,便于事后分析
4. **终态安全**: COMPLETED/FAILED/CANCELLED 是终态,一旦进入不可逆转

### 组件协作关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent (编排器)                            │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │  Policy   │  │ StepExecutor│  │   Reducer    │  │  Hooks  │ │
│  │ (决定做什么)│  │ (执行操作)    │  │ (更新状态)    │  │ (扩展点) │ │
│  └──────┬───┘  └──────┬──────┘  └──────┬───────┘  └────┬────┘ │
│         │              │               │                │      │
│         v              v               v                v      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              StateManager (状态管理 + 检查点)              │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────v───────────────────────────────┐  │
│  │            ProgressDetector (循环检测 + 进度监控)           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Agent 主循环

### 初始化

Agent 类采用**依赖注入**模式,所有核心组件通过构造函数传入:

```python
# src/arcana/runtime/agent.py:42-74
class Agent:
    def __init__(
        self,
        *,
        policy: BasePolicy,           # 策略:决定每一步做什么
        reducer: BaseReducer,         # 归约器:决定状态如何更新
        gateway: ModelGatewayRegistry, # LLM 网关
        config: RuntimeConfig | None = None,
        trace_writer: TraceWriter | None = None,
        budget_tracker: BudgetTracker | None = None,
        tool_gateway: ToolGateway | None = None,
        hooks: list[RuntimeHook] | None = None,
    ) -> None:
        self.policy = policy
        self.reducer = reducer
        self.gateway = gateway
        self.config = config or RuntimeConfig()
        # ... 其余字段
```

**设计要点**: 内部组件 (`StepExecutor`, `StateManager`, `ProgressDetector`) 采用**延迟初始化** (lazy initialization):

```python
# src/arcana/runtime/agent.py:84-96
@property
def step_executor(self) -> StepExecutor:
    """Get or create step executor."""
    if self._step_executor is None:
        from arcana.runtime.step import StepExecutor
        self._step_executor = StepExecutor(
            gateway=self.gateway,
            tool_gateway=self.tool_gateway,
            trace_writer=self.trace_writer,
            budget_tracker=self.budget_tracker,
        )
    return self._step_executor
```

**为什么延迟初始化?**
- 避免循环导入 (import cycle)
- 如果某些组件未使用,不会浪费资源创建
- 使 Agent 的构造更轻量,测试更方便

### 执行循环 (run loop)

`run()` 方法是 Agent 的入口。它实现了一个经典的 **while-true 循环**:

```python
# src/arcana/runtime/agent.py:122-196 (简化)
async def run(self, goal: str, *, initial_state=None, task_id=None) -> AgentState:
    # 1. 初始化状态
    state = initial_state or self._create_initial_state(goal, task_id)
    state = self.state_manager.transition(state, ExecutionStatus.RUNNING)

    # 2. 触发生命周期钩子
    await self._call_hooks("on_run_start", state, trace_ctx)

    try:
        while True:
            # 3. 检查停止条件
            stop_reason = self._check_stop_conditions(state)
            if stop_reason:
                state = await self._handle_stop(state, stop_reason, trace_ctx)
                break

            # 4. 执行单步
            step_result = await self._execute_step(state, trace_ctx)

            # 5. 通过 Reducer 更新状态
            state = await self.reducer.reduce(state, step_result)

            # 6. 更新进度追踪
            self.progress_detector.record_step(step_result)
            if not self.progress_detector.is_making_progress():
                state.consecutive_no_progress += 1
            else:
                state.consecutive_no_progress = 0

            # 7. 检查目标是否达成
            if step_result.state_updates.get("goal_reached"):
                state = await self._handle_stop(state, StopReason.GOAL_REACHED, trace_ctx)
                break

            # 8. 按需创建检查点
            checkpoint_reason = self._should_checkpoint(state, step_result)
            if checkpoint_reason:
                await self.state_manager.checkpoint(state, trace_ctx, reason=checkpoint_reason)

            # 9. 通知 hooks
            await self._call_hooks("on_step_complete", state, step_result, trace_ctx)

    except Exception as e:
        state = await self._handle_error(state, e, trace_ctx)

    await self._call_hooks("on_run_end", state, trace_ctx)
    return state
```

### 每个循环步骤的详细流程

```
┌─────────────────────────────────────────────────────────┐
│                    Agent.run() 主循环                     │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────v───────────┐
          │ _check_stop_conditions │ <── 最大步数? 预算超限? 无进展?
          └────────────┬───────────┘
                       │ (未触发)
          ┌────────────v───────────┐
          │     policy.decide()    │ <── 策略决定下一步做什么
          └────────────┬───────────┘
                       │
          ┌────────────v───────────┐
          │  step_executor.execute │ <── 执行 LLM 调用或工具调用
          └────────────┬───────────┘
                       │
          ┌────────────v───────────┐
          │    reducer.reduce()    │ <── 将结果归约到状态
          └────────────┬───────────┘
                       │
          ┌────────────v───────────┐
          │ progress_detector      │ <── 检测循环/卡死
          │   .record_step()       │
          └────────────┬───────────┘
                       │
          ┌────────────v───────────┐
          │  _should_checkpoint()  │ <── 是否需要保存检查点
          └────────────┬───────────┘
                       │
          ┌────────────v───────────┐
          │  hooks.on_step_complete│ <── 通知所有钩子
          └────────────┬───────────┘
                       │
                       └──> 回到循环顶部
```

### 停止条件

Agent 有多种停止条件,定义在 `_check_stop_conditions` (`agent.py:244-271`):

| 停止原因 | 判断条件 | 说明 |
|----------|----------|------|
| `MAX_STEPS` | `state.has_reached_max_steps` | 达到最大步数上限 |
| `NO_PROGRESS` | 连续无进展 >= 阈值 | Agent 陷入循环 |
| `MAX_TOKENS` | 预算追踪器报告 token 超限 | Token 配额用尽 |
| `MAX_COST` | 预算追踪器报告成本超限 | 费用超出预算 |
| `MAX_TIME` | 预算追踪器报告时间超限 | 执行超时 |
| `ERROR` | 连续错误 >= 阈值 | 持续出错,无法恢复 |
| `GOAL_REACHED` | step_result 标记目标达成 | 任务完成 |

### 检查点策略

Agent 在以下时机创建检查点 (`agent.py:273-317`):

```python
# src/arcana/runtime/agent.py:273-317
def _should_checkpoint(self, state, step_result) -> str | None:
    # 1. 出错时保存
    if not step_result.success and self.config.checkpoint_on_error:
        return "error"

    # 2. 计划步骤完成时保存
    if step_result.state_updates.get("plan_step_completed"):
        return "plan_step"

    # 3. 验证步骤后保存
    if step_result.step_type == StepType.VERIFY:
        return "verification"

    # 4. 固定间隔保存
    if state.current_step % self.config.checkpoint_interval_steps == 0:
        return "interval"

    # 5. 预算阈值保存 (如 50%, 75%, 90%)
    if self.budget_tracker:
        current_ratio = self._get_budget_ratio()
        for threshold in self.config.checkpoint_budget_thresholds:
            if self._last_checkpoint_budget_ratio < threshold <= current_ratio:
                return "budget"
    return None
```

**为什么在预算阈值处保存检查点?** 这是一个精妙的设计 -- 当预算消耗接近上限时,Agent 随时可能被强制停止。提前保存检查点意味着用户可以在之后追加预算并从检查点恢复,而不是从头开始。

### 恢复执行 (resume)

```python
# src/arcana/runtime/agent.py:198-222
async def resume(self, snapshot: StateSnapshot) -> AgentState:
    # 1. 验证快照完整性 (哈希校验)
    self.state_manager.verify_snapshot(snapshot)

    # 2. 重置进度检测器
    self.progress_detector.reset()

    # 3. 从快照状态恢复运行
    return await self.run(
        goal=snapshot.state.goal or "",
        initial_state=snapshot.state,
        task_id=snapshot.state.task_id,
    )
```

---

## 3. 策略模式 (Policy)

### 概念

Policy 是 Agent 的 **"大脑"** -- 它根据当前状态决定下一步该做什么。不同的 Policy 实现了不同的推理范式。

### 基础接口

```python
# src/arcana/runtime/policies/base.py:13-46
class BasePolicy(ABC):
    """A policy decides what action to take given the current state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for identification."""
        ...

    @abstractmethod
    async def decide(self, state: AgentState) -> PolicyDecision:
        """
        Decide the next action based on current state.
        Returns: PolicyDecision describing the next action
        """
        ...

    def build_system_prompt(self, state: AgentState) -> str:
        """Build system prompt for LLM. Override in subclasses."""
        return f"You are a helpful AI assistant. Your goal is: {state.goal}"
```

Policy 的输出是 `PolicyDecision`,它包含:
- `action_type`: `"llm_call"` | `"tool_call"` | `"complete"` | `"verify"` | `"fail"`
- `messages`: 发给 LLM 的消息列表
- `tool_calls`: 要执行的工具调用列表
- `reasoning`: 决策理由
- `metadata`: 附加元数据

### ReActPolicy -- 思考-行动循环

ReAct (Reasoning + Acting) 是最经典的 Agent 推理范式,来自论文 [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)。

```
┌─────────────────────────────────────────┐
│           ReAct 循环                     │
│                                         │
│  ┌──────────┐                           │
│  │  Think   │ Thought: 我应该先搜索...   │
│  └────┬─────┘                           │
│       │                                 │
│  ┌────v─────┐                           │
│  │   Act    │ Action: search("query")   │
│  └────┬─────┘                           │
│       │                                 │
│  ┌────v─────┐                           │
│  │ Observe  │ Observation: 搜索结果...   │
│  └────┬─────┘                           │
│       │                                 │
│       └──> 回到 Think (或 FINISH)        │
└─────────────────────────────────────────┘
```

核心实现 (`react.py:34-92`):

```python
# src/arcana/runtime/policies/react.py:14-31
REACT_SYSTEM_PROMPT = """You are an AI assistant that follows the ReAct framework.

For each step, you must:
1. Think about what to do next
2. Decide on an action (or conclude if the goal is reached)

Format your response EXACTLY as:
Thought: <your reasoning about the current situation and what to do>
Action: <the action to take, or "FINISH" if the goal is achieved>

Goal: {goal}

Previous steps:
{history}

Working memory:
{memory}
"""

# src/arcana/runtime/policies/react.py:45-67
class ReActPolicy(BasePolicy):
    async def decide(self, state: AgentState) -> PolicyDecision:
        history = self._format_history(state)   # 最近 5 步历史
        memory = self._format_memory(state)     # 工作记忆快照

        system_prompt = REACT_SYSTEM_PROMPT.format(
            goal=state.goal or "No goal specified",
            history=history or "No previous steps",
            memory=memory or "Empty",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is your next step?"},
        ]

        return PolicyDecision(
            action_type="llm_call",
            messages=messages,
            reasoning="ReAct step: generate thought and action",
        )
```

**设计细节**:
- 历史窗口限制为最近 5 步 (`react.py:76`),避免 context window 溢出
- 工作记忆中长值被截断到 200 字符 (`react.py:88-89`),保持 prompt 简洁
- 每次 decide() 都只返回一个 `llm_call` 决策,由 StepExecutor 负责执行

### PlanExecutePolicy -- 计划-执行-验证循环

PlanExecutePolicy 实现了更结构化的推理方式,将任务分解为三个阶段:

```
┌──────────────────────────────────────────────────────────────┐
│              Plan-Execute-Verify 三阶段循环                    │
│                                                              │
│  阶段 1: PLAN (规划)                                          │
│  ┌─────────────────────────────────────────────────┐         │
│  │ LLM 生成结构化计划 (JSON)                         │         │
│  │ {goal, acceptance_criteria, steps: [{id, desc}]} │         │
│  └──────────────────────┬──────────────────────────┘         │
│                         │                                    │
│  阶段 2: EXECUTE (执行)  │                                    │
│  ┌──────────────────────v──────────────────────────┐         │
│  │ 逐步执行计划:                                     │         │
│  │   step_1 -> step_2 -> ... -> step_n             │         │
│  │ 每步执行后检查 STEP_COMPLETE                      │         │
│  └──────────────────────┬──────────────────────────┘         │
│                         │                                    │
│  阶段 3: VERIFY (验证)   │                                    │
│  ┌──────────────────────v──────────────────────────┐         │
│  │ 验证所有验收标准是否满足                            │         │
│  │ 通过 -> GOAL_REACHED                             │         │
│  │ 失败 -> 标记 FAILED                               │         │
│  └─────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

三阶段的切换逻辑 (`plan_execute.py:96-122`):

```python
# src/arcana/runtime/policies/plan_execute.py:96-122
async def decide(self, state: AgentState) -> PolicyDecision:
    plan = self._load_plan(state)  # 从 working_memory 加载计划

    if plan is None:
        # 没有计划 -> 进入规划阶段
        return self._plan_phase(state)

    if plan.has_failed:
        # 计划有失败步骤 -> 停止
        return PolicyDecision(action_type="fail", ...)

    if not plan.is_complete:
        # 计划未完成 -> 执行下一步
        next_step = plan.next_step()
        if next_step is not None:
            return self._execute_phase(state, plan, next_step.id)
        # 所有步骤要么完成、要么失败、要么阻塞
        return PolicyDecision(action_type="fail", ...)

    # 所有步骤完成 -> 进入验证阶段
    return self._verify_phase(state, plan)
```

**规划阶段** 要求 LLM 输出结构化 JSON:

```python
# src/arcana/runtime/policies/plan_execute.py:16-41
PLAN_SYSTEM_PROMPT = """You are an AI assistant that follows the Plan-and-Execute framework.

You must create a structured plan to achieve the goal. Respond with a JSON object:
{
  "goal": "<the goal to achieve>",
  "acceptance_criteria": ["<criterion 1>", "<criterion 2>"],
  "steps": [
    {
      "id": "step_1",
      "description": "<what to do>",
      "acceptance_criteria": ["<how to verify this step>"],
      "dependencies": []
    }
  ]
}
"""
```

**执行阶段** 为每个步骤构建独立的 prompt,包含进度信息和验收标准:

```python
# src/arcana/runtime/policies/plan_execute.py:164-214
def _execute_phase(self, state, plan, step_id) -> PolicyDecision:
    # 查找当前步骤
    current_step = next(s for s in plan.steps if s.id == step_id)

    progress = f"{plan.progress_ratio:.0%} ({completed}/{total} steps)"

    system_prompt = EXECUTE_SYSTEM_PROMPT.format(
        goal=state.goal,
        progress=progress,
        step_id=current_step.id,
        step_description=current_step.description,
        step_criteria=criteria_str,
        history=history,
        memory=memory,
    )
    # ...返回 PolicyDecision(action_type="llm_call", ...)
```

**ReAct vs PlanExecute 对比**:

| 特性 | ReAct | PlanExecute |
|------|-------|-------------|
| 规划方式 | 即时的、隐式的 | 预先的、显式的 |
| 适用场景 | 简单任务、探索性任务 | 复杂多步任务 |
| 可恢复性 | 弱 (无结构化进度) | 强 (每步独立可恢复) |
| Token 效率 | 较高 (prompt 简短) | 较低 (需要传递计划) |
| 可解释性 | 一般 | 优秀 (结构化计划可审计) |

---

## 4. 步骤执行器 (StepExecutor)

StepExecutor 是 Runtime 的 **"手"** -- 它负责将 Policy 的决策转化为实际操作。

### 执行流程

```python
# src/arcana/runtime/step.py:69-133
async def execute(self, *, state, decision, trace_ctx) -> StepResult:
    step_id = trace_ctx.new_step_id()

    try:
        if decision.action_type == "llm_call":
            return await self._execute_llm_call(state, decision, step_id, trace_ctx)
        elif decision.action_type == "tool_call":
            return await self._execute_tool_calls(state, decision, step_id, trace_ctx)
        elif decision.action_type == "complete":
            return StepResult(step_type=StepType.VERIFY, success=True,
                              state_updates={"goal_reached": True})
        elif decision.action_type == "verify":
            return await self._execute_verify(state, decision, step_id)
        elif decision.action_type == "fail":
            return StepResult(step_type=StepType.VERIFY, success=False,
                              error=decision.stop_reason, is_recoverable=False)
    except Exception as e:
        return StepResult(success=False, error=str(e),
                          is_recoverable=self._is_recoverable_error(e))
```

### LLM 调用 -- 带验证的重试

这是 StepExecutor 最复杂的部分 (`step.py:135-270`)。核心流程:

```
   LLM 请求
      │
      v
  ┌─────────────┐     验证失败    ┌─────────────────┐
  │  调用 LLM    │──────────────>│  构建重试 prompt   │
  └──────┬──────┘               └────────┬──────────┘
         │                               │
         │ 验证通过                        └──> 重新调用 LLM (最多 3 次)
         v
  ┌─────────────┐
  │  解析响应     │  提取 Thought 和 Action
  └──────┬──────┘
         │
         v
  ┌─────────────┐
  │ 返回结果     │  Action=="FINISH" ? goal_reached : 继续
  └─────────────┘
```

验证支持两种模式:
1. **JSON Schema 验证**: 用 Pydantic 模型验证 LLM 输出的 JSON 结构
2. **结构化格式验证**: 检查 `Thought:` / `Action:` 等必填字段是否存在

```python
# src/arcana/runtime/step.py:163-240 (简化)
for attempt in range(max_attempts):
    response = await self.gateway.generate(request=request, config=self.model_config)

    # JSON Schema 验证
    if expected_schema:
        validation = self.validator.validate_json(response, expected_schema)
        if not validation.valid:
            # 构建重试 prompt,将错误信息反馈给 LLM
            retry_prompt = self.validator.create_retry_prompt(validation)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": retry_prompt})
            continue  # 重试

    # 结构化格式验证
    if validate_structured:
        validation = self.validator.validate_structured_format(response, required_fields)
        if not validation.valid:
            # 同样构建重试 prompt
            continue
    break  # 验证通过
```

**为什么在 StepExecutor 而非 Policy 中做验证?** 因为验证是执行层面的关注点。Policy 只负责 "做什么",StepExecutor 负责 "怎么做" 和 "做对了吗"。这种分离让 Policy 保持简洁。

### 工具执行

工具调用通过 `ToolGateway` 批量执行:

```python
# src/arcana/runtime/step.py:272-319
async def _execute_tool_calls(self, state, decision, step_id, trace_ctx) -> StepResult:
    if self.tool_gateway is None:
        return StepResult(success=False, error="ToolGateway not configured")

    tool_calls = [
        ToolCall(
            id=tc_dict.get("id", str(uuid4())),
            name=tc_dict["name"],
            arguments=tc_dict.get("arguments", {}),
            idempotency_key=tc_dict.get("idempotency_key"),
            run_id=state.run_id,
            step_id=step_id,
        )
        for tc_dict in decision.tool_calls
    ]

    results = await self.tool_gateway.call_many(tool_calls, trace_ctx=trace_ctx)

    all_success = all(r.success for r in results)
    observation = "\n".join(r.output_str for r in results)

    return StepResult(
        step_type=StepType.ACT,
        success=all_success,
        tool_results=results,
        observation=observation,
        is_recoverable=any(r.error and r.error.is_retryable for r in results if not r.success),
    )
```

### LLM 响应解析

`_parse_llm_response` (`step.py:366-422`) 从 LLM 自由文本中提取结构化的 Thought 和 Action:

```python
# src/arcana/runtime/step.py:366-422
def _parse_llm_response(self, content: str) -> tuple[str | None, str | None]:
    """
    Expected format:
    Thought: <reasoning>
    Action: <action to take>
    """
    thought = None
    action = None
    # ... 逐行解析,支持多行 Thought/Action
    # 如果没有结构化格式,将整个内容视为 thought
    if thought is None and action is None:
        thought = content.strip() if content.strip() else None
    return thought, action
```

**容错设计**: 如果 LLM 没有遵循格式要求,整段内容会被当作 thought 处理,而不是报错。这让系统在 LLM 输出不稳定时也能继续运行。

---

## 5. 状态归约器 (Reducer)

### 设计理念

Reducer 的概念借鉴自 Redux -- **给定当前状态和一个事件(步骤结果),产出新状态**。这是一种纯函数式的状态更新模式。

```
(State, StepResult) → Reducer → NewState
```

### 基础接口

```python
# src/arcana/runtime/reducers/base.py:13-43
class BaseReducer(ABC):
    """
    A reducer takes the current state and a step result,
    producing a new state.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def reduce(self, state: AgentState, step_result: StepResult) -> AgentState: ...
```

### DefaultReducer

DefaultReducer 处理所有标准的状态更新 (`reducers/default.py:14-88`):

```python
# src/arcana/runtime/reducers/default.py:29-70
async def reduce(self, state, step_result) -> AgentState:
    # 1. 记录完成的步骤到历史
    step_summary = self._summarize_step(step_result)
    if step_summary:
        state.completed_steps.append(step_summary)

    # 2. 应用状态更新 (如 tokens_used, goal_reached)
    for key, value in step_result.state_updates.items():
        if hasattr(state, key):
            setattr(state, key, value)

    # 3. 应用工作记忆更新 (None 表示删除)
    for key, value in step_result.memory_updates.items():
        if value is None:
            state.working_memory.pop(key, None)
        else:
            state.working_memory[key] = value

    # 4. 追踪错误
    if step_result.success:
        state.consecutive_errors = 0
        state.last_error = None
    else:
        state.consecutive_errors += 1
        state.last_error = step_result.error

    # 5. 追加 LLM 响应到消息历史
    if step_result.llm_response and step_result.llm_response.content:
        state.messages.append({
            "role": "assistant",
            "content": step_result.llm_response.content,
        })
    return state
```

步骤摘要的生成 (`default.py:72-88`):

```python
def _summarize_step(self, step_result) -> str | None:
    parts = []
    if step_result.thought:
        parts.append(f"Thought: {step_result.thought[:100]}")  # 截断到 100 字符
    if step_result.action:
        parts.append(f"Action: {step_result.action}")
    if step_result.observation:
        parts.append(f"Observation: {step_result.observation[:100]}")
    if step_result.error:
        parts.append(f"Error: {step_result.error[:50]}")
    return " | ".join(parts) if parts else None
```

### PlanReducer -- 计划感知的归约器

PlanReducer 继承 DefaultReducer,额外处理计划相关的状态更新 (`reducers/plan_reducer.py:16-137`):

```python
# src/arcana/runtime/reducers/plan_reducer.py:30-57
class PlanReducer(DefaultReducer):
    async def reduce(self, state, step_result) -> AgentState:
        # 先执行父类的标准归约
        state = await super().reduce(state, step_result)

        # 检查是否有新计划数据
        plan_data = step_result.state_updates.get("plan")
        if plan_data is not None:
            self._update_plan(state, plan_data)

        # 检查计划步骤完成信号
        completed_step_id = step_result.memory_updates.get("plan_step_completed")
        if completed_step_id is not None:
            self._mark_step_completed(state, completed_step_id)

        # 更新进度追踪元数据
        self._update_progress(state)
        return state
```

PlanReducer 在 `working_memory` 中维护三个计划追踪字段:

```python
# src/arcana/runtime/reducers/plan_reducer.py:107-115
def _update_progress(self, state) -> None:
    plan = self._load_plan(state)
    if plan is None:
        return
    state.working_memory["plan_progress"] = plan.progress_ratio   # 0.0 ~ 1.0
    state.working_memory["plan_complete"] = plan.is_complete       # bool
    state.working_memory["plan_failed"] = plan.has_failed          # bool
```

**Reducer 的组合模式**: PlanReducer 通过继承 DefaultReducer 并调用 `super().reduce()`,实现了**行为叠加**。这是一种简洁的扩展方式 -- 未来如果需要支持新的推理范式,只需创建新的 Reducer 子类,叠加在 DefaultReducer 之上。

---

## 6. 状态管理器 (StateManager)

StateManager 负责三件事:状态转换验证、检查点创建、完整性校验。

### 状态转换验证

```python
# src/arcana/runtime/state_manager.py:65-88
def transition(self, state, new_status) -> AgentState:
    valid_next = VALID_TRANSITIONS.get(state.status, set())
    if new_status not in valid_next:
        raise StateTransitionError(state.status.value, new_status.value)
    state.status = new_status
    return state
```

这个看似简单的方法其实是系统的**安全阀** -- 它防止了状态的非法跳转。例如,你不能从 COMPLETED 跳回 RUNNING。

### 检查点创建

```python
# src/arcana/runtime/state_manager.py:90-148
async def checkpoint(self, state, trace_ctx, reason="step_complete") -> StateSnapshot:
    # 1. 计算状态哈希 (排除易变字段)
    serializable = state.model_dump(exclude={"start_time", "elapsed_ms"})
    state_hash = canonical_hash(serializable)

    # 2. 捕获计划进度
    plan_progress = {}
    plan_data = state.working_memory.get("plan")
    if plan_data and isinstance(plan_data, dict):
        plan_progress = dict(plan_data)

    # 3. 创建快照
    snapshot = StateSnapshot(
        run_id=state.run_id,
        step_id=trace_ctx.new_step_id(),
        state_hash=state_hash,
        state=state,
        checkpoint_reason=reason,
        plan_progress=plan_progress,
        is_resumable=state.status in {ExecutionStatus.RUNNING, ExecutionStatus.PAUSED},
    )

    # 4. 持久化到 JSONL 文件
    await self._persist_snapshot(snapshot)

    # 5. 记录 trace 事件
    if self.trace_writer:
        event = TraceEvent(event_type=EventType.CHECKPOINT, state_after_hash=state_hash, ...)
        self.trace_writer.write(event)

    return snapshot
```

**存储格式**: 检查点以 **JSONL** (JSON Lines) 格式存储,每个快照一行。文件命名为 `{run_id}.checkpoints.jsonl`。

```
checkpoints/
  └── abc-123-def.checkpoints.jsonl   # 每行一个 StateSnapshot JSON
```

### 完整性校验

恢复检查点时,StateManager 会验证哈希以确保状态未被篡改:

```python
# src/arcana/runtime/state_manager.py:197-218
def verify_snapshot(self, snapshot) -> bool:
    serializable = snapshot.state.model_dump(exclude={"start_time", "elapsed_ms"})
    if not verify_hash(serializable, snapshot.state_hash):
        actual = canonical_hash(serializable)
        raise HashVerificationError(
            expected=snapshot.state_hash,
            actual=actual,
            run_id=snapshot.run_id,
        )
    return True
```

**为什么排除 `start_time` 和 `elapsed_ms`?** 因为这两个字段是易变的 (volatile) -- 恢复时它们的值必然不同于保存时。将它们排除在哈希计算之外,确保了恢复后的状态校验能通过。

---

## 7. 进度检测 (ProgressDetector)

ProgressDetector 是 Agent 的 **"自省机制"** -- 它监控 Agent 是否陷入了死循环或重复行为。

### 三重检测机制

```python
# src/arcana/runtime/progress.py:73-96
def is_making_progress(self) -> bool:
    if len(self._step_hashes) < 2:
        return True  # 数据不足,假定有进展

    # 检测 1: 完全重复的步骤
    if self._has_duplicate_steps():
        return False

    # 检测 2: 循环模式 (A->B->A->B)
    if self._has_cyclic_pattern():
        return False

    # 检测 3: 输出过于相似
    if self._outputs_too_similar():
        return False

    return True
```

### 检测 1: 重复步骤检测

```python
# src/arcana/runtime/progress.py:98-108
def _has_duplicate_steps(self) -> bool:
    last_hash = self._step_hashes[-1]
    duplicates = sum(1 for h in list(self._step_hashes)[:-1] if h == last_hash)
    return duplicates >= 2  # 窗口内出现 3 次相同步骤 = 卡住
```

每个步骤被哈希为 `canonical_hash({thought, action, observation})`。如果同一个哈希在窗口内出现 3 次以上,说明 Agent 在重复同样的思考和行动。

### 检测 2: 循环模式检测

```python
# src/arcana/runtime/progress.py:110-137
def _has_cyclic_pattern(self) -> bool:
    if len(self._action_sequence) < 4:
        return False
    actions = list(self._action_sequence)
    # 检查长度为 2, 3, 4... 的循环
    for cycle_length in range(2, len(actions) // 2 + 1):
        if self._is_repeating_cycle(actions, cycle_length):
            return True
    return False

def _is_repeating_cycle(self, sequence, cycle_length) -> bool:
    recent = sequence[-cycle_length * 2:]
    first_half = recent[:cycle_length]
    second_half = recent[cycle_length:]
    return first_half == second_half  # A-B-A-B 或 A-B-C-A-B-C
```

例如,如果 Agent 的行动序列是 `search -> read -> search -> read`,这就是长度为 2 的循环。

### 检测 3: 输出相似度检测

```python
# src/arcana/runtime/progress.py:139-151
def _outputs_too_similar(self) -> bool:
    outputs = list(self._recent_outputs)
    unique_outputs = set(outputs)
    similarity = 1 - (len(unique_outputs) / len(outputs))
    return similarity >= self.similarity_threshold  # 默认 0.95
```

如果窗口内 95% 以上的输出都相同,说明 Agent 虽然在运行,但产出无变化。

**滑动窗口设计**: 所有检测都基于固定大小的 `deque` 窗口 (默认 5 步),这意味着:
- 内存占用恒定,不随运行时间增长
- 只关注近期行为,允许 Agent 在早期犯错后恢复

---

## 8. 输出验证 (Validator)

OutputValidator (`validator.py`) 确保 LLM 输出符合预期格式,并在失败时自动重试。

### JSON 验证

```python
# src/arcana/runtime/validator.py:41-102
def validate_json(self, response, schema=None) -> ValidationResult:
    content = response.content
    if not content:
        return ValidationResult(valid=False, errors=["Empty response content"])

    # 尝试直接解析 JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # 尝试从 markdown 代码块中提取
        data = self._extract_json_from_markdown(content)
        if data is None:
            return ValidationResult(valid=False, errors=[f"Invalid JSON: {e}"])

    # Pydantic 模型验证
    if schema:
        try:
            validated = schema.model_validate(data)
            return ValidationResult(valid=True, data=validated.model_dump())
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            return ValidationResult(valid=False, errors=errors, data=data)
    return ValidationResult(valid=True, data=data)
```

**智能 JSON 提取**: LLM 经常将 JSON 包裹在 markdown 代码块中。`_extract_json_from_markdown` (`validator.py:183-226`) 能处理多种格式:

```python
# 支持的格式:
# ```json\n{...}\n```
# ```\n{...}\n```
# 或者直接嵌入文本中的 JSON 对象
```

### 重试 Prompt 生成

验证失败时,Validator 构建详细的重试提示:

```python
# src/arcana/runtime/validator.py:150-181
def create_retry_prompt(self, validation_result, schema_description=None) -> str:
    errors_str = "\n".join(f"- {err}" for err in validation_result.errors)

    prompt = f"""Your previous response was invalid. Please try again.

Errors found:
{errors_str}

Your previous response:
{validation_result.raw_content}
"""
    if schema_description:
        prompt += f"\n\nExpected format:\n{schema_description}"
    prompt += "\n\nPlease provide a valid response."
    return prompt
```

这种 "展示错误 + 展示原始输出 + 展示期望格式" 的三段式提示,极大提高了重试成功率。

---

## 9. Replay 引擎

### 设计目标

Replay 引擎 (`replay.py`) 允许**确定性地重放**之前的 Agent 执行过程,用于:
- 调试失败的运行
- 比较两次运行的差异
- 测试时的确定性验证

### ReplayCache

```python
# src/arcana/runtime/replay.py:15-103
class ReplayCache:
    """Cache for storing and retrieving replay data."""

    def load_from_trace(self, reader, run_id) -> None:
        events = reader.read_events(run_id)
        for event in events:
            if event.event_type == EventType.LLM_CALL:
                # 按请求摘要索引 LLM 响应
                if event.llm_request_digest and event.llm_response_digest:
                    self._llm_responses[event.llm_request_digest] = {
                        "content": event.llm_response_content,
                        "model": event.model,
                        "usage": event.llm_usage,
                    }

            elif event.event_type == EventType.TOOL_CALL:
                # 按幂等键索引工具结果
                if event.tool_call and event.tool_result:
                    call_digest = event.tool_call.get("idempotency_key")
                    if call_digest:
                        self._tool_results[call_digest] = event.tool_result
```

**关键设计**: LLM 响应按 **请求摘要** (request_digest) 索引。同样的请求在重放时会返回完全相同的响应,确保执行路径一致。

### 分歧点检测

Replay 引擎最强大的功能之一是**比较两次运行**:

```python
# src/arcana/runtime/replay.py:178-221
async def get_divergence_point(self, run_id_a, run_id_b):
    """Find where two runs diverged."""
    events_a = self.reader.read_events(run_id_a)
    events_b = self.reader.read_events(run_id_b)

    for i in range(min(len(events_a), len(events_b))):
        # 比较状态哈希
        if events_a[i].state_after_hash != events_b[i].state_after_hash:
            return (i, events_a[i], events_b[i])
        # 比较 LLM 响应摘要
        if (events_a[i].llm_response_digest and events_b[i].llm_response_digest
                and events_a[i].llm_response_digest != events_b[i].llm_response_digest):
            return (i, events_a[i], events_b[i])
    return None
```

**使用场景**: 当一个之前成功的任务突然失败时,你可以用 `get_divergence_point()` 找到两次运行开始产生不同结果的精确步骤,然后检查那个步骤的输入 (prompt) 和输出 (LLM response) 差异。

### 逐步状态重建

```python
# src/arcana/runtime/replay.py:223-251
async def get_step_states(self, run_id) -> list[tuple[str, AgentState]]:
    """Get (step_id, state) pairs for each step in a run."""
    events = self.reader.read_events(run_id)
    state = self._reconstruct_initial_state(events)
    step_states = []

    for event in events:
        state = self._apply_event(state, event)
        state_copy = state.model_copy(deep=True)
        step_states.append((event.step_id, state_copy))

    return step_states
```

这让你可以 "时间旅行" -- 查看 Agent 在每个步骤后的完整状态。

---

## 10. 错误处理

### 异常层级

Arcana 定义了细粒度的异常层级 (`exceptions.py`):

```
RuntimeError (基类)
  ├── StateTransitionError   # 非法状态转换
  ├── CheckpointError        # 检查点操作失败
  │   └── HashVerificationError  # 哈希校验失败
  ├── StepExecutionError     # 步骤执行失败
  ├── PolicyError            # 策略决策失败
  └── ProgressStallError     # Agent 停滞
```

每个异常都携带分类信息:

```python
# src/arcana/runtime/exceptions.py:8-24
class ErrorType(str, Enum):
    # 可恢复
    RETRYABLE = "retryable"        # 临时问题 (限流、超时)
    VALIDATION = "validation"      # 格式验证失败
    PARTIAL_FAILURE = "partial_failure"  # 部分成功

    # 不可恢复
    PERMANENT = "permanent"        # 永久性失败
    BUDGET_EXCEEDED = "budget_exceeded"  # 预算超限
    AUTHORIZATION = "authorization"     # 权限不足

    # 需要人工介入
    REQUIRES_HUMAN = "requires_human"
    SAFETY_VIOLATION = "safety_violation"
```

### ErrorHandler -- 智能错误分类

```python
# src/arcana/runtime/error_handler.py:150-212
def _classify_error(self, error: Exception) -> RuntimeError:
    error_str = str(error).lower()

    # 限流和超时 -> 可重试
    if any(x in error_str for x in ["rate limit", "429", "too many requests"]):
        return RuntimeError(str(error), recoverable=True, error_type=ErrorType.RETRYABLE)

    if any(x in error_str for x in ["timeout", "timed out", "503"]):
        return RuntimeError(str(error), recoverable=True, error_type=ErrorType.RETRYABLE)

    # 预算超限 -> 不可恢复
    if any(x in error_str for x in ["budget", "quota", "limit exceeded"]):
        return RuntimeError(str(error), recoverable=False, error_type=ErrorType.BUDGET_EXCEEDED)

    # 授权问题 -> 不可恢复,需要升级
    if any(x in error_str for x in ["unauthorized", "forbidden", "401", "403"]):
        return RuntimeError(str(error), recoverable=False, error_type=ErrorType.AUTHORIZATION)

    # 默认: 永久性错误
    return RuntimeError(str(error), recoverable=False, error_type=ErrorType.PERMANENT)
```

### RetryStrategy -- 指数退避重试

```python
# src/arcana/runtime/error_handler.py:15-65
class RetryStrategy:
    def __init__(self, *, max_attempts=3, initial_delay_ms=1000,
                 max_delay_ms=10000, backoff_multiplier=2.0, jitter=True):
        # ...

    def get_delay(self, attempt: int) -> float:
        delay_ms = min(
            self.initial_delay_ms * (self.backoff_multiplier ** attempt),
            self.max_delay_ms,
        )
        if self.jitter:
            jitter_range = delay_ms * 0.25
            delay_ms += random.uniform(-jitter_range, jitter_range)
        return delay_ms / 1000
```

重试延迟序列 (默认配置): `1s -> 2s -> 4s` (加 25% 随机抖动)

**为什么需要 jitter?** 当多个 Agent 同时遭遇限流时,如果它们都在完全相同的时间重试,会造成 "惊群效应" (thundering herd)。添加随机抖动让重试时间分散开来。

### 错误处理决策矩阵

```python
# src/arcana/runtime/error_handler.py:94-148
async def handle_error(self, error, state, *, context=None) -> tuple[bool, str | None]:
    runtime_error = self._classify_error(error) if not isinstance(error, RuntimeError) else error

    if runtime_error.error_type == ErrorType.RETRYABLE:
        return (True, None)                               # 静默重试

    elif runtime_error.error_type == ErrorType.VALIDATION:
        return (True, f"Validation failed: {msg}")       # 带错误消息重试

    elif runtime_error.error_type == ErrorType.PARTIAL_FAILURE:
        return (False, f"Partial failure: {msg}")        # 继续但记录

    elif runtime_error.error_type in {ErrorType.BUDGET_EXCEEDED, ErrorType.PERMANENT}:
        return (False, msg)                               # 停止

    elif runtime_error.error_type in {ErrorType.REQUIRES_HUMAN, ErrorType.SAFETY_VIOLATION}:
        if self.escalation_callback:
            self.escalation_callback(runtime_error, state)  # 升级给人类
        return (False, f"Escalation required: {msg}")
```

---

## 11. Hook 系统

### 设计理念

Hook 系统采用 **Protocol** (鸭子类型) 而非继承,提供最大灵活性:

```python
# src/arcana/runtime/hooks/base.py:13-63
@runtime_checkable
class RuntimeHook(Protocol):
    """
    Protocol for runtime hooks.
    All methods are optional - implement only what you need.
    """

    async def on_run_start(self, state, trace_ctx) -> None: ...
    async def on_run_end(self, state, trace_ctx) -> None: ...
    async def on_step_complete(self, state, step_result, trace_ctx) -> None: ...
    async def on_checkpoint(self, state, trace_ctx) -> None: ...
    async def on_error(self, state, error, trace_ctx) -> None: ...
```

Agent 在各生命周期节点调用 hooks (`agent.py:404-418`):

```python
# src/arcana/runtime/agent.py:404-418
async def _call_hooks(self, hook_name, *args, **kwargs) -> None:
    for hook in self.hooks:
        method = getattr(hook, hook_name, None)
        if method:
            try:
                await method(*args, **kwargs)
            except Exception:
                pass  # hook 异常不影响主流程
```

**关键设计**: hook 的异常被静默捕获。这是因为 hook 是可选的扩展点,它的失败不应该导致 Agent 执行中断。

### MemoryHook -- 记忆系统集成

MemoryHook (`hooks/memory_hook.py`) 是 Hook 系统的典型应用,它将 Agent 的工作记忆与持久化记忆系统打通:

```python
# src/arcana/runtime/hooks/memory_hook.py:17-27
class MemoryHook:
    """
    - on_run_start: 从持久化存储加载工作记忆到 AgentState
    - on_step_complete: 将 StepResult 中的 memory_updates 持久化
    - on_run_end: 在成功完成时,将标记的条目提升为长期记忆
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory_manager = memory_manager
```

**运行开始时** -- 加载持久化记忆:

```python
# src/arcana/runtime/hooks/memory_hook.py:29-38
async def on_run_start(self, state, trace_ctx) -> None:
    entries = await self.memory_manager.working.get_all(state.run_id)
    for key, entry in entries.items():
        if key not in state.working_memory:
            state.working_memory[key] = entry.content
```

**每步完成后** -- 持久化记忆更新:

```python
# src/arcana/runtime/hooks/memory_hook.py:40-62
async def on_step_complete(self, state, step_result, trace_ctx) -> None:
    for key, value in step_result.memory_updates.items():
        if value is None:
            await self.memory_manager.working.delete(state.run_id, key)
            continue

        content = value if isinstance(value, str) else json.dumps(value)
        request = MemoryWriteRequest(
            memory_type=MemoryType.WORKING,
            key=key,
            content=content,
            confidence=1.0,  # 步骤结果是可信的
            source="step_result",
            run_id=state.run_id,
            step_id=step_result.step_id,
        )
        await self.memory_manager.write(request)
```

**运行成功结束时** -- 提升记忆:

```python
# src/arcana/runtime/hooks/memory_hook.py:64-85
async def on_run_end(self, state, trace_ctx) -> None:
    if state.status.value != "completed":
        return  # 只有成功完成才提升

    entries = await self.memory_manager.working.get_all(state.run_id)
    for _key, entry in entries.items():
        if entry.metadata.get("promote_to_long_term"):
            lt_request = MemoryWriteRequest(
                memory_type=MemoryType.LONG_TERM,
                key=entry.key,
                content=entry.content,
                confidence=entry.confidence,
                source=f"promoted_from_working:{state.run_id}",
                tags=[*entry.tags, "promoted"],
            )
            await self.memory_manager.write(lt_request)
```

**记忆生命周期**:
```
工作记忆 (Working Memory)
    │
    │ on_step_complete: 持久化每步更新
    │
    v
持久化工作记忆 (Persisted Working)
    │
    │ on_run_end: 如果标记了 "promote_to_long_term"
    │
    v
长期记忆 (Long-Term Memory)
```

---

## 12. 验证器 (Verifier)

### 基础接口

```python
# src/arcana/runtime/verifiers/base.py:13-37
class BaseVerifier(ABC):
    """Checks whether an agent's goal or plan criteria have been satisfied."""

    @abstractmethod
    async def verify(self, state, plan=None) -> GoalVerificationResult: ...
```

### GoalVerifier -- 目标验证

GoalVerifier (`verifiers/goal_verifier.py`) 通过匹配已完成的步骤与验收标准来判断目标是否达成:

```python
# src/arcana/runtime/verifiers/goal_verifier.py:27-100
async def verify(self, state, plan=None) -> GoalVerificationResult:
    criteria_results = {}
    failed_criteria = []

    if plan is not None:
        # 1. 检查计划步骤完成状态
        for step in plan.steps:
            is_met = step.status == PlanStepStatus.COMPLETED
            criteria_results[step.description] = is_met
            if not is_met:
                failed_criteria.append(step.description)

        # 2. 检查全局验收标准
        for criterion in plan.acceptance_criteria:
            is_met = self._criterion_matched(criterion, state.completed_steps)
            criteria_results[criterion] = is_met
    else:
        # 没有计划时,简单检查是否有已完成步骤
        if state.goal:
            criteria_results[state.goal] = len(state.completed_steps) > 0

    # 3. 计算覆盖率
    total = len(criteria_results)
    passed = sum(1 for v in criteria_results.values() if v)
    coverage = passed / total if total > 0 else 0.0

    # 4. 判定结果: PASSED / PARTIAL / FAILED
    if passed == total:
        outcome = VerificationOutcome.PASSED
    elif passed > 0:
        outcome = VerificationOutcome.PARTIAL
    else:
        outcome = VerificationOutcome.FAILED
```

**标准匹配算法** -- 基于内容词重叠:

```python
# src/arcana/runtime/verifiers/goal_verifier.py:102-140
def _criterion_matched(self, criterion, completed_steps) -> bool:
    criterion_words = self._extract_content_words(criterion)  # 长度 > 3 的词

    for step in completed_steps:
        step_words = self._extract_content_words(step)
        overlap = criterion_words & step_words
        min_overlap = min(2, len(criterion_words))
        if len(overlap) >= min_overlap:
            return True
    return False

def _extract_content_words(self, text) -> set[str]:
    return {w.lower() for w in text.split() if len(w) > 3}
```

这是一种轻量级的文本匹配方案。它不需要 LLM 调用,只检查关键词重叠。阈值设为 "至少 2 个内容词重叠",在准确率和召回率之间取得平衡。

---

## 13. 数据流图

### 完整请求流程

```
用户请求: "帮我写一篇关于 Python 的博客"
                │
                v
┌───────────────────────────────────────────────────────────────────┐
│  create_agent() / create_react_agent()                            │
│  创建 Agent 实例,注入 Policy + Reducer + Gateway                   │
└──────────────────────────┬────────────────────────────────────────┘
                           │
                           v
┌───────────────────────────────────────────────────────────────────┐
│  Agent.run(goal="写一篇关于 Python 的博客")                        │
│                                                                   │
│  1. 创建 AgentState(run_id=uuid, goal=goal, status=PENDING)      │
│  2. state_manager.transition(PENDING -> RUNNING)                  │
│  3. hooks.on_run_start()                                          │
│                                                                   │
│  ┌─── 主循环 ─────────────────────────────────────────────────┐   │
│  │                                                            │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │ Step 1: _check_stop_conditions() -> None (继续)      │  │   │
│  │  │                                                      │  │   │
│  │  │ Policy.decide(state) -> PolicyDecision               │  │   │
│  │  │   action_type="llm_call"                             │  │   │
│  │  │   messages=[{system: ReAct prompt}, {user: "Next?"}] │  │   │
│  │  │                                                      │  │   │
│  │  │ StepExecutor.execute(decision)                       │  │   │
│  │  │   -> gateway.generate(LLMRequest)                    │  │   │
│  │  │   <- LLMResponse("Thought: 需要先列大纲...")          │  │   │
│  │  │   -> _parse_llm_response() -> (thought, action)      │  │   │
│  │  │   -> StepResult(thought=..., action=...)             │  │   │
│  │  │                                                      │  │   │
│  │  │ Reducer.reduce(state, step_result) -> updated_state  │  │   │
│  │  │   - completed_steps.append("Thought: 需要...")       │  │   │
│  │  │   - state.tokens_used += response.usage.total_tokens │  │   │
│  │  │                                                      │  │   │
│  │  │ ProgressDetector.record_step(step_result)            │  │   │
│  │  │   is_making_progress() -> True                       │  │   │
│  │  │                                                      │  │   │
│  │  │ _should_checkpoint() -> "interval" (每 N 步)         │  │   │
│  │  │ state_manager.checkpoint(state)                      │  │   │
│  │  │                                                      │  │   │
│  │  │ hooks.on_step_complete()                             │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                                                            │   │
│  │  ... (重复 Step 2, 3, ... N) ...                           │   │
│  │                                                            │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │ Step N: LLM 返回 "Action: FINISH"                    │  │   │
│  │  │ step_result.state_updates["goal_reached"] = True     │  │   │
│  │  │ -> _handle_stop(GOAL_REACHED)                        │  │   │
│  │  │ -> state_manager.transition(RUNNING -> COMPLETED)    │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
│  4. hooks.on_run_end()                                            │
│  5. return state (status=COMPLETED)                               │
└───────────────────────────────────────────────────────────────────┘
```

### 组件交互时序图

```
Agent          Policy         StepExecutor      Gateway         Reducer       StateManager
  │               │                │                │               │               │
  │ decide(state) │                │                │               │               │
  │──────────────>│                │                │               │               │
  │  PolicyDecision               │                │               │               │
  │<──────────────│                │                │               │               │
  │               │                │                │               │               │
  │ execute(decision)              │                │               │               │
  │───────────────────────────────>│                │               │               │
  │               │                │  generate()    │               │               │
  │               │                │───────────────>│               │               │
  │               │                │  LLMResponse   │               │               │
  │               │                │<───────────────│               │               │
  │               │  StepResult    │                │               │               │
  │<───────────────────────────────│                │               │               │
  │               │                │                │               │               │
  │ reduce(state, step_result)     │                │               │               │
  │────────────────────────────────────────────────────────────────>│               │
  │               │  new_state     │                │               │               │
  │<────────────────────────────────────────────────────────────────│               │
  │               │                │                │               │               │
  │ checkpoint(state)              │                │               │               │
  │─────────────────────────────────────────────────────────────────────────────────>│
  │               │                │                │               │  snapshot      │
  │<─────────────────────────────────────────────────────────────────────────────────│
  │               │                │                │               │               │
```

---

## 14. 生产注意事项

### 1. 预算管理

在生产环境中,**必须** 配置 BudgetTracker:

```python
from arcana.contracts.llm import Budget
from arcana.runtime import create_agent

agent = create_agent(
    gateway=gateway,
    budget=Budget(
        max_tokens=100_000,
        max_cost_usd=5.0,
        max_time_ms=300_000,  # 5 分钟
    ),
)
```

检查点预算阈值 (`RuntimeConfig.checkpoint_budget_thresholds`) 默认为 `[0.5, 0.75, 0.9]`,即在 50%、75%、90% 预算消耗时自动保存检查点。

### 2. 检查点存储

默认使用本地文件系统 (`./checkpoints/`):
- 生产环境应改为对象存储 (S3/GCS) 或数据库
- JSONL 格式支持追加写入,但不适合并发高的场景
- 建议为检查点文件设置 TTL 自动清理策略

### 3. 错误升级

为需要人工介入的场景配置升级回调:

```python
from arcana.runtime.error_handler import ErrorHandler, RetryStrategy

def escalate(error, state):
    # 发送告警到 PagerDuty、Slack 等
    notify_team(f"Agent {state.run_id} needs human intervention: {error}")

handler = ErrorHandler(
    retry_strategy=RetryStrategy(max_attempts=5, max_delay_ms=30000),
    escalation_callback=escalate,
)
```

### 4. Hook 性能

Hook 是同步调用的 (虽然是 async,但是顺序执行)。如果 hook 处理耗时较长,会拖慢整个执行循环。建议:
- Hook 内部使用异步队列,避免阻塞
- 设置 hook 超时 (当前实现未强制)
- 监控 hook 执行时间

### 5. 进度检测调参

默认参数 (`window_size=5`, `similarity_threshold=0.95`) 适用于大多数场景:
- 如果任务涉及大量重复操作 (如批量数据处理),应适当放宽阈值
- 如果任务需要快速检测循环,可以减小窗口大小

### 6. 状态大小控制

`completed_steps` 和 `messages` 列表会随运行时间线性增长:
- 考虑设置最大历史长度
- ReActPolicy 已经只使用最近 5 步历史 (`react.py:76`)
- 检查点中的完整状态可能很大,注意序列化/反序列化性能

### 7. 幂等性

工具调用通过 `idempotency_key` 支持幂等性。在恢复执行时,已执行的工具调用不会重复执行。确保你的工具实现也支持幂等性。

---

## 15. 本章小结

### 核心架构

Runtime 引擎的设计遵循了几个关键原则:

1. **状态机驱动**: Agent 的生命周期是一个严格的有限状态机,防止非法状态跳转
2. **策略模式**: Policy 将 "做什么" 的决策与 "怎么做" 的执行分离,支持多种推理范式
3. **Redux 式归约**: Reducer 以纯函数方式更新状态,使状态变更可预测、可追踪
4. **检查点恢复**: 自动的检查点机制支持长时间运行的任务在中断后恢复
5. **自我监控**: ProgressDetector 三重检测机制防止 Agent 陷入无意义循环
6. **Hook 扩展**: Protocol-based 的 hook 系统在不修改核心代码的前提下支持任意扩展

### 关键文件索引

| 文件 | 职责 | 核心类 |
|------|------|--------|
| `runtime/agent.py` | 执行编排 | `Agent` |
| `runtime/step.py` | 步骤执行 | `StepExecutor` |
| `runtime/state_manager.py` | 状态转换 + 检查点 | `StateManager` |
| `runtime/progress.py` | 进度监控 | `ProgressDetector` |
| `runtime/validator.py` | 输出验证 | `OutputValidator` |
| `runtime/replay.py` | 确定性重放 | `ReplayEngine` |
| `runtime/error_handler.py` | 错误分类 + 重试 | `ErrorHandler` |
| `runtime/exceptions.py` | 异常定义 | `RuntimeError` 等 |
| `runtime/factory.py` | 便捷创建 | `create_agent()` |
| `runtime/policies/base.py` | 策略接口 | `BasePolicy` |
| `runtime/policies/react.py` | ReAct 策略 | `ReActPolicy` |
| `runtime/policies/plan_execute.py` | 计划执行策略 | `PlanExecutePolicy` |
| `runtime/reducers/base.py` | 归约器接口 | `BaseReducer` |
| `runtime/reducers/default.py` | 默认归约器 | `DefaultReducer` |
| `runtime/reducers/plan_reducer.py` | 计划归约器 | `PlanReducer` |
| `runtime/hooks/base.py` | Hook 协议 | `RuntimeHook` |
| `runtime/hooks/memory_hook.py` | 记忆集成 | `MemoryHook` |
| `runtime/verifiers/base.py` | 验证器接口 | `BaseVerifier` |
| `runtime/verifiers/goal_verifier.py` | 目标验证 | `GoalVerifier` |

### 面试高频考点

- **为什么用状态机而不是简单的 while 循环?** 状态机提供了形式化的状态转换验证,防止非法跳转;终态概念确保 Agent 不会在完成后被意外重启;PAUSED 状态支持优雅的暂停恢复。
- **Policy 和 StepExecutor 为什么分开?** 单一职责原则。Policy 负责决策 (what),StepExecutor 负责执行 (how)。这让你可以在不改变执行逻辑的前提下切换推理范式,也可以在不改变策略的前提下优化执行效率。
- **Reducer 为什么叫 Reducer?** 借鉴 Redux 的 `(state, action) => newState` 模式。好处是状态变更可预测、可测试、可重放。
- **检查点哈希为什么排除时间字段?** 因为时间字段在恢复时必然不同,包含它们会导致哈希校验总是失败。
- **ProgressDetector 的三重检测为什么必要?** 单一检测手段有盲区: 重复检测抓不到变体循环,循环检测抓不到非精确重复,相似度检测抓不到不同内容但相同效果的步骤。三者互补才能全面覆盖。

---

> **下一章**: [Part 4: Graph 引擎](./tutorial_part4.md) -- 多节点 DAG 编排、条件分支和并行执行
