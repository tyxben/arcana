# Graph Engine TODO — 学习路线

> 按优先级排列。每个 TODO 都是一个独立的学习单元，完成后你会深入理解 Agent 框架的某个核心概念。

---

## P0: 并发安全与可靠性（不补就不能上线）

### TODO-1: Executor 状态隔离
**文件**: `graph/executor.py`
**问题**: `GraphExecutor._execution_state` 是实例变量，多次并发 `ainvoke()` 共享状态会互相污染。`prebuilt/react_agent.py` 的 `iteration_count` 用闭包更严重——编译一次，调用多次，计数器不会重置。
**学习点**: 理解 stateless vs stateful 设计、execution context 隔离模式
**做法**:
- 每次 `execute()` 创建独立的 `ExecutionContext` dataclass
- 把 `_execution_state`、`_event_queue` 等移入 context
- prebuilt 里的计数器改为从 state 读取而非闭包

### TODO-2: 节点超时控制
**文件**: `graph/node_runner.py`
**问题**: 节点函数没有超时，一个 LLM 调用卡死 30 秒整个图就挂了
**学习点**: `asyncio.wait_for`、deadline propagation、graceful cancellation
**做法**:
```python
# GraphNodeSpec 添加 timeout_ms 字段
# NodeRunner.run() 包裹 asyncio.wait_for
result = await asyncio.wait_for(fn(state), timeout=timeout_s)
```

### TODO-3: 节点重试机制
**文件**: `graph/node_runner.py`, `contracts/graph.py`
**问题**: 节点失败直接上抛，没有重试
**学习点**: 指数退避、重试策略、幂等性、可恢复 vs 不可恢复错误
**做法**:
- `GraphNodeSpec` 添加 `retry_policy: RetryPolicy | None`
- `RetryPolicy(max_retries, backoff_base, retryable_exceptions)`
- NodeRunner 实现 retry loop

### TODO-4: 结构化日志
**文件**: 新建 `graph/logging.py`
**问题**: 当前只有 trace event，没有运行时日志
**学习点**: structlog vs logging、日志级别策略、correlation ID
**做法**: 每个关键路径加结构化日志，包含 run_id、node、duration

---

## P1: 核心能力扩展

### TODO-5: 并行节点执行（fan-out / fan-in）
**文件**: `graph/executor.py`, `graph/state_graph.py`
**问题**: 当前是纯串行，一个节点执行完才能执行下一个
**学习点**: DAG 调度、`asyncio.gather`、并行状态合并冲突
**做法**:
- 允许一个节点有多条出边指向不同节点（fan-out）
- 多个节点的出边指向同一个节点（fan-in，需要 barrier）
- fan-in 节点等所有上游完成后，用 reducer 合并状态
- 这是 LangGraph 最强大的能力之一，理解这个就理解了图引擎的核心

### TODO-6: Checkpointer 后端抽象
**文件**: `graph/checkpointer.py`
**问题**: 只有本地 JSON 文件，不适合生产
**学习点**: 存储抽象、依赖倒置、适配器模式
**做法**:
```python
class BaseCheckpointer(ABC):
    async def save(self, ...) -> str: ...
    async def load(self, ...) -> dict | None: ...

class FileCheckpointer(BaseCheckpointer): ...  # 当前实现
class RedisCheckpointer(BaseCheckpointer): ...  # 新增
class PostgresCheckpointer(BaseCheckpointer): ...  # 新增
```

### TODO-7: 流式输出优化
**文件**: `graph/streaming.py`
**问题**: 用 `asyncio.wait_for(timeout=0.1)` 轮询队列，浪费 CPU
**学习点**: 生产者-消费者模式、sentinel value、async iterator protocol
**做法**:
- executor 完成后放一个 sentinel（如 `None`）到队列
- astream 用 `await queue.get()` 阻塞等待，不需要 timeout 轮询
- 实现 `__aiter__` / `__anext__` 让 CompiledGraph 自身可迭代

### TODO-8: 图的序列化与反序列化
**文件**: 新建 `graph/serialization.py`
**问题**: 当前图定义只存在内存中，无法持久化、版本化、跨进程传递
**学习点**: 图的 JSON 表示、函数注册表、声明式 vs 命令式
**做法**:
- `StateGraph.to_dict()` / `StateGraph.from_dict()`
- node function 通过 name 注册到 registry，序列化时只保存 name

---

## P2: 深入理解框架设计

### TODO-9: 中间件/钩子系统
**文件**: 新建 `graph/middleware.py`
**问题**: 无法在节点执行前后注入通用逻辑（日志、鉴权、缓存）
**学习点**: 中间件模式、装饰器链、AOP（面向切面编程）
**做法**:
```python
class GraphMiddleware(ABC):
    async def before_node(self, node_name, state): ...
    async def after_node(self, node_name, state, output): ...

# 用法
graph.compile(middleware=[LoggingMiddleware(), CacheMiddleware()])
```

### TODO-10: 动态图（运行时修改）
**文件**: `graph/executor.py`
**问题**: 图编译后是静态的，无法运行时增删节点
**学习点**: 静态图 vs 动态图的 trade-off、LangGraph 的 Send API
**做法**: 实现 `Command(send=[Send("node", data)])` 支持运行时动态分发

### TODO-11: 可视化
**文件**: 新建 `graph/visualization.py`
**问题**: 无法直观看到图的结构和执行路径
**学习点**: Mermaid diagram 生成、DOT 格式、ASCII art
**做法**:
```python
app = graph.compile()
print(app.get_graph().draw_mermaid())
# 输出 Mermaid 语法，粘贴到 GitHub/Notion 即可渲染
```

### TODO-12: 与现有 Agent 运行时集成
**文件**: `graph/nodes/agent_node.py`
**问题**: 图引擎和 Policy-Step-Reducer 运行时是完全独立的
**学习点**: 组合模式、如何让两套系统互操作
**做法**: 创建 `AgentNode`，将现有 `Agent.run()` 包装为图节点，让图可以调度传统 Agent

---

## 学习建议

1. **从 TODO-1 开始** — 并发安全是最基础的工程素养
2. **TODO-5 是核心** — 理解并行节点 = 理解图引擎为什么比简单 chain 强
3. **每完成一个 TODO，写对应的测试** — 测试驱动是理解框架边界的最好方式
4. **对比 LangGraph 源码** — 看 `langgraph/pregel/` 目录，理解它如何解决同样的问题
5. **看 trace 输出** — 每次执行后用 `TraceReader` 回放，观察事件流

### 推荐阅读顺序（理解框架全貌）

```
1. contracts/graph.py     → 数据模型是一切的基础
2. graph/state_graph.py   → 构建器模式 + 编译时验证
3. graph/executor.py      → 核心执行循环（最重要的文件）
4. graph/reducers.py      → 状态管理的精髓
5. graph/streaming.py     → 异步流式的实现模式
6. graph/checkpointer.py  → 中断/恢复 = 有状态工作流
7. prebuilt/react_agent.py → 看实际 Agent 模式如何用图表达
```
