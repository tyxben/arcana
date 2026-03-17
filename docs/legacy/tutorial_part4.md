# 第四部分：Graph Engine -- 声明式图编排引擎

> **源码路径**: `src/arcana/graph/`
> **核心依赖**: `contracts/graph.py` (数据模型), `gateway/` (LLM 调用), `tool_gateway/` (工具执行)

---

## 目录

1. [Graph Engine 概述](#1-graph-engine-概述)
2. [StateGraph 构建器](#2-stategraph-构建器)
3. [CompiledGraph 编译产物](#3-compiledgraph-编译产物)
4. [GraphExecutor 执行引擎](#4-graphexecutor-执行引擎)
5. [NodeRunner 节点执行器](#5-noderunner-节点执行器)
6. [流式执行 (Streaming)](#6-流式执行-streaming)
7. [Checkpointer 状态持久化](#7-checkpointer-状态持久化)
8. [中断与恢复 (Interrupt)](#8-中断与恢复-interrupt)
9. [Reducers 状态合并策略](#9-reducers-状态合并策略)
10. [预置图 (Prebuilt)](#10-预置图-prebuilt)
11. [节点类型](#11-节点类型)
12. [与 LangGraph 的对比](#12-与-langgraph-的对比)
13. [生产注意事项](#13-生产注意事项)
14. [本章小结](#14-本章小结)

---

## 1. Graph Engine 概述

### 1.1 为什么需要图引擎？

在第三部分中，我们介绍了 Arcana 的 Runtime 引擎 -- 一个基于状态机的线性执行模型。Runtime 擅长处理 `think -> act -> observe` 这类单一循环模式。但现实世界中的 Agent 工作流往往更加复杂：

- **多分支决策**：根据 LLM 输出决定走不同的处理路径
- **循环模式**：Agent 需要反复调用工具直到任务完成
- **子图嵌套**：一个大流程中嵌入独立的子流程
- **人机协作**：在关键节点暂停等待人类审批

Graph Engine 通过**声明式图编排**解决这些问题。你不需要手写控制流逻辑，只需声明"节点"和"边"，引擎会自动处理执行、路由、状态管理和中断恢复。

### 1.2 Runtime vs Graph：何时用哪个？

```
┌──────────────────┬─────────────────────┬──────────────────────┐
│     维度         │    Runtime 引擎      │     Graph 引擎       │
├──────────────────┼─────────────────────┼──────────────────────┤
│ 执行模式         │ 线性状态机           │ 有向图               │
│ 分支控制         │ Policy 决策          │ 条件边 (path_fn)     │
│ 适用场景         │ 简单 ReAct 循环      │ 复杂多步工作流       │
│ 子流程           │ 不支持               │ SubgraphNode         │
│ 人机协作         │ 不内置               │ interrupt_before/after│
│ 状态管理         │ StateManager         │ Checkpointer         │
│ 流式输出         │ 通过回调             │ astream + 事件队列   │
│ 学习曲线         │ 较低                 │ 中等                 │
└──────────────────┴─────────────────────┴──────────────────────┘
```

**经验法则**：如果你的 Agent 只需要"循环调用 LLM + 工具"，用 Runtime；如果需要多步编排、条件分支、人工审批，用 Graph。

### 1.3 整体架构

```
                         用户代码
                           │
                    ┌──────▼──────┐
                    │  StateGraph  │  ← 构建阶段：声明节点和边
                    │  (Builder)   │
                    └──────┬──────┘
                           │ .compile()
                    ┌──────▼──────┐
                    │ CompiledGraph│  ← 编译阶段：验证 + 冻结
                    │  (Frozen)    │
                    └──────┬──────┘
                           │ .ainvoke() / .astream()
                    ┌──────▼──────┐
                    │ GraphExecutor│  ← 执行阶段：主循环
                    │  (Engine)    │
                    └──┬───┬───┬──┘
                       │   │   │
              ┌────────┘   │   └────────┐
              ▼            ▼            ▼
         NodeRunner   Checkpointer  Streaming
         (节点执行)   (状态持久化)  (流式输出)
```

---

## 2. StateGraph 构建器

> **文件**: `src/arcana/graph/state_graph.py`

StateGraph 是图的**构建器** (Builder)。它遵循 Builder 模式，让你通过链式调用声明图的结构，最终通过 `compile()` 生成不可变的执行图。

### 2.1 核心数据结构

```python
# state_graph.py:43-51
class StateGraph:
    def __init__(self, state_schema: type[BaseModel] | None = None) -> None:
        self._state_schema = state_schema
        self._nodes: dict[str, GraphNodeSpec] = {}       # 节点注册表
        self._node_fns: dict[str, Callable[..., Any]] = {} # 节点函数
        self._edges: list[GraphEdgeSpec] = []             # 直接边
        self._conditional_edges: list[ConditionalEdgeSpec] = [] # 条件边
        self._conditional_fns: dict[str, Callable[..., str]] = {} # 条件函数
        self._entry_point: str | None = None              # 入口点
        self._finish_points: list[str] = []               # 终点列表
```

**设计决策**：`state_schema` 是可选的。如果提供了 Pydantic Model，引擎会：
1. 用它的默认值初始化状态
2. 从 `Annotated` 类型中提取 Reducer（详见第 9 节）

### 2.2 添加节点

```python
# state_graph.py:53-74
def add_node(
    self,
    name: str,
    fn: Callable[..., Any],
    *,
    node_type: NodeType = NodeType.FUNCTION,
    metadata: dict[str, Any] | None = None,
) -> StateGraph:
```

每个节点就是一个**函数**（同步或异步），签名为 `(state: dict) -> dict | None`。节点接收完整的图状态，返回需要更新的字段。

```python
# 示例：添加一个搜索节点
async def search(state: dict[str, Any]) -> dict[str, Any]:
    query = state.get("query", "")
    results = await do_search(query)
    return {"results": results}  # 只返回需要更新的字段

graph = StateGraph()
graph.add_node("search", search)
```

**保护机制**（`state_graph.py:62-65`）：
- 不允许使用 `START` / `END` 作为节点名
- 不允许重复注册同名节点

### 2.3 添加边

Arcana 支持两种边：

#### 直接边 (Direct Edge)

```python
# state_graph.py:76-81
def add_edge(self, source: str, target: str) -> StateGraph:
    self._edges.append(
        GraphEdgeSpec(source=source, target=target, edge_type=EdgeType.DIRECT)
    )
    return self
```

直接边表示无条件跳转。`START` 和 `END` 是两个特殊常量：

```python
# constants.py:3-4
START = "__start__"
END = "__end__"
```

#### 条件边 (Conditional Edge)

```python
# state_graph.py:83-102
def add_conditional_edges(
    self,
    source: str,
    path_fn: Callable[..., str],
    path_map: dict[str, str] | None = None,
) -> StateGraph:
```

条件边是图引擎的核心能力。`path_fn` 接收当前状态，返回一个字符串键。如果提供了 `path_map`，该键会被映射到目标节点名；否则键直接作为目标节点名。

```python
# 示例：根据 LLM 是否返回工具调用来路由
def should_continue(state: dict) -> str:
    last_msg = state["messages"][-1]
    if last_msg.get("tool_calls"):
        return "tools"       # 走工具分支
    return END               # 结束

graph.add_conditional_edges(
    "agent",                           # 从 agent 节点出发
    should_continue,                   # 路由函数
    {"tools": "tools", END: END}       # 键 -> 目标节点映射
)
```

### 2.4 入口和出口的便捷方法

```python
# state_graph.py:104-112
def set_entry_point(self, name: str) -> StateGraph:
    """等价于 add_edge(START, name)"""
    self._entry_point = name
    return self

def set_finish_point(self, name: str) -> StateGraph:
    """等价于 add_edge(name, END)"""
    self._finish_points.append(name)
    return self
```

### 2.5 编译与验证

`compile()` 方法是构建阶段的终点（`state_graph.py:114-160`）。它做三件事：

1. **解析入口点**（`_resolve_entry_point`，`state_graph.py:162-178`）：
   - 优先使用 `set_entry_point()` 设置的值
   - 否则从 `START` 边推导
   - 如果有多条 `START` 边或没有入口，报错

2. **验证图结构**（`_validate`，`state_graph.py:180-213`）：
   - 入口点必须是已注册的节点
   - 所有边的 source/target 必须是已注册节点（或 START/END）
   - 条件边的 `path_map` 目标必须合法

3. **可达性检查**（`_can_reach_end`，`state_graph.py:222-257`）：
   - 用 BFS 从入口点遍历，确认至少有一条路径到达 `END`
   - 这防止了图中出现"死循环无出口"的错误

```python
# 完整示例
graph = StateGraph()
graph.add_node("search", search)
graph.add_node("summarize", summarize)
graph.add_edge(START, "search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", END)

app = graph.compile(name="search-summarize")  # 验证通过，返回 CompiledGraph
```

---

## 3. CompiledGraph 编译产物

> **文件**: `src/arcana/graph/compiled_graph.py`

CompiledGraph 是**编译后的不可变图**。它是用户与图引擎交互的主要接口，提供三个核心方法。

### 3.1 三大执行方法

```
┌─────────────────────────────────────────────────────────┐
│                    CompiledGraph                        │
├─────────────────────────────────────────────────────────┤
│  ainvoke(input, config)  → dict    # 一次性执行到完成   │
│  astream(input, mode)    → AsyncGen # 流式执行           │
│  aresume(checkpoint_id)  → dict    # 从中断点恢复执行   │
└─────────────────────────────────────────────────────────┘
```

#### ainvoke -- 同步执行到完成

```python
# compiled_graph.py:58-76
async def ainvoke(
    self,
    input: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    executor = GraphExecutor(...)
    return await executor.execute(input, config)
```

最简单的使用方式。传入初始状态，获得最终状态：

```python
result = await app.ainvoke({"query": "Python", "messages": []})
print(result["summary"])  # 包含完整的处理结果
```

#### astream -- 流式执行

```python
# compiled_graph.py:78-101
async def astream(
    self,
    input: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    mode: str = "values",
) -> AsyncGenerator[dict[str, Any], None]:
```

支持三种流式模式（详见第 6 节）。每个节点完成后都会产出一个事件。

#### aresume -- 从中断恢复

```python
# compiled_graph.py:103-128
async def aresume(
    self,
    checkpoint_id: str,
    command: Command | None = None,
) -> dict[str, Any]:
```

从检查点恢复执行，支持通过 `Command` 注入状态更新、跳转到指定节点，或传递恢复值。

---

## 4. GraphExecutor 执行引擎

> **文件**: `src/arcana/graph/executor.py`

GraphExecutor 是整个图引擎的**心脏**。它实现了核心执行循环。

### 4.1 执行循环

```
┌─────────────────────────────────────────────────┐
│            GraphExecutor.execute()               │
│                                                 │
│  current = entry_point                          │
│  while current != END:                          │
│    ┌─────────────────────────────────┐          │
│    │ 1. interrupt_before 检查        │ ──→ 中断? │
│    │ 2. 执行节点函数 (NodeRunner)    │          │
│    │ 3. apply_reducers 合并输出到状态 │          │
│    │ 4. interrupt_after 检查         │ ──→ 中断? │
│    │ 5. 发送流式事件                 │          │
│    │ 6. _route() 路由到下一节点      │          │
│    └─────────────────────────────────┘          │
│  return final_state                             │
└─────────────────────────────────────────────────┘
```

对应代码（`executor.py:82-139`）：

```python
async def execute(self, initial_state, config=None):
    state = self._init_state(initial_state)
    current = self._config.entry_point

    while current != END:
        # 1. interrupt_before
        if current in self._config.interrupt_before:
            await self._handle_interrupt(current, state, trace_ctx, phase="before")

        # 2. 执行节点
        fn = self._node_fns.get(current)
        output = await self._node_runner.run(current, fn, state, trace_ctx)

        # 3. 合并状态
        state = apply_reducers(state, output, self._reducers)

        # 4. interrupt_after
        if current in self._config.interrupt_after:
            await self._handle_interrupt(current, state, trace_ctx, phase="after")

        # 5. 流式事件
        await self._emit_event(current, state, output)

        # 6. 路由
        current = self._route(current, state, output)

    return state
```

### 4.2 路由机制

`_route` 方法（`executor.py:216-254`）实现了路由逻辑，优先级是：

1. **条件边优先**：如果当前节点有条件边，调用 `path_fn(state)` 获取路由键
   - 有 `path_map`：将键映射到目标节点
   - 无 `path_map`：键直接作为目标节点名
2. **直接边兜底**：如果没有条件边，使用直接边
3. **无路由报错**：如果两者都没有，抛出 `GraphExecutionError`

```python
# executor.py:216-254
def _route(self, current, state, output):
    # 条件边优先
    for cond in self._conditional_edges:
        if cond.source == current:
            path_fn = self._conditional_fns.get(current)
            key = path_fn(state)
            if cond.path_map:
                target = cond.path_map.get(key)
                if target is None:
                    raise GraphExecutionError(f"Unmapped key '{key}'")
                return target
            else:
                return key  # 直接用键作为节点名

    # 直接边
    for edge in self._edges:
        if edge.source == current:
            return edge.target

    raise GraphExecutionError(f"No outgoing edge from '{current}'")
```

### 4.3 状态初始化

`_init_state` 方法（`executor.py:201-214`）处理状态初始化：

```python
def _init_state(self, initial_state):
    if self._state_schema:
        # 从 schema 提取默认值，然后叠加用户输入
        defaults = self._state_schema.model_fields
        state = {}
        for field_name, field_info in defaults.items():
            if field_info.default is not None:
                state[field_name] = field_info.default
            elif field_info.default_factory is not None:
                state[field_name] = field_info.default_factory()
        state.update(initial_state)
        return state
    return dict(initial_state)
```

**设计决策**：图状态始终是 `dict[str, Any]`，即使定义了 schema。这是为了保持灵活性 -- 节点可以返回 schema 中未定义的键，系统不会报错。Schema 仅用于提供默认值和 Reducer 提取。

### 4.4 恢复执行

`resume` 方法（`executor.py:141-199`）支持从检查点恢复：

```python
async def resume(self, checkpoint_data, command=None):
    state = dict(checkpoint_data.get("state", {}))
    resume_node = checkpoint_data.get("resume_node", "")
    phase = checkpoint_data.get("phase", "before")

    if command:
        if command.update:
            state.update(command.update)      # 注入状态更新
        if command.goto:
            resume_node = command.goto         # 跳转到指定节点
            phase = "before"
        if command.resume is not None:
            state["__resume_value__"] = command.resume  # 传递恢复值
```

恢复逻辑区分两种情况：
- `phase == "before"`：节点尚未执行，从该节点重新开始
- `phase == "after"`：节点已执行完毕，从其下游节点开始

---

## 5. NodeRunner 节点执行器

> **文件**: `src/arcana/graph/node_runner.py`

NodeRunner 负责执行单个节点函数，处理同步/异步兼容和追踪。

### 5.1 同步/异步兼容

```python
# node_runner.py:59-65
if asyncio.iscoroutinefunction(fn) or (
    callable(fn) and inspect.iscoroutinefunction(fn.__call__)
):
    result = await fn(state)
else:
    result = fn(state)
```

这段代码做了两层检查：
1. 函数本身是否是 `async def`
2. 如果是可调用对象（如类实例），检查其 `__call__` 方法是否是 `async`

这意味着你可以混用同步和异步节点：

```python
# 同步节点 -- 适合纯计算
def format_output(state: dict) -> dict:
    return {"display": f"Result: {state['result']}"}

# 异步节点 -- 适合 I/O 操作
async def call_api(state: dict) -> dict:
    result = await http_client.get(state["url"])
    return {"result": result}

# 类实例节点（如 LLMNode）-- __call__ 是 async
llm_node = LLMNode(gateway)

graph.add_node("format", format_output)
graph.add_node("api", call_api)
graph.add_node("llm", llm_node)
```

### 5.2 输出规范化

```python
# node_runner.py:82-88
if result is None:
    result = {}
elif not isinstance(result, dict):
    raise TypeError(
        f"Node '{node_name}' must return a dict or None, got {type(result).__name__}"
    )
```

节点函数必须返回 `dict` 或 `None`。返回 `None` 等价于返回空字典（不修改状态）。

### 5.3 追踪集成

NodeRunner 在节点执行前后自动写入 Trace 事件：

```
GRAPH_NODE_START  → 节点开始执行
GRAPH_NODE_COMPLETE → 节点完成（包含耗时 duration_ms 和输出键列表）
ERROR → 节点执行异常
```

这为调试和监控提供了细粒度的可观测性。

---

## 6. 流式执行 (Streaming)

> **文件**: `src/arcana/graph/streaming.py`

### 6.1 架构设计

流式执行基于 `asyncio.Queue` 实现生产者-消费者模式：

```
┌──────────────┐    事件队列     ┌──────────────┐
│ GraphExecutor│ ──────────────→ │   astream    │
│ (后台任务)   │   Queue<dict>  │ (AsyncGen)   │
│              │                │              │
│ _emit_event()│                │ 格式化+yield  │
└──────────────┘                └──────────────┘
```

```python
# streaming.py:33-48
queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
executor = GraphExecutor(...)
executor.event_queue = queue

# 在后台任务中执行图
task = asyncio.create_task(executor.execute(input, config))
```

### 6.2 三种流式模式

```python
# streaming.py:27-29
if mode not in ("values", "updates", "messages"):
    raise ValueError(...)
```

#### mode="values" -- 完整状态快照

每个节点完成后 yield 完整的当前状态。适合需要持续展示全局状态的场景：

```python
async for state in app.astream(input, mode="values"):
    print(f"当前状态键: {list(state.keys())}")
    # 输出: {'query': '...', 'results': [...], 'messages': [...]}
```

#### mode="updates" -- 增量更新

每个节点完成后 yield 节点名和输出。适合追踪执行进度：

```python
async for event in app.astream(input, mode="updates"):
    print(f"节点 '{event['node']}' 输出: {event['output'].keys()}")
    # 输出: 节点 'search' 输出: dict_keys(['results', 'messages'])
```

#### mode="messages" -- 消息增量

仅 yield 新增的消息。适合聊天式 Agent：

```python
async for event in app.astream(input, mode="messages"):
    for msg in event["messages"]:
        print(f"[{msg['role']}] {msg['content'][:50]}...")
```

### 6.3 事件格式化

`_format_event` 函数（`streaming.py:93-112`）将原始事件转换为对应模式的输出：

```python
def _format_event(event, mode, previous_messages):
    if event.get("type") != "node_complete":
        return None

    if mode == "values":
        return dict(event.get("state", {}))
    elif mode == "updates":
        return {"node": event["node"], "output": event.get("output", {})}
    elif mode == "messages":
        current_messages = event.get("state", {}).get("messages", [])
        new_messages = current_messages[len(previous_messages):]
        if new_messages:
            return {"node": event["node"], "messages": new_messages}
        return None
```

**消息去重机制**：`messages` 模式通过记录 `previous_messages` 的长度，用切片 `current[len(previous):]` 提取新增消息。简单高效。

### 6.4 优雅关闭

流式执行在 `finally` 块中处理清理工作（`streaming.py:79-90`）：

```python
finally:
    if not task.done():
        task.cancel()      # 如果消费端提前退出，取消后台任务
    else:
        exc = task.exception()
        if exc:
            raise exc       # 传播执行器的异常
```

---

## 7. Checkpointer 状态持久化

> **文件**: `src/arcana/graph/checkpointer.py`

### 7.1 设计思路

Checkpointer 是中断/恢复功能的存储层。它将图执行状态持久化到磁盘，使得执行可以在任意检查点暂停和恢复。

```python
# checkpointer.py:11-22
class GraphCheckpointer:
    """
    Persists graph execution state for interrupt/resume.
    Stores checkpoints as JSON files.
    """
    def __init__(self, checkpoint_dir: str | Path = "./checkpoints/graph") -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
```

### 7.2 三个核心操作

#### save -- 保存检查点

```python
# checkpointer.py:23-54
async def save(self, state, node_id, metadata=None) -> str:
    checkpoint_id = str(uuid4())
    checkpoint = {
        "checkpoint_id": checkpoint_id,
        "state": state,
        "node_id": node_id,
        "resume_node": node_id,
        **(metadata or {}),
    }
    checkpoint_file = self._checkpoint_dir / f"{checkpoint_id}.json"
    checkpoint_file.write_text(json.dumps(checkpoint, default=str))
    return checkpoint_id
```

每个检查点包含：
- `checkpoint_id`：UUID 唯一标识
- `state`：完整的图状态快照
- `node_id` / `resume_node`：中断发生的节点
- `metadata`：额外信息（如 `phase: "before"/"after"`）

#### load -- 加载检查点

```python
# checkpointer.py:56-68
async def load(self, checkpoint_id: str) -> dict[str, Any] | None:
    checkpoint_file = self._checkpoint_dir / f"{checkpoint_id}.json"
    if not checkpoint_file.exists():
        return None
    return json.loads(checkpoint_file.read_text())
```

#### delete -- 删除检查点

```python
# checkpointer.py:70-76
async def delete(self, checkpoint_id: str) -> bool:
    checkpoint_file = self._checkpoint_dir / f"{checkpoint_id}.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        return True
    return False
```

**设计决策**：Checkpointer 使用 JSON 文件而非数据库，保持简单。生产环境可以替换为 Redis/PostgreSQL 实现（只需实现相同的 `save`/`load`/`delete` 接口）。

---

## 8. 中断与恢复 (Interrupt)

> **文件**: `src/arcana/graph/interrupt.py`

中断机制是图引擎支持**人机协作** (Human-in-the-Loop) 的核心能力。

### 8.1 GraphInterrupt 异常

```python
# interrupt.py:18-32
class GraphInterrupt(Exception):
    def __init__(self, *, node_id, state, checkpoint_id, message=""):
        self.node_id = node_id          # 中断发生的节点
        self.state = state              # 中断时的状态快照
        self.checkpoint_id = checkpoint_id  # 检查点 ID（用于恢复）
        super().__init__(message or f"Graph interrupted at node '{node_id}'")
```

GraphInterrupt 是一个特殊异常。它不表示错误，而是一个**控制信号**："请暂停执行，等待人类介入"。

### 8.2 中断时机

在 `graph.compile()` 时通过 `interrupt_before` 和 `interrupt_after` 指定：

```python
app = graph.compile(
    checkpointer=GraphCheckpointer(),
    interrupt_before=["human_review"],   # 在 human_review 节点执行前中断
    interrupt_after=["sensitive_action"], # 在 sensitive_action 节点执行后中断
)
```

```
interrupt_before:                interrupt_after:

  ──→ [中断] ──→ 节点执行         ──→ 节点执行 ──→ [中断]
       ↑                                            ↑
    在节点执行之前暂停              在节点执行之后暂停
    人类可以修改输入                人类可以审查输出
```

### 8.3 Command -- 恢复指令

```python
# interrupt.py:10-15
class Command(BaseModel):
    resume: Any = None              # 传递给中断节点的值
    update: dict[str, Any] | None = None  # 状态更新
    goto: str | None = None         # 跳转到指定节点
```

Command 提供三种恢复策略：

1. **传递值** (`resume`)：将值注入状态的 `__resume_value__` 键，被中断节点可以读取
2. **更新状态** (`update`)：直接修改图状态
3. **跳转节点** (`goto`)：跳过当前节点，直接到指定节点

### 8.4 完整的中断/恢复流程

```python
# 1. 编译时声明中断点
app = graph.compile(
    checkpointer=GraphCheckpointer(),
    interrupt_before=["human_review"],
)

# 2. 执行，遇到中断会抛出 GraphInterrupt
try:
    result = await app.ainvoke({"task": "deploy to production"})
except GraphInterrupt as interrupt:
    print(f"需要人工审批: 节点={interrupt.node_id}")
    print(f"当前状态: {interrupt.state}")
    checkpoint_id = interrupt.checkpoint_id

    # 3. 人工审批后恢复
    human_decision = await get_human_approval()
    result = await app.aresume(
        checkpoint_id,
        Command(
            resume=human_decision,           # 传递审批结果
            update={"approved_by": "admin"}, # 更新状态
        ),
    )
```

### 8.5 中断处理内部实现

```python
# executor.py:256-291
async def _handle_interrupt(self, node_id, state, trace_ctx, phase):
    # 1. 保存检查点
    if self._checkpointer:
        checkpoint_id = await self._checkpointer.save(
            state=state,
            node_id=node_id,
            metadata={"phase": phase, "resume_node": node_id},
        )

    # 2. 记录追踪事件
    if self._trace_writer:
        self._trace_writer.write(TraceEvent(...))

    # 3. 更新执行状态
    self._execution_state.is_interrupted = True
    self._execution_state.interrupt_node = node_id

    # 4. 抛出中断异常
    raise GraphInterrupt(
        node_id=node_id,
        state=dict(state),
        checkpoint_id=checkpoint_id,
    )
```

---

## 9. Reducers 状态合并策略

> **文件**: `src/arcana/graph/reducers.py`

### 9.1 为什么需要 Reducer？

图中的每个节点返回一个 `dict`，表示"我要更新这些字段"。但状态更新并非总是简单的"替换"。考虑以下场景：

- `messages` 字段：应该**追加**新消息，而不是替换
- `counter` 字段：应该**累加**，而不是替换
- `metadata` 字段：应该**合并**字典，而不是替换

Reducer 就是解决"如何将节点输出合并到现有状态"的函数。

### 9.2 内置 Reducer

```python
# reducers.py:12-33
def replace_reducer(old, new):      # 直接替换（默认行为）
    return new

def append_reducer(old, new):       # 追加到列表
    old_list = old if isinstance(old, list) else ([] if old is None else [old])
    if isinstance(new, list):
        return old_list + new
    return old_list + [new]

def add_reducer(old, new):          # 数值累加
    return (old or 0) + new

def merge_reducer(old, new):        # 字典合并
    return {**(old or {}), **(new or {})}
```

### 9.3 add_messages -- 消息专用 Reducer

```python
# reducers.py:35-61
def add_messages(existing, new):
    """消息追加 + ID 去重"""
    existing = existing or []
    if not isinstance(new, list):
        new = [new]

    # 按 id 建索引，支持更新已有消息
    existing_by_id = {}
    for i, msg in enumerate(existing):
        msg_id = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
        if msg_id is not None:
            existing_by_id[msg_id] = i

    result = list(existing)
    for msg in new:
        msg_id = ...
        if msg_id is not None and msg_id in existing_by_id:
            result[existing_by_id[msg_id]] = msg  # 更新已有消息
        else:
            result.append(msg)                     # 追加新消息
    return result
```

`add_messages` 的去重逻辑很重要：如果新消息带有 `id` 且该 `id` 已存在，则**更新**而非追加。这与 LangGraph 的同名 Reducer 行为一致。

### 9.4 通过 Annotated 声明 Reducer

```python
# reducers.py:64-90
def extract_reducers(schema):
    """从 Pydantic 模型的 Annotated 类型提取 Reducer"""
    hints = get_type_hints(schema, include_extras=True)
    for field_name, hint in hints.items():
        if get_origin(hint) is typing.Annotated:
            args = get_args(hint)
            for arg in args[1:]:
                if callable(arg) and not isinstance(arg, type):
                    reducers[field_name] = arg
                    break
    return reducers
```

使用方式：

```python
from typing import Annotated
from pydantic import BaseModel
from arcana.graph.reducers import add_messages, add_reducer

class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = []    # 消息追加+去重
    counter: Annotated[int, add_reducer] = 0        # 数值累加
    result: str = ""                                 # 默认：替换

graph = StateGraph(state_schema=AgentState)
```

### 9.5 Reducer 应用

```python
# reducers.py:93-106
def apply_reducers(state, output, reducers):
    new_state = dict(state)
    for key, value in output.items():
        if key in reducers:
            new_state[key] = reducers[key](state.get(key), value)
        else:
            new_state[key] = value   # 无 Reducer 的字段直接替换
    return new_state
```

这个函数在执行循环的第 3 步被调用（`executor.py:108`），将每个节点的输出合并到全局状态中。

---

## 10. 预置图 (Prebuilt)

> **文件**: `src/arcana/graph/prebuilt/`

Arcana 提供两种开箱即用的 Agent 模式。

### 10.1 ReAct Agent

> **文件**: `src/arcana/graph/prebuilt/react_agent.py`

ReAct (Reasoning + Acting) 是最经典的 Agent 模式：LLM 思考 -> 调用工具 -> 观察结果 -> 继续思考。

```
┌─────────┐     有 tool_calls     ┌─────────┐
│         │ ────────────────────→ │         │
│  agent  │                       │  tools  │
│ (LLM)   │ ←──────────────────── │         │
│         │     工具结果回传       │         │
└────┬────┘                       └─────────┘
     │ 无 tool_calls
     ▼
   [END]
```

```python
# react_agent.py:20-82
def create_react_agent(gateway, tool_gateway, *, max_iterations=25, ...):
    llm_node = LLMNode(gateway, ...)
    tool_node = ToolNode(tool_gateway)

    def should_continue(state):
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count > max_iterations:
            return END
        last_msg = state["messages"][-1]
        if last_msg.get("tool_calls"):
            return "tools"
        return END

    graph = StateGraph()
    graph.add_node("agent", llm_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
```

**安全机制**：`max_iterations` 防止 Agent 陷入无限循环。默认 25 次，可根据任务复杂度调整。

使用示例（参见 `examples/demo_graph_react.py`）：

```python
agent = create_react_agent(
    gateway,
    tool_gateway,
    system_prompt="You are a helpful research assistant.",
    max_iterations=10,
)

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "What is the latest Python release?"}]
})
```

### 10.2 Plan-Execute Agent

> **文件**: `src/arcana/graph/prebuilt/plan_execute.py`

Plan-Execute 模式将"规划"和"执行"解耦，适合复杂多步任务：

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ planner  │ ──→ │ executor │ ──→ │ verifier │
│ (规划)   │     │ (执行)   │     │ (验证)   │
└──────────┘     └──────────┘     └────┬─────┘
     ↑                                  │
     │ 未完成 + replan_count < max      │ 已完成 或 达到上限
     └──────────────────────────────────┘
                                        │
                                        ▼
                                      [END]
```

```python
# plan_execute.py:18-162
def create_plan_execute_agent(gateway, tool_gateway, *, max_replans=3, ...):
    async def planner(state):
        # 生成或修改计划
        if state.get("verification_feedback"):
            # 有反馈，请求修改计划
            ...
        return {"plan": response.content, "messages": [...]}

    async def executor(state):
        # 按计划执行
        return {"messages": [...]}

    async def verifier(state):
        # 验证结果
        is_complete = "COMPLETE" in content.upper()
        return {"is_complete": is_complete, "verification_feedback": ...}

    def should_replan(state):
        if state.get("is_complete", False):
            return END
        replan_count += 1
        if replan_count > max_replans:
            return END
        return "planner"
```

**与 ReAct 的对比**：
- ReAct 是"边想边做"，每步只做一个决定
- Plan-Execute 是"先想后做再验证"，有全局规划能力
- Plan-Execute 的验证环节能捕获偏离，自动修正

---

## 11. 节点类型

> **文件**: `src/arcana/graph/nodes/`

### 11.1 LLMNode -- LLM 调用节点

> **文件**: `src/arcana/graph/nodes/llm_node.py`

```python
# llm_node.py:14-73
class LLMNode:
    def __init__(self, gateway, *, model_config=None, system_prompt=None):
        self._gateway = gateway
        self._model_config = model_config
        self._system_prompt = system_prompt

    async def __call__(self, state):
        messages = state.get("messages", [])
        request_messages = []

        # 添加 system prompt
        if self._system_prompt:
            request_messages.append(Message(role="system", content=self._system_prompt))

        # 转换 state messages
        for msg in messages:
            if isinstance(msg, Message):
                request_messages.append(msg)
            elif isinstance(msg, dict):
                request_messages.append(Message(**msg))

        response = await self._gateway.generate(LLMRequest(...))

        # 构建返回消息（包含 tool_calls）
        response_msg = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            response_msg["tool_calls"] = [tc.model_dump() for tc in response.tool_calls]

        return {"messages": [response_msg]}
```

**约定**：LLMNode 从 `state["messages"]` 读取消息历史，将 LLM 响应追加到 `messages` 中。这与 Reducer 配合：如果 `messages` 字段使用 `add_messages` Reducer，新消息会被追加而非替换。

### 11.2 ToolNode -- 工具执行节点

> **文件**: `src/arcana/graph/nodes/tool_node.py`

```python
# tool_node.py:13-71
class ToolNode:
    def __init__(self, tool_gateway):
        self._tool_gateway = tool_gateway

    async def __call__(self, state):
        messages = state.get("messages", [])

        # 从最后一条 assistant 消息中提取 tool_calls
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls_data = msg["tool_calls"]
                break

        # 转换为 ToolCall 对象
        tool_calls = [ToolCall(id=tc["id"], name=tc["function"]["name"], ...)]

        # 批量执行
        results = await self._tool_gateway.call_many(tool_calls)

        # 构建 tool 消息
        return {"messages": [
            {"role": "tool", "content": result.output, "tool_call_id": tc.id}
            for tc, result in zip(tool_calls, results)
        ]}
```

**关键设计**：ToolNode 从消息历史中**反向搜索**最近的 assistant 消息获取 tool_calls，而不是从单独的状态字段读取。这保持了与 OpenAI 消息格式的兼容性。

### 11.3 SubgraphNode -- 子图节点

> **文件**: `src/arcana/graph/nodes/subgraph_node.py`

```python
# subgraph_node.py:11-58
class SubgraphNode:
    def __init__(self, graph, *, input_map=None, output_map=None):
        self._graph = graph          # 嵌套的 CompiledGraph
        self._input_map = input_map  # 父状态键 -> 子图输入键
        self._output_map = output_map # 子图输出键 -> 父状态键

    async def __call__(self, state):
        # 映射输入
        if self._input_map:
            subgraph_input = {
                sub_key: state[parent_key]
                for parent_key, sub_key in self._input_map.items()
            }
        else:
            subgraph_input = dict(state)  # 传递完整状态

        # 执行子图
        result = await self._graph.ainvoke(subgraph_input)

        # 映射输出
        if self._output_map:
            return {parent_key: result[sub_key] ...}
        return result
```

SubgraphNode 允许将一个完整的 CompiledGraph 嵌入为另一个图的节点。`input_map` 和 `output_map` 处理父子图之间的状态映射：

```python
# 构建子图
sub_graph = StateGraph()
sub_graph.add_node("analyze", analyze_fn)
sub_graph.add_edge(START, "analyze")
sub_graph.add_edge("analyze", END)
compiled_sub = sub_graph.compile()

# 嵌入主图
main_graph = StateGraph()
main_graph.add_node("sub_task", SubgraphNode(
    compiled_sub,
    input_map={"raw_data": "input"},     # 父 raw_data -> 子 input
    output_map={"result": "analysis"},    # 子 result -> 父 analysis
))
```

---

## 12. 与 LangGraph 的对比

Arcana 的 Graph Engine 在 API 设计上深受 LangGraph 的启发，但有若干关键差异：

### 12.1 相似之处

| 特性 | LangGraph | Arcana Graph |
|------|-----------|-------------|
| Builder 模式 | `StateGraph` | `StateGraph` |
| 编译步骤 | `.compile()` | `.compile()` |
| 特殊常量 | `START`, `END` | `START`, `END` |
| 条件边 | `add_conditional_edges` | `add_conditional_edges` |
| 消息 Reducer | `add_messages` | `add_messages` |
| 中断/恢复 | `interrupt_before/after` | `interrupt_before/after` |
| 流式模式 | `values/updates/messages` | `values/updates/messages` |
| Checkpointer | 多种后端 | JSON 文件（可扩展） |

### 12.2 关键差异

```
┌────────────────┬─────────────────────────┬─────────────────────────┐
│    差异点       │      LangGraph          │      Arcana Graph       │
├────────────────┼─────────────────────────┼─────────────────────────┤
│ 状态类型       │ TypedDict / Pydantic    │ 纯 dict + 可选 schema   │
│ Reducer 声明   │ Annotated 在 TypedDict  │ Annotated 在 Pydantic   │
│ 并行节点       │ Send API 支持           │ 暂不支持                │
│ 子图           │ 内置 Subgraph 协议      │ SubgraphNode 封装       │
│ Checkpointer   │ SQLite/Postgres/Redis   │ JSON 文件               │
│ 追踪           │ LangSmith 集成          │ 内置 TraceWriter        │
│ 代码量         │ ~数千行                 │ ~600 行                 │
│ 依赖           │ langchain-core          │ 仅 Pydantic             │
│ 设计目标       │ 生产级功能完整          │ 教学级精简实现          │
└────────────────┴─────────────────────────┴─────────────────────────┘
```

### 12.3 设计哲学差异

1. **契约优先**：Arcana 所有数据模型在 `contracts/graph.py` 中预先定义（`GraphNodeSpec`, `GraphEdgeSpec`, `ConditionalEdgeSpec`, `GraphConfig`, `GraphExecutionState`），遵循"先定义契约，再写实现"的原则

2. **极简依赖**：不依赖 LangChain 生态，唯一的外部依赖是 Pydantic

3. **可观测性内置**：追踪不是可选插件，而是引擎的组成部分（`NodeRunner` 自动写入 Trace 事件）

---

## 13. 生产注意事项

### 13.1 Checkpointer 替换

当前的 JSON 文件 Checkpointer 适合开发和测试。生产环境需要：

```python
# 自定义 Checkpointer 只需实现三个方法
class RedisCheckpointer:
    async def save(self, state, node_id, metadata=None) -> str: ...
    async def load(self, checkpoint_id: str) -> dict | None: ...
    async def delete(self, checkpoint_id: str) -> bool: ...
```

### 13.2 max_iterations 安全限制

ReAct Agent 的 `max_iterations` 参数至关重要。在生产中：
- 为每种任务类型设置合理的上限
- 监控实际迭代次数，调整阈值
- 结合 `BudgetTracker` 的 token/cost 限制

### 13.3 Reducer 选择

- 对消息历史字段，始终使用 `add_messages`（支持去重）
- 对计数器字段，使用 `add_reducer`
- 对累积结果字段，使用 `append_reducer`
- 未声明 Reducer 的字段默认是 `replace`（后写覆盖前写）

### 13.4 错误处理

节点函数抛出的异常会直接传播到调用方。建议：

```python
async def safe_node(state):
    try:
        result = await risky_operation()
        return {"result": result, "error": None}
    except Exception as e:
        return {"error": str(e)}  # 将错误写入状态，让下游节点决策
```

### 13.5 图验证的局限

`_can_reach_end` 使用 BFS 验证可达性，但无法检测所有运行时问题：
- 条件边的 `path_fn` 可能永远不返回通向 END 的键
- 状态依赖的路由可能在特定输入下陷入循环
- 建议为复杂图编写集成测试，覆盖各种执行路径

### 13.6 流式执行的超时

流式模式使用 `asyncio.wait_for(queue.get(), timeout=0.1)` 轮询事件队列（`streaming.py:55`）。100ms 的超时在大多数场景下是合适的，但对极低延迟场景可能需要调整。

---

## 14. 本章小结

### 核心文件清单

| 文件 | 职责 | 行数 |
|------|------|------|
| `contracts/graph.py` | 数据模型（节点、边、配置、执行状态） | 69 |
| `graph/constants.py` | START / END 常量 | 4 |
| `graph/state_graph.py` | Builder 模式构建器 | 257 |
| `graph/compiled_graph.py` | 编译后的不可变图 | 129 |
| `graph/executor.py` | 核心执行循环 | 307 |
| `graph/node_runner.py` | 节点执行 + 追踪 | 107 |
| `graph/streaming.py` | 流式输出（三种模式） | 113 |
| `graph/checkpointer.py` | 状态持久化（JSON） | 77 |
| `graph/interrupt.py` | 中断/恢复 + Command | 33 |
| `graph/reducers.py` | 状态合并策略 | 107 |
| `graph/nodes/llm_node.py` | LLM 调用节点 | 73 |
| `graph/nodes/tool_node.py` | 工具执行节点 | 71 |
| `graph/nodes/subgraph_node.py` | 子图嵌套节点 | 58 |
| `graph/prebuilt/react_agent.py` | ReAct Agent 模板 | 82 |
| `graph/prebuilt/plan_execute.py` | Plan-Execute 模板 | 162 |

### 知识点回顾

1. **Builder 模式**：StateGraph 通过 `add_node` / `add_edge` / `add_conditional_edges` 声明图结构，`compile()` 验证并冻结
2. **执行循环**：interrupt_before -> 执行节点 -> apply_reducers -> interrupt_after -> 流式事件 -> 路由
3. **路由优先级**：条件边 > 直接边 > 报错
4. **Reducer 机制**：通过 `Annotated[type, reducer_fn]` 声明字段级合并策略
5. **中断/恢复**：GraphInterrupt 异常 + Checkpointer 持久化 + Command 恢复指令
6. **流式模式**：values（全量快照）/ updates（增量更新）/ messages（消息增量）
7. **预置模式**：ReAct（思考-行动循环）和 Plan-Execute（规划-执行-验证循环）
8. **子图嵌套**：SubgraphNode 通过 input_map / output_map 实现父子图状态映射

### 下一步

在第五部分中，我们将介绍 Tool Gateway（工具网关）、Memory（记忆系统）、RAG（检索增强生成）以及 Multi-Agent 协调等高级模块。
