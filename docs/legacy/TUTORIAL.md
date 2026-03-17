# Arcana Agent Framework 深度教程

> **Arcana** — 一个**可控、可复现、可评测**的 Agent 平台框架
>
> 本教程从架构设计到每一行核心代码，带你理解一个生产级 Agent 框架是如何构建的。

---

## 总目录

> 点击链接跳转到对应章节的详细文档。每章都是独立的 Markdown 文件，包含完整的代码分析和 `file:line` 引用。

### [第一章：架构总览与 Contracts 层](./tutorial_part1.md)

框架的基石——理解 Arcana 的设计哲学和数据合约。

| 章节 | 内容 |
|------|------|
| [1. 框架总览](./tutorial_part1.md#1-框架总览) | 设计哲学：可控、可复现、可评测；Contracts-First 方法论 |
| [2. 架构分层](./tutorial_part1.md#2-架构分层) | 层次结构图、模块依赖关系 |
| [3. Contracts 层详解](./tutorial_part1.md#3-contracts-层详解) | 13 个合约文件逐一剖析（trace/tool/state/llm/runtime/plan/graph/rag/memory/multi_agent/orchestrator/eval） |
| [4. Trace 系统](./tutorial_part1.md#4-trace-系统) | TraceWriter/TraceReader 深度分析，JSONL 审计日志 |
| [5. Canonical Hashing](./tutorial_part1.md#5-canonical-hashing) | 值规范化、浮点精度、SHA-256 截断、为什么需要 Canonical JSON |
| [6. 配置系统](./tutorial_part1.md#6-配置系统) | ArcanaConfig、环境变量加载、单例模式 |
| [7. 本章小结](./tutorial_part1.md#7-本章小结) | 设计模式汇总、关键收获 |

**核心技术点**：Pydantic v2 Schema、JSONL 追加写入、SHA-256 Canonical Hash、枚举驱动设计

---

### [第二章：Model Gateway 与 Provider 系统](./tutorial_part2.md)

统一模型接入层——如何用一个类对接 8+ 家 LLM 供应商。

| 章节 | 内容 |
|------|------|
| [1. Model Gateway 概述](./tutorial_part2.md#1-model-gateway-概述) | 为什么需要网关？三大核心问题（锁定、成本、单点故障） |
| [2. 抽象基类设计](./tutorial_part2.md#2-抽象基类设计) | ModelGateway ABC、generate() 接口、异常体系与 retryable 标志 |
| [3. Provider 适配器模式](./tutorial_part2.md#3-provider-适配器模式) | OpenAICompatibleProvider 一个类通吃所有供应商、消息转换、工厂函数 |
| [4. 注册表与路由](./tutorial_part2.md#4-注册表与路由) | ModelGatewayRegistry、fallback chain、路由决策流程图 |
| [5. 预算控制](./tutorial_part2.md#5-预算控制) | BudgetTracker 三维限制（token/cost/time）、线程安全、预测性检查 |
| [6. Trace 集成](./tutorial_part2.md#6-trace-集成) | 自动追踪、request_digest/response_digest |
| [7. 设计模式总结](./tutorial_part2.md#7-设计模式总结) | Strategy、Adapter、Registry、Factory、Chain of Responsibility、Null Object |
| [8. 生产注意事项](./tutorial_part2.md#8-生产注意事项) | 限流退避、熔断器、成本优化、供应商多样性、安全性 |

**核心技术点**：AsyncIO、OpenAI SDK 复用、Pydantic model_copy()、threading.Lock 线程安全

---

### [第三章：Runtime 引擎](./tutorial_part3.md)

Agent 执行的心脏——状态机驱动的自治循环。

| 章节 | 内容 |
|------|------|
| [1. Runtime 引擎概述](./tutorial_part3.md#1-runtime-引擎概述) | 状态机执行模型、状态转换图 |
| [2. Agent 主循环](./tutorial_part3.md#2-agent-主循环) | 初始化、run loop、5 种停止条件、checkpoint 策略、resume |
| [3. 策略模式 (Policy)](./tutorial_part3.md#3-策略模式-policy) | BasePolicy → ReActPolicy（Think-Act-Observe）→ PlanExecutePolicy（Plan-Execute-Verify） |
| [4. 步骤执行器 (StepExecutor)](./tutorial_part3.md#4-步骤执行器-stepexecutor) | LLM 调用 + 验证重试、工具执行、响应解析 |
| [5. 状态归约器 (Reducer)](./tutorial_part3.md#5-状态归约器-reducer) | DefaultReducer、PlanReducer、继承组合 |
| [6. 状态管理器 (StateManager)](./tutorial_part3.md#6-状态管理器-statemanager) | 状态转换验证、checkpoint JSONL 持久化、哈希完整性校验 |
| [7. 进度检测 (ProgressDetector)](./tutorial_part3.md#7-进度检测-progressdetector) | 重复检测、循环模式检测、输出相似度检测 |
| [8. 输出验证 (Validator)](./tutorial_part3.md#8-输出验证-validator) | JSON Schema 校验、Markdown 提取、带错误提示的重试 |
| [9. Replay 引擎](./tutorial_part3.md#9-replay-引擎) | ReplayCache、确定性回放、分歧点检测、逐步状态重建 |
| [10. 错误处理](./tutorial_part3.md#10-错误处理) | 异常层级、ErrorType 分类、RetryStrategy 指数退避+抖动 |
| [11. Hook 系统](./tutorial_part3.md#11-hook-系统) | Protocol-based RuntimeHook、MemoryHook 集成 |
| [12. 验证器 (Verifier)](./tutorial_part3.md#12-验证器-verifier) | GoalVerifier 词汇重叠判定 |
| [13. 数据流图](./tutorial_part3.md#13-数据流图) | 完整请求流 ASCII 图、组件交互序列图 |
| [14. 生产注意事项](./tutorial_part3.md#14-生产注意事项) | 预算、checkpoint 存储、错误升级、Hook 性能、幂等 |

**核心技术点**：有限状态机、策略模式、Protocol 类型协议、指数退避+抖动、Canonical Hash 去重

---

### [第四章：Graph Engine 声明式图编排引擎](./tutorial_part4.md)

复杂工作流的解决方案——声明式图编排 + 人机协作。

| 章节 | 内容 |
|------|------|
| [1. Graph Engine 概述](./tutorial_part4.md#1-graph-engine-概述) | Runtime vs Graph 对比表、何时用哪个 |
| [2. StateGraph 构建器](./tutorial_part4.md#2-stategraph-构建器) | Builder 模式、add_node/add_edge/add_conditional_edges、编译验证 |
| [3. CompiledGraph](./tutorial_part4.md#3-compiledgraph-编译产物) | ainvoke/astream/aresume 三大执行方法 |
| [4. GraphExecutor 执行引擎](./tutorial_part4.md#4-graphexecutor-执行引擎) | 执行循环 6 步、路由机制（条件边 > 直接边）、状态初始化、恢复执行 |
| [5. NodeRunner](./tutorial_part4.md#5-noderunner-节点执行器) | 同步/异步兼容、输出规范化、追踪集成 |
| [6. 流式执行 (Streaming)](./tutorial_part4.md#6-流式执行-streaming) | 生产者-消费者架构、三种模式（values/updates/messages）、消息去重 |
| [7. Checkpointer](./tutorial_part4.md#7-checkpointer-状态持久化) | JSON 文件持久化、save/load/delete |
| [8. 中断与恢复 (Interrupt)](./tutorial_part4.md#8-中断与恢复-interrupt) | GraphInterrupt 异常、Command（resume/update/goto）、Human-in-the-Loop 完整流程 |
| [9. Reducers](./tutorial_part4.md#9-reducers-状态合并策略) | 5 种内置 Reducer、Annotated 声明、add_messages 去重 |
| [10. 预置图 (Prebuilt)](./tutorial_part4.md#10-预置图-prebuilt) | ReAct Agent、Plan-Execute Agent 模板 |
| [11. 节点类型](./tutorial_part4.md#11-节点类型) | LLMNode、ToolNode、SubgraphNode |
| [12. 与 LangGraph 的对比](./tutorial_part4.md#12-与-langgraph-的对比) | API 相似性、关键差异、设计哲学对比 |
| [13. 生产注意事项](./tutorial_part4.md#13-生产注意事项) | Checkpointer 替换、max_iterations、错误处理、图验证局限 |

**核心技术点**：Builder 模式、AsyncGenerator 流式、asyncio.Queue 生产者-消费者、BFS 可达性验证、Annotated 类型元编程

---

### [第五章：平台服务层](./tutorial_part5.md)

Tool Gateway、Memory、RAG、Storage、Multi-Agent、Orchestrator、Observability、Eval——完整的平台治理能力。

| 章节 | 内容 |
|------|------|
| [1. Tool Gateway（工具网关）](./tutorial_part5.md#1-tool-gateway工具网关) | 7 步授权流水线、Capability 权限模型、幂等性缓存、写操作确认、读写分离并发、LangChain 适配器 |
| [2. Memory 系统](./tutorial_part5.md#2-memory-系统) | 三层架构（Working/Long-Term/Episodic）、Write Governance 置信度门控、记忆去污染 |
| [3. RAG 流水线](./tutorial_part5.md#3-rag-流水线) | 分块策略（固定/段落/递归）、OpenAI Embedder、BM25 混合重排、CitationVerifier 引用验证 |
| [4. Storage 抽象层](./tutorial_part5.md#4-storage-抽象层) | StorageBackend + VectorStore 双抽象、InMemory 实现、ChromaDB 实现、懒导入 |
| [5. Multi-Agent 协作](./tutorial_part5.md#5-multi-agent-协作) | PEC 模式（Planner-Executor-Critic）、RoleConfig、MessageBus、预算与错误控制 |
| [6. Orchestrator（编排器）](./tutorial_part5.md#6-orchestrator编排器) | TaskGraph DAG + 环路检测、优先级调度、Semaphore 并发池、指数退避重试 |
| [7. Observability（可观测性）](./tutorial_part5.md#7-observability可观测性) | MetricsCollector P50/P95/P99、MetricsHook 实时监控、告警建议 |
| [8. Eval Harness（评测框架）](./tutorial_part5.md#8-eval-harness评测框架) | EvalRunner 6 种判定标准、RegressionGate 绝对+相对检查 |
| [9. 生产部署全景](./tutorial_part5.md#9-生产部署全景) | 8 个模块的开发 vs 生产环境对比表 |

**核心技术点**：Pipeline 模式、Capability-based AuthZ、BM25 算法、asyncio.Semaphore、DFS 环路检测、百分位数统计

---

## 快速导航：按主题查找

### 设计模式

| 模式 | 位置 | 说明 |
|------|------|------|
| Contracts-First | [Part 1](./tutorial_part1.md#1-框架总览) | 先定义 Schema，再写实现 |
| Strategy | [Part 2](./tutorial_part2.md#7-设计模式总结), [Part 3](./tutorial_part3.md#3-策略模式-policy) | Provider/Policy 可互换 |
| Adapter | [Part 2](./tutorial_part2.md#3-provider-适配器模式) | OpenAI 格式统一适配 |
| Registry | [Part 2](./tutorial_part2.md#4-注册表与路由) | Provider/Tool 注册与路由 |
| Builder | [Part 4](./tutorial_part4.md#2-stategraph-构建器) | StateGraph 声明式构建 |
| Observer/Hook | [Part 3](./tutorial_part3.md#11-hook-系统) | Runtime 生命周期回调 |
| Pipeline | [Part 5](./tutorial_part5.md#1-tool-gateway工具网关) | Tool Gateway 7 步流水线 |
| State Machine | [Part 3](./tutorial_part3.md#1-runtime-引擎概述) | Agent 状态转换 |
| Null Object | [Part 2](./tutorial_part2.md#5-预算控制) | BudgetTracker 无限制模式 |

### 生产关键能力

| 能力 | 位置 | 说明 |
|------|------|------|
| 预算控制 | [Part 2 §5](./tutorial_part2.md#5-预算控制) | Token/Cost/Time 三维限制 |
| 故障转移 | [Part 2 §4](./tutorial_part2.md#4-注册表与路由) | Fallback chain 自动切换 |
| 幂等性 | [Part 5 §1.4](./tutorial_part5.md#1-tool-gateway工具网关) | Tool Gateway 幂等缓存 |
| 审计追踪 | [Part 1 §4](./tutorial_part1.md#4-trace-系统) | JSONL Trace 全链路 |
| 循环检测 | [Part 3 §7](./tutorial_part3.md#7-进度检测-progressdetector) | 3 种检测机制 |
| 确定性回放 | [Part 3 §9](./tutorial_part3.md#9-replay-引擎) | Replay 引擎 |
| 人机协作 | [Part 4 §8](./tutorial_part4.md#8-中断与恢复-interrupt) | Graph Interrupt/Resume |
| 质量门禁 | [Part 5 §8](./tutorial_part5.md#8-eval-harness评测框架) | RegressionGate |
| 记忆治理 | [Part 5 §2.5](./tutorial_part5.md#2-memory-系统) | Write Governance |
| 引用验证 | [Part 5 §3.6](./tutorial_part5.md#3-rag-流水线) | CitationVerifier |

### 核心源文件索引

| 文件 | 章节 | 职责 |
|------|------|------|
| `contracts/*.py` | [Part 1 §3](./tutorial_part1.md#3-contracts-层详解) | 全部数据模型 |
| `trace/writer.py` | [Part 1 §4](./tutorial_part1.md#4-trace-系统) | JSONL 事件写入 |
| `utils/hashing.py` | [Part 1 §5](./tutorial_part1.md#5-canonical-hashing) | Canonical Hash |
| `gateway/base.py` | [Part 2 §2](./tutorial_part2.md#2-抽象基类设计) | ModelGateway ABC |
| `gateway/providers/openai_compatible.py` | [Part 2 §3](./tutorial_part2.md#3-provider-适配器模式) | 万能 Provider |
| `gateway/registry.py` | [Part 2 §4](./tutorial_part2.md#4-注册表与路由) | 路由 + Fallback |
| `gateway/budget.py` | [Part 2 §5](./tutorial_part2.md#5-预算控制) | 预算追踪器 |
| `runtime/agent.py` | [Part 3 §2](./tutorial_part3.md#2-agent-主循环) | 主执行循环 |
| `runtime/policies/react.py` | [Part 3 §3](./tutorial_part3.md#3-策略模式-policy) | ReAct 策略 |
| `runtime/progress.py` | [Part 3 §7](./tutorial_part3.md#7-进度检测-progressdetector) | 循环检测 |
| `runtime/replay.py` | [Part 3 §9](./tutorial_part3.md#9-replay-引擎) | 确定性回放 |
| `graph/state_graph.py` | [Part 4 §2](./tutorial_part4.md#2-stategraph-构建器) | 图构建器 |
| `graph/executor.py` | [Part 4 §4](./tutorial_part4.md#4-graphexecutor-执行引擎) | 图执行引擎 |
| `graph/interrupt.py` | [Part 4 §8](./tutorial_part4.md#8-中断与恢复-interrupt) | 中断/恢复 |
| `graph/reducers.py` | [Part 4 §9](./tutorial_part4.md#9-reducers-状态合并策略) | 状态合并 |
| `tool_gateway/gateway.py` | [Part 5 §1](./tutorial_part5.md#1-tool-gateway工具网关) | 工具授权流水线 |
| `memory/manager.py` | [Part 5 §2](./tutorial_part5.md#2-memory-系统) | 记忆统一门面 |
| `rag/retriever.py` | [Part 5 §3](./tutorial_part5.md#3-rag-流水线) | RAG 检索核心 |
| `orchestrator/orchestrator.py` | [Part 5 §6](./tutorial_part5.md#6-orchestrator编排器) | 任务编排 |
| `eval/gate.py` | [Part 5 §8](./tutorial_part5.md#8-eval-harness评测框架) | 质量门禁 |

---

## 技术栈总览

| 技术 | 用途 | 为什么选它 |
|------|------|-----------|
| **Python 3.11+** | 主语言 | asyncio 原生支持、类型提示成熟 |
| **Pydantic v2** | 数据合约 | 快速校验、JSON Schema 生成、跨语言迁移基础 |
| **AsyncIO** | 并发模型 | LLM 调用 I/O 密集，异步提升吞吐 |
| **httpx** | HTTP 客户端 | 异步原生、比 requests 更现代 |
| **OpenAI SDK** | LLM 调用 | 复用成熟连接池和重试机制 |
| **JSONL** | Trace 存储 | 追加写入、人类可读、流式解析 |
| **SHA-256** | 内容哈希 | 密码学安全、确定性、标准化 |
| **threading.Lock** | 线程安全 | 保护 BudgetTracker、幂等缓存等共享状态 |
| **asyncio.Semaphore** | 并发控制 | ExecutorPool 槽位管理 |
| **ChromaDB** | 向量存储 | 开源、支持持久化、HNSW 索引 |
| **pytest + pytest-asyncio** | 测试 | 异步测试原生支持 |
| **ruff** | Lint/Format | 极快、替代 flake8+black+isort |
| **mypy (strict)** | 类型检查 | 编译期捕获类型错误 |

---

## 生产部署 Checklist

在将 Arcana 部署到生产环境前，确认以下事项：

### 必须完成

- [ ] **存储替换**：将 InMemoryBackend 替换为 PostgreSQL/Redis
- [ ] **向量库**：将 InMemoryVectorStore 替换为 ChromaDB（持久模式）/ pgvector
- [ ] **密钥管理**：从 .env 迁移到 Vault / AWS Secrets Manager
- [ ] **预算配置**：为每种任务类型设置 token/cost/time 上限
- [ ] **Fallback 链**：配置跨供应商的故障转移链
- [ ] **质量门禁**：在 CI 中集成 EvalRunner + RegressionGate

### 强烈建议

- [ ] **熔断器**：为频繁失败的 Provider 添加 Circuit Breaker
- [ ] **指标导出**：将 MetricsCollector 输出接入 Prometheus/Datadog
- [ ] **幂等缓存**：将 Tool Gateway 缓存迁移到 Redis + TTL
- [ ] **Checkpointer**：将 JSON 文件 Checkpointer 替换为 Redis/PostgreSQL
- [ ] **消息总线**：将 MessageBus 从进程内 Queue 迁移到 Redis Streams
- [ ] **日志级别**：生产环境设为 WARNING，开发环境 DEBUG

### 可选增强

- [ ] **OpenTelemetry**：分布式追踪集成
- [ ] **PII 检测**：Trace 中的敏感信息过滤
- [ ] **RBAC**：基于角色的工具访问控制（替代 Capability-based）
- [ ] **Grafana 仪表盘**：实时监控面板
- [ ] **对抗测试集**：注入、越权、循环、超时等异常场景
- [ ] **Trusted Execution**：敏感工具隔离执行（TEE/签名 receipt）

---

## 阅读建议

1. **初学者**：按顺序 Part 1 → 2 → 3 → 4 → 5 阅读，每章结束后看"本章小结"
2. **赶时间**：只看本文件的目录 + 每章的"小结"部分
3. **面试准备**：重点看设计模式表格 + 生产注意事项
4. **想动手**：先跑 `examples/` 下的 demo，然后对照源码阅读对应章节
5. **找特定内容**：用上面的"按主题查找"表格快速定位

---

> 本教程由 5 个并行 Agent 分析 90+ 个源文件后生成，涵盖 6800+ 行技术文档。
