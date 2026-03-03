# Arcana 学习路线图

> 从零到高级 Agent 工程师的完整学习路径

---

## 📚 学习资源导航

### 核心知识文档

1. **[KNOWLEDGE.md](./KNOWLEDGE.md)** - Agent 平台工程知识体系
   - 全面的知识点覆盖（1500+ 行）
   - Model Gateway、Tool Gateway、RAG 等核心模块
   - 面试高频问题汇总
   - 技术栈选型与原因

2. **[RUNTIME_KNOWLEDGE.md](./RUNTIME_KNOWLEDGE.md)** - Agent Runtime 详解
   - Runtime 核心概念与架构
   - Policy-Step-Reducer 模式深入分析
   - 执行控制、进度检测、错误处理

3. **[WEEK3-4_RUNTIME_LEARNING.md](./WEEK3-4_RUNTIME_LEARNING.md)** - Week 3-4 学习要点 ⭐ NEW
   - Policy-Step-Reducer 模式详解
   - Schema Validation 实现细节
   - Replay 机制设计思路
   - 错误分类与处理策略
   - 面试高频问题与实战经验

### 规范文档

- **[specs/trace_spec.md](./specs/trace_spec.md)** - Trace 系统规范
- **[specs/tool_spec.md](./specs/tool_spec.md)** - Tool 契约规范
- **[specs/state_spec.md](./specs/state_spec.md)** - State 状态规范
- **[specs/rag_spec.md](./specs/rag_spec.md)** - RAG 系统规范

---

## 🗓️ 14 周学习计划

### Phase 1: 基础设施（Week 1-2）✅

**学习目标**：
- 理解"平台优先"思想
- 掌握 Contract-First 设计
- 建立可观测性基础

**知识点**：
- Canonical Hash 规范
- JSONL 事件日志
- Model Gateway 抽象
- Budget Tracking

**产出**：
- ✅ Trace Writer/Reader
- ✅ Model Gateway + 多 Provider
- ✅ Budget Tracker
- ✅ 完整文档

**学习资料**：
- [KNOWLEDGE.md - 核心概念](./KNOWLEDGE.md#1-核心概念)
- [KNOWLEDGE.md - Trace 系统](./KNOWLEDGE.md#2-trace-系统审计与可追溯)
- [KNOWLEDGE.md - Model Gateway](./KNOWLEDGE.md#3-model-gateway模型网关)

---

### Phase 2: 执行引擎（Week 3-4）✅

**学习目标**：
- 掌握状态机设计模式
- 理解执行控制机制
- 学会调试与复现技巧

**核心知识点**：

1. **Policy-Step-Reducer 模式**
   - Redux 单向数据流思想
   - 关注点分离
   - 纯函数 Reducer

2. **Schema Validation**
   - LLM 输出不稳定问题
   - 宽容解析策略
   - 自动重试机制

3. **Replay 机制**
   - 确定性保证（temperature=0 + seed）
   - Cache 索引设计
   - 分歧点检测

4. **错误处理**
   - 9 种错误类型分类
   - 指数退避重试
   - 升级策略

**产出**：
- ✅ Agent 执行引擎
- ✅ Progress Detector
- ✅ Schema Validator
- ✅ Replay Engine
- ✅ Error Handler
- ✅ 集成测试 + Demo

**学习资料**：
- 📖 **[WEEK3-4_RUNTIME_LEARNING.md](./WEEK3-4_RUNTIME_LEARNING.md)** - 必读！
- [RUNTIME_KNOWLEDGE.md](./RUNTIME_KNOWLEDGE.md)

**面试准备**：
- Q: 解释 Policy-Step-Reducer 模式
- Q: Schema Validation 为什么要自动重试？
- Q: Replay 如何保证确定性？
- Q: 错误分类的意义是什么？
- Q: ProgressDetector 如何检测死循环？

---

### Phase 3: 工具系统（Week 5）🔜

**学习目标**：
- 掌握 Capability-based 权限模型
- 理解幂等性设计
- 学会注入防护

**核心知识点**：

1. **Tool Contract**
   - Input/Output Schema
   - Side Effect 分类（read/write）
   - Idempotency Key

2. **权限系统**
   - Capability 授权
   - 最小权限原则
   - 审计日志

3. **安全防护**
   - Prompt 注入检测
   - 参数校验
   - 写操作保护（HITL）

**即将产出**：
- Tool Gateway 接口
- Capability Manager
- Idempotency Handler
- 注入防护规则

**预习资料**：
- [KNOWLEDGE.md - Tool Gateway](./KNOWLEDGE.md#4-tool-gateway工具网关)

---

### Phase 4: 知识系统（Week 6-7）🔜

**Week 6: RAG v1**
- 索引与检索
- Rerank 策略
- Citation 约束

**Week 7: Memory**
- Working Memory
- Long-term Memory
- Episodic Memory
- 记忆污染处理

**预习资料**：
- [KNOWLEDGE.md - RAG 系统](./KNOWLEDGE.md#6-rag-系统)

---

### Phase 5: 编排系统（Week 8-10）🔜

**Week 8: Plan-and-Execute**
- Planner/Executor/Verifier
- 验收条件
- 重规划机制

**Week 9-10: Orchestrator**
- 任务队列
- 并发控制
- 配额管理
- 背压策略

---

### Phase 6: 治理系统（Week 11-12）🔜

**Week 11-12: Observability + Multi-Agent**
- Trace Viewer
- Eval Harness
- 对抗集
- Multi-Agent 协作协议

---

### Phase 7: 项目整合（Week 13-14）🔜

**Week 13-14: Capstone**
- 端到端 Demo
- 生产强化
- 文档完善
- 架构决策记录（ADR）

---

## 📊 当前进度

```
✅ Week 1-2:  Contracts + Trace + Model Gateway  [████████████████████] 100%
✅ Week 3-4:  Agent Runtime                       [████████████████████] 100%
🔜 Week 5:    Tool Gateway                        [                    ]   0%
⏳ Week 6-14: RAG + Memory + Orchestrator         [                    ]   0%
```

**总体进度**: 28% (4/14 周)

---

## 🎯 学习方法建议

### 1. 先理论后实践

```
每个模块学习流程：
1. 阅读知识文档（理解"为什么"）
2. 查看代码实现（理解"怎么做"）
3. 运行测试和 Demo（验证理解）
4. 独立实现小功能（深化掌握）
```

### 2. 问题驱动学习

**好问题示例**：
- "为什么需要 Canonical Hash？"
- "Schema Validation 失败后如何重试？"
- "Replay 如何保证确定性？"
- "什么时候需要 HITL？"

### 3. 对比学习

**横向对比**：
| Arcana | LangChain | AutoGPT |
|--------|-----------|---------|
| 平台化设计 | 框架化 | 应用化 |
| 可控可测 | 易用性强 | 自主性强 |
| 生产导向 | 快速原型 | 演示导向 |

**纵向对比**：
- Week 1 的 Trace vs Week 11 的 Trace Viewer
- Week 3 的简单 State vs Week 9 的 Checkpoint
- Week 5 的 Tool vs Week 12 的 Multi-Agent Tool 共享

### 4. 面试准备策略

**高频题目分类**：
1. **架构设计**：为什么这么设计？有什么好处？
2. **技术选型**：为什么选 X 而不是 Y？
3. **故障排查**：出了问题怎么调试？
4. **性能优化**：如何提高性能？
5. **安全考虑**：如何防止注入/越权？

**回答框架**：
```
1. 背景/问题
2. 方案/设计
3. 实现细节
4. 权衡与取舍
5. 实际效果
```

---

## 🔍 深度学习资源

### 推荐论文

**Agent 相关**：
- ReAct: Synergizing Reasoning and Acting
- Reflexion: Language Agents with Verbal Reinforcement Learning
- AutoGPT: An Autonomous GPT-4 Experiment

**RAG 相关**：
- Self-RAG: Learning to Retrieve, Generate, and Critique
- Dense Passage Retrieval for Open-Domain Question Answering

**Tool Use**：
- Toolformer: Language Models Can Teach Themselves to Use Tools
- Gorilla: Large Language Model Connected with Massive APIs

### 开源项目研究

**必看**：
- LangChain/LangGraph - 学习编排思想
- DSPy - 学习优化思想
- AutoGPT - 学习自主 Agent 设计

**进阶**：
- Semantic Kernel (Microsoft)
- Haystack (deepset)
- BabyAGI

### 技术博客

**官方文档**：
- Anthropic: Claude Best Practices
- OpenAI: Function Calling Guide
- Google: Gemini Agent Patterns

**社区博客**：
- Pinecone: Vector DB Tutorials
- LangChain Blog
- Hugging Face Blog

---

## 💡 实战项目建议

### 小项目（练手）

1. **简单 RAG 问答**
   - 索引本地文档
   - 实现检索+回答
   - 强制引用约束

2. **带预算的 Agent**
   - 设定 token 限制
   - 实现降级策略
   - 记录成本

3. **Tool Calling Agent**
   - 集成 2-3 个工具
   - 实现权限控制
   - 记录审计日志

### 中型项目（深化）

1. **可 Replay 的 Agent**
   - 完整的 Trace 系统
   - Replay Engine
   - 分歧点检测

2. **Multi-Agent 协作**
   - Planner + Executor + Critic
   - 协作协议
   - 冲突解决

### 大型项目（综合）

1. **企业知识助手**
   - RAG + 工具编排
   - HITL + 审计
   - 回归门禁

2. **Agent Platform**
   - 完整的 Orchestrator
   - 多租户隔离
   - 可观测性平台

---

## ✅ 自测清单

### Week 1-2 自测

- [ ] 能解释为什么需要 Canonical Hash
- [ ] 能说出 JSONL 的 3 个优点
- [ ] 能实现一个简单的 ModelGateway Provider
- [ ] 能设计 Budget Tracker 的数据结构

### Week 3-4 自测

- [ ] 能画出 Policy-Step-Reducer 的数据流图
- [ ] 能解释 Schema Validation 的重试策略
- [ ] 能说出 Replay 确定性的 3 个保证
- [ ] 能列举 5 种以上错误类型
- [ ] 能实现一个简单的 ProgressDetector

### Week 5+ 自测（待补充）

---

## 🎓 进阶路径

### 初级 Agent 工程师（Week 1-4）

**能力要求**：
- ✅ 理解 Agent 基本架构
- ✅ 能使用现有 Agent 框架
- ✅ 能调试简单的 Agent 问题

**对应岗位**：
- Agent 应用开发
- AI 产品工程师（初级）

### 中级 Agent 工程师（Week 5-10）

**能力要求**：
- 能设计 Agent 系统架构
- 能实现核心模块（Tool/RAG/Memory）
- 能处理生产环境问题

**对应岗位**：
- Agent 平台工程师
- AI 基础设施工程师
- 高级后端工程师（AI 方向）

### 高级 Agent 架构师（Week 11-14+）

**能力要求**：
- 能设计大规模 Agent 平台
- 能权衡技术选型
- 能指导团队实施

**对应岗位**：
- Agent 平台架构师
- AI Infra Tech Lead
- 资深/专家工程师（AI 方向）

---

## 📞 学习支持

### 遇到问题？

1. **查文档**：先看对应周的学习文档
2. **看代码**：阅读相关模块的实现
3. **跑测试**：运行测试理解行为
4. **问 AI**：Claude/GPT 可以解答具体问题

### 贡献文档

发现文档有误或可以改进？欢迎提交 PR！

---

## 🎉 总结

**学习原则**：
1. **理解为先**：不要死记硬背，理解"为什么"
2. **实践验证**：动手实现才能真正掌握
3. **持续总结**：每周整理知识点和心得
4. **面试导向**：注重高频问题的准备

**下一步**：
- 📖 完成当前周的学习文档阅读
- 💻 运行对应的 Demo 和测试
- ✍️ 总结学习笔记和疑问
- 🚀 开始下一周的学习

祝学习顺利！🎯
