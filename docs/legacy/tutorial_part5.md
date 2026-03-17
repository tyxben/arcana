# Part 5: 平台服务层 -- Tool Gateway、Memory、RAG、Storage、Multi-Agent、Orchestrator、Observability、Eval

> 本章覆盖 Arcana 框架中除核心运行时以外的所有**平台服务模块**。这些模块共同构成了一个生产级 Agent 平台的"地基"：工具调用如何被管控、记忆如何持久化与检索、文档如何被索引与引用、多 Agent 如何协作、任务如何被编排调度、运行时指标如何采集、以及质量如何被自动化守门。

---

## 目录

1. [Tool Gateway（工具网关）](#1-tool-gateway工具网关)
2. [Memory 系统](#2-memory-系统)
3. [RAG 流水线](#3-rag-流水线)
4. [Storage 抽象层](#4-storage-抽象层)
5. [Multi-Agent 协作](#5-multi-agent-协作)
6. [Orchestrator（编排器）](#6-orchestrator编排器)
7. [Observability（可观测性）](#7-observability可观测性)
8. [Eval Harness（评测框架）](#8-eval-harness评测框架)
9. [生产部署全景](#9-生产部署全景)
10. [本章小结](#10-本章小结)

---

## 1. Tool Gateway（工具网关）

### 1.1 为什么需要工具网关？

LLM Agent 的核心能力之一是调用外部工具（搜索、数据库、API 等）。但直接让 LLM 调用工具存在严重隐患：

- **安全**：Agent 可能调用它没有权限的工具
- **正确性**：参数可能不符合 schema
- **幂等性**：重复调用同一操作可能产生副作用
- **可审计**：生产环境必须知道"谁在什么时候调用了什么"
- **可靠性**：网络抖动、超时需要重试

Arcana 的 ToolGateway 将这些横切关注点统一为一条**授权流水线**：

```
Resolve → Authorize → Validate → Idempotency → Confirm → Execute+Retry → Audit
```

### 1.2 ToolProvider 抽象基类

每个工具实现为一个 `ToolProvider`（`src/arcana/tool_gateway/base.py:10`）：

```python
class ToolProvider(ABC):
    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """返回工具规格说明。"""
        ...

    @abstractmethod
    async def execute(self, call: ToolCall) -> ToolResult:
        """执行工具调用。"""
        ...

    async def health_check(self) -> bool:
        """检查工具是否可用。"""
        return True
```

**设计决策**：ToolProvider 只负责"执行"，所有治理逻辑（授权、校验、重试、审计）都在 ToolGateway 层处理。这种职责分离让工具实现者可以专注于业务逻辑，不需要重复编写安全检查代码。

`ToolExecutionError`（`base.py:43`）区分了**可重试**和**不可重试**错误：

```python
class ToolExecutionError(Exception):
    def __init__(self, message: str, tool_name: str,
                 retryable: bool = False, error_code: str | None = None):
        self.retryable = retryable  # 关键：决定是否重试
        self.error_code = error_code
```

### 1.3 ToolRegistry -- 工具注册中心

`ToolRegistry`（`src/arcana/tool_gateway/registry.py:11`）管理所有已注册的工具：

```python
registry = ToolRegistry()
registry.register(my_search_tool)
registry.register(my_database_tool)

# 列出所有工具
print(registry.list_tools())  # ["search", "database"]

# 转换为 OpenAI function calling 格式
openai_tools = registry.to_openai_tools()

# 健康检查
health = await registry.health_check_all()
# {"search": True, "database": False}
```

`to_openai_tools()`（`registry.py:67`）将内部 ToolSpec 转换为 OpenAI 格式，这意味着无论底层使用哪个 LLM 提供商，工具定义只需要写一次。

### 1.4 Gateway 流水线详解

`ToolGateway.call()`（`src/arcana/tool_gateway/gateway.py:100`）是整条流水线的入口：

**第 1 步：Resolve -- 查找工具提供者**

```python
provider = self.registry.get(tool_call.name)
if provider is None:
    return ToolResult(success=False, error=ToolError(
        error_type=ErrorType.NON_RETRYABLE,
        message=f"Tool '{tool_call.name}' not found in registry",
        code="TOOL_NOT_FOUND",
    ))
```

**第 2 步：Authorize -- 能力检查**

```python
def _check_capabilities(self, spec: ToolSpec) -> list[str]:
    required = set(spec.capabilities)
    missing = required - self.granted_capabilities
    return sorted(missing)
```

这是**基于能力（capability-based）的授权模型**。每个工具声明自己需要的能力（如 `["web_search", "file_read"]`），每个 Agent 被授予一组能力。不匹配则拒绝。授权失败会被单独审计（`_log_authorization_failure`，`gateway.py:390`），这对安全分析至关重要。

**第 3 步：Validate -- 参数校验**

`validate_arguments()`（`src/arcana/tool_gateway/validators.py:21`）对参数进行 JSON Schema 校验：

```python
def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> ToolError | None:
    # 检查必填字段
    missing = check_required_fields(schema, arguments)
    # 检查类型匹配
    type_errors = check_types(schema, arguments)
```

校验错误是 `NON_RETRYABLE` -- 参数不对重试也没用。

**第 4 步：Idempotency Cache -- 幂等性缓存**

```python
def _check_idempotency(self, key: str | None) -> ToolResult | None:
    if key is None:
        return None
    with self._cache_lock:
        return self._idempotency_cache.get(key)
```

当 `ToolCall.idempotency_key` 存在时，相同的 key 会返回缓存结果。这避免了 LLM 重复生成相同调用时产生副作用（如重复发邮件、重复下单）。

**第 5 步：Confirm -- 写操作确认**

```python
async def _confirm_execution(self, tool_call, spec) -> ToolResult | None:
    if spec.side_effect != SideEffect.WRITE and not spec.requires_confirmation:
        return None  # 读操作直接放行

    if self.confirmation_callback is None:
        return ToolResult(success=False, error=ToolError(
            error_type=ErrorType.REQUIRES_HUMAN,
            code="CONFIRMATION_REQUIRED",
        ))

    confirmed = await self.confirmation_callback(tool_call, spec)
    if not confirmed:
        return ToolResult(success=False, code="CONFIRMATION_REJECTED")
```

对于有副作用的写操作（如删除文件、发送消息），Gateway 要求人类确认。这是**人机协作（Human-in-the-Loop）**的关键机制。

**第 6 步：Execute with Retry -- 带重试的执行**

`_execute_with_retry()`（`gateway.py:282`）实现了指数退避重试：

```python
for attempt in range(1 + spec.max_retries):
    try:
        result = await asyncio.wait_for(
            provider.execute(call),
            timeout=spec.timeout_ms / 1000.0,
        )
        if result.success:
            return result
        if result.error and result.error.is_retryable and attempt < spec.max_retries:
            delay = spec.retry_delay_ms * (2**attempt) / 1000.0
            await asyncio.sleep(delay)
            continue
        return result
    except TimeoutError:
        # 超时是可重试的
        ...
    except ToolExecutionError as e:
        # 根据 retryable 标记决定
        ...
    except Exception:
        # 未知异常不重试
        break
```

退避策略是 `delay_ms * 2^attempt`，防止雪崩式重试。

**第 7 步：Audit -- 审计日志**

```python
record = ToolCallRecord(
    name=tool_call.name,
    args_digest=canonical_hash(tool_call.arguments),   # 参数摘要，不存明文
    idempotency_key=tool_call.idempotency_key,
    result_digest=canonical_hash(result_data),
    side_effect=spec.side_effect.value,
    duration_ms=result.duration_ms,
)
```

注意：审计日志存储的是参数和结果的**摘要（digest）**而非明文。这既满足了审计需求，又避免了敏感数据泄露。

### 1.5 批量调用：读写分离

`call_many()`（`gateway.py:182`）对多个工具调用实现了智能并发：

```python
# 读操作并发执行
read_tasks = [self.call(tc, trace_ctx=trace_ctx) for _, tc in read_calls]
read_results = await asyncio.gather(*read_tasks)

# 写操作顺序执行
for idx, tc in write_calls:
    result = await self.call(tc, trace_ctx=trace_ctx)
```

**为什么？** 读操作无副作用，并发安全；写操作可能有依赖关系（如先创建文件再写入），必须顺序执行。

### 1.6 LangChain 适配器

`LangChainToolAdapter`（`src/arcana/tool_gateway/adapters/langchain.py:37`）让 LangChain 生态的工具无缝接入 Arcana：

```python
from langchain_community.tools import WikipediaQueryRun
from arcana.tool_gateway.adapters.langchain import LangChainToolAdapter

lc_tool = WikipediaQueryRun(...)
adapter = LangChainToolAdapter(lc_tool, side_effect=SideEffect.READ)
registry.register(adapter)
# 现在 Wikipedia 工具也走 Arcana 的授权/审计流水线了
```

适配器自动从 LangChain 工具提取 schema（`langchain.py:68`），通过 `ainvoke` 执行。关键约束：**所有治理逻辑仍由 ToolGateway 处理**，适配器只桥接执行接口。

### 1.7 生产注意事项

| 关注点 | 建议 |
|--------|------|
| 幂等性缓存 | 当前是内存字典，生产环境需换 Redis 并设置 TTL |
| 线程安全 | `_cache_lock` 使用 `threading.Lock`，适合单进程；多进程需分布式锁 |
| 确认回调 | 可接入 Slack/Teams 审批流 |
| 健康检查 | 定期执行 `registry.health_check_all()` 并接入告警 |

---

## 2. Memory 系统

### 2.1 三层记忆架构

Arcana 的记忆系统借鉴人类认知模型，分为三层：

| 层级 | 类比 | 实现 | 用途 |
|------|------|------|------|
| Working Memory | 工作记忆 | KV 存储 | 单次运行的临时状态 |
| Long-Term Memory | 长期记忆 | 向量数据库 | 跨运行的持久知识 |
| Episodic Memory | 情景记忆 | 事件日志 | 运行轨迹回放 |

### 2.2 Working Memory -- 运行时KV

`WorkingMemoryStore`（`src/arcana/memory/working.py:15`）是最简单的一层：

```python
class WorkingMemoryStore:
    def _namespace(self, run_id: str) -> str:
        return f"{self.config.working_namespace_prefix}:{run_id}"

    async def put(self, run_id: str, key: str, entry: MemoryEntry) -> None:
        ns = self._namespace(run_id)
        await self.backend.put(ns, key, entry.model_dump(mode="json"))
        # 维护 __keys__ 元键用于遍历
        keys = await self.backend.get(ns, "__keys__") or []
        if key not in keys:
            keys.append(key)
            await self.backend.put(ns, "__keys__", keys)
```

**设计要点**：
- 以 `run_id` 为命名空间隔离不同运行
- `__keys__` 元键追踪所有键，支持 `get_all()` 遍历
- 软删除（revoke）保留审计历史，硬删除（delete）彻底移除

```python
async def revoke(self, run_id: str, key: str, reason: str) -> bool:
    entry.revoked = True
    entry.revoked_at = datetime.now(UTC)
    entry.revoked_reason = reason  # 必须说明原因
    # 更新存储，不删除
```

### 2.3 Long-Term Memory -- 语义检索

`LongTermMemoryStore`（`src/arcana/memory/long_term.py:15`）结合向量检索和 KV 存储：

```python
async def store(self, entry: MemoryEntry) -> None:
    # 1. 生成 embedding
    embeddings = await self.embedder.embed([entry.content])

    # 2. 写入向量库（用于语义搜索）
    metadata = self._build_metadata(entry)
    await self.vector_store.upsert(
        id=entry.id, embedding=embeddings[0],
        metadata=metadata, content=entry.content,
    )

    # 3. 写入 KV（存储完整 MemoryEntry）
    await self.backend.put(_LTM_NAMESPACE, entry.id,
                           entry.model_dump(mode="json"))
```

**为什么需要双写？**

向量库适合搜索但不适合存储复杂结构；KV 存储完整 entry 但不能语义搜索。两者配合实现了"搜索用向量，水合用 KV"的模式（`long_term.py:76`）：

```python
async def search(self, query: MemoryQuery) -> list[MemoryEntry]:
    # 向量搜索得到 ID
    results = await self.vector_store.search(
        query_embedding=query_embedding, top_k=query.top_k,
        filters=filters,
    )
    # 用 ID 从 KV 水合完整条目
    for result in results:
        data = await self.backend.get(_LTM_NAMESPACE, result.id)
        entry = MemoryEntry.model_validate(data)
        if entry.confidence >= query.min_confidence:
            entries.append(entry)
```

撤销操作（`long_term.py:93`）需要同步更新两处：

```python
async def revoke(self, entry_id: str, reason: str) -> bool:
    # 更新 KV
    await self.backend.put(_LTM_NAMESPACE, entry.id, entry.model_dump(mode="json"))
    # 更新向量库元数据（需要重新 embed 才能更新 metadata）
    embeddings = await self.embedder.embed([entry.content])
    await self.vector_store.upsert(id=entry.id, embedding=embeddings[0],
                                    metadata=metadata, content=entry.content)
```

### 2.4 Episodic Memory -- 事件轨迹

`EpisodicMemoryStore`（`src/arcana/memory/episodic.py:16`）基于 Trace 系统记录记忆操作历史：

```python
async def record_event(self, run_id: str, entry: MemoryEntry) -> None:
    if self.trace_writer:
        event = TraceEvent(
            run_id=run_id,
            event_type=EventType.MEMORY_WRITE,
            metadata={
                "memory_entry_id": entry.id,
                "content_preview": entry.content[:200],  # 只存预览
            },
        )
        self.trace_writer.write(event)
```

情景记忆的价值在于**回放和调试**：可以追溯"某个记忆是在哪次运行的哪个步骤被创建的"。`get_trajectory()`（`episodic.py:60`）重建完整轨迹。

### 2.5 Write Governance -- 写入治理

`WritePolicy`（`src/arcana/memory/governance.py:16`）是记忆系统的守门人：

```python
class WritePolicy:
    def evaluate(self, request: MemoryWriteRequest) -> MemoryWriteResult:
        # 低于最低置信度：拒绝
        if request.confidence < self.config.min_write_confidence:
            return MemoryWriteResult(
                success=False,
                rejected_reason=f"Confidence {request.confidence:.2f} below threshold",
            )
        # 低于警告阈值：记录警告但放行
        if request.confidence < self.config.warn_confidence_threshold:
            logger.warning("Low confidence memory write: %.2f", request.confidence)
        return MemoryWriteResult(success=True)
```

**为什么需要置信度门控？** Agent 可能从不可靠来源获取信息（如未经验证的网页）。低置信度写入被拒绝可以防止"记忆污染"，这在长期运行的 Agent 中至关重要。

撤销验证确保同一条目不会被重复撤销：

```python
def validate_revocation(self, entry: MemoryEntry, request: RevocationRequest) -> bool:
    return not entry.revoked  # 已撤销的不能再撤销
```

### 2.6 MemoryManager -- 统一门面

`MemoryManager`（`src/arcana/memory/manager.py:24`）是记忆系统的唯一入口：

```python
async def write(self, request: MemoryWriteRequest) -> MemoryWriteResult:
    # 1. 治理检查
    result = self.governance.evaluate(request)
    if not result.success:
        return result

    # 2. 创建 MemoryEntry
    entry = MemoryEntry(id=str(uuid4()), ...)

    # 3. 根据类型路由到对应存储
    if request.memory_type == MemoryType.WORKING:
        await self.working.put(request.run_id, request.key, entry)
    elif request.memory_type == MemoryType.LONG_TERM:
        await self.long_term.store(entry)
    elif request.memory_type == MemoryType.EPISODIC:
        await self.episodic.record_event(request.run_id, entry)

    # 4. 所有写入都记录到情景记忆（审计）
    if request.run_id and request.memory_type != MemoryType.EPISODIC:
        await self.episodic.record_event(request.run_id, entry)
```

`query()`（`manager.py:98`）支持跨类型查询：如果不指定 `memory_type`，会同时查询三层记忆并合并结果。

`find_and_revoke_by_content()`（`manager.py:153`）用于**记忆去污染**：

```python
async def find_and_revoke_by_content(self, content_pattern: str, reason: str):
    """语义搜索匹配内容并批量撤销 -- 用于清除错误信息"""
    matches = await self.long_term.search(MemoryQuery(query=content_pattern, top_k=50))
    for entry in matches:
        await self.revoke(RevocationRequest(
            entry_id=entry.id, reason=reason, revoked_by="decontamination",
        ))
```

---

## 3. RAG 流水线

### 3.1 完整 RAG 架构

```
Document → Chunker → Embedder → VectorStore (索引)
Query → Embedder → VectorStore → Reranker → CitationVerifier (检索)
```

### 3.2 Chunker -- 文档分块

`Chunker`（`src/arcana/rag/chunker.py:11`）支持三种策略：

| 策略 | 枚举值 | 适用场景 |
|------|--------|---------|
| 固定窗口 | `FIXED` | 通用场景，简单可靠 |
| 段落分割 | `PARAGRAPH` | 结构化文档（Markdown、文章） |
| 递归分割 | `RECURSIVE` | 长文档，自适应粒度 |

递归策略（`chunker.py:107`）是最智能的，它按层次尝试：

```python
def _recursive_split(self, text, chunk_size, overlap):
    # Level 1: 尝试段落分割
    paragraphs = re.split(r"\n\n+", text)
    if len(paragraphs) > 1:
        return self._recursive_merge(text, paragraphs, chunk_size, overlap)

    # Level 2: 尝试句子分割
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        return self._recursive_merge(text, sentences, chunk_size, overlap)

    # Level 3: 回退到固定分割
    return self._fixed_split(text, chunk_size, overlap)
```

`_recursive_merge()`（`chunker.py:132`）负责将小段合并到不超过 `chunk_size` 的块中，超大段会被二次分割。

每个 Chunk 都有确定性 ID（`chunker.py:42`）：

```python
chunk_id = canonical_hash({"document_id": document.id, "start_offset": start})
```

这意味着同一文档重复索引不会产生重复块。

### 3.3 Embedder -- 向量化

`Embedder`（`src/arcana/rag/embedder.py:11`）定义了嵌入接口：

```python
class Embedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
```

`OpenAIEmbedder`（`embedder.py:28`）直接调用 OpenAI 兼容 API，不依赖 openai SDK：

```python
async def embed(self, texts: list[str]) -> list[list[float]]:
    url = f"{self.base_url}/v1/embeddings"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
    # 按 index 排序确保顺序正确
    embeddings_data = sorted(data["data"], key=lambda x: x["index"])
```

`MockEmbedder`（`embedder.py:71`）用于测试，基于 SHA-256 生成确定性向量：

```python
def _hash_to_vector(self, text: str) -> list[float]:
    while len(values) < self.dimensions:
        digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
        for byte_val in digest:
            values.append((byte_val / 127.5) - 1.0)  # 映射到 [-1, 1]
```

### 3.4 Retriever -- 检索核心

`Retriever`（`src/arcana/rag/retriever.py:22`）封装了完整的索引和检索流程：

**索引流程（ingest）**：

```python
async def ingest(self, document, config=None):
    # 1. 分块
    chunks = self._chunker.chunk(document, config)
    # 2. 批量嵌入
    texts = [c.content for c in chunks]
    embeddings = await self.embedder.embed(texts)
    # 3. 写入向量库
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        await self.vector_store.upsert(
            id=chunk.id, embedding=embedding,
            metadata={"document_id": chunk.document_id, ...},
            content=chunk.content,
        )
```

**检索流程（retrieve）**：

```python
async def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
    # 1. 嵌入查询
    query_embedding = (await self.embedder.embed([query.query]))[0]
    # 2. 向量搜索
    search_results = await self.vector_store.search(
        query_embedding=query_embedding,
        top_k=query.top_k, filters=query.filters,
    )
    # 3. 构建引用
    for sr in search_results:
        citation = Citation(
            source=sr.metadata.get("source", ""),
            chunk_id=sr.id,
            snippet=sr.content[:200],
            score=sr.score,
        )
        results.append(RetrievalResult(
            chunk_id=sr.id, score=sr.score,
            content=sr.content, citation=citation,
        ))
```

每次检索都自动生成 Citation 对象，为后续的引用验证做准备。

### 3.5 BM25Reranker -- 混合重排

`BM25Reranker`（`src/arcana/rag/reranker.py:36`）将向量相似度和文本相关度融合：

```python
final_score = alpha * vector_score + (1 - alpha) * bm25_norm
```

默认 `alpha=0.7`，即向量分数权重 70%，BM25 权重 30%。

BM25 的核心公式（`reranker.py:97`）：

```python
idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
numerator = tf * (self.k1 + 1)
denominator = tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
score += idf * numerator / denominator
```

**为什么需要混合排序？** 纯向量检索擅长语义匹配（"汽车"匹配"轿车"），但可能遗漏精确关键词。BM25 擅长精确匹配。两者互补可以显著提升检索质量。

### 3.6 CitationVerifier -- 引用验证

`CitationVerifier`（`src/arcana/rag/verifier.py:13`）检查 RAG 回答的引用可靠性：

```python
class CitationVerifier:
    def verify(self, answer: RAGAnswer, retrieved_results) -> VerificationResult:
        # 检查 1：有没有引用？
        if not answer.citations:
            return VerificationResult(valid=False, coverage=0.0)

        # 检查 2：引用的 chunk_id 是否存在于检索结果中？
        invalid_ids = cited_chunk_ids - valid_chunk_ids

        # 检查 3-4：覆盖率 -- 多少句子有引用支撑？
        for sentence in sentences:
            if self._sentence_is_supported(sentence, citation_snippets, cited_sources):
                supported_count += 1

        coverage = supported_count / len(sentences)

        # 检查 5：弱引用 -- 分数低于阈值的引用
        for citation in answer.citations:
            if citation.score < self.weak_threshold:
                weak.append(citation.chunk_id or citation.source)
```

支撑度判断基于词汇重叠（`verifier.py:108`）：

```python
def _sentence_is_supported(self, sentence, citation_snippets, cited_sources):
    content_overlap = {w for w in overlap if len(w) > 3}
    if len(content_overlap) >= 2:
        return True  # 至少 2 个内容词重叠
```

---

## 4. Storage 抽象层

### 4.1 为什么需要存储抽象？

Arcana 的 Memory、RAG、Trace 等多个模块都需要持久化，但各自的存储需求不同：

- Trace 需要追加写入和按 run_id 查询
- Memory 需要 KV 和向量搜索
- Orchestrator 需要 checkpoint

如果每个模块都绑定具体实现（PostgreSQL、Redis、Chroma），就会产生强耦合。`StorageBackend` 和 `VectorStore` 两个抽象接口解决了这个问题。

### 4.2 StorageBackend -- 结构化存储

`StorageBackend`（`src/arcana/storage/base.py:9`）统一了三类操作：

```python
class StorageBackend(ABC):
    # Trace Events
    async def store_trace_event(self, run_id, event) -> None: ...
    async def get_trace_events(self, run_id, *, event_type=None) -> list: ...

    # Checkpoints
    async def store_checkpoint(self, run_id, step_id, state) -> None: ...
    async def get_latest_checkpoint(self, run_id) -> dict | None: ...

    # Key-Value
    async def put(self, namespace, key, value) -> None: ...
    async def get(self, namespace, key) -> Any | None: ...
    async def delete(self, namespace, key) -> bool: ...
```

### 4.3 VectorStore -- 向量存储

`VectorStore`（`base.py:77`）专注于嵌入式检索：

```python
class VectorStore(ABC):
    async def upsert(self, id, embedding, metadata=None, content=None) -> None: ...
    async def search(self, query_embedding, *, top_k=10,
                     filters=None, min_score=0.0) -> list[VectorSearchResult]: ...
    async def delete(self, id) -> bool: ...
    async def count(self) -> int: ...
```

### 4.4 InMemory 实现

`InMemoryBackend`（`src/arcana/storage/memory.py:12`）和 `InMemoryVectorStore`（`memory.py:105`）用于测试和开发：

```python
class InMemoryVectorStore(VectorStore):
    async def search(self, query_embedding, *, top_k=10, filters=None, min_score=0.0):
        for vec_id, entry in candidates:
            if filters and not _matches_filters(entry.metadata, filters):
                continue
            score = _cosine_similarity(query_embedding, entry.embedding)
            if score >= min_score:
                results.append(VectorSearchResult(id=vec_id, score=score, ...))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
```

纯 Python 余弦相似度（`memory.py:209`）不依赖 numpy：

```python
def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

### 4.5 ChromaDB 实现

`ChromaVectorStore`（`src/arcana/storage/chroma.py:24`）是生产级向量存储：

```python
class ChromaVectorStore(VectorStore):
    def __init__(self, *, persist_directory=None, collection_name="arcana_vectors"):
        _require_chromadb()  # 懒加载检查

    async def initialize(self):
        if self._persist_directory:
            self._client = chromadb.PersistentClient(path=self._persist_directory)
        else:
            self._client = chromadb.Client()  # 临时模式

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦距离
        )
```

ChromaDB 返回距离而非相似度，需要转换（`chroma.py:132`）：

```python
score = 1.0 - distances[i]  # cosine space: similarity = 1 - distance
```

元数据扁平化（`chroma.py:162`）处理 ChromaDB 只支持标量值的限制：

```python
def _flatten_metadata(metadata):
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)  # 复杂类型转字符串
```

懒导入设计（`src/arcana/storage/__init__.py:15`）避免未安装 chromadb 时的导入错误：

```python
def get_chroma_store() -> type:
    from arcana.storage.chroma import ChromaVectorStore
    return ChromaVectorStore
```

---

## 5. Multi-Agent 协作

### 5.1 Planner-Executor-Critic 模式

Arcana 的多 Agent 协作基于经典的 **PEC（Planner-Executor-Critic）** 模式：

```
Goal → Planner → Plan → Executor → Result → Critic → Verdict
                                                        ↓
                                              Approved? → Done
                                              Rejected? → Feedback → Planner (下一轮)
```

### 5.2 TeamOrchestrator

`TeamOrchestrator`（`src/arcana/multi_agent/team.py:57`）协调整个循环：

```python
class TeamOrchestrator:
    async def run(self, goal: str) -> HandoffResult:
        session = CollaborationSession(goal=goal, max_rounds=self._max_rounds)

        for round_num in range(1, self._max_rounds + 1):
            # 预算守卫
            if self._is_budget_exhausted():
                return HandoffResult(final_status="budget_exhausted", ...)

            # 1. Planner 制定计划
            planner_state = await self._run_role(AgentRole.PLANNER, planner_goal, session_id)
            plan_content = planner_state.working_memory.get(WM_KEY_PLAN, planner_goal)
            await self._bus.publish(plan_msg)

            # 2. Executor 执行计划
            executor_state = await self._run_role(AgentRole.EXECUTOR, executor_goal, session_id)
            result_content = executor_state.working_memory.get(WM_KEY_RESULT, "...")
            await self._bus.publish(result_msg)

            # 3. Critic 验证结果
            critic_state = await self._run_role(AgentRole.CRITIC, critic_goal, session_id)
            verdict = self._extract_verdict(critic_state)

            if verdict:
                return HandoffResult(final_status="completed", ...)

            # 拒绝：构建反馈进入下一轮
            feedback = AgentMessage(
                sender_role=AgentRole.CRITIC,
                content={WM_KEY_FEEDBACK: feedback_content},
            )

        # 达到最大轮次：升级处理
        return HandoffResult(final_status="escalated", ...)
```

**Working Memory 键约定**（`team.py:31`）：

```python
WM_KEY_PLAN = "plan"       # Planner 输出
WM_KEY_RESULT = "result"   # Executor 输出
WM_KEY_FEEDBACK = "feedback"  # Critic 反馈
WM_KEY_VERDICT = "verdict"    # Critic 判定
```

**判定值**（`team.py:37`）：

```python
APPROVED_VERDICTS = frozenset({"pass", "true", "yes", "approved"})
```

### 5.3 RoleConfig -- 角色配置

每个角色可以有独立的策略、reducer 和步骤限制（`team.py:40`）：

```python
class RoleConfig:
    def __init__(self, *, role: AgentRole, policy: BasePolicy,
                 reducer: BaseReducer, max_steps: int = 50):
        ...
```

这允许为不同角色选择不同的 LLM 配置。例如：Planner 用高推理能力的大模型，Executor 用工具调用能力强的模型，Critic 用便宜的小模型做验证。

### 5.4 MessageBus -- 消息总线

`MessageBus`（`src/arcana/multi_agent/message_bus.py:14`）提供进程内异步消息传递：

```python
class MessageBus:
    def __init__(self):
        self._queues: dict[AgentRole, asyncio.Queue[AgentMessage]] = {}
        self._history: dict[str, list[AgentMessage]] = defaultdict(list)

    async def publish(self, message: AgentMessage) -> None:
        role = message.recipient_role
        if role not in self._queues:
            self._queues[role] = asyncio.Queue()
        await self._queues[role].put(message)
        self._history[message.session_id].append(message)  # 审计日志
```

`subscribe()`（`message_bus.py:39`）提供非阻塞消费，但默认的 TeamOrchestrator 并不使用它 -- 它通过直接传递状态来通信，MessageBus 主要用于**审计日志和自定义编排器**。

`history()`（`message_bus.py:66`）返回一个 session 的完整消息历史，这对调试多 Agent 交互非常有价值。

### 5.5 预算与错误控制

TeamOrchestrator 有两层安全网：

1. **全局预算**（`team.py:319`）：每轮开始前检查，耗尽则立即停止
2. **角色错误处理**（`team.py:329`）：任何角色失败都生成 `HandoffResult(final_status="error")`

```python
def _handle_role_error(self, role, exc, session_id, round_num, ...):
    logger.exception("Agent role %s failed in round %d", role.value, round_num)
    self._write_trace_event(session_id, EventType.TASK_FAIL,
                            {"role": role.value, "error": str(exc)})
    return HandoffResult(final_status="error", ...)
```

---

## 6. Orchestrator（编排器）

### 6.1 与 Multi-Agent 的区别

Multi-Agent（TeamOrchestrator）关注的是**角色间协作**（Planner/Executor/Critic 互相对话）。Orchestrator 关注的是**任务调度**（多个独立任务的 DAG 依赖、并发执行、重试）。

```
TeamOrchestrator: 角色驱动，一个 goal → 多轮 PEC 对话
Orchestrator:     任务驱动，多个 tasks → DAG 调度 → 并发执行
```

### 6.2 TaskGraph -- DAG 依赖管理

`TaskGraph`（`src/arcana/orchestrator/task_graph.py:22`）管理任务间的有向无环图：

```python
class TaskGraph:
    def add_task(self, task: Task) -> None:
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' already exists")
        # 环路检测
        self._tasks[task.id] = task
        cycle = self._detect_cycle(task.id)
        if cycle:
            del self._tasks[task.id]
            raise CycleError(task.id, cycle)
```

DFS 环路检测（`task_graph.py:152`）在添加任务时即刻运行，防止构建出不可执行的图。

`ready_tasks()`（`task_graph.py:63`）返回所有依赖已满足的 PENDING 任务，支持并行执行：

```python
def ready_tasks(self) -> list[Task]:
    completed_ids = {tid for tid, t in self._tasks.items()
                     if t.status == TaskStatus.COMPLETED}
    return [t for t in self._tasks.values()
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.dependencies)]
```

`is_stuck`（`task_graph.py:121`）检测死锁状态：没有任务在运行/排队，也没有任务可执行，但还有未完成的任务。

### 6.3 TaskScheduler -- 优先级调度

`TaskScheduler`（`src/arcana/orchestrator/scheduler.py:16`）基于多因素排序选择任务：

```python
def select_tasks(self, max_count: int) -> list[Task]:
    ready = self._graph.ready_tasks()
    # 预算准入控制
    admissible = [t for t in ready if self._can_admit(t)]
    # 排序：截止时间 ASC → 优先级 DESC → 创建时间 ASC
    admissible.sort(key=lambda t: (
        t.deadline if t.deadline is not None else _MAX_DATETIME,
        -t.priority,
        t.created_at,
    ))
    return admissible[:max_count]
```

**准入控制**（`scheduler.py:65`）：如果全局预算不足以承担某任务的 token 需求，该任务不会被调度。

### 6.4 ExecutorPool -- 并发执行池

`ExecutorPool`（`src/arcana/orchestrator/executor_pool.py:30`）使用 `asyncio.Semaphore` 控制并发度：

```python
class ExecutorPool:
    def __init__(self, agent_factory, *, max_concurrent=4):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running: dict[str, asyncio.Task[TaskResult]] = {}

    async def _execute(self, task: Task) -> TaskResult:
        async with self._semaphore:  # 信号量控制并发
            agent = self._factory.create_agent(task)
            state = await agent.run(goal=task.goal, task_id=task.id)
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED if state.status.value == "completed"
                       else TaskStatus.FAILED,
                tokens_used=state.tokens_used,
                cost_usd=state.cost_usd,
                ...
            )
```

`wait_any()`（`executor_pool.py:124`）使用 `asyncio.wait(FIRST_COMPLETED)` 等待任意一个任务完成，让调度循环可以及时处理结果并释放槽位。

### 6.5 Orchestrator -- 主协调器

`Orchestrator`（`src/arcana/orchestrator/orchestrator.py:24`）串联所有组件：

```python
class Orchestrator:
    async def run(self) -> dict[str, TaskResult]:
        while not self._graph.is_complete:
            if self._graph.is_stuck:
                break  # 死锁检测

            # 调度就绪任务
            available = self._pool.available_slots
            if available > 0:
                to_run = self._scheduler.select_tasks(available)
                for task in to_run:
                    self._graph.mark_task(task.id, TaskStatus.RUNNING)
                    self._pool.submit(task)

            # 等待完成
            if self._pool.running_count > 0:
                completed_results = await self._pool.wait_any()
                for result in completed_results:
                    await self._handle_result(result)
```

结果处理包含重试逻辑（`orchestrator.py:152`）：

```python
async def _handle_result(self, result: TaskResult):
    if result.status == TaskStatus.FAILED:
        if task.attempt < task.retry_policy.max_retries:
            # 指数退避后重新排队
            delay_ms = min(
                task.retry_policy.delay_ms * (backoff_multiplier ** (attempt - 1)),
                task.retry_policy.max_delay_ms,
            )
            await asyncio.sleep(delay_ms / 1000)
            self._graph.mark_task(task.id, TaskStatus.PENDING)  # 重新排队
```

### 6.6 OrchestratorHook -- 生命周期钩子

`OrchestratorHook`（`src/arcana/orchestrator/hooks.py:12`）使用 Protocol 定义了可选钩子：

```python
@runtime_checkable
class OrchestratorHook(Protocol):
    async def on_task_submitted(self, task: Task) -> None: ...
    async def on_task_started(self, task: Task) -> None: ...
    async def on_task_completed(self, task: Task, result: TaskResult) -> None: ...
    async def on_task_failed(self, task: Task, result: TaskResult) -> None: ...
    async def on_task_retrying(self, task: Task, attempt: int) -> None: ...
    async def on_orchestrator_complete(self) -> None: ...
```

所有钩子调用都被 try/except 包裹（`orchestrator.py:229`），保证钩子异常不会中断调度：

```python
async def _call_hooks(self, hook_name, *args):
    for hook in self._hooks:
        method = getattr(hook, hook_name, None)
        if method:
            try:
                await method(*args)
            except Exception:
                pass  # 钩子错误不影响主流程
```

---

## 7. Observability（可观测性）

### 7.1 MetricsCollector -- 指标采集

`MetricsCollector`（`src/arcana/observability/metrics.py:44`）从 Trace 事件中提取运行指标：

```python
class MetricsCollector:
    @staticmethod
    def summarize_run(events: list[TraceEvent]) -> RunSummary:
        for event in events:
            if event.event_type == EventType.LLM_CALL:
                llm_calls += 1
            elif event.event_type == EventType.TOOL_CALL:
                tool_calls += 1
            elif event.event_type == EventType.ERROR:
                errors += 1
            if event.budgets:
                tokens_used = max(tokens_used, event.budgets.tokens_used)
                cost_usd = max(cost_usd, event.budgets.cost_usd)
```

`RunSummary`（`metrics.py:17`）包含核心运行指标：

```python
class RunSummary(BaseModel):
    run_id: str
    total_steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    errors: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    stop_reason: str | None = None
```

`aggregate()`（`metrics.py:124`）跨多次运行计算百分位数：

```python
@staticmethod
def aggregate(summaries: list[RunSummary]) -> AggregateMetrics:
    return AggregateMetrics(
        count=n,
        avg_steps=sum(s.total_steps for s in summaries) / n,
        p50_tokens=_percentile(token_values, 50),
        p95_tokens=_percentile(token_values, 95),
        p99_tokens=_percentile(token_values, 99),
    )
```

P95/P99 百分位数对识别异常运行（token 爆炸、成本失控）特别有用。

### 7.2 MetricsHook -- 实时步骤级监控

`MetricsHook`（`src/arcana/observability/hooks.py:37`）实现了 `RuntimeHook` 协议，在运行时实时采集：

```python
class MetricsHook:
    async def on_step_complete(self, state, step_result, trace_ctx):
        tokens = 0
        if step_result.llm_response and step_result.llm_response.usage:
            tokens = step_result.llm_response.usage.total_tokens
            self.total_tokens += tokens

        metric = StepMetric(
            step_number=state.current_step,
            step_type=step_result.step_type.value,
            tokens_used=tokens,
            success=step_result.success,
        )
        self.step_metrics.append(metric)
```

`to_summary()`（`hooks.py:109`）将采集的原始数据转化为 `RunSummary`，可以在运行结束后立即获得指标而无需读取 Trace 文件。

### 7.3 应该监控什么？

| 指标 | 告警条件 | 说明 |
|------|---------|------|
| `tokens_used` | 超过 P95 2 倍 | Token 爆炸，可能陷入循环 |
| `cost_usd` | 超过单次预算 | 成本失控 |
| `errors` | > 0 | 每次错误都值得关注 |
| `duration_ms` | 超过 SLA | 超时风险 |
| `stop_reason` | 非 `goal_reached` | 异常终止 |
| 工具调用失败率 | > 10% | 下游服务可能故障 |

---

## 8. Eval Harness（评测框架）

### 8.1 为什么 Agent 需要评测？

传统软件通过单元测试验证正确性。但 Agent 的行为是非确定性的（LLM 输出随机），需要一套**统计性质量保证**框架：

- 定义"正确"的标准（期望结果）
- 批量运行测试用例
- 统计通过率
- 检测版本间的质量退化

### 8.2 EvalRunner -- 测试执行器

`EvalRunner`（`src/arcana/eval/runner.py:23`）逐个执行评测用例：

```python
class EvalRunner:
    def __init__(self, agent_factory: Callable[[], Agent]):
        self._agent_factory = agent_factory  # 每个用例创建全新 Agent

    async def run_case(self, case: EvalCase) -> EvalResult:
        agent = self._agent_factory()  # 隔离环境
        state = await agent.run(case.goal)

        # 从 Trace 提取指标
        events = reader.read_events(state.run_id)
        summary = MetricsCollector.summarize_run(events)

        # 检查预期结果
        passed = self._check_outcome(case, state, events)

        return EvalResult(
            case_id=case.id, passed=passed,
            tokens_used=state.tokens_used,
            cost_usd=state.cost_usd,
        )
```

**每个用例使用新 Agent 实例**，这保证了用例间的隔离性。

### 8.3 OutcomeCriterion -- 多维判定

`_check_outcome()`（`runner.py:128`）支持六种判定标准：

| 标准 | 说明 | 示例 |
|------|------|------|
| `STATUS` | 最终状态 | `"completed"` |
| `STOP_REASON` | 停止原因 | `"goal_reached"` |
| `MAX_STEPS` | 步骤不超过 N | `10` |
| `MAX_COST` | 成本不超过 $ | `0.05` |
| `CONTAINS_KEYS` | working_memory 包含指定键 | `["result", "plan"]` |
| `KEY_VALUES` | working_memory 键值匹配 | `{"answer": "42"}` |

```python
if criterion == OutcomeCriterion.KEY_VALUES:
    return all(state.working_memory.get(k) == v
               for k, v in expected.items())
```

### 8.4 RegressionGate -- 质量守门

`RegressionGate`（`src/arcana/eval/gate.py:13`）提供两种检查模式：

**绝对检查**（`gate.py:25`）：

```python
def check(self, report: EvalReport) -> RegressionResult:
    violations = []
    if report.pass_rate < self._config.min_pass_rate:
        violations.append(f"pass_rate {report.pass_rate:.2%} < ...")
    if avg_cost > self._config.max_avg_cost_usd:
        violations.append(f"avg_cost ${avg_cost:.4f} > ...")
    if avg_tokens > self._config.max_avg_tokens:
        violations.append(f"avg_tokens {avg_tokens:.0f} > ...")
```

**相对检查（回归检测）**（`gate.py:62`）：

```python
def compare(self, current: EvalReport, baseline: EvalReport) -> RegressionResult:
    regression_pct = (baseline.pass_rate - current.pass_rate) / baseline.pass_rate
    if regression_pct > self._config.max_regression_pct:
        violations.append(f"regression {regression_pct:.2%} > ...")
```

在 CI/CD 流水线中，可以这样使用：

```python
gate = RegressionGate(GateConfig(min_pass_rate=0.95, max_regression_pct=0.05))

# 绝对检查：通过率必须 >= 95%
result = gate.check(current_report)

# 回归检查：相比 baseline 退化不超过 5%
result = gate.compare(current_report, baseline_report)

if not result.passed:
    print(f"Quality gate failed: {result.gate_violations}")
    sys.exit(1)
```

---

## 9. 生产部署全景

将 Arcana 部署到生产环境时，每个模块都有需要特别关注的点：

### 9.1 Tool Gateway

| 项目 | 开发环境 | 生产环境 |
|------|---------|---------|
| 幂等性缓存 | 内存 dict | Redis + TTL |
| 确认回调 | CLI 输入 | Slack/Teams 审批 |
| 并发控制 | 无限制 | 限速 + 熔断器 |
| 审计日志 | JSONL 文件 | 持久化 + 合规保留 |

### 9.2 Memory

| 项目 | 开发环境 | 生产环境 |
|------|---------|---------|
| Working Memory | InMemoryBackend | Redis（带 TTL 自动过期） |
| Long-Term Memory | InMemoryVectorStore | pgvector / Pinecone |
| 治理策略 | 宽松阈值 | 严格置信度门控 |
| 去污染 | 手动 | 定期扫描 + 自动撤销 |

### 9.3 RAG

| 项目 | 开发环境 | 生产环境 |
|------|---------|---------|
| Embedder | MockEmbedder | OpenAIEmbedder + 缓存层 |
| VectorStore | InMemoryVectorStore | ChromaDB / pgvector |
| 分块策略 | 固定大小 | 递归策略 + 语义分块 |
| Reranker | 可选 | 必须启用，提升精度 |
| 引用验证 | 可选 | 必须启用，防止幻觉 |

### 9.4 Storage

| 项目 | 开发环境 | 生产环境 |
|------|---------|---------|
| StorageBackend | InMemoryBackend | PostgreSQL |
| VectorStore | InMemoryVectorStore | ChromaDB（持久模式）/ pgvector |
| 备份 | 无 | 定期快照 + 增量备份 |
| 连接池 | N/A | 必须配置 |

### 9.5 Multi-Agent

| 项目 | 开发环境 | 生产环境 |
|------|---------|---------|
| MessageBus | 进程内 Queue | Redis Streams / Kafka |
| 预算控制 | 宽松 | 每角色独立预算 |
| 超时 | 无 | 每角色设置超时 |
| 最大轮次 | 大（调试用） | 小（3-5 轮控制成本） |

### 9.6 Orchestrator

| 项目 | 开发环境 | 生产环境 |
|------|---------|---------|
| 并发度 | 低（2-4） | 根据资源调整 |
| 任务持久化 | 可选 | 必须（故障恢复） |
| 死锁检测 | `is_stuck` 属性 | + 告警通知 |
| 重试策略 | 默认 | 根据任务类型定制 |

### 9.7 Observability

| 项目 | 开发环境 | 生产环境 |
|------|---------|---------|
| 指标输出 | 控制台 / RunSummary | Prometheus / Datadog |
| 告警 | 无 | P95 token 偏差 > 2x、成本超限、错误率 > 阈值 |
| 仪表盘 | 无 | Grafana 实时面板 |
| 日志级别 | DEBUG | WARNING |

### 9.8 Eval

| 项目 | 开发环境 | 生产环境 |
|------|---------|---------|
| 运行频率 | 手动 | CI/CD 每次提交 |
| 用例数量 | 少量 | 覆盖核心场景的完整套件 |
| 质量门 | 宽松 | min_pass_rate >= 0.95, max_regression_pct <= 0.05 |
| 基线管理 | 手动快照 | 自动存储最佳报告作为基线 |

---

## 10. 本章小结

本章覆盖了 Arcana 平台的八个服务模块。让我们回顾核心设计思想：

### 关键架构模式

1. **流水线模式（Tool Gateway）**：将横切关注点（授权、校验、重试、审计）串联为标准化管道，每个工具只需实现 `execute()`。

2. **三层记忆（Memory）**：Working（临时）、Long-Term（持久语义）、Episodic（轨迹回放），通过 MemoryManager 统一门面，WritePolicy 守门。

3. **索引-检索-验证（RAG）**：从分块到嵌入到检索到重排到引用验证，每一步都是可替换组件。BM25 与向量的混合排序是工程实践的最佳选择。

4. **抽象层（Storage）**：两个 ABC（StorageBackend + VectorStore）解耦了上层业务和底层实现，使得从内存到 PostgreSQL/ChromaDB 的切换只需改一行配置。

5. **PEC 协作（Multi-Agent）**：Planner-Executor-Critic 循环提供了自我验证和迭代改进能力。MessageBus 保留完整消息历史用于审计。

6. **DAG 调度（Orchestrator）**：TaskGraph 管理依赖，TaskScheduler 处理优先级和准入，ExecutorPool 控制并发。三者正交，各司其职。

7. **Trace 驱动的可观测性**：所有模块的行为都通过 TraceEvent 记录，MetricsCollector 从中提取指标，MetricsHook 提供实时监控。

8. **统计质量保证（Eval）**：Agent 的非确定性行为需要统计性验证。EvalRunner + RegressionGate 构成了 CI/CD 质量守门的完整闭环。

### 各模块间的协作关系

```
Tool Gateway ←→ Trace（审计）
Memory ←→ Storage（持久化）、RAG（长期记忆的向量检索）、Trace（情景记忆）
RAG ←→ Storage（向量存储）、Trace（检索日志）
Multi-Agent ←→ Runtime（Agent 执行）、Trace（协作日志）、Budget（预算控制）
Orchestrator ←→ Runtime（Agent 执行）、Storage（任务持久化）、Trace（调度日志）
Observability ←→ Trace（指标提取）、Runtime（实时钩子）
Eval ←→ Runtime（Agent 执行）、Trace（结果分析）、Observability（指标采集）
```

至此，Arcana 框架的所有核心模块已经完整覆盖。从最底层的 Contracts 和 Trace，到 Model Gateway 和 Runtime Engine，再到本章的平台服务层，构成了一个可审计、可观测、可评测的生产级 Agent 框架。
