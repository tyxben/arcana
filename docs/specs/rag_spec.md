# RAG Specification

> Version: 1.0.0
> Status: Draft (Week 6 Implementation)

## Overview

The RAG (Retrieval-Augmented Generation) system provides knowledge retrieval and citation capabilities. Every answer must include traceable citations to source documents.

## Design Principles

1. **Citation-first**: Every claim must be traceable
2. **Evidence-based**: Answers derived from retrieved context
3. **Transparent**: Retrieval process is fully logged
4. **Evaluable**: Metrics for retrieval quality

## Document Schema

### Document

```python
class Document:
    id: str                   # Unique document identifier
    source: str               # Origin (URL, file path, etc.)
    timestamp: datetime       # Document date/import time
    tags: list[str]           # Classification tags
    content: str              # Full document content
    metadata: dict            # Additional metadata

    # Chunking
    chunk_id: str | None      # If this is a chunk
    parent_id: str | None     # Parent document ID
```

### Chunk

```python
class Chunk:
    id: str                   # Unique chunk identifier
    document_id: str          # Parent document
    content: str              # Chunk text
    start_offset: int         # Start position in document
    end_offset: int           # End position in document

    # Embeddings
    embedding: list[float] | None
    embedding_model: str | None
```

## Retrieval Schema

### RetrievalQuery

```python
class RetrievalQuery:
    query: str                # Search query
    top_k: int                # Number of results
    filters: dict | None      # Metadata filters
    rerank: bool              # Apply reranking
```

### RetrievalResult

```python
class RetrievalResult:
    doc_id: str               # Document ID
    chunk_id: str | None      # Chunk ID if applicable
    score: float              # Relevance score (0-1)
    snippet: str              # Matched text
    citation: Citation        # Citation info
```

### Citation

```python
class Citation:
    source: str               # Source reference
    page: int | None          # Page number
    section: str | None       # Section name
    url: str | None           # Source URL
    retrieved_at: datetime    # Retrieval timestamp
```

## Answer Schema

### RAGAnswer

```python
class RAGAnswer:
    content: str              # Answer text
    citations: list[Citation] # Supporting citations
    confidence: float         # Confidence score
    retrieved_chunks: list[str]  # Chunk IDs used

    # Trace info
    query_digest: str         # Hash of query
    retrieval_trace: dict     # Full retrieval metadata
```

## RAG Pipeline

```
┌─────────────┐     ┌───────────┐     ┌──────────────┐
│    Query    │────▶│  Rewrite  │────▶│   Retrieve   │
└─────────────┘     └───────────┘     └──────┬───────┘
                                             │
                                      ┌──────▼───────┐
                                      │   Rerank     │
                                      └──────┬───────┘
                                             │
┌─────────────┐     ┌───────────┐     ┌──────▼───────┐
│   Answer    │◀────│  Generate │◀────│   Context    │
└─────────────┘     └───────────┘     └──────────────┘
```

### Pipeline Steps

1. **Query Rewrite**: Optimize query for retrieval
2. **Retrieve**: Vector search + keyword search
3. **Rerank**: Score and filter results
4. **Context Assembly**: Build prompt context
5. **Generate**: LLM generates answer with citations
6. **Verify**: Check citations are valid

## Ingestion Contract

### IngestionRequest

```python
class IngestionRequest:
    source: str               # Document source
    content: str              # Raw content
    content_type: str         # MIME type
    metadata: dict            # Custom metadata

    # Chunking config
    chunk_size: int           # Target chunk size
    chunk_overlap: int        # Overlap between chunks
```

### ChunkingStrategy

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `fixed` | Fixed token count | General text |
| `semantic` | Sentence boundaries | Articles |
| `paragraph` | Paragraph breaks | Structured docs |
| `recursive` | Hierarchical | Code, long docs |

## Reranking

### Reranker Options

| Reranker | Description | Performance |
|----------|-------------|-------------|
| `bge-reranker-base` | Local model (~400MB) | Fast, good quality |
| `cohere-rerank` | Cohere API | High quality |
| `jina-rerank` | Jina API | Good quality |
| `bm25` | BM25 scoring | Fast, no model |

### Rerank Config

```python
class RerankConfig:
    model: str                # Reranker to use
    top_n: int                # Results after rerank
    min_score: float          # Minimum score threshold
```

## Citation Verification

Every answer must pass citation verification:

```python
class CitationVerifier:
    def verify(self, answer: RAGAnswer) -> VerificationResult:
        # 1. Check all claims have citations
        # 2. Verify citations exist in retrieved chunks
        # 3. Check citation relevance
        # 4. Flag unsupported claims
```

### Verification Result

```python
class VerificationResult:
    valid: bool
    coverage: float           # % of claims with citations
    unsupported_claims: list[str]
    weak_citations: list[str] # Low relevance
```

## Trace Integration

RAG operations generate trace events:

```python
# Retrieval event
TraceEvent(
    event_type=EventType.TOOL_CALL,
    tool_call=ToolCallRecord(
        name="rag_retrieve",
        args_digest=canonical_hash(query),
        result_digest=canonical_hash(results),
    ),
    metadata={
        "query": query.query,
        "top_k": query.top_k,
        "results_count": len(results),
        "scores": [r.score for r in results],
    }
)
```

## Metrics

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| `recall@k` | Relevant docs in top-k |
| `precision@k` | Precision in top-k |
| `mrr` | Mean reciprocal rank |
| `ndcg` | Normalized DCG |

### Answer Metrics

| Metric | Description |
|--------|-------------|
| `citation_coverage` | Claims with citations |
| `evidence_relevance` | Citation-claim relevance |
| `hallucination_rate` | Unsupported claims |
| `answer_faithfulness` | Alignment with sources |

## Example Usage

```python
# Query
results = await rag.retrieve(
    query="What is the capital of France?",
    top_k=5,
    rerank=True,
)

# Generate answer
answer = await rag.generate(
    query="What is the capital of France?",
    context=results,
    require_citations=True,
)

# Verify
verification = verifier.verify(answer)
assert verification.valid
```

## Best Practices

1. **Always require citations**: No unsupported claims
2. **Log retrieval details**: For debugging and evaluation
3. **Set score thresholds**: Filter low-relevance results
4. **Implement fallbacks**: Handle empty retrieval
5. **Version embeddings**: Track embedding model changes
