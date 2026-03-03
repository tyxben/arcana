"""RAG-related contracts for retrieval-augmented generation."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    """Strategy for splitting documents into chunks."""

    FIXED = "fixed"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"


class Document(BaseModel):
    """A source document for RAG."""

    id: str
    source: str  # Origin (URL, file path, etc.)
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_type: str = "text/plain"


class Chunk(BaseModel):
    """A chunk of a document after splitting."""

    id: str
    document_id: str
    content: str
    start_offset: int = 0
    end_offset: int = 0

    # Embedding info
    embedding: list[float] | None = None
    embedding_model: str | None = None

    # Inherited metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Citation reference for a claim."""

    source: str
    chunk_id: str | None = None
    document_id: str | None = None
    snippet: str = ""  # Relevant text excerpt
    page: int | None = None
    section: str | None = None
    url: str | None = None
    score: float = 0.0
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RetrievalQuery(BaseModel):
    """Query for retrieval."""

    query: str
    top_k: int = 10
    filters: dict[str, Any] | None = None
    rerank: bool = True
    min_score: float = 0.0


class RetrievalResult(BaseModel):
    """Single retrieval result."""

    chunk_id: str
    document_id: str
    score: float
    content: str
    citation: Citation


class RetrievalResponse(BaseModel):
    """Full retrieval response with results and metadata."""

    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
    total_candidates: int = 0  # Before reranking
    query_embedding_model: str | None = None


class RerankConfig(BaseModel):
    """Configuration for reranking."""

    method: str = "bm25"  # "bm25", "cross_encoder", "cohere", "jina"
    top_n: int = 5
    min_score: float = 0.0
    model: str | None = None  # For model-based rerankers


class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    embedding_model: str = "text-embedding-ada-002"


class RAGAnswer(BaseModel):
    """Answer with citations from RAG pipeline."""

    content: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    retrieved_chunks: list[str] = Field(default_factory=list)  # Chunk IDs used

    # Trace info
    query_digest: str | None = None
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Result of citation verification."""

    valid: bool
    coverage: float = 0.0  # % of claims with citations
    unsupported_claims: list[str] = Field(default_factory=list)
    weak_citations: list[str] = Field(default_factory=list)
