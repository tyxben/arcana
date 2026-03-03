"""Abstract base classes for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """
    Abstract storage backend for structured data.

    Supports trace events, checkpoints, and general key-value storage.
    Implementations: InMemoryBackend (tests), PostgresBackend (production).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend (create tables, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections and clean up."""
        ...

    # ── Trace Events ─────────────────────────────────────────

    @abstractmethod
    async def store_trace_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Store a trace event."""
        ...

    @abstractmethod
    async def get_trace_events(
        self,
        run_id: str,
        *,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get trace events for a run."""
        ...

    # ── Checkpoints ──────────────────────────────────────────

    @abstractmethod
    async def store_checkpoint(
        self, run_id: str, step_id: str, state: dict[str, Any]
    ) -> None:
        """Store a checkpoint."""
        ...

    @abstractmethod
    async def get_latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """Get the latest checkpoint for a run."""
        ...

    # ── Key-Value ────────────────────────────────────────────

    @abstractmethod
    async def put(self, namespace: str, key: str, value: Any) -> None:
        """Store a key-value pair."""
        ...

    @abstractmethod
    async def get(self, namespace: str, key: str) -> Any | None:
        """Get a value by key."""
        ...

    @abstractmethod
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete a key-value pair. Returns True if existed."""
        ...


class VectorStore(ABC):
    """
    Abstract vector store for embedding-based retrieval.

    Used by the RAG system for document indexing and similarity search.
    Implementations: InMemoryVectorStore (tests), PgVectorStore (production).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections."""
        ...

    @abstractmethod
    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> None:
        """Insert or update a vector with metadata."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Get total number of stored vectors."""
        ...


class VectorSearchResult:
    """Result from a vector similarity search."""

    __slots__ = ("id", "score", "metadata", "content")

    def __init__(
        self,
        id: str,
        score: float,
        metadata: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> None:
        self.id = id
        self.score = score
        self.metadata = metadata
        self.content = content

    def __repr__(self) -> str:
        return f"VectorSearchResult(id={self.id!r}, score={self.score:.4f})"
