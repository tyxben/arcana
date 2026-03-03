"""In-memory storage backends for testing and development."""

from __future__ import annotations

import math
import threading
from typing import Any

from arcana.storage.base import StorageBackend, VectorSearchResult, VectorStore


class InMemoryBackend(StorageBackend):
    """
    Dict-based in-memory storage backend.

    Suitable for tests and local development. Not persistent across restarts.
    All shared state is protected by a threading lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # run_id -> list of events
        self._traces: dict[str, list[dict[str, Any]]] = {}
        # run_id -> list of (step_id, state) tuples (append-only for ordering)
        self._checkpoints: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        # namespace -> key -> value
        self._kv: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """No-op for in-memory backend."""

    async def close(self) -> None:
        """Clear all data."""
        with self._lock:
            self._traces.clear()
            self._checkpoints.clear()
            self._kv.clear()

    # ── Trace Events ─────────────────────────────────────────

    async def store_trace_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Store a trace event for a given run."""
        with self._lock:
            self._traces.setdefault(run_id, []).append(event)

    async def get_trace_events(
        self,
        run_id: str,
        *,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get trace events for a run, optionally filtered by type."""
        with self._lock:
            events = list(self._traces.get(run_id, []))

        if event_type is not None:
            events = [e for e in events if e.get("event_type") == event_type]

        if limit is not None:
            events = events[:limit]

        return events

    # ── Checkpoints ──────────────────────────────────────────

    async def store_checkpoint(
        self, run_id: str, step_id: str, state: dict[str, Any]
    ) -> None:
        """Store a checkpoint for a run."""
        with self._lock:
            self._checkpoints.setdefault(run_id, []).append((step_id, state))

    async def get_latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """Get the most recently stored checkpoint for a run."""
        with self._lock:
            checkpoints = self._checkpoints.get(run_id, [])
            if not checkpoints:
                return None
            step_id, state = checkpoints[-1]
            return {"step_id": step_id, **state}

    # ── Key-Value ────────────────────────────────────────────

    async def put(self, namespace: str, key: str, value: Any) -> None:
        """Store a key-value pair in a namespace."""
        with self._lock:
            self._kv.setdefault(namespace, {})[key] = value

    async def get(self, namespace: str, key: str) -> Any | None:
        """Get a value by namespace and key. Returns None if not found."""
        with self._lock:
            return self._kv.get(namespace, {}).get(key)

    async def delete(self, namespace: str, key: str) -> bool:
        """Delete a key-value pair. Returns True if it existed."""
        with self._lock:
            ns = self._kv.get(namespace, {})
            if key in ns:
                del ns[key]
                return True
            return False


class InMemoryVectorStore(VectorStore):
    """
    List-based in-memory vector store with pure-Python cosine similarity.

    Suitable for tests and local development with small datasets.
    Uses math.sqrt and sum() -- no numpy dependency.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # id -> {embedding, metadata, content}
        self._vectors: dict[str, _VectorEntry] = {}

    async def initialize(self) -> None:
        """No-op for in-memory vector store."""

    async def close(self) -> None:
        """Clear all vectors."""
        with self._lock:
            self._vectors.clear()

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> None:
        """Insert or update a vector with optional metadata and content."""
        with self._lock:
            self._vectors[id] = _VectorEntry(
                embedding=embedding,
                metadata=metadata or {},
                content=content,
            )

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using cosine similarity."""
        with self._lock:
            candidates = list(self._vectors.items())

        results: list[VectorSearchResult] = []
        for vec_id, entry in candidates:
            # Apply metadata filters
            if filters is not None:
                if not _matches_filters(entry.metadata, filters):
                    continue

            score = _cosine_similarity(query_embedding, entry.embedding)

            if score >= min_score:
                results.append(
                    VectorSearchResult(
                        id=vec_id,
                        score=score,
                        metadata=entry.metadata,
                        content=entry.content,
                    )
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def delete(self, id: str) -> bool:
        """Delete a vector by ID. Returns True if it existed."""
        with self._lock:
            if id in self._vectors:
                del self._vectors[id]
                return True
            return False

    async def count(self) -> int:
        """Get the total number of stored vectors."""
        with self._lock:
            return len(self._vectors)


# ── Internal Helpers ─────────────────────────────────────────────


class _VectorEntry:
    """Internal storage record for a single vector."""

    __slots__ = ("embedding", "metadata", "content")

    def __init__(
        self,
        embedding: list[float],
        metadata: dict[str, Any],
        content: str | None,
    ) -> None:
        self.embedding = embedding
        self.metadata = metadata
        self.content = content


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero magnitude.
    Formula: dot(a, b) / (norm(a) * norm(b))
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Check if metadata matches all filter key-value pairs."""
    return all(metadata.get(k) == v for k, v in filters.items())
