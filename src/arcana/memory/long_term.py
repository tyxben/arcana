"""Long-term memory store — persistent, vector-indexed facts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from arcana.contracts.memory import MemoryEntry, MemoryQuery
from arcana.rag.embedder import Embedder
from arcana.storage.base import StorageBackend, VectorStore

_LTM_NAMESPACE = "ltm"


class LongTermMemoryStore:
    """
    Persistent memory backed by VectorStore (semantic search) and
    StorageBackend KV (full entry storage).

    Vector metadata stores filtering fields (revoked, confidence, etc.).
    The complete MemoryEntry is stored in KV under namespace "ltm".
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        backend: StorageBackend,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.backend = backend

    async def store(self, entry: MemoryEntry) -> None:
        """Store a long-term memory entry with vector embedding."""
        # Embed content
        embeddings = await self.embedder.embed([entry.content])
        embedding = embeddings[0]

        # Upsert vector with filterable metadata
        metadata = self._build_metadata(entry)
        await self.vector_store.upsert(
            id=entry.id,
            embedding=embedding,
            metadata=metadata,
            content=entry.content,
        )

        # Store full entry in KV
        await self.backend.put(_LTM_NAMESPACE, entry.id, entry.model_dump(mode="json"))

    async def search(self, query: MemoryQuery) -> list[MemoryEntry]:
        """Search long-term memory semantically."""
        if not query.query:
            return []

        # Embed query
        query_embeddings = await self.embedder.embed([query.query])
        query_embedding = query_embeddings[0]

        # Build filters
        filters: dict[str, Any] = {}
        if not query.include_revoked:
            filters["revoked"] = False

        # Vector search
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=query.top_k,
            filters=filters,
            min_score=-1.0,  # Accept all; filter by confidence later
        )

        # Hydrate full entries from KV
        entries: list[MemoryEntry] = []
        for result in results:
            data = await self.backend.get(_LTM_NAMESPACE, result.id)
            if data is not None:
                entry = MemoryEntry.model_validate(data)
                if query.include_revoked or not entry.revoked:
                    if entry.confidence >= query.min_confidence:
                        entries.append(entry)

        return entries

    async def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific entry by ID."""
        data = await self.backend.get(_LTM_NAMESPACE, entry_id)
        if data is None:
            return None
        return MemoryEntry.model_validate(data)

    async def revoke(self, entry_id: str, reason: str) -> bool:
        """Revoke a long-term memory entry."""
        entry = await self.get_by_id(entry_id)
        if entry is None or entry.revoked:
            return False

        entry.revoked = True
        entry.revoked_at = datetime.now(UTC)
        entry.revoked_reason = reason
        entry.updated_at = datetime.now(UTC)

        # Update KV
        await self.backend.put(
            _LTM_NAMESPACE, entry.id, entry.model_dump(mode="json")
        )

        # Update vector metadata (re-embed same content to update metadata)
        embeddings = await self.embedder.embed([entry.content])
        metadata = self._build_metadata(entry)
        await self.vector_store.upsert(
            id=entry.id,
            embedding=embeddings[0],
            metadata=metadata,
            content=entry.content,
        )
        return True

    @staticmethod
    def _build_metadata(entry: MemoryEntry) -> dict[str, Any]:
        """Build vector metadata from a MemoryEntry."""
        return {
            "memory_type": entry.memory_type.value,
            "key": entry.key,
            "confidence": entry.confidence,
            "source": entry.source,
            "revoked": entry.revoked,
            "tags": entry.tags,
            **entry.metadata,
        }
