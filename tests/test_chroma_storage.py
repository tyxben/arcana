"""Tests for the ChromaDB vector store backend."""

from __future__ import annotations

import uuid

import pytest

from arcana.storage.chroma import ChromaVectorStore, _flatten_metadata

# ── ChromaVectorStore Tests ─────────────────────────────────────────


class TestChromaVectorStore:
    """Tests for ChromaDB-backed vector store."""

    @pytest.fixture
    async def store(self) -> ChromaVectorStore:
        """Create an ephemeral ChromaDB store with unique collection per test."""
        name = f"test_{uuid.uuid4().hex[:8]}"
        s = ChromaVectorStore(collection_name=name)
        await s.initialize()
        return s

    async def test_upsert_and_count(self, store: ChromaVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0, 0.0])
        await store.upsert("v2", [0.0, 1.0, 0.0])
        assert await store.count() == 2

    async def test_upsert_overwrites(self, store: ChromaVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0, 0.0], content="old")
        await store.upsert("v1", [0.0, 1.0, 0.0], content="new")
        assert await store.count() == 1

    async def test_search_returns_similar(self, store: ChromaVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0, 0.0], content="doc-a")
        await store.upsert("v2", [0.0, 1.0, 0.0], content="doc-b")
        await store.upsert("v3", [0.9, 0.1, 0.0], content="doc-c")

        results = await store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # Most similar should be v1 (exact match)
        assert results[0].id == "v1"
        assert results[0].score > 0.9

    async def test_search_min_score_filter(self, store: ChromaVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0, 0.0])
        await store.upsert("v2", [0.0, 1.0, 0.0])  # orthogonal, score ~0

        results = await store.search([1.0, 0.0, 0.0], top_k=10, min_score=0.5)
        # Only v1 should pass the min_score filter
        assert all(r.score >= 0.5 for r in results)

    async def test_search_with_metadata_filter(self, store: ChromaVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0, 0.0], metadata={"type": "a"})
        await store.upsert("v2", [0.9, 0.1, 0.0], metadata={"type": "b"})

        results = await store.search(
            [1.0, 0.0, 0.0], top_k=10, filters={"type": "b"}
        )
        assert len(results) == 1
        assert results[0].id == "v2"

    async def test_search_returns_content(self, store: ChromaVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0, 0.0], content="hello world")
        results = await store.search([1.0, 0.0, 0.0], top_k=1)
        assert results[0].content == "hello world"

    async def test_delete_existing(self, store: ChromaVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0, 0.0])
        assert await store.delete("v1") is True
        assert await store.count() == 0

    async def test_delete_nonexistent(self, store: ChromaVectorStore) -> None:
        assert await store.delete("nope") is False

    async def test_count_empty(self, store: ChromaVectorStore) -> None:
        assert await store.count() == 0

    async def test_not_initialized_raises(self) -> None:
        store = ChromaVectorStore()
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.upsert("v1", [1.0])

    async def test_close_and_reinitialize(self, store: ChromaVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0, 0.0])
        await store.close()
        with pytest.raises(RuntimeError):
            await store.count()


# ── Metadata Flattening Tests ───────────────────────────────────────


class TestFlattenMetadata:
    def test_flat_values_unchanged(self) -> None:
        meta = {"key": "value", "num": 42, "flag": True, "score": 3.14}
        assert _flatten_metadata(meta) == meta

    def test_complex_values_stringified(self) -> None:
        meta = {"tags": ["a", "b"], "nested": {"x": 1}}
        flat = _flatten_metadata(meta)
        assert flat["tags"] == "['a', 'b']"
        assert flat["nested"] == "{'x': 1}"

    def test_empty_dict(self) -> None:
        assert _flatten_metadata({}) == {}
