"""Unit tests for the in-memory storage backends."""

from __future__ import annotations

import math

import pytest

from arcana.storage.memory import InMemoryBackend, InMemoryVectorStore, _cosine_similarity


# ═══════════════════════════════════════════════════════════════════
# InMemoryBackend Tests
# ═══════════════════════════════════════════════════════════════════


class TestInMemoryBackendTraceEvents:
    """Tests for trace event storage and retrieval."""

    @pytest.fixture
    async def backend(self) -> InMemoryBackend:
        backend = InMemoryBackend()
        await backend.initialize()
        return backend

    async def test_store_and_retrieve_single_event(self, backend: InMemoryBackend) -> None:
        event = {"event_type": "llm_call", "model": "gpt-4", "tokens": 100}
        await backend.store_trace_event("run-1", event)

        events = await backend.get_trace_events("run-1")
        assert len(events) == 1
        assert events[0] == event

    async def test_store_multiple_events_preserves_order(self, backend: InMemoryBackend) -> None:
        for i in range(5):
            await backend.store_trace_event("run-1", {"event_type": "llm_call", "step": i})

        events = await backend.get_trace_events("run-1")
        assert len(events) == 5
        assert [e["step"] for e in events] == [0, 1, 2, 3, 4]

    async def test_filter_by_event_type(self, backend: InMemoryBackend) -> None:
        await backend.store_trace_event("run-1", {"event_type": "llm_call", "data": "a"})
        await backend.store_trace_event("run-1", {"event_type": "tool_call", "data": "b"})
        await backend.store_trace_event("run-1", {"event_type": "llm_call", "data": "c"})

        llm_events = await backend.get_trace_events("run-1", event_type="llm_call")
        assert len(llm_events) == 2
        assert all(e["event_type"] == "llm_call" for e in llm_events)

    async def test_limit_events(self, backend: InMemoryBackend) -> None:
        for i in range(10):
            await backend.store_trace_event("run-1", {"event_type": "llm_call", "step": i})

        events = await backend.get_trace_events("run-1", limit=3)
        assert len(events) == 3
        assert events[0]["step"] == 0

    async def test_filter_and_limit_combined(self, backend: InMemoryBackend) -> None:
        for i in range(5):
            await backend.store_trace_event("run-1", {"event_type": "llm_call", "step": i})
            await backend.store_trace_event("run-1", {"event_type": "tool_call", "step": i})

        events = await backend.get_trace_events("run-1", event_type="tool_call", limit=2)
        assert len(events) == 2
        assert all(e["event_type"] == "tool_call" for e in events)

    async def test_get_events_empty_run(self, backend: InMemoryBackend) -> None:
        events = await backend.get_trace_events("nonexistent-run")
        assert events == []

    async def test_events_isolated_by_run_id(self, backend: InMemoryBackend) -> None:
        await backend.store_trace_event("run-1", {"event_type": "llm_call", "run": 1})
        await backend.store_trace_event("run-2", {"event_type": "llm_call", "run": 2})

        events_1 = await backend.get_trace_events("run-1")
        events_2 = await backend.get_trace_events("run-2")

        assert len(events_1) == 1
        assert events_1[0]["run"] == 1
        assert len(events_2) == 1
        assert events_2[0]["run"] == 2


class TestInMemoryBackendCheckpoints:
    """Tests for checkpoint storage and retrieval."""

    @pytest.fixture
    async def backend(self) -> InMemoryBackend:
        backend = InMemoryBackend()
        await backend.initialize()
        return backend

    async def test_store_and_retrieve_checkpoint(self, backend: InMemoryBackend) -> None:
        state = {"messages": ["hello"], "step": 1}
        await backend.store_checkpoint("run-1", "step-1", state)

        checkpoint = await backend.get_latest_checkpoint("run-1")
        assert checkpoint is not None
        assert checkpoint["step_id"] == "step-1"
        assert checkpoint["messages"] == ["hello"]
        assert checkpoint["step"] == 1

    async def test_latest_checkpoint_returns_most_recent(self, backend: InMemoryBackend) -> None:
        await backend.store_checkpoint("run-1", "step-1", {"version": 1})
        await backend.store_checkpoint("run-1", "step-2", {"version": 2})
        await backend.store_checkpoint("run-1", "step-3", {"version": 3})

        checkpoint = await backend.get_latest_checkpoint("run-1")
        assert checkpoint is not None
        assert checkpoint["step_id"] == "step-3"
        assert checkpoint["version"] == 3

    async def test_no_checkpoint_returns_none(self, backend: InMemoryBackend) -> None:
        checkpoint = await backend.get_latest_checkpoint("nonexistent-run")
        assert checkpoint is None

    async def test_checkpoints_isolated_by_run_id(self, backend: InMemoryBackend) -> None:
        await backend.store_checkpoint("run-1", "step-a", {"data": "a"})
        await backend.store_checkpoint("run-2", "step-b", {"data": "b"})

        cp1 = await backend.get_latest_checkpoint("run-1")
        cp2 = await backend.get_latest_checkpoint("run-2")

        assert cp1 is not None
        assert cp1["data"] == "a"
        assert cp2 is not None
        assert cp2["data"] == "b"


class TestInMemoryBackendKV:
    """Tests for key-value storage operations."""

    @pytest.fixture
    async def backend(self) -> InMemoryBackend:
        backend = InMemoryBackend()
        await backend.initialize()
        return backend

    async def test_put_and_get(self, backend: InMemoryBackend) -> None:
        await backend.put("config", "model", "gpt-4")
        value = await backend.get("config", "model")
        assert value == "gpt-4"

    async def test_put_overwrites_existing(self, backend: InMemoryBackend) -> None:
        await backend.put("config", "model", "gpt-3.5")
        await backend.put("config", "model", "gpt-4")
        value = await backend.get("config", "model")
        assert value == "gpt-4"

    async def test_get_nonexistent_key(self, backend: InMemoryBackend) -> None:
        value = await backend.get("config", "nonexistent")
        assert value is None

    async def test_get_nonexistent_namespace(self, backend: InMemoryBackend) -> None:
        value = await backend.get("nonexistent", "key")
        assert value is None

    async def test_delete_existing_key(self, backend: InMemoryBackend) -> None:
        await backend.put("config", "model", "gpt-4")
        deleted = await backend.delete("config", "model")
        assert deleted is True

        value = await backend.get("config", "model")
        assert value is None

    async def test_delete_nonexistent_key(self, backend: InMemoryBackend) -> None:
        deleted = await backend.delete("config", "nonexistent")
        assert deleted is False

    async def test_delete_nonexistent_namespace(self, backend: InMemoryBackend) -> None:
        deleted = await backend.delete("nonexistent", "key")
        assert deleted is False

    async def test_namespaces_are_isolated(self, backend: InMemoryBackend) -> None:
        await backend.put("ns-1", "key", "value-1")
        await backend.put("ns-2", "key", "value-2")

        assert await backend.get("ns-1", "key") == "value-1"
        assert await backend.get("ns-2", "key") == "value-2"

    async def test_complex_values(self, backend: InMemoryBackend) -> None:
        complex_value = {"nested": {"list": [1, 2, 3], "flag": True}}
        await backend.put("data", "complex", complex_value)
        retrieved = await backend.get("data", "complex")
        assert retrieved == complex_value


class TestInMemoryBackendClose:
    """Tests for backend lifecycle."""

    async def test_close_clears_all_data(self) -> None:
        backend = InMemoryBackend()
        await backend.initialize()

        await backend.store_trace_event("run-1", {"event_type": "llm_call"})
        await backend.store_checkpoint("run-1", "step-1", {"data": "test"})
        await backend.put("ns", "key", "value")

        await backend.close()

        assert await backend.get_trace_events("run-1") == []
        assert await backend.get_latest_checkpoint("run-1") is None
        assert await backend.get("ns", "key") is None


# ═══════════════════════════════════════════════════════════════════
# InMemoryVectorStore Tests
# ═══════════════════════════════════════════════════════════════════


class TestCosineSimility:
    """Tests for the cosine similarity helper function."""

    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert math.isclose(_cosine_similarity(v, v), 1.0, abs_tol=1e-9)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert math.isclose(_cosine_similarity(a, b), 0.0, abs_tol=1e-9)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert math.isclose(_cosine_similarity(a, b), -1.0, abs_tol=1e-9)

    def test_zero_vector_returns_zero(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0
        assert _cosine_similarity(b, a) == 0.0

    def test_known_similarity(self) -> None:
        # cos([1,1], [1,0]) = 1/sqrt(2) ~= 0.7071
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert math.isclose(_cosine_similarity(a, b), expected, abs_tol=1e-9)


class TestInMemoryVectorStore:
    """Tests for the in-memory vector store."""

    @pytest.fixture
    async def store(self) -> InMemoryVectorStore:
        store = InMemoryVectorStore()
        await store.initialize()
        return store

    async def test_upsert_and_count(self, store: InMemoryVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0, 0.0])
        await store.upsert("v2", [0.0, 1.0, 0.0])

        assert await store.count() == 2

    async def test_upsert_overwrites(self, store: InMemoryVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0], content="old")
        await store.upsert("v1", [0.0, 1.0], content="new")

        assert await store.count() == 1
        results = await store.search([0.0, 1.0], top_k=1)
        assert len(results) == 1
        assert results[0].id == "v1"
        assert results[0].content == "new"

    async def test_search_returns_correct_ordering(self, store: InMemoryVectorStore) -> None:
        # v1 is most similar to query [1, 0, 0]
        await store.upsert("v1", [1.0, 0.0, 0.0], content="closest")
        await store.upsert("v2", [0.7, 0.7, 0.0], content="medium")
        await store.upsert("v3", [0.0, 0.0, 1.0], content="orthogonal")

        results = await store.search([1.0, 0.0, 0.0])

        assert len(results) == 3
        assert results[0].id == "v1"  # Most similar
        assert results[0].content == "closest"
        assert results[1].id == "v2"  # Medium similarity
        assert results[2].id == "v3"  # Least similar (orthogonal)

        # Scores should be descending
        assert results[0].score >= results[1].score >= results[2].score

    async def test_search_top_k(self, store: InMemoryVectorStore) -> None:
        for i in range(10):
            await store.upsert(f"v{i}", [float(i), 1.0])

        results = await store.search([9.0, 1.0], top_k=3)
        assert len(results) == 3

    async def test_search_min_score(self, store: InMemoryVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0], content="aligned")
        await store.upsert("v2", [0.0, 1.0], content="orthogonal")

        results = await store.search([1.0, 0.0], min_score=0.5)

        # Only v1 should match (score ~1.0); v2 is orthogonal (score ~0.0)
        assert len(results) == 1
        assert results[0].id == "v1"

    async def test_search_with_metadata_filters(self, store: InMemoryVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0], metadata={"source": "web", "lang": "en"})
        await store.upsert("v2", [0.9, 0.1], metadata={"source": "file", "lang": "en"})
        await store.upsert("v3", [0.8, 0.2], metadata={"source": "web", "lang": "fr"})

        # Filter by source=web
        results = await store.search([1.0, 0.0], filters={"source": "web"})
        assert len(results) == 2
        assert {r.id for r in results} == {"v1", "v3"}

        # Filter by source=web AND lang=en
        results = await store.search([1.0, 0.0], filters={"source": "web", "lang": "en"})
        assert len(results) == 1
        assert results[0].id == "v1"

    async def test_search_with_no_matching_filters(self, store: InMemoryVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0], metadata={"source": "web"})
        results = await store.search([1.0, 0.0], filters={"source": "nonexistent"})
        assert results == []

    async def test_search_empty_store(self, store: InMemoryVectorStore) -> None:
        results = await store.search([1.0, 0.0])
        assert results == []

    async def test_delete_existing(self, store: InMemoryVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0])
        assert await store.count() == 1

        deleted = await store.delete("v1")
        assert deleted is True
        assert await store.count() == 0

    async def test_delete_nonexistent(self, store: InMemoryVectorStore) -> None:
        deleted = await store.delete("nonexistent")
        assert deleted is False

    async def test_close_clears_all_vectors(self, store: InMemoryVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0])
        await store.upsert("v2", [0.0, 1.0])

        await store.close()

        assert await store.count() == 0

    async def test_upsert_with_content_and_metadata(self, store: InMemoryVectorStore) -> None:
        await store.upsert(
            "v1",
            [1.0, 0.0],
            metadata={"source": "docs"},
            content="Hello, world!",
        )

        results = await store.search([1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].metadata == {"source": "docs"}
        assert results[0].content == "Hello, world!"

    async def test_search_result_has_correct_attributes(self, store: InMemoryVectorStore) -> None:
        await store.upsert("v1", [1.0, 0.0], metadata={"k": "v"}, content="text")

        results = await store.search([1.0, 0.0])
        result = results[0]

        assert result.id == "v1"
        assert isinstance(result.score, float)
        assert result.metadata == {"k": "v"}
        assert result.content == "text"

    async def test_cosine_similarity_ordering_is_correct(
        self, store: InMemoryVectorStore
    ) -> None:
        """Verify that search ordering matches manual cosine similarity computation."""
        query = [3.0, 4.0]
        vectors = {
            "a": [3.0, 4.0],   # identical -> sim = 1.0
            "b": [4.0, 3.0],   # close -> sim = 24/25 = 0.96
            "c": [0.0, 5.0],   # partial -> sim = 20/25 = 0.8
            "d": [-3.0, -4.0], # opposite -> sim = -1.0
        }

        for vid, vec in vectors.items():
            await store.upsert(vid, vec)

        # Use min_score=-1.0 to include negative similarities (default is 0.0)
        results = await store.search(query, min_score=-1.0)

        assert len(results) == 4
        assert results[0].id == "a"
        assert math.isclose(results[0].score, 1.0, abs_tol=1e-9)
        assert results[1].id == "b"
        assert results[2].id == "c"
        assert results[3].id == "d"
        assert math.isclose(results[3].score, -1.0, abs_tol=1e-9)

        # Default min_score=0.0 should exclude the opposite vector
        results_default = await store.search(query)
        assert len(results_default) == 3
        assert all(r.score >= 0.0 for r in results_default)
