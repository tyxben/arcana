"""Comprehensive tests for the Memory system."""

from __future__ import annotations

import pytest

from arcana.contracts.memory import (
    MemoryConfig,
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    MemoryWriteRequest,
    MemoryWriteResult,
    RevocationRequest,
)
from arcana.contracts.trace import EventType
from arcana.memory.episodic import EpisodicMemoryStore
from arcana.memory.governance import WritePolicy
from arcana.memory.long_term import LongTermMemoryStore
from arcana.memory.manager import MemoryManager
from arcana.memory.working import WorkingMemoryStore
from arcana.rag.embedder import MockEmbedder
from arcana.storage.memory import InMemoryBackend, InMemoryVectorStore
from arcana.trace.writer import TraceWriter


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder(dimensions=64)


@pytest.fixture
def config() -> MemoryConfig:
    return MemoryConfig()


@pytest.fixture
def governance(config: MemoryConfig) -> WritePolicy:
    return WritePolicy(config)


@pytest.fixture
def working_store(backend: InMemoryBackend, config: MemoryConfig) -> WorkingMemoryStore:
    return WorkingMemoryStore(backend, config)


@pytest.fixture
def long_term_store(
    vector_store: InMemoryVectorStore,
    embedder: MockEmbedder,
    backend: InMemoryBackend,
) -> LongTermMemoryStore:
    return LongTermMemoryStore(vector_store, embedder, backend)


@pytest.fixture
def episodic_store(
    backend: InMemoryBackend, tmp_path
) -> EpisodicMemoryStore:
    trace_writer = TraceWriter(trace_dir=tmp_path)
    from arcana.trace.reader import TraceReader

    trace_reader = TraceReader(trace_dir=tmp_path)
    return EpisodicMemoryStore(
        trace_writer=trace_writer,
        trace_reader=trace_reader,
        backend=backend,
    )


@pytest.fixture
def memory_manager(
    working_store: WorkingMemoryStore,
    long_term_store: LongTermMemoryStore,
    episodic_store: EpisodicMemoryStore,
    governance: WritePolicy,
    config: MemoryConfig,
) -> MemoryManager:
    return MemoryManager(
        working=working_store,
        long_term=long_term_store,
        episodic=episodic_store,
        governance=governance,
        config=config,
    )


# ── Contract Tests ───────────────────────────────────────────────


class TestContracts:
    """Test Pydantic contract serialization and defaults."""

    def test_memory_entry_serialization(self) -> None:
        entry = MemoryEntry(
            id="e1",
            memory_type=MemoryType.LONG_TERM,
            key="fact",
            content="The sky is blue",
            confidence=0.9,
        )
        data = entry.model_dump(mode="json")
        restored = MemoryEntry.model_validate(data)
        assert restored.id == "e1"
        assert restored.content == "The sky is blue"
        assert restored.confidence == 0.9

    def test_memory_entry_revoked_defaults(self) -> None:
        entry = MemoryEntry(
            id="e1",
            memory_type=MemoryType.WORKING,
            key="k",
            content="c",
        )
        assert entry.revoked is False
        assert entry.revoked_at is None
        assert entry.revoked_reason is None

    def test_memory_query_defaults(self) -> None:
        query = MemoryQuery(query="test")
        assert query.include_revoked is False
        assert query.min_confidence == 0.0
        assert query.top_k == 10

    def test_memory_config_defaults(self) -> None:
        config = MemoryConfig()
        assert config.min_write_confidence == 0.5
        assert config.warn_confidence_threshold == 0.7


# ── WritePolicy Tests ────────────────────────────────────────────


class TestWritePolicy:
    """Test write governance logic."""

    def test_write_accepted_above_threshold(self, governance: WritePolicy) -> None:
        request = MemoryWriteRequest(
            memory_type=MemoryType.LONG_TERM,
            key="fact",
            content="test",
            confidence=0.8,
        )
        result = governance.evaluate(request)
        assert result.success is True

    def test_write_rejected_below_threshold(self, governance: WritePolicy) -> None:
        request = MemoryWriteRequest(
            memory_type=MemoryType.LONG_TERM,
            key="fact",
            content="test",
            confidence=0.3,
        )
        result = governance.evaluate(request)
        assert result.success is False
        assert result.confidence_below_threshold is True
        assert "0.30" in (result.rejected_reason or "")

    def test_write_at_exact_threshold(self) -> None:
        config = MemoryConfig(min_write_confidence=0.5)
        policy = WritePolicy(config)
        request = MemoryWriteRequest(
            memory_type=MemoryType.LONG_TERM,
            key="fact",
            content="test",
            confidence=0.5,
        )
        result = policy.evaluate(request)
        assert result.success is True

    def test_revocation_already_revoked(self, governance: WritePolicy) -> None:
        entry = MemoryEntry(
            id="e1",
            memory_type=MemoryType.LONG_TERM,
            key="k",
            content="c",
            revoked=True,
        )
        request = RevocationRequest(entry_id="e1", reason="test")
        assert governance.validate_revocation(entry, request) is False


# ── WorkingMemoryStore Tests ─────────────────────────────────────


class TestWorkingMemoryStore:
    """Test working memory (run-scoped KV)."""

    async def test_put_and_get(self, working_store: WorkingMemoryStore) -> None:
        entry = MemoryEntry(
            id="w1",
            memory_type=MemoryType.WORKING,
            key="progress",
            content="Step 1 done",
        )
        await working_store.put("run-1", "progress", entry)
        result = await working_store.get("run-1", "progress")
        assert result is not None
        assert result.content == "Step 1 done"

    async def test_get_nonexistent(self, working_store: WorkingMemoryStore) -> None:
        result = await working_store.get("run-x", "missing")
        assert result is None

    async def test_get_all_entries(self, working_store: WorkingMemoryStore) -> None:
        for i in range(3):
            entry = MemoryEntry(
                id=f"w{i}",
                memory_type=MemoryType.WORKING,
                key=f"key{i}",
                content=f"value{i}",
            )
            await working_store.put("run-1", f"key{i}", entry)

        all_entries = await working_store.get_all("run-1")
        assert len(all_entries) == 3

    async def test_revoke_hides_entry(self, working_store: WorkingMemoryStore) -> None:
        entry = MemoryEntry(
            id="w1",
            memory_type=MemoryType.WORKING,
            key="k",
            content="to revoke",
        )
        await working_store.put("run-1", "k", entry)
        assert await working_store.revoke("run-1", "k", "bad data")

        # Default get returns None for revoked
        assert await working_store.get("run-1", "k") is None
        # But include_revoked=True still finds it
        revoked = await working_store.get("run-1", "k", include_revoked=True)
        assert revoked is not None
        assert revoked.revoked is True
        assert revoked.revoked_reason == "bad data"

    async def test_namespace_isolation(self, working_store: WorkingMemoryStore) -> None:
        entry1 = MemoryEntry(
            id="w1", memory_type=MemoryType.WORKING, key="k", content="run1"
        )
        entry2 = MemoryEntry(
            id="w2", memory_type=MemoryType.WORKING, key="k", content="run2"
        )
        await working_store.put("run-1", "k", entry1)
        await working_store.put("run-2", "k", entry2)

        r1 = await working_store.get("run-1", "k")
        r2 = await working_store.get("run-2", "k")
        assert r1 is not None and r1.content == "run1"
        assert r2 is not None and r2.content == "run2"


# ── LongTermMemoryStore Tests ────────────────────────────────────


class TestLongTermMemoryStore:
    """Test long-term memory (vector-indexed facts)."""

    async def test_store_and_search(
        self, long_term_store: LongTermMemoryStore
    ) -> None:
        entry = MemoryEntry(
            id="lt1",
            memory_type=MemoryType.LONG_TERM,
            key="fact",
            content="Python is a programming language",
            confidence=0.9,
        )
        await long_term_store.store(entry)

        query = MemoryQuery(
            query="programming language",
            memory_type=MemoryType.LONG_TERM,
            min_confidence=0.0,
        )
        results = await long_term_store.search(query)
        assert len(results) >= 1
        assert results[0].content == "Python is a programming language"

    async def test_search_excludes_revoked(
        self, long_term_store: LongTermMemoryStore
    ) -> None:
        entry = MemoryEntry(
            id="lt1",
            memory_type=MemoryType.LONG_TERM,
            key="bad-fact",
            content="The moon is made of cheese",
            confidence=0.9,
        )
        await long_term_store.store(entry)
        await long_term_store.revoke("lt1", "incorrect")

        query = MemoryQuery(
            query="moon cheese", memory_type=MemoryType.LONG_TERM
        )
        results = await long_term_store.search(query)
        assert len(results) == 0

    async def test_search_includes_revoked_when_requested(
        self, long_term_store: LongTermMemoryStore
    ) -> None:
        entry = MemoryEntry(
            id="lt1",
            memory_type=MemoryType.LONG_TERM,
            key="fact",
            content="A revoked fact",
            confidence=0.9,
        )
        await long_term_store.store(entry)
        await long_term_store.revoke("lt1", "wrong")

        query = MemoryQuery(
            query="revoked fact",
            memory_type=MemoryType.LONG_TERM,
            include_revoked=True,
        )
        results = await long_term_store.search(query)
        assert len(results) >= 1
        assert results[0].revoked is True

    async def test_get_by_id(self, long_term_store: LongTermMemoryStore) -> None:
        entry = MemoryEntry(
            id="lt-direct",
            memory_type=MemoryType.LONG_TERM,
            key="fact",
            content="Direct lookup test",
        )
        await long_term_store.store(entry)

        result = await long_term_store.get_by_id("lt-direct")
        assert result is not None
        assert result.content == "Direct lookup test"

        assert await long_term_store.get_by_id("nonexistent") is None


# ── EpisodicMemoryStore Tests ────────────────────────────────────


class TestEpisodicMemoryStore:
    """Test episodic memory (event trajectory)."""

    async def test_record_event_creates_trace(
        self, episodic_store: EpisodicMemoryStore, tmp_path
    ) -> None:
        entry = MemoryEntry(
            id="ep1",
            memory_type=MemoryType.EPISODIC,
            key="observation",
            content="User asked about Python",
            confidence=0.8,
        )
        await episodic_store.record_event("run-ep", entry)

        # Check trace file was created
        trace_files = list(tmp_path.glob("*.jsonl"))
        assert len(trace_files) > 0

    async def test_get_trajectory(
        self, episodic_store: EpisodicMemoryStore
    ) -> None:
        for i in range(3):
            entry = MemoryEntry(
                id=f"ep{i}",
                memory_type=MemoryType.EPISODIC,
                key=f"step{i}",
                content=f"Event {i} happened",
            )
            await episodic_store.record_event("run-traj", entry)

        trajectory = await episodic_store.get_trajectory("run-traj")
        assert len(trajectory) == 3

    async def test_trajectory_excludes_revoked(
        self, episodic_store: EpisodicMemoryStore
    ) -> None:
        entry = MemoryEntry(
            id="ep-revoked",
            memory_type=MemoryType.EPISODIC,
            key="bad",
            content="Bad event",
            revoked=True,
        )
        await episodic_store.record_event("run-rev", entry)

        trajectory = await episodic_store.get_trajectory("run-rev")
        assert len(trajectory) == 0

        # But include_revoked=True shows it
        trajectory_all = await episodic_store.get_trajectory(
            "run-rev", include_revoked=True
        )
        assert len(trajectory_all) == 1


# ── MemoryManager Integration Tests ─────────────────────────────


class TestMemoryManager:
    """Test the MemoryManager orchestration layer."""

    async def test_write_routes_to_working(
        self, memory_manager: MemoryManager
    ) -> None:
        request = MemoryWriteRequest(
            memory_type=MemoryType.WORKING,
            key="progress",
            content="Step 1 complete",
            run_id="run-wm",
        )
        result = await memory_manager.write(request)
        assert result.success is True
        assert result.entry_id is not None

        # Verify it's in working store
        entry = await memory_manager.working.get("run-wm", "progress")
        assert entry is not None
        assert entry.content == "Step 1 complete"

    async def test_write_routes_to_long_term(
        self, memory_manager: MemoryManager
    ) -> None:
        request = MemoryWriteRequest(
            memory_type=MemoryType.LONG_TERM,
            key="fact",
            content="Water boils at 100 degrees Celsius",
            confidence=0.95,
            run_id="run-lt",
        )
        result = await memory_manager.write(request)
        assert result.success is True

        # Verify it's searchable
        query = MemoryQuery(
            query="water boiling point", memory_type=MemoryType.LONG_TERM
        )
        results = await memory_manager.query(query)
        assert len(results) >= 1

    async def test_write_rejected_by_governance(
        self, memory_manager: MemoryManager
    ) -> None:
        request = MemoryWriteRequest(
            memory_type=MemoryType.LONG_TERM,
            key="uncertain",
            content="Maybe the earth is flat",
            confidence=0.2,  # Below threshold
            run_id="run-gov",
        )
        result = await memory_manager.write(request)
        assert result.success is False
        assert result.confidence_below_threshold is True

    async def test_revoke_and_requery(
        self, memory_manager: MemoryManager
    ) -> None:
        # Write a fact
        request = MemoryWriteRequest(
            memory_type=MemoryType.LONG_TERM,
            key="wrong-fact",
            content="Paris is the capital of Germany",
            confidence=0.8,
            run_id="run-rev",
        )
        result = await memory_manager.write(request)
        assert result.success and result.entry_id

        # Revoke it
        revoked = await memory_manager.revoke(
            RevocationRequest(
                entry_id=result.entry_id,
                reason="Incorrect: Paris is the capital of France",
            )
        )
        assert revoked is True

        # Query should NOT return it
        query = MemoryQuery(
            query="capital of Germany", memory_type=MemoryType.LONG_TERM
        )
        results = await memory_manager.query(query)
        assert all(r.id != result.entry_id for r in results)

    async def test_find_and_revoke_by_content(
        self, memory_manager: MemoryManager
    ) -> None:
        # Write several facts
        for content in [
            "The sun revolves around the earth",
            "Earth is the center of the universe",
            "Gravity pulls objects together",
        ]:
            await memory_manager.write(
                MemoryWriteRequest(
                    memory_type=MemoryType.LONG_TERM,
                    key="fact",
                    content=content,
                    confidence=0.8,
                    run_id="run-decon",
                )
            )

        # Decontaminate
        revoked_ids = await memory_manager.find_and_revoke_by_content(
            "earth revolves sun universe",
            reason="Geocentric model is incorrect",
        )
        # At least some entries should be revoked
        assert len(revoked_ids) >= 1

    async def test_working_memory_requires_run_id(
        self, memory_manager: MemoryManager
    ) -> None:
        request = MemoryWriteRequest(
            memory_type=MemoryType.WORKING,
            key="k",
            content="v",
            run_id=None,
        )
        result = await memory_manager.write(request)
        assert result.success is False
        assert "run_id" in (result.rejected_reason or "")


# ── MemoryHook Tests ─────────────────────────────────────────────


class TestMemoryHook:
    """Test MemoryHook integration with the Agent Runtime."""

    async def test_on_step_complete_persists_updates(
        self, memory_manager: MemoryManager
    ) -> None:
        from arcana.contracts.runtime import StepResult
        from arcana.contracts.state import AgentState
        from arcana.contracts.trace import TraceContext
        from arcana.runtime.hooks.memory_hook import MemoryHook

        hook = MemoryHook(memory_manager)

        state = AgentState(run_id="run-hook", current_step=1)
        step_result = StepResult(
            step_id="step-1",
            step_type="act",
            success=True,
            memory_updates={"conclusion": "Task is progressing well"},
        )
        trace_ctx = TraceContext(run_id="run-hook", task_id="task-1")

        await hook.on_step_complete(state, step_result, trace_ctx)

        # Check it was persisted
        entry = await memory_manager.working.get("run-hook", "conclusion")
        assert entry is not None
        assert entry.content == "Task is progressing well"
        assert entry.source == "step_result"

    async def test_on_run_start_loads_memory(
        self, memory_manager: MemoryManager
    ) -> None:
        from arcana.contracts.state import AgentState
        from arcana.contracts.trace import TraceContext
        from arcana.runtime.hooks.memory_hook import MemoryHook

        hook = MemoryHook(memory_manager)

        # Pre-populate working memory
        await memory_manager.write(
            MemoryWriteRequest(
                memory_type=MemoryType.WORKING,
                key="existing",
                content="Previously saved data",
                run_id="run-load",
            )
        )

        state = AgentState(run_id="run-load")
        trace_ctx = TraceContext(run_id="run-load", task_id="task-1")

        await hook.on_run_start(state, trace_ctx)
        assert state.working_memory.get("existing") == "Previously saved data"

    async def test_on_run_end_promotes_flagged(
        self, memory_manager: MemoryManager
    ) -> None:
        from arcana.contracts.memory import MemoryEntry, MemoryType
        from arcana.contracts.state import AgentState, ExecutionStatus
        from arcana.contracts.trace import TraceContext
        from arcana.runtime.hooks.memory_hook import MemoryHook

        hook = MemoryHook(memory_manager)

        # Write working memory with promote flag
        entry = MemoryEntry(
            id="promote-me",
            memory_type=MemoryType.WORKING,
            key="learned-fact",
            content="Important discovery to keep",
            confidence=0.9,
            metadata={"promote_to_long_term": True},
        )
        await memory_manager.working.put("run-promote", "learned-fact", entry)

        state = AgentState(run_id="run-promote", status=ExecutionStatus.COMPLETED)
        trace_ctx = TraceContext(run_id="run-promote", task_id="task-1")

        await hook.on_run_end(state, trace_ctx)

        # Should now be in long-term memory
        query = MemoryQuery(
            query="important discovery",
            memory_type=MemoryType.LONG_TERM,
        )
        results = await memory_manager.query(query)
        assert len(results) >= 1
        assert any("Important discovery" in r.content for r in results)


# ── Acceptance Criteria Tests ────────────────────────────────────


class TestAcceptanceCriteria:
    """End-to-end tests for Week 7 acceptance criteria."""

    async def test_locate_and_revoke_contaminated_memory(
        self, memory_manager: MemoryManager
    ) -> None:
        """
        Acceptance: can locate and revoke contaminated memory;
        after revocation, queries no longer return it.
        """
        # 1. Write contaminated memory
        result = await memory_manager.write(
            MemoryWriteRequest(
                memory_type=MemoryType.LONG_TERM,
                key="wrong-answer",
                content="The capital of Australia is Sydney",
                confidence=0.8,
                source="tool_result",
                run_id="run-contaminated",
            )
        )
        assert result.success and result.entry_id
        contaminated_id = result.entry_id

        # 2. Locate it by semantic search
        query = MemoryQuery(
            query="capital of Australia",
            memory_type=MemoryType.LONG_TERM,
        )
        found = await memory_manager.query(query)
        assert any(e.id == contaminated_id for e in found)

        # 3. Revoke it
        revoked = await memory_manager.revoke(
            RevocationRequest(
                entry_id=contaminated_id,
                reason="Incorrect: the capital of Australia is Canberra",
            )
        )
        assert revoked is True

        # 4. Query again — should NOT return the contaminated entry
        found_after = await memory_manager.query(query)
        assert all(e.id != contaminated_id for e in found_after)

    async def test_revoked_entry_preserves_history(
        self, memory_manager: MemoryManager
    ) -> None:
        """
        Acceptance: revoked entries preserve history with include_revoked=True.
        """
        result = await memory_manager.write(
            MemoryWriteRequest(
                memory_type=MemoryType.LONG_TERM,
                key="fact",
                content="Some disputed fact",
                confidence=0.8,
                run_id="run-hist",
            )
        )
        assert result.entry_id

        await memory_manager.revoke(
            RevocationRequest(
                entry_id=result.entry_id,
                reason="Disputed and found to be incorrect",
            )
        )

        # Direct lookup still finds it with revoked=True
        entry = await memory_manager.long_term.get_by_id(result.entry_id)
        assert entry is not None
        assert entry.revoked is True
        assert entry.revoked_reason == "Disputed and found to be incorrect"
        assert entry.revoked_at is not None
