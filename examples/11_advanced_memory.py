"""
Arcana: Advanced Memory (Application-Layer Composition)

Demonstrates MemoryManager as a composable module outside the Runtime.
Runtime has built-in lightweight memory (RunMemoryStore) for cross-run context.
For governed multi-tier memory, compose MemoryManager yourself.

    Runtime memory  → RunMemoryStore  → built-in, process-scoped
    Advanced memory → MemoryManager   → you compose, you control

Usage:
    uv run python examples/11_advanced_memory.py
"""

from __future__ import annotations

import asyncio

from arcana.contracts.memory import (
    MemoryConfig,
    MemoryQuery,
    MemoryType,
    MemoryWriteRequest,
    RevocationRequest,
)
from arcana.memory import (
    EpisodicMemoryStore,
    LongTermMemoryStore,
    MemoryManager,
    WorkingMemoryStore,
    WritePolicy,
)
from arcana.rag.embedder import MockEmbedder
from arcana.storage.memory import InMemoryBackend, InMemoryVectorStore
from arcana.trace.reader import TraceReader
from arcana.trace.writer import TraceWriter


def build_memory_manager(trace_dir: str = "/tmp/arcana-memory-demo") -> MemoryManager:
    """
    Build a MemoryManager with in-memory backends.

    In production you'd swap InMemoryBackend for SqliteBackend, Redis, etc.
    The MemoryManager doesn't care — it programs to the StorageBackend protocol.
    """
    backend = InMemoryBackend()
    vector_store = InMemoryVectorStore()
    embedder = MockEmbedder(dimensions=64)
    config = MemoryConfig()

    return MemoryManager(
        working=WorkingMemoryStore(backend, config),
        long_term=LongTermMemoryStore(vector_store, embedder, backend),
        episodic=EpisodicMemoryStore(
            trace_writer=TraceWriter(trace_dir=trace_dir),
            trace_reader=TraceReader(trace_dir=trace_dir),
            backend=backend,
        ),
        governance=WritePolicy(config),
        config=config,
    )


async def main():
    memory = build_memory_manager()

    # ── 1. Write facts with governance ───────────────────────────
    print("=== Writing facts ===")

    result = await memory.write(MemoryWriteRequest(
        memory_type=MemoryType.LONG_TERM,
        key="user-preference",
        content="User prefers concise responses",
        confidence=0.9,
        source="observation",
        run_id="run-001",
    ))
    print(f"  Wrote (confidence=0.9): success={result.success}")

    # Low confidence → rejected by governance
    result = await memory.write(MemoryWriteRequest(
        memory_type=MemoryType.LONG_TERM,
        key="uncertain-fact",
        content="User might like cats",
        confidence=0.3,
        source="guess",
        run_id="run-001",
    ))
    print(f"  Wrote (confidence=0.3): success={result.success}, reason={result.rejected_reason}")

    # ── 2. Query across memory types ─────────────────────────────
    print("\n=== Querying ===")

    results = await memory.query(MemoryQuery(
        query="user preference response style",
        memory_type=MemoryType.LONG_TERM,
    ))
    for entry in results:
        print(f"  Found: {entry.content} (confidence={entry.confidence})")

    # ── 3. Working memory (run-scoped) ───────────────────────────
    print("\n=== Working memory ===")

    await memory.write(MemoryWriteRequest(
        memory_type=MemoryType.WORKING,
        key="current-topic",
        content="Discussing memory architecture",
        run_id="run-002",
    ))
    entry = await memory.working.get("run-002", "current-topic")
    print(f"  Run run-002 topic: {entry.content if entry else 'none'}")

    # Different run can't see it
    entry2 = await memory.working.get("run-003", "current-topic")
    print(f"  Run run-003 topic: {entry2.content if entry2 else 'none (isolated)'}")

    # ── 4. Revocation (decontamination) ──────────────────────────
    print("\n=== Revocation ===")

    bad_fact = await memory.write(MemoryWriteRequest(
        memory_type=MemoryType.LONG_TERM,
        key="wrong-fact",
        content="The capital of Australia is Sydney",
        confidence=0.8,
        run_id="run-001",
    ))
    print(f"  Wrote bad fact: id={bad_fact.entry_id}")

    revoked = await memory.revoke(RevocationRequest(
        entry_id=bad_fact.entry_id,
        reason="Incorrect: capital is Canberra",
    ))
    print(f"  Revoked: {revoked}")

    # Query no longer returns it
    results = await memory.query(MemoryQuery(
        query="capital Australia",
        memory_type=MemoryType.LONG_TERM,
    ))
    print(f"  Query after revocation: {len(results)} results (bad fact excluded)")

    print("\nDone. MemoryManager is an application-layer module — you compose it, you control it.")


if __name__ == "__main__":
    asyncio.run(main())
