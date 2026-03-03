"""MemoryManager — central orchestrator for the memory system."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from arcana.contracts.memory import (
    MemoryConfig,
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    MemoryWriteRequest,
    MemoryWriteResult,
    RevocationRequest,
)
from arcana.memory.episodic import EpisodicMemoryStore
from arcana.memory.governance import WritePolicy
from arcana.memory.long_term import LongTermMemoryStore
from arcana.memory.working import WorkingMemoryStore
from arcana.utils.hashing import canonical_hash


class MemoryManager:
    """
    Unified API for all memory operations.

    Routes writes/queries to the appropriate store based on memory type,
    enforces write governance, and logs all operations to episodic trace.
    """

    def __init__(
        self,
        working: WorkingMemoryStore,
        long_term: LongTermMemoryStore,
        episodic: EpisodicMemoryStore,
        governance: WritePolicy,
        config: MemoryConfig | None = None,
    ) -> None:
        self.working = working
        self.long_term = long_term
        self.episodic = episodic
        self.governance = governance
        self.config = config or MemoryConfig()

    async def write(self, request: MemoryWriteRequest) -> MemoryWriteResult:
        """
        Write a memory entry with governance checks.

        Flow: governance check → create entry → route to store → log episodic.
        """
        # 1. Governance check
        result = self.governance.evaluate(request)
        if not result.success:
            return result

        # 2. Create MemoryEntry
        now = datetime.now(UTC)
        entry = MemoryEntry(
            id=str(uuid4()),
            memory_type=request.memory_type,
            key=request.key,
            content=request.content,
            confidence=request.confidence,
            source=request.source,
            source_run_id=request.run_id,
            source_step_id=request.step_id,
            created_at=now,
            updated_at=now,
            tags=request.tags,
            metadata=request.metadata,
            content_hash=canonical_hash({"content": request.content}),
        )

        # 3. Route to appropriate store
        if request.memory_type == MemoryType.WORKING:
            if request.run_id is None:
                return MemoryWriteResult(
                    success=False,
                    rejected_reason="Working memory requires run_id",
                )
            await self.working.put(request.run_id, request.key, entry)

        elif request.memory_type == MemoryType.LONG_TERM:
            await self.long_term.store(entry)

        elif request.memory_type == MemoryType.EPISODIC:
            if request.run_id:
                await self.episodic.record_event(request.run_id, entry)

        # 4. Log all writes to episodic trace
        if request.run_id and request.memory_type != MemoryType.EPISODIC:
            await self.episodic.record_event(request.run_id, entry)

        result.entry_id = entry.id
        return result

    async def query(self, query: MemoryQuery) -> list[MemoryEntry]:
        """
        Query memory across types.

        If memory_type is specified, queries only that type.
        Otherwise queries all applicable types and merges results.
        """
        results: list[MemoryEntry] = []

        if query.memory_type is None or query.memory_type == MemoryType.WORKING:
            if query.run_id and query.key:
                entry = await self.working.get(query.run_id, query.key)
                if entry:
                    results.append(entry)
            elif query.run_id:
                entries = await self.working.get_all(query.run_id)
                results.extend(entries.values())

        if query.memory_type is None or query.memory_type == MemoryType.LONG_TERM:
            if query.query:
                lt_results = await self.long_term.search(query)
                results.extend(lt_results)

        if query.memory_type is None or query.memory_type == MemoryType.EPISODIC:
            ep_results = await self.episodic.get_cross_run_episodes(query)
            results.extend(ep_results)

        # Filter by min_confidence
        if query.min_confidence > 0:
            results = [e for e in results if e.confidence >= query.min_confidence]

        return results[: query.top_k]

    async def revoke(self, request: RevocationRequest) -> bool:
        """
        Revoke a memory entry by ID.

        Sets revoked=True but preserves the entry for audit history.
        """
        # Try long-term store (most common revocation target)
        entry = await self.long_term.get_by_id(request.entry_id)
        if entry is not None:
            if not self.governance.validate_revocation(entry, request):
                return False
            success = await self.long_term.revoke(
                request.entry_id, request.reason
            )
            if success and entry.source_run_id:
                entry.revoked = True
                entry.revoked_reason = request.reason
                await self.episodic.record_event(entry.source_run_id, entry)
            return success

        return False

    async def find_and_revoke_by_content(
        self, content_pattern: str, reason: str
    ) -> list[str]:
        """
        Find entries matching content semantically and revoke them.

        Used for decontamination: locate and revoke incorrect information.
        """
        query = MemoryQuery(
            query=content_pattern,
            memory_type=MemoryType.LONG_TERM,
            include_revoked=False,
            top_k=50,
        )
        matches = await self.long_term.search(query)

        revoked_ids: list[str] = []
        for entry in matches:
            success = await self.revoke(
                RevocationRequest(
                    entry_id=entry.id,
                    reason=reason,
                    revoked_by="decontamination",
                )
            )
            if success:
                revoked_ids.append(entry.id)

        return revoked_ids
