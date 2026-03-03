"""Episodic memory store — event trajectory logs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.memory import MemoryEntry, MemoryQuery
from arcana.contracts.trace import EventType, TraceEvent

if TYPE_CHECKING:
    from arcana.storage.base import StorageBackend
    from arcana.trace.reader import TraceReader
    from arcana.trace.writer import TraceWriter


class EpisodicMemoryStore:
    """
    Event-based memory backed by the trace system.

    Records memory operations as TraceEvents and stores full entries
    in KV for hydration. Episodic memory is primarily read-oriented —
    the trace system captures the event log automatically.
    """

    def __init__(
        self,
        trace_writer: TraceWriter | None = None,
        trace_reader: TraceReader | None = None,
        backend: StorageBackend | None = None,
    ) -> None:
        self.trace_writer = trace_writer
        self.trace_reader = trace_reader
        self.backend = backend

    async def record_event(self, run_id: str, entry: MemoryEntry) -> None:
        """Record a memory event to the trace log and KV store."""
        if self.trace_writer:
            event = TraceEvent(
                run_id=run_id,
                event_type=EventType.MEMORY_WRITE,
                metadata={
                    "memory_entry_id": entry.id,
                    "memory_type": entry.memory_type.value,
                    "key": entry.key,
                    "content_preview": entry.content[:200],
                    "confidence": entry.confidence,
                    "source": entry.source,
                    "revoked": entry.revoked,
                },
            )
            self.trace_writer.write(event)

        if self.backend:
            await self.backend.put(
                f"episodic:{run_id}",
                entry.id,
                entry.model_dump(mode="json"),
            )

    async def get_trajectory(
        self,
        run_id: str,
        *,
        include_revoked: bool = False,
    ) -> list[MemoryEntry]:
        """Get the memory trajectory for a run."""
        if not self.trace_reader:
            return []

        events = self.trace_reader.filter_events(
            run_id,
            event_types=[EventType.MEMORY_WRITE],
        )

        entries: list[MemoryEntry] = []
        seen_ids: set[str] = set()

        for event in events:
            entry_id = event.metadata.get("memory_entry_id")
            if not entry_id or entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)

            if self.backend:
                data = await self.backend.get(f"episodic:{run_id}", entry_id)
                if data is not None:
                    entry = MemoryEntry.model_validate(data)
                    if include_revoked or not entry.revoked:
                        entries.append(entry)

        return entries

    async def get_cross_run_episodes(
        self, query: MemoryQuery
    ) -> list[MemoryEntry]:
        """Search episodic memory across runs (by run_id filter)."""
        if query.run_id:
            return await self.get_trajectory(
                query.run_id,
                include_revoked=query.include_revoked,
            )
        return []
