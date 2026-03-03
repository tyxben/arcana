"""Working memory store — short-lived, run-scoped KV."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from arcana.contracts.memory import MemoryEntry

if TYPE_CHECKING:
    from arcana.contracts.memory import MemoryConfig
    from arcana.storage.base import StorageBackend


class WorkingMemoryStore:
    """
    Run-scoped working memory backed by StorageBackend KV.

    Each run gets its own namespace. Entries are MemoryEntry objects
    serialized to the KV store. A meta-key ``__keys__`` tracks all
    keys in the namespace for iteration.
    """

    def __init__(self, backend: StorageBackend, config: MemoryConfig) -> None:
        self.backend = backend
        self.config = config

    def _namespace(self, run_id: str) -> str:
        return f"{self.config.working_namespace_prefix}:{run_id}"

    async def put(self, run_id: str, key: str, entry: MemoryEntry) -> None:
        """Store a working memory entry."""
        ns = self._namespace(run_id)
        await self.backend.put(ns, key, entry.model_dump(mode="json"))

        # Track keys via meta-key
        keys: list[str] = await self.backend.get(ns, "__keys__") or []
        if key not in keys:
            keys.append(key)
            await self.backend.put(ns, "__keys__", keys)

    async def get(
        self, run_id: str, key: str, *, include_revoked: bool = False
    ) -> MemoryEntry | None:
        """Retrieve a working memory entry. Filters revoked by default."""
        data = await self.backend.get(self._namespace(run_id), key)
        if data is None:
            return None
        entry = MemoryEntry.model_validate(data)
        if entry.revoked and not include_revoked:
            return None
        return entry

    async def get_all(
        self, run_id: str, *, include_revoked: bool = False
    ) -> dict[str, MemoryEntry]:
        """Get all working memory entries for a run."""
        ns = self._namespace(run_id)
        keys: list[str] = await self.backend.get(ns, "__keys__") or []

        result: dict[str, MemoryEntry] = {}
        for key in keys:
            entry = await self.get(run_id, key, include_revoked=include_revoked)
            if entry is not None:
                result[key] = entry
        return result

    async def revoke(self, run_id: str, key: str, reason: str) -> bool:
        """Revoke a working memory entry (soft-delete, preserves history)."""
        entry = await self.get(run_id, key, include_revoked=True)
        if entry is None or entry.revoked:
            return False
        entry.revoked = True
        entry.revoked_at = datetime.now(UTC)
        entry.revoked_reason = reason
        entry.updated_at = datetime.now(UTC)
        await self.backend.put(
            self._namespace(run_id), key, entry.model_dump(mode="json")
        )
        return True

    async def delete(self, run_id: str, key: str) -> bool:
        """Hard-delete a working memory entry."""
        ns = self._namespace(run_id)
        deleted = await self.backend.delete(ns, key)
        if deleted:
            keys: list[str] = await self.backend.get(ns, "__keys__") or []
            if key in keys:
                keys.remove(key)
                await self.backend.put(ns, "__keys__", keys)
        return deleted
