"""Tests for SharedContext -- thread-safe key-value store."""

from __future__ import annotations

import threading

from arcana.multi_agent.shared_context import SharedContext


class TestSharedContext:
    def test_get_set_basic(self):
        ctx = SharedContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_get_missing_returns_default(self):
        ctx = SharedContext()
        assert ctx.get("nope") is None
        assert ctx.get("nope", 42) == 42

    def test_delete_existing_returns_true(self):
        ctx = SharedContext()
        ctx.set("x", 1)
        assert ctx.delete("x") is True
        assert ctx.get("x") is None

    def test_delete_missing_returns_false(self):
        ctx = SharedContext()
        assert ctx.delete("ghost") is False

    def test_keys_lists_all(self):
        ctx = SharedContext()
        ctx.set("a", 1)
        ctx.set("b", 2)
        ctx.set("c", 3)
        assert sorted(ctx.keys()) == ["a", "b", "c"]

    def test_snapshot_returns_copy(self):
        ctx = SharedContext()
        ctx.set("x", [1, 2])
        snap = ctx.snapshot()

        # Mutating the snapshot dict should not affect the store
        snap["y"] = 999
        assert ctx.get("y") is None

    def test_clear_empties_everything(self):
        ctx = SharedContext()
        ctx.set("a", 1)
        ctx.set("b", 2)
        ctx.clear()
        assert ctx.keys() == []
        assert ctx.snapshot() == {}

    def test_thread_safety_concurrent_writes(self):
        """50 threads writing concurrently -- no crashes, all keys present."""
        ctx = SharedContext()
        barrier = threading.Barrier(50)

        def writer(idx: int) -> None:
            barrier.wait()
            ctx.set(f"key-{idx}", idx)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = ctx.snapshot()
        assert len(snap) == 50
        for i in range(50):
            assert snap[f"key-{i}"] == i

    def test_overwrite_existing_key(self):
        ctx = SharedContext()
        ctx.set("k", "v1")
        ctx.set("k", "v2")
        assert ctx.get("k") == "v2"

    def test_keys_empty_initially(self):
        ctx = SharedContext()
        assert ctx.keys() == []
