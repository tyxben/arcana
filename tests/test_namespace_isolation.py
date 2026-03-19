"""Tests for namespace/tenant isolation in Runtime, RunMemoryStore, and TraceWriter."""

from __future__ import annotations

from arcana.memory.run_memory import RunMemoryStore
from arcana.runtime_core import Runtime, RuntimeConfig
from arcana.trace.writer import TraceWriter

# ── RunMemoryStore namespace isolation ───────────────────────────


class TestRunMemoryStoreNamespace:
    """Two stores sharing the same backing list but different namespaces."""

    def test_different_namespaces_dont_share_facts(self) -> None:
        """Facts stored under namespace A are invisible to namespace B."""
        store_a = RunMemoryStore(namespace="tenant-a")
        store_b = RunMemoryStore(namespace="tenant-b")

        # Simulate shared backing store by merging internal lists
        shared: list = []  # type: ignore[type-arg]
        store_a._facts = shared
        store_b._facts = shared

        store_a.store("secret-a", run_id="r1", importance=0.8, tags=["a"])
        store_b.store("secret-b", run_id="r2", importance=0.8, tags=["b"])

        # Each store only sees its own facts
        assert store_a.fact_count == 1
        assert store_b.fact_count == 1
        assert store_a.facts[0].content == "secret-a"
        assert store_b.facts[0].content == "secret-b"

    def test_retrieve_respects_namespace(self) -> None:
        """retrieve() only returns facts from the same namespace."""
        store_a = RunMemoryStore(namespace="tenant-a")
        store_b = RunMemoryStore(namespace="tenant-b")

        shared: list = []  # type: ignore[type-arg]
        store_a._facts = shared
        store_b._facts = shared

        store_a.store("Python is great", run_id="r1", importance=0.9)
        store_b.store("Go is fast", run_id="r2", importance=0.9)

        ctx_a = store_a.retrieve("Python")
        ctx_b = store_b.retrieve("Go")

        assert "Python" in ctx_a
        assert "Go" not in ctx_a
        assert "Go" in ctx_b
        assert "Python" not in ctx_b

    def test_store_run_result_respects_namespace(self) -> None:
        """store_run_result tags facts with the namespace."""
        store_a = RunMemoryStore(namespace="ns-a")
        store_b = RunMemoryStore(namespace="ns-b")

        shared: list = []  # type: ignore[type-arg]
        store_a._facts = shared
        store_b._facts = shared

        store_a.store_run_result(goal="hello", answer="world", run_id="r1")
        store_b.store_run_result(goal="foo", answer="bar", run_id="r2")

        # Each sees only its own results
        assert store_a.fact_count >= 1
        assert store_b.fact_count >= 1
        assert all("hello" in f.content or "world" in f.content for f in store_a.facts)
        assert all("foo" in f.content or "bar" in f.content for f in store_b.facts)

    def test_namespace_none_sees_all_facts(self) -> None:
        """namespace=None preserves backward-compatible behavior (no filtering)."""
        store = RunMemoryStore()  # namespace=None
        store.store("fact one", run_id="r1")
        store.store("fact two very different content", run_id="r2")

        assert store.fact_count == 2
        ctx = store.retrieve("fact")
        assert "fact one" in ctx
        assert "fact two" in ctx

    def test_get_context_respects_namespace(self) -> None:
        """get_context() only returns facts from the current namespace."""
        store_a = RunMemoryStore(namespace="a")
        store_b = RunMemoryStore(namespace="b")

        shared: list = []  # type: ignore[type-arg]
        store_a._facts = shared
        store_b._facts = shared

        store_a.store("alpha fact", run_id="r1")
        store_b.store("beta fact", run_id="r2")

        ctx_a = store_a.get_context()
        ctx_b = store_b.get_context()

        assert "alpha" in ctx_a
        assert "beta" not in ctx_a
        assert "beta" in ctx_b
        assert "alpha" not in ctx_b


# ── TraceWriter namespace isolation ──────────────────────────────


class TestTraceWriterNamespace:
    def test_namespace_creates_subdirectory(self, tmp_path) -> None:
        """When namespace is set, trace files go to {trace_dir}/{namespace}/."""
        writer = TraceWriter(trace_dir=tmp_path, namespace="tenant-x")
        assert writer.trace_dir == tmp_path / "tenant-x"
        assert writer.trace_dir.exists()

    def test_namespace_none_uses_base_dir(self, tmp_path) -> None:
        """When namespace is None, trace files go to trace_dir directly."""
        writer = TraceWriter(trace_dir=tmp_path)
        assert writer.trace_dir == tmp_path

    def test_trace_files_written_to_namespace_subdir(self, tmp_path) -> None:
        """write_raw puts files in the namespace subdirectory."""
        writer_a = TraceWriter(trace_dir=tmp_path, namespace="ns-a")
        writer_b = TraceWriter(trace_dir=tmp_path, namespace="ns-b")

        writer_a.write_raw("run-1", {"event": "hello"})
        writer_b.write_raw("run-2", {"event": "world"})

        assert (tmp_path / "ns-a" / "run-1.jsonl").exists()
        assert (tmp_path / "ns-b" / "run-2.jsonl").exists()
        # No files in the base directory
        assert not list(tmp_path.glob("*.jsonl"))

    def test_list_runs_scoped_to_namespace(self, tmp_path) -> None:
        """list_runs only returns runs from the namespace subdir."""
        writer_a = TraceWriter(trace_dir=tmp_path, namespace="ns-a")
        writer_b = TraceWriter(trace_dir=tmp_path, namespace="ns-b")

        writer_a.write_raw("run-a1", {"x": 1})
        writer_a.write_raw("run-a2", {"x": 2})
        writer_b.write_raw("run-b1", {"x": 3})

        assert sorted(writer_a.list_runs()) == ["run-a1", "run-a2"]
        assert writer_b.list_runs() == ["run-b1"]


# ── Runtime namespace wiring ─────────────────────────────────────


class TestRuntimeNamespace:
    def test_namespace_stored(self) -> None:
        rt = Runtime(namespace="my-ns")
        assert rt.namespace == "my-ns"

    def test_namespace_default_none(self) -> None:
        rt = Runtime()
        assert rt.namespace is None

    def test_namespace_passed_to_memory(self) -> None:
        """Runtime passes namespace to RunMemoryStore."""
        rt = Runtime(memory=True, namespace="mem-ns")
        assert rt._memory_store is not None
        assert rt._memory_store._namespace == "mem-ns"

    def test_namespace_passed_to_trace(self, tmp_path) -> None:
        """Runtime passes namespace to TraceWriter."""
        rt = Runtime(
            trace=True,
            namespace="trace-ns",
            config=RuntimeConfig(trace_dir=str(tmp_path)),
        )
        assert rt._trace_writer is not None
        assert rt._trace_writer.trace_dir == tmp_path / "trace-ns"

    def test_no_namespace_memory_unchanged(self) -> None:
        """namespace=None preserves default memory behavior."""
        rt = Runtime(memory=True)
        assert rt._memory_store is not None
        assert rt._memory_store._namespace is None

    def test_no_namespace_trace_unchanged(self, tmp_path) -> None:
        """namespace=None preserves default trace behavior."""
        rt = Runtime(
            trace=True,
            config=RuntimeConfig(trace_dir=str(tmp_path)),
        )
        assert rt._trace_writer is not None
        assert rt._trace_writer.trace_dir == tmp_path
