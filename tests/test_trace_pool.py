"""Tests for pool-aware trace metadata (``source_agent``).

v0.8.0 invariant: every trace event emitted during a pool run carries
``metadata["source_agent"] = <pool_agent_name>``. The ``TraceEvent``
schema itself is unchanged — v0.6.0/v0.7.0 consumers keep working.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arcana.contracts.llm import LLMResponse, TokenUsage
from arcana.contracts.trace import AgentRole, EventType, TraceEvent
from arcana.multi_agent.agent_pool import AgentPool
from arcana.runtime_core import Runtime, _PoolTaggedTraceWriter
from arcana.trace.writer import TraceWriter


# ── helpers ─────────────────────────────────────────────────────────────


class _MockProvider:
    provider_name = "mock"
    default_model = "mock-model"
    profile = None

    async def generate(self, request, config, trace_ctx=None):
        return LLMResponse(
            content="ok",
            model="mock-model",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def close(self) -> None:
        pass


def _make_runtime_with_mock(trace_dir: Path) -> Runtime:
    rt = Runtime()
    rt._gateway._providers["mock"] = _MockProvider()
    rt._gateway._default_provider = "mock"
    rt._config.default_provider = "mock"
    # Inject a real TraceWriter so the pool-wrapping path has something to wrap.
    rt._trace_writer = TraceWriter(trace_dir=str(trace_dir))
    return rt


def _event(run_id: str = "r1", **meta_extra) -> TraceEvent:
    md = dict(meta_extra)
    return TraceEvent(
        run_id=run_id,
        role=AgentRole.SYSTEM,
        event_type=EventType.LLM_CALL,
        metadata=md,
    )


# ── _PoolTaggedTraceWriter (unit level) ─────────────────────────────────


class TestPoolTaggedTraceWriter:
    def test_tags_event_metadata_with_source_agent(self, tmp_path):
        inner = TraceWriter(trace_dir=str(tmp_path))
        tagged = _PoolTaggedTraceWriter(inner, source_agent="planner")

        evt = _event()
        assert "source_agent" not in evt.metadata

        tagged.write(evt)
        assert evt.metadata["source_agent"] == "planner"

    def test_preserves_existing_source_agent_if_caller_set_it(self, tmp_path):
        inner = TraceWriter(trace_dir=str(tmp_path))
        tagged = _PoolTaggedTraceWriter(inner, source_agent="planner")

        evt = _event(source_agent="something-else")
        tagged.write(evt)

        assert evt.metadata["source_agent"] == "something-else"

    def test_proxies_other_attributes(self, tmp_path):
        inner = TraceWriter(trace_dir=str(tmp_path))
        tagged = _PoolTaggedTraceWriter(inner, source_agent="worker")

        # exists / list_runs are pass-throughs
        assert tagged.exists("nonexistent") is False
        assert tagged.list_runs() == []
        # trace_dir attribute is proxied
        assert tagged.trace_dir == inner.trace_dir

    def test_persisted_event_json_contains_source_agent(self, tmp_path):
        """End-to-end: after write, the JSONL file contains the tag."""
        inner = TraceWriter(trace_dir=str(tmp_path))
        tagged = _PoolTaggedTraceWriter(inner, source_agent="critic")

        tagged.write(_event(run_id="r-xyz"))

        trace_file = Path(tmp_path) / "r-xyz.jsonl"
        assert trace_file.exists()

        import json

        line = trace_file.read_text().strip()
        record = json.loads(line)
        assert record["metadata"]["source_agent"] == "critic"

    def test_non_event_write_does_not_crash(self, tmp_path):
        """Defensive: if something non-TraceEvent-shaped is passed, we
        don't blow up — just delegate."""
        captured = []

        class _Capture:
            def write(self, obj):
                captured.append(obj)

        tagged = _PoolTaggedTraceWriter(_Capture(), source_agent="x")
        tagged.write(object())  # no metadata attr

        assert len(captured) == 1


# ── ChatSession → pool wrapping (integration) ───────────────────────────


class TestChatSessionPoolWiring:
    def test_pool_session_wraps_trace_writer(self, tmp_path):
        rt = _make_runtime_with_mock(tmp_path)
        pool = AgentPool(rt)
        session = pool.add("planner")

        # Build an agent so the wiring decision runs.
        agent = session._build_agent(goal="hello")

        # The agent's trace_writer should be the wrapper, tagging "planner"
        tw = agent.trace_writer
        assert isinstance(tw, _PoolTaggedTraceWriter)
        # Wrapped inner is the runtime's TraceWriter (proxied attrs work).
        assert tw._source_agent == "planner"

    def test_bare_chat_session_is_not_wrapped(self, tmp_path):
        """Non-pool sessions pass the bare TraceWriter through unchanged —
        no perf overhead, no metadata tagging."""
        rt = _make_runtime_with_mock(tmp_path)

        async def _run():
            async with rt.chat() as c:
                agent = c._build_agent(goal="hi")
                assert not isinstance(agent.trace_writer, _PoolTaggedTraceWriter)

        import asyncio

        asyncio.run(_run())

    def test_different_pool_agents_get_different_tags(self, tmp_path):
        rt = _make_runtime_with_mock(tmp_path)
        pool = AgentPool(rt)
        a = pool.add("alpha")
        b = pool.add("beta")

        agent_a = a._build_agent(goal="g")
        agent_b = b._build_agent(goal="g")

        assert agent_a.trace_writer._source_agent == "alpha"
        assert agent_b.trace_writer._source_agent == "beta"

    def test_pool_session_stores_pool_agent_name(self, tmp_path):
        rt = _make_runtime_with_mock(tmp_path)
        pool = AgentPool(rt)
        session = pool.add("solo")

        assert session._pool_agent_name == "solo"
