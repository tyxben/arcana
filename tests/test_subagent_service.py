"""Tests for the experimental subagent service (Phase 3 slice).

Covers the load-bearing behaviours of ``arcana.experimental.subagents``:
registration, isolated single-shot asks, no-recursion enforcement,
per-subtask budget resolution, delegation-as-tool, and trace correlation
(``source_agent`` / ``bundle_id`` / ``delegated_by_run_id``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from arcana.contracts.llm import LLMResponse, TokenUsage
from arcana.experimental import (
    SubagentRecursionError,
    SubagentResult,
    SubagentService,
    subagents,
)
from arcana.experimental.subagents import _DELEGATION_ACTIVE
from arcana.runtime_core import Budget, Runtime, RuntimeConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_response(text: str, pt: int = 10, ct: int = 20) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=None,
        usage=TokenUsage(
            prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct
        ),
        model="test-model",
        finish_reason="stop",
    )


def _make_runtime(trace_writer=None) -> Runtime:
    rt = Runtime(
        providers={"ollama": ""},
        config=RuntimeConfig(default_provider="ollama"),
    )
    if trace_writer is not None:
        rt._trace_writer = trace_writer
    return rt


def _mock_generate(rt: Runtime, *responses: LLMResponse) -> AsyncMock:
    """Wire the gateway to return the given responses, bypassing streaming."""
    gen = AsyncMock(side_effect=list(responses) or [_text_response("ok")])
    rt._gateway.generate = gen
    rt._gateway.stream = MagicMock(side_effect=NotImplementedError)
    return gen


class _CaptureWriter:
    """Trace writer that records every event written to it."""

    def __init__(self) -> None:
        self.events: list = []

    def write(self, event) -> None:
        self.events.append(event)

    def __getattr__(self, name):  # tolerate flush/close/etc.
        return lambda *a, **k: None


# ---------------------------------------------------------------------------
# SubagentResult contract
# ---------------------------------------------------------------------------


class TestSubagentResult:
    def test_defaults(self):
        r = SubagentResult(agent="x")
        assert r.agent == "x"
        assert r.content == ""
        assert r.run_id == ""
        assert r.tokens == 0
        assert r.cost == 0.0
        assert r.trace_refs == []

    def test_populated(self):
        r = SubagentResult(
            agent="researcher",
            content="done",
            run_id="run-1",
            tokens=42,
            cost=0.01,
            trace_refs=["run-1"],
        )
        assert r.content == "done"
        assert r.run_id == "run-1"
        assert r.tokens == 42
        assert r.trace_refs == ["run-1"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_add_and_names(self):
        svc = subagents(_make_runtime())
        svc.add("a", system="A")
        svc.add("b", system="B")
        assert svc.names == ["a", "b"]

    def test_duplicate_name_raises(self):
        svc = subagents(_make_runtime())
        svc.add("a")
        with pytest.raises(ValueError, match="already registered"):
            svc.add("a")

    def test_bundle_id_stable_and_unique(self):
        s1 = subagents(_make_runtime())
        s2 = subagents(_make_runtime())
        assert s1.bundle_id == s1.bundle_id
        assert s1.bundle_id != s2.bundle_id

    def test_explicit_bundle_id(self):
        svc = subagents(_make_runtime(), bundle_id="my-bundle")
        assert svc.bundle_id == "my-bundle"

    def test_factory_returns_service(self):
        assert isinstance(subagents(_make_runtime()), SubagentService)


# ---------------------------------------------------------------------------
# ask() execution
# ---------------------------------------------------------------------------


class TestAsk:
    @pytest.mark.asyncio
    async def test_unknown_name_raises(self):
        svc = subagents(_make_runtime())
        with pytest.raises(KeyError, match="Unknown subagent"):
            await svc.ask("nope", "task")

    @pytest.mark.asyncio
    async def test_returns_populated_result(self):
        rt = _make_runtime()
        _mock_generate(rt, _text_response("the answer", pt=10, ct=20))
        svc = subagents(rt)
        svc.add("researcher", system="You research.")

        res = await svc.ask("researcher", "find X")

        assert isinstance(res, SubagentResult)
        assert res.agent == "researcher"
        assert res.content == "the answer"
        assert res.tokens == 30
        assert res.cost > 0
        assert res.run_id
        assert res.trace_refs == [res.run_id]

    @pytest.mark.asyncio
    async def test_context_injected_into_message(self):
        rt = _make_runtime()
        gen = _mock_generate(rt, _text_response("ok"))
        svc = subagents(rt)
        svc.add("a", system="A")

        await svc.ask("a", "do it", context={"prior": "value"})

        # The composed user message should carry the context block.
        sent = str(gen.call_args_list)
        assert "<context>" in sent
        assert "prior" in sent


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


class TestIsolation:
    @pytest.mark.asyncio
    async def test_asks_do_not_share_history(self):
        rt = _make_runtime()
        gen = _mock_generate(
            rt,
            _text_response("first answer"),
            _text_response("second answer"),
        )
        svc = subagents(rt)
        svc.add("a", system="A")

        r1 = await svc.ask("a", "FIRST_TASK")
        r2 = await svc.ask("a", "SECOND_TASK")

        # Distinct isolated runs.
        assert r1.run_id != r2.run_id
        # The second ask's request must not contain the first task -- a fresh
        # session is built per ask, so history does not accumulate.
        second_call = str(gen.call_args_list[1])
        assert "FIRST_TASK" not in second_call
        assert "SECOND_TASK" in second_call


# ---------------------------------------------------------------------------
# No recursion (v1)
# ---------------------------------------------------------------------------


class TestNoRecursion:
    @pytest.mark.asyncio
    async def test_ask_inside_delegation_raises(self):
        svc = subagents(_make_runtime())
        svc.add("a")
        token = _DELEGATION_ACTIVE.set(True)
        try:
            with pytest.raises(SubagentRecursionError):
                await svc.ask("a", "task")
        finally:
            _DELEGATION_ACTIVE.reset(token)

    @pytest.mark.asyncio
    async def test_delegate_tool_refuses_when_active(self):
        svc = subagents(_make_runtime())
        svc.add("a")
        delegate = svc.as_tool("a")

        token = _DELEGATION_ACTIVE.set(True)
        try:
            out = await delegate._fn("task")
        finally:
            _DELEGATION_ACTIVE.reset(token)

        assert "refused" in out.lower()
        assert "recursive" in out.lower()

    @pytest.mark.asyncio
    async def test_flag_cleared_after_ask(self):
        rt = _make_runtime()
        _mock_generate(rt, _text_response("ok"))
        svc = subagents(rt)
        svc.add("a")
        await svc.ask("a", "task")
        # The context flag must be reset once the ask completes.
        assert _DELEGATION_ACTIVE.get() is False


# ---------------------------------------------------------------------------
# Per-subtask budget resolution
# ---------------------------------------------------------------------------


class TestBudget:
    def test_no_budget_anywhere_yields_none(self):
        svc = subagents(_make_runtime())
        svc.add("a")
        assert svc._tracker_for(svc._configs["a"]) is None

    def test_service_budget_shared_across_asks(self):
        svc = subagents(_make_runtime(), budget=Budget(max_cost_usd=1.0))
        svc.add("a")
        t1 = svc._tracker_for(svc._configs["a"])
        t2 = svc._tracker_for(svc._configs["a"])
        # Same shared tracker instance -> accumulates across asks.
        assert t1 is t2
        assert t1.max_cost_usd == 1.0

    def test_per_subagent_budget_is_fresh_per_ask(self):
        svc = subagents(_make_runtime(), budget=Budget(max_cost_usd=1.0))
        svc.add("a", budget=Budget(max_cost_usd=0.25, max_tokens=500))
        t1 = svc._tracker_for(svc._configs["a"])
        t2 = svc._tracker_for(svc._configs["a"])
        # Fresh isolated cap each ask.
        assert t1 is not t2
        assert t1.max_cost_usd == 0.25
        assert t1.max_tokens == 500


# ---------------------------------------------------------------------------
# Delegation as a tool
# ---------------------------------------------------------------------------


class TestAsTool:
    def test_unknown_name_raises(self):
        svc = subagents(_make_runtime())
        with pytest.raises(KeyError):
            svc.as_tool("nope")

    def test_default_spec(self):
        svc = subagents(_make_runtime())
        svc.add("researcher")
        t = svc.as_tool("researcher")
        assert t._spec.name == "ask_researcher"
        # Conservative side-effect: delegation is not a pure read.
        assert t._spec.side_effect.value == "write"

    def test_custom_name_and_side_effect(self):
        svc = subagents(_make_runtime())
        svc.add("worker")
        t = svc.as_tool("worker", tool_name="delegate", side_effect="read")
        assert t._spec.name == "delegate"
        assert t._spec.side_effect.value == "read"

    @pytest.mark.asyncio
    async def test_delegate_runs_ask(self):
        rt = _make_runtime()
        _mock_generate(rt, _text_response("delegated answer"))
        svc = subagents(rt)
        svc.add("researcher", system="You research.")
        delegate = svc.as_tool("researcher")

        out = await delegate._fn("find something")
        assert out == "delegated answer"


# ---------------------------------------------------------------------------
# Trace correlation
# ---------------------------------------------------------------------------


class TestTraceCorrelation:
    @pytest.mark.asyncio
    async def test_events_stamped_with_source_agent_and_bundle(self):
        cap = _CaptureWriter()
        rt = _make_runtime(trace_writer=cap)
        _mock_generate(rt, _text_response("answer"))
        svc = subagents(rt, bundle_id="bundle-test")
        svc.add("researcher", system="You research.")

        await svc.ask("researcher", "task")

        assert cap.events, "expected at least one trace event"
        for evt in cap.events:
            assert evt.metadata.get("source_agent") == "researcher"
            assert evt.metadata.get("bundle_id") == "bundle-test"

    @pytest.mark.asyncio
    async def test_delegated_by_run_id_stamped_when_provided(self):
        cap = _CaptureWriter()
        rt = _make_runtime(trace_writer=cap)
        _mock_generate(rt, _text_response("answer"))
        svc = subagents(rt)
        svc.add("a", system="A")

        await svc.ask("a", "task", delegated_by_run_id="parent-run-99")

        assert cap.events
        assert all(
            evt.metadata.get("delegated_by_run_id") == "parent-run-99"
            for evt in cap.events
        )

    @pytest.mark.asyncio
    async def test_no_delegated_by_run_id_key_when_absent(self):
        cap = _CaptureWriter()
        rt = _make_runtime(trace_writer=cap)
        _mock_generate(rt, _text_response("answer"))
        svc = subagents(rt)
        svc.add("a", system="A")

        await svc.ask("a", "task")

        assert cap.events
        # Key is omitted entirely (not stamped empty) when not supplied.
        assert all(
            "delegated_by_run_id" not in evt.metadata for evt in cap.events
        )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_async_context_manager_closes(self):
        rt = _make_runtime()
        async with subagents(rt) as svc:
            svc.add("a")
            assert svc.names == ["a"]
        # close() drops registrations.
        assert svc.names == []
