"""Tests for Runtime event hooks (_EventBus, on/off, event emission)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arcana.runtime_core import RunResult, Runtime, _EventBus

# ---------------------------------------------------------------------------
# _EventBus unit tests
# ---------------------------------------------------------------------------


class TestEventBus:
    """Low-level tests for _EventBus pub/sub."""

    def test_on_registers_callback(self) -> None:
        bus = _EventBus()
        cb = MagicMock()
        bus.on("test", cb)
        assert cb in bus._listeners["test"]

    def test_off_removes_callback(self) -> None:
        bus = _EventBus()
        cb = MagicMock()
        bus.on("test", cb)
        bus.off("test", cb)
        assert cb not in bus._listeners["test"]

    def test_off_noop_when_not_registered(self) -> None:
        bus = _EventBus()
        cb = MagicMock()
        # Should not raise
        bus.off("test", cb)

    @pytest.mark.asyncio
    async def test_emit_calls_sync_callback(self) -> None:
        bus = _EventBus()
        cb = MagicMock()
        bus.on("ping", cb)
        await bus.emit("ping", data="hello")
        cb.assert_called_once_with(data="hello")

    @pytest.mark.asyncio
    async def test_emit_calls_async_callback(self) -> None:
        bus = _EventBus()
        received: dict[str, Any] = {}

        async def async_cb(**kwargs: Any) -> None:
            received.update(kwargs)

        bus.on("ping", async_cb)
        await bus.emit("ping", value=42)
        assert received == {"value": 42}

    @pytest.mark.asyncio
    async def test_emit_calls_multiple_callbacks(self) -> None:
        bus = _EventBus()
        calls: list[str] = []
        bus.on("ev", lambda **kw: calls.append("a"))
        bus.on("ev", lambda **kw: calls.append("b"))
        await bus.emit("ev")
        assert calls == ["a", "b"]

    @pytest.mark.asyncio
    async def test_emit_noop_for_unknown_event(self) -> None:
        bus = _EventBus()
        # Should not raise
        await bus.emit("nonexistent", x=1)


# ---------------------------------------------------------------------------
# Runtime.on / .off API tests
# ---------------------------------------------------------------------------


class TestRuntimeOnOff:
    """Tests for the public on()/off() API on Runtime."""

    def _make_runtime(self) -> Runtime:
        """Create a minimal Runtime with no providers (for unit testing hooks)."""
        return Runtime(providers={})

    def test_on_returns_self_for_chaining(self) -> None:
        rt = self._make_runtime()
        cb = MagicMock()
        result = rt.on("run_start", cb)
        assert result is rt

    def test_off_returns_self_for_chaining(self) -> None:
        rt = self._make_runtime()
        cb = MagicMock()
        rt.on("run_start", cb)
        result = rt.off("run_start", cb)
        assert result is rt

    def test_chaining_multiple_on(self) -> None:
        rt = self._make_runtime()
        cb1 = MagicMock()
        cb2 = MagicMock()
        result = rt.on("run_start", cb1).on("run_end", cb2)
        assert result is rt
        assert cb1 in rt._events._listeners["run_start"]
        assert cb2 in rt._events._listeners["run_end"]


# ---------------------------------------------------------------------------
# Event emission during run()
# ---------------------------------------------------------------------------


class TestRuntimeEventEmission:
    """Tests that events fire during Runtime.run()."""

    def _make_runtime(self) -> Runtime:
        return Runtime(providers={})

    @pytest.mark.asyncio
    async def test_run_emits_start_and_end(self) -> None:
        rt = self._make_runtime()
        events: list[tuple[str, dict[str, Any]]] = []

        def on_start(**kw: Any) -> None:
            events.append(("run_start", kw))

        def on_end(**kw: Any) -> None:
            events.append(("run_end", kw))

        rt.on("run_start", on_start).on("run_end", on_end)

        fake_result = RunResult(
            output="done", success=True, steps=1,
            tokens_used=10, cost_usd=0.001, run_id="test-123",
        )

        with patch.object(
            rt, "_create_session", return_value=MagicMock(
                run=AsyncMock(return_value=fake_result),
                run_id="test-123",
            ),
        ):
            result = await rt.run("Hello")

        assert result.success
        assert len(events) == 2

        start_ev = events[0]
        assert start_ev[0] == "run_start"
        assert start_ev[1]["run_id"] == "test-123"
        assert start_ev[1]["goal"] == "Hello"

        end_ev = events[1]
        assert end_ev[0] == "run_end"
        assert end_ev[1]["run_id"] == "test-123"
        assert end_ev[1]["result"] is result

    @pytest.mark.asyncio
    async def test_run_emits_error_on_exception(self) -> None:
        rt = self._make_runtime()
        errors: list[dict[str, Any]] = []

        def on_error(**kw: Any) -> None:
            errors.append(kw)

        rt.on("error", on_error)

        boom = RuntimeError("boom")

        with patch.object(
            rt, "_create_session", return_value=MagicMock(
                run=AsyncMock(side_effect=boom),
                run_id="err-run",
            ),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await rt.run("fail")

        assert len(errors) == 1
        assert errors[0]["run_id"] == "err-run"
        assert errors[0]["error"] is boom

    @pytest.mark.asyncio
    async def test_async_callbacks_work_during_run(self) -> None:
        rt = self._make_runtime()
        received: list[str] = []

        async def async_on_start(**kw: Any) -> None:
            received.append(kw["goal"])

        rt.on("run_start", async_on_start)

        fake_result = RunResult(
            output="ok", success=True, run_id="async-test",
        )

        with patch.object(
            rt, "_create_session", return_value=MagicMock(
                run=AsyncMock(return_value=fake_result),
                run_id="async-test",
            ),
        ):
            await rt.run("async goal")

        assert received == ["async goal"]

    @pytest.mark.asyncio
    async def test_no_error_event_on_success(self) -> None:
        rt = self._make_runtime()
        errors: list[Any] = []

        rt.on("error", lambda **kw: errors.append(kw))

        fake_result = RunResult(
            output="ok", success=True, run_id="no-err",
        )

        with patch.object(
            rt, "_create_session", return_value=MagicMock(
                run=AsyncMock(return_value=fake_result),
                run_id="no-err",
            ),
        ):
            await rt.run("succeed")

        assert errors == []

    @pytest.mark.asyncio
    async def test_off_prevents_callback_from_firing(self) -> None:
        rt = self._make_runtime()
        calls: list[str] = []

        def cb(**kw: Any) -> None:
            calls.append("fired")

        rt.on("run_start", cb)
        rt.off("run_start", cb)

        fake_result = RunResult(
            output="ok", success=True, run_id="off-test",
        )

        with patch.object(
            rt, "_create_session", return_value=MagicMock(
                run=AsyncMock(return_value=fake_result),
                run_id="off-test",
            ),
        ):
            await rt.run("test")

        assert calls == []
