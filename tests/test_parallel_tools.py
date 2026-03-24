"""Tests for parallel (concurrent) tool execution via ToolGateway.call_many_concurrent."""

from __future__ import annotations

import asyncio
import time

import pytest

from arcana.contracts.tool import (
    SideEffect,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from arcana.tool_gateway.base import ToolProvider
from arcana.tool_gateway.gateway import ToolGateway
from arcana.tool_gateway.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Test tool providers
# ---------------------------------------------------------------------------


class SlowEchoTool(ToolProvider):
    """Tool that sleeps before returning -- used to verify concurrency."""

    def __init__(self, delay: float = 0.2) -> None:
        self._delay = delay

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="slow_echo",
            description="Echo with delay",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            side_effect=SideEffect.READ,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        await asyncio.sleep(self._delay)
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output=call.arguments.get("message", ""),
        )


class FailingTool(ToolProvider):
    """Tool that always raises an exception."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="failing",
            description="Always fails",
            input_schema={"type": "object", "properties": {}},
            side_effect=SideEffect.READ,
            max_retries=0,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        raise RuntimeError("boom")


class InstantTool(ToolProvider):
    """Tool that returns instantly."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="instant",
            description="Returns immediately",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
            side_effect=SideEffect.READ,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output=call.arguments.get("value", "ok"),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_gateway(*providers: ToolProvider) -> ToolGateway:
    registry = ToolRegistry()
    for p in providers:
        registry.register(p)
    return ToolGateway(registry=registry)


def _make_call(name: str, call_id: str, **kwargs: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCallManyConcurrent:
    """Tests for ToolGateway.call_many_concurrent."""

    @pytest.mark.asyncio
    async def test_concurrent_execution_is_faster_than_sequential(self) -> None:
        """Multiple slow tools should complete in ~1x delay, not Nx delay."""
        delay = 0.2
        n = 4
        gw = _make_gateway(SlowEchoTool(delay=delay))

        calls = [
            _make_call("slow_echo", f"c{i}", message=f"msg-{i}") for i in range(n)
        ]

        start = time.monotonic()
        results = await gw.call_many_concurrent(calls)
        elapsed = time.monotonic() - start

        # All should succeed
        assert len(results) == n
        assert all(r.success for r in results)

        # If truly concurrent: elapsed ~ delay.  Sequential would be ~ n*delay.
        # Allow generous margin but reject sequential timing.
        assert elapsed < delay * (n - 1), (
            f"Expected concurrent execution in ~{delay}s but took {elapsed:.2f}s "
            f"(sequential would be ~{delay * n:.1f}s)"
        )

    @pytest.mark.asyncio
    async def test_result_order_matches_input_order(self) -> None:
        """Results must appear in the same order as the input calls."""
        gw = _make_gateway(SlowEchoTool(delay=0.05))

        calls = [
            _make_call("slow_echo", f"id-{i}", message=f"msg-{i}") for i in range(5)
        ]
        results = await gw.call_many_concurrent(calls)

        for i, result in enumerate(results):
            assert result.tool_call_id == f"id-{i}"
            assert result.output == f"msg-{i}"

    @pytest.mark.asyncio
    async def test_one_failure_does_not_block_others(self) -> None:
        """A failing tool should not prevent other tools from completing."""
        gw = _make_gateway(SlowEchoTool(delay=0.05), FailingTool())

        calls = [
            _make_call("slow_echo", "ok-1", message="hello"),
            _make_call("failing", "fail-1"),
            _make_call("slow_echo", "ok-2", message="world"),
        ]
        results = await gw.call_many_concurrent(calls)

        assert len(results) == 3

        # First and third should succeed
        assert results[0].success is True
        assert results[0].output == "hello"
        assert results[2].success is True
        assert results[2].output == "world"

        # Second should have failed gracefully
        assert results[1].success is False
        assert results[1].error is not None

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self) -> None:
        """Calling with an empty list should return an empty list."""
        gw = _make_gateway(InstantTool())
        results = await gw.call_many_concurrent([])
        assert results == []

    @pytest.mark.asyncio
    async def test_single_call(self) -> None:
        """A single tool call should work fine through the concurrent path."""
        gw = _make_gateway(InstantTool())
        calls = [_make_call("instant", "solo", value="42")]
        results = await gw.call_many_concurrent(calls)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == "42"

    @pytest.mark.asyncio
    async def test_unknown_tool_handled_gracefully(self) -> None:
        """A call to an unregistered tool should return an error result, not crash."""
        gw = _make_gateway(InstantTool())
        calls = [
            _make_call("instant", "good", value="yes"),
            _make_call("nonexistent", "bad"),
        ]
        results = await gw.call_many_concurrent(calls)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert "not found" in results[1].error.message.lower()


class TestCallManyBackwardCompatibility:
    """Verify that the original call_many method still works unchanged."""

    @pytest.mark.asyncio
    async def test_call_many_still_works(self) -> None:
        """The original call_many must remain functional."""
        gw = _make_gateway(InstantTool())
        calls = [
            _make_call("instant", f"id-{i}", value=f"v{i}") for i in range(3)
        ]
        results = await gw.call_many(calls)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.success is True
            assert r.tool_call_id == f"id-{i}"
            assert r.output == f"v{i}"

    @pytest.mark.asyncio
    async def test_call_many_read_write_separation(self) -> None:
        """call_many should still separate read and write tools."""
        from arcana.tool_gateway.base import ToolProvider as _TP

        class WriteCounter(_TP):
            """Write tool that records execution order."""

            call_order: list[str] = []

            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="write_counter",
                    description="Counts writes",
                    input_schema={
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                    },
                    side_effect=SideEffect.WRITE,
                    requires_confirmation=False,
                )

            async def execute(self, call: ToolCall) -> ToolResult:
                WriteCounter.call_order.append(call.arguments.get("label", ""))
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    success=True,
                    output="done",
                )

        WriteCounter.call_order = []

        # Provide a confirmation callback that auto-approves so write tools
        # actually execute (the gateway gates WRITE tools behind confirmation).
        async def _auto_confirm(tc: ToolCall, spec: ToolSpec) -> bool:
            return True

        registry = ToolRegistry()
        registry.register(InstantTool())
        registry.register(WriteCounter())
        gw = ToolGateway(registry=registry, confirmation_callback=_auto_confirm)

        calls = [
            _make_call("instant", "r1", value="a"),
            _make_call("write_counter", "w1", label="first"),
            _make_call("write_counter", "w2", label="second"),
        ]
        results = await gw.call_many(calls)

        assert len(results) == 3
        # Write tools executed sequentially in order
        assert WriteCounter.call_order == ["first", "second"]
