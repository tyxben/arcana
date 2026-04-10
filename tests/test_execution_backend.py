"""Tests for ExecutionBackend abstraction."""

from __future__ import annotations

import pytest

from arcana.contracts.tool import (
    SideEffect,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from arcana.tool_gateway.base import ToolProvider
from arcana.tool_gateway.execution_backend import ExecutionBackend, InProcessBackend
from arcana.tool_gateway.gateway import ToolGateway
from arcana.tool_gateway.registry import ToolRegistry

# ── Helpers ─────────────────────────────────────────────────────────


class EchoProvider(ToolProvider):
    """Minimal provider for backend tests."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echo input",
            input_schema={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
            side_effect=SideEffect.NONE,
            capabilities=[],
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output=call.arguments.get("msg", ""),
        )


def _make_call(msg: str = "hello") -> ToolCall:
    return ToolCall(id="call-1", name="echo", arguments={"msg": msg})


class TrackingBackend:
    """Custom backend that records calls and delegates to provider."""

    def __init__(self) -> None:
        self.calls: list[tuple[ToolProvider, ToolCall]] = []

    async def execute(self, provider: ToolProvider, call: ToolCall) -> ToolResult:
        self.calls.append((provider, call))
        return await provider.execute(call)

    async def cleanup(self) -> None:
        pass


class TransformingBackend:
    """Backend that wraps the result output with a prefix."""

    async def execute(self, provider: ToolProvider, call: ToolCall) -> ToolResult:
        result = await provider.execute(call)
        result.output = f"[wrapped] {result.output}"
        return result

    async def cleanup(self) -> None:
        pass


# ── InProcessBackend tests ──────────────────────────────────────────


class TestInProcessBackend:
    @pytest.mark.asyncio
    async def test_delegates_to_provider(self) -> None:
        backend = InProcessBackend()
        provider = EchoProvider()
        call = _make_call("world")

        result = await backend.execute(provider, call)

        assert result.success is True
        assert result.output == "world"
        assert result.name == "echo"
        assert result.tool_call_id == "call-1"

    @pytest.mark.asyncio
    async def test_cleanup_is_noop(self) -> None:
        backend = InProcessBackend()
        # Should not raise
        await backend.cleanup()

    def test_satisfies_protocol(self) -> None:
        assert isinstance(InProcessBackend(), ExecutionBackend)


# ── ToolGateway + backend integration ───────────────────────────────


class TestToolGatewayBackendIntegration:
    def _build_gateway(
        self, backend: ExecutionBackend | None = None
    ) -> ToolGateway:
        registry = ToolRegistry()
        registry.register(EchoProvider())
        return ToolGateway(registry=registry, backend=backend)

    @pytest.mark.asyncio
    async def test_gateway_uses_backend_for_execution(self) -> None:
        tracking = TrackingBackend()
        gw = self._build_gateway(backend=tracking)

        call = _make_call("via-backend")
        result = await gw.call(call)

        assert result.success is True
        assert result.output == "via-backend"
        assert len(tracking.calls) == 1
        assert tracking.calls[0][1].arguments["msg"] == "via-backend"

    @pytest.mark.asyncio
    async def test_gateway_defaults_to_in_process_backend(self) -> None:
        gw = self._build_gateway(backend=None)
        assert isinstance(gw.backend, InProcessBackend)

        call = _make_call("default")
        result = await gw.call(call)

        assert result.success is True
        assert result.output == "default"

    @pytest.mark.asyncio
    async def test_custom_backend_can_transform_result(self) -> None:
        gw = self._build_gateway(backend=TransformingBackend())

        call = _make_call("data")
        result = await gw.call(call)

        assert result.success is True
        assert result.output == "[wrapped] data"

    @pytest.mark.asyncio
    async def test_custom_backend_satisfies_protocol(self) -> None:
        assert isinstance(TrackingBackend(), ExecutionBackend)
        assert isinstance(TransformingBackend(), ExecutionBackend)

    @pytest.mark.asyncio
    async def test_gateway_close_calls_backend_cleanup(self) -> None:
        """ToolGateway.close() must invoke backend.cleanup()."""

        class CleanupTracker:
            def __init__(self) -> None:
                self.cleaned = False

            async def execute(self, provider: ToolProvider, call: ToolCall) -> ToolResult:
                return await provider.execute(call)

            async def cleanup(self) -> None:
                self.cleaned = True

        tracker = CleanupTracker()
        gw = self._build_gateway(backend=tracker)

        assert not tracker.cleaned
        await gw.close()
        assert tracker.cleaned

    @pytest.mark.asyncio
    async def test_backend_exception_propagates_through_gateway(self) -> None:
        """Backend raising an exception is handled by gateway retry logic."""

        class FailingBackend:
            async def execute(self, provider: ToolProvider, call: ToolCall) -> ToolResult:
                raise RuntimeError("backend crashed")

            async def cleanup(self) -> None:
                pass

        gw = self._build_gateway(backend=FailingBackend())
        call = _make_call("boom")
        result = await gw.call(call)

        # Gateway catches unexpected errors and returns a ToolResult with error
        assert result.success is False
        assert "backend crashed" in result.error.message

    @pytest.mark.asyncio
    async def test_cleanup_exception_propagates(self) -> None:
        """If backend.cleanup() raises, gateway.close() propagates it."""

        class BadCleanup:
            async def execute(self, provider: ToolProvider, call: ToolCall) -> ToolResult:
                return await provider.execute(call)

            async def cleanup(self) -> None:
                raise OSError("cleanup failed")

        gw = self._build_gateway(backend=BadCleanup())
        with pytest.raises(OSError, match="cleanup failed"):
            await gw.close()

    @pytest.mark.asyncio
    async def test_backend_used_in_call_many_concurrent(self) -> None:
        """call_many_concurrent routes each call through the backend."""
        tracking = TrackingBackend()
        gw = self._build_gateway(backend=tracking)

        calls = [_make_call("a"), ToolCall(id="call-2", name="echo", arguments={"msg": "b"})]
        results = await gw.call_many_concurrent(calls)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert len(tracking.calls) == 2
        msgs = [c[1].arguments["msg"] for c in tracking.calls]
        assert sorted(msgs) == ["a", "b"]
