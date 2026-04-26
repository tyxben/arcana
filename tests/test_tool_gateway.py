"""Unit tests for Tool Gateway."""

import asyncio

import pytest

from arcana.contracts.tool import (
    SideEffect,
    ToolCall,
    ToolError,
    ToolErrorCategory,
    ToolResult,
    ToolSpec,
)
from arcana.contracts.trace import TraceContext
from arcana.tool_gateway.base import ToolProvider
from arcana.tool_gateway.formatter import format_tool_for_llm, format_tool_list_for_llm
from arcana.tool_gateway.gateway import ToolGateway
from arcana.tool_gateway.lazy_registry import KeywordToolMatcher, LazyToolRegistry
from arcana.tool_gateway.registry import ToolRegistry
from arcana.tool_gateway.validators import validate_arguments
from arcana.trace.writer import TraceWriter

# ── Test Tool Providers ──────────────────────────────────────────


class EchoTool(ToolProvider):
    """Tool that echoes its input."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echo input back",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            side_effect=SideEffect.NONE,
            capabilities=[],
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output=call.arguments.get("message", ""),
        )


class WriteTool(ToolProvider):
    """Tool with WRITE side effect requiring capabilities."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_write",
            description="Write a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            side_effect=SideEffect.WRITE,
            requires_confirmation=True,
            capabilities=["fs:write", "fs:read"],
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output="written",
        )


class FailingTool(ToolProvider):
    """Tool that fails with retryable errors, then succeeds."""

    def __init__(self, fail_count: int = 1) -> None:
        self._fail_count = fail_count
        self._attempt = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="flaky",
            description="Flaky tool",
            input_schema={"type": "object", "properties": {}},
            side_effect=SideEffect.READ,
            capabilities=[],
            max_retries=3,
            retry_delay_ms=1,  # Fast retries for tests
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        self._attempt += 1
        if self._attempt <= self._fail_count:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=ToolError(
                    category=ToolErrorCategory.TRANSPORT,
                    message="Temporary failure",
                ),
            )
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output="ok",
        )


class NonRetryableFailTool(ToolProvider):
    """Tool that fails with non-retryable error."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="hard_fail",
            description="Always fails permanently",
            input_schema={"type": "object", "properties": {}},
            side_effect=SideEffect.READ,
            capabilities=[],
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=False,
            error=ToolError(
                category=ToolErrorCategory.LOGIC,
                message="Permanent failure",
                code="FATAL",
            ),
        )


class SlowTool(ToolProvider):
    """Tool that takes too long (for timeout tests)."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="slow",
            description="Slow tool",
            input_schema={"type": "object", "properties": {}},
            side_effect=SideEffect.READ,
            capabilities=[],
            timeout_ms=50,
            max_retries=0,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        await asyncio.sleep(1)  # Sleep longer than timeout
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output="done",
        )


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def registry():
    """Create a registry with test tools."""
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(WriteTool())
    return reg


@pytest.fixture
def trace_ctx():
    return TraceContext(run_id="test-run")


def _make_call(name: str = "echo", **kwargs) -> ToolCall:
    """Helper to create a ToolCall."""
    defaults = {
        "id": "call-1",
        "name": name,
        "arguments": {"message": "hello"},
    }
    defaults.update(kwargs)
    return ToolCall(**defaults)


# ── TestToolRegistry ─────────────────────────────────────────────


class TestToolRegistry:
    """Test the tool registry."""

    def test_register_and_get(self, registry):
        provider = registry.get("echo")
        assert provider is not None
        assert provider.spec.name == "echo"

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    def test_unregister(self, registry):
        assert registry.unregister("echo")
        assert registry.get("echo") is None

    def test_unregister_nonexistent(self, registry):
        assert not registry.unregister("nonexistent")

    def test_list_tools(self, registry):
        tools = registry.list_tools()
        assert "echo" in tools
        assert "file_write" in tools

    def test_get_specs(self, registry):
        specs = registry.get_specs()
        assert len(specs) == 2
        names = [s.name for s in specs]
        assert "echo" in names

    def test_to_openai_tools(self, registry):
        tools = registry.to_openai_tools()
        assert len(tools) == 2
        echo_tool = next(t for t in tools if t["function"]["name"] == "echo")
        assert echo_tool["type"] == "function"
        assert "description" in echo_tool["function"]
        assert "parameters" in echo_tool["function"]


# ── TestAuthorization ────────────────────────────────────────────


class TestAuthorization:
    """Test capability-based authorization."""

    async def test_no_capabilities_required(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry, granted_capabilities=set())
        result = await gw.call(_make_call("echo"), trace_ctx=trace_ctx)
        assert result.success

    async def test_authorized_call(self, registry, trace_ctx):
        gw = ToolGateway(
            registry=registry,
            granted_capabilities={"fs:write", "fs:read"},
            confirmation_callback=_auto_confirm,
        )
        call = _make_call("file_write", arguments={"path": "a.txt", "content": "hi"})
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert result.success

    async def test_missing_capability_rejected(self, registry, trace_ctx):
        gw = ToolGateway(
            registry=registry,
            granted_capabilities={"fs:read"},  # Missing fs:write
        )
        call = _make_call("file_write", arguments={"path": "a.txt", "content": "hi"})
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert not result.success
        assert result.error is not None
        assert result.error.code == "UNAUTHORIZED"

    async def test_unauthorized_write_is_audited(self, registry, tmp_path, trace_ctx):
        trace_writer = TraceWriter(trace_dir=tmp_path)
        gw = ToolGateway(
            registry=registry,
            trace_writer=trace_writer,
            granted_capabilities=set(),  # No capabilities
        )
        call = _make_call("file_write", arguments={"path": "x", "content": "y"})
        result = await gw.call(call, trace_ctx=trace_ctx)

        assert not result.success
        # Check that trace event was written
        trace_files = list(tmp_path.glob("*.jsonl"))
        assert len(trace_files) > 0


# ── TestValidation ───────────────────────────────────────────────


class TestValidation:
    """Test argument validation."""

    def test_valid_arguments(self):
        spec = ToolSpec(
            name="t",
            description="t",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            },
        )
        assert validate_arguments(spec, {"name": "Alice", "age": 30}) is None

    def test_missing_required_field(self):
        spec = ToolSpec(
            name="t",
            description="t",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        error = validate_arguments(spec, {})
        assert error is not None
        assert error.code == "VALIDATION_ERROR"

    def test_wrong_type(self):
        spec = ToolSpec(
            name="t",
            description="t",
            input_schema={
                "type": "object",
                "properties": {"age": {"type": "integer"}},
            },
        )
        error = validate_arguments(spec, {"age": "not_a_number"})
        assert error is not None
        assert "integer" in error.message

    def test_empty_schema_always_valid(self):
        spec = ToolSpec(name="t", description="t", input_schema={})
        assert validate_arguments(spec, {"anything": "goes"}) is None

    async def test_validation_failure_in_gateway(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        call = _make_call("echo", arguments={})  # Missing required 'message'
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert not result.success
        assert result.error is not None
        assert result.error.code == "VALIDATION_ERROR"


# ── TestIdempotency ──────────────────────────────────────────────


class TestIdempotency:
    """Test idempotency cache."""

    async def test_same_key_returns_cached(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        call1 = _make_call("echo", idempotency_key="key-1")
        call2 = _make_call("echo", id="call-2", idempotency_key="key-1")

        result1 = await gw.call(call1, trace_ctx=trace_ctx)
        result2 = await gw.call(call2, trace_ctx=trace_ctx)

        assert result1.success
        assert result2.success
        # Should be the same cached result
        assert result2.tool_call_id == result1.tool_call_id

    async def test_no_key_not_cached(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        call1 = _make_call("echo")
        call2 = _make_call("echo", id="call-2")

        result1 = await gw.call(call1, trace_ctx=trace_ctx)
        result2 = await gw.call(call2, trace_ctx=trace_ctx)

        assert result1.success
        assert result2.success
        # Different executions
        assert result2.tool_call_id == "call-2"

    async def test_different_keys_execute_separately(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        call1 = _make_call("echo", idempotency_key="key-1")
        call2 = _make_call("echo", id="call-2", idempotency_key="key-2")

        await gw.call(call1, trace_ctx=trace_ctx)
        result2 = await gw.call(call2, trace_ctx=trace_ctx)

        assert result2.tool_call_id == "call-2"

    async def test_cache_bounded_evicts_oldest(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry, idempotency_cache_limit=2)

        await gw.call(_make_call("echo", id="c1", idempotency_key="k1"), trace_ctx=trace_ctx)
        await gw.call(_make_call("echo", id="c2", idempotency_key="k2"), trace_ctx=trace_ctx)
        await gw.call(_make_call("echo", id="c3", idempotency_key="k3"), trace_ctx=trace_ctx)

        assert len(gw._idempotency_cache) == 2
        assert "k1" not in gw._idempotency_cache
        assert "k2" in gw._idempotency_cache
        assert "k3" in gw._idempotency_cache

    async def test_cache_hit_refreshes_lru(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry, idempotency_cache_limit=2)

        await gw.call(_make_call("echo", id="c1", idempotency_key="k1"), trace_ctx=trace_ctx)
        await gw.call(_make_call("echo", id="c2", idempotency_key="k2"), trace_ctx=trace_ctx)

        # Hit k1 — marks it MRU so k2 becomes the oldest.
        hit = await gw.call(
            _make_call("echo", id="c1-hit", idempotency_key="k1"),
            trace_ctx=trace_ctx,
        )
        assert hit.tool_call_id == "c1"

        await gw.call(_make_call("echo", id="c3", idempotency_key="k3"), trace_ctx=trace_ctx)

        assert len(gw._idempotency_cache) == 2
        assert "k1" in gw._idempotency_cache
        assert "k2" not in gw._idempotency_cache
        assert "k3" in gw._idempotency_cache

    async def test_cache_limit_zero_disables_dedup(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry, idempotency_cache_limit=0)

        call1 = _make_call("echo", id="c1", idempotency_key="k1")
        call2 = _make_call("echo", id="c2", idempotency_key="k1")

        result1 = await gw.call(call1, trace_ctx=trace_ctx)
        result2 = await gw.call(call2, trace_ctx=trace_ctx)

        # With limit=0 every insert is immediately evicted; second call must
        # re-execute against its own ToolCall, not return the first result.
        assert result1.tool_call_id == "c1"
        assert result2.tool_call_id == "c2"
        assert len(gw._idempotency_cache) == 0

    async def test_cache_limit_negative_raises(self, registry):
        with pytest.raises(ValueError, match="idempotency_cache_limit"):
            ToolGateway(registry=registry, idempotency_cache_limit=-1)

    async def test_cache_limit_none_is_unbounded(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry, idempotency_cache_limit=None)

        for i in range(100):
            await gw.call(
                _make_call("echo", id=f"c{i}", idempotency_key=f"k{i}"),
                trace_ctx=trace_ctx,
            )

        assert len(gw._idempotency_cache) == 100

    async def test_close_clears_idempotency_cache(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        await gw.call(
            _make_call("echo", id="c1", idempotency_key="k1"),
            trace_ctx=trace_ctx,
        )
        assert len(gw._idempotency_cache) == 1

        await gw.close()

        assert len(gw._idempotency_cache) == 0


# ── TestConfirmation ─────────────────────────────────────────────


async def _auto_confirm(call: ToolCall, spec: ToolSpec) -> bool:
    return True


async def _auto_reject(call: ToolCall, spec: ToolSpec) -> bool:
    return False


class TestConfirmation:
    """Test write confirmation gate."""

    async def test_read_tool_auto_confirmed(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)  # No callback needed for reads
        result = await gw.call(_make_call("echo"), trace_ctx=trace_ctx)
        assert result.success

    async def test_write_tool_no_callback_returns_error(self, registry, trace_ctx):
        gw = ToolGateway(
            registry=registry,
            granted_capabilities={"fs:write", "fs:read"},
            # No confirmation_callback
        )
        call = _make_call("file_write", arguments={"path": "a", "content": "b"})
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert not result.success
        assert result.error is not None
        assert result.error.code == "CONFIRMATION_REQUIRED"

    async def test_write_tool_with_confirm_callback(self, registry, trace_ctx):
        gw = ToolGateway(
            registry=registry,
            granted_capabilities={"fs:write", "fs:read"},
            confirmation_callback=_auto_confirm,
        )
        call = _make_call("file_write", arguments={"path": "a", "content": "b"})
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert result.success

    async def test_write_tool_rejected_by_callback(self, registry, trace_ctx):
        gw = ToolGateway(
            registry=registry,
            granted_capabilities={"fs:write", "fs:read"},
            confirmation_callback=_auto_reject,
        )
        call = _make_call("file_write", arguments={"path": "a", "content": "b"})
        result = await gw.call(call, trace_ctx=trace_ctx)
        assert not result.success
        assert result.error is not None
        assert result.error.code == "CONFIRMATION_REJECTED"


# ── TestRetry ────────────────────────────────────────────────────


class TestRetry:
    """Test retry behavior."""

    async def test_retryable_error_retries_and_succeeds(self, trace_ctx):
        reg = ToolRegistry()
        reg.register(FailingTool(fail_count=2))

        gw = ToolGateway(registry=reg)
        call = _make_call("flaky", arguments={})
        result = await gw.call(call, trace_ctx=trace_ctx)

        assert result.success
        assert result.retry_count == 2

    async def test_non_retryable_error_fails_immediately(self, trace_ctx):
        reg = ToolRegistry()
        reg.register(NonRetryableFailTool())

        gw = ToolGateway(registry=reg)
        call = _make_call("hard_fail", arguments={})
        result = await gw.call(call, trace_ctx=trace_ctx)

        assert not result.success
        assert result.error is not None
        assert result.error.code == "FATAL"
        assert result.retry_count == 0

    async def test_max_retries_exhausted(self, trace_ctx):
        reg = ToolRegistry()
        reg.register(FailingTool(fail_count=100))  # Always fails

        gw = ToolGateway(registry=reg)
        call = _make_call("flaky", arguments={})
        result = await gw.call(call, trace_ctx=trace_ctx)

        assert not result.success
        assert result.retry_count == 3  # max_retries from spec

    async def test_timeout_returns_error(self, trace_ctx):
        reg = ToolRegistry()
        reg.register(SlowTool())

        gw = ToolGateway(registry=reg)
        call = _make_call("slow", arguments={})
        result = await gw.call(call, trace_ctx=trace_ctx)

        assert not result.success
        assert result.error is not None
        assert result.error.code == "TIMEOUT"


# ── TestTracing ──────────────────────────────────────────────────


class TestTracing:
    """Test audit/trace logging."""

    async def test_successful_call_logged(self, registry, tmp_path, trace_ctx):
        trace_writer = TraceWriter(trace_dir=tmp_path)
        gw = ToolGateway(registry=registry, trace_writer=trace_writer)

        await gw.call(_make_call("echo"), trace_ctx=trace_ctx)

        trace_files = list(tmp_path.glob("*.jsonl"))
        assert len(trace_files) > 0

    async def test_failed_call_logged(self, tmp_path, trace_ctx):
        reg = ToolRegistry()
        reg.register(NonRetryableFailTool())
        trace_writer = TraceWriter(trace_dir=tmp_path)

        gw = ToolGateway(registry=reg, trace_writer=trace_writer)
        await gw.call(_make_call("hard_fail", arguments={}), trace_ctx=trace_ctx)

        trace_files = list(tmp_path.glob("*.jsonl"))
        assert len(trace_files) > 0

    async def test_tool_not_found(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        result = await gw.call(_make_call("nonexistent"), trace_ctx=trace_ctx)
        assert not result.success
        assert result.error is not None
        assert result.error.code == "TOOL_NOT_FOUND"


# ── TestCallMany ─────────────────────────────────────────────────


class TestCallMany:
    """Test batch execution."""

    async def test_multiple_read_tools(self, registry, trace_ctx):
        gw = ToolGateway(registry=registry)
        calls = [
            _make_call("echo", id="c1", arguments={"message": "a"}),
            _make_call("echo", id="c2", arguments={"message": "b"}),
        ]
        results = await gw.call_many(calls, trace_ctx=trace_ctx)
        assert len(results) == 2
        assert all(r.success for r in results)
        # Results should be in original order
        assert results[0].tool_call_id == "c1"
        assert results[1].tool_call_id == "c2"


# ── Helper: ToolSpec + ToolProvider factories for lazy registry tests ──


def _make_spec(
    name: str,
    description: str = "A tool",
    **kwargs,
) -> ToolSpec:
    """Create a ToolSpec with sensible defaults."""
    return ToolSpec(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
        **kwargs,
    )


class _SimpleProvider:
    """Minimal ToolProvider for registry population."""

    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id, name=call.name, success=True, output="ok"
        )

    async def health_check(self) -> bool:
        return True


def _make_registry(*specs: ToolSpec) -> ToolRegistry:
    """Build a ToolRegistry populated with SimpleProviders."""
    reg = ToolRegistry()
    for s in specs:
        reg.register(_SimpleProvider(s))
    return reg


# ── TestKeywordToolMatcher ──────────────────────────────────────


class TestKeywordToolMatcher:
    """Test the keyword-based tool matching strategy."""

    def test_prefers_name_match(self):
        specs = [
            _make_spec("web_search", "Search the internet"),
            _make_spec("file_read", "Read a file from disk"),
        ]
        matcher = KeywordToolMatcher()
        ranked = matcher.rank("search the web", specs)
        assert ranked[0].name == "web_search"

    def test_description_keyword_overlap(self):
        specs = [
            _make_spec("tool_a", "Parse CSV files into tables"),
            _make_spec("tool_b", "Send HTTP requests"),
        ]
        matcher = KeywordToolMatcher()
        ranked = matcher.rank("parse a CSV file", specs)
        assert ranked[0].name == "tool_a"

    def test_when_to_use_boosts_score(self):
        specs = [
            _make_spec(
                "tool_a",
                "Generic tool",
                when_to_use="When you need current news articles",
            ),
            _make_spec("tool_b", "Generic tool"),
        ]
        matcher = KeywordToolMatcher()
        ranked = matcher.rank("find current news", specs)
        assert ranked[0].name == "tool_a"

    def test_deterministic_tie_breaking(self):
        """Tools with equal scores are sorted alphabetically by name."""
        specs = [
            _make_spec("zzz_tool", "Does stuff"),
            _make_spec("aaa_tool", "Does stuff"),
        ]
        matcher = KeywordToolMatcher()
        ranked = matcher.rank("unrelated query", specs)
        assert ranked[0].name == "aaa_tool"
        assert ranked[1].name == "zzz_tool"

    def test_category_keyword_match(self):
        specs = [
            _make_spec("my_search", "Search for information"),
            _make_spec("my_writer", "Write documents"),
        ]
        matcher = KeywordToolMatcher()
        ranked = matcher.rank("find some data", specs)
        # "find" maps to "search" category, which matches "search" in "my_search" name
        assert ranked[0].name == "my_search"

    def test_empty_candidates(self):
        matcher = KeywordToolMatcher()
        ranked = matcher.rank("anything", [])
        assert ranked == []


# ── TestLazyToolRegistry ────────────────────────────────────────


class TestLazyToolRegistry:
    """Test the lazy tool registry."""

    def _make_full_registry(self, n: int = 10) -> ToolRegistry:
        """Create a registry with n tools."""
        specs = [_make_spec(f"tool_{i:02d}", f"Tool number {i}") for i in range(n)]
        return _make_registry(*specs)

    def test_initial_selection_respects_max(self):
        reg = self._make_full_registry(50)
        lazy = LazyToolRegistry(reg, max_initial_tools=5)
        selected = lazy.select_initial_tools("search the web for Python tutorials")
        assert len(selected) <= 5

    def test_working_set_populated_after_initial_selection(self):
        reg = self._make_full_registry(10)
        lazy = LazyToolRegistry(reg, max_initial_tools=3)
        lazy.select_initial_tools("do something")
        assert len(lazy.working_set) == 3

    def test_available_but_hidden(self):
        reg = self._make_full_registry(10)
        lazy = LazyToolRegistry(reg, max_initial_tools=3)
        lazy.select_initial_tools("do something")
        hidden = lazy.available_but_hidden
        assert len(hidden) == 7
        # Hidden tools should not be in working set
        ws_names = {s.name for s in lazy.working_set}
        for name in hidden:
            assert name not in ws_names

    def test_expand_returns_new_tools_only(self):
        specs = [
            _make_spec("web_search", "Search the internet"),
            _make_spec("file_read", "Read a file from disk"),
            _make_spec("shell_exec", "Run a shell command"),
        ]
        reg = _make_registry(*specs)
        lazy = LazyToolRegistry(reg, max_initial_tools=2)
        lazy.select_initial_tools("search and read files")
        initial_names = {s.name for s in lazy.working_set}

        new_tools = lazy.expand("run a shell command")
        assert len(new_tools) > 0
        for t in new_tools:
            assert t.name not in initial_names

    def test_expand_does_not_exceed_max_working_set(self):
        reg = self._make_full_registry(20)
        lazy = LazyToolRegistry(reg, max_initial_tools=10, max_working_set=12)
        lazy.select_initial_tools("do something")
        assert len(lazy.working_set) == 10

        lazy.expand("need more tools")
        assert len(lazy.working_set) <= 12

    def test_expand_always_adds_at_least_one(self):
        """Even when working set is at capacity, expand adds at least one tool."""
        reg = self._make_full_registry(5)
        lazy = LazyToolRegistry(reg, max_initial_tools=3, max_working_set=3)
        lazy.select_initial_tools("do something")
        assert len(lazy.working_set) == 3

        new_tools = lazy.expand("need more tools")
        # Should add at least 1 even though at max
        assert len(new_tools) >= 1

    def test_get_tool_on_demand_existing(self):
        specs = [
            _make_spec("web_search", "Search the internet"),
            _make_spec("file_read", "Read a file from disk"),
            _make_spec("shell_exec", "Run a shell command"),
        ]
        reg = _make_registry(*specs)
        lazy = LazyToolRegistry(reg, max_initial_tools=1)
        lazy.select_initial_tools("search the web")
        assert "shell_exec" not in [s.name for s in lazy.working_set]

        spec = lazy.get_tool_on_demand("shell_exec")
        assert spec is not None
        assert spec.name == "shell_exec"
        assert "shell_exec" in [s.name for s in lazy.working_set]

    def test_get_tool_on_demand_already_in_working_set(self):
        specs = [_make_spec("web_search", "Search the internet")]
        reg = _make_registry(*specs)
        lazy = LazyToolRegistry(reg, max_initial_tools=5)
        lazy.select_initial_tools("search")

        # Requesting a tool already in working set should return it
        spec = lazy.get_tool_on_demand("web_search")
        assert spec is not None
        assert spec.name == "web_search"

    def test_get_tool_on_demand_nonexistent(self):
        reg = self._make_full_registry(3)
        lazy = LazyToolRegistry(reg, max_initial_tools=1)
        lazy.select_initial_tools("do something")

        spec = lazy.get_tool_on_demand("does_not_exist")
        assert spec is None

    def test_to_openai_tools_only_includes_working_set(self):
        reg = self._make_full_registry(10)
        lazy = LazyToolRegistry(reg, max_initial_tools=2)
        lazy.select_initial_tools("use tool_00")
        openai_tools = lazy.to_openai_tools()
        assert len(openai_tools) <= 2
        for tool in openai_tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_reset_clears_working_set(self):
        reg = self._make_full_registry(5)
        lazy = LazyToolRegistry(reg, max_initial_tools=3)
        lazy.select_initial_tools("do something")
        assert len(lazy.working_set) == 3

        lazy.reset()
        assert len(lazy.working_set) == 0
        assert len(lazy.expansion_log) == 0

    def test_expansion_log_records_events(self):
        reg = self._make_full_registry(20)
        lazy = LazyToolRegistry(reg, max_initial_tools=3, max_working_set=5)
        lazy.select_initial_tools("do something")
        # expand adds up to (5-3)=2 hidden tools
        lazy.expand("need more tools")
        # Pick a tool that won't be added by either initial or expand
        # (initial gets 3, expand gets 2, so at least 15 remain hidden)
        ws_names = {s.name for s in lazy.working_set}
        remaining = [n for n in reg.list_tools() if n not in ws_names]
        assert len(remaining) > 0
        lazy.get_tool_on_demand(remaining[0])

        log = lazy.expansion_log
        assert len(log) == 3
        assert log[0].trigger == "initial_selection"
        assert log[1].trigger == "on_demand_expansion"
        assert log[2].trigger == "explicit_request"


# ── TestFormatToolForLLM ────────────────────────────────────────


class TestFormatToolForLLM:
    """Test tool description formatting for LLM consumption."""

    def test_basic_description_without_affordances(self):
        spec = _make_spec("web_search", "Search the web")
        formatted = format_tool_for_llm(spec)
        assert "**web_search**: Search the web" in formatted
        # No affordance lines
        assert "Use when:" not in formatted
        assert "Expect:" not in formatted

    def test_includes_when_to_use(self):
        spec = _make_spec(
            "web_search",
            "Search the web",
            when_to_use="When you need current information",
        )
        formatted = format_tool_for_llm(spec)
        assert "Use when: When you need current information" in formatted

    def test_includes_what_to_expect(self):
        spec = _make_spec(
            "web_search",
            "Search the web",
            what_to_expect="Returns a list of search results",
        )
        formatted = format_tool_for_llm(spec)
        assert "Expect: Returns a list of search results" in formatted

    def test_includes_failure_meaning(self):
        spec = _make_spec(
            "web_search",
            "Search the web",
            failure_meaning="Query was too broad",
        )
        formatted = format_tool_for_llm(spec)
        assert "If it fails: Query was too broad" in formatted

    def test_includes_success_next_step(self):
        spec = _make_spec(
            "web_search",
            "Search the web",
            success_next_step="Summarize and cite results",
        )
        formatted = format_tool_for_llm(spec)
        assert "After success: Summarize and cite results" in formatted

    def test_includes_write_warning(self):
        spec = _make_spec(
            "file_write",
            "Write a file",
            side_effect=SideEffect.WRITE,
        )
        formatted = format_tool_for_llm(spec)
        assert "[WRITE]" in formatted

    def test_no_write_warning_for_read_tools(self):
        spec = _make_spec("file_read", "Read a file")
        formatted = format_tool_for_llm(spec)
        assert "[WRITE]" not in formatted

    def test_full_affordance_spec(self):
        spec = ToolSpec(
            name="web_search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            when_to_use="When you need current information",
            what_to_expect="Returns search results with snippets",
            failure_meaning="Query was too broad",
            success_next_step="Summarize the most relevant results",
        )
        formatted = format_tool_for_llm(spec)
        assert "**web_search**" in formatted
        assert "Use when:" in formatted
        assert "Expect:" in formatted
        assert "If it fails:" in formatted
        assert "After success:" in formatted


# ── TestFormatToolListForLLM ────────────────────────────────────


class TestFormatToolListForLLM:
    """Test batch tool list formatting."""

    def test_empty_list(self):
        formatted = format_tool_list_for_llm([])
        assert formatted == "No tools are currently available."

    def test_single_tool(self):
        specs = [_make_spec("echo", "Echo input back")]
        formatted = format_tool_list_for_llm(specs)
        assert "You have access to 1 tools:" in formatted
        assert "**echo**" in formatted
        assert "additional tools may be made available" in formatted

    def test_multiple_tools(self):
        specs = [
            _make_spec("tool_a", "First tool"),
            _make_spec("tool_b", "Second tool"),
            _make_spec("tool_c", "Third tool"),
        ]
        formatted = format_tool_list_for_llm(specs)
        assert "You have access to 3 tools:" in formatted
        assert "**tool_a**" in formatted
        assert "**tool_b**" in formatted
        assert "**tool_c**" in formatted


# ── TestToolSpecAffordanceFields ────────────────────────────────


class TestToolSpecAffordanceFields:
    """Test that ToolSpec affordance fields are properly handled."""

    def test_backward_compatible_defaults(self):
        """Existing ToolSpec usage works unchanged."""
        spec = ToolSpec(
            name="echo",
            description="Echo input",
            input_schema={"type": "object", "properties": {}},
        )
        assert spec.when_to_use is None
        assert spec.what_to_expect is None
        assert spec.failure_meaning is None
        assert spec.success_next_step is None
        assert spec.category is None
        assert spec.related_tools == []

    def test_all_affordance_fields_settable(self):
        spec = ToolSpec(
            name="web_search",
            description="Search the web",
            input_schema={"type": "object", "properties": {}},
            when_to_use="When you need current info",
            what_to_expect="Returns search results",
            failure_meaning="Query too broad",
            success_next_step="Summarize results",
            category="search",
            related_tools=["summarize", "extract_citations"],
        )
        assert spec.when_to_use == "When you need current info"
        assert spec.what_to_expect == "Returns search results"
        assert spec.failure_meaning == "Query too broad"
        assert spec.success_next_step == "Summarize results"
        assert spec.category == "search"
        assert spec.related_tools == ["summarize", "extract_citations"]

    def test_serialization_roundtrip(self):
        spec = ToolSpec(
            name="test",
            description="test",
            input_schema={},
            when_to_use="use it now",
            category="code",
            related_tools=["other"],
        )
        data = spec.model_dump()
        restored = ToolSpec.model_validate(data)
        assert restored.when_to_use == "use it now"
        assert restored.category == "code"
        assert restored.related_tools == ["other"]
