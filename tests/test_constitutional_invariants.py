"""Constitutional invariant tests.

These tests pin architectural promises from ``CONSTITUTION.md`` against the
implementation. Each test class targets one Principle or Inviolable Rule.
A failure here means the constitution is no longer being honored — fix the
code, not the test.

Invariants covered:

- **Side-effect aware tool dispatch** (Principle 3 + 6) — write tools must
  serialize when dispatched in batch; the runtime, not the LLM, owns this
  safety boundary.
- **Cognitive primitives are opt-in** (Principle 9 + Chapter IV Inviolable
  Rule) — default ``Runtime`` does not auto-expose ``recall``/``pin``/
  ``unpin`` to the LLM.
- **ask_user never blocks the user** (Principle 8 + Chapter IV) — without
  an ``input_handler``, the handler returns a fallback message rather than
  awaiting indefinitely.
- **Structured output coexists with tools** (Principle 6) — setting
  ``response_format_schema`` does not disable the tool surface.
- **No mechanical retry** (Fourth Prohibition) — only TRANSPORT, TIMEOUT,
  and RATE_LIMIT errors are eligible for the gateway's retry loop;
  VALIDATION, PERMISSION, LOGIC, CONFIRMATION_REQUIRED, and UNEXPECTED
  surface immediately so the LLM can react with a different strategy.
- **Capability provenance is real** (Principle 3 + v3.5/v3.6) — imported
  (MCP) capabilities carry origin/server provenance that the permission
  policy acts on: a server-scoped deny rule fires for the remote tool and
  never for a local one. Provenance is not a decorative recorded field; it
  changes a real authorization outcome and is non-bypassable. (The
  refuse-when-provenance-missing *exposure* gate is future work, not this
  invariant.)

Future invariants (not yet pinned, tracked as work):

- Pinned content survives ``WorkingSetBuilder`` compression at L0 fidelity.
- Final-answer detection does not depend on forced-output markers.
- Framework-authored context notes preserve provenance and never impersonate
  user intent, system policy, or assistant conclusions.
- Semantic-weakening provider/capability downgrades surface as structured
  feedback or trace evidence rather than changing the caller contract silently.
- Passive cognitive surfacing only happens after the user enabled the
  primitive and the LLM explicitly armed it.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from arcana.contracts.mcp import MCPToolSpec
from arcana.contracts.permission import (
    PermissionAction,
    PermissionMatch,
    PermissionPolicy,
    PermissionRule,
)
from arcana.contracts.tool import (
    ASK_USER_TOOL_NAME,
    SideEffect,
    ToolCall,
    ToolError,
    ToolErrorCategory,
    ToolResult,
    ToolSpec,
)
from arcana.contracts.trace import TraceContext
from arcana.mcp.protocol import mcp_tool_to_arcana_spec
from arcana.runtime.ask_user import AskUserHandler
from arcana.runtime.cognitive import (
    PIN_TOOL_NAME,
    RECALL_TOOL_NAME,
    UNPIN_TOOL_NAME,
)
from arcana.runtime.conversation import ConversationAgent
from arcana.tool_gateway.base import ToolProvider
from arcana.tool_gateway.gateway import ToolGateway
from arcana.tool_gateway.local_channel import LocalChannel
from arcana.tool_gateway.registry import ToolRegistry
from arcana.trace.writer import TraceWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DelayedTool(ToolProvider):
    """Tool that records its execution start/end timestamps."""

    def __init__(self, name: str, *, side_effect: SideEffect, delay: float) -> None:
        self._name = name
        self._side_effect = side_effect
        self._delay = delay
        self.starts: list[float] = []
        self.ends: list[float] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=f"{self._name} tool ({self._side_effect.value})",
            input_schema={"type": "object", "properties": {}},
            side_effect=self._side_effect,
            max_retries=0,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        self.starts.append(time.monotonic())
        await asyncio.sleep(self._delay)
        self.ends.append(time.monotonic())
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=True,
            output="ok",
        )


async def _auto_confirm(call: ToolCall, spec: ToolSpec) -> bool:
    """Auto-approve write tools in tests; the invariant under test is the
    *dispatch ordering*, not the confirmation gate itself."""
    return True


def _gateway_with(*providers: ToolProvider) -> ToolGateway:
    registry = ToolRegistry()
    for p in providers:
        registry.register(p)
    return ToolGateway(registry=registry, confirmation_callback=_auto_confirm)


def _agent(**overrides: object) -> ConversationAgent:
    """Build a ConversationAgent with the minimal arg set tests need.

    The agent is never run — these tests only inspect schema-shaped state.
    Passing ``gateway=None`` is fine because no LLM call is made.
    """
    return ConversationAgent(gateway=None, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Principle 3 + 6 — side-effect aware tool dispatch
# ---------------------------------------------------------------------------


class TestSideEffectDispatch:
    """Write tools must serialize; read tools may run concurrently."""

    @pytest.mark.asyncio
    async def test_write_tools_serialize_in_call_many(self) -> None:
        delay = 0.1
        write_tool = _DelayedTool("w", side_effect=SideEffect.WRITE, delay=delay)
        gw = _gateway_with(write_tool)

        calls = [ToolCall(id=f"w{i}", name="w", arguments={}) for i in range(3)]
        results = await gw.call_many(calls)

        assert all(r.success for r in results)
        # Three serial writes must finish AFTER the previous one started.
        # Prove non-overlap: start[i+1] >= end[i] (modulo a small jitter).
        for i in range(len(write_tool.starts) - 1):
            assert write_tool.starts[i + 1] >= write_tool.ends[i] - 0.01, (
                f"Write tools overlapped: start[{i + 1}]={write_tool.starts[i + 1]:.3f} "
                f"end[{i}]={write_tool.ends[i]:.3f}"
            )

    @pytest.mark.asyncio
    async def test_read_tools_run_concurrently_in_call_many(self) -> None:
        delay = 0.15
        read_tool = _DelayedTool("r", side_effect=SideEffect.READ, delay=delay)
        gw = _gateway_with(read_tool)

        calls = [ToolCall(id=f"r{i}", name="r", arguments={}) for i in range(3)]
        start = time.monotonic()
        results = await gw.call_many(calls)
        elapsed = time.monotonic() - start

        assert all(r.success for r in results)
        # Concurrent: ~delay total. Sequential would be ~3*delay.
        assert elapsed < delay * 2, (
            f"Read tools serialized: {elapsed:.2f}s for 3x{delay}s tools"
        )

    @pytest.mark.asyncio
    async def test_local_channel_uses_side_effect_aware_dispatch(self) -> None:
        """LocalChannel.execute_many must route through call_many, not call_many_concurrent.

        This is the regression guard for the runtime's default execution
        path — ConversationAgent dispatches batched tool calls through
        LocalChannel, so the channel's choice of dispatcher is the actual
        boundary the constitution rests on.
        """
        delay = 0.1
        write_tool = _DelayedTool("w", side_effect=SideEffect.WRITE, delay=delay)
        gw = _gateway_with(write_tool)
        channel = LocalChannel(gw)

        calls = [ToolCall(id=f"w{i}", name="w", arguments={}) for i in range(3)]
        await channel.execute_many(calls)

        for i in range(len(write_tool.starts) - 1):
            assert write_tool.starts[i + 1] >= write_tool.ends[i] - 0.01, (
                "LocalChannel.execute_many ran write tools concurrently"
            )


# ---------------------------------------------------------------------------
# Principle 9 — cognitive primitives are opt-in
# ---------------------------------------------------------------------------


class TestCognitivePrimitivesOptIn:
    """Default Runtime must not expose recall/pin/unpin to the LLM."""

    def test_default_agent_does_not_expose_cognitive_tools(self) -> None:
        agent = _agent()

        tool_names = {
            t["function"]["name"] for t in (agent._get_current_tools() or [])
        }

        assert RECALL_TOOL_NAME not in tool_names
        assert PIN_TOOL_NAME not in tool_names
        assert UNPIN_TOOL_NAME not in tool_names
        # ask_user is the only built-in always exposed (Principle 8).
        assert ASK_USER_TOOL_NAME in tool_names

    def test_opting_in_to_pin_exposes_pin_and_unpin(self) -> None:
        agent = _agent(cognitive_primitives=["pin"])

        tool_names = {
            t["function"]["name"] for t in (agent._get_current_tools() or [])
        }

        assert PIN_TOOL_NAME in tool_names
        # unpin rides with pin — they are a symmetric pair.
        assert UNPIN_TOOL_NAME in tool_names
        # recall is independent: opting into pin does not enable recall.
        assert RECALL_TOOL_NAME not in tool_names


# ---------------------------------------------------------------------------
# Principle 8 + Chapter IV — ask_user never blocks the user
# ---------------------------------------------------------------------------


class TestAskUserNeverBlocks:
    """Without an input_handler, ask_user falls back synchronously."""

    @pytest.mark.asyncio
    async def test_ask_user_without_handler_returns_fallback(self) -> None:
        handler = AskUserHandler(input_handler=None)

        # Wrap in wait_for to assert non-blocking even with no answer source.
        result = await asyncio.wait_for(handler.handle("anything?"), timeout=0.5)

        assert isinstance(result, str)
        assert result  # non-empty
        # The exact fallback wording is internal; the invariant is that the
        # LLM gets *some* string and is not stuck. Spot-check a stable token.
        assert "proceed" in result.lower()


# ---------------------------------------------------------------------------
# Principle 6 — structured output does not disable tools
# ---------------------------------------------------------------------------


class TestStructuredOutputCoexistsWithTools:
    """response_format must not strip the tool surface from the LLM."""

    def test_tools_remain_when_response_format_is_set(self) -> None:
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        agent = _agent(response_format_schema=schema)

        tools = agent._get_current_tools()
        assert tools is not None
        names = {t["function"]["name"] for t in tools}
        # The runtime never silently disables tools when structured output
        # is requested. ask_user is the always-on built-in; its presence
        # proves the tool list survived.
        assert ASK_USER_TOOL_NAME in names


# ---------------------------------------------------------------------------
# Fourth Prohibition — No Mechanical Retry
# ---------------------------------------------------------------------------


class _AlwaysFailsTool(ToolProvider):
    """Tool that emits a configurable ToolError category on every call."""

    def __init__(self, category: ToolErrorCategory) -> None:
        self._category = category
        self.attempts = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="always_fails",
            description=f"Always fails with {self._category.value}",
            input_schema={"type": "object", "properties": {}},
            side_effect=SideEffect.READ,
            # Generous retry budget so the gateway never short-circuits via
            # max_retries — any retries observed must be policy-driven.
            max_retries=5,
            retry_delay_ms=1,
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        self.attempts += 1
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            success=False,
            error=ToolError(
                category=self._category,
                message=f"failed with {self._category.value}",
            ),
        )


class TestNoMechanicalRetry:
    """Only structurally transient categories enter the retry loop."""

    @pytest.mark.parametrize(
        "category",
        [
            ToolErrorCategory.VALIDATION,
            ToolErrorCategory.PERMISSION,
            ToolErrorCategory.LOGIC,
            ToolErrorCategory.UNEXPECTED,
        ],
    )
    @pytest.mark.asyncio
    async def test_non_retryable_categories_do_not_retry(
        self, category: ToolErrorCategory
    ) -> None:
        tool = _AlwaysFailsTool(category=category)
        gw = _gateway_with(tool)

        result = await gw.call(ToolCall(id="c1", name="always_fails", arguments={}))

        assert not result.success
        assert tool.attempts == 1, (
            f"Category {category.value} retried {tool.attempts} times — "
            f"only TRANSPORT/TIMEOUT/RATE_LIMIT are retry-eligible."
        )

    @pytest.mark.asyncio
    async def test_transport_errors_do_retry(self) -> None:
        """The retry loop must still fire for genuine transient failures."""
        tool = _AlwaysFailsTool(category=ToolErrorCategory.TRANSPORT)
        gw = _gateway_with(tool)

        result = await gw.call(ToolCall(id="c1", name="always_fails", arguments={}))

        assert not result.success
        # 1 initial + 5 retries = 6 attempts when all fail
        assert tool.attempts == 6, (
            f"TRANSPORT errors should retry up to max_retries; "
            f"saw {tool.attempts} attempt(s)"
        )

    def test_default_max_retries_is_conservative(self) -> None:
        """Default max_retries should be small enough to avoid masking real failures."""
        spec = ToolSpec(
            name="t",
            description="t",
            input_schema={"type": "object", "properties": {}},
        )
        # Conservative default — one retry covers transient flap, more than
        # two starts to mask real problems.
        assert spec.max_retries <= 2, (
            f"Default max_retries={spec.max_retries} is too aggressive; "
            f"see CONSTITUTION fourth prohibition (No Mechanical Retry)."
        )


# ---------------------------------------------------------------------------
# Principle 3 + v3.5/v3.6 — capability provenance is real, not decorative
# ---------------------------------------------------------------------------


class _StaticSpecTool(ToolProvider):
    """Provider that returns a fixed spec and succeeds — used to register a
    provenance-bearing capability into a real ToolGateway."""

    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id, name=call.name, success=True, output="ok"
        )


class TestCapabilityProvenanceIsReal:
    """Imported (MCP) capabilities carry real provenance the policy acts on.

    This is the anti-"fake provenance" guard. Before provenance was wired,
    the gateway evaluated every tool as origin='local' / server_name=None, so
    the ``origins`` / ``server_names`` policy dimensions were structurally
    dead. A future refactor that drops provenance at the gateway evaluation
    site re-collapses everything to local and breaks these tests.

    Out of scope here (tracked separately): this slice RECORDS provenance and
    lets policy act on it. It does NOT implement the roadmap's
    refuse-when-provenance-missing *exposure* gate (Phase 1, item 2) — these
    tests must not be read as proof that gate exists.
    """

    def test_mcp_registration_stamps_origin_and_server(self) -> None:
        mcp = MCPToolSpec(
            name="read_file", description="Read", input_schema={"type": "object"}
        )
        spec = mcp_tool_to_arcana_spec(mcp, server_name="fs")
        assert spec.provenance is not None
        assert spec.provenance.origin == "mcp"
        assert spec.provenance.server_name == "fs"

    @pytest.mark.asyncio
    async def test_server_rule_fires_for_mcp_and_not_for_local(self, tmp_path) -> None:
        mcp = MCPToolSpec(
            name="read",
            description="Read",
            input_schema={"type": "object", "properties": {}},
        )
        mcp_spec = mcp_tool_to_arcana_spec(mcp, server_name="untrusted")

        registry = ToolRegistry()
        registry.register(_StaticSpecTool(mcp_spec))  # name: "untrusted.read"
        registry.register(
            _StaticSpecTool(
                ToolSpec(
                    name="local_read",
                    description="local",
                    input_schema={"type": "object", "properties": {}},
                )
            )
        )

        policy = PermissionPolicy(
            rules=[
                PermissionRule(
                    id="deny-untrusted-server",
                    action=PermissionAction.DENY,
                    reason="Untrusted MCP server.",
                    match=PermissionMatch(server_names=["untrusted"]),
                )
            ]
        )
        trace_writer = TraceWriter(trace_dir=tmp_path)
        trace_ctx = TraceContext(run_id="invariant-run")
        gw = ToolGateway(
            registry=registry, trace_writer=trace_writer, permission_policy=policy
        )

        # 1. Authorization outcome: the server-scoped rule fires for the MCP tool.
        denied = await gw.call(
            ToolCall(id="1", name="untrusted.read", arguments={}), trace_ctx=trace_ctx
        )
        assert not denied.success
        assert denied.error is not None
        assert denied.error.category == ToolErrorCategory.PERMISSION
        assert denied.error.code == "PERMISSION_DENIED"

        # 2. Labeled evidence (v3.5 mandatory context provenance): the same
        #    decision records real provenance in the trace, not just a denial.
        #    This pins the trace-emission half so "non-bypassable" covers both
        #    the authorization outcome AND the evidence trail.
        event = json.loads(
            (tmp_path / "invariant-run.jsonl").read_text().splitlines()[0]
        )
        assert event["metadata"]["provenance"]["origin"] == "mcp"
        assert event["metadata"]["provenance"]["server_name"] == "untrusted"

        # 3. The same server-scoped rule must NOT touch a local capability.
        allowed = await gw.call(
            ToolCall(id="2", name="local_read", arguments={}), trace_ctx=trace_ctx
        )
        assert allowed.success
