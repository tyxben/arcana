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
  changes a real authorization outcome and is non-bypassable.
- **Imported capabilities are admission-gated** (v3.5 + Amendment 5) — an
  imported tool with no authoritative read-only declaration is exposed as
  WRITE + confirmation (never a guessed confirmation-free READ), a spec
  claiming a remote origin without a server identity is refused, and both
  decisions leave a CAPABILITY_ADMISSION trace event.
- **Protocol discovery is traceable but not trust** (Amendment 5) — querying a
  capability transport leaves PROTOCOL_DISCOVERY evidence that is distinct from
  per-tool CAPABILITY_ADMISSION decisions.

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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arcana.contracts.mcp import MCPMessage, MCPServerConfig, MCPToolSpec
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
    ToolProvenance,
    ToolResult,
    ToolSpec,
)
from arcana.contracts.trace import (
    EventType,
    GuardrailDecisionRecord,
    ProtocolDiscoveryRecord,
    TraceContext,
)
from arcana.mcp.client import MCPClient
from arcana.mcp.protocol import mcp_tool_to_arcana_spec
from arcana.mcp.setup import _admit, register_mcp_tools, setup_mcp_tools
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

    The complementary *exposure* gate (conservative side-effect downgrade +
    provenance-integrity refuse, applied before a tool reaches the LLM) is
    pinned separately by ``TestImportedCapabilityExposureGate`` below.
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


class TestImportedCapabilityExposureGate:
    """Imported capabilities are admission-gated before reaching the LLM.

    Two boundaries, both non-bypassable:

    1. *No silent semantic downgrade* (v3.5): an imported tool with no
       authoritative read-only declaration is exposed as WRITE + confirmation,
       never a guessed confirmation-free READ. A regression that re-introduces
       name-keyword guessing would let a real writer reach the LLM unconfirmed.
    2. *Refuse contradictory provenance*: a spec that claims a remote origin
       but carries no server identity cannot be attributed or authorized, so it
       is refused rather than exposed (roadmap test matrix).

    Both decisions leave labeled evidence (CAPABILITY_ADMISSION trace event).
    A server's own ``readOnlyHint`` is the only thing that buys a tool out of
    conservative treatment — recorded as the source's claim, not an Arcana
    guarantee (Amendment 5).
    """

    @pytest.mark.asyncio
    async def test_unannotated_tool_reaches_gateway_requiring_confirmation(
        self, tmp_path
    ) -> None:
        registry = ToolRegistry()
        trace_writer = TraceWriter(trace_dir=tmp_path)
        trace_ctx = TraceContext(run_id="gate-run")

        # A read-*named* tool with no annotations: the old heuristic would have
        # exposed it as a confirmation-free READ.
        admitted = register_mcp_tools(
            client=MCPClient(),
            server_name="svc",
            mcp_tools=[MCPToolSpec(name="read_thing", description="Read", input_schema={})],
            registry=registry,
            trace_writer=trace_writer,
            trace_ctx=trace_ctx,
        )
        assert admitted == ["svc.read_thing"]

        spec = registry.get("svc.read_thing").spec
        assert spec.side_effect == SideEffect.WRITE
        assert spec.requires_confirmation is True

        # The gateway actually gates it: no confirmation callback -> blocked,
        # never executed.
        gw = ToolGateway(registry=registry)
        result = await gw.call(ToolCall(id="1", name="svc.read_thing", arguments={}))
        assert not result.success
        assert result.error is not None
        assert result.error.category == ToolErrorCategory.CONFIRMATION_REQUIRED

        # Labeled evidence: the downgrade is in the trace.
        events = [
            json.loads(line)
            for line in (tmp_path / "gate-run.jsonl").read_text().splitlines()
        ]
        admission = [
            e for e in events if e["event_type"] == EventType.CAPABILITY_ADMISSION.value
        ]
        assert len(admission) == 1
        assert admission[0]["metadata"]["decision"] == "downgraded"
        assert admission[0]["metadata"]["side_effect_basis"] == "inferred"

    @pytest.mark.asyncio
    async def test_declared_read_only_tool_is_not_gated(self, tmp_path) -> None:
        registry = ToolRegistry()
        register_mcp_tools(
            client=MCPClient(),
            server_name="fs",
            mcp_tools=[
                MCPToolSpec(
                    name="list_dir",
                    description="List",
                    input_schema={},
                    annotations={"readOnlyHint": True},
                )
            ],
            registry=registry,
        )
        spec = registry.get("fs.list_dir").spec
        assert spec.side_effect == SideEffect.READ
        assert spec.requires_confirmation is False

        gw = ToolGateway(registry=registry)
        # No confirmation callback, yet a declared read-only tool is admitted
        # through (it would fail at execution since the client is unconnected,
        # but it is NOT confirmation-gated).
        result = await gw.call(ToolCall(id="1", name="fs.list_dir", arguments={}))
        assert (
            result.error is None
            or result.error.category != ToolErrorCategory.CONFIRMATION_REQUIRED
        )

    def test_contradictory_provenance_is_refused(self) -> None:
        # origin claims remote, but no server identity -> cannot be attributed.
        bad = ToolSpec(
            name="ghost",
            description="x",
            input_schema={},
            provenance=ToolProvenance(origin="mcp", server_name=None),
        )
        ok, reason = _admit(bad)
        assert ok is False
        assert "server_name" in reason

        # A local tool with no provenance is fine (implicitly local).
        local = ToolSpec(name="local", description="x", input_schema={})
        assert _admit(local) == (True, "admitted")


class TestBoundaryTraceContracts:
    """Trace metadata for protocol discovery and guardrails is structured evidence."""

    def test_protocol_discovery_record_serializes_adapter_decision(self) -> None:
        record = ProtocolDiscoveryRecord(
            protocol="mcp",
            server_name="srv",
            transport="stdio",
            action="initial_tools_list",
            decision="discovered",
            tool_count=2,
            tool_names_digest="a" * 16,
            tool_specs_digest="b" * 16,
            removed_count=0,
            admitted_count=2,
        )

        data = record.model_dump(mode="json", exclude_none=True)

        assert EventType.PROTOCOL_DISCOVERY.value == "protocol_discovery"
        assert data["protocol"] == "mcp"
        assert data["decision"] == "discovered"
        assert data["admitted_count"] == 2

    def test_guardrail_record_is_boundary_evidence_not_workflow(self) -> None:
        record = GuardrailDecisionRecord(
            guardrail_name="remote_tool_boundary",
            boundary="tool_call",
            action="block",
            subject_digest="c" * 16,
            reason="untrusted capability requested write access",
        )

        data = record.model_dump(mode="json", exclude_none=True)

        assert EventType.GUARDRAIL_DECISION.value == "guardrail_decision"
        assert data["action"] == "block"
        assert "next_tool" not in data
        assert "replacement_goal" not in data


class TestProtocolDiscoveryIsTraceableButNotTrust:
    """Protocol discovery is visible evidence, not an authorization shortcut.

    Amendment 5 says protocols are transports, not trust boundaries. Discovering
    tools from a protocol source therefore needs its own trace evidence, while
    each imported tool still needs a separate admission decision before LLM
    exposure.
    """

    @pytest.mark.asyncio
    async def test_mcp_discovery_trace_is_separate_from_admission(
        self,
        tmp_path,
    ) -> None:
        init_response = MCPMessage(id=1, result={"capabilities": {}})
        tools_response = MCPMessage(
            id=2,
            result={
                "tools": [
                    {
                        "name": "list_dir",
                        "description": "List",
                        "inputSchema": {},
                        "annotations": {"readOnlyHint": True},
                    }
                ]
            },
        )
        transport = AsyncMock()
        transport.connect = AsyncMock()
        transport.send = AsyncMock()
        transport.receive = AsyncMock(side_effect=[init_response, tools_response])
        transport.close = AsyncMock()

        registry = ToolRegistry()
        trace_writer = TraceWriter(trace_dir=tmp_path)

        with patch("arcana.mcp.client._create_transport", return_value=transport):
            client = await setup_mcp_tools(
                [MCPServerConfig(name="fs", command="echo")],
                registry,
                trace_writer=trace_writer,
            )

        assert "fs.list_dir" in registry.list_tools()

        run_ids = trace_writer.list_runs()
        assert len(run_ids) == 1
        events = [
            json.loads(line)
            for line in (tmp_path / f"{run_ids[0]}.jsonl").read_text().splitlines()
        ]
        discovery = [
            e for e in events if e["event_type"] == EventType.PROTOCOL_DISCOVERY.value
        ]
        admissions = [
            e for e in events if e["event_type"] == EventType.CAPABILITY_ADMISSION.value
        ]

        assert len(discovery) == 1
        assert len(admissions) == 1
        discovery_metadata = discovery[0]["metadata"]
        admission_metadata = admissions[0]["metadata"]

        assert discovery_metadata["protocol"] == "mcp"
        assert discovery_metadata["server_name"] == "fs"
        assert discovery_metadata["decision"] == "discovered"
        assert discovery_metadata["tool_count"] == 1
        assert "tool_names_digest" in discovery_metadata
        assert "tool_name" not in discovery_metadata

        assert admission_metadata["tool_name"] == "fs.list_dir"
        assert admission_metadata["decision"] == "admitted"
        assert len(admission_metadata["capability_digest"]) == 16

        await client.disconnect_all()


# ---------------------------------------------------------------------------
# Amendment 3 + Phase 3 — subagents impose no default topology, no recursion
# ---------------------------------------------------------------------------


class TestSubagentServiceImposesNoTopology:
    """The experimental subagent service is a user-directed facade.

    Amendment 3 (v3.4) rejected any default main-agent/subagent hierarchy,
    scheduler, or supervisor. The Phase 3 facade must therefore spawn nothing
    on its own — the caller explicitly registers and asks — and must forbid a
    subagent from recursively spawning another (v1 non-goal). It must also
    confine a subagent to its granted authority: a DENY permission policy is
    non-bypassable (Principle 3 — the runtime owns the boundary).
    """

    def test_fresh_service_spawns_no_agents(self) -> None:
        from arcana.experimental import subagents
        from arcana.runtime_core import Runtime, RuntimeConfig

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )
        svc = subagents(rt)

        # No main agent, no pre-registered worker, no scheduler attribute.
        assert svc.names == []
        assert not hasattr(svc, "scheduler")
        assert not hasattr(svc, "supervisor")

    @pytest.mark.asyncio
    async def test_subagent_cannot_spawn_subagent(self) -> None:
        from arcana.experimental import SubagentRecursionError, subagents
        from arcana.experimental.subagents import _DELEGATION_ACTIVE
        from arcana.runtime_core import Runtime, RuntimeConfig

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )
        svc = subagents(rt)
        svc.add("worker")

        # Simulate being inside an active delegation (as if a subagent's run
        # tried to delegate again). Both ask() and a delegation tool must
        # refuse rather than building a recursive hierarchy.
        token = _DELEGATION_ACTIVE.set(True)
        try:
            with pytest.raises(SubagentRecursionError):
                await svc.ask("worker", "spawn another")

            delegate = svc.as_tool("worker")
            refusal = await delegate._fn("spawn another")
            assert "refused" in refusal.lower()
        finally:
            _DELEGATION_ACTIVE.reset(token)

    @pytest.mark.asyncio
    async def test_subagent_cannot_exceed_granted_policy(self) -> None:
        """A subagent's DENY policy is non-bypassable.

        The runtime, not the LLM, owns the authorization boundary (Principle
        3). A subagent told to call a denied tool must not execute it, even
        when an approval handler would otherwise confirm it.
        """
        from unittest.mock import MagicMock

        import arcana
        from arcana.contracts.llm import (
            LLMResponse,
            TokenUsage,
            ToolCallRequest,
        )
        from arcana.experimental import subagents
        from arcana.runtime_core import Runtime, RuntimeConfig

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )
        rt._gateway.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc-1", name="writer", arguments='{"x": "data"}'
                        )
                    ],
                    usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                    model="test-model",
                    finish_reason="tool_calls",
                ),
                LLMResponse(
                    content="blocked",
                    tool_calls=None,
                    usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                    model="test-model",
                    finish_reason="stop",
                ),
            ]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        ran = {"writer": False}

        @arcana.tool(side_effect="write")
        async def writer(x: str) -> str:
            ran["writer"] = True
            return "wrote"

        async def approve(call, spec):  # would say yes; DENY precedes it
            return True

        deny = PermissionPolicy(
            rules=[
                PermissionRule(
                    action=PermissionAction.DENY,
                    reason="subagent may not write",
                    match=PermissionMatch(side_effects=[SideEffect.WRITE]),
                )
            ]
        )
        svc = subagents(rt, approval_handler=approve)
        svc.add("w", system="x", tools=[writer], permission_policy=deny)

        await svc.ask("w", "write the file")

        # Non-bypassable: the denied tool never executed.
        assert ran["writer"] is False


# ---------------------------------------------------------------------------
# Constitution v3.6 — lifecycle hooks are observers, not hidden planners
# ---------------------------------------------------------------------------


class TestLifecycleHooksAreObservers:
    """V2 lifecycle hooks observe; they never block or redirect the run.

    Constitution v3.6: guardrails are boundaries, not hidden workflows;
    blocking happens only at explicit boundaries (the tool-call guardrail).
    Lifecycle hooks therefore have no return channel and fail open — a
    raising or "blocking" observer changes nothing about execution.
    """

    @pytest.mark.asyncio
    async def test_observer_cannot_block_or_crash_a_tool(self) -> None:
        from unittest.mock import MagicMock

        import arcana
        from arcana.contracts.llm import (
            LLMResponse,
            TokenUsage,
            ToolCallRequest,
        )
        from arcana.runtime_core import Runtime, RuntimeConfig

        ran = {"look": False}

        @arcana.tool(side_effect="read")
        async def look(x: str) -> str:
            ran["look"] = True
            return "data"

        rt = Runtime(
            providers={"ollama": ""},
            tools=[look],
            config=RuntimeConfig(default_provider="ollama"),
        )
        rt._gateway.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc-1", name="look", arguments='{"x": "a"}'
                        )
                    ],
                    usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                    model="m",
                    finish_reason="tool_calls",
                ),
                LLMResponse(
                    content="done",
                    tool_calls=None,
                    usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    model="m",
                    finish_reason="stop",
                ),
            ]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        # Observers that try to block (return False) AND crash (raise). Neither
        # has any effect: the tool still runs and the conversation completes.
        rt.on("tool_start", lambda **k: False)

        def crash(**k):
            raise RuntimeError("observer blew up")

        rt.on("turn_start", crash)
        rt.on("tool_end", crash)

        async with rt.chat() as c:
            result = await c.send("use the tool")

        assert ran["look"] is True
        assert result.content == "done"


# ---------------------------------------------------------------------------
# Design pattern 4 (Trace Everything) — audited on the LIVE V2 path
# ---------------------------------------------------------------------------


class TestTraceEverythingOnLivePath:
    """A real ConversationAgent run must audit its tool calls.

    The provider LLM_CALL and gateway TOOL_CALL audit events are gated on a
    trace context. The V2 agent must thread one through so the evidence chain
    is real on the live path — not only in tests that drive the gateway
    directly. (Otherwise the Phase 1/3 permission/guardrail audit metadata
    never reaches the trace in production.)
    """

    @pytest.mark.asyncio
    async def test_live_tool_run_writes_tool_call_audit(self, tmp_path) -> None:
        from collections import Counter

        import arcana
        from arcana.contracts.llm import (
            LLMResponse,
            TokenUsage,
            ToolCallRequest,
        )
        from arcana.runtime_core import Runtime, RuntimeConfig
        from arcana.trace.reader import TraceReader

        @arcana.tool(side_effect="read")
        async def look(x: str) -> str:
            return "data"

        rt = Runtime(
            providers={"ollama": ""},
            tools=[look],
            trace=True,
            config=RuntimeConfig(
                default_provider="ollama", trace_dir=str(tmp_path)
            ),
        )
        rt._gateway.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc-1", name="look", arguments='{"x": "a"}'
                        )
                    ],
                    usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=5, total_tokens=15
                    ),
                    model="m",
                    finish_reason="tool_calls",
                ),
                LLMResponse(
                    content="done",
                    tool_calls=None,
                    usage=TokenUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    model="m",
                    finish_reason="stop",
                ),
            ]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        async with rt.chat() as c:
            await c.send("use the tool")

        reader = TraceReader(trace_dir=tmp_path)
        types: Counter = Counter()
        for f in tmp_path.glob("*.jsonl"):
            types += Counter(
                e.event_type.value for e in reader.read_events(f.stem)
            )
        assert types["tool_call"] >= 1


# ---------------------------------------------------------------------------
# Second Prohibition (No Controllability Theater) — degraded providers
# deliver real tool calls, not pseudo-capabilities
# ---------------------------------------------------------------------------


class TestDegradedToolCallsAreReal:
    """A provider without native tool-calling must still execute tools.

    The prompt-based fallback asks the model to emit the call as JSON text.
    If that JSON is never parsed back into tool_calls, the framework shows a
    tool surface it cannot honor (controllability theater). The provider must
    recover the call so the runtime actually executes it (finding F2).
    """

    @pytest.mark.asyncio
    async def test_text_fallback_tool_call_is_executable(self) -> None:
        from types import SimpleNamespace

        from arcana.contracts.llm import (
            LLMRequest,
            Message,
            MessageRole,
            ModelConfig,
        )
        from arcana.gateway.providers.openai_compatible import (
            OpenAICompatibleProvider,
            ProviderProfile,
        )

        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_key="sk-test",
            base_url="http://localhost",
            profile=ProviderProfile(tool_calls=False),
        )
        msg = SimpleNamespace(
            content=(
                '```json\n{"tool_call": {"name": "search", '
                '"arguments": {"q": "x"}}}\n```'
            ),
            tool_calls=None,
        )
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
            prompt_tokens_details=None,
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=usage,
            model="m",
        )
        provider.client.chat.completions.create = AsyncMock(
            return_value=completion
        )

        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="search x")],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        resp = await provider.generate(
            request, ModelConfig(provider="test", model_id="m")
        )

        # The text JSON became a real, executable tool call — not a final
        # answer of raw JSON.
        assert resp.tool_calls, "degraded provider must surface the tool call"
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == '{"q": "x"}'

    @pytest.mark.asyncio
    async def test_capability_downgrade_leaves_a_trace_marker(self, tmp_path) -> None:
        """A silent capability downgrade is the theater the constitution forbids.

        When a provider degrades to the prompt-based tool fallback, that
        downgrade must be visible evidence in the trace (a counted, gateable
        signal), not only a log line.
        """
        from types import SimpleNamespace

        from arcana.contracts.llm import (
            LLMRequest,
            Message,
            MessageRole,
            ModelConfig,
        )
        from arcana.contracts.trace import TraceContext
        from arcana.gateway.providers.openai_compatible import (
            OpenAICompatibleProvider,
            ProviderProfile,
        )
        from arcana.trace.reader import TraceReader
        from arcana.trace.writer import TraceWriter

        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_key="sk-test",
            base_url="http://localhost",
            profile=ProviderProfile(tool_calls=False),
            trace_writer=TraceWriter(trace_dir=str(tmp_path)),
        )
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1, total_tokens=2,
                prompt_tokens_details=None,
            ),
            model="m",
        )
        provider.client.chat.completions.create = AsyncMock(
            return_value=completion
        )
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="x")],
            tools=[{"type": "function", "function": {"name": "t"}}],
        )
        await provider.generate(
            request,
            ModelConfig(provider="test", model_id="m"),
            TraceContext(run_id="run-deg"),
        )

        events = TraceReader(trace_dir=tmp_path).read_events("run-deg")
        llm_events = [e for e in events if e.event_type.value == "llm_call"]
        assert llm_events
        assert any(
            e.metadata.get("degraded_capabilities") for e in llm_events
        ), "capability downgrade must leave a trace marker"


# ---------------------------------------------------------------------------
# Principle 7 Corollary — eval is evidence/gate, never runtime governance
# ---------------------------------------------------------------------------


class TestEvalIsGateNotGovernance:
    """The regression gate must never run inside the agent runtime.

    Principle 7 Corollary: a failing eval blocks a release or returns
    structured risk evidence; it does not become a hidden supervisor that
    rewrites the running LLM's strategy. So ``RegressionGate`` (and the F5
    signal/golden gating) must have NO call site under ``src/arcana/runtime/``.
    """

    def test_regression_gate_not_invoked_in_runtime(self) -> None:
        import pathlib

        runtime_dir = pathlib.Path("src/arcana/runtime")
        offenders = []
        for py in runtime_dir.rglob("*.py"):
            text = py.read_text()
            if "RegressionGate" in text:
                offenders.append(str(py))
        assert not offenders, (
            f"RegressionGate referenced inside runtime/: {offenders} — eval is "
            f"gate-not-governance (Principle 7 Corollary)."
        )


# ---------------------------------------------------------------------------
# Amendment 6 — a running agent must never mutate itself
# ---------------------------------------------------------------------------


class TestEvolutionContractsNotInRuntime:
    """The self-evolution contracts must never be reached from the live runtime.

    Amendment 6 §1: a self-evolution loop's output is a proposal, never an
    in-place mutation of running Arcana state. Structurally, no module under
    ``src/arcana/runtime/`` may reference the evolution contracts — so "a
    running agent cannot mutate itself" is enforced by the import graph, not by
    trust. (Mirrors TestEvalIsGateNotGovernance.)
    """

    def test_no_evolution_contract_referenced_in_runtime(self) -> None:
        import pathlib

        forbidden = (
            "EvolutionProposal",
            "EvidenceBundle",
            "PromotionRecord",
            "classify_authority",
            "contracts.evolution",
        )
        runtime_dir = pathlib.Path("src/arcana/runtime")
        offenders = []
        for py in runtime_dir.rglob("*.py"):
            text = py.read_text()
            for name in forbidden:
                if name in text:
                    offenders.append(f"{py}: {name}")
        assert not offenders, (
            f"self-evolution contract referenced inside runtime/: {offenders} "
            f"— a running agent must never mutate itself (Amendment 6 §1)."
        )
