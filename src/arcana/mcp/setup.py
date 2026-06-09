"""Setup MCP tools in Arcana's tool system."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arcana.contracts.mcp import MCPServerConfig, MCPToolSpec
from arcana.contracts.tool import ToolSpec
from arcana.contracts.trace import EventType, TraceContext, TraceEvent
from arcana.mcp.client import MCPClient
from arcana.mcp.protocol import mcp_tool_to_arcana_spec, resolve_side_effect
from arcana.mcp.tool_provider import MCPToolProvider
from arcana.tool_gateway.registry import ToolRegistry

if TYPE_CHECKING:
    from arcana.trace.writer import TraceWriter

logger = logging.getLogger(__name__)


def _admit(spec: ToolSpec) -> tuple[bool, str]:
    """Provenance-integrity gate for an imported capability. Pure.

    A capability that *claims* a remote origin must carry the identity of the
    server it came from. A spec with ``origin != "local"`` but no
    ``server_name`` is contradictory provenance -- it cannot be authorized,
    traced, or attributed against a policy's ``server_names`` dimension, so it
    is refused rather than silently exposed to the LLM (roadmap test matrix:
    "remote capabilities without provenance ... are not exposed").
    """
    prov = spec.provenance
    if prov is not None and prov.origin != "local" and not prov.server_name:
        return False, (
            f"refused: origin={prov.origin!r} claims a remote capability but "
            f"carries no server_name"
        )
    return True, "admitted"


def register_mcp_tools(
    *,
    client: MCPClient,
    server_name: str,
    mcp_tools: list[MCPToolSpec],
    registry: ToolRegistry,
    capability_prefix: str | None = None,
    trace_writer: TraceWriter | None = None,
    trace_ctx: TraceContext | None = None,
) -> list[str]:
    """Admit and register one server's MCP tools into ``registry``.

    This is the single admission chokepoint for imported capabilities. Every
    tool is conservatively classified at spec-construction time
    (``resolve_side_effect``: an un-declared tool is exposed as WRITE +
    confirmation, never a silent confirmation-free READ), checked for
    provenance integrity (``_admit``), and the decision is written to trace as
    labeled evidence before the tool can reach the LLM.

    Returns the qualified names of the tools that were admitted. It is designed
    to be re-invoked on ``tools/list_changed`` by a future dynamic-discovery
    bridge: calling it again re-classifies, re-authorizes, and re-traces every
    tool for free.
    """
    admitted: list[str] = []
    for mcp_tool in mcp_tools:
        arcana_spec = mcp_tool_to_arcana_spec(
            mcp_tool,
            server_name=server_name,
            capability_prefix=capability_prefix,
        )
        _side_effect, _confirm, basis = resolve_side_effect(mcp_tool)
        ok, reason = _admit(arcana_spec)

        if not ok:
            logger.warning(
                "Refused MCP tool '%s' from server '%s': %s",
                arcana_spec.name,
                server_name,
                reason,
            )
            _trace_admission(
                trace_writer,
                trace_ctx,
                tool_name=arcana_spec.name,
                server_name=server_name,
                decision="refused",
                basis=basis,
                spec=arcana_spec,
                reason=reason,
            )
            continue

        provider = MCPToolProvider(
            client=client,
            server_name=server_name,
            mcp_tool_name=mcp_tool.name,
            arcana_spec=arcana_spec,
        )
        registry.register(provider)
        admitted.append(arcana_spec.name)
        _trace_admission(
            trace_writer,
            trace_ctx,
            tool_name=arcana_spec.name,
            server_name=server_name,
            decision="downgraded" if basis == "inferred" else "admitted",
            basis=basis,
            spec=arcana_spec,
            reason=reason,
        )

    return admitted


def _trace_admission(
    trace_writer: TraceWriter | None,
    trace_ctx: TraceContext | None,
    *,
    tool_name: str,
    server_name: str,
    decision: str,
    basis: str,
    spec: ToolSpec,
    reason: str,
) -> None:
    """Write a CAPABILITY_ADMISSION trace event. No-op without a writer."""
    if trace_writer is None or trace_ctx is None:
        return
    event = TraceEvent(
        run_id=trace_ctx.run_id,
        task_id=trace_ctx.task_id,
        step_id=trace_ctx.new_step_id(),
        event_type=EventType.CAPABILITY_ADMISSION,
        metadata={
            "tool_name": tool_name,
            "origin": "mcp",
            "server_name": server_name,
            "decision": decision,
            "side_effect_basis": basis,
            "side_effect": spec.side_effect.value,
            "requires_confirmation": spec.requires_confirmation,
            "reason": reason,
        },
    )
    trace_writer.write(event)


async def setup_mcp_tools(
    configs: list[MCPServerConfig],
    registry: ToolRegistry,
    trace_writer: TraceWriter | None = None,
) -> MCPClient:
    """
    Connect to MCP servers and register their tools in ToolRegistry.

    After this call, MCP tools are available through ToolGateway
    with full authorization, validation, and audit. Every imported tool passes
    through the admission gate (``register_mcp_tools``): conservative
    side-effect classification, provenance-integrity check, and a traced
    admission decision before exposure to the LLM.

    Returns the MCPClient (caller should keep it alive).
    """
    client = MCPClient()
    trace_ctx = trace_writer.create_context() if trace_writer is not None else None

    for config in configs:
        mcp_tools = await client.connect(config)
        register_mcp_tools(
            client=client,
            server_name=config.name,
            mcp_tools=mcp_tools,
            registry=registry,
            capability_prefix=config.capability_prefix,
            trace_writer=trace_writer,
            trace_ctx=trace_ctx,
        )

    return client
