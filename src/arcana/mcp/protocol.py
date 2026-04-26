"""MCP protocol pure functions. No I/O, no side effects."""

from __future__ import annotations

from typing import Any

from arcana.contracts.mcp import MCPError, MCPMessage, MCPToolSpec
from arcana.contracts.tool import SideEffect, ToolError, ToolErrorCategory, ToolSpec


def serialize_message(msg: MCPMessage) -> bytes:
    """Serialize MCPMessage to JSON-RPC bytes.

    Excludes None fields to produce valid JSON-RPC messages.
    A request must not contain 'result'/'error'; a response must not contain 'method'/'params'.
    """
    return (msg.model_dump_json(exclude_none=True) + "\n").encode()


def deserialize_message(data: bytes) -> MCPMessage:
    """Deserialize JSON-RPC bytes to MCPMessage."""
    return MCPMessage.model_validate_json(data.strip())


def mcp_tool_to_arcana_spec(
    mcp_spec: MCPToolSpec,
    server_name: str,
    capability_prefix: str | None = None,
) -> ToolSpec:
    """Convert MCP tool spec to Arcana ToolSpec. Pure."""
    capabilities: list[str] = []
    if capability_prefix:
        capabilities.append(f"{capability_prefix}.{mcp_spec.name}")

    return ToolSpec(
        name=f"{server_name}.{mcp_spec.name}",
        description=mcp_spec.description or f"MCP tool: {mcp_spec.name}",
        input_schema=mcp_spec.input_schema,
        side_effect=_infer_side_effect(mcp_spec),
        capabilities=capabilities,
        when_to_use=(
            f"MCP tool from {server_name}: "
            f"{mcp_spec.description or mcp_spec.name}"
        ),
    )


def arcana_spec_to_mcp_tool(spec: ToolSpec) -> MCPToolSpec:
    """Convert Arcana ToolSpec to MCP tool spec. Pure."""
    return MCPToolSpec(
        name=spec.name,
        description=spec.description,
        input_schema=spec.input_schema,
    )


def mcp_error_to_tool_error(error: MCPError) -> ToolError:
    """Map MCP error to Arcana ToolError. Pure."""
    # MCP follows JSON-RPC conventions:
    #   -32700 parse error, -32600 invalid request, -32601 method not found,
    #   -32602 invalid params → caller error, won't change on retry → VALIDATION
    #   -32603 internal error, plus the impl-defined server-error band
    #   (-32099..-32000) → server-side transient → TRANSPORT
    #   everything else → upstream tool's clean failure → LOGIC
    if error.code in (-32700, -32600, -32601, -32602):
        category = ToolErrorCategory.VALIDATION
    elif error.code == -32603 or -32099 <= error.code <= -32000:
        category = ToolErrorCategory.TRANSPORT
    else:
        category = ToolErrorCategory.LOGIC

    return ToolError(
        category=category,
        message=error.message,
        code=str(error.code),
        details={"mcp_data": error.data} if error.data else {},
    )


def is_notification(message: MCPMessage) -> bool:
    """Check if a JSON-RPC message is a server notification.

    Notifications have a ``method`` field but no ``id``.  They are
    server-initiated and do not expect a response.
    """
    return message.method is not None and message.id is None


def make_request(
    method: str,
    params: dict[str, Any] | None = None,
    msg_id: str | int | None = None,
) -> MCPMessage:
    """Create a JSON-RPC request message. Pure."""
    return MCPMessage(
        id=msg_id,
        method=method,
        params=params,
    )


def make_response(msg_id: str | int | None, result: Any) -> MCPMessage:
    """Create a JSON-RPC response message. Pure."""
    return MCPMessage(id=msg_id, result=result)


def make_error_response(
    msg_id: str | int | None, code: int, message: str
) -> MCPMessage:
    """Create a JSON-RPC error response. Pure."""
    return MCPMessage(id=msg_id, error=MCPError(code=code, message=message))


def _infer_side_effect(mcp_spec: MCPToolSpec) -> SideEffect:
    """Infer side effect from MCP tool name/description. Pure heuristic."""
    name_lower = mcp_spec.name.lower()
    desc_lower = (mcp_spec.description or "").lower()

    write_indicators = [
        "write",
        "create",
        "delete",
        "update",
        "set",
        "put",
        "post",
        "remove",
        "send",
        "execute",
    ]
    for indicator in write_indicators:
        if indicator in name_lower or indicator in desc_lower:
            return SideEffect.WRITE

    return SideEffect.READ
