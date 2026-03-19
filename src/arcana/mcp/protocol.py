"""MCP protocol pure functions. No I/O, no side effects."""

from __future__ import annotations

from typing import Any

from arcana.contracts.mcp import MCPError, MCPMessage, MCPToolSpec
from arcana.contracts.tool import ErrorType, SideEffect, ToolError, ToolSpec


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
    # MCP error codes follow JSON-RPC conventions
    if error.code in (-32600, -32601, -32602):
        error_type = ErrorType.NON_RETRYABLE
    elif error.code in (-32603, -32000):
        error_type = ErrorType.RETRYABLE
    else:
        error_type = (
            ErrorType.RETRYABLE
            if error.code >= -32099
            else ErrorType.NON_RETRYABLE
        )

    return ToolError(
        error_type=error_type,
        message=error.message,
        code=str(error.code),
        details={"mcp_data": error.data} if error.data else {},
    )


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
