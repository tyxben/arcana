"""MCP (Model Context Protocol) contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MCPTransportType(str, Enum):
    """Transport type for MCP server connections."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


class MCPServerConfig(BaseModel):
    """Configuration for connecting to an MCP server."""

    name: str
    transport: MCPTransportType = MCPTransportType.STDIO
    # stdio
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # HTTP
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    # Common
    timeout_ms: int = 30000
    reconnect_attempts: int = 3
    reconnect_delay_ms: int = 1000
    capability_prefix: str | None = None


class MCPToolSpec(BaseModel):
    """MCP tool spec from server's tools/list."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPResource(BaseModel):
    """MCP resource spec."""

    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None


class MCPError(BaseModel):
    """JSON-RPC error."""

    code: int
    message: str
    data: Any | None = None


class MCPMessage(BaseModel):
    """JSON-RPC 2.0 message."""

    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str | None = None
    params: dict[str, Any] | None = None
    result: Any | None = None
    error: MCPError | None = None
