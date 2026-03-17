"""Setup MCP tools in Arcana's tool system."""

from __future__ import annotations

from arcana.contracts.mcp import MCPServerConfig
from arcana.mcp.client import MCPClient
from arcana.mcp.protocol import mcp_tool_to_arcana_spec
from arcana.mcp.tool_provider import MCPToolProvider
from arcana.tool_gateway.registry import ToolRegistry


async def setup_mcp_tools(
    configs: list[MCPServerConfig],
    registry: ToolRegistry,
) -> MCPClient:
    """
    Connect to MCP servers and register their tools in ToolRegistry.

    After this call, MCP tools are available through ToolGateway
    with full authorization, validation, and audit.

    Returns the MCPClient (caller should keep it alive).
    """
    client = MCPClient()

    for config in configs:
        mcp_tools = await client.connect(config)

        for mcp_tool in mcp_tools:
            # Convert MCP spec to Arcana spec
            arcana_spec = mcp_tool_to_arcana_spec(
                mcp_tool,
                server_name=config.name,
                capability_prefix=config.capability_prefix,
            )

            # Create bridge provider
            provider = MCPToolProvider(
                client=client,
                server_name=config.name,
                mcp_tool_name=mcp_tool.name,
                arcana_spec=arcana_spec,
            )

            # Register in ToolRegistry -- now it's a first-class Arcana tool
            registry.register(provider)

    return client
