"""
Integration test: Real MCP server via npx @modelcontextprotocol/server-filesystem.

Requires: npx (Node.js)
Run: uv run pytest tests/integration/test_mcp_real.py -v
"""

from __future__ import annotations

import shutil

import pytest

from arcana.contracts.mcp import MCPServerConfig
from arcana.mcp.client import MCPClient

NPX_PATH = shutil.which("npx")

pytestmark = pytest.mark.skipif(
    NPX_PATH is None,
    reason="npx not available",
)


@pytest.fixture
def test_dir(tmp_path):
    """Create a temp directory with test files."""
    (tmp_path / "hello.txt").write_text("Hello from Arcana MCP test!")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("Nested file content")
    return tmp_path


@pytest.fixture
def fs_config(test_dir):
    """MCP filesystem server config scoped to test_dir."""
    return MCPServerConfig(
        name="filesystem",
        command=NPX_PATH,  # Use absolute path — venv may not have npx on PATH
        args=["-y", "@modelcontextprotocol/server-filesystem", str(test_dir)],
        timeout_ms=60000,
    )


class TestMCPRealFilesystem:
    """End-to-end tests with a real MCP filesystem server."""

    async def test_connect_and_discover_tools(self, fs_config):
        """Should connect and discover filesystem tools."""
        client = MCPClient()
        try:
            tools = await client.connect(fs_config)
            tool_names = [t.name for t in tools]
            assert len(tools) > 0, "Should discover at least one tool"
            # filesystem-server should expose read_file at minimum
            assert any("read" in name.lower() for name in tool_names), (
                f"Expected a read-related tool, got: {tool_names}"
            )
            print(f"\nDiscovered {len(tools)} tools: {tool_names}")
        finally:
            await client.disconnect_all()

    async def test_read_file_tool(self, fs_config, test_dir):
        """Should be able to read a file via MCP tool call."""
        client = MCPClient()
        try:
            tools = await client.connect(fs_config)
            tool_names = [t.name for t in tools]

            # Find the read_file tool
            read_tool = None
            for name in tool_names:
                if "read" in name.lower() and "file" in name.lower():
                    read_tool = name
                    break

            if not read_tool:
                pytest.skip(f"No read_file tool found in: {tool_names}")

            result = await client.call_tool(
                "filesystem", read_tool, {"path": str(test_dir / "hello.txt")}
            )
            assert "Hello from Arcana MCP test!" in str(result)
            print(f"\nRead result: {result}")
        finally:
            await client.disconnect_all()

    async def test_list_directory_tool(self, fs_config, test_dir):
        """Should be able to list directory contents."""
        client = MCPClient()
        try:
            tools = await client.connect(fs_config)
            tool_names = [t.name for t in tools]

            # Find list_directory or similar
            list_tool = None
            for name in tool_names:
                if "list" in name.lower() and "dir" in name.lower():
                    list_tool = name
                    break
                if "ls" in name.lower():
                    list_tool = name
                    break

            if not list_tool:
                pytest.skip(f"No list_directory tool found in: {tool_names}")

            result = await client.call_tool(
                "filesystem", list_tool, {"path": str(test_dir)}
            )
            result_str = str(result)
            assert "hello.txt" in result_str
            print(f"\nList result: {result}")
        finally:
            await client.disconnect_all()

    async def test_runtime_mcp_integration(self, test_dir):
        """Test MCP through Runtime.connect_mcp()."""
        from arcana.contracts.mcp import MCPServerConfig
        from arcana.runtime_core import Runtime

        runtime = Runtime(
            providers={"deepseek": "fake-key"},
            mcp_servers=[
                MCPServerConfig(
                    name="fs",
                    command=NPX_PATH,
                    args=["-y", "@modelcontextprotocol/server-filesystem", str(test_dir)],
                ),
            ],
        )

        tool_names = await runtime.connect_mcp()
        assert len(tool_names) > 0, "Runtime should discover MCP tools"
        print(f"\nRuntime MCP tools: {tool_names}")

        # Verify tools are in the tool registry
        assert runtime.tools is not None
        assert len(runtime.tools) > 0

        # Cleanup
        if runtime._mcp_client:
            await runtime._mcp_client.disconnect_all()
