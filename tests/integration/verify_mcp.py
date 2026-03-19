"""
Standalone MCP verification script.
Run: uv run python tests/integration/verify_mcp.py
"""

import asyncio
import os
import tempfile

from arcana.contracts.mcp import MCPServerConfig
from arcana.mcp.client import MCPClient


async def main():
    # Create test directory (realpath to resolve macOS /var -> /private/var symlink)
    td = os.path.realpath(tempfile.mkdtemp(prefix="arcana-mcp-test-"))
    with open(os.path.join(td, "hello.txt"), "w") as f:
        f.write("Hello from Arcana MCP test!")
    os.makedirs(os.path.join(td, "sub"), exist_ok=True)
    with open(os.path.join(td, "sub", "nested.txt"), "w") as f:
        f.write("Nested content")

    print(f"Test dir: {td}")

    client = MCPClient()
    config = MCPServerConfig(
        name="filesystem",
        command="/Users/ty/.nvm/versions/node/v22.22.0/bin/npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", td],
        timeout_ms=60000,
    )

    try:
        print("Connecting to MCP filesystem server...")
        tools = await client.connect(config)
        tool_names = [t.name for t in tools]
        print(f"OK: Discovered {len(tools)} tools: {tool_names}")

        # Test read_file
        if "read_file" in tool_names:
            result = await client.call_tool("filesystem", "read_file", {"path": os.path.join(td, "hello.txt")})
            print(f"  read_file result: {repr(result)}")
            assert "Hello from Arcana MCP test!" in str(result), f"Unexpected: {result}"
            print(f"OK: read_file → {result}")

        # Test list_directory
        if "list_directory" in tool_names:
            result = await client.call_tool("filesystem", "list_directory", {"path": td})
            assert "hello.txt" in str(result)
            print(f"OK: list_directory → {result}")

        # Test write_file
        if "write_file" in tool_names:
            test_path = os.path.join(td, "written_by_mcp.txt")
            await client.call_tool("filesystem", "write_file", {"path": test_path, "content": "Written by Arcana MCP!"})
            assert os.path.exists(test_path)
            print(f"OK: write_file → {open(test_path).read()}")

        print("\nAll MCP tests passed!")

    except Exception as e:
        print(f"\nFAILED: {e}")
        raise
    finally:
        await client.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
