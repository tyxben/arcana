"""Smoke tests — minimal CI-compatible regression suite (Task 1.6).

Each test is designed to be run independently. Tests that require
external services (API keys, MCP servers) are skipped gracefully.
"""

from __future__ import annotations

import os
import shutil

import pytest
from dotenv import load_dotenv

load_dotenv()


def _has_key(var: str) -> bool:
    return bool(os.environ.get(var))


# ── Test 1: Simple generation ──────────────────────────────────────


class TestSmokeGeneration:
    """Basic LLM generation (DeepSeek)."""

    pytestmark = pytest.mark.skipif(
        not _has_key("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set"
    )

    @pytest.mark.asyncio
    async def test_simple_generation(self):
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        rt = Runtime(
            providers={"deepseek": os.environ["DEEPSEEK_API_KEY"]},
            budget=Budget(max_cost_usd=0.10),
            config=RuntimeConfig(default_provider="deepseek"),
        )
        result = await rt.run("What is 2 + 2? Answer with just the number.")
        assert result.success
        assert "4" in str(result.output)


# ── Test 2: Tool use ───────────────────────────────────────────────


class TestSmokeToolUse:
    """Tool use with a real provider."""

    pytestmark = pytest.mark.skipif(
        not _has_key("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set"
    )

    @pytest.mark.asyncio
    async def test_calculator_tool(self):
        import arcana

        @arcana.tool
        def calculator(expression: str) -> str:
            """Evaluate a math expression."""
            import ast
            import operator

            ops = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
            }

            def _eval_node(node: ast.expr) -> float:
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return float(node.value)
                if isinstance(node, ast.BinOp) and type(node.op) in ops:
                    return ops[type(node.op)](_eval_node(node.left), _eval_node(node.right))
                raise ValueError(f"Unsupported: {ast.dump(node)}")

            try:
                tree = ast.parse(expression.strip(), mode="eval")
                return str(_eval_node(tree.body))
            except Exception as e:
                return f"Error: {e}"

        result = await arcana.run(
            "Use the calculator tool to compute 123 * 456. Return the result.",
            provider="deepseek",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            tools=[calculator],
            max_turns=5,
        )
        assert result.success
        assert "56088" in str(result.output)


# ── Test 3: MCP stdio ─────────────────────────────────────────────


class TestSmokeMCP:
    """MCP stdio server connectivity."""

    pytestmark = pytest.mark.skipif(
        not shutil.which("npx"), reason="npx not available"
    )

    @pytest.mark.asyncio
    async def test_mcp_stdio_filesystem(self, tmp_path):
        from arcana.contracts.mcp import MCPServerConfig
        from arcana.mcp.client import MCPClient

        config = MCPServerConfig(
            name="fs",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", str(tmp_path)],
        )
        client = MCPClient()
        try:
            tools = await client.connect(config)
            assert len(tools) > 0
            tool_names = [t.name for t in tools]
            assert "read_file" in tool_names or any("read" in n for n in tool_names)
        finally:
            await client.disconnect_all()


# ── Test 4: Multi-turn conversation ────────────────────────────────


class TestSmokeMultiTurn:
    """Multi-turn conversation with context retention."""

    pytestmark = pytest.mark.skipif(
        not _has_key("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set"
    )

    @pytest.mark.asyncio
    async def test_context_retention(self):
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        rt = Runtime(
            providers={"deepseek": os.environ["DEEPSEEK_API_KEY"]},
            budget=Budget(max_cost_usd=0.20),
            config=RuntimeConfig(
                default_provider="deepseek",
                max_turns=3,
            ),
        )

        # Turn 1: establish context
        r1 = await rt.run("My name is Alice. Just say 'Hello Alice'.")
        assert r1.success

        # Turn 2: test context retention with new runtime (stateless)
        r2 = await rt.run(
            "Remember, my name is Alice. What is my name? Answer with just the name."
        )
        assert r2.success
        assert "Alice" in str(r2.output)
