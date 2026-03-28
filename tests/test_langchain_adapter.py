"""Tests for the LangChain tool adapter.

Since langchain_core is not a required dependency, all tests mock it.
"""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arcana.contracts.tool import ErrorType, SideEffect, ToolCall

# ---------------------------------------------------------------------------
# Helpers: fake langchain_core.tools module so the adapter can import it
# ---------------------------------------------------------------------------

def _make_lc_tool(
    name: str = "wiki_search",
    description: str = "Search Wikipedia",
    args_schema: Any = None,
    args: dict[str, Any] | None = None,
    ainvoke_return: str = "Paris is the capital of France.",
    ainvoke_side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock LangChain BaseTool."""
    tool = MagicMock()
    tool.name = name
    tool.description = description

    # args_schema is a Pydantic model with .model_json_schema()
    if args_schema is not None:
        tool.args_schema = args_schema
    else:
        tool.args_schema = None

    # fallback .args dict
    if args is not None:
        tool.args = args
    elif args_schema is None:
        tool.args = {"query": "str"}

    # ainvoke
    if ainvoke_side_effect:
        tool.ainvoke = AsyncMock(side_effect=ainvoke_side_effect)
    else:
        tool.ainvoke = AsyncMock(return_value=ainvoke_return)

    return tool


@pytest.fixture(autouse=True)
def _patch_langchain_import():
    """Ensure langchain_core.tools is importable (mocked) for all tests."""
    fake_mod = MagicMock()
    fake_mod.BaseTool = MagicMock  # just needs to be a class-ish thing
    with patch.dict(sys.modules, {"langchain_core": MagicMock(), "langchain_core.tools": fake_mod}):
        # Force re-evaluation of the LANGCHAIN_AVAILABLE flag
        import arcana.tool_gateway.adapters.langchain as mod
        mod.LANGCHAIN_AVAILABLE = True
        mod.LCBaseTool = fake_mod.BaseTool
        yield


# ---------------------------------------------------------------------------
# Import adapter (must happen after fixture patches for module-level guard)
# ---------------------------------------------------------------------------

def _get_adapter_class():
    from arcana.tool_gateway.adapters.langchain import LangChainToolAdapter
    return LangChainToolAdapter


# =========================================================================
# 1. Construction — spec extraction
# =========================================================================

class TestAdapterConstruction:
    """Test that the adapter correctly extracts ToolSpec from LangChain tools."""

    def test_basic_construction(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool()
        adapter = Adapter(lc_tool)

        assert adapter.spec.name == "wiki_search"
        assert adapter.spec.description == "Search Wikipedia"
        assert adapter.spec.side_effect == SideEffect.READ  # default

    def test_custom_side_effect(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool()
        adapter = Adapter(lc_tool, side_effect=SideEffect.WRITE)

        assert adapter.spec.side_effect == SideEffect.WRITE

    def test_custom_capabilities(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool()
        adapter = Adapter(lc_tool, capabilities=["search", "knowledge"])

        assert adapter.spec.capabilities == ["search", "knowledge"]

    def test_requires_confirmation(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool()
        adapter = Adapter(lc_tool, requires_confirmation=True)

        assert adapter.spec.requires_confirmation is True

    def test_custom_retries_and_timeout(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool()
        adapter = Adapter(lc_tool, max_retries=5, timeout_ms=60000)

        assert adapter.spec.max_retries == 5
        assert adapter.spec.timeout_ms == 60000

    def test_schema_from_args_schema(self):
        """When lc_tool has args_schema (Pydantic model), use model_json_schema()."""
        Adapter = _get_adapter_class()

        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }

        lc_tool = _make_lc_tool(args_schema=mock_schema)
        adapter = Adapter(lc_tool)

        assert adapter.spec.input_schema["type"] == "object"
        assert "query" in adapter.spec.input_schema["properties"]
        assert "limit" in adapter.spec.input_schema["properties"]
        mock_schema.model_json_schema.assert_called_once()

    def test_schema_from_args_fallback(self):
        """When no args_schema, fall back to .args dict → simple string schema."""
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool(args_schema=None, args={"query": "str", "lang": "str"})
        adapter = Adapter(lc_tool)

        assert adapter.spec.input_schema["type"] == "object"
        assert "query" in adapter.spec.input_schema["properties"]
        assert "lang" in adapter.spec.input_schema["properties"]
        # Fallback produces {"type": "string"} for each arg
        assert adapter.spec.input_schema["properties"]["query"] == {"type": "string"}

    def test_empty_description_defaults(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool(description=None)
        adapter = Adapter(lc_tool)

        assert adapter.spec.description == ""

    def test_no_args_schema_no_args(self):
        """When tool has neither args_schema nor args, schema is empty."""
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool(args_schema=None)
        # Remove the fallback .args
        del lc_tool.args
        lc_tool.args_schema = None
        # Also need to make hasattr(lc_tool, "args") return False
        type(lc_tool).args = property(lambda self: (_ for _ in ()).throw(AttributeError))

        adapter = Adapter(lc_tool)
        assert adapter.spec.input_schema == {}


# =========================================================================
# 2. Execution — success path
# =========================================================================

class TestAdapterExecution:
    """Test that execute() correctly calls ainvoke and returns ToolResult."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool(ainvoke_return="42")
        adapter = Adapter(lc_tool)

        call = ToolCall(id="call-001", name="wiki_search", arguments={"query": "population of France"})
        result = await adapter.execute(call)

        assert result.success is True
        assert result.output == "42"
        assert result.tool_call_id == "call-001"
        assert result.name == "wiki_search"
        assert result.error is None
        lc_tool.ainvoke.assert_awaited_once_with({"query": "population of France"})

    @pytest.mark.asyncio
    async def test_execution_with_complex_return(self):
        """ainvoke can return dicts, lists, etc — adapter passes through."""
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool(ainvoke_return={"title": "France", "summary": "A country"})
        adapter = Adapter(lc_tool)

        call = ToolCall(id="call-002", name="wiki_search", arguments={"query": "France"})
        result = await adapter.execute(call)

        assert result.success is True
        assert result.output == {"title": "France", "summary": "A country"}

    @pytest.mark.asyncio
    async def test_execution_with_empty_arguments(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool(ainvoke_return="ok")
        adapter = Adapter(lc_tool)

        call = ToolCall(id="call-003", name="wiki_search", arguments={})
        result = await adapter.execute(call)

        assert result.success is True
        lc_tool.ainvoke.assert_awaited_once_with({})


# =========================================================================
# 3. Execution — error handling
# =========================================================================

class TestAdapterErrorHandling:
    """Test that execute() catches exceptions and returns proper ToolError."""

    @pytest.mark.asyncio
    async def test_generic_exception(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool(ainvoke_side_effect=RuntimeError("API rate limited"))
        adapter = Adapter(lc_tool)

        call = ToolCall(id="call-err-1", name="wiki_search", arguments={"query": "test"})
        result = await adapter.execute(call)

        assert result.success is False
        assert result.error is not None
        assert result.error.error_type == ErrorType.NON_RETRYABLE
        assert "API rate limited" in result.error.message
        assert result.error.code == "LANGCHAIN_ERROR"
        assert result.tool_call_id == "call-err-1"

    @pytest.mark.asyncio
    async def test_value_error(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool(ainvoke_side_effect=ValueError("Invalid input"))
        adapter = Adapter(lc_tool)

        call = ToolCall(id="call-err-2", name="wiki_search", arguments={"query": ""})
        result = await adapter.execute(call)

        assert result.success is False
        assert "Invalid input" in result.error.message

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool(ainvoke_side_effect=TimeoutError("Connection timed out"))
        adapter = Adapter(lc_tool)

        call = ToolCall(id="call-err-3", name="wiki_search", arguments={"query": "test"})
        result = await adapter.execute(call)

        assert result.success is False
        assert "timed out" in result.error.message


# =========================================================================
# 4. Protocol compliance — ToolProvider
# =========================================================================

class TestToolProviderProtocol:
    """Verify the adapter satisfies the ToolProvider protocol."""

    def test_has_spec_property(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool()
        adapter = Adapter(lc_tool)

        # spec should be a ToolSpec
        from arcana.contracts.tool import ToolSpec
        assert isinstance(adapter.spec, ToolSpec)

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self):
        Adapter = _get_adapter_class()
        lc_tool = _make_lc_tool()
        adapter = Adapter(lc_tool)

        from arcana.contracts.tool import ToolResult
        call = ToolCall(id="call-proto", name="wiki_search", arguments={"query": "test"})
        result = await adapter.execute(call)
        assert isinstance(result, ToolResult)


# =========================================================================
# 5. Import guard
# =========================================================================

class TestImportGuard:
    """Test behavior when langchain_core is not installed."""

    def test_raises_import_error_when_unavailable(self):
        import arcana.tool_gateway.adapters.langchain as mod
        from arcana.tool_gateway.adapters.langchain import LangChainToolAdapter

        # Temporarily disable
        original = mod.LANGCHAIN_AVAILABLE
        mod.LANGCHAIN_AVAILABLE = False
        try:
            lc_tool = _make_lc_tool()
            with pytest.raises(ImportError, match="langchain-core is required"):
                LangChainToolAdapter(lc_tool)
        finally:
            mod.LANGCHAIN_AVAILABLE = original
