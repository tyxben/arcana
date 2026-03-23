"""Adapter to wrap LangChain tools as Arcana ToolProviders.

Hard rules:
1. Tool execution MUST go through the ToolGateway pipeline (authz, validation, retry, trace)
2. Trace MUST use Arcana's own trace_spec

This adapter only bridges the execution interface — all governance
is handled by the ToolGateway layer above.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcana.contracts.tool import (
    ErrorType,
    SideEffect,
    ToolCall,
    ToolError,
    ToolResult,
    ToolSpec,
)
from arcana.tool_gateway.base import ToolProvider

if TYPE_CHECKING:
    pass

try:
    from langchain_core.tools import BaseTool as LCBaseTool  # type: ignore[import-not-found]

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LCBaseTool = None


class LangChainToolAdapter(ToolProvider):
    """
    Wraps a LangChain BaseTool as an Arcana ToolProvider.

    Usage:
        from langchain_community.tools import WikipediaQueryRun
        lc_tool = WikipediaQueryRun(...)
        adapter = LangChainToolAdapter(lc_tool, side_effect=SideEffect.READ)
        registry.register(adapter)
    """

    def __init__(
        self,
        lc_tool: Any,
        *,
        side_effect: SideEffect = SideEffect.READ,
        capabilities: list[str] | None = None,
        requires_confirmation: bool = False,
        max_retries: int = 3,
        timeout_ms: int = 30000,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for LangChain adapter. "
                "Install with: pip install arcana[langchain]"
            )

        self._lc_tool = lc_tool

        # Build input schema from LangChain tool
        input_schema: dict[str, Any] = {}
        if hasattr(lc_tool, "args_schema") and lc_tool.args_schema is not None:
            input_schema = lc_tool.args_schema.model_json_schema()
        elif hasattr(lc_tool, "args"):
            input_schema = {
                "type": "object",
                "properties": {
                    k: {"type": "string"} for k in lc_tool.args
                },
            }

        self._spec = ToolSpec(
            name=lc_tool.name,
            description=lc_tool.description or "",
            input_schema=input_schema,
            side_effect=side_effect,
            capabilities=capabilities or [],
            requires_confirmation=requires_confirmation,
            max_retries=max_retries,
            timeout_ms=timeout_ms,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute the LangChain tool via ainvoke."""
        try:
            output = await self._lc_tool.ainvoke(call.arguments)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=True,
                output=output,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                success=False,
                error=ToolError(
                    error_type=ErrorType.NON_RETRYABLE,
                    message=str(e),
                    code="LANGCHAIN_ERROR",
                ),
            )
