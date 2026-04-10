"""Tool Gateway - controlled tool execution with authorization, validation, and audit."""

from arcana.tool_gateway.base import ToolExecutionError, ToolProvider
from arcana.tool_gateway.execution_backend import ExecutionBackend, InProcessBackend
from arcana.tool_gateway.formatter import format_tool_for_llm, format_tool_list_for_llm
from arcana.tool_gateway.gateway import AuthorizationError, ConfirmationRequired, ToolGateway
from arcana.tool_gateway.lazy_registry import (
    KeywordToolMatcher,
    LazyToolRegistry,
    ToolExpansionEvent,
    ToolMatcher,
)
from arcana.tool_gateway.local_channel import LocalChannel
from arcana.tool_gateway.registry import ToolRegistry

__all__ = [
    "AuthorizationError",
    "ConfirmationRequired",
    "ExecutionBackend",
    "InProcessBackend",
    "KeywordToolMatcher",
    "LazyToolRegistry",
    "LocalChannel",
    "ToolExecutionError",
    "ToolExpansionEvent",
    "ToolGateway",
    "ToolMatcher",
    "ToolProvider",
    "ToolRegistry",
    "format_tool_for_llm",
    "format_tool_list_for_llm",
]
