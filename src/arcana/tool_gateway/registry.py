"""Registry for managing tool providers."""

from __future__ import annotations

from typing import Any

from arcana.contracts.tool import ToolSpec
from arcana.tool_gateway.base import ToolProvider


class ToolRegistry:
    """
    Registry for managing tool providers.

    Supports:
    - Registering/unregistering tool providers by name
    - Looking up tools by name
    - Listing available tools and their specs
    - Converting specs to OpenAI function calling format
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._providers: dict[str, ToolProvider] = {}

    def register(self, provider: ToolProvider) -> None:
        """
        Register a tool provider. Name comes from provider.spec.name.

        Args:
            provider: The tool provider to register
        """
        self._providers[provider.spec.name] = provider

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Returns:
            True if removed, False if not found
        """
        if name in self._providers:
            del self._providers[name]
            return True
        return False

    def get(self, name: str) -> ToolProvider | None:
        """
        Look up a tool provider by name.

        Args:
            name: Tool name

        Returns:
            The provider or None if not found
        """
        return self._providers.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._providers.keys())

    def get_specs(self) -> list[ToolSpec]:
        """Get specs for all registered tools."""
        return [p.spec for p in self._providers.values()]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """
        Convert all registered tool specs to OpenAI function calling format.

        Returns:
            List of tool definitions in OpenAI format
        """
        tools = []
        for provider in self._providers.values():
            spec = provider.spec
            tool_def: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
            tools.append(tool_def)
        return tools

    async def health_check_all(self) -> dict[str, bool]:
        """
        Check health of all registered tools.

        Returns:
            Dict mapping tool name to health status
        """
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception:
                results[name] = False
        return results
