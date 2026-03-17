"""Base transport for MCP communication."""

from __future__ import annotations

from abc import ABC, abstractmethod

from arcana.contracts.mcp import MCPMessage


class MCPTransport(ABC):
    """Abstract transport for MCP JSON-RPC communication."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""
        ...

    @abstractmethod
    async def send(self, message: MCPMessage) -> None:
        """Send a message."""
        ...

    @abstractmethod
    async def receive(self) -> MCPMessage:
        """Receive a message."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connection."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        ...
