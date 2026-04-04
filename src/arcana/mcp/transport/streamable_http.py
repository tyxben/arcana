"""Streamable HTTP transport for MCP (spec 2025-03-26).

Replaces the deprecated HTTP+SSE transport. Each client message is a POST
to a single MCP endpoint. The server may respond with:
  - application/json — a single JSON-RPC response
  - text/event-stream — an SSE stream containing responses, notifications, requests

Session management via Mcp-Session-Id header.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from arcana.contracts.mcp import MCPMessage, MCPServerConfig
from arcana.mcp.protocol import deserialize_message, serialize_message
from arcana.mcp.transport.base import MCPTransport

logger = logging.getLogger(__name__)


class StreamableHTTPTransport(MCPTransport):
    """MCP transport over Streamable HTTP (spec 2025-03-26).

    Communication pattern:
      Client → POST JSON-RPC message → Server
      Server → application/json OR text/event-stream → Client
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._connected = False
        # Queue for messages received via SSE that aren't the direct response
        self._inbox: asyncio.Queue[MCPMessage] = asyncio.Queue()

    async def connect(self) -> None:
        """Establish HTTP connection."""
        if not self._config.url:
            raise ValueError(
                f"MCP server '{self._config.name}' has no URL configured. "
                f"Set url in MCPServerConfig for Streamable HTTP transport."
            )

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout_ms / 1000),
            headers=self._config.headers,
        )
        self._connected = True
        logger.info("StreamableHTTP transport connected: %s", self._config.url)

    async def send(self, message: MCPMessage) -> None:
        """Send a JSON-RPC message via HTTP POST.

        The response is parsed and queued. For SSE streams, all events
        are read and queued.
        """
        if not self._client or not self._connected:
            raise ConnectionError(
                f"MCP server '{self._config.name}' transport not connected."
            )

        data = serialize_message(message)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        assert self._config.url is not None, "MCP server URL must not be None"
        try:
            response = await self._client.post(
                self._config.url,
                content=data,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"MCP server '{self._config.name}' request timed out "
                f"(timeout={self._config.timeout_ms}ms)."
            ) from exc
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to MCP server '{self._config.name}' at {self._config.url}. "
                f"Error: {exc}"
            ) from exc

        # Capture session ID from response
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

        # 202 Accepted — notification/response acknowledged, no body
        if response.status_code == 202:
            return

        # Error responses
        if response.status_code >= 400:
            raise ConnectionError(
                f"MCP server '{self._config.name}' returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        content_type = response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            # SSE stream — parse all events and queue them
            await self._parse_sse_response(response.text)
        elif "application/json" in content_type:
            # Single JSON response
            msg = deserialize_message(response.content)
            await self._inbox.put(msg)
        else:
            raise ConnectionError(
                f"MCP server '{self._config.name}' returned unexpected content-type: "
                f"{content_type}"
            )

    async def receive(self) -> MCPMessage:
        """Receive the next message from the inbox queue."""
        if not self._connected:
            raise ConnectionError(
                f"MCP server '{self._config.name}' transport not connected."
            )

        try:
            return await asyncio.wait_for(
                self._inbox.get(),
                timeout=self._config.timeout_ms / 1000,
            )
        except TimeoutError:
            raise ConnectionError(
                f"MCP server '{self._config.name}' did not respond "
                f"within {self._config.timeout_ms}ms."
            ) from None

    async def close(self) -> None:
        """Close HTTP connection. Optionally terminate session."""
        if self._client:
            # Try to terminate session if we have one
            if self._session_id and self._config.url:
                try:
                    await self._client.delete(
                        self._config.url,
                        headers={"Mcp-Session-Id": self._session_id},
                    )
                except Exception:
                    pass  # Best-effort session cleanup

            await self._client.aclose()
            self._client = None

        self._session_id = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    # ------------------------------------------------------------------
    # SSE parsing
    # ------------------------------------------------------------------

    async def _parse_sse_response(self, body: str) -> None:
        """Parse SSE event stream and queue all JSON-RPC messages."""
        for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = line.strip()
            if not line or line.startswith(":"):
                continue  # comment or empty line
            if line.startswith("data:"):
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    msg = deserialize_message(data.encode())
                    await self._inbox.put(msg)
                except Exception as exc:
                    logger.warning(
                        "Failed to parse SSE data from '%s': %s",
                        self._config.name, exc,
                    )
