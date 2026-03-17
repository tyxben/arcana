"""Stdio transport -- communicates with MCP server via subprocess stdin/stdout."""

from __future__ import annotations

import asyncio
import logging
import os

from arcana.contracts.mcp import MCPMessage, MCPServerConfig
from arcana.mcp.protocol import deserialize_message, serialize_message
from arcana.mcp.transport.base import MCPTransport

logger = logging.getLogger(__name__)


class StdioTransport(MCPTransport):
    """Transport via subprocess stdin/stdout."""

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._connected = False

    async def connect(self) -> None:
        if not self._config.command:
            raise ValueError("StdioTransport requires a command")

        cmd = [self._config.command, *self._config.args]
        env = {**dict(os.environ), **self._config.env} if self._config.env else None

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._connected = True
        logger.info("StdioTransport connected: %s", " ".join(cmd))

    async def send(self, message: MCPMessage) -> None:
        if not self._process or not self._process.stdin:
            raise ConnectionError("Transport not connected")

        data = serialize_message(message)
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def receive(self) -> MCPMessage:
        if not self._process or not self._process.stdout:
            raise ConnectionError("Transport not connected")

        line = await asyncio.wait_for(
            self._process.stdout.readline(),
            timeout=self._config.timeout_ms / 1000,
        )

        if not line:
            raise ConnectionError("MCP server closed connection")

        return deserialize_message(line)

    async def close(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except TimeoutError:
                self._process.kill()
            self._process = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return (
            self._connected
            and self._process is not None
            and self._process.returncode is None
        )
