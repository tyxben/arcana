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
            raise ValueError(
                f"MCP server '{self._config.name}' has no command configured. "
                f"Set command in MCPServerConfig, e.g.: "
                f"MCPServerConfig(name='{self._config.name}', command='npx', args=['-y', 'your-mcp-server'])"
            )

        cmd = [self._config.command, *self._config.args]
        env = {**dict(os.environ), **self._config.env} if self._config.env else None

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Brief check: if the process exited immediately, fail fast.
        await asyncio.sleep(0.5)
        if self._process.returncode is not None:
            stderr_bytes = await self._process.stderr.read() if self._process.stderr else b""
            stderr_text = stderr_bytes.decode(errors="replace").strip()
            raise ConnectionError(
                f"MCP server process exited immediately (code={self._process.returncode}). "
                f"stderr: {stderr_text[:500]}"
            )

        self._connected = True
        logger.info("StdioTransport connected: %s", " ".join(cmd))

    async def send(self, message: MCPMessage) -> None:
        if not self._process or not self._process.stdin:
            raise ConnectionError(
                f"MCP server '{self._config.name}' transport not connected. "
                f"Call connect() before sending messages."
            )

        data = serialize_message(message)
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def receive(self) -> MCPMessage:
        if not self._process or not self._process.stdout:
            raise ConnectionError(
                f"MCP server '{self._config.name}' transport not connected. "
                f"Call connect() before receiving messages."
            )

        line = await asyncio.wait_for(
            self._process.stdout.readline(),
            timeout=self._config.timeout_ms / 1000,
        )

        if not line:
            cmd_str = f"{self._config.command} {' '.join(self._config.args)}" if self._config.command else "(unknown)"
            raise ConnectionError(
                f"MCP server '{self._config.name}' closed the connection unexpectedly "
                f"(command: {cmd_str}). "
                f"The server process may have crashed. Check its stderr output for details."
            )

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
