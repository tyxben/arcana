"""SSE (Server-Sent Events) adapter for streaming."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from arcana.contracts.streaming import StreamEvent


async def stream_to_sse(
    events: AsyncGenerator[StreamEvent, None],
) -> AsyncGenerator[str, None]:
    """Convert StreamEvent stream to SSE format for HTTP responses."""
    async for event in events:
        data = event.model_dump_json()
        yield f"event: {event.event_type.value}\ndata: {data}\n\n"
