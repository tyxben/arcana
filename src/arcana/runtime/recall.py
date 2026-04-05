"""Built-in recall tool -- lets the LLM retrieve compressed conversation context.

Part of the Virtual Memory system. When conversation history is compressed,
the LLM can use this tool to retrieve the full original content on demand.

Per the Constitution: the framework provides the capability (recall),
the LLM decides whether and when to use it. Recall is a capability,
not a dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.tool import SideEffect, ToolSpec

if TYPE_CHECKING:
    from arcana.context.page_table import ContextPageTable

from arcana.contracts.tool import RECALL_TOOL_NAME

RECALL_SPEC = ToolSpec(
    name=RECALL_TOOL_NAME,
    description=(
        "Retrieve full details of earlier conversation that was compressed. "
        "Use when the compressed summary lacks information you need."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "chunk_id": {
                "type": "string",
                "description": "The chunk ID from a compression notice (e.g., 'ctx-a1b2c3d4').",
            },
        },
        "required": ["chunk_id"],
    },
    when_to_use=(
        "When you see a compressed message with a chunk_id and need the full "
        "original content to answer accurately. Only recall what you actually need."
    ),
    what_to_expect="The full original messages that were compressed, formatted as text.",
    failure_meaning=(
        "Chunk not found. The chunk_id may be incorrect. "
        "Work with the compressed summary instead."
    ),
    side_effect=SideEffect.READ,
)


class RecallHandler:
    """Handles recall tool calls by looking up the context page table."""

    def __init__(self, page_table: ContextPageTable) -> None:
        self._page_table = page_table

    async def handle(self, chunk_id: str) -> str:
        """Retrieve evicted messages and format them for the LLM."""
        messages = self._page_table.recall(chunk_id)
        if messages is None:
            return f"Chunk '{chunk_id}' not found. Available chunks: {', '.join(c['chunk_id'] for c in self._page_table.index()) or 'none'}"

        # Format messages for LLM consumption
        lines: list[str] = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"[{role}] {content}")

        return f"[Recalled context — chunk {chunk_id}]\n" + "\n".join(lines)
