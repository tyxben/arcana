"""Context page table -- stores evicted message content for on-demand recall.

Part of the Virtual Memory system: when the context builder compresses
messages, the full content is stored here. The LLM can retrieve it via
the built-in recall tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from arcana.contracts.llm import Message


@dataclass
class ContextPageTable:
    """In-memory page table for evicted conversation messages.

    Stores full-fidelity messages that were compressed or evicted during
    context building. Each eviction produces a chunk_id that the LLM can
    use with the recall tool to retrieve the original content.

    This is an in-memory, run-scoped structure -- no external storage needed.
    """

    _pages: dict[str, list[Message]] = field(default_factory=dict)
    _summaries: dict[str, str] = field(default_factory=dict)

    def evict(self, messages: list[Message], summary: str) -> str:
        """Store messages and return a chunk_id for later recall.

        Args:
            messages: The original full-fidelity messages being compressed.
            summary: A brief description of what was evicted (for the index).

        Returns:
            A chunk_id string that can be passed to recall().
        """
        chunk_id = f"ctx-{uuid4().hex[:8]}"
        self._pages[chunk_id] = list(messages)
        self._summaries[chunk_id] = summary
        return chunk_id

    def recall(self, chunk_id: str) -> list[Message] | None:
        """Retrieve evicted messages by chunk_id.

        Returns None if chunk_id not found (e.g., typo or hallucination).
        """
        return self._pages.get(chunk_id)

    def get_summary(self, chunk_id: str) -> str:
        """Get the summary for a chunk_id."""
        return self._summaries.get(chunk_id, "")

    @property
    def has_pages(self) -> bool:
        """True if any pages have been evicted (recall tool should be offered)."""
        return len(self._pages) > 0

    @property
    def page_count(self) -> int:
        """Number of evicted page chunks."""
        return len(self._pages)

    def index(self) -> list[dict[str, str]]:
        """Return a compact index of all stored chunks for context injection."""
        return [
            {"chunk_id": cid, "summary": self._summaries.get(cid, "")}
            for cid in self._pages
        ]

    def clear(self) -> None:
        """Clear all stored pages (call at start of new run)."""
        self._pages.clear()
        self._summaries.clear()
