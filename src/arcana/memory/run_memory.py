"""Run Memory -- stores key facts from each run for cross-run context."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    """A single fact extracted from a run."""

    content: str
    source_run_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunMemoryStore:
    """Simple in-memory store for cross-run facts.

    After each run: store goal + answer as facts.
    Before each run: retrieve recent facts as context.

    This is intentionally simple. No vector search, no embeddings.
    Just the most recent N facts, injected into the system prompt.
    """

    def __init__(self, max_facts: int = 50) -> None:
        self._facts: list[MemoryFact] = []
        self._max_facts = max_facts

    def store(self, content: str, run_id: str = "", **metadata: Any) -> None:
        """Store a fact."""
        fact = MemoryFact(
            content=content,
            source_run_id=run_id,
            metadata=metadata,
        )
        self._facts.append(fact)
        # Evict oldest if over limit
        if len(self._facts) > self._max_facts:
            self._facts = self._facts[-self._max_facts :]

    def store_run_result(self, goal: str, answer: str, run_id: str = "") -> None:
        """Store key facts from a completed run."""
        if goal:
            self.store(f"User asked: {goal}", run_id=run_id, type="goal")
        if answer:
            # Truncate long answers to key info
            summary = answer[:500] if len(answer) > 500 else answer
            self.store(f"Result: {summary}", run_id=run_id, type="answer")

    def get_context(self, max_facts: int = 10) -> str:
        """Get recent facts as a context string for injection."""
        if not self._facts:
            return ""

        recent = self._facts[-max_facts:]
        lines = ["[Memory from previous interactions]"]
        for fact in recent:
            lines.append(f"- {fact.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored facts."""
        self._facts.clear()

    @property
    def fact_count(self) -> int:
        """Number of stored facts."""
        return len(self._facts)
