"""Run Memory — cross-run fact storage with relevance retrieval.

Core principle: memory is a retrieval problem, not a storage problem.
Store everything, but only inject what's relevant to the current goal.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    """A single fact with relevance metadata."""

    content: str
    source_run_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    importance: float = 0.5  # 0.0 (trivial) to 1.0 (critical)
    tags: list[str] = Field(default_factory=list)
    access_count: int = 0  # How many times retrieved
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunMemoryStore:
    """Cross-run fact storage with relevance-based retrieval.

    Stores facts from completed runs. Before each new run, retrieves
    the most relevant facts within a token budget.

    Retrieval ranking factors:
    1. Keyword relevance (query ↔ fact word overlap)
    2. Recency (newer facts score higher, exponential decay)
    3. Importance (user-assigned or auto-detected)
    4. Access frequency (frequently useful facts score higher)

    Args:
        namespace: Optional namespace for tenant isolation. When set,
            facts are prefixed with the namespace so that different
            tenants sharing one store don't see each other's data.
            When ``None``, no prefix is applied (backward-compatible).
    """

    def __init__(
        self,
        *,
        max_facts: int = 500,
        default_budget_tokens: int = 800,
        recency_half_life_hours: float = 24.0,
        namespace: str | None = None,
    ) -> None:
        self._facts: list[MemoryFact] = []
        self._max_facts = max_facts
        self.default_budget_tokens = default_budget_tokens
        self._half_life_seconds = recency_half_life_hours * 3600
        self._namespace = namespace

    # ------------------------------------------------------------------
    # Namespace filtering
    # ------------------------------------------------------------------

    @property
    def _namespace_facts(self) -> list[MemoryFact]:
        """Return facts visible to the current namespace.

        - namespace=None  → all facts (backward-compatible)
        - namespace=X     → only facts tagged with ``_namespace == X``
        """
        if self._namespace is None:
            return self._facts
        return [
            f for f in self._facts
            if f.metadata.get("_namespace") == self._namespace
        ]

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        *,
        run_id: str = "",
        importance: float = 0.5,
        tags: list[str] | None = None,
        **metadata: Any,
    ) -> None:
        """Store a fact."""
        # Tag fact with namespace for later filtering
        if self._namespace is not None:
            metadata["_namespace"] = self._namespace

        # Deduplicate: if very similar fact exists, update instead of append
        for existing in self._namespace_facts:
            if self._similarity(existing.content, content) > 0.85:
                existing.content = content
                existing.timestamp = datetime.now(UTC)
                existing.importance = max(existing.importance, importance)
                return

        fact = MemoryFact(
            content=content,
            source_run_id=run_id,
            importance=importance,
            tags=tags or [],
            metadata=metadata,
        )
        self._facts.append(fact)

        # Evict lowest-scoring facts if over limit
        if len(self._facts) > self._max_facts:
            self._facts.sort(key=lambda f: self._base_score(f), reverse=True)
            self._facts = self._facts[: self._max_facts]

    def store_run_result(
        self,
        goal: str,
        answer: str,
        run_id: str = "",
    ) -> None:
        """Extract and store facts from a completed run."""
        if goal:
            self.store(
                f"User asked: {goal}",
                run_id=run_id,
                importance=0.4,
                tags=["goal"],
            )
        if answer:
            summary = answer[:300] if len(answer) > 300 else answer
            # Strip [DONE] markers
            summary = summary.replace("[DONE]", "").replace("[done]", "").strip()
            if summary:
                self.store(
                    f"Result: {summary}",
                    run_id=run_id,
                    importance=0.5,
                    tags=["answer"],
                )

        # Try to extract key facts (simple heuristic)
        for fact in self._extract_key_facts(goal, answer):
            self.store(fact, run_id=run_id, importance=0.7, tags=["extracted"])

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, *, budget_tokens: int | None = None) -> str:
        """Retrieve relevant facts within a token budget.

        Args:
            query: The current goal/question to match against.
            budget_tokens: Max tokens for the context string.
                          Defaults to self.default_budget_tokens.

        Returns:
            Context string ready for injection into system prompt.
        """
        visible = self._namespace_facts
        if not visible:
            return ""

        budget = budget_tokens or self.default_budget_tokens

        # Score all facts against the query
        scored = [
            (fact, self._relevance_score(query, fact)) for fact in visible
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Pack into budget
        lines = ["[Memory from previous interactions]"]
        used_tokens = _estimate_tokens(lines[0])

        for fact, score in scored:
            if score < 0.05:
                break  # Below relevance threshold
            line = f"- {fact.content}"
            line_tokens = _estimate_tokens(line)
            if used_tokens + line_tokens > budget:
                break
            lines.append(line)
            used_tokens += line_tokens
            fact.access_count += 1

        if len(lines) <= 1:
            return ""  # Only header, no facts

        return "\n".join(lines)

    def get_context(self, max_facts: int = 10) -> str:
        """Simple retrieval: most recent N facts (backward compat)."""
        visible = self._namespace_facts
        if not visible:
            return ""
        recent = visible[-max_facts:]
        lines = ["[Memory from previous interactions]"]
        for fact in recent:
            lines.append(f"- {fact.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Update / Forget
    # ------------------------------------------------------------------

    def update(self, old_pattern: str, new_content: str) -> bool:
        """Update facts matching a pattern. Returns True if any updated."""
        updated = False
        pattern_lower = old_pattern.lower()
        for fact in self._facts:
            if pattern_lower in fact.content.lower():
                fact.content = new_content
                fact.timestamp = datetime.now(UTC)
                updated = True
        return updated

    def forget(self, pattern: str) -> int:
        """Remove facts matching a pattern. Returns count removed."""
        pattern_lower = pattern.lower()
        before = len(self._facts)
        self._facts = [
            f for f in self._facts if pattern_lower not in f.content.lower()
        ]
        return before - len(self._facts)

    def clear(self) -> None:
        """Clear all stored facts."""
        self._facts.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fact_count(self) -> int:
        return len(self._namespace_facts)

    @property
    def facts(self) -> list[MemoryFact]:
        """Read-only access to facts visible in the current namespace."""
        return list(self._namespace_facts)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _relevance_score(self, query: str, fact: MemoryFact) -> float:
        """Compute relevance score for a fact given a query.

        Combines: keyword match + recency + importance + access frequency.
        """
        keyword_score = self._keyword_overlap(query, fact.content)
        recency_score = self._recency_decay(fact.timestamp)
        importance_score = fact.importance
        access_score = min(fact.access_count * 0.05, 0.2)  # Cap at 0.2

        # Weighted combination
        return (
            keyword_score * 0.45
            + recency_score * 0.25
            + importance_score * 0.20
            + access_score * 0.10
        )

    def _base_score(self, fact: MemoryFact) -> float:
        """Base score for eviction ranking (no query context)."""
        recency = self._recency_decay(fact.timestamp)
        return fact.importance * 0.4 + recency * 0.4 + min(fact.access_count * 0.05, 0.2)

    def _keyword_overlap(self, query: str, content: str) -> float:
        """Word overlap between query and content. 0.0 to 1.0."""
        query_words = set(self._tokenize(query))
        content_words = set(self._tokenize(content))
        if not query_words:
            return 0.0
        overlap = query_words & content_words
        return len(overlap) / len(query_words)

    def _recency_decay(self, timestamp: datetime) -> float:
        """Exponential decay based on age. 1.0 = just now, 0.0 = ancient."""
        age_seconds = (datetime.now(UTC) - timestamp).total_seconds()
        if age_seconds <= 0:
            return 1.0
        return math.exp(-0.693 * age_seconds / self._half_life_seconds)

    def _similarity(self, a: str, b: str) -> float:
        """Simple Jaccard similarity between two texts."""
        words_a = set(self._tokenize(a))
        words_b = set(self._tokenize(b))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenization, lowercase, skip short words."""
        words = re.findall(r"\w+", text.lower())
        return [w for w in words if len(w) > 2]

    @staticmethod
    def _extract_key_facts(goal: str, answer: str) -> list[str]:
        """Extract key facts from goal + answer. Simple heuristic."""
        facts = []
        combined = f"{goal} {answer}".lower()

        # Pattern: "my name is X"
        name_match = re.search(r"(?:my name is|i am|i'm|我叫|我是)\s+(\S+)", combined)
        if name_match:
            facts.append(f"User's name: {name_match.group(1)}")

        # Pattern: "I prefer/use/like X"
        pref_match = re.search(
            r"(?:i prefer|i use|i like|i work with|我喜欢|我用)\s+(\S+(?:\s+\S+)?)",
            combined,
        )
        if pref_match:
            facts.append(f"User preference: {pref_match.group(1)}")

        return facts


def _estimate_tokens(text: str) -> int:
    """Rough token estimation."""
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_count = len(text) - cjk_count
    return (cjk_count // 2) + (other_count // 4) + 1
