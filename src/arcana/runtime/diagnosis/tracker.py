"""Recovery state tracking -- prevents infinite recovery loops.

``RecoveryTracker`` is the only stateful component in the diagnostic
recovery system.  All diagnoser functions are pure; this class
accumulates history so that diagnosers can make escalation decisions.
"""

from __future__ import annotations

from collections import Counter

from arcana.contracts.diagnosis import ErrorCategory, ErrorDiagnosis


class RecoveryTracker:
    """Track error recovery attempts per (tool, category) pair.

    Keyed by ``(tool_name, error_category)`` to detect repeated failures
    of the same kind for the same tool.

    Parameters
    ----------
    max_recoveries_per_category:
        Maximum recovery attempts for the same ``(tool, category)`` pair
        before ``should_escalate`` returns ``True``.
    max_total_recoveries:
        Maximum total recovery attempts across all tools/categories
        before ``should_abort`` returns ``True``.
    """

    def __init__(
        self,
        max_recoveries_per_category: int = 2,
        max_total_recoveries: int = 5,
    ) -> None:
        self._history: list[ErrorDiagnosis] = []
        self._by_key: dict[tuple[str, str], list[ErrorDiagnosis]] = {}
        self._max_per_category = max_recoveries_per_category
        self._max_total = max_total_recoveries

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, diagnosis: ErrorDiagnosis) -> None:
        """Record a diagnosis in the tracker's history."""
        self._history.append(diagnosis)
        tool = diagnosis.related_tool or "__unknown__"
        key = (tool, diagnosis.error_category.value)
        self._by_key.setdefault(key, []).append(diagnosis)

    # ------------------------------------------------------------------
    # Escalation queries
    # ------------------------------------------------------------------

    def should_escalate(self) -> bool:
        """Whether recovery attempts should escalate to a harder strategy.

        Returns ``True`` when any single ``(tool, category)`` pair has
        exceeded ``max_recoveries_per_category``.
        """
        return any(len(v) >= self._max_per_category for v in self._by_key.values())

    def should_abort(self) -> bool:
        """Whether the system should abort recovery entirely.

        Returns ``True`` when total recovery attempts exceed
        ``max_total_recoveries`` **or** the same error category appears
        repeatedly across different tools (indicating a systemic issue).
        """
        if len(self._history) >= self._max_total:
            return True
        # Systemic check: same category across 3+ distinct tools
        cat_tools: dict[str, set[str]] = {}
        for diag in self._history:
            cat = diag.error_category.value
            tool = diag.related_tool or "__unknown__"
            cat_tools.setdefault(cat, set()).add(tool)
        return any(len(tools) >= 3 for tools in cat_tools.values())

    # ------------------------------------------------------------------
    # Pattern detection
    # ------------------------------------------------------------------

    def get_pattern(self) -> str | None:
        """Detect a recurring error pattern in the history.

        Returns a human-readable description of the pattern, or ``None``
        if no notable pattern is detected.

        Detected patterns:
        - Consecutive identical categories (e.g. "3 consecutive FACT_ERROR")
        - Dominant category (>60% of all errors)
        """
        if len(self._history) < 2:
            return None

        # Check consecutive identical categories (look at last N)
        consecutive_count = 1
        for i in range(len(self._history) - 1, 0, -1):
            if self._history[i].error_category == self._history[i - 1].error_category:
                consecutive_count += 1
            else:
                break
        if consecutive_count >= 3:
            cat = self._history[-1].error_category.value
            return f"{consecutive_count} consecutive {cat} errors"

        # Check dominant category
        counts = Counter(d.error_category.value for d in self._history)
        total = len(self._history)
        for cat, count in counts.most_common(1):
            if count / total > 0.6 and count >= 3:
                return f"{cat} is dominant ({count}/{total} errors)"

        return None

    # ------------------------------------------------------------------
    # History access
    # ------------------------------------------------------------------

    def get_history(self, tool_name: str | None = None) -> list[ErrorDiagnosis]:
        """Return diagnosis history, optionally filtered by tool name."""
        if tool_name is None:
            return list(self._history)
        return [d for d in self._history if d.related_tool == tool_name]

    def total_attempts(self) -> int:
        """Total number of recorded recovery attempts."""
        return len(self._history)

    def attempts_for(self, tool_name: str, category: ErrorCategory) -> int:
        """Count attempts for a specific (tool, category) pair."""
        key = (tool_name, category.value)
        return len(self._by_key.get(key, []))

    def reset(self) -> None:
        """Clear all tracked history."""
        self._history.clear()
        self._by_key.clear()
