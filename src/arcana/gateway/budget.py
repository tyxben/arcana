"""Budget tracking for LLM usage."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from arcana.contracts.llm import Budget, TokenUsage
from arcana.contracts.trace import BudgetSnapshot
from arcana.gateway.base import BudgetExceededError


@dataclass
class BudgetTracker:
    """
    Tracks token usage, cost, and time against budget limits.

    Thread-safe budget tracking with real-time limit enforcement.
    """

    # Limits
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_time_ms: int | None = None

    # Consumed
    tokens_used: int = field(default=0)
    cost_usd: float = field(default=0.0)
    start_time_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @classmethod
    def from_budget(cls, budget: Budget | None) -> BudgetTracker:
        """Create a tracker from a Budget object."""
        if budget is None:
            return cls()

        return cls(
            max_tokens=budget.max_tokens,
            max_cost_usd=budget.max_cost_usd,
            max_time_ms=budget.max_time_ms,
        )

    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        return int(time.time() * 1000) - self.start_time_ms

    @property
    def tokens_remaining(self) -> int | None:
        """Get remaining token budget."""
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def cost_remaining(self) -> float | None:
        """Get remaining cost budget."""
        if self.max_cost_usd is None:
            return None
        return max(0.0, self.max_cost_usd - self.cost_usd)

    @property
    def time_remaining_ms(self) -> int | None:
        """Get remaining time budget in milliseconds."""
        if self.max_time_ms is None:
            return None
        return max(0, self.max_time_ms - self.elapsed_ms)

    def check_budget(self) -> None:
        """
        Check if any budget limit has been exceeded.

        Raises:
            BudgetExceededError: If a budget limit is exceeded
        """
        with self._lock:
            if self.max_tokens and self.tokens_used >= self.max_tokens:
                raise BudgetExceededError(
                    f"Token budget exceeded: used {self.tokens_used:,} / limit {self.max_tokens:,} tokens. "
                    f"Increase budget with Budget(max_tokens={self.max_tokens * 2:,}) or reduce max_turns.",
                    budget_type="tokens",
                )

            if self.max_cost_usd and self.cost_usd >= self.max_cost_usd:
                raise BudgetExceededError(
                    f"Cost budget exceeded: spent ${self.cost_usd:.4f} / limit ${self.max_cost_usd:.4f}. "
                    f"Increase budget with Budget(max_cost_usd={self.max_cost_usd * 2:.2f}) or reduce max_turns.",
                    budget_type="cost",
                )

            if self.max_time_ms and self.elapsed_ms >= self.max_time_ms:
                raise BudgetExceededError(
                    f"Time budget exceeded: elapsed {self.elapsed_ms:,}ms / limit {self.max_time_ms:,}ms. "
                    f"Increase budget with Budget(max_time_ms={self.max_time_ms * 2:,}) or simplify the task.",
                    budget_type="time",
                )

    def add_usage(self, usage: TokenUsage, cost: float | None = None) -> None:
        """
        Add token usage and optionally cost.

        Args:
            usage: Token usage from LLM response
            cost: Optional explicit cost (uses estimate if not provided)
        """
        with self._lock:
            self.tokens_used += usage.total_tokens
            self.cost_usd += cost if cost is not None else usage.cost_estimate

    def can_afford(self, estimated_tokens: int) -> bool:
        """
        Check if the budget can afford an estimated token count.

        Args:
            estimated_tokens: Estimated tokens for the next call

        Returns:
            True if affordable, False otherwise
        """
        if self.max_tokens and (self.tokens_used + estimated_tokens) > self.max_tokens:
            return False
        return True

    def to_snapshot(self) -> BudgetSnapshot:
        """Create a snapshot of current budget state."""
        return BudgetSnapshot(
            max_tokens=self.max_tokens,
            max_cost_usd=self.max_cost_usd,
            max_time_ms=self.max_time_ms,
            tokens_used=self.tokens_used,
            cost_usd=self.cost_usd,
            time_ms=self.elapsed_ms,
        )

    def reset(self) -> None:
        """Reset usage counters."""
        with self._lock:
            self.tokens_used = 0
            self.cost_usd = 0.0
            self.start_time_ms = int(time.time() * 1000)
