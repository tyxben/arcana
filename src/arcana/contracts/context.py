"""Context management contracts for Working Set."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ContextLayer(str, Enum):
    IDENTITY = "identity"
    TASK = "task"
    WORKING = "working"
    EXTERNAL = "external"


class ContextBlock(BaseModel):
    """A discrete block of context content."""

    layer: ContextLayer
    key: str  # e.g., "tool:file_read", "history:step_3"
    content: str
    token_count: int
    priority: float = 0.5  # 0.0 (drop first) to 1.0 (drop last)
    compressible: bool = True
    source: str | None = None


class TokenBudget(BaseModel):
    """Token allocation for a single LLM call."""

    total_window: int = 128000
    identity_tokens: int = 200
    task_tokens: int = 300
    response_reserve: int = 4096

    # Per-layer hard caps (None = no cap, use remaining budget)
    tool_budget: int | None = None
    history_budget: int | None = None
    memory_budget: int | None = None

    @property
    def working_budget(self) -> int:
        return (
            self.total_window
            - self.identity_tokens
            - self.task_tokens
            - self.response_reserve
        )


class StepContext(BaseModel):
    """What the current step needs."""

    step_type: str = "think"
    needs_tools: bool = False
    needs_memory: bool = False
    relevant_tool_names: list[str] | None = None
    memory_query: str | None = None
    previous_error: dict[str, Any] | None = None  # ErrorDiagnosis dict
    focus_instruction: str | None = None


class WorkingSet(BaseModel):
    """The assembled context for a single LLM call."""

    identity: ContextBlock
    task: ContextBlock
    working_blocks: list[ContextBlock] = Field(default_factory=list)
    total_tokens: int = 0
    dropped_keys: list[str] = Field(default_factory=list)
    compressed_keys: list[str] = Field(default_factory=list)


class MessageDecision(BaseModel):
    """Structured per-message evidence for context composition.

    Every message in the input to WorkingSetBuilder produces one of these
    in the resulting ContextDecision. Answers, for each message: was it
    kept verbatim, compressed (and to what fidelity), dropped, or folded
    into a summary?
    """

    index: int  # position in the original messages list
    role: str  # user / assistant / tool / system
    outcome: Literal["kept", "compressed", "dropped", "summarized"]

    # Populated only when the path applies fidelity grading
    fidelity: Literal["L0", "L1", "L2", "L3"] | None = None
    # Populated only when the path scores messages
    relevance_score: float | None = None

    token_count_before: int
    token_count_after: int = 0  # 0 when dropped

    # Short machine-readable reason, e.g.:
    #   "passthrough", "tail_preserve_head", "tail_preserve_tail",
    #   "tail_preserve_middle_compressed", "stale_tool_result",
    #   "aggressive_truncate_kept", "aggressive_truncate_drop",
    #   "llm_summarized_into_single", "relevance_compression"
    reason: str = ""


class ContextDecision(BaseModel):
    """Record of why context was composed this way for a single LLM call.

    Every turn, the WorkingSetBuilder produces one of these. It answers:
    - Was anything compressed or dropped?
    - How full is the context window?
    - Where did the tokens go?
    - What information was lost?
    """

    turn: int = 0
    strategy: str = "passthrough"

    # Budget breakdown (tokens)
    budget_total: int = 0
    budget_used: int = 0
    budget_tools: int = 0
    budget_reserve: int = 0

    # Message counts
    messages_in: int = 0
    messages_out: int = 0
    compressed_count: int = 0

    # Flags
    memory_injected: bool = False
    history_compressed: bool = False

    # Structured per-message evidence (one entry per input message)
    decisions: list[MessageDecision] = Field(default_factory=list)

    # What was compressed/dropped (message role:summary pairs)
    # Kept for backward compatibility; authoritative evidence is in `decisions`.
    compressed_messages: list[str] = Field(default_factory=list)

    # Human-readable explanation
    explanation: str = ""


class ContextStrategy(BaseModel):
    """Configuration for adaptive context compression strategy.

    Defines thresholds for when to apply different compression levels
    based on context window utilization ratio.

    Modes:
        - "adaptive": Select strategy based on utilization thresholds (default)
        - "off": Never compress
        - "always_compress": Always apply compression
    """

    mode: str = "adaptive"

    # Thresholds (as fraction of window_size, must be ascending)
    passthrough_threshold: float = 0.50   # Below: no compression
    tail_preserve_threshold: float = 0.75  # 0.50-0.75: compress middle, keep tail
    llm_summarize_threshold: float = 0.90  # 0.75-0.90: use LLM summarization
    # Above 0.90: aggressive truncate

    # tail_preserve config
    tail_preserve_keep_recent: int = 6  # Recent messages to keep verbatim

    # aggressive_truncate config
    aggressive_keep_turns: int = 2  # Keep only last N user+assistant pairs

    # Tool result pruning (Phase 0 — zero-cost, applied before compression)
    tool_result_staleness_turns: int = 4    # tool results older than N turns are stale
    tool_result_prune_max_chars: int = 200  # keep this many chars as summary


class ContextReport(BaseModel):
    """Report of how context was composed for a single LLM call.

    Produced by WorkingSetBuilder and attached to RunResult, ChatResponse,
    TraceEvent, and StreamEvent for full visibility into context window usage.
    """

    # Turn metadata
    turn: int = 0
    strategy_used: str = "passthrough"

    # Token allocation breakdown
    total_tokens: int = 0
    identity_tokens: int = 0
    task_tokens: int = 0
    tools_tokens: int = 0
    history_tokens: int = 0
    memory_tokens: int = 0

    # Compression metrics
    compression_applied: bool = False
    compression_savings: int = 0
    messages_compressed: int = 0

    # Utilization
    window_size: int = 128_000
    utilization: float = 0.0

    # Phase 0: tool result pruning stats
    tool_results_pruned: int = 0        # how many tool results Phase 0 pruned
    tool_results_tokens_saved: int = 0  # tokens saved by Phase 0

    # Fidelity distribution (when compression uses fidelity spectrum)
    fidelity_distribution: dict[str, int] = Field(default_factory=dict)

    # Tool loading info (for lazy registry)
    tools_loaded: int = 0
    tools_available: int = 0
