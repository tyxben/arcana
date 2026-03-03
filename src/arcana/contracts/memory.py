"""Memory-related contracts for the agent memory system."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Type of memory store."""

    WORKING = "working"  # Short-lived, run-scoped KV
    LONG_TERM = "long_term"  # Persistent, vector-indexed facts
    EPISODIC = "episodic"  # Event trajectory logs


class MemoryEntry(BaseModel):
    """A single memory entry, regardless of memory type."""

    id: str
    memory_type: MemoryType
    key: str
    content: str

    # Governance metadata
    confidence: float = 1.0  # 0.0 to 1.0
    source: str = ""  # Origin: "tool", "llm", "user", "step_result", etc.
    source_run_id: str | None = None
    source_step_id: str | None = None

    # Lifecycle
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Revocation (soft-delete with history preservation)
    revoked: bool = False
    revoked_at: datetime | None = None
    revoked_reason: str | None = None

    # Classification
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Integrity
    content_hash: str = ""


class MemoryQuery(BaseModel):
    """Query parameters for memory retrieval."""

    query: str = ""  # Semantic search text (long-term)
    key: str | None = None  # Exact key lookup (working)
    memory_type: MemoryType | None = None  # Filter by type
    tags: list[str] = Field(default_factory=list)
    run_id: str | None = None  # Filter by run
    top_k: int = 10
    min_confidence: float = 0.0
    include_revoked: bool = False  # Default: hide revoked entries


class MemoryWriteRequest(BaseModel):
    """Request to write a memory entry."""

    memory_type: MemoryType
    key: str
    content: str
    confidence: float = 1.0
    source: str = "agent"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    step_id: str | None = None


class MemoryWriteResult(BaseModel):
    """Result of a memory write operation."""

    success: bool
    entry_id: str | None = None
    rejected_reason: str | None = None
    confidence_below_threshold: bool = False


class RevocationRequest(BaseModel):
    """Request to revoke a memory entry."""

    entry_id: str
    reason: str
    revoked_by: str = "system"


class MemoryConfig(BaseModel):
    """Configuration for the memory system."""

    # Write governance thresholds
    min_write_confidence: float = 0.5
    warn_confidence_threshold: float = 0.7

    # Working memory
    working_namespace_prefix: str = "wm"

    # Long-term memory
    embedding_model: str = "text-embedding-ada-002"

    # Episodic memory
    max_episodic_results: int = 50
