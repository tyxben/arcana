"""Graph interrupt and resume commands."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Command(BaseModel):
    """Command for resuming from an interrupt."""

    resume: Any = None  # Value passed to the interrupted node
    update: dict[str, Any] | None = None  # State updates to apply
    goto: str | None = None  # Jump to a specific node


class GraphInterrupt(Exception):
    """Raised when graph execution is interrupted (human-in-the-loop)."""

    def __init__(
        self,
        *,
        node_id: str,
        state: dict[str, Any],
        checkpoint_id: str,
        message: str = "",
    ) -> None:
        self.node_id = node_id
        self.state = state
        self.checkpoint_id = checkpoint_id
        super().__init__(message or f"Graph interrupted at node '{node_id}'")
