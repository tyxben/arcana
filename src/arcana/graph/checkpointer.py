"""Graph-level checkpoint for interrupt/resume support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class GraphCheckpointer:
    """
    Persists graph execution state for interrupt/resume (human-in-the-loop).

    Stores checkpoints as JSONL files, following the same pattern
    as arcana's StateManager.
    """

    def __init__(self, checkpoint_dir: str | Path = "./checkpoints/graph") -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def save(
        self,
        state: dict[str, Any],
        node_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a checkpoint and return its ID.

        Args:
            state: Current graph state
            node_id: Node where execution was interrupted
            metadata: Additional checkpoint metadata

        Returns:
            checkpoint_id: Unique ID for this checkpoint
        """
        checkpoint_id = str(uuid4())
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "state": state,
            "node_id": node_id,
            "resume_node": node_id,
            **(metadata or {}),
        }

        checkpoint_file = self._checkpoint_dir / f"{checkpoint_id}.json"
        checkpoint_file.write_text(
            json.dumps(checkpoint, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> dict[str, Any] | None:
        """
        Load a checkpoint by ID.

        Returns:
            Checkpoint data dict, or None if not found
        """
        checkpoint_file = self._checkpoint_dir / f"{checkpoint_id}.json"
        if not checkpoint_file.exists():
            return None

        data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        return data  # type: ignore[no-any-return]

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint. Returns True if it existed."""
        checkpoint_file = self._checkpoint_dir / f"{checkpoint_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False
