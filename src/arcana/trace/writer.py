"""TraceWriter - JSONL-based trace event writer."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arcana.contracts.trace import TraceContext, TraceEvent


class TraceWriter:
    """
    Writes trace events to JSONL files.

    Each run gets its own trace file: {trace_dir}/{run_id}.jsonl
    Events are appended atomically with file locking.
    """

    def __init__(
        self,
        trace_dir: str | Path = "./traces",
        enabled: bool = True,
        namespace: str | None = None,
    ):
        """
        Initialize the trace writer.

        Args:
            trace_dir: Directory to store trace files
            enabled: Whether tracing is enabled
            namespace: Optional namespace for tenant isolation. When set,
                trace files are written to ``{trace_dir}/{namespace}/``
                instead of ``trace_dir`` directly.
        """
        base_dir = Path(trace_dir)
        self.trace_dir = base_dir / namespace if namespace else base_dir
        self.enabled = enabled
        self._lock = threading.Lock()

        if self.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def _get_trace_file(self, run_id: str) -> Path:
        """Get the trace file path for a run."""
        return self.trace_dir / f"{run_id}.jsonl"

    def write(self, event: TraceEvent) -> None:
        """
        Write a trace event to the appropriate file.

        Args:
            event: The trace event to write
        """
        if not self.enabled:
            return

        trace_file = self._get_trace_file(event.run_id)

        # Serialize event to JSON
        event_json = event.model_dump_json()

        # Append atomically with lock
        with self._lock:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(event_json + "\n")

    def write_raw(self, run_id: str, data: dict[str, Any]) -> None:
        """
        Write raw data to the trace file.

        Args:
            run_id: The run ID
            data: Raw data dictionary to write
        """
        if not self.enabled:
            return

        trace_file = self._get_trace_file(run_id)

        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(UTC).isoformat()

        json_line = json.dumps(data, ensure_ascii=False, default=str)

        with self._lock:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")

    def create_context(self, task_id: str | None = None) -> TraceContext:
        """
        Create a new trace context.

        Args:
            task_id: Optional task ID

        Returns:
            New TraceContext instance
        """
        return TraceContext(task_id=task_id)

    def exists(self, run_id: str) -> bool:
        """Check if a trace file exists for the given run."""
        return self._get_trace_file(run_id).exists()

    def delete(self, run_id: str) -> bool:
        """
        Delete a trace file.

        Args:
            run_id: The run ID

        Returns:
            True if deleted, False if not found
        """
        trace_file = self._get_trace_file(run_id)
        if trace_file.exists():
            trace_file.unlink()
            return True
        return False

    def list_runs(self) -> list[str]:
        """
        List all run IDs with trace files.

        Returns:
            List of run IDs
        """
        if not self.trace_dir.exists():
            return []

        return [
            f.stem for f in self.trace_dir.glob("*.jsonl") if f.is_file()
        ]
