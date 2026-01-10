"""StateManager - state transitions and checkpointing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from arcana.contracts.state import AgentState, ExecutionStatus, StateSnapshot
from arcana.contracts.trace import AgentRole, EventType, TraceEvent
from arcana.runtime.exceptions import HashVerificationError, StateTransitionError
from arcana.utils.hashing import canonical_hash, verify_hash

if TYPE_CHECKING:
    from arcana.contracts.runtime import RuntimeConfig
    from arcana.contracts.trace import TraceContext
    from arcana.trace.writer import TraceWriter


# Valid state transitions
VALID_TRANSITIONS: dict[ExecutionStatus, set[ExecutionStatus]] = {
    ExecutionStatus.PENDING: {ExecutionStatus.RUNNING},
    ExecutionStatus.RUNNING: {
        ExecutionStatus.PAUSED,
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
    },
    ExecutionStatus.PAUSED: {ExecutionStatus.RUNNING, ExecutionStatus.CANCELLED},
    ExecutionStatus.COMPLETED: set(),
    ExecutionStatus.FAILED: set(),
    ExecutionStatus.CANCELLED: set(),
}


class StateManager:
    """
    Manages agent state transitions and checkpointing.

    Responsibilities:
    - Validate and apply state transitions
    - Create and verify state snapshots
    - Persist checkpoints to storage
    """

    def __init__(
        self,
        *,
        trace_writer: TraceWriter | None = None,
        config: RuntimeConfig | None = None,
        checkpoint_dir: str | Path = "./checkpoints",
    ) -> None:
        """
        Initialize the state manager.

        Args:
            trace_writer: Optional trace writer for logging
            config: Runtime configuration
            checkpoint_dir: Directory for checkpoint storage
        """
        self.trace_writer = trace_writer
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)

    def transition(
        self,
        state: AgentState,
        new_status: ExecutionStatus,
    ) -> AgentState:
        """
        Transition state to a new status.

        Args:
            state: Current state
            new_status: Target status

        Returns:
            Updated state

        Raises:
            StateTransitionError: If transition is invalid
        """
        valid_next = VALID_TRANSITIONS.get(state.status, set())
        if new_status not in valid_next:
            raise StateTransitionError(state.status.value, new_status.value)

        state.status = new_status
        return state

    async def checkpoint(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
        reason: str = "step_complete",
    ) -> StateSnapshot:
        """
        Create a checkpoint of current state.

        Args:
            state: State to checkpoint
            trace_ctx: Trace context
            reason: Reason for checkpoint

        Returns:
            Created snapshot
        """
        # Compute state hash (excluding volatile fields)
        serializable = state.model_dump(exclude={"start_time", "elapsed_ms"})
        state_hash = canonical_hash(serializable)

        # Create snapshot
        snapshot = StateSnapshot(
            run_id=state.run_id,
            step_id=trace_ctx.new_step_id(),
            state_hash=state_hash,
            state=state,
            checkpoint_reason=reason,
            is_resumable=state.status
            in {
                ExecutionStatus.RUNNING,
                ExecutionStatus.PAUSED,
            },
        )

        # Persist to storage
        await self._persist_snapshot(snapshot)

        # Log checkpoint event
        if self.trace_writer:
            event = TraceEvent(
                run_id=state.run_id,
                task_id=state.task_id,
                step_id=snapshot.step_id,
                role=AgentRole.SYSTEM,
                event_type=EventType.CHECKPOINT,
                state_after_hash=state_hash,
                metadata={"checkpoint_reason": reason},
            )
            self.trace_writer.write(event)

        return snapshot

    async def load_snapshot(
        self,
        run_id: str,
        step_id: str | None = None,
    ) -> StateSnapshot | None:
        """
        Load a snapshot from storage.

        Args:
            run_id: Run ID
            step_id: Optional specific step ID (latest if None)

        Returns:
            Loaded snapshot or None if not found
        """
        checkpoint_file = self._get_checkpoint_file(run_id)
        if not checkpoint_file.exists():
            return None

        snapshots = self._read_checkpoints(checkpoint_file)

        if step_id:
            # Find specific snapshot
            for snapshot in snapshots:
                if snapshot.step_id == step_id:
                    return snapshot
            return None
        else:
            # Return latest
            return snapshots[-1] if snapshots else None

    def verify_snapshot(self, snapshot: StateSnapshot) -> bool:
        """
        Verify snapshot integrity.

        Args:
            snapshot: Snapshot to verify

        Returns:
            True if valid

        Raises:
            HashVerificationError: If verification fails
        """
        serializable = snapshot.state.model_dump(exclude={"start_time", "elapsed_ms"})
        if not verify_hash(serializable, snapshot.state_hash):
            actual = canonical_hash(serializable)
            raise HashVerificationError(
                expected=snapshot.state_hash,
                actual=actual,
                run_id=snapshot.run_id,
            )
        return True

    async def _persist_snapshot(self, snapshot: StateSnapshot) -> None:
        """Persist snapshot to storage."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = self._get_checkpoint_file(snapshot.run_id)

        # Append to JSONL file
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(snapshot.model_dump_json() + "\n")

    def _get_checkpoint_file(self, run_id: str) -> Path:
        """Get checkpoint file path for a run."""
        return self.checkpoint_dir / f"{run_id}.checkpoints.jsonl"

    def _read_checkpoints(self, path: Path) -> list[StateSnapshot]:
        """Read all checkpoints from file."""
        snapshots = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        snapshots.append(StateSnapshot.model_validate(data))
                    except (json.JSONDecodeError, ValueError):
                        continue
        return snapshots
