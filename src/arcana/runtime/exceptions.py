"""Runtime-specific exceptions."""

from __future__ import annotations


class RuntimeError(Exception):
    """Base exception for runtime errors."""

    def __init__(self, message: str, *, recoverable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.recoverable = recoverable


class StateTransitionError(RuntimeError):
    """Invalid state transition attempted."""

    def __init__(self, from_status: str, to_status: str) -> None:
        super().__init__(
            f"Invalid state transition: {from_status} -> {to_status}",
            recoverable=False,
        )
        self.from_status = from_status
        self.to_status = to_status


class CheckpointError(RuntimeError):
    """Error during checkpoint operations."""

    def __init__(self, message: str, *, run_id: str | None = None) -> None:
        super().__init__(message, recoverable=False)
        self.run_id = run_id


class HashVerificationError(CheckpointError):
    """State hash verification failed."""

    def __init__(self, expected: str, actual: str, run_id: str | None = None) -> None:
        super().__init__(
            f"Hash verification failed: expected {expected}, got {actual}",
            run_id=run_id,
        )
        self.expected = expected
        self.actual = actual


class StepExecutionError(RuntimeError):
    """Error during step execution."""

    def __init__(
        self,
        message: str,
        *,
        step_id: str | None = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(message, recoverable=recoverable)
        self.step_id = step_id


class PolicyError(RuntimeError):
    """Error in policy decision making."""

    def __init__(self, message: str, *, policy_name: str | None = None) -> None:
        super().__init__(message, recoverable=False)
        self.policy_name = policy_name


class ProgressStallError(RuntimeError):
    """Agent has stalled and is not making progress."""

    def __init__(self, consecutive_no_progress: int) -> None:
        super().__init__(
            f"Agent stalled after {consecutive_no_progress} steps without progress",
            recoverable=False,
        )
        self.consecutive_no_progress = consecutive_no_progress
