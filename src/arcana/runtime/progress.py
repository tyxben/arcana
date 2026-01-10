"""ProgressDetector - monitors for loops and stuck states."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from arcana.utils.hashing import canonical_hash

if TYPE_CHECKING:
    from arcana.contracts.runtime import StepResult


class ProgressDetector:
    """
    Detects lack of progress in agent execution.

    Monitors:
    - Repeated identical outputs
    - Cyclic patterns in actions
    - Similarity between consecutive steps
    """

    def __init__(
        self,
        *,
        window_size: int = 5,
        similarity_threshold: float = 0.95,
    ) -> None:
        """
        Initialize the progress detector.

        Args:
            window_size: Number of recent steps to track
            similarity_threshold: Threshold for considering steps similar
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold

        # Recent step hashes for duplicate detection
        self._step_hashes: deque[str] = deque(maxlen=window_size)

        # Recent outputs for similarity detection
        self._recent_outputs: deque[str] = deque(maxlen=window_size)

        # Action sequence for cycle detection
        self._action_sequence: deque[str] = deque(maxlen=window_size * 2)

    def record_step(self, step_result: StepResult) -> None:
        """
        Record a step for progress tracking.

        Args:
            step_result: Result of the executed step
        """
        # Hash the step result (excluding step_id and timing)
        step_data = {
            "thought": step_result.thought,
            "action": step_result.action,
            "observation": step_result.observation,
        }
        step_hash = canonical_hash(step_data)
        self._step_hashes.append(step_hash)

        # Track outputs
        output = step_result.thought or step_result.observation or ""
        self._recent_outputs.append(output)

        # Track actions
        if step_result.action:
            self._action_sequence.append(step_result.action)

    def is_making_progress(self) -> bool:
        """
        Check if the agent is making progress.

        Returns:
            True if making progress, False if stuck
        """
        # Not enough data yet
        if len(self._step_hashes) < 2:
            return True

        # Check for exact duplicate steps
        if self._has_duplicate_steps():
            return False

        # Check for cyclic patterns
        if self._has_cyclic_pattern():
            return False

        # Check output similarity
        if self._outputs_too_similar():
            return False

        return True

    def _has_duplicate_steps(self) -> bool:
        """Check if recent steps are duplicates."""
        if len(self._step_hashes) < 2:
            return False

        # Check if last step matches any recent step
        last_hash = self._step_hashes[-1]
        duplicates = sum(1 for h in list(self._step_hashes)[:-1] if h == last_hash)

        # More than 1 duplicate in window indicates stuck
        return duplicates >= 2

    def _has_cyclic_pattern(self) -> bool:
        """Check for cyclic patterns in actions."""
        if len(self._action_sequence) < 4:
            return False

        actions = list(self._action_sequence)

        # Check for simple cycles (A-B-A-B pattern)
        for cycle_length in range(2, len(actions) // 2 + 1):
            if self._is_repeating_cycle(actions, cycle_length):
                return True

        return False

    def _is_repeating_cycle(
        self,
        sequence: list[str],
        cycle_length: int,
    ) -> bool:
        """Check if sequence contains a repeating cycle."""
        if len(sequence) < cycle_length * 2:
            return False

        recent = sequence[-cycle_length * 2 :]
        first_half = recent[:cycle_length]
        second_half = recent[cycle_length:]

        return first_half == second_half

    def _outputs_too_similar(self) -> bool:
        """Check if recent outputs are too similar."""
        if len(self._recent_outputs) < 2:
            return False

        outputs = list(self._recent_outputs)

        # Simple similarity: check if outputs are nearly identical
        # (Could be enhanced with cosine similarity of embeddings)
        unique_outputs = set(outputs)
        similarity = 1 - (len(unique_outputs) / len(outputs))

        return similarity >= self.similarity_threshold

    def reset(self) -> None:
        """Reset progress tracking."""
        self._step_hashes.clear()
        self._recent_outputs.clear()
        self._action_sequence.clear()
