"""Write governance for the memory system."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arcana.contracts.memory import MemoryConfig, MemoryWriteRequest, MemoryWriteResult

if TYPE_CHECKING:
    from arcana.contracts.memory import MemoryEntry, RevocationRequest

logger = logging.getLogger(__name__)


class WritePolicy:
    """
    Governs memory writes via confidence thresholds and source tracking.

    Decides whether a write is allowed, warns on borderline confidence,
    and validates revocation requests.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()

    def evaluate(self, request: MemoryWriteRequest) -> MemoryWriteResult:
        """Evaluate whether a memory write should be allowed."""
        # Reject below minimum confidence
        if request.confidence < self.config.min_write_confidence:
            return MemoryWriteResult(
                success=False,
                rejected_reason=(
                    f"Confidence {request.confidence:.2f} below threshold "
                    f"{self.config.min_write_confidence:.2f}"
                ),
                confidence_below_threshold=True,
            )

        # Warn between min and warn thresholds
        if request.confidence < self.config.warn_confidence_threshold:
            logger.warning(
                "Low confidence memory write: %.2f (key=%s, source=%s)",
                request.confidence,
                request.key,
                request.source,
            )

        return MemoryWriteResult(success=True)

    def validate_revocation(
        self, entry: MemoryEntry, request: RevocationRequest
    ) -> bool:
        """Validate that a revocation is valid. Cannot revoke twice."""
        return not entry.revoked
