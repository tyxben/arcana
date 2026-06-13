"""Guardrail contracts for explicit runtime boundaries."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from arcana.contracts.tool import SideEffect, ToolSpec
from arcana.utils.hashing import canonical_hash


class GuardrailAction(str, Enum):
    """Action selected by a guardrail boundary check."""

    ALLOW = "allow"
    WARN = "warn"
    REDACT = "redact"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


class ToolGuardrailRequest(BaseModel):
    """Normalized input for a tool-call guardrail."""

    boundary: str = "tool_call"
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_digest: str
    side_effect: SideEffect
    capabilities: list[str] = Field(default_factory=list)
    origin: str = "local"
    server_name: str | None = None

    @classmethod
    def from_tool_call(
        cls,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        spec: ToolSpec,
    ) -> ToolGuardrailRequest:
        """Build a guardrail request from a tool call and its resolved spec."""

        provenance = spec.provenance
        return cls(
            tool_name=tool_name,
            arguments=dict(arguments),
            arguments_digest=canonical_hash(arguments),
            side_effect=spec.side_effect,
            capabilities=list(spec.capabilities),
            origin=provenance.origin if provenance is not None else "local",
            server_name=provenance.server_name if provenance is not None else None,
        )


class GuardrailDecision(BaseModel):
    """Structured guardrail result.

    The decision is boundary-shaped: it can block, warn, redact arguments, or
    require approval. It does not choose a next tool, replace the user's goal,
    or encode hidden workflow state.
    """

    guardrail_name: str
    action: GuardrailAction
    reason: str
    redacted_arguments: dict[str, Any] | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def blocks_execution(self) -> bool:
        """Whether this decision prevents immediate execution."""

        return self.action == GuardrailAction.BLOCK

    @property
    def requires_approval(self) -> bool:
        """Whether this decision requests explicit approval before execution."""

        return self.action == GuardrailAction.REQUIRE_APPROVAL
