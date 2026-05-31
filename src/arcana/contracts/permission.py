"""Permission policy contracts for tool execution boundaries."""

from __future__ import annotations

from enum import Enum
from fnmatch import fnmatchcase

from pydantic import BaseModel, Field

from arcana.contracts.tool import SideEffect, ToolSpec


class PermissionAction(str, Enum):
    """Runtime action selected by a permission policy."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionScope(str, Enum):
    """Scope at which a permission rule applies."""

    CALL = "call"
    SESSION = "session"
    PROJECT = "project"
    GLOBAL = "global"


class PermissionRequest(BaseModel):
    """Normalized tool permission evaluation input."""

    tool_name: str
    capabilities: list[str] = Field(default_factory=list)
    origin: str = "local"
    server_name: str | None = None
    side_effect: SideEffect = SideEffect.READ

    @classmethod
    def from_tool_spec(cls, spec: ToolSpec) -> PermissionRequest:
        """Build a permission request from a stable ToolSpec.

        Provenance is read from the spec's own ``provenance`` -- the single
        source of truth. A spec without provenance is treated as origin
        ``"local"``. Provenance is supplied by stamping the spec at its
        construction site (e.g. ``mcp_tool_to_arcana_spec``); there is
        deliberately no out-of-band override, so a partial or contradictory
        ``origin`` / ``server_name`` pair cannot be expressed at this security
        boundary.
        """

        prov = spec.provenance
        return cls(
            tool_name=spec.name,
            capabilities=list(spec.capabilities),
            origin=prov.origin if prov else "local",
            server_name=prov.server_name if prov else None,
            side_effect=spec.side_effect,
        )


class PermissionMatch(BaseModel):
    """Rule matcher. Empty fields are wildcards for that dimension."""

    tool_names: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    origins: list[str] = Field(default_factory=list)
    server_names: list[str] = Field(default_factory=list)
    side_effects: list[SideEffect] = Field(default_factory=list)

    def matches(self, request: PermissionRequest) -> bool:
        """Return true when all populated match dimensions accept the request."""

        return (
            self._matches_any(self.tool_names, request.tool_name)
            and self._matches_any(self.origins, request.origin)
            and self._matches_optional_any(self.server_names, request.server_name)
            and self._matches_side_effect(request.side_effect)
            and self._matches_capabilities(request.capabilities)
        )

    @staticmethod
    def _matches_any(patterns: list[str], value: str) -> bool:
        if not patterns:
            return True
        return any(fnmatchcase(value, pattern) for pattern in patterns)

    @staticmethod
    def _matches_optional_any(patterns: list[str], value: str | None) -> bool:
        if not patterns:
            return True
        if value is None:
            return False
        return any(fnmatchcase(value, pattern) for pattern in patterns)

    def _matches_side_effect(self, side_effect: SideEffect) -> bool:
        if not self.side_effects:
            return True
        return side_effect in self.side_effects

    def _matches_capabilities(self, capabilities: list[str]) -> bool:
        if not self.capabilities:
            return True
        return any(
            fnmatchcase(capability, pattern)
            for capability in capabilities
            for pattern in self.capabilities
        )


class PermissionRule(BaseModel):
    """Ordered permission rule."""

    action: PermissionAction
    reason: str
    match: PermissionMatch = Field(default_factory=PermissionMatch)
    scope: PermissionScope = PermissionScope.CALL
    id: str | None = None


class PermissionDecision(BaseModel):
    """Structured permission policy result."""

    action: PermissionAction
    allowed: bool
    reason: str
    scope: PermissionScope
    matched_rule_id: str | None = None

    @property
    def requires_approval(self) -> bool:
        """Whether the caller must ask before executing."""

        return self.action == PermissionAction.ASK


class PermissionPolicy(BaseModel):
    """Ordered allow/deny/ask policy for tool permission checks."""

    rules: list[PermissionRule] = Field(default_factory=list)
    default_action: PermissionAction = PermissionAction.ALLOW
    default_reason: str = "No permission rule matched."
    default_scope: PermissionScope = PermissionScope.CALL

    def evaluate(self, request: PermissionRequest) -> PermissionDecision:
        """Evaluate a tool permission request using first-match-wins semantics."""

        for rule in self.rules:
            if rule.match.matches(request):
                return PermissionDecision(
                    action=rule.action,
                    allowed=rule.action == PermissionAction.ALLOW,
                    reason=rule.reason,
                    scope=rule.scope,
                    matched_rule_id=rule.id,
                )

        return PermissionDecision(
            action=self.default_action,
            allowed=self.default_action == PermissionAction.ALLOW,
            reason=self.default_reason,
            scope=self.default_scope,
            matched_rule_id=None,
        )
