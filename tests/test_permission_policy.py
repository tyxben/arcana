"""Tests for permission policy contracts."""

from arcana.contracts.permission import (
    PermissionAction,
    PermissionMatch,
    PermissionPolicy,
    PermissionRequest,
    PermissionRule,
    PermissionScope,
)
from arcana.contracts.tool import SideEffect, ToolProvenance, ToolSpec


def test_default_policy_allows_unmatched_call() -> None:
    policy = PermissionPolicy()

    decision = policy.evaluate(
        PermissionRequest(tool_name="search_docs", side_effect=SideEffect.READ)
    )

    assert decision.action == PermissionAction.ALLOW
    assert decision.allowed is True
    assert decision.requires_approval is False
    assert decision.reason == "No permission rule matched."
    assert decision.matched_rule_id is None


def test_deny_rule_matches_tool_capability_origin_server_and_side_effect() -> None:
    policy = PermissionPolicy(
        rules=[
            PermissionRule(
                id="deny-remote-shell-write",
                action=PermissionAction.DENY,
                reason="Remote shell writes are not trusted.",
                scope=PermissionScope.PROJECT,
                match=PermissionMatch(
                    tool_names=["shell_*"],
                    capabilities=["shell:*"],
                    origins=["mcp"],
                    server_names=["untrusted-*"],
                    side_effects=[SideEffect.WRITE],
                ),
            )
        ]
    )

    decision = policy.evaluate(
        PermissionRequest(
            tool_name="shell_exec",
            capabilities=["shell:exec"],
            origin="mcp",
            server_name="untrusted-local",
            side_effect=SideEffect.WRITE,
        )
    )

    assert decision.action == PermissionAction.DENY
    assert decision.allowed is False
    assert decision.reason == "Remote shell writes are not trusted."
    assert decision.scope == PermissionScope.PROJECT
    assert decision.matched_rule_id == "deny-remote-shell-write"


def test_ask_rule_surfaces_confirmation_required_decision() -> None:
    policy = PermissionPolicy(
        rules=[
            PermissionRule(
                id="ask-file-write",
                action=PermissionAction.ASK,
                reason="File writes require approval.",
                match=PermissionMatch(
                    capabilities=["fs:write"],
                    side_effects=[SideEffect.WRITE],
                ),
            )
        ]
    )

    decision = policy.evaluate(
        PermissionRequest(
            tool_name="file_write",
            capabilities=["fs:read", "fs:write"],
            side_effect=SideEffect.WRITE,
        )
    )

    assert decision.action == PermissionAction.ASK
    assert decision.allowed is False
    assert decision.requires_approval is True
    assert decision.reason == "File writes require approval."


def test_first_matching_rule_wins() -> None:
    policy = PermissionPolicy(
        rules=[
            PermissionRule(
                id="deny-write",
                action=PermissionAction.DENY,
                reason="Writes denied.",
                match=PermissionMatch(side_effects=[SideEffect.WRITE]),
            ),
            PermissionRule(
                id="allow-file-write",
                action=PermissionAction.ALLOW,
                reason="Specific file write allowed.",
                match=PermissionMatch(tool_names=["file_write"]),
            ),
        ]
    )

    decision = policy.evaluate(
        PermissionRequest(tool_name="file_write", side_effect=SideEffect.WRITE)
    )

    assert decision.action == PermissionAction.DENY
    assert decision.reason == "Writes denied."
    assert decision.matched_rule_id == "deny-write"


def test_nonmatching_populated_dimension_falls_through_to_default() -> None:
    policy = PermissionPolicy(
        default_action=PermissionAction.DENY,
        default_reason="No explicit grant.",
        rules=[
            PermissionRule(
                action=PermissionAction.ALLOW,
                reason="Trusted docs read.",
                match=PermissionMatch(
                    tool_names=["docs_*"],
                    origins=["local"],
                    side_effects=[SideEffect.READ],
                ),
            )
        ],
    )

    decision = policy.evaluate(
        PermissionRequest(
            tool_name="docs_search",
            origin="mcp",
            side_effect=SideEffect.READ,
        )
    )

    assert decision.action == PermissionAction.DENY
    assert decision.allowed is False
    assert decision.reason == "No explicit grant."
    assert decision.matched_rule_id is None


def test_permission_request_can_be_built_from_tool_spec() -> None:
    spec = ToolSpec(
        name="file_read",
        description="Read a file",
        input_schema={"type": "object", "properties": {}},
        capabilities=["fs:read"],
        side_effect=SideEffect.READ,
        provenance=ToolProvenance(origin="mcp", server_name="filesystem"),
    )

    # Provenance is read from the spec itself -- the single source of truth.
    request = PermissionRequest.from_tool_spec(spec)

    assert request.tool_name == "file_read"
    assert request.capabilities == ["fs:read"]
    assert request.origin == "mcp"
    assert request.server_name == "filesystem"
    assert request.side_effect == SideEffect.READ


def test_from_tool_spec_without_provenance_is_local() -> None:
    spec = ToolSpec(
        name="local_tool",
        description="A local tool",
        input_schema={"type": "object", "properties": {}},
    )

    request = PermissionRequest.from_tool_spec(spec)

    assert request.origin == "local"
    assert request.server_name is None
