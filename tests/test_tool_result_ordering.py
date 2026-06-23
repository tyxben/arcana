"""Tool results are returned in input order (finding F4).

_execute_tools buckets calls into ask_user -> cognitive -> gateway and runs
each bucket, but must re-assemble the returned results (and therefore the
TOOL_END stream events) in the order the LLM requested them. The LLM
conversation is already tool_call_id-matched; this pins the returned-order
contract that mixed built-in + gateway turns used to violate.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import arcana
from arcana.contracts.llm import ToolCallRequest
from arcana.runtime.conversation import ConversationAgent
from arcana.sdk import _FunctionToolProvider
from arcana.tool_gateway.gateway import ToolGateway
from arcana.tool_gateway.registry import ToolRegistry


def _gateway_with_echo() -> ToolGateway:
    @arcana.tool(side_effect="read")
    async def echo(value: str) -> str:
        return f"echo:{value}"

    registry = ToolRegistry()
    registry.register(
        _FunctionToolProvider(spec=echo._arcana_tool_spec, func=echo)
    )
    return ToolGateway(registry=registry)


def _agent() -> ConversationAgent:
    return ConversationAgent(
        gateway=MagicMock(),
        tool_gateway=_gateway_with_echo(),
        input_handler=lambda q: "answer",
    )


def _gw_call(cid: str, value: str) -> ToolCallRequest:
    return ToolCallRequest(id=cid, name="echo", arguments=f'{{"value": "{value}"}}')


def _ask_call(cid: str) -> ToolCallRequest:
    return ToolCallRequest(id=cid, name="ask_user", arguments='{"question": "q?"}')


class TestToolResultOrdering:
    @pytest.mark.asyncio
    async def test_builtin_after_gateway_preserves_input_order(self):
        """Input [gateway, ask_user] used to return [ask_user, gateway]."""
        agent = _agent()
        calls = [_gw_call("g1", "a"), _ask_call("u1")]

        results, _ = await agent._execute_tools(calls)

        assert [r.tool_call_id for r in results] == ["g1", "u1"]

    @pytest.mark.asyncio
    async def test_interleaved_order_preserved(self):
        """A built-in sandwiched between gateway calls stays in place."""
        agent = _agent()
        calls = [_gw_call("g1", "a"), _ask_call("u1"), _gw_call("g2", "b")]

        results, _ = await agent._execute_tools(calls)

        assert [r.tool_call_id for r in results] == ["g1", "u1", "g2"]
        # And the payloads line up with their calls (id-matched, not shuffled).
        by_id = {r.tool_call_id: r for r in results}
        assert "echo:a" in by_id["g1"].output_str
        assert "echo:b" in by_id["g2"].output_str

    @pytest.mark.asyncio
    async def test_pure_gateway_order_preserved(self):
        agent = _agent()
        calls = [_gw_call("g1", "a"), _gw_call("g2", "b"), _gw_call("g3", "c")]

        results, _ = await agent._execute_tools(calls)

        assert [r.tool_call_id for r in results] == ["g1", "g2", "g3"]

    @pytest.mark.asyncio
    async def test_builtin_first_unchanged(self):
        """Input order already built-in-first is returned unchanged."""
        agent = _agent()
        calls = [_ask_call("u1"), _gw_call("g1", "a")]

        results, _ = await agent._execute_tools(calls)

        assert [r.tool_call_id for r in results] == ["u1", "g1"]

    @pytest.mark.asyncio
    async def test_duplicate_ids_not_merged_or_dropped(self):
        """Colliding tool_call_ids must not last-wins-collapse to one result."""
        agent = _agent()
        calls = [_gw_call("dup", "a"), _gw_call("dup", "b")]

        results, _ = await agent._execute_tools(calls)

        # Both executed results survive (a by-id dict would keep only one).
        assert len(results) == 2
        outputs = sorted(r.output_str for r in results)
        assert any("echo:a" in o for o in outputs)
        assert any("echo:b" in o for o in outputs)

    @pytest.mark.asyncio
    async def test_empty_ids_preserved(self):
        """Empty ids are ambiguous to pair, but no result may be dropped."""
        agent = _agent()
        calls = [_gw_call("", "a"), _ask_call("")]

        results, _ = await agent._execute_tools(calls)

        assert len(results) == 2
