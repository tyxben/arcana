"""Evidence + budget chain on the live V2 path (findings F1 + F3).

The V2 ConversationAgent must thread a TraceContext into its LLM and tool
calls so the provider's LLM_CALL and the gateway's TOOL_CALL audit events
(and the permission/guardrail decision metadata) are actually written -- not
only in unit tests that drive the gateway directly. LLM compression must also
count against budget + reported cost and be traced.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import arcana
from arcana.context.builder import WorkingSetBuilder
from arcana.contracts.context import TokenBudget
from arcana.contracts.llm import LLMResponse, Message, MessageRole, TokenUsage, ToolCallRequest
from arcana.contracts.permission import (
    PermissionAction,
    PermissionMatch,
    PermissionPolicy,
    PermissionRule,
)
from arcana.contracts.tool import SideEffect
from arcana.runtime_core import Runtime, RuntimeConfig
from arcana.trace.reader import TraceReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(t: str) -> LLMResponse:
    return LLMResponse(
        content=t, tool_calls=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        model="m", finish_reason="stop",
    )


def _call(name: str, args: str, cid: str = "tc1") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=cid, name=name, arguments=args)],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="m", finish_reason="tool_calls",
    )


def _event_types(trace_dir, run_id) -> Counter:
    reader = TraceReader(trace_dir=trace_dir)
    return Counter(e.event_type.value for e in reader.read_events(run_id))


# ---------------------------------------------------------------------------
# F1 — tool calls are audited on the live agent path
# ---------------------------------------------------------------------------


class TestToolCallAuditOnLivePath:
    @pytest.mark.asyncio
    async def test_tool_call_event_written(self, tmp_path):
        @arcana.tool(side_effect="read")
        async def look(x: str) -> str:
            return "data"

        rt = Runtime(
            providers={"ollama": ""},
            tools=[look],
            trace=True,
            config=RuntimeConfig(default_provider="ollama", trace_dir=str(tmp_path)),
        )
        rt._gateway.generate = AsyncMock(
            side_effect=[_call("look", '{"x": "a"}'), _text("done")]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        async with rt.chat() as c:
            await c.send("use the tool")

        files = list(Path(tmp_path).glob("*.jsonl"))
        assert files
        all_types: Counter = Counter()
        for f in files:
            all_types += _event_types(tmp_path, f.stem)
        # The TOOL_CALL audit event now fires on the real V2 path.
        assert all_types["tool_call"] >= 1

    @pytest.mark.asyncio
    async def test_permission_decision_metadata_reaches_trace(self, tmp_path):
        """A permission policy's decision is recorded on the live tool call."""
        @arcana.tool(side_effect="read")
        async def look(x: str) -> str:
            return "data"

        rt = Runtime(
            providers={"ollama": ""},
            tools=[look],
            trace=True,
            config=RuntimeConfig(default_provider="ollama", trace_dir=str(tmp_path)),
        )
        # Allow-all policy so the call proceeds but a decision is recorded.
        rt._tool_gateway.permission_policy = PermissionPolicy(
            rules=[
                PermissionRule(
                    action=PermissionAction.ALLOW,
                    reason="reads ok",
                    match=PermissionMatch(side_effects=[SideEffect.READ]),
                )
            ]
        )
        rt._gateway.generate = AsyncMock(
            side_effect=[_call("look", '{"x": "a"}'), _text("done")]
        )
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        async with rt.chat() as c:
            await c.send("use the tool")

        reader = TraceReader(trace_dir=tmp_path)
        tool_events = [
            e
            for f in Path(tmp_path).glob("*.jsonl")
            for e in reader.read_events(f.stem)
            if e.event_type.value == "tool_call"
        ]
        assert tool_events
        assert any(
            "permission_decision" in (e.metadata or {}) for e in tool_events
        )

    @pytest.mark.asyncio
    async def test_agent_passes_trace_ctx_to_generate(self, tmp_path):
        """The agent threads a non-None trace_ctx into the LLM call (so the
        provider's LLM_CALL event can fire)."""
        rt = Runtime(
            providers={"ollama": ""},
            trace=True,
            config=RuntimeConfig(default_provider="ollama", trace_dir=str(tmp_path)),
        )
        seen = {}

        async def spy(request=None, config=None, trace_ctx=None, **kw):
            seen["trace_ctx"] = trace_ctx
            return _text("hi")

        rt._gateway.generate = spy
        rt._gateway.stream = MagicMock(side_effect=NotImplementedError)

        async with rt.chat() as c:
            await c.send("hello there please answer in full")

        assert seen.get("trace_ctx") is not None
        assert seen["trace_ctx"].run_id


# ---------------------------------------------------------------------------
# F3 — LLM compression is budgeted + traced
# ---------------------------------------------------------------------------


class TestCompressionAccounting:
    @pytest.mark.asyncio
    async def test_compression_usage_recorded_and_consumed(self):
        """A builder LLM summarization records usage; consume returns it once."""
        gateway = MagicMock()
        summary_usage = TokenUsage(
            prompt_tokens=200, completion_tokens=50, total_tokens=250
        )
        gateway.generate = AsyncMock(
            return_value=LLMResponse(
                content="short summary",
                tool_calls=None,
                usage=summary_usage,
                model="m",
                finish_reason="stop",
            )
        )
        gateway.default_provider = "ollama"
        gateway.get = MagicMock(return_value=None)

        builder = WorkingSetBuilder(
            identity="You are helpful.",
            token_budget=TokenBudget(total_window=2000, response_reserve=200),
            goal="do the thing",
            gateway=gateway,
        )

        # A long history that forces LLM summarization of the middle section.
        messages = [Message(role=MessageRole.SYSTEM, content="sys")]
        for i in range(40):
            messages.append(
                Message(
                    role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                    content=f"message {i} " + ("filler " * 60),
                )
            )

        await builder.abuild_conversation_context(messages, turn=1)

        # The compression call happened and its usage is available exactly once.
        consumed = builder.consume_compression_usage()
        assert consumed is not None
        assert consumed.total_tokens == 250
        # Second consume returns None (cleared).
        assert builder.consume_compression_usage() is None
        # And the gateway was actually invoked for summarization.
        assert gateway.generate.await_count >= 1

    @pytest.mark.asyncio
    async def test_no_compression_means_no_usage(self):
        gateway = MagicMock()
        gateway.generate = AsyncMock()
        builder = WorkingSetBuilder(
            identity="id",
            token_budget=TokenBudget(total_window=128_000),
            goal="g",
            gateway=gateway,
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="short"),
        ]
        await builder.abuild_conversation_context(messages, turn=0)
        # Nothing to compress -> no summarization call, no usage.
        assert builder.consume_compression_usage() is None
        gateway.generate.assert_not_awaited()
