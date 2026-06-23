"""Prompt-based tool fallback is a real capability, not theater (finding F2).

A provider without native tool-calling asks the model to emit the call as JSON
in the response text. These tests prove that JSON is parsed back into
``LLMResponse.tool_calls`` (so the runtime executes the tool), across the
non-streaming path, the streaming path, and a full degraded agent loop.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import arcana
from arcana.contracts.llm import LLMRequest, Message, MessageRole, ModelConfig
from arcana.gateway.providers.openai_compatible import (
    OpenAICompatibleProvider,
    ProviderProfile,
    parse_text_tool_calls,
)
from arcana.runtime.stream_accumulator import StreamAccumulator

# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------

FENCE = "```"


def _block(inner: str) -> str:
    return f"{FENCE}json\n{inner}\n{FENCE}"


class TestParseTextToolCalls:
    def test_happy_path_fenced(self):
        calls = parse_text_tool_calls(
            "I'll look that up.\n"
            + _block('{"tool_call": {"name": "search", "arguments": {"q": "weather"}}}')
        )
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments == '{"q": "weather"}'
        assert calls[0].id.startswith("call_")

    def test_bare_unfenced_json(self):
        calls = parse_text_tool_calls(
            'Sure: {"tool_call": {"name": "now", "arguments": {"tz": "UTC"}}}'
        )
        assert [c.name for c in calls] == ["now"]
        assert calls[0].arguments == '{"tz": "UTC"}'

    def test_multiple_blocks_preserve_order(self):
        calls = parse_text_tool_calls(
            _block('{"tool_call": {"name": "a", "arguments": {"x": 1}}}')
            + "\nthen\n"
            + _block('{"tool_call": {"name": "b", "arguments": {"y": 2}}}')
        )
        assert [c.name for c in calls] == ["a", "b"]

    def test_nested_braces_and_braces_in_strings(self):
        calls = parse_text_tool_calls(
            _block(
                '{"tool_call": {"name": "echo", "arguments": '
                '{"text": "use {curly} and } x {", "deep": {"a": {"b": 1}}}}}'
            )
        )
        assert len(calls) == 1
        import json

        args = json.loads(calls[0].arguments)
        assert args["text"] == "use {curly} and } x {"
        assert args["deep"] == {"a": {"b": 1}}

    def test_double_encoded_arguments(self):
        calls = parse_text_tool_calls(
            '{"tool_call": {"name": "q", "arguments": "{\\"k\\": \\"v\\"}"}}'
        )
        assert calls[0].arguments == '{"k": "v"}'

    def test_malformed_json_is_skipped(self):
        assert parse_text_tool_calls(
            _block('{"tool_call": {"name": "x", "arguments": {"q":')
        ) == []

    def test_non_tool_call_json_rejected(self):
        assert parse_text_tool_calls(_block('{"result": {"name": "x"}}')) == []

    def test_missing_name_rejected(self):
        assert parse_text_tool_calls(_block('{"tool_call": {"arguments": {}}}')) == []

    def test_plain_answer_yields_nothing(self):
        assert parse_text_tool_calls("The capital of France is Paris.") == []

    def test_empty_and_none(self):
        assert parse_text_tool_calls(None) == []
        assert parse_text_tool_calls("") == []

    def test_deeply_nested_arguments_do_not_raise(self):
        # ~1500 levels -> json.loads RecursionError must be swallowed.
        depth = 1500
        deep = (
            '{"tool_call": {"name": "x", "arguments": '
            + '{"a":' * depth + "1" + "}" * depth + "}}"
        )
        assert parse_text_tool_calls(deep) == []

    def test_many_unbalanced_starts_are_safe(self):
        # Many never-closing `{"tool_call"` starts must stay linear (the
        # cursor skip), and yield nothing since none balance.
        assert parse_text_tool_calls('{"tool_call"' * 5000) == []

    def test_whitespace_only_name_rejected(self):
        assert parse_text_tool_calls(
            '{"tool_call": {"name": "   ", "arguments": {}}}'
        ) == []

    def test_surrounding_whitespace_in_name_stripped(self):
        calls = parse_text_tool_calls(
            '{"tool_call": {"name": "  search  ", "arguments": {}}}'
        )
        assert [c.name for c in calls] == ["search"]

    def test_nested_tool_call_in_arguments_not_double_counted(self):
        # A `tool_call`-looking object inside arguments is part of the outer
        # call, not a second call.
        calls = parse_text_tool_calls(
            '{"tool_call": {"name": "wrap", "arguments": '
            '{"inner": {"tool_call": {"name": "x", "arguments": {}}}}}}'
        )
        assert [c.name for c in calls] == ["wrap"]

    def test_arguments_as_array_coerced_to_empty(self):
        calls = parse_text_tool_calls(
            '{"tool_call": {"name": "f", "arguments": [1, 2, 3]}}'
        )
        assert calls[0].arguments == "{}"


# ---------------------------------------------------------------------------
# Provider end-to-end (non-streaming + streaming)
# ---------------------------------------------------------------------------


def _completion(content, *, finish_reason="stop", model="test-model", pt=10, ct=5):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
        prompt_tokens_details=None,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _degraded_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider_name="test",
        api_key="sk-test",
        base_url="http://localhost",
        profile=ProviderProfile(tool_calls=False),
    )


_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}


class TestProviderFallbackEndToEnd:
    @pytest.mark.asyncio
    async def test_generate_recovers_tool_call_from_text(self):
        provider = _degraded_provider()
        provider.client.chat.completions.create = AsyncMock(
            return_value=_completion(
                _block('{"tool_call": {"name": "search", "arguments": {"q": "x"}}}')
            )
        )
        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="search x")],
            tools=[_TOOL],
        )
        resp = await provider.generate(req, ModelConfig(provider="test", model_id="m"))

        assert resp.tool_calls is not None
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == '{"q": "x"}'
        # finish_reason promoted so assessment treats it as "wants to act".
        assert resp.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_generate_plain_answer_has_no_tool_calls(self):
        provider = _degraded_provider()
        provider.client.chat.completions.create = AsyncMock(
            return_value=_completion("Paris is the capital.")
        )
        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="capital?")],
            tools=[_TOOL],
        )
        resp = await provider.generate(req, ModelConfig(provider="test", model_id="m"))
        assert not resp.tool_calls
        assert resp.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_parity_recovers_tool_call(self):
        """The streaming path also surfaces text-fallback tool calls."""
        provider = _degraded_provider()
        provider.client.chat.completions.create = AsyncMock(
            return_value=_completion(
                _block('{"tool_call": {"name": "search", "arguments": {"q": "y"}}}')
            )
        )
        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="search y")],
            tools=[_TOOL],
        )
        acc = StreamAccumulator()
        async for chunk in provider.stream(req, ModelConfig(provider="test", model_id="m")):
            acc.feed(chunk)
        resp = acc.to_response()
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == '{"q": "y"}'


# ---------------------------------------------------------------------------
# Full agent loop with a degraded provider — the tool actually executes
# ---------------------------------------------------------------------------


class TestReactiveDegrade:
    """A provider that starts with native tools but the API rejects them
    (400) auto-degrades and recovers the tool call via the text fallback."""

    @pytest.mark.asyncio
    async def test_400_degrades_and_recovers_tool_call(self):
        import httpx
        from openai import BadRequestError

        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_key="sk-test",
            base_url="http://localhost",
            profile=ProviderProfile(tool_calls=True),  # starts native-capable
        )
        bad_request = BadRequestError(
            "tools are not supported by this model",
            response=httpx.Response(
                400, request=httpx.Request("POST", "http://localhost/v1")
            ),
            body=None,
        )
        provider.client.chat.completions.create = AsyncMock(
            side_effect=[
                bad_request,
                _completion(
                    _block('{"tool_call": {"name": "search", "arguments": {"q": "z"}}}')
                ),
            ]
        )
        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="search z")],
            tools=[_TOOL],
        )
        resp = await provider.generate(req, ModelConfig(provider="test", model_id="m"))

        # Capability degraded (fail-once) and the retry recovered the call.
        assert provider.profile.tool_calls is False
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == '{"q": "z"}'
        assert provider.client.chat.completions.create.await_count == 2


class TestDegradedAgentExecutesTool:
    @pytest.mark.asyncio
    async def test_tool_runs_end_to_end(self):
        # str-typed arg: this module uses `from __future__ import annotations`,
        # so @arcana.tool sees the annotation as the string "str" and schemas
        # it as "string" — matching string args keeps the test about the
        # fallback, not about signature-schema typing.
        ran = {"value": None}

        @arcana.tool(side_effect="read")
        async def record(value: str) -> str:
            ran["value"] = value
            return f"recorded {value}"

        rt = arcana.Runtime(
            providers={"ollama": ""},
            tools=[record],
            config=arcana.RuntimeConfig(default_provider="ollama"),
        )
        provider = rt._gateway.get("ollama")
        # Force the prompt-based fallback path.
        provider.profile.tool_calls = False
        provider.client = MagicMock()
        provider.client.chat.completions.create = AsyncMock(
            side_effect=[
                _completion(
                    _block('{"tool_call": {"name": "record", "arguments": {"value": "hi"}}}')
                ),
                _completion("Done — recorded hi."),
            ]
        )

        async with rt.chat() as c:
            resp = await c.send("record hi")

        # The degraded provider's text tool call was parsed AND executed with
        # the parsed arguments — proof it is a real capability, not theater.
        assert ran["value"] == "hi"
        assert "recorded" in resp.content.lower() or "hi" in resp.content
