"""Deterministic mock provider for CI-safe eval tests.

Returns canned responses based on message content patterns.
No API keys needed — same input always gives same output.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from arcana.contracts.llm import (
    LLMRequest,
    LLMResponse,
    ModelConfig,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)

if TYPE_CHECKING:
    from arcana.contracts.trace import TraceContext


class MockProvider:
    """Deterministic LLM provider for eval and testing.

    Implements BaseProvider protocol. Routes responses based on
    keyword matching against the last user message.

    Attributes:
        call_count: Number of generate() calls made.
        call_log: List of (messages, tools) tuples for each call.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.call_log: list[tuple[list[dict[str, str]], list[dict[str, object]] | None]] = []
        self._response_rules: list[tuple[re.Pattern[str], LLMResponse]] = []
        self._tool_call_rules: list[tuple[re.Pattern[str], list[ToolCallRequest]]] = []
        self._default_response = "I don't have a specific answer for that."

        # Install default rules
        self._install_defaults()

    # ------------------------------------------------------------------
    # BaseProvider protocol
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return "mock-v1"

    @property
    def supported_models(self) -> list[str]:
        return ["mock-v1"]

    async def generate(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> LLMResponse:
        self.call_count += 1

        # Extract last user message
        user_text = self._extract_user_text(request)
        tools = request.tools

        self.call_log.append((
            [{"role": m.role.value, "content": str(m.content)} for m in request.messages],
            tools,
        ))

        # Check if we have tool results in messages — if so, synthesize a final answer
        has_tool_results = any(m.role.value == "tool" for m in request.messages)
        if has_tool_results:
            return self._respond_to_tool_results(request)

        # Check tool call rules first (when tools are available)
        if tools:
            for pattern, tool_calls in self._tool_call_rules:
                if pattern.search(user_text):
                    return LLMResponse(
                        content=None,
                        tool_calls=tool_calls,
                        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
                        model="mock-v1",
                        finish_reason="tool_calls",
                    )

        # Check text response rules
        for pattern, response in self._response_rules:
            if pattern.search(user_text):
                return response

        # Default
        return LLMResponse(
            content=self._default_response,
            usage=TokenUsage(prompt_tokens=30, completion_tokens=20, total_tokens=50),
            model="mock-v1",
            finish_reason="stop",
        )

    async def stream(
        self,
        request: LLMRequest,
        config: ModelConfig,
        trace_ctx: TraceContext | None = None,
    ) -> AsyncIterator[StreamChunk]:
        response = await self.generate(request, config, trace_ctx)
        if response.content:
            yield StreamChunk(type="text_delta", text=response.content)
        if response.tool_calls:
            for tc in response.tool_calls:
                yield StreamChunk(
                    type="tool_call_delta",
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    arguments_delta=tc.arguments,
                )
        yield StreamChunk(type="done", usage=response.usage)

    async def health_check(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_response_rule(self, pattern: str, content: str, *, done: bool = True) -> None:
        """Add a text response rule. Prepended so custom rules override defaults.

        The ``done`` parameter is kept for backward compatibility but no longer
        appends a [DONE] marker. Completion is determined by the LLM's natural
        stop signal (finish_reason=stop), not by artificial markers.
        """
        text = content
        response = LLMResponse(
            content=text,
            usage=TokenUsage(prompt_tokens=30, completion_tokens=len(text.split()), total_tokens=30 + len(text.split())),
            model="mock-v1",
            finish_reason="stop",
        )
        # Prepend: custom rules take priority over defaults
        self._response_rules.insert(0, (re.compile(pattern, re.IGNORECASE), response))

    def add_tool_call_rule(self, pattern: str, tool_name: str, arguments: dict[str, object]) -> None:
        """Add a rule that triggers a tool call when pattern matches. Prepended so custom rules override defaults."""
        tool_calls = [
            ToolCallRequest(
                id=f"call_{tool_name}_001",
                name=tool_name,
                arguments=json.dumps(arguments),
            )
        ]
        # Prepend: custom rules take priority over defaults
        self._tool_call_rules.insert(0, (re.compile(pattern, re.IGNORECASE), tool_calls))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _install_defaults(self) -> None:
        """Install default response rules for common eval patterns."""
        # Math
        self.add_response_rule(
            r"what is 2\s*\+\s*2|2\s*\+\s*2",
            "The answer is 4.",
        )
        self.add_response_rule(
            r"what is 3\s*\*\s*7|3\s*\*\s*7|3\s*times\s*7",
            "The answer is 21.",
        )

        # Tool call triggers
        self.add_tool_call_rule(
            r"calculat|compute|what is \d+\s*[\+\-\*\/]\s*\d+.*tool|use.*tool.*\d+",
            "calculator",
            {"expression": "2 + 2"},
        )

        # Context retention: respond with remembered info
        self.add_response_rule(
            r"what.*name|who am i|remember.*name|recall.*name",
            "Your name is Alice, as you told me earlier.",
        )

        # Greetings
        self.add_response_rule(r"^(hi|hello|hey)\b", "Hello! How can I help you?")

    def _extract_user_text(self, request: LLMRequest) -> str:
        """Get the concatenated text of all user messages."""
        parts: list[str] = []
        for m in request.messages:
            if m.role.value == "user" and isinstance(m.content, str):
                parts.append(m.content)
        return " ".join(parts)

    def _respond_to_tool_results(self, request: LLMRequest) -> LLMResponse:
        """Generate a final answer after tool results are available."""
        # Find the last tool result
        tool_output = ""
        for m in reversed(request.messages):
            if m.role.value == "tool" and isinstance(m.content, str):
                tool_output = m.content
                break

        content = f"Based on the tool result: {tool_output}"
        return LLMResponse(
            content=content,
            usage=TokenUsage(prompt_tokens=60, completion_tokens=15, total_tokens=75),
            model="mock-v1",
            finish_reason="stop",
        )
