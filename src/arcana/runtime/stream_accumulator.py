"""Stream accumulator for assembling LLM streaming chunks into a complete response."""

from __future__ import annotations

from dataclasses import dataclass, field

from arcana.contracts.llm import LLMResponse, StreamChunk, TokenUsage, ToolCallRequest


@dataclass
class StreamAccumulator:
    """Accumulates streaming chunks into a complete LLMResponse.

    Single state-management point for all streaming chunk types.
    """

    text_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    _tool_names: dict[str, str] = field(default_factory=dict)
    _tool_args: dict[str, list[str]] = field(default_factory=dict)
    usage: TokenUsage | None = None
    finish_reason: str = "stop"
    model: str = ""

    def feed(self, chunk: StreamChunk) -> None:
        """Process a single streaming chunk."""
        if chunk.type == "text_delta" and chunk.text:
            self.text_parts.append(chunk.text)
        elif chunk.type == "tool_call_delta" and chunk.tool_call_id:
            tc_id = chunk.tool_call_id
            if tc_id not in self._tool_names:
                self._tool_names[tc_id] = chunk.tool_name or ""
            if chunk.tool_name:
                self._tool_names[tc_id] = chunk.tool_name
            if chunk.arguments_delta:
                self._tool_args.setdefault(tc_id, []).append(chunk.arguments_delta)
        elif chunk.type == "thinking_delta" and chunk.thinking:
            self.thinking_parts.append(chunk.thinking)
        elif chunk.type == "usage" and chunk.usage:
            self.usage = chunk.usage
        elif chunk.type == "done":
            if chunk.metadata:
                self.finish_reason = chunk.metadata.get("finish_reason", self.finish_reason)
                self.model = chunk.metadata.get("model", self.model)
            if chunk.usage:
                self.usage = chunk.usage

    @property
    def text(self) -> str | None:
        """Assembled full text, or None if no text chunks received."""
        return "".join(self.text_parts) if self.text_parts else None

    @property
    def thinking(self) -> str | None:
        """Assembled full thinking text, or None if no thinking chunks received."""
        return "".join(self.thinking_parts) if self.thinking_parts else None

    @property
    def tool_calls(self) -> list[ToolCallRequest] | None:
        """Assembled tool calls, or None if no tool call chunks received."""
        if not self._tool_names:
            return None
        return [
            ToolCallRequest(
                id=tc_id,
                name=self._tool_names.get(tc_id, ""),
                arguments="".join(self._tool_args.get(tc_id, [])),
            )
            for tc_id in self._tool_names
        ]

    def to_response(self) -> LLMResponse:
        """Assemble accumulated chunks into a complete LLMResponse."""
        # Build anthropic extension if thinking was captured
        anthropic_ext = None
        if self.thinking_parts:
            from arcana.contracts.llm import AnthropicResponseExt, ThinkingBlock

            anthropic_ext = AnthropicResponseExt(
                thinking_blocks=[ThinkingBlock(thinking=self.thinking)]  # type: ignore[arg-type]
            )

        return LLMResponse(
            content=self.text,
            tool_calls=self.tool_calls,
            usage=self.usage or TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model=self.model,
            finish_reason=self.finish_reason,
            anthropic=anthropic_ext,
        )
