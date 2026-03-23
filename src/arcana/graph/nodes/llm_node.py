"""LLMNode - wraps gateway.generate() for use in graph workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcana.contracts.llm import LLMRequest, Message, MessageRole, ModelConfig

if TYPE_CHECKING:
    from arcana.gateway.base import ModelGateway
    from arcana.gateway.registry import ModelGatewayRegistry


class LLMNode:
    """
    Graph node that calls an LLM via the gateway.

    Reads messages from state and returns the LLM response.
    """

    def __init__(
        self,
        gateway: ModelGateway | ModelGatewayRegistry,
        *,
        model_config: ModelConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._gateway = gateway
        self._model_config = model_config
        self._system_prompt = system_prompt

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute LLM call with messages from state."""
        messages = state.get("messages", [])

        # Build message list
        request_messages: list[Message] = []

        # Add system prompt if configured
        if self._system_prompt:
            request_messages.append(
                Message(role=MessageRole.SYSTEM, content=self._system_prompt)
            )

        # Convert state messages to Message objects
        for msg in messages:
            if isinstance(msg, Message):
                request_messages.append(msg)
            elif isinstance(msg, dict):
                request_messages.append(Message(**msg))

        # Build request
        config = self._model_config or ModelConfig(provider="default", model_id="default")
        request = LLMRequest(messages=request_messages)

        # Call gateway
        response = await self._gateway.generate(request, config)

        # Build response message
        response_msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.content or "",
        }

        # Include tool calls if present
        if response.tool_calls:
            response_msg["tool_calls"] = [
                tc.model_dump() for tc in response.tool_calls
            ]

        return {"messages": [response_msg]}
