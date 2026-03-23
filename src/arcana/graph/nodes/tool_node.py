"""ToolNode - wraps tool_gateway for use in graph workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcana.contracts.tool import ToolCall

if TYPE_CHECKING:
    from arcana.tool_gateway.gateway import ToolGateway


class ToolNode:
    """
    Graph node that executes tool calls from state.

    Reads tool_calls from the last assistant message and executes them
    via the tool gateway.
    """

    def __init__(self, tool_gateway: ToolGateway) -> None:
        self._tool_gateway = tool_gateway

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute tool calls from the last assistant message."""
        messages = state.get("messages", [])

        # Find tool calls from the last assistant message
        tool_calls_data: list[dict[str, Any]] = []
        for msg in reversed(messages):
            msg_data = msg if isinstance(msg, dict) else msg.model_dump()
            if msg_data.get("role") == "assistant" and msg_data.get("tool_calls"):
                tool_calls_data = msg_data["tool_calls"]
                break

        if not tool_calls_data:
            return {}

        # Convert to ToolCall objects
        tool_call_objects: list[ToolCall] = []
        for tc in tool_calls_data:
            func_data = tc.get("function", tc)
            arguments = func_data.get("arguments", {})
            if isinstance(arguments, str):
                import json
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
            tool_call_objects.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=func_data.get("name", ""),
                    arguments=arguments,
                )
            )

        # Execute via gateway
        results = await self._tool_gateway.call_many(tool_call_objects)

        # Build tool result messages
        result_messages = []
        for tool_call, result in zip(tool_call_objects, results, strict=True):
            result_messages.append({
                "role": "tool",
                "content": result.output if result.output else str(result.error),
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
            })

        return {"messages": result_messages}
