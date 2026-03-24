"""Built-in ask_user tool -- lets the LLM ask the user clarifying questions.

Per the Constitution (Chapter IV): interaction is a capability, not a dependency.
If no input_handler is provided, the LLM gets a graceful fallback and proceeds
with best judgment. The user is never forced to interact mid-execution.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from arcana.contracts.tool import ASK_USER_TOOL_NAME, SideEffect, ToolSpec

ASK_USER_SPEC = ToolSpec(
    name=ASK_USER_TOOL_NAME,
    description=(
        "Ask the user a clarifying question when you need information "
        "that cannot be inferred from context."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user. Be specific and concise.",
            },
        },
        "required": ["question"],
    },
    when_to_use=(
        "When you need specific information from the user that cannot be "
        "reasonably inferred from the goal or available context. Use sparingly "
        "-- attempt to solve the problem first."
    ),
    what_to_expect="The user's answer as a text string. May be brief or detailed.",
    failure_meaning=(
        "User declined or couldn't answer. Proceed with your best judgment "
        "using available information."
    ),
    side_effect=SideEffect.READ,
)

_FALLBACK_MESSAGE = "No user input available. Proceed with your best judgment."


class AskUserHandler:
    """Handles ask_user tool calls by delegating to a user-provided callback.

    When no callback is provided, returns a graceful fallback message so the
    LLM can continue without blocking.
    """

    def __init__(self, input_handler: Callable[..., Any] | None = None) -> None:
        self._handler = input_handler

    async def handle(self, question: str) -> str:
        """Process an ask_user request, returning the user's answer or fallback."""
        if self._handler is None:
            return _FALLBACK_MESSAGE
        result = self._handler(question)
        if asyncio.iscoroutine(result):
            return await result
        return str(result)
