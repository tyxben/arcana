"""Fast-path executors for DIRECT_ANSWER and SINGLE_TOOL intents."""

from __future__ import annotations

import json
import uuid
from typing import Any

from arcana.contracts.llm import (
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
)
from arcana.contracts.tool import ToolCall
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.tool_gateway.gateway import ToolGateway


class DirectExecutor:
    """Execute DIRECT_ANSWER and SINGLE_TOOL via fast paths.

    Bypasses the full agent loop for simple requests that need at most
    one LLM call or one tool call followed by a summary.
    """

    async def direct_answer(
        self,
        goal: str,
        gateway: ModelGatewayRegistry,
        config: ModelConfig,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Answer with a single LLM call, no tools.

        Args:
            goal: The user's request.
            gateway: Model gateway for LLM access.
            config: Model configuration to use.
            system_prompt: Optional system prompt (includes memory context).

        Returns:
            The LLM's response content.
        """
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=system_prompt or "You are a helpful assistant."),
                Message(role=MessageRole.USER, content=goal),
            ]
        )
        response = await gateway.generate(request=request, config=config)
        return response.content or ""

    async def single_tool_call(
        self,
        goal: str,
        tool_name: str,
        tool_args: dict[str, Any],
        gateway: ModelGatewayRegistry,
        config: ModelConfig,
        tool_gateway: ToolGateway,
    ) -> str:
        """Execute one tool call, then summarise the result with one LLM call.

        Args:
            goal: The user's original request.
            tool_name: Name of the tool to invoke.
            tool_args: Arguments to pass to the tool.
            gateway: Model gateway for LLM access.
            config: Model configuration to use.
            tool_gateway: Gateway for tool execution.

        Returns:
            LLM summary incorporating the tool result.
        """
        # Step 0: If no args provided, ask LLM to generate them
        if not tool_args:
            tool_args = await self._generate_tool_args(
                goal, tool_name, gateway, config, tool_gateway
            )

        # Step 1: Execute the tool
        call_id = uuid.uuid4().hex[:16]
        tool_call = ToolCall(id=call_id, name=tool_name, arguments=tool_args)
        tool_result = await tool_gateway.call(tool_call)

        # Step 2: Summarise the result
        summary_request = LLMRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You are a helpful assistant. "
                        "Summarise the tool result to answer the user's request."
                    ),
                ),
                Message(role=MessageRole.USER, content=goal),
                Message(
                    role=MessageRole.TOOL,
                    content=tool_result.output_str,
                    tool_call_id=call_id,
                ),
            ]
        )
        response = await gateway.generate(request=summary_request, config=config)
        return response.content or ""

    async def _generate_tool_args(
        self,
        goal: str,
        tool_name: str,
        gateway: ModelGatewayRegistry,
        config: ModelConfig,
        tool_gateway: ToolGateway,
    ) -> dict[str, Any]:
        """Ask LLM to generate arguments for a tool call.

        When the single-tool fast path is invoked without explicit arguments,
        this method asks the LLM to infer the correct arguments from the
        user's goal and the tool's schema.

        Args:
            goal: The user's original request.
            tool_name: Name of the tool to invoke.
            gateway: Model gateway for LLM access.
            config: Model configuration to use.
            tool_gateway: Gateway for tool execution.

        Returns:
            Dict of tool arguments inferred by the LLM.
        """
        tool_provider = tool_gateway.registry.get(tool_name)
        if not tool_provider:
            return {}

        schema_desc = json.dumps(tool_provider.spec.input_schema, ensure_ascii=False)
        arg_request = LLMRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        f"Generate the JSON arguments for calling the tool '{tool_name}'.\n"
                        f"Tool description: {tool_provider.spec.description}\n"
                        f"Parameter schema: {schema_desc}\n"
                        f"Return ONLY a JSON object with the arguments. No other text."
                    ),
                ),
                Message(role=MessageRole.USER, content=goal),
            ]
        )
        arg_response = await gateway.generate(request=arg_request, config=config)

        try:
            content = arg_response.content or "{}"
            # Strip markdown code fences if present
            stripped = content.strip()
            if stripped.startswith("```"):
                lines = stripped.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else "{}"
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return {}
