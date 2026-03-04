"""ReAct agent - prebuilt agent ↔ tools cycle graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcana.graph.constants import END, START
from arcana.graph.nodes.llm_node import LLMNode
from arcana.graph.nodes.tool_node import ToolNode
from arcana.graph.state_graph import StateGraph

if TYPE_CHECKING:
    from arcana.contracts.llm import ModelConfig
    from arcana.gateway.base import ModelGateway
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.graph.compiled_graph import CompiledGraph
    from arcana.tool_gateway.gateway import ToolGateway


def create_react_agent(
    gateway: ModelGateway | ModelGatewayRegistry,
    tool_gateway: ToolGateway,
    *,
    model_config: ModelConfig | None = None,
    system_prompt: str | None = None,
    max_iterations: int = 25,
    checkpointer: Any | None = None,
) -> CompiledGraph:
    """
    Create a ReAct agent as a compiled graph.

    The agent follows the cycle:
        agent (LLM call) → tools (if tool_calls) → agent → ...
        agent (no tool_calls) → END

    Args:
        gateway: Model gateway for LLM calls
        tool_gateway: Tool gateway for tool execution
        model_config: Optional model configuration
        system_prompt: Optional system prompt
        max_iterations: Max loop iterations (safety limit)
        checkpointer: Optional checkpointer for interrupt/resume

    Returns:
        CompiledGraph ready to execute
    """
    iteration_count = 0

    llm_node = LLMNode(
        gateway,
        model_config=model_config,
        system_prompt=system_prompt,
    )
    tool_node = ToolNode(tool_gateway)

    def should_continue(state: dict[str, Any]) -> str:
        """Route: if last message has tool_calls → tools, else → END."""
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count > max_iterations:
            return END

        messages = state.get("messages", [])
        if not messages:
            return END

        last_msg = messages[-1]
        msg_data = last_msg if isinstance(last_msg, dict) else last_msg.model_dump()

        if msg_data.get("tool_calls"):
            return "tools"
        return END

    graph = StateGraph()
    graph.add_node("agent", llm_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
