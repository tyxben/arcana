"""Plan-Execute agent - planner → executor → verifier → (replan | END)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcana.graph.constants import END, START
from arcana.graph.state_graph import StateGraph

if TYPE_CHECKING:
    from arcana.contracts.llm import ModelConfig
    from arcana.gateway.base import ModelGateway
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.graph.compiled_graph import CompiledGraph
    from arcana.tool_gateway.gateway import ToolGateway


def create_plan_execute_agent(
    gateway: ModelGateway | ModelGatewayRegistry,
    tool_gateway: ToolGateway,
    *,
    model_config: ModelConfig | None = None,
    planner_prompt: str = "Create a step-by-step plan to accomplish the task.",
    max_replans: int = 3,
    checkpointer: Any | None = None,
) -> CompiledGraph:
    """
    Create a Plan-Execute agent as a compiled graph.

    Flow: planner → executor → verifier → (replan | END)

    Args:
        gateway: Model gateway for LLM calls
        tool_gateway: Tool gateway for tool execution
        model_config: Optional model configuration
        planner_prompt: System prompt for the planner
        max_replans: Maximum number of replan iterations
        checkpointer: Optional checkpointer for interrupt/resume

    Returns:
        CompiledGraph ready to execute
    """
    from arcana.contracts.llm import LLMRequest, Message, MessageRole
    from arcana.contracts.llm import ModelConfig as MC

    config = model_config or MC(provider="default", model_id="default")
    replan_count = 0

    async def planner(state: dict[str, Any]) -> dict[str, Any]:
        """Generate or revise a plan."""
        messages = state.get("messages", [])
        plan_messages = [
            Message(role=MessageRole.SYSTEM, content=planner_prompt),
            *[
                Message(**m) if isinstance(m, dict) else m
                for m in messages
            ],
        ]

        if state.get("verification_feedback"):
            plan_messages.append(
                Message(
                    role=MessageRole.USER,
                    content=f"Previous plan feedback: {state['verification_feedback']}. "
                    "Please revise the plan.",
                )
            )

        request = LLMRequest(messages=plan_messages)
        response = await gateway.generate(request, config)

        return {
            "plan": response.content or "",
            "messages": [{"role": "assistant", "content": response.content or ""}],
        }

    async def executor(state: dict[str, Any]) -> dict[str, Any]:
        """Execute the current plan step."""
        plan = state.get("plan", "")
        messages = state.get("messages", [])

        exec_messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=f"Execute the following plan step by step:\n{plan}",
            ),
            *[
                Message(**m) if isinstance(m, dict) else m
                for m in messages
            ],
        ]

        request = LLMRequest(messages=exec_messages)
        response = await gateway.generate(request, config)

        result: dict[str, Any] = {
            "messages": [{"role": "assistant", "content": response.content or ""}],
        }

        # If the executor wants to call tools, include them
        if response.tool_calls:
            result["messages"][-1]["tool_calls"] = [
                tc.model_dump() for tc in response.tool_calls
            ]

        return result

    async def verifier(state: dict[str, Any]) -> dict[str, Any]:
        """Verify if the plan execution achieved the goal."""
        messages = state.get("messages", [])

        verify_messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="Evaluate whether the task has been completed successfully. "
                "Respond with either 'COMPLETE' if done, or provide specific "
                "feedback on what needs to be improved.",
            ),
            *[
                Message(**m) if isinstance(m, dict) else m
                for m in messages
            ],
        ]

        request = LLMRequest(messages=verify_messages)
        response = await gateway.generate(request, config)

        content = response.content or ""
        is_complete = "COMPLETE" in content.upper()

        return {
            "is_complete": is_complete,
            "verification_feedback": "" if is_complete else content,
            "messages": [{"role": "assistant", "content": content}],
        }

    def should_replan(state: dict[str, Any]) -> str:
        """Decide whether to replan or finish."""
        nonlocal replan_count

        if state.get("is_complete", False):
            return END

        replan_count += 1
        if replan_count > max_replans:
            return END

        return "planner"

    graph = StateGraph()
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("verifier", verifier)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "verifier")
    graph.add_conditional_edges(
        "verifier", should_replan, {"planner": "planner", END: END}
    )

    return graph.compile(checkpointer=checkpointer)
