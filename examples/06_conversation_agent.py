"""
Demo: ConversationAgent -- LLM-native execution.

Demonstrates ConversationAgent (V2 engine):
  - Direct answer
  - Tool usage via native tool_use
  - Streaming events

Usage:
    DEEPSEEK_API_KEY="sk-xxx" uv run python examples/06_conversation_agent.py
"""

import asyncio
import os
import sys

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    print("ERROR: Set DEEPSEEK_API_KEY environment variable")
    sys.exit(1)


async def test_v2_direct():
    """V2: Simple question, should complete in 1 turn."""
    from arcana.contracts.llm import ModelConfig
    from arcana.gateway.providers.openai_compatible import create_deepseek_provider
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.runtime.conversation import ConversationAgent

    gateway = ModelGatewayRegistry()
    gateway.register("deepseek", create_deepseek_provider(DEEPSEEK_API_KEY))
    gateway.set_default("deepseek")

    agent = ConversationAgent(
        gateway=gateway,
        model_config=ModelConfig(provider="deepseek", model_id="deepseek-chat"),
        max_turns=5,
    )

    state = await agent.run("What is the capital of France?")
    print(f"  Status: {state.status.value}")
    print(f"  Turns: {state.current_step}")
    print(f"  Answer: {str(state.working_memory.get('answer', 'N/A'))[:200]}")
    print(f"  Tokens: {state.tokens_used}")
    assert state.status.value == "completed"


async def test_v2_with_tools():
    """V2: Tool usage via native tool_use."""
    from arcana.contracts.llm import ModelConfig
    from arcana.contracts.tool import ToolCall, ToolError, ToolResult, ToolSpec
    from arcana.gateway.providers.openai_compatible import create_deepseek_provider
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.runtime.conversation import ConversationAgent
    from arcana.tool_gateway.gateway import ToolGateway
    from arcana.tool_gateway.registry import ToolRegistry

    gateway = ModelGatewayRegistry()
    gateway.register("deepseek", create_deepseek_provider(DEEPSEEK_API_KEY))
    gateway.set_default("deepseek")

    # Register calculator tool
    class CalcProvider:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="calculator",
                description="Calculate math expressions. Input: {'expression': '<math expr>'}",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate, e.g. '15 * 37 + 89'",
                        }
                    },
                    "required": ["expression"],
                },
            )

        async def execute(self, call: ToolCall) -> ToolResult:
            try:
                expr = call.arguments.get("expression", "0")
                result = str(eval(expr))  # noqa: S307
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    success=True,
                    output=result,
                )
            except Exception as e:
                from arcana.contracts.tool import ToolErrorCategory

                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    success=False,
                    error=ToolError(
                        category=ToolErrorCategory.UNEXPECTED, message=str(e)
                    ),
                )

        async def health_check(self) -> bool:
            return True

    registry = ToolRegistry()
    registry.register(CalcProvider())
    tool_gw = ToolGateway(registry=registry)

    agent = ConversationAgent(
        gateway=gateway,
        model_config=ModelConfig(provider="deepseek", model_id="deepseek-chat"),
        tool_gateway=tool_gw,
        max_turns=5,
    )

    state = await agent.run("What is 15 * 37 + 89?")
    print(f"  Status: {state.status.value}")
    print(f"  Turns: {state.current_step}")
    print(f"  Answer: {str(state.working_memory.get('answer', 'N/A'))[:200]}")
    assert "644" in str(state.working_memory.get("answer", ""))


async def test_v2_streaming():
    """V2: Stream events during execution."""
    from arcana.contracts.llm import ModelConfig
    from arcana.gateway.providers.openai_compatible import create_deepseek_provider
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.runtime.conversation import ConversationAgent

    gateway = ModelGatewayRegistry()
    gateway.register("deepseek", create_deepseek_provider(DEEPSEEK_API_KEY))
    gateway.set_default("deepseek")

    agent = ConversationAgent(
        gateway=gateway,
        model_config=ModelConfig(provider="deepseek", model_id="deepseek-chat"),
        max_turns=3,
    )

    events = []
    async for event in agent.astream("Explain what Python is in 2 sentences."):
        events.append(event)
        print(f"    Event: {event.event_type.value} | {(event.content or '')[:80]}")

    assert len(events) >= 2  # At least RUN_START + RUN_COMPLETE
    print(f"  Total events: {len(events)}")


async def main():
    """Run all demos."""
    demos = [
        ("V2 Direct Answer", test_v2_direct),
        ("V2 Tool Usage", test_v2_with_tools),
        ("V2 Streaming", test_v2_streaming),
    ]

    passed = 0
    for name, fn in demos:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            await fn()
            print("  [PASS]")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{len(demos)} passed")
    print(f"{'='*60}")

    if passed == len(demos):
        print("  All demos passed!")
    else:
        print("  Some demos failed. Check output above.")


if __name__ == "__main__":
    asyncio.run(main())
