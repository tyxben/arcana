"""
Arcana: Ask User Tool

Demonstrates how the LLM can ask clarifying questions mid-execution.
The input_handler callback is called when the LLM invokes the ask_user tool.

Without an input_handler, the LLM receives a graceful fallback message
and proceeds with its best judgment -- interaction is a capability, not
a dependency.

Usage:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python examples/14_ask_user.py
"""

from __future__ import annotations

import asyncio
import os


async def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Set DEEPSEEK_API_KEY")
        return

    # --- With input_handler: LLM can ask the user questions ---
    # NOTE: input_handler on arcana.run() is being wired by another agent.
    # When available, usage will be:
    #
    #   result = await arcana.run(
    #       "Write me a short poem",
    #       provider="deepseek",
    #       api_key=api_key,
    #       input_handler=lambda q: input(f"Agent asks: {q}\nYour answer: "),
    #   )
    #   print(f"Result: {result.output}")

    # --- Working implementation using ConversationAgent directly ---
    # This demonstrates ask_user with the existing ConversationAgent API.
    from arcana.contracts.llm import ModelConfig
    from arcana.gateway.budget import BudgetTracker
    from arcana.gateway.providers.openai_compatible import create_deepseek_provider
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.routing.classifier import RuleBasedClassifier
    from arcana.runtime.conversation import ConversationAgent

    # Setup provider
    gateway = ModelGatewayRegistry()
    gateway.register("deepseek", create_deepseek_provider(api_key))
    gateway.set_default("deepseek")

    model_config = ModelConfig(provider="deepseek", model_id="deepseek-chat")
    budget = BudgetTracker(max_cost_usd=1.0)

    # With input_handler -- the LLM can ask questions
    print("--- With input handler (interactive) ---")
    agent = ConversationAgent(
        gateway=gateway,
        model_config=model_config,
        budget_tracker=budget,
        intent_classifier=RuleBasedClassifier(),
        max_turns=10,
        input_handler=lambda q: input(f"  Agent asks: {q}\n  Your answer: "),
    )
    state = await agent.run("Write me a short poem")
    answer = state.working_memory.get("answer", "")
    print(f"Result: {answer}")
    print()

    # Without input_handler -- LLM proceeds autonomously
    print("--- Without input handler (autonomous) ---")
    budget2 = BudgetTracker(max_cost_usd=1.0)
    agent2 = ConversationAgent(
        gateway=gateway,
        model_config=model_config,
        budget_tracker=budget2,
        intent_classifier=RuleBasedClassifier(),
        max_turns=10,
        # No input_handler -- ask_user returns fallback, LLM proceeds on its own
    )
    state2 = await agent2.run("Write me a short poem")
    answer2 = state2.working_memory.get("answer", "")
    print(f"Result: {answer2}")


if __name__ == "__main__":
    asyncio.run(main())
