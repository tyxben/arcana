"""
Arcana: Budget Control

Agent stops when budget is exhausted -- never runs away.
"""

import asyncio
import os


async def main():
    from arcana.contracts.runtime import RuntimeConfig
    from arcana.gateway.budget import BudgetTracker
    from arcana.gateway.providers.openai_compatible import create_deepseek_provider
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.runtime.agent import Agent
    from arcana.runtime.policies.adaptive import AdaptivePolicy
    from arcana.runtime.reducers.default import DefaultReducer

    gateway = ModelGatewayRegistry()
    gateway.register(
        "deepseek",
        create_deepseek_provider(api_key=os.environ["DEEPSEEK_API_KEY"]),
    )
    gateway.set_default("deepseek")

    # Strict budget: max 2000 tokens
    budget = BudgetTracker(max_tokens=2000, max_cost_usd=0.01)

    agent = Agent(
        policy=AdaptivePolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=20),
        budget_tracker=budget,
    )

    state = await agent.run(
        "Write a detailed essay about the history of artificial intelligence"
    )

    print(f"Status: {state.status.value}")
    print(f"Steps: {state.current_step}")
    print(f"Tokens: {state.tokens_used}")
    print(f"Budget stopped execution: {state.status.value != 'completed'}")


if __name__ == "__main__":
    asyncio.run(main())
