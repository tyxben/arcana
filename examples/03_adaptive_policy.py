"""
Arcana: Adaptive Policy

The agent chooses its own strategy -- direct answer, tool call, plan, or pivot.
"""

import asyncio
import os


async def main():
    from arcana.contracts.runtime import RuntimeConfig
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

    agent = Agent(
        policy=AdaptivePolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=10),
    )

    state = await agent.run(
        "Compare Python and Rust for web development. "
        "Consider performance, ecosystem, and learning curve."
    )

    print(f"Status: {state.status.value}")
    print(f"Steps: {state.current_step}")
    print(f"Answer:\n{state.working_memory.get('answer', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
