"""
Arcana: Intent Router

Simple questions get direct answers (1 LLM call).
Complex tasks enter the agent loop.
"""

import asyncio
import os


async def main():
    from arcana.contracts.runtime import RuntimeConfig
    from arcana.gateway.providers.openai_compatible import create_deepseek_provider
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.routing.classifier import RuleBasedClassifier
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
        intent_classifier=RuleBasedClassifier(),
        auto_route=True,
    )

    # Simple question -> direct answer (1 step)
    print("--- Simple question ---")
    state = await agent.run("What is the capital of Japan?")
    print(f"Steps: {state.current_step}, Answer: {state.working_memory.get('answer', 'N/A')[:100]}")

    # Complex task -> agent loop (multiple steps)
    print("\n--- Complex task ---")
    state = await agent.run(
        "Design a REST API for a todo app. Include endpoints, data models, and error handling."
    )
    print(
        f"Steps: {state.current_step}, "
        f"Answer: {state.working_memory.get('answer', 'N/A')[:200]}..."
    )


if __name__ == "__main__":
    asyncio.run(main())
