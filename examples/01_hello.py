"""
Arcana: Hello World

The simplest possible agent -- one LLM call.
"""

import asyncio
import os


async def main():
    from arcana.contracts.llm import LLMRequest, Message, MessageRole, ModelConfig
    from arcana.gateway.providers.openai_compatible import create_deepseek_provider
    from arcana.gateway.registry import ModelGatewayRegistry

    # Setup
    gateway = ModelGatewayRegistry()
    gateway.register(
        "deepseek",
        create_deepseek_provider(api_key=os.environ["DEEPSEEK_API_KEY"]),
    )
    gateway.set_default("deepseek")

    # One call
    request = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="Say hello in 3 languages")]
    )
    response = await gateway.generate(
        request=request,
        config=ModelConfig(provider="deepseek", model_id="deepseek-chat"),
    )
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
