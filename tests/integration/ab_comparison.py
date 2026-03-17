"""
A/B Comparison: ConversationAgent (V2) vs Agent+AdaptivePolicy (V1)

Same tasks, same provider, compare metrics.
Usage: DEEPSEEK_API_KEY=sk-xxx uv run python tests/integration/ab_comparison.py
"""

import asyncio
import os
import time


TASKS = [
    {
        "goal": "What is the capital of France?",
        "check": lambda answer: "paris" in answer.lower(),
        "label": "Simple factual",
    },
    {
        "goal": (
            "Compare Python and Rust for web development. "
            "Consider performance, ecosystem, and learning curve. "
            "Give a recommendation."
        ),
        "check": lambda answer: (
            len(answer) > 200
            and "python" in answer.lower()
            and "rust" in answer.lower()
        ),
        "label": "Multi-aspect analysis",
    },
    {
        "goal": (
            "Explain quantum computing in simple terms, "
            "then give 3 real-world applications."
        ),
        "check": lambda answer: len(answer) > 100,
        "label": "Explain + enumerate",
    },
    {
        "goal": (
            "Write a Python function that checks if a string is a palindrome. "
            "Include docstring and type hints."
        ),
        "check": lambda answer: "def " in answer and "palindrome" in answer.lower(),
        "label": "Code generation",
    },
    {
        "goal": (
            "What are the pros and cons of microservices vs monolith architecture? "
            "Give a decision framework."
        ),
        "check": lambda answer: (
            "micro" in answer.lower() and "monolith" in answer.lower()
        ),
        "label": "Technical comparison",
    },
]


async def run_v2(goal: str, gateway, model_config) -> dict:
    """V2: ConversationAgent"""
    from arcana.runtime.conversation import ConversationAgent

    agent = ConversationAgent(
        gateway=gateway,
        model_config=model_config,
        max_turns=10,
    )
    start = time.monotonic()
    state = await agent.run(goal)
    elapsed = time.monotonic() - start
    return {
        "engine": "V2 Conversation",
        "status": state.status.value,
        "steps": state.current_step,
        "tokens": state.tokens_used,
        "time_s": round(elapsed, 1),
        "answer": state.working_memory.get("answer", ""),
    }


async def run_v1(goal: str, gateway, model_config) -> dict:
    """V1: Agent + AdaptivePolicy"""
    from arcana.contracts.runtime import RuntimeConfig
    from arcana.runtime.agent import Agent
    from arcana.runtime.policies.adaptive import AdaptivePolicy
    from arcana.runtime.reducers.default import DefaultReducer

    agent = Agent(
        policy=AdaptivePolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=10),
        auto_route=False,
    )
    start = time.monotonic()
    state = await agent.run(goal)
    elapsed = time.monotonic() - start
    return {
        "engine": "V1 Adaptive",
        "status": state.status.value,
        "steps": state.current_step,
        "tokens": state.tokens_used,
        "time_s": round(elapsed, 1),
        "answer": state.working_memory.get("answer", ""),
    }


async def main() -> None:
    from arcana.contracts.llm import ModelConfig
    from arcana.gateway.providers.openai_compatible import create_deepseek_provider
    from arcana.gateway.registry import ModelGatewayRegistry

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Set DEEPSEEK_API_KEY")
        return

    gateway = ModelGatewayRegistry()
    gateway.register("deepseek", create_deepseek_provider(api_key))
    gateway.set_default("deepseek")
    config = ModelConfig(provider="deepseek", model_id="deepseek-chat")

    header = (
        f"{'Task':<25} {'Engine':<18} {'Status':<10} "
        f"{'Steps':<6} {'Tokens':<8} {'Time':<6} {'Pass'}"
    )
    print(header)
    print("-" * 90)

    v1_total = {"tokens": 0, "steps": 0, "time": 0.0, "pass": 0}
    v2_total = {"tokens": 0, "steps": 0, "time": 0.0, "pass": 0}

    for task in TASKS:
        # --- Run V2 ---
        try:
            r2 = await run_v2(task["goal"], gateway, config)
            p2 = task["check"](r2["answer"]) if r2["answer"] else False
            v2_total["tokens"] += r2["tokens"]
            v2_total["steps"] += r2["steps"]
            v2_total["time"] += r2["time_s"]
            v2_total["pass"] += int(p2)
        except Exception as e:
            r2 = {
                "engine": "V2",
                "status": "error",
                "steps": 0,
                "tokens": 0,
                "time_s": 0,
                "answer": "",
            }
            p2 = False
            print(f"  [V2 ERROR] {e}")

        # --- Run V1 ---
        try:
            r1 = await run_v1(task["goal"], gateway, config)
            p1 = task["check"](r1["answer"]) if r1["answer"] else False
            v1_total["tokens"] += r1["tokens"]
            v1_total["steps"] += r1["steps"]
            v1_total["time"] += r1["time_s"]
            v1_total["pass"] += int(p1)
        except Exception as e:
            r1 = {
                "engine": "V1",
                "status": "error",
                "steps": 0,
                "tokens": 0,
                "time_s": 0,
                "answer": "",
            }
            p1 = False
            print(f"  [V1 ERROR] {e}")

        pass_v2 = "PASS" if p2 else "FAIL"
        pass_v1 = "PASS" if p1 else "FAIL"

        print(
            f"{task['label']:<25} {'V2 Conversation':<18} "
            f"{r2['status']:<10} {r2['steps']:<6} "
            f"{r2['tokens']:<8} {r2['time_s']:<6} {pass_v2}"
        )
        print(
            f"{'':<25} {'V1 Adaptive':<18} "
            f"{r1['status']:<10} {r1['steps']:<6} "
            f"{r1['tokens']:<8} {r1['time_s']:<6} {pass_v1}"
        )
        print()

    print("=" * 90)
    print(
        f"{'TOTALS':<25} {'V2 Conversation':<18} "
        f"{'':10} {v2_total['steps']:<6} "
        f"{v2_total['tokens']:<8} {v2_total['time']:<6} "
        f"{v2_total['pass']}/5"
    )
    print(
        f"{'':25} {'V1 Adaptive':<18} "
        f"{'':10} {v1_total['steps']:<6} "
        f"{v1_total['tokens']:<8} {v1_total['time']:<6} "
        f"{v1_total['pass']}/5"
    )

    # Token savings
    if v1_total["tokens"] > 0:
        savings = (1 - v2_total["tokens"] / v1_total["tokens"]) * 100
        print(f"\nV2 token savings vs V1: {savings:.0f}%")


if __name__ == "__main__":
    asyncio.run(main())
