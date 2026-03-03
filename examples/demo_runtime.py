"""
Demo of Agent Runtime capabilities.

This example demonstrates:
- Basic agent execution with ReAct policy
- Budget tracking
- Checkpointing and resume
- Trace logging
- Progress detection
"""

import asyncio
import os
from pathlib import Path

from arcana.contracts.runtime import RuntimeConfig
from arcana.gateway.budget import BudgetTracker
from arcana.gateway.providers import create_deepseek_provider
from arcana.gateway.registry import ModelGatewayRegistry
from arcana.runtime.agent import Agent
from arcana.runtime.policies.react import ReActPolicy
from arcana.runtime.reducers.default import DefaultReducer
from arcana.trace.writer import TraceWriter


async def demo_basic_execution():
    """Demonstrate basic agent execution."""
    print("=" * 60)
    print("Demo 1: Basic Agent Execution")
    print("=" * 60)

    # Setup
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DEEPSEEK_API_KEY not set, skipping real execution demo")
        return

    gateway = ModelGatewayRegistry()
    gateway.register("deepseek", create_deepseek_provider(api_key))
    gateway.set_default("deepseek")

    # Create agent
    agent = Agent(
        policy=ReActPolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(max_steps=10),
    )

    # Run
    print("\n🚀 Running agent with goal: 'Calculate 15 + 27'")
    state = await agent.run("Calculate 15 + 27 and explain the result")

    print(f"\n✅ Execution completed!")
    print(f"   Status: {state.status.value}")
    print(f"   Steps: {state.current_step}")
    print(f"   Tokens used: {state.tokens_used}")


async def demo_with_budget_and_trace():
    """Demonstrate budget tracking and trace logging."""
    print("\n" + "=" * 60)
    print("Demo 2: Budget Tracking & Trace Logging")
    print("=" * 60)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DEEPSEEK_API_KEY not set, skipping demo")
        return

    # Setup directories
    trace_dir = Path("./demo_traces")
    checkpoint_dir = Path("./demo_checkpoints")

    gateway = ModelGatewayRegistry()
    gateway.register("deepseek", create_deepseek_provider(api_key))
    gateway.set_default("deepseek")

    # Budget tracker (limit to 1000 tokens)
    budget_tracker = BudgetTracker(max_tokens=1000, max_cost_usd=0.01)

    # Trace writer
    trace_writer = TraceWriter(trace_dir=trace_dir)

    # Create agent
    agent = Agent(
        policy=ReActPolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(
            max_steps=10,
            checkpoint_interval_steps=2,
        ),
        budget_tracker=budget_tracker,
        trace_writer=trace_writer,
    )

    # Configure checkpoint directory
    agent.state_manager.checkpoint_dir = checkpoint_dir

    # Run
    print("\n🚀 Running agent with budget limits...")
    goal = "Explain what the Fibonacci sequence is and give the first 5 numbers"
    state = await agent.run(goal)

    print(f"\n✅ Execution completed!")
    print(f"   Status: {state.status.value}")
    print(f"   Steps: {state.current_step}")

    # Show budget usage
    budget_snapshot = budget_tracker.to_snapshot()
    print(f"\n💰 Budget Usage:")
    print(f"   Tokens: {budget_snapshot.tokens_used}/{budget_snapshot.max_tokens}")
    print(f"   Cost: ${budget_snapshot.cost_usd:.4f}")
    print(f"   Time: {budget_snapshot.time_ms}ms")

    # Show trace file
    trace_files = list(trace_dir.glob(f"{state.run_id}.jsonl"))
    if trace_files:
        print(f"\n📝 Trace file: {trace_files[0]}")
        print(f"   (Use TraceReader to analyze)")

    # Show checkpoints
    checkpoint_files = list(checkpoint_dir.glob(f"{state.run_id}.checkpoints.jsonl"))
    if checkpoint_files:
        print(f"\n💾 Checkpoint file: {checkpoint_files[0]}")


async def demo_progress_detection():
    """Demonstrate progress detection with a stuck agent."""
    print("\n" + "=" * 60)
    print("Demo 3: Progress Detection")
    print("=" * 60)

    # Mock gateway that repeats same response
    class MockStuckGateway:
        """Gateway that simulates a stuck agent."""

        async def generate(self, request, config, trace_ctx=None):
            from arcana.contracts.llm import LLMResponse, Usage

            return LLMResponse(
                content="Thought: I need to think about this\nAction: Think more",
                model="mock",
                usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

    gateway = ModelGatewayRegistry()
    gateway.register("mock", MockStuckGateway())
    gateway.set_default("mock")

    agent = Agent(
        policy=ReActPolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(
            max_steps=10,
            max_consecutive_no_progress=3,
        ),
    )

    print("\n🚀 Running agent that will get stuck...")
    state = await agent.run("Some task")

    print(f"\n⚠️  Agent stopped!")
    print(f"   Status: {state.status.value}")
    print(f"   Reason: No progress detected")
    print(f"   Consecutive no-progress steps: {state.consecutive_no_progress}")
    print(f"   Total steps: {state.current_step}")


async def demo_checkpointing_and_resume():
    """Demonstrate checkpointing and resume."""
    print("\n" + "=" * 60)
    print("Demo 4: Checkpointing and Resume")
    print("=" * 60)

    checkpoint_dir = Path("./demo_checkpoints")

    # Simulate agent that stops mid-execution
    class MockPartialGateway:
        """Gateway that simulates stopping mid-task."""

        def __init__(self):
            self.call_count = 0

        async def generate(self, request, config, trace_ctx=None):
            from arcana.contracts.llm import LLMResponse, Usage

            self.call_count += 1

            if self.call_count <= 2:
                return LLMResponse(
                    content=f"Thought: Step {self.call_count}\nAction: Continue",
                    model="mock",
                    usage=Usage(
                        prompt_tokens=10, completion_tokens=10, total_tokens=20
                    ),
                )
            else:
                return LLMResponse(
                    content="Thought: Done\nAction: FINISH",
                    model="mock",
                    usage=Usage(
                        prompt_tokens=10, completion_tokens=10, total_tokens=20
                    ),
                )

    gateway = ModelGatewayRegistry()
    gateway.register("mock", MockPartialGateway())
    gateway.set_default("mock")

    agent = Agent(
        policy=ReActPolicy(),
        reducer=DefaultReducer(),
        gateway=gateway,
        config=RuntimeConfig(
            max_steps=5,
            checkpoint_interval_steps=1,
        ),
    )

    agent.state_manager.checkpoint_dir = checkpoint_dir

    print("\n🚀 Running agent with checkpointing...")
    state = await agent.run("Multi-step task")

    print(f"\n✅ Execution completed!")
    print(f"   Steps: {state.current_step}")

    # Load and show checkpoints
    snapshot = await agent.state_manager.load_snapshot(state.run_id)
    if snapshot:
        print(f"\n💾 Latest checkpoint:")
        print(f"   Step ID: {snapshot.step_id}")
        print(f"   Resumable: {snapshot.is_resumable}")
        print(f"   State hash: {snapshot.state_hash}")


async def main():
    """Run all demos."""
    print("\n🎯 Arcana Agent Runtime Demos\n")

    # Demo 1: Basic execution (requires API key)
    await demo_basic_execution()

    # Demo 2: Budget & Trace (requires API key)
    await demo_with_budget_and_trace()

    # Demo 3: Progress detection (mock)
    await demo_progress_detection()

    # Demo 4: Checkpointing (mock)
    await demo_checkpointing_and_resume()

    print("\n" + "=" * 60)
    print("✨ All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
