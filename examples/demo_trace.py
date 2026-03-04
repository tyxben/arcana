#!/usr/bin/env python3
"""
Demo: Trace System + Model Gateway

This demo shows:
1. Creating a trace context
2. Making LLM calls with tracing (DeepSeek + Gemini)
3. Reading and analyzing the trace

Run with:
    uv run python examples/demo_trace.py

Or directly:
    python examples/demo_trace.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcana.contracts.llm import LLMRequest, Message, MessageRole, ModelConfig
from arcana.contracts.trace import AgentRole, EventType, TraceEvent
from arcana.trace.reader import TraceReader
from arcana.trace.writer import TraceWriter
from arcana.utils.config import load_config
from arcana.utils.hashing import canonical_hash


async def demo_basic_trace():
    """Demo basic trace writing and reading."""
    print("=" * 60)
    print("Demo 1: Basic Trace Writing and Reading")
    print("=" * 60)

    # Create trace writer
    trace_dir = Path("./traces")
    writer = TraceWriter(trace_dir=trace_dir)

    # Create a context
    ctx = writer.create_context(task_id="demo-task")
    print(f"Created trace context: run_id={ctx.run_id}")

    # Write some events
    event1 = TraceEvent(
        run_id=ctx.run_id,
        task_id=ctx.task_id,
        step_id=ctx.new_step_id(),
        role=AgentRole.SYSTEM,
        event_type=EventType.LLM_CALL,
        llm_request_digest=canonical_hash({"message": "Hello"}),
        llm_response_digest=canonical_hash({"response": "World"}),
        model="demo-model",
    )
    writer.write(event1)
    print(f"Wrote event 1: {event1.event_type.value}")

    event2 = TraceEvent(
        run_id=ctx.run_id,
        task_id=ctx.task_id,
        step_id=ctx.new_step_id(),
        role=AgentRole.EXECUTOR,
        event_type=EventType.TOOL_CALL,
        metadata={"tool": "search", "query": "Python async"},
    )
    writer.write(event2)
    print(f"Wrote event 2: {event2.event_type.value}")

    # Read back the trace
    reader = TraceReader(trace_dir=trace_dir)
    events = reader.read_events(ctx.run_id)
    print(f"\nRead {len(events)} events from trace")

    # Get summary
    summary = reader.get_summary(ctx.run_id)
    print("\nTrace Summary:")
    print(f"  - Total events: {summary['total_events']}")
    print(f"  - LLM calls: {summary['llm_calls']}")
    print(f"  - Tool calls: {summary['tool_calls']}")

    # Cleanup
    writer.delete(ctx.run_id)
    print("\nDeleted trace file")


async def test_provider(provider_name: str, provider, model_config: ModelConfig, writer: TraceWriter):
    """Test a single provider."""
    print(f"\n--- Testing {provider_name} ({model_config.model_id}) ---")

    ctx = writer.create_context(task_id=f"{provider_name}-test")

    # Create request
    request = LLMRequest(
        messages=[
            Message(role=MessageRole.USER, content="What is 2 + 2? Reply with just the number."),
        ],
    )

    try:
        response = await provider.generate(request, model_config, ctx)
        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Tokens: {response.usage.total_tokens}")
        return True, ctx.run_id
    except Exception as e:
        print(f"Error: {e}")
        return False, ctx.run_id


async def demo_llm_with_trace():
    """Demo LLM calls with trace for both providers."""
    print("\n" + "=" * 60)
    print("Demo 2: LLM Calls with Trace (DeepSeek + Gemini)")
    print("=" * 60)

    # Load config
    config = load_config()

    # Setup trace
    trace_dir = Path("./traces")
    writer = TraceWriter(trace_dir=trace_dir)
    reader = TraceReader(trace_dir=trace_dir)

    results = []

    # Test DeepSeek
    if config.deepseek.api_key:
        from arcana.gateway.providers.deepseek import DeepSeekProvider

        provider = DeepSeekProvider(
            api_key=config.deepseek.api_key,
            base_url=config.deepseek.base_url,
            trace_writer=writer,
        )
        model_config = ModelConfig(
            provider="deepseek",
            model_id="deepseek-chat",
            temperature=0.0,
        )
        success, run_id = await test_provider("DeepSeek", provider, model_config, writer)
        results.append(("DeepSeek", success, run_id))
    else:
        print("\nDeepSeek: No API key configured")

    # Test Gemini
    if config.gemini.api_key:
        from arcana.gateway.providers.gemini import GeminiProvider

        provider = GeminiProvider(
            api_key=config.gemini.api_key,
            base_url=config.gemini.base_url,
            trace_writer=writer,
        )
        model_config = ModelConfig(
            provider="gemini",
            model_id="gemini-2.0-flash",
            temperature=0.0,
        )
        success, run_id = await test_provider("Gemini", provider, model_config, writer)
        results.append(("Gemini", success, run_id))
    else:
        print("\nGemini: No API key configured")

    # Summary
    print("\n" + "-" * 40)
    print("Results Summary:")
    for name, success, run_id in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")

        # Show trace summary
        if success:
            summary = reader.get_summary(run_id)
            print(f"    - Trace events: {summary['total_events']}")

    # Keep traces for inspection
    print(f"\nTrace files saved in: {trace_dir.absolute()}")


async def demo_hash_consistency():
    """Demo canonical hash consistency."""
    print("\n" + "=" * 60)
    print("Demo 3: Hash Consistency")
    print("=" * 60)

    # Same data, different order
    data1 = {"b": 2, "a": 1, "c": 3}
    data2 = {"a": 1, "b": 2, "c": 3}
    data3 = {"c": 3, "a": 1, "b": 2}

    hash1 = canonical_hash(data1)
    hash2 = canonical_hash(data2)
    hash3 = canonical_hash(data3)

    print(f"Data 1: {data1} -> {hash1}")
    print(f"Data 2: {data2} -> {hash2}")
    print(f"Data 3: {data3} -> {hash3}")
    print(f"\nAll hashes equal: {hash1 == hash2 == hash3}")

    # Float normalization
    float_data1 = {"value": 3.141592653589793}
    float_data2 = {"value": 3.1415926535}

    hash_f1 = canonical_hash(float_data1)
    hash_f2 = canonical_hash(float_data2)

    print(f"\nFloat data 1: {float_data1} -> {hash_f1}")
    print(f"Float data 2: {float_data2} -> {hash_f2}")
    print(f"Hashes equal (6 decimal normalization): {hash_f1 == hash_f2}")


async def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# Arcana Agent Platform - Week 1 Demo")
    print("#" * 60)

    await demo_basic_trace()
    await demo_hash_consistency()
    await demo_llm_with_trace()

    print("\n" + "#" * 60)
    print("# Demo Complete!")
    print("#" * 60)


if __name__ == "__main__":
    asyncio.run(main())
