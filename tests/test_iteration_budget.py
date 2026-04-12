"""Tests for iteration budget (max_iterations) in BudgetTracker and ConversationAgent."""

from __future__ import annotations

import threading

import pytest

from arcana.contracts.llm import Budget, LLMResponse, ModelConfig, TokenUsage
from arcana.gateway.base import BudgetExceededError
from arcana.gateway.budget import BudgetTracker
from arcana.gateway.registry import ModelGatewayRegistry

# ── BudgetTracker unit tests ───────────────────────────────────────────


class TestConsumeIteration:
    """consume_iteration increments the counter."""

    def test_increments_counter(self) -> None:
        tracker = BudgetTracker(max_iterations=10)
        assert tracker.iterations_used == 0
        tracker.consume_iteration()
        assert tracker.iterations_used == 1
        tracker.consume_iteration()
        assert tracker.iterations_used == 2

    def test_increments_without_limit(self) -> None:
        """consume_iteration works even when max_iterations is None."""
        tracker = BudgetTracker()
        tracker.consume_iteration()
        tracker.consume_iteration()
        assert tracker.iterations_used == 2


class TestCheckBudgetIterations:
    """check_budget raises BudgetExceededError when iterations exceeded."""

    def test_raises_when_exceeded(self) -> None:
        tracker = BudgetTracker(max_iterations=3, iterations_used=3)
        with pytest.raises(BudgetExceededError, match="Iteration budget exceeded") as exc_info:
            tracker.check_budget()
        assert exc_info.value.budget_type == "iterations"

    def test_no_raise_when_under_limit(self) -> None:
        tracker = BudgetTracker(max_iterations=5, iterations_used=4)
        tracker.check_budget()  # Should not raise

    def test_no_raise_when_unlimited(self) -> None:
        tracker = BudgetTracker(max_iterations=None, iterations_used=1000)
        tracker.check_budget()  # Should not raise

    def test_raises_at_exact_limit(self) -> None:
        """max_iterations=3, iterations_used=3 means we've used all 3 -- should raise."""
        tracker = BudgetTracker(max_iterations=3, iterations_used=3)
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()

    def test_consume_then_check(self) -> None:
        """Consume iterations until budget is exceeded."""
        tracker = BudgetTracker(max_iterations=2)
        tracker.consume_iteration()  # 1
        tracker.check_budget()  # OK
        tracker.consume_iteration()  # 2
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()


class TestIterationsRemaining:
    """iterations_remaining property."""

    def test_returns_remaining(self) -> None:
        tracker = BudgetTracker(max_iterations=10, iterations_used=3)
        assert tracker.iterations_remaining == 7

    def test_returns_zero_when_exhausted(self) -> None:
        tracker = BudgetTracker(max_iterations=5, iterations_used=5)
        assert tracker.iterations_remaining == 0

    def test_returns_zero_when_over(self) -> None:
        tracker = BudgetTracker(max_iterations=5, iterations_used=8)
        assert tracker.iterations_remaining == 0

    def test_returns_none_when_unlimited(self) -> None:
        tracker = BudgetTracker()
        assert tracker.iterations_remaining is None


class TestCanAffordIterations:
    """can_afford checks iterations parameter."""

    def test_can_afford_within_budget(self) -> None:
        tracker = BudgetTracker(max_iterations=10, iterations_used=5)
        assert tracker.can_afford(0, iterations=3) is True

    def test_cannot_afford_over_budget(self) -> None:
        tracker = BudgetTracker(max_iterations=10, iterations_used=8)
        assert tracker.can_afford(0, iterations=3) is False

    def test_can_afford_at_exact_limit(self) -> None:
        tracker = BudgetTracker(max_iterations=10, iterations_used=7)
        assert tracker.can_afford(0, iterations=3) is True

    def test_can_afford_unlimited(self) -> None:
        tracker = BudgetTracker()
        assert tracker.can_afford(0, iterations=1000) is True

    def test_can_afford_checks_all_dimensions(self) -> None:
        """Tokens OK but iterations over budget -> cannot afford."""
        tracker = BudgetTracker(max_tokens=1000, max_iterations=5, iterations_used=4)
        assert tracker.can_afford(100, iterations=2) is False
        assert tracker.can_afford(100, iterations=1) is True

    def test_backward_compatible_no_iterations_arg(self) -> None:
        """Calling can_afford without iterations still works."""
        tracker = BudgetTracker(max_iterations=5, iterations_used=4)
        assert tracker.can_afford(0) is True


class TestFromBudget:
    """from_budget reads max_iterations from Budget."""

    def test_reads_max_iterations(self) -> None:
        budget = Budget(max_tokens=1000, max_iterations=50)
        tracker = BudgetTracker.from_budget(budget)
        assert tracker.max_iterations == 50
        assert tracker.max_tokens == 1000

    def test_none_budget(self) -> None:
        tracker = BudgetTracker.from_budget(None)
        assert tracker.max_iterations is None

    def test_no_iterations_in_budget(self) -> None:
        budget = Budget(max_tokens=1000)
        tracker = BudgetTracker.from_budget(budget)
        assert tracker.max_iterations is None


class TestToSnapshot:
    """to_snapshot includes iteration fields."""

    def test_snapshot_includes_iterations(self) -> None:
        tracker = BudgetTracker(max_iterations=20, iterations_used=5)
        snapshot = tracker.to_snapshot()
        assert snapshot.max_iterations == 20
        assert snapshot.iterations_used == 5

    def test_snapshot_iterations_remaining(self) -> None:
        tracker = BudgetTracker(max_iterations=20, iterations_used=5)
        snapshot = tracker.to_snapshot()
        assert snapshot.iterations_remaining == 15

    def test_snapshot_budget_exhausted_by_iterations(self) -> None:
        tracker = BudgetTracker(max_iterations=5, iterations_used=5)
        snapshot = tracker.to_snapshot()
        assert snapshot.budget_exhausted is True


class TestReset:
    """reset clears iterations_used."""

    def test_reset_clears_iterations(self) -> None:
        tracker = BudgetTracker(max_iterations=10, iterations_used=7)
        tracker.reset()
        assert tracker.iterations_used == 0

    def test_reset_preserves_limits(self) -> None:
        tracker = BudgetTracker(max_iterations=10, max_tokens=5000, iterations_used=7)
        tracker.reset()
        assert tracker.max_iterations == 10
        assert tracker.max_tokens == 5000


class TestThreadSafety:
    """Thread safety for concurrent consume_iteration calls."""

    def test_concurrent_consume_iterations(self) -> None:
        """Many threads consuming iterations concurrently should not lose counts."""
        tracker = BudgetTracker(max_iterations=10000)
        num_threads = 50
        increments_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(increments_per_thread):
                tracker.consume_iteration()

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.iterations_used == num_threads * increments_per_thread


# ── ConversationAgent integration tests ────────────────────────────────

_MOCK_CONFIG = ModelConfig(provider="mock", model_id="mock-v1")


def _ok_response(text: str = "Done.") -> LLMResponse:
    return LLMResponse(
        content=text,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="mock",
        finish_reason="stop",
    )


def _continuing_response(text: str = "Thinking...") -> LLMResponse:
    """Response with finish_reason='length' — agent treats as 'cut off', continues."""
    return LLMResponse(
        content=text,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="mock",
        finish_reason="length",
    )


def _make_mock_gateway(responses: list[LLMResponse] | None = None) -> ModelGatewayRegistry:
    """Create a mock gateway registry with a fake provider."""
    if responses is None:
        responses = [_ok_response()]

    class FakeProvider:
        provider_name = "mock"
        supported_models = ["mock-v1"]
        default_model = "mock-v1"
        _idx = 0

        async def generate(self, request, config, trace_ctx=None):
            resp = responses[min(self._idx, len(responses) - 1)]
            self._idx += 1
            return resp

        async def stream(self, request, config, trace_ctx=None):
            # Force fallback to generate() so finish_reason is preserved
            raise NotImplementedError

        async def health_check(self):
            return True

    gateway = ModelGatewayRegistry()
    gateway.register("mock", FakeProvider())
    gateway.set_default("mock")
    return gateway


class TestConversationAgentIterationBudget:
    """ConversationAgent stops when iteration budget exhausted."""

    @pytest.mark.asyncio
    async def test_stops_on_iteration_budget(self) -> None:
        """Agent should stop when global iteration budget is exhausted."""
        from arcana.runtime.conversation import ConversationAgent

        # Use "length" finish_reason so the agent keeps looping (thinks LLM was cut off)
        gateway = _make_mock_gateway([_continuing_response()])
        # Budget allows only 2 iterations, but agent has max_turns=10
        tracker = BudgetTracker(max_iterations=2)

        agent = ConversationAgent(
            gateway=gateway,
            model_config=_MOCK_CONFIG,
            budget_tracker=tracker,
            max_turns=10,
        )

        # The agent consumes 1 iteration at start of each turn.
        # Turn 1: check (0 < 2 OK), consume -> 1, LLM returns "length"
        # Turn 2: check (1 < 2 OK), consume -> 2, LLM returns "length"
        # Turn 3: check (2 >= 2 RAISE)
        with pytest.raises(BudgetExceededError, match="Iteration budget exceeded"):
            await agent.run("test goal")

        assert tracker.iterations_used == 2

    @pytest.mark.asyncio
    async def test_no_limit_runs_to_max_turns(self) -> None:
        """Agent without iteration limit should run up to max_turns."""
        from arcana.runtime.conversation import ConversationAgent

        gateway = _make_mock_gateway()
        tracker = BudgetTracker()  # No iteration limit

        agent = ConversationAgent(
            gateway=gateway,
            model_config=_MOCK_CONFIG,
            budget_tracker=tracker,
            max_turns=3,
        )

        state = await agent.run("test goal")
        # Should complete all turns without error (LLM returns stop on turn 1)
        assert state is not None
        # At least 1 iteration was consumed
        assert tracker.iterations_used >= 1

    @pytest.mark.asyncio
    async def test_shared_budget_across_agents(self) -> None:
        """Two agents sharing a budget tracker deplete the shared pool."""
        from arcana.runtime.conversation import ConversationAgent

        gateway = _make_mock_gateway()
        # Shared budget: 3 iterations total
        shared_tracker = BudgetTracker(max_iterations=3)

        # First agent runs 1 turn (LLM returns stop immediately)
        agent1 = ConversationAgent(
            gateway=gateway,
            model_config=_MOCK_CONFIG,
            budget_tracker=shared_tracker,
            max_turns=5,
        )
        await agent1.run("goal 1")
        used_after_agent1 = shared_tracker.iterations_used
        assert used_after_agent1 >= 1

        # Second agent also shares the same tracker
        agent2 = ConversationAgent(
            gateway=gateway,
            model_config=_MOCK_CONFIG,
            budget_tracker=shared_tracker,
            max_turns=5,
        )
        await agent2.run("goal 2")
        # Both agents consumed from the same pool
        assert shared_tracker.iterations_used >= used_after_agent1 + 1
