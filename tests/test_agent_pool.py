"""Tests for AgentPool, runtime.collaborate(), and ChatSession pool integration."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from arcana.contracts.llm import LLMResponse, TokenUsage
from arcana.gateway.budget import BudgetTracker
from arcana.multi_agent.agent_pool import AgentPool
from arcana.multi_agent.channel import Channel
from arcana.multi_agent.shared_context import SharedContext
from arcana.runtime_core import Budget, ChatSession, Runtime

# ── Helpers ─────────────────────────────────────────────────────────────


def _make_runtime(**kwargs) -> Runtime:
    """Create a minimal runtime with a mock provider."""
    rt = Runtime(providers={"deepseek": "test-key"}, **kwargs)
    return rt


class _MockProvider:
    """Minimal mock LLM provider."""

    provider_name = "mock"
    default_model = "mock-model"
    profile = None

    async def generate(self, request, config, trace_ctx=None):
        return LLMResponse(
            content="mock response",
            model="mock-model",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def close(self) -> None:
        pass


def _make_runtime_with_mock() -> Runtime:
    """Create a runtime backed by a mock provider (no real HTTP)."""
    rt = Runtime()
    mock_provider = _MockProvider()
    rt._gateway._providers["mock"] = mock_provider
    rt._gateway._default_provider = "mock"
    rt._config.default_provider = "mock"
    return rt


# ── AgentPool unit tests ──────────���─────────────────────────────────────


class TestAgentPoolAdd:
    def test_add_creates_named_agent(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        session = pool.add("planner", system="You are a planner")

        assert isinstance(session, ChatSession)
        assert "planner" in pool.agents

    def test_add_duplicate_name_raises(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        pool.add("planner", system="You plan")

        with pytest.raises(ValueError, match="already exists"):
            pool.add("planner", system="Duplicate")

    def test_agents_returns_all(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        pool.add("a", system="Agent A")
        pool.add("b", system="Agent B")
        pool.add("c", system="Agent C")

        agents = pool.agents
        assert set(agents.keys()) == {"a", "b", "c"}
        # Returned dict is a copy
        agents["x"] = None  # type: ignore[assignment]
        assert "x" not in pool.agents


class TestAgentPoolPrimitives:
    def test_channel_accessible(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        assert isinstance(pool.channel, Channel)

    def test_shared_accessible(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        assert isinstance(pool.shared, SharedContext)

    def test_add_registers_in_channel(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        pool.add("alpha", system="Agent alpha")

        assert "alpha" in pool.channel.agents


class TestAgentPoolClose:
    @pytest.mark.asyncio
    async def test_close_clears_everything(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        pool.add("a", system="Agent A")
        pool.shared.set("key", "value")

        await pool.close()

        assert pool.agents == {}
        assert pool.shared.keys() == []
        assert pool.channel.agents == []

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        rt = _make_runtime_with_mock()

        async with AgentPool(rt) as pool:
            pool.add("a", system="Agent A")
            pool.shared.set("key", "value")
            assert "a" in pool.agents

        # After exit, everything is cleared
        assert pool.agents == {}
        assert pool.shared.keys() == []


# ── Runtime.collaborate() tests ─────────────────────────────────────────


class TestCollaborate:
    def test_collaborate_returns_agent_pool(self):
        rt = _make_runtime_with_mock()
        pool = rt.collaborate()

        assert isinstance(pool, AgentPool)

    def test_collaborate_with_budget(self):
        rt = _make_runtime_with_mock()
        pool = rt.collaborate(budget=Budget(max_cost_usd=1.0, max_tokens=10_000))

        assert pool._budget_tracker is not None
        assert pool._budget_tracker.max_cost_usd == 1.0
        assert pool._budget_tracker.max_tokens == 10_000

    def test_collaborate_without_budget(self):
        rt = _make_runtime_with_mock()
        pool = rt.collaborate()

        assert pool._budget_tracker is None

    def test_collaborate_is_sync_not_coroutine(self):
        """Bug 1 regression: runtime.collaborate() must be a sync factory,
        not a coroutine. ``async with runtime.collaborate()`` is the
        documented usage and must work without ``await``."""
        import inspect

        rt = _make_runtime_with_mock()
        result = rt.collaborate()

        assert not inspect.iscoroutine(result), (
            "collaborate() returned a coroutine — would break `async with` usage"
        )
        assert isinstance(result, AgentPool)

    @pytest.mark.asyncio
    async def test_collaborate_async_with_usage(self):
        """The documented ``async with runtime.collaborate() as pool:`` works."""
        rt = _make_runtime_with_mock()

        async with rt.collaborate() as pool:
            pool.add("worker", system="You work")
            assert "worker" in pool.agents

        # __aexit__ clears state
        assert pool.agents == {}


# ── Agent.send() with mock gateway ──────────────────────────────────────


class TestAgentSend:
    @pytest.mark.asyncio
    async def test_agent_send_works(self):
        """Pool agent send() produces a ChatResponse via mocked LLM."""
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)
        agent = pool.add("worker", system="You are a worker")

        # Mock the ConversationAgent's astream to yield nothing and set state
        with patch(
            "arcana.runtime_core.ChatSession._build_agent"
        ) as mock_build:
            mock_agent = MagicMock()

            # Make astream an async generator
            async def _fake_stream(msg):
                return
                yield  # noqa: unreachable — makes this an async generator

            mock_agent.astream = _fake_stream
            mock_agent._state = MagicMock()
            mock_agent._state.tokens_used = 15
            mock_agent._state.cost_usd = 0.001
            mock_agent._state.working_memory = {"answer": "mock response"}
            mock_agent._state.last_context_report = None
            mock_agent.final_messages = []
            mock_build.return_value = mock_agent

            response = await agent.send("Do something")

        assert response.content == "mock response"
        assert response.tokens_used == 15


# ── Two agents communicating through shared context ──────────────────────


class TestTwoAgentsCommunication:
    @pytest.mark.asyncio
    async def test_agents_share_context(self):
        """Two agents can communicate through shared context."""
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        pool.add("writer", system="You write")
        pool.add("reader", system="You read")

        # Writer puts data in shared context
        pool.shared.set("plan", "Step 1: Do X, Step 2: Do Y")

        # Reader retrieves it
        plan = pool.shared.get("plan")
        assert plan == "Step 1: Do X, Step 2: Do Y"


# ── Budget sharing ──────���────────────────────────────────────────────────


class TestBudgetSharing:
    def test_pool_agents_share_budget_tracker(self):
        """All agents in a pool with a budget share the same tracker."""
        rt = _make_runtime_with_mock()
        tracker = BudgetTracker(max_cost_usd=5.0, max_tokens=100_000)
        pool = AgentPool(rt, budget_tracker=tracker)

        a = pool.add("a", system="Agent A")
        b = pool.add("b", system="Agent B")

        assert a._budget_tracker is b._budget_tracker
        assert a._budget_tracker is tracker

    def test_pool_without_budget_agents_get_own_tracker(self):
        """Without a shared budget, each agent creates its own tracker."""
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        a = pool.add("a", system="Agent A")
        b = pool.add("b", system="Agent B")

        # Each gets its own tracker (created from runtime default)
        assert a._budget_tracker is not b._budget_tracker


# ── runtime.team() deprecation ──────���───────────────────────────────────


class TestTeamDeprecation:
    @pytest.mark.asyncio
    async def test_team_emits_deprecation_warning(self):
        """runtime.team() emits a DeprecationWarning."""
        rt = _make_runtime_with_mock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                await rt.team(
                    "test goal",
                    [],
                    max_rounds=1,
                )
            except Exception:
                pass  # team() may fail with empty agents, that's fine

            # Check that a DeprecationWarning was issued
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "collaborate" in str(deprecation_warnings[0].message)


# ── Module export tests ──────────────────────────────────────────────────


class TestModuleExports:
    def test_agent_pool_in_multi_agent_module(self):
        import arcana.multi_agent as ma

        assert hasattr(ma, "AgentPool")

    def test_agent_pool_in_arcana_package(self):
        import arcana

        assert hasattr(arcana, "AgentPool")


# ── ChatSession per-session overrides ───────────────────────────────────


class TestChatSessionOverrides:
    def test_provider_override_stored(self):
        rt = _make_runtime_with_mock()
        session = ChatSession(
            rt, system_prompt="test", _provider="openai", _model="gpt-4"
        )
        assert session._provider_override == "openai"
        assert session._model_override == "gpt-4"

    def test_shared_budget_tracker(self):
        rt = _make_runtime_with_mock()
        tracker = BudgetTracker(max_cost_usd=2.0)
        session = ChatSession(rt, system_prompt="test", _budget_tracker=tracker)
        assert session._budget_tracker is tracker

    def test_default_budget_tracker_when_none(self):
        rt = _make_runtime_with_mock()
        session = ChatSession(rt, system_prompt="test")
        assert isinstance(session._budget_tracker, BudgetTracker)


# ── Per-agent cognitive primitives (v0.8.0) ─────────────────────────────


class TestPerAgentCognitivePrimitives:
    """v0.8.0: each pool agent owns its cognitive primitives config.

    Pool-level default is inherited when the per-agent arg is omitted.
    Per-agent ``[]`` is an explicit opt-out even when pool default opts in.
    """

    def test_session_resolves_per_agent_override(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)
        session = pool.add("planner", cognitive_primitives=["recall", "pin"])

        assert session._resolve_cognitive_primitives() == ["recall", "pin"]

    def test_session_resolves_pool_default_when_no_override(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt, default_cognitive_primitives=["pin"])
        session = pool.add("worker")  # no per-agent arg

        assert session._resolve_cognitive_primitives() == ["pin"]

    def test_per_agent_empty_opts_out_of_pool_default(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt, default_cognitive_primitives=["recall", "pin"])
        session = pool.add("silent", cognitive_primitives=[])

        assert session._resolve_cognitive_primitives() == []

    def test_per_agent_overrides_pool_default_with_subset(self):
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt, default_cognitive_primitives=["recall", "pin"])
        session = pool.add("minimal", cognitive_primitives=["pin"])

        assert session._resolve_cognitive_primitives() == ["pin"]

    def test_bare_runtime_chat_unaffected(self):
        """Sessions built via runtime.chat() still read runtime config."""
        rt = _make_runtime_with_mock()
        rt._config.cognitive_primitives = ["recall"]
        session = ChatSession(rt, system_prompt="x")

        assert session._resolve_cognitive_primitives() == ["recall"]

    def test_each_pool_agent_is_independent_session(self):
        """Per-agent cognitive state lives on independent ChatSessions —
        no shared handler, no shared config."""
        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        a = pool.add("a", cognitive_primitives=["pin"])
        b = pool.add("b", cognitive_primitives=["recall"])

        assert a is not b
        assert a._cognitive_primitives_override == ["pin"]
        assert b._cognitive_primitives_override == ["recall"]

    def test_pool_collaborate_default_propagates(self):
        """runtime.collaborate(cognitive_primitives=[...]) sets pool default."""
        rt = _make_runtime_with_mock()
        pool = rt.collaborate(cognitive_primitives=["pin"])
        session = pool.add("w")

        assert session._resolve_cognitive_primitives() == ["pin"]


class TestPerAgentCognitiveCollision:
    """Tool-name collision with an active cognitive primitive is a
    configuration error — either rename the user tool or drop the
    primitive (Principle 5: structured feedback, not silent shadowing)."""

    def test_user_tool_named_recall_collides_when_recall_active(self):
        import arcana

        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        @arcana.tool()
        def recall(turn: int) -> str:
            """User tool."""
            return f"user recall {turn}"

        with pytest.raises(ValueError, match="recall"):
            pool.add(
                "collider",
                cognitive_primitives=["recall"],
                tools=[recall],
            )

    def test_user_tool_named_pin_collides(self):
        import arcana

        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        @arcana.tool()
        def pin(content: str) -> str:
            """User tool."""
            return content

        with pytest.raises(ValueError, match="pin"):
            pool.add("collider", cognitive_primitives=["pin"], tools=[pin])

    def test_user_tool_named_unpin_collides_when_pin_active(self):
        """Activating 'pin' also reserves 'unpin' — they're a family."""
        import arcana

        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        @arcana.tool()
        def unpin(pin_id: str) -> str:
            """User tool."""
            return pin_id

        with pytest.raises(ValueError, match="unpin"):
            pool.add("collider", cognitive_primitives=["pin"], tools=[unpin])

    def test_user_tool_with_same_name_is_fine_when_primitive_inactive(self):
        """If the primitive isn't active for this agent, the user tool
        is free to use the name — no shadowing happens."""
        import arcana

        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        @arcana.tool()
        def recall(turn: int) -> str:
            """User tool."""
            return f"user recall {turn}"

        # cognitive_primitives=[] means this agent has no recall primitive;
        # the user's own 'recall' tool should be allowed.
        session = pool.add(
            "plain",
            cognitive_primitives=[],
            tools=[recall],
        )
        assert session is not None

    def test_unrelated_user_tools_pass_through(self):
        import arcana

        rt = _make_runtime_with_mock()
        pool = AgentPool(rt)

        @arcana.tool()
        def search(query: str) -> str:
            """User tool."""
            return f"results for {query}"

        session = pool.add(
            "searcher",
            cognitive_primitives=["recall", "pin"],
            tools=[search],
        )
        assert session is not None
