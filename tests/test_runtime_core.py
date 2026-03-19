"""Tests for Runtime, Session, Budget, and SDK entry points."""

from unittest.mock import AsyncMock, patch

import pytest

from arcana.runtime_core import (
    AgentConfig,
    Budget,
    RunResult,
    Runtime,
    RuntimeConfig,
    Session,
    TeamResult,
)


class TestBudget:
    def test_defaults(self):
        b = Budget()
        assert b.max_cost_usd == 10.0
        assert b.max_tokens == 500_000

    def test_custom(self):
        b = Budget(max_cost_usd=0.5, max_tokens=1000)
        assert b.max_cost_usd == 0.5
        assert b.max_tokens == 1000


class TestRuntimeConfig:
    def test_defaults(self):
        c = RuntimeConfig()
        assert c.default_provider == "deepseek"
        assert c.max_turns == 20

    def test_custom(self):
        c = RuntimeConfig(default_provider="openai", max_turns=5)
        assert c.default_provider == "openai"
        assert c.max_turns == 5


class TestRunResult:
    def test_defaults(self):
        r = RunResult()
        assert not r.success
        assert r.steps == 0
        assert r.tokens_used == 0
        assert r.cost_usd == 0.0
        assert r.run_id == ""

    def test_populated(self):
        r = RunResult(output="hello", success=True, steps=3, tokens_used=100)
        assert r.output == "hello"
        assert r.success
        assert r.steps == 3
        assert r.tokens_used == 100


class TestTeamResult:
    def test_defaults(self):
        t = TeamResult()
        assert not t.success
        assert t.rounds == 0
        assert t.agent_outputs == {}
        assert t.conversation_log == []
        assert t.total_tokens == 0

    def test_conversation_log(self):
        t = TeamResult(
            success=True,
            rounds=2,
            conversation_log=[{"round": 1, "agent": "a", "content": "hi"}],
        )
        assert len(t.conversation_log) == 1
        assert t.success
        assert t.rounds == 2


class TestAgentConfig:
    def test_basic(self):
        a = AgentConfig(name="writer", prompt="You write")
        assert a.name == "writer"
        assert a.prompt == "You write"
        assert a.model is None
        assert a.provider is None

    def test_with_overrides(self):
        a = AgentConfig(name="x", prompt="y", provider="openai", model="gpt-4o")
        assert a.provider == "openai"
        assert a.model == "gpt-4o"


class TestRuntimeInit:
    def test_no_providers(self):
        rt = Runtime()
        assert rt.providers == []

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            Runtime(providers={"nonexistent": "key"})

    def test_ollama_provider(self):
        """Ollama provider with matching default_provider config."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )
        assert "ollama" in rt.providers

    def test_default_provider_mismatch_raises(self):
        """When default_provider is not in the registered providers, set_default raises."""
        with pytest.raises(KeyError, match="not registered"):
            Runtime(providers={"ollama": ""})

    def test_tools_registered(self):
        from arcana.sdk import tool

        @tool()
        def my_tool(x: str) -> str:
            return x

        rt = Runtime(tools=[my_tool])
        assert "my_tool" in rt.tools

    def test_no_tools_empty_list(self):
        rt = Runtime()
        assert rt.tools == []

    def test_budget_config(self):
        rt = Runtime(budget=Budget(max_cost_usd=0.1))
        assert rt._budget_policy.max_cost_usd == 0.1

    def test_default_budget(self):
        rt = Runtime()
        assert rt._budget_policy.max_cost_usd == 10.0
        assert rt._budget_policy.max_tokens == 500_000


class TestRuntimeRun:
    @pytest.mark.asyncio
    async def test_run_requires_provider(self):
        """No providers registered -> _resolve_model_config should fail."""
        rt = Runtime()
        with pytest.raises(ValueError):
            await rt.run("test")

    @pytest.mark.asyncio
    async def test_run_with_mock_provider(self):
        """Mock the ConversationAgent to test run() flow without API."""
        from arcana.contracts.state import AgentState, ExecutionStatus

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        mock_state = AgentState(
            run_id="test",
            status=ExecutionStatus.COMPLETED,
            working_memory={"answer": "42"},
            tokens_used=10,
        )

        with patch("arcana.runtime_core.ConversationAgent", create=True):
            # Patch the import inside Session.run
            with patch("arcana.runtime.conversation.ConversationAgent") as MockAgent2:
                instance = MockAgent2.return_value
                instance.run = AsyncMock(return_value=mock_state)

                result = await rt.run("test")
                assert result.success
                assert result.output == "42"


class TestRuntimeStream:
    @pytest.mark.asyncio
    async def test_stream_rejects_non_conversation_engine(self):
        """Streaming only supports conversation engine."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )
        with pytest.raises(ValueError, match="Streaming only supported"):
            async for _ in rt.stream("test", engine="adaptive"):
                pass


class TestRuntimeTeam:
    @pytest.mark.asyncio
    async def test_team_with_mock(self):
        from arcana.contracts.llm import LLMResponse, TokenUsage

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        mock_response = LLMResponse(
            content="Great work! [DONE]",
            usage=TokenUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
            model="test",
            finish_reason="stop",
        )
        rt._gateway.generate = AsyncMock(return_value=mock_response)

        result = await rt.team(
            goal="test",
            agents=[
                AgentConfig(name="a", prompt="agent a"),
                AgentConfig(name="b", prompt="agent b"),
            ],
            max_rounds=1,
        )
        assert result.success
        assert result.rounds == 1

    @pytest.mark.asyncio
    async def test_team_max_rounds_reached(self):
        """When no agent says [DONE], max_rounds is reached and success=False."""
        from arcana.contracts.llm import LLMResponse, TokenUsage

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        mock_response = LLMResponse(
            content="Still working on it...",
            usage=TokenUsage(
                prompt_tokens=5, completion_tokens=10, total_tokens=15
            ),
            model="test",
            finish_reason="stop",
        )
        rt._gateway.generate = AsyncMock(return_value=mock_response)

        result = await rt.team(
            goal="test",
            agents=[AgentConfig(name="a", prompt="agent a")],
            max_rounds=2,
        )
        assert not result.success
        assert result.rounds == 2
        assert len(result.conversation_log) == 2


class TestRuntimeClose:
    @pytest.mark.asyncio
    async def test_close_without_mcp(self):
        rt = Runtime()
        await rt.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_with_mcp_client(self):
        rt = Runtime()
        mock_mcp = AsyncMock()
        rt._mcp_client = mock_mcp
        await rt.close()
        mock_mcp.disconnect_all.assert_awaited_once()
        assert rt._mcp_client is None


class TestMakeLLMNode:
    def test_returns_llm_node_with_gateway(self):
        """make_llm_node returns an LLMNode wired to runtime's gateway."""
        from arcana.graph.nodes.llm_node import LLMNode

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )
        node = rt.make_llm_node()
        assert isinstance(node, LLMNode)
        assert node._gateway is rt._gateway

    def test_passes_model_config(self):
        """make_llm_node resolves model config from runtime."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )
        node = rt.make_llm_node()
        assert node._model_config is not None
        assert node._model_config.provider == "ollama"

    def test_passes_system_prompt(self):
        """make_llm_node forwards system_prompt to the node."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )
        node = rt.make_llm_node(system_prompt="You are helpful.")
        assert node._system_prompt == "You are helpful."

    def test_no_system_prompt_by_default(self):
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )
        node = rt.make_llm_node()
        assert node._system_prompt is None


class TestMakeToolNode:
    def test_returns_tool_node_with_gateway(self):
        """make_tool_node returns a ToolNode wired to runtime's tool gateway."""
        from arcana.graph.nodes.tool_node import ToolNode
        from arcana.sdk import tool

        @tool()
        def my_tool(x: str) -> str:
            return x

        rt = Runtime(tools=[my_tool])
        node = rt.make_tool_node()
        assert isinstance(node, ToolNode)
        assert node._tool_gateway is rt._tool_gateway

    def test_raises_if_no_tools(self):
        """make_tool_node raises ValueError when no tools are registered."""
        rt = Runtime()
        with pytest.raises(ValueError, match="No tools registered"):
            rt.make_tool_node()


class TestSession:
    def test_session_has_run_id(self):
        rt = Runtime()
        session = Session(runtime=rt)
        assert session.run_id  # Should be a non-empty UUID string
        assert len(session.run_id) > 0

    def test_session_budget_defaults(self):
        rt = Runtime()
        session = Session(runtime=rt)
        assert session.budget.max_cost_usd == 10.0

    def test_session_custom_budget(self):
        rt = Runtime()
        session = Session(runtime=rt, budget=Budget(max_cost_usd=0.5))
        assert session.budget.max_cost_usd == 0.5
