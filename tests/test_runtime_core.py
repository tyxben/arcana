"""Tests for Runtime, Session, Budget, and SDK entry points."""

from unittest.mock import AsyncMock, patch

import pytest

from arcana.runtime_core import (
    AgentConfig,
    Budget,
    ChainResult,
    ChainStep,
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

    def test_run_result_has_parsed_field(self):
        """RunResult exposes a parsed field (None by default)."""
        r = RunResult()
        assert r.parsed is None

        r2 = RunResult(output="raw text", parsed={"name": "Alice"}, success=True)
        assert r2.parsed == {"name": "Alice"}
        assert r2.output == "raw text"


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


class TestRuntimeChain:
    @pytest.mark.asyncio
    async def test_chain_sequential_execution(self):
        """Chain runs steps sequentially, passing output as context."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        call_count = 0
        calls: list[dict] = []

        async def mock_run(goal, **kwargs):
            nonlocal call_count
            call_count += 1
            calls.append({"goal": goal, **kwargs})
            return RunResult(
                output=f"step-{call_count}-output",
                success=True,
                steps=1,
                tokens_used=50,
                cost_usd=0.001,
                run_id=f"run-{call_count}",
            )

        rt.run = mock_run  # type: ignore[assignment]

        result = await rt.chain(
            steps=[
                ChainStep(name="filter", goal="Filter items"),
                ChainStep(name="classify", goal="Classify items"),
            ],
            input="raw data",
        )

        assert result.success
        assert call_count == 2
        assert result.steps["filter"] == "step-1-output"
        assert result.steps["classify"] == "step-2-output"
        assert result.output == "step-2-output"
        assert result.total_tokens == 100
        # First step gets input as context
        assert calls[0]["context"] == "raw data"
        # Second step gets first step's output as context
        assert calls[1]["context"] == "step-1-output"

    @pytest.mark.asyncio
    async def test_chain_stops_on_failure(self):
        """Chain stops and returns failure if a step fails."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        call_count = 0

        async def mock_run(goal, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return RunResult(output="error", success=False, run_id="fail")
            return RunResult(
                output="ok", success=True, steps=1,
                tokens_used=10, cost_usd=0.001, run_id=f"run-{call_count}",
            )

        rt.run = mock_run  # type: ignore[assignment]

        result = await rt.chain(
            steps=[
                ChainStep(name="a", goal="Step A"),
                ChainStep(name="b", goal="Step B"),
                ChainStep(name="c", goal="Step C"),
            ],
        )

        assert not result.success
        assert call_count == 2  # Step C never ran
        assert "a" in result.steps
        assert "b" in result.steps
        assert "c" not in result.steps

    @pytest.mark.asyncio
    async def test_chain_passes_system_and_response_format(self):
        """Chain passes per-step system and response_format to run()."""
        from pydantic import BaseModel

        class Output(BaseModel):
            value: str

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        captured_kwargs: list[dict] = []

        async def mock_run(goal, **kwargs):
            captured_kwargs.append(kwargs)
            return RunResult(
                output=Output(value="test") if kwargs.get("response_format") else "plain",
                success=True, steps=1, tokens_used=10, cost_usd=0.001, run_id="r",
            )

        rt.run = mock_run  # type: ignore[assignment]

        await rt.chain(
            steps=[
                ChainStep(
                    name="step1",
                    goal="Do something",
                    system="You are a filter",
                    response_format=Output,
                ),
            ],
        )

        assert captured_kwargs[0]["system"] == "You are a filter"
        assert captured_kwargs[0]["response_format"] is Output

    @pytest.mark.asyncio
    async def test_chain_empty_steps(self):
        """Chain with no steps returns empty result."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        result = await rt.chain(steps=[])
        assert result.success
        assert result.output is None
        assert result.steps == {}


class TestRuntimeRunContext:
    @pytest.mark.asyncio
    async def test_run_context_dict_injected_into_goal(self):
        """Context dict is serialized and injected into goal."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        captured_goal = None

        original_create_session = rt._create_session

        def mock_create_session(**kwargs):
            return original_create_session(**kwargs)

        # Mock at the session level to capture the goal
        from arcana.contracts.llm import LLMResponse, TokenUsage

        mock_response = LLMResponse(
            content="done",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model="test",
            finish_reason="stop",
        )
        rt._gateway.generate = AsyncMock(return_value=mock_response)
        rt._gateway.stream = AsyncMock(side_effect=AttributeError)

        result = await rt.run(
            "Classify these items",
            context={"batch_insight": "all tech news", "count": 42},
        )

        # Verify context was injected by checking the gateway call
        call_args = rt._gateway.generate.call_args
        if call_args:
            messages = call_args[1]["request"].messages if "request" in call_args[1] else call_args[0][0].messages
            user_msg = [m for m in messages if m.role.value == "user"][0]
            assert "<context>" in user_msg.content
            assert "all tech news" in user_msg.content

    @pytest.mark.asyncio
    async def test_run_context_string_injected_into_goal(self):
        """Context string is injected as-is."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        from arcana.contracts.llm import LLMResponse, TokenUsage

        mock_response = LLMResponse(
            content="done",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model="test",
            finish_reason="stop",
        )
        rt._gateway.generate = AsyncMock(return_value=mock_response)
        rt._gateway.stream = AsyncMock(side_effect=AttributeError)

        await rt.run(
            "Classify items",
            context="Previous step found all items are tech news",
        )

        call_args = rt._gateway.generate.call_args
        if call_args:
            messages = call_args[1]["request"].messages if "request" in call_args[1] else call_args[0][0].messages
            user_msg = [m for m in messages if m.role.value == "user"][0]
            assert "<context>" in user_msg.content
            assert "Previous step found all items are tech news" in user_msg.content

    @pytest.mark.asyncio
    async def test_run_no_context_no_injection(self):
        """Without context, goal is unchanged."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        from arcana.contracts.llm import LLMResponse, TokenUsage

        mock_response = LLMResponse(
            content="done",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model="test",
            finish_reason="stop",
        )
        rt._gateway.generate = AsyncMock(return_value=mock_response)
        rt._gateway.stream = AsyncMock(side_effect=AttributeError)

        await rt.run("Classify items")

        call_args = rt._gateway.generate.call_args
        if call_args:
            messages = call_args[1]["request"].messages if "request" in call_args[1] else call_args[0][0].messages
            user_msg = [m for m in messages if m.role.value == "user"][0]
            assert "<context>" not in user_msg.content


class TestRuntimeRunSystem:
    @pytest.mark.asyncio
    async def test_run_system_passed_to_agent(self):
        """run(system=...) is passed through to ConversationAgent."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        from arcana.contracts.llm import LLMResponse, TokenUsage

        mock_response = LLMResponse(
            content="done",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model="test",
            finish_reason="stop",
        )
        rt._gateway.generate = AsyncMock(return_value=mock_response)
        rt._gateway.stream = AsyncMock(side_effect=AttributeError)

        await rt.run("Hello", system="You are a pirate")

        call_args = rt._gateway.generate.call_args
        if call_args:
            messages = call_args[1]["request"].messages if "request" in call_args[1] else call_args[0][0].messages
            system_msg = [m for m in messages if m.role.value == "system"][0]
            assert "You are a pirate" in system_msg.content

    @pytest.mark.asyncio
    async def test_run_config_system_prompt_used_as_fallback(self):
        """RuntimeConfig.system_prompt is used when run(system=) is not set."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(
                default_provider="ollama",
                system_prompt="You are a butler",
            ),
        )

        from arcana.contracts.llm import LLMResponse, TokenUsage

        mock_response = LLMResponse(
            content="done",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model="test",
            finish_reason="stop",
        )
        rt._gateway.generate = AsyncMock(return_value=mock_response)
        rt._gateway.stream = AsyncMock(side_effect=AttributeError)

        await rt.run("Hello")

        call_args = rt._gateway.generate.call_args
        if call_args:
            messages = call_args[1]["request"].messages if "request" in call_args[1] else call_args[0][0].messages
            system_msg = [m for m in messages if m.role.value == "system"][0]
            assert "You are a butler" in system_msg.content

    @pytest.mark.asyncio
    async def test_run_system_overrides_config(self):
        """run(system=) overrides RuntimeConfig.system_prompt."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(
                default_provider="ollama",
                system_prompt="You are a butler",
            ),
        )

        from arcana.contracts.llm import LLMResponse, TokenUsage

        mock_response = LLMResponse(
            content="done",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model="test",
            finish_reason="stop",
        )
        rt._gateway.generate = AsyncMock(return_value=mock_response)
        rt._gateway.stream = AsyncMock(side_effect=AttributeError)

        await rt.run("Hello", system="You are a pirate")

        call_args = rt._gateway.generate.call_args
        if call_args:
            messages = call_args[1]["request"].messages if "request" in call_args[1] else call_args[0][0].messages
            system_msg = [m for m in messages if m.role.value == "system"][0]
            assert "You are a pirate" in system_msg.content
            assert "butler" not in system_msg.content


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


class TestRuntimeBudgetQuery:
    def test_budget_remaining_initial(self):
        """Fresh runtime returns full budget."""
        rt = Runtime(budget=Budget(max_cost_usd=5.0, max_tokens=100_000))
        assert rt.budget_remaining_usd == 5.0
        assert rt.budget_used_usd == 0.0
        assert rt.tokens_remaining == 100_000
        assert rt.tokens_used == 0

    def test_budget_remaining_default(self):
        """Default budget returns default values."""
        rt = Runtime()
        assert rt.budget_remaining_usd == 10.0
        assert rt.tokens_remaining == 500_000

    @pytest.mark.asyncio
    async def test_budget_tracks_across_runs(self):
        """After a run, budget_used_usd reflects usage."""
        from arcana.contracts.state import AgentState, ExecutionStatus

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
            budget=Budget(max_cost_usd=5.0, max_tokens=100_000),
        )

        mock_state = AgentState(
            run_id="test-1",
            status=ExecutionStatus.COMPLETED,
            working_memory={"answer": "hello"},
            tokens_used=150,
            cost_usd=0.5,
        )

        with patch("arcana.runtime.conversation.ConversationAgent") as MockAgent:
            instance = MockAgent.return_value
            instance.run = AsyncMock(return_value=mock_state)
            await rt.run("first task")

        assert rt.budget_used_usd == 0.5
        assert rt.budget_remaining_usd == 4.5
        assert rt.tokens_used == 150
        assert rt.tokens_remaining == 99_850

    @pytest.mark.asyncio
    async def test_tokens_used_accumulates(self):
        """Multiple runs accumulate token counts."""
        from arcana.contracts.state import AgentState, ExecutionStatus

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
            budget=Budget(max_cost_usd=10.0, max_tokens=1000),
        )

        states = [
            AgentState(
                run_id=f"test-{i}",
                status=ExecutionStatus.COMPLETED,
                working_memory={"answer": f"result-{i}"},
                tokens_used=tokens,
                cost_usd=cost,
            )
            for i, (tokens, cost) in enumerate([(100, 0.1), (200, 0.3), (50, 0.05)])
        ]

        with patch("arcana.runtime.conversation.ConversationAgent") as MockAgent:
            for state in states:
                instance = MockAgent.return_value
                instance.run = AsyncMock(return_value=state)
                await rt.run(f"task {state.run_id}")

        assert rt.tokens_used == 350
        assert rt.budget_used_usd == pytest.approx(0.45)
        assert rt.tokens_remaining == 650
        assert rt.budget_remaining_usd == pytest.approx(9.55)


class TestRuntimeFallbackOrder:
    def test_single_provider(self):
        rt = Runtime(
            providers={"deepseek": "sk-xxx"},
            config=RuntimeConfig(default_provider="deepseek"),
        )
        assert rt.fallback_order == ["deepseek"]

    def test_multi_provider_auto_chain(self):
        rt = Runtime(
            providers={"deepseek": "sk-xxx", "openai": "sk-yyy"},
            config=RuntimeConfig(default_provider="deepseek"),
        )
        assert rt.fallback_order == ["deepseek", "openai"]

    def test_no_providers(self):
        rt = Runtime()
        assert rt.fallback_order == []
