"""Tests for Runtime, Session, Budget, and SDK entry points."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arcana.runtime_core import (
    AgentConfig,
    Budget,
    BudgetScope,
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


class TestRuntimeTeamSession:
    @pytest.mark.asyncio
    async def test_team_session_independent_context(self):
        """Session mode: each agent gets independent conversation history."""
        from arcana.contracts.llm import LLMResponse, TokenUsage

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        call_count = 0
        messages_seen: list[list] = []

        async def mock_generate(request, config, trace_ctx=None):
            nonlocal call_count
            call_count += 1
            # Record what messages each agent saw
            messages_seen.append([
                (m.role.value if hasattr(m.role, "value") else m.role, m.content[:80])
                for m in request.messages
            ])
            return LLMResponse(
                content=f"Response from agent {call_count} [DONE]",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                model="test",
                finish_reason="stop",
            )

        rt._gateway.generate = mock_generate

        result = await rt.team(
            goal="test goal",
            agents=[
                AgentConfig(name="a", prompt="agent a"),
                AgentConfig(name="b", prompt="agent b"),
            ],
            mode="session",
            max_rounds=1,
        )

        assert result.success
        # Agent A saw: system + user(goal) — no messages from B yet
        a_msgs = messages_seen[0]
        assert a_msgs[0][0] == "system"
        assert "agent a" in a_msgs[0][1]
        assert a_msgs[1][0] == "user"
        assert "test goal" in a_msgs[1][1]
        assert len(a_msgs) == 2  # Only system + goal, no shared history

    @pytest.mark.asyncio
    async def test_team_session_message_delivery(self):
        """Session mode: agent A's response is delivered to agent B's inbox."""
        from arcana.contracts.llm import LLMResponse, TokenUsage

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        call_count = 0
        messages_seen: list[list] = []

        async def mock_generate(request, config, trace_ctx=None):
            nonlocal call_count
            call_count += 1
            messages_seen.append([
                (m.role.value if hasattr(m.role, "value") else m.role, m.content)
                for m in request.messages
            ])
            if call_count >= 3:
                return LLMResponse(
                    content="Done [DONE]",
                    usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                    model="test", finish_reason="stop",
                )
            return LLMResponse(
                content=f"Hello from call {call_count}",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
                model="test", finish_reason="stop",
            )

        rt._gateway.generate = mock_generate

        await rt.team(
            goal="test",
            agents=[
                AgentConfig(name="alpha", prompt="agent alpha"),
                AgentConfig(name="beta", prompt="agent beta"),
            ],
            mode="session",
            max_rounds=3,
        )

        # Round 1: alpha speaks (call 1), beta speaks (call 2)
        # Round 2: alpha speaks (call 3) — should see beta's round-1 message
        # Call 3 = alpha's second turn
        if len(messages_seen) >= 3:
            alpha_r2_msgs = messages_seen[2]
            # Should have: system, user(goal), assistant(alpha r1), user(from beta)
            contents = [m[1] for m in alpha_r2_msgs]
            assert any("Message from beta" in c for c in contents)

    @pytest.mark.asyncio
    async def test_team_session_done_detection(self):
        """Session mode: [DONE] ends collaboration."""
        from arcana.contracts.llm import LLMResponse, TokenUsage

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        mock_response = LLMResponse(
            content="All done [DONE]",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            model="test", finish_reason="stop",
        )
        rt._gateway.generate = AsyncMock(return_value=mock_response)

        result = await rt.team(
            goal="test",
            agents=[AgentConfig(name="a", prompt="a"), AgentConfig(name="b", prompt="b")],
            mode="session",
            max_rounds=5,
        )

        assert result.success
        assert result.rounds == 1

    @pytest.mark.asyncio
    async def test_team_session_max_rounds(self):
        """Session mode: max_rounds reached returns success=False."""
        from arcana.contracts.llm import LLMResponse, TokenUsage

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        mock_response = LLMResponse(
            content="Still working...",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model="test", finish_reason="stop",
        )
        rt._gateway.generate = AsyncMock(return_value=mock_response)

        result = await rt.team(
            goal="test",
            agents=[AgentConfig(name="a", prompt="a")],
            mode="session",
            max_rounds=2,
        )

        assert not result.success
        assert result.rounds == 2

    @pytest.mark.asyncio
    async def test_team_mode_invalid_raises(self):
        """Invalid mode raises ValueError."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        with pytest.raises(ValueError, match="Invalid team mode"):
            await rt.team("test", [AgentConfig(name="a", prompt="a")], mode="invalid")

    @pytest.mark.asyncio
    async def test_team_mode_shared_unchanged(self):
        """Default mode='shared' preserves existing behavior."""
        from arcana.contracts.llm import LLMResponse, TokenUsage

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        mock_response = LLMResponse(
            content="Great work! [DONE]",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            model="test", finish_reason="stop",
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

    @pytest.mark.asyncio
    async def test_chain_parallel_steps(self):
        """Parallel steps run concurrently and both receive same context."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        calls: list[dict] = []

        async def mock_run(goal, **kwargs):
            calls.append({"goal": goal, **kwargs})
            # Return output keyed by goal for identification
            if "Classify" in goal:
                return RunResult(
                    output="classified", success=True, steps=1,
                    tokens_used=30, cost_usd=0.001, run_id="classify",
                )
            else:
                return RunResult(
                    output="analyzed", success=True, steps=1,
                    tokens_used=20, cost_usd=0.002, run_id="analyze",
                )

        rt.run = mock_run  # type: ignore[assignment]

        result = await rt.chain(
            steps=[
                [
                    ChainStep(name="classify", goal="Classify items"),
                    ChainStep(name="analyze", goal="Analyze items"),
                ],
            ],
            input="raw data",
        )

        assert result.success
        assert len(calls) == 2
        assert result.steps["classify"] == "classified"
        assert result.steps["analyze"] == "analyzed"
        # Both parallel steps receive the same input context
        assert calls[0]["context"] == "raw data"
        assert calls[1]["context"] == "raw data"
        assert result.total_tokens == 50
        assert result.total_cost_usd == pytest.approx(0.003)
        # Output is dict when last step is a parallel group
        assert result.output == {"classify": "classified", "analyze": "analyzed"}

    @pytest.mark.asyncio
    async def test_chain_parallel_then_sequential(self):
        """[A, B] -> C: C gets merged context from A and B."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        calls: list[dict] = []
        call_count = 0

        async def mock_run(goal, **kwargs):
            nonlocal call_count
            call_count += 1
            calls.append({"goal": goal, **kwargs})
            if "Classify" in goal:
                return RunResult(
                    output="classified-result", success=True, steps=1,
                    tokens_used=10, cost_usd=0.001, run_id="r1",
                )
            elif "Analyze" in goal:
                return RunResult(
                    output="analyzed-result", success=True, steps=1,
                    tokens_used=10, cost_usd=0.001, run_id="r2",
                )
            else:
                return RunResult(
                    output="integrated", success=True, steps=1,
                    tokens_used=10, cost_usd=0.001, run_id="r3",
                )

        rt.run = mock_run  # type: ignore[assignment]

        result = await rt.chain(
            steps=[
                [
                    ChainStep(name="classify", goal="Classify items"),
                    ChainStep(name="analyze", goal="Analyze items"),
                ],
                ChainStep(name="integrate", goal="Integrate results"),
            ],
            input="raw data",
        )

        assert result.success
        assert call_count == 3
        # Integrate step should receive merged context from both parallel steps
        integrate_call = [c for c in calls if "Integrate" in c["goal"]][0]
        ctx = integrate_call["context"]
        assert "[classify]:" in ctx
        assert "classified-result" in ctx
        assert "[analyze]:" in ctx
        assert "analyzed-result" in ctx
        assert result.output == "integrated"

    @pytest.mark.asyncio
    async def test_chain_sequential_then_parallel(self):
        """A -> [B, C]: B and C both get A's output as context."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        calls: list[dict] = []

        async def mock_run(goal, **kwargs):
            calls.append({"goal": goal, **kwargs})
            if "Filter" in goal:
                return RunResult(
                    output="filtered-data", success=True, steps=1,
                    tokens_used=20, cost_usd=0.001, run_id="r1",
                )
            elif "Classify" in goal:
                return RunResult(
                    output="classified", success=True, steps=1,
                    tokens_used=15, cost_usd=0.001, run_id="r2",
                )
            else:
                return RunResult(
                    output="analyzed", success=True, steps=1,
                    tokens_used=15, cost_usd=0.001, run_id="r3",
                )

        rt.run = mock_run  # type: ignore[assignment]

        result = await rt.chain(
            steps=[
                ChainStep(name="filter", goal="Filter items"),
                [
                    ChainStep(name="classify", goal="Classify items"),
                    ChainStep(name="analyze", goal="Analyze items"),
                ],
            ],
            input="raw data",
        )

        assert result.success
        assert result.steps["filter"] == "filtered-data"
        assert result.steps["classify"] == "classified"
        assert result.steps["analyze"] == "analyzed"
        # Both parallel steps should receive filter's output as context
        classify_call = [c for c in calls if "Classify" in c["goal"]][0]
        analyze_call = [c for c in calls if "Analyze" in c["goal"]][0]
        assert classify_call["context"] == "filtered-data"
        assert analyze_call["context"] == "filtered-data"
        # Output is dict since last step is parallel
        assert result.output == {"classify": "classified", "analyze": "analyzed"}

    @pytest.mark.asyncio
    async def test_chain_parallel_failure_stops_chain(self):
        """If one parallel step fails, the chain returns failure."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        call_count = 0

        async def mock_run(goal, **kwargs):
            nonlocal call_count
            call_count += 1
            if "Classify" in goal:
                return RunResult(
                    output="classified", success=True, steps=1,
                    tokens_used=10, cost_usd=0.001, run_id="r1",
                )
            elif "Analyze" in goal:
                return RunResult(
                    output="error", success=False, steps=1,
                    tokens_used=5, cost_usd=0.0005, run_id="r2",
                )
            else:
                # This should never be reached
                return RunResult(
                    output="integrated", success=True, steps=1,
                    tokens_used=10, cost_usd=0.001, run_id="r3",
                )

        rt.run = mock_run  # type: ignore[assignment]

        result = await rt.chain(
            steps=[
                [
                    ChainStep(name="classify", goal="Classify items"),
                    ChainStep(name="analyze", goal="Analyze items"),
                ],
                ChainStep(name="integrate", goal="Integrate results"),
            ],
        )

        assert not result.success
        # Both parallel steps ran (gather runs all)
        assert "classify" in result.steps
        assert "analyze" in result.steps
        # Integrate never ran
        assert "integrate" not in result.steps

    @pytest.mark.asyncio
    async def test_chain_mixed_sequential_and_parallel(self):
        """Full DAG: A -> [B, C] -> D."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        calls: list[dict] = []
        call_count = 0

        async def mock_run(goal, **kwargs):
            nonlocal call_count
            call_count += 1
            calls.append({"goal": goal, **kwargs})
            if "Filter" in goal:
                return RunResult(
                    output="filtered", success=True, steps=1,
                    tokens_used=20, cost_usd=0.001, run_id="r1",
                )
            elif "Classify" in goal:
                return RunResult(
                    output="classified", success=True, steps=1,
                    tokens_used=15, cost_usd=0.001, run_id="r2",
                )
            elif "Analyze" in goal:
                return RunResult(
                    output="analyzed", success=True, steps=1,
                    tokens_used=15, cost_usd=0.001, run_id="r3",
                )
            else:
                return RunResult(
                    output="integrated", success=True, steps=1,
                    tokens_used=25, cost_usd=0.002, run_id="r4",
                )

        rt.run = mock_run  # type: ignore[assignment]

        result = await rt.chain(
            steps=[
                ChainStep(name="filter", goal="Filter items", system="You are a filter"),
                [
                    ChainStep(name="classify", goal="Classify items", system="You are a classifier"),
                    ChainStep(name="analyze", goal="Analyze items", system="You are an analyst"),
                ],
                ChainStep(name="integrate", goal="Integrate results", system="You are an integrator"),
            ],
            input="raw data",
        )

        assert result.success
        assert call_count == 4
        assert result.steps["filter"] == "filtered"
        assert result.steps["classify"] == "classified"
        assert result.steps["analyze"] == "analyzed"
        assert result.steps["integrate"] == "integrated"
        assert result.output == "integrated"
        assert result.total_tokens == 75
        assert result.total_cost_usd == pytest.approx(0.005)

        # Verify context flow:
        # 1. Filter gets input
        filter_call = [c for c in calls if "Filter" in c["goal"]][0]
        assert filter_call["context"] == "raw data"
        # 2. Classify and Analyze both get filter's output
        classify_call = [c for c in calls if "Classify" in c["goal"]][0]
        analyze_call = [c for c in calls if "Analyze" in c["goal"]][0]
        assert classify_call["context"] == "filtered"
        assert analyze_call["context"] == "filtered"
        # 3. Integrate gets merged context from classify + analyze
        integrate_call = [c for c in calls if "Integrate" in c["goal"]][0]
        ctx = integrate_call["context"]
        assert "[classify]:" in ctx
        assert "classified" in ctx
        assert "[analyze]:" in ctx
        assert "analyzed" in ctx


class TestRuntimeRunContext:
    @pytest.mark.asyncio
    async def test_run_context_dict_injected_into_goal(self):
        """Context dict is serialized and injected into goal."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )


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

        await rt.run(
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


class TestExtendedProviderConfig:
    """Tests for extended provider config format (dict values)."""

    def test_extended_provider_config_dict(self):
        """Verify dict config sets api_key and overrides default_model."""
        rt = Runtime(
            providers={"deepseek": {"api_key": "sk-xxx", "model": "deepseek-reasoner"}},
            config=RuntimeConfig(default_provider="deepseek"),
        )
        assert "deepseek" in rt.providers
        # Verify the model was overridden on the provider instance
        provider = rt._gateway.get("deepseek")
        assert provider.default_model == "deepseek-reasoner"

    def test_extended_provider_config_string(self):
        """Verify string config still works (backward compat)."""
        rt = Runtime(
            providers={"deepseek": "sk-xxx"},
            config=RuntimeConfig(default_provider="deepseek"),
        )
        assert "deepseek" in rt.providers
        # Default model should be the factory default
        provider = rt._gateway.get("deepseek")
        assert provider.default_model == "deepseek-chat"

    def test_extended_provider_config_dict_no_model(self):
        """Dict config without model key keeps the factory default."""
        rt = Runtime(
            providers={"deepseek": {"api_key": "sk-xxx"}},
            config=RuntimeConfig(default_provider="deepseek"),
        )
        provider = rt._gateway.get("deepseek")
        assert provider.default_model == "deepseek-chat"

    def test_extended_provider_config_mixed(self):
        """Mix of string and dict configs in the same providers dict."""
        rt = Runtime(
            providers={
                "deepseek": {"api_key": "sk-xxx", "model": "deepseek-reasoner"},
                "openai": "sk-yyy",
            },
            config=RuntimeConfig(default_provider="deepseek"),
        )
        ds_provider = rt._gateway.get("deepseek")
        assert ds_provider.default_model == "deepseek-reasoner"
        oai_provider = rt._gateway.get("openai")
        assert oai_provider.default_model == "gpt-4o-mini"


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


class TestRunProviderModelOverride:
    """Tests for per-run provider/model selection via run(provider=, model=)."""

    @pytest.mark.asyncio
    async def test_run_with_provider_override(self):
        """run(provider='openai') passes the override to Session."""
        rt = Runtime(
            providers={"ollama": "", "openai": "sk-test"},
            config=RuntimeConfig(default_provider="ollama"),
        )

        captured_sessions: list[Session] = []
        original_create = rt._create_session

        def spy_create(**kwargs):
            session = original_create(**kwargs)
            captured_sessions.append(session)
            return session

        async def fake_run(self_session, goal: str):
            return MagicMock(
                output="result",
                success=True,
                tokens_used=10,
                cost_usd=0.001,
                run_id="test",
            )

        with (
            patch.object(rt, "_create_session", side_effect=spy_create),
            patch.object(Session, "run", fake_run),
        ):
            await rt.run("test goal", provider="openai")

        assert len(captured_sessions) == 1
        assert captured_sessions[0]._provider_override == "openai"

    @pytest.mark.asyncio
    async def test_run_with_model_override(self):
        """run(model='custom-model') passes the override to Session."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        captured_sessions: list[Session] = []
        original_create = rt._create_session

        def spy_create(**kwargs):
            session = original_create(**kwargs)
            captured_sessions.append(session)
            return session

        async def fake_run(self_session, goal: str):
            return MagicMock(
                output="result",
                success=True,
                tokens_used=10,
                cost_usd=0.001,
                run_id="test",
            )

        with (
            patch.object(rt, "_create_session", side_effect=spy_create),
            patch.object(Session, "run", fake_run),
        ):
            await rt.run("test goal", model="custom-model-v2")

        assert len(captured_sessions) == 1
        assert captured_sessions[0]._model_override == "custom-model-v2"

    @pytest.mark.asyncio
    async def test_run_with_both_provider_and_model(self):
        """run(provider='openai', model='gpt-4o') overrides both."""
        rt = Runtime(
            providers={"ollama": "", "openai": "sk-test"},
            config=RuntimeConfig(default_provider="ollama"),
        )

        captured_sessions: list[Session] = []
        original_create = rt._create_session

        def spy_create(**kwargs):
            session = original_create(**kwargs)
            captured_sessions.append(session)
            return session

        async def fake_run(self_session, goal: str):
            return MagicMock(
                output="result",
                success=True,
                tokens_used=10,
                cost_usd=0.001,
                run_id="test",
            )

        with (
            patch.object(rt, "_create_session", side_effect=spy_create),
            patch.object(Session, "run", fake_run),
        ):
            await rt.run("test goal", provider="openai", model="gpt-4o-mini")

        assert len(captured_sessions) == 1
        assert captured_sessions[0]._provider_override == "openai"
        assert captured_sessions[0]._model_override == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_session_builds_custom_model_config_with_overrides(self):
        """Session.run() builds a custom ModelConfig when overrides are set."""
        from arcana.contracts.llm import ModelConfig

        rt = Runtime(
            providers={"ollama": "", "openai": "sk-test"},
            config=RuntimeConfig(default_provider="ollama"),
        )

        session = rt._create_session(provider="openai", model="gpt-4o")

        # Verify the override fields are stored
        assert session._provider_override == "openai"
        assert session._model_override == "gpt-4o"

        # Simulate the model config resolution logic from Session.run()
        resolved_provider = (
            session._provider_override or rt._config.default_provider
        )
        resolved_model_id = (
            session._model_override or rt._config.default_model
        )
        if not resolved_model_id:
            p = rt._gateway.get(resolved_provider)
            if (
                p
                and hasattr(p, "default_model")
                and isinstance(p.default_model, str)
            ):
                resolved_model_id = p.default_model

        mc = ModelConfig(provider=resolved_provider, model_id=resolved_model_id)
        assert mc.provider == "openai"
        assert mc.model_id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_session_without_override_uses_default(self):
        """Session without overrides uses _resolve_model_config()."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        session = rt._create_session()
        assert session._provider_override is None
        assert session._model_override is None


class TestChainStepProviderModel:
    """Tests for provider/model fields on ChainStep."""

    def test_chain_step_accepts_provider_and_model(self):
        """ChainStep accepts provider and model fields."""
        step = ChainStep(
            name="analyze",
            goal="Analyze data",
            provider="openai",
            model="gpt-4o",
        )
        assert step.provider == "openai"
        assert step.model == "gpt-4o"

    def test_chain_step_defaults_none(self):
        """ChainStep provider/model default to None."""
        step = ChainStep(name="step1", goal="Do something")
        assert step.provider is None
        assert step.model is None

    @pytest.mark.asyncio
    async def test_chain_passes_provider_model_to_run(self):
        """chain() passes ChainStep provider/model through to run()."""
        rt = Runtime(
            providers={"ollama": "", "openai": "sk-test"},
            config=RuntimeConfig(default_provider="ollama"),
        )

        call_args: list[dict] = []

        async def fake_run(goal, *, provider=None, model=None, **kwargs):
            call_args.append(
                {"goal": goal, "provider": provider, "model": model}
            )
            return RunResult(
                output="step result",
                success=True,
                steps=1,
                tokens_used=5,
                cost_usd=0.0005,
                run_id="test",
            )

        rt.run = fake_run  # type: ignore[assignment]

        steps = [
            ChainStep(
                name="filter",
                goal="Filter data",
                provider="ollama",
                model="llama3",
            ),
            ChainStep(
                name="analyze",
                goal="Analyze data",
                provider="openai",
                model="gpt-4o",
            ),
        ]
        result = await rt.chain(steps)

        assert result.success
        assert len(call_args) == 2
        assert call_args[0]["provider"] == "ollama"
        assert call_args[0]["model"] == "llama3"
        assert call_args[1]["provider"] == "openai"
        assert call_args[1]["model"] == "gpt-4o"


class TestBudgetScope:
    @pytest.mark.asyncio
    async def test_budget_scope_basic(self):
        """BudgetScope tracks usage across multiple runs."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        call_count = 0

        async def mock_run(goal, **kwargs):
            nonlocal call_count
            call_count += 1
            return RunResult(
                output=f"result-{call_count}",
                success=True,
                steps=1,
                tokens_used=100,
                cost_usd=0.01,
                run_id=f"run-{call_count}",
            )

        rt.run = mock_run  # type: ignore[assignment]

        async with rt.budget_scope(max_cost_usd=0.10) as scoped:
            r1 = await scoped.run("Task 1")
            r2 = await scoped.run("Task 2")

        assert r1.output == "result-1"
        assert r2.output == "result-2"
        assert scoped.budget_used_usd == pytest.approx(0.02)
        assert scoped.tokens_used == 200

    @pytest.mark.asyncio
    async def test_budget_scope_exhausted(self):
        """BudgetScope raises BudgetExceededError when scope budget exceeded."""
        from arcana.gateway.base import BudgetExceededError

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        async def mock_run(goal, **kwargs):
            return RunResult(
                output="done",
                success=True,
                steps=1,
                tokens_used=50,
                cost_usd=0.05,
                run_id="run",
            )

        rt.run = mock_run  # type: ignore[assignment]

        async with rt.budget_scope(max_cost_usd=0.08) as scoped:
            await scoped.run("Task 1")  # costs 0.05
            assert scoped.budget_used_usd == pytest.approx(0.05)

            await scoped.run("Task 2")  # costs 0.05, total = 0.10 > 0.08 but runs before check
            # Now the scope has used 0.10 which exceeds 0.08
            assert scoped.budget_used_usd == pytest.approx(0.10)

            # Third run should raise because scope is already exhausted
            with pytest.raises(BudgetExceededError, match="Scoped budget exhausted"):
                await scoped.run("Task 3")

    @pytest.mark.asyncio
    async def test_budget_scope_token_exhausted(self):
        """BudgetScope raises BudgetExceededError when token budget exceeded."""
        from arcana.gateway.base import BudgetExceededError

        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        async def mock_run(goal, **kwargs):
            return RunResult(
                output="done",
                success=True,
                steps=1,
                tokens_used=600,
                cost_usd=0.001,
                run_id="run",
            )

        rt.run = mock_run  # type: ignore[assignment]

        async with rt.budget_scope(max_tokens=1000) as scoped:
            await scoped.run("Task 1")  # uses 600 tokens
            assert scoped.tokens_used == 600

            await scoped.run("Task 2")  # uses 600 more, total = 1200 > 1000
            assert scoped.tokens_used == 1200

            # Third run should raise
            with pytest.raises(BudgetExceededError, match="Scoped token budget exhausted"):
                await scoped.run("Task 3")

    @pytest.mark.asyncio
    async def test_budget_scope_runtime_also_tracks(self):
        """Runtime._total_cost_usd also accumulates from scoped runs."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
            budget=Budget(max_cost_usd=5.0),
        )

        call_count = 0

        async def mock_run(goal, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate runtime._total_cost_usd accumulation
            rt._total_cost_usd += 0.02
            rt._total_tokens_used += 100
            return RunResult(
                output="done",
                success=True,
                steps=1,
                tokens_used=100,
                cost_usd=0.02,
                run_id=f"run-{call_count}",
            )

        rt.run = mock_run  # type: ignore[assignment]

        assert rt.budget_used_usd == 0.0

        async with rt.budget_scope(max_cost_usd=0.10) as scoped:
            await scoped.run("Task 1")
            await scoped.run("Task 2")

        # Scope tracked its own usage
        assert scoped.budget_used_usd == pytest.approx(0.04)
        assert scoped.tokens_used == 200

        # Runtime also tracked the same usage
        assert rt.budget_used_usd == pytest.approx(0.04)
        assert rt.tokens_used == 200

    @pytest.mark.asyncio
    async def test_budget_scope_properties(self):
        """Verify remaining/used properties of BudgetScope."""
        rt = Runtime()

        scope = BudgetScope(rt, max_cost_usd=1.0, max_tokens=10_000)
        assert scope.budget_used_usd == 0.0
        assert scope.budget_remaining_usd == 1.0
        assert scope.tokens_used == 0
        assert scope.tokens_remaining == 10_000

        # Simulate usage
        scope._cost_used = 0.3
        scope._tokens_used = 4000
        assert scope.budget_used_usd == pytest.approx(0.3)
        assert scope.budget_remaining_usd == pytest.approx(0.7)
        assert scope.tokens_used == 4000
        assert scope.tokens_remaining == 6000

    @pytest.mark.asyncio
    async def test_budget_scope_no_limits(self):
        """BudgetScope with no limits returns None for remaining properties."""
        rt = Runtime()

        scope = BudgetScope(rt)
        assert scope.budget_remaining_usd is None
        assert scope.tokens_remaining is None

    @pytest.mark.asyncio
    async def test_budget_scope_merges_user_budget(self):
        """BudgetScope merges user-provided budget with scope budget (stricter wins)."""
        rt = Runtime(
            providers={"ollama": ""},
            config=RuntimeConfig(default_provider="ollama"),
        )

        captured_kwargs: list[dict] = []

        async def mock_run(goal, **kwargs):
            captured_kwargs.append(kwargs)
            return RunResult(
                output="done",
                success=True,
                steps=1,
                tokens_used=10,
                cost_usd=0.001,
                run_id="run",
            )

        rt.run = mock_run  # type: ignore[assignment]

        async with rt.budget_scope(max_cost_usd=0.50) as scoped:
            # User passes a tighter budget
            await scoped.run("Task", budget=Budget(max_cost_usd=0.01, max_tokens=100))

        # The merged budget should use the stricter (lower) values
        merged = captured_kwargs[0]["budget"]
        assert merged.max_cost_usd == 0.01  # user's 0.01 < scope's 0.50
        assert merged.max_tokens == 100  # user's 100 < scope's 500_000
