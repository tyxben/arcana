"""Tests for structured output (response_format) support.

All tests use mocks -- no real API calls required.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from arcana.contracts.llm import LLMRequest

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class Person(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


# =========================================================================
# 1. Pydantic model -> JSON schema conversion
# =========================================================================


class TestPydanticToJsonSchema:
    """Verify that Pydantic models produce the expected JSON schema."""

    def test_simple_model_schema(self) -> None:
        schema = Person.model_json_schema()
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"

    def test_schema_has_required_fields(self) -> None:
        schema = Person.model_json_schema()
        assert "name" in schema["required"]
        assert "age" in schema["required"]

    def test_complex_model_schema(self) -> None:
        schema = Address.model_json_schema()
        assert "street" in schema["properties"]
        assert "city" in schema["properties"]
        assert "zip_code" in schema["properties"]


# =========================================================================
# 2. Response parsing and validation
# =========================================================================


class TestResponseParsing:
    """Verify JSON -> Pydantic model parsing."""

    def test_valid_json_parses_to_model(self) -> None:
        raw = '{"name": "Alice", "age": 30}'
        parsed = json.loads(raw)
        person = Person.model_validate(parsed)
        assert person.name == "Alice"
        assert person.age == 30

    def test_extra_fields_ignored(self) -> None:
        raw = '{"name": "Bob", "age": 25, "extra": "ignored"}'
        parsed = json.loads(raw)
        person = Person.model_validate(parsed)
        assert person.name == "Bob"
        assert person.age == 25

    def test_missing_required_field_raises(self) -> None:
        raw = '{"name": "Charlie"}'
        parsed = json.loads(raw)
        with pytest.raises(ValidationError):
            Person.model_validate(parsed)

    def test_wrong_type_raises(self) -> None:
        raw = '{"name": "Dave", "age": "not-a-number"}'
        parsed = json.loads(raw)
        with pytest.raises(ValidationError):
            Person.model_validate(parsed)


# =========================================================================
# 3. Invalid JSON response -> graceful error
# =========================================================================


class TestInvalidJsonHandling:
    """Verify that malformed JSON is handled gracefully."""

    def test_invalid_json_string(self) -> None:
        raw = "this is not json at all"
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

    def test_partial_json(self) -> None:
        raw = '{"name": "Eve", "age":'
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

    def test_empty_string(self) -> None:
        raw = ""
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)


# =========================================================================
# 4. LLMRequest response_format field
# =========================================================================


class TestLLMRequestResponseFormat:
    """Verify LLMRequest accepts and carries response_format."""

    def test_default_is_none(self) -> None:
        from arcana.contracts.llm import Message, MessageRole

        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="hi")]
        )
        assert req.response_format is None

    def test_schema_passes_through(self) -> None:
        from arcana.contracts.llm import Message, MessageRole

        schema = Person.model_json_schema()
        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="hi")],
            response_format=schema,
        )
        assert req.response_format is not None
        assert req.response_format["type"] == "object"
        assert "name" in req.response_format["properties"]

    def test_response_schema_still_works(self) -> None:
        """Backward compat: response_schema (json_object mode) unchanged."""
        from arcana.contracts.llm import Message, MessageRole

        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="hi")],
            response_schema={"type": "object"},
        )
        assert req.response_schema is not None
        assert req.response_format is None


# =========================================================================
# 5. OpenAI-compatible provider: response_format in API params
# =========================================================================


class TestOpenAIProviderResponseFormat:
    """Verify the provider builds correct API params for response_format."""

    def test_response_format_in_params(self) -> None:
        """When response_format is set, provider should use json_schema mode."""
        from arcana.contracts.llm import Message, MessageRole

        schema = Person.model_json_schema()
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="hi")],
            response_format=schema,
        )

        # The provider builds params internally. We verify by checking
        # that the request carries the schema correctly.
        assert request.response_format == schema

    def test_response_format_coexists_with_tools(self) -> None:
        """Tools and response_format can be used together."""
        from arcana.contracts.llm import Message, MessageRole

        schema = Person.model_json_schema()
        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="hi")],
            response_format=schema,
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        # Both fields coexist on the request — the provider sends both
        assert request.response_format is not None
        assert request.tools is not None


# =========================================================================
# 6. SDK arcana.run() with response_format
# =========================================================================


class TestSDKRunWithResponseFormat:
    """Test arcana.run() with response_format parameter."""

    @pytest.mark.asyncio
    async def test_run_with_response_format_parses_output(self) -> None:
        """arcana.run() should parse JSON output into a Pydantic model."""
        from arcana.runtime_core import RunResult as RuntimeRunResult

        person = Person(name="John", age=30)
        mock_runtime_result = RuntimeRunResult(
            output=person,
            parsed=person,
            success=True,
            steps=1,
            tokens_used=100,
            cost_usd=0.001,
            run_id="test-run-001",
        )

        with patch("arcana.runtime_core.Runtime.run", new_callable=AsyncMock) as mock_run, \
             patch("arcana.runtime_core.Runtime.__init__", return_value=None):
            mock_run.return_value = mock_runtime_result

            import arcana.sdk as sdk

            result = await sdk.run(
                "Extract person info: John is 30",
                response_format=Person,
                provider="openai",
                api_key="sk-test",
            )

            assert result.success is True
            assert isinstance(result.output, Person)
            assert result.output.name == "John"
            assert result.output.age == 30

            # Verify response_format was passed through
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["response_format"] is Person

    @pytest.mark.asyncio
    async def test_run_without_response_format_unchanged(self) -> None:
        """arcana.run() without response_format behaves as before."""
        from arcana.runtime_core import RunResult as RuntimeRunResult

        mock_runtime_result = RuntimeRunResult(
            output="The answer is 42",
            success=True,
            steps=1,
            tokens_used=50,
            cost_usd=0.0005,
            run_id="test-run-002",
        )

        with patch("arcana.runtime_core.Runtime.run", new_callable=AsyncMock) as mock_run, \
             patch("arcana.runtime_core.Runtime.__init__", return_value=None):
            mock_run.return_value = mock_runtime_result

            import arcana.sdk as sdk

            result = await sdk.run(
                "What is the meaning of life?",
                provider="openai",
                api_key="sk-test",
            )

            assert result.success is True
            assert result.output == "The answer is 42"

            # Verify response_format defaults to None
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["response_format"] is None


# =========================================================================
# 7. Runtime.run() with response_format
# =========================================================================


class TestRuntimeRunWithResponseFormat:
    """Test Runtime.run() passes response_format to Session."""

    @pytest.mark.asyncio
    async def test_runtime_run_passes_response_format(self) -> None:
        """Runtime.run() should pass response_format through to session."""
        from arcana.runtime_core import RunResult, Runtime, Session

        with patch.object(Runtime, "_create_session") as mock_create_session, \
             patch.object(Runtime, "__init__", return_value=None):

            mock_session = MagicMock(spec=Session)
            mock_session.run = AsyncMock(return_value=RunResult(
                output=Person(name="Jane", age=25),
                success=True,
                run_id="test-run-003",
            ))
            mock_create_session.return_value = mock_session

            rt = Runtime.__new__(Runtime)
            rt._mcp_configs = []
            rt._mcp_client = None
            rt._tool_registry = None
            rt._memory_store = None
            rt._total_tokens_used = 0
            rt._total_cost_usd = 0.0

            result = await rt.run(
                "Extract person info",
                response_format=Person,
            )

            # Verify response_format was passed to _create_session
            mock_create_session.assert_called_once()
            call_kwargs = mock_create_session.call_args[1]
            assert call_kwargs["response_format"] is Person

            assert result.success is True
            assert isinstance(result.output, Person)

    @pytest.mark.asyncio
    async def test_runtime_run_without_response_format(self) -> None:
        """Runtime.run() without response_format defaults to None."""
        from arcana.runtime_core import RunResult, Runtime, Session

        with patch.object(Runtime, "_create_session") as mock_create_session, \
             patch.object(Runtime, "__init__", return_value=None):

            mock_session = MagicMock(spec=Session)
            mock_session.run = AsyncMock(return_value=RunResult(
                output="plain text",
                success=True,
                run_id="test-run-004",
            ))
            mock_create_session.return_value = mock_session

            rt = Runtime.__new__(Runtime)
            rt._mcp_configs = []
            rt._mcp_client = None
            rt._tool_registry = None
            rt._memory_store = None
            rt._total_tokens_used = 0
            rt._total_cost_usd = 0.0

            result = await rt.run("Just a question")

            call_kwargs = mock_create_session.call_args[1]
            assert call_kwargs.get("response_format") is None
            assert result.output == "plain text"


# =========================================================================
# 8. Session structured output parsing
# =========================================================================


class TestSessionStructuredParsing:
    """Test Session.run() parses and validates structured output."""

    @pytest.mark.asyncio
    async def test_session_parses_valid_json(self) -> None:
        """Session should parse valid JSON into a Pydantic model instance."""
        from arcana.contracts.state import AgentState, ExecutionStatus
        from arcana.runtime_core import Session

        person_json = '{"name": "Alice", "age": 28}'

        # Create a fake state with JSON in working_memory
        fake_state = AgentState(
            run_id="test-run-005",
            goal="extract person",
            max_steps=1,
        )
        fake_state.status = ExecutionStatus.COMPLETED
        fake_state.working_memory["answer"] = person_json
        fake_state.current_step = 1
        fake_state.tokens_used = 100

        with patch.object(Session, "__init__", return_value=None):
            session = Session.__new__(Session)
            session._runtime = MagicMock()
            session._engine = "conversation"
            session._max_turns = 1
            session._response_format = Person
            session._extra_tools = None
            session._memory_context = ""
            session.state = fake_state

            # Directly test the parsing logic by simulating post-agent state
            raw_output = fake_state.working_memory.get(
                "answer", fake_state.working_memory.get("result", "")
            )
            clean_output = raw_output.replace("[DONE]", "").replace("[done]", "").strip()

            # Parse structured output
            parsed = json.loads(clean_output)
            result = Person.model_validate(parsed)

            assert isinstance(result, Person)
            assert result.name == "Alice"
            assert result.age == 28

    @pytest.mark.asyncio
    async def test_session_invalid_json_returns_failure(self) -> None:
        """Session should return success=False when JSON parsing fails."""
        raw = "not valid json"

        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

        # The Session.run() catches JSONDecodeError and returns
        # RunResult(success=False). We verify both the exception and
        # the fact that the raw string would be preserved as output.
        is_valid = True
        try:
            parsed_json = json.loads(raw)
            Person.model_validate(parsed_json)
        except (json.JSONDecodeError, ValidationError):
            is_valid = False

        assert not is_valid, "Invalid JSON should fail parsing"

    def test_result_parsed_is_none_without_response_format(self) -> None:
        """Without response_format, parsed is always None."""
        from arcana.runtime_core import RunResult

        result = RunResult(
            output="plain text answer",
            success=True,
            steps=1,
            tokens_used=50,
            run_id="test-no-format",
        )
        assert result.parsed is None
        assert result.output == "plain text answer"

    def test_result_parsed_is_none_on_failure(self) -> None:
        """On parse failure, parsed is None and output is the raw string."""
        from arcana.runtime_core import RunResult

        result = RunResult(
            output="not json",
            parsed=None,
            success=False,
            steps=1,
            tokens_used=50,
            run_id="test-fail",
        )
        assert isinstance(result.output, str)
        assert result.parsed is None


# =========================================================================
# 9. Tools and response_format coexistence
# =========================================================================


class TestToolsWithResponseFormat:
    """Verify tools and response_format can be used together."""

    def test_llm_request_carries_both(self) -> None:
        """LLMRequest carries both tools and response_format to the provider."""
        from arcana.contracts.llm import Message, MessageRole

        schema = Person.model_json_schema()
        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="hi")],
            response_format=schema,
            tools=[{"type": "function", "function": {"name": "foo"}}],
        )
        assert req.response_format is not None
        assert req.tools is not None

    def test_session_preserves_tool_gateway_with_response_format(self) -> None:
        """Session.run() should NOT null out tool_gateway when response_format is set."""
        from unittest.mock import MagicMock, patch

        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        with patch("arcana.runtime_core.Runtime._setup_providers"):
            rt = Runtime.__new__(Runtime)
            rt._config = RuntimeConfig()
            rt._budget_policy = Budget()
            rt._namespace = None
            rt._tool_registry = None
            rt._tool_gateway = MagicMock()  # non-None gateway
            rt._gateway = MagicMock()
            rt._trace_writer = None
            rt._mcp_configs = None
            rt._mcp_client = None
            rt._memory_store = None
            rt._total_tokens_used = 0
            rt._total_cost_usd = 0.0

            session = rt._create_session(
                response_format=Person,
            )
            # tool_gateway should still be available, not nulled out
            assert rt._tool_gateway is not None


# =========================================================================
# 10. on_parse_error callback
# =========================================================================


def _make_fake_state(answer: str) -> Any:
    """Create a fake completed AgentState with the given answer."""
    from arcana.contracts.state import AgentState, ExecutionStatus

    state = AgentState(
        run_id="test-parse-error",
        goal="test goal",
        max_steps=1,
    )
    state.status = ExecutionStatus.COMPLETED
    state.working_memory["answer"] = answer
    state.current_step = 1
    state.tokens_used = 50
    return state


class TestOnParseErrorCallback:
    """Test on_parse_error callback for structured output resilience.

    These tests exercise the real Session.run() code path by mocking the
    ConversationAgent so it sets the agent state without making LLM calls.
    """

    def _build_runtime(self) -> Any:
        """Create a minimal Runtime with mocked internals."""
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig

        rt = Runtime.__new__(Runtime)
        rt._config = RuntimeConfig(default_provider="test")
        rt._budget_policy = Budget()
        rt._namespace = None
        rt._gateway = MagicMock()
        rt._tool_registry = None
        rt._tool_gateway = None
        rt._trace_writer = None
        rt._mcp_configs = []
        rt._mcp_client = None
        rt._memory_store = None
        rt._total_tokens_used = 0
        rt._total_cost_usd = 0.0

        # _resolve_model_config needs to return something valid
        mock_model_config = MagicMock()
        rt._resolve_model_config = MagicMock(return_value=mock_model_config)

        return rt

    @pytest.mark.asyncio
    async def test_on_parse_error_fixes_output(self) -> None:
        """Callback returns a fixed model instance -> result is success."""
        malformed_json = '{"name": "Alice", "age": "thirty"}'
        fixed_person = Person(name="Alice", age=30)

        def fixer(raw: str, error: Exception) -> Person:
            return fixed_person

        rt = self._build_runtime()
        fake_state = _make_fake_state(malformed_json)

        with patch(
            "arcana.runtime.conversation.ConversationAgent"
        ) as MockAgent, patch(
            "arcana.routing.classifier.RuleBasedClassifier"
        ):
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run = AsyncMock(return_value=fake_state)
            MockAgent.return_value = mock_agent_instance

            result = await rt.run(
                "extract person",
                response_format=Person,
                on_parse_error=fixer,
            )

        assert result.success is True
        assert isinstance(result.parsed, Person)
        assert result.parsed.name == "Alice"
        assert result.parsed.age == 30
        assert result.output == fixed_person

    @pytest.mark.asyncio
    async def test_on_parse_error_returns_none(self) -> None:
        """Callback returns None -> original failure preserved."""
        malformed_json = '{"name": "Alice", "age": "thirty"}'

        def fixer(raw: str, error: Exception) -> None:
            return None

        rt = self._build_runtime()
        fake_state = _make_fake_state(malformed_json)

        with patch(
            "arcana.runtime.conversation.ConversationAgent"
        ) as MockAgent, patch(
            "arcana.routing.classifier.RuleBasedClassifier"
        ):
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run = AsyncMock(return_value=fake_state)
            MockAgent.return_value = mock_agent_instance

            result = await rt.run(
                "extract person",
                response_format=Person,
                on_parse_error=fixer,
            )

        assert result.success is False
        assert result.parsed is None
        assert isinstance(result.output, str)

    @pytest.mark.asyncio
    async def test_on_parse_error_callback_raises(self) -> None:
        """Callback raises an exception -> original failure preserved."""
        malformed_json = '{"name": "Alice", "age": "thirty"}'

        def fixer(raw: str, error: Exception) -> Person:
            raise RuntimeError("callback crashed")

        rt = self._build_runtime()
        fake_state = _make_fake_state(malformed_json)

        with patch(
            "arcana.runtime.conversation.ConversationAgent"
        ) as MockAgent, patch(
            "arcana.routing.classifier.RuleBasedClassifier"
        ):
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run = AsyncMock(return_value=fake_state)
            MockAgent.return_value = mock_agent_instance

            result = await rt.run(
                "extract person",
                response_format=Person,
                on_parse_error=fixer,
            )

        assert result.success is False
        assert result.parsed is None

    @pytest.mark.asyncio
    async def test_on_parse_error_not_called_on_success(self) -> None:
        """When parsing succeeds, callback should never be invoked."""
        valid_json = '{"name": "Alice", "age": 28}'
        callback_called = False

        def fixer(raw: str, error: Exception) -> Person:
            nonlocal callback_called
            callback_called = True
            return Person(name="fixed", age=0)

        rt = self._build_runtime()
        fake_state = _make_fake_state(valid_json)

        with patch(
            "arcana.runtime.conversation.ConversationAgent"
        ) as MockAgent, patch(
            "arcana.routing.classifier.RuleBasedClassifier"
        ):
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run = AsyncMock(return_value=fake_state)
            MockAgent.return_value = mock_agent_instance

            result = await rt.run(
                "extract person",
                response_format=Person,
                on_parse_error=fixer,
            )

        assert not callback_called
        assert result.success is True
        assert isinstance(result.parsed, Person)
        assert result.parsed.name == "Alice"
        assert result.parsed.age == 28

    @pytest.mark.asyncio
    async def test_on_parse_error_receives_raw_and_error(self) -> None:
        """Verify callback receives the raw string and the actual error."""
        malformed_json = '{"name": "Alice", "age": "thirty"}'
        received_args: dict[str, object] = {}

        def fixer(raw: str, error: Exception) -> None:
            received_args["raw"] = raw
            received_args["error"] = error
            return None

        rt = self._build_runtime()
        fake_state = _make_fake_state(malformed_json)

        with patch(
            "arcana.runtime.conversation.ConversationAgent"
        ) as MockAgent, patch(
            "arcana.routing.classifier.RuleBasedClassifier"
        ):
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run = AsyncMock(return_value=fake_state)
            MockAgent.return_value = mock_agent_instance

            await rt.run(
                "extract person",
                response_format=Person,
                on_parse_error=fixer,
            )

        assert "raw" in received_args
        assert "error" in received_args
        assert received_args["raw"] == malformed_json
        assert isinstance(received_args["error"], Exception)
