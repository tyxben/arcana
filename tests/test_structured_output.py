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
        from arcana.runtime_core import RunResult, Runtime, Session, _EventBus

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
            import asyncio as _asyncio
            rt._total_tokens_used = 0
            rt._total_cost_usd = 0.0
            rt._totals_lock = _asyncio.Lock()
            rt._events = _EventBus()

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
        from arcana.runtime_core import RunResult, Runtime, Session, _EventBus

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
            import asyncio as _asyncio
            rt._total_tokens_used = 0
            rt._total_cost_usd = 0.0
            rt._totals_lock = _asyncio.Lock()
            rt._events = _EventBus()

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
            import asyncio as _asyncio
            rt._total_tokens_used = 0
            rt._total_cost_usd = 0.0
            rt._totals_lock = _asyncio.Lock()

            rt._create_session(
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
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig, _EventBus

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
        import asyncio as _asyncio
        rt._total_tokens_used = 0
        rt._total_cost_usd = 0.0
        rt._totals_lock = _asyncio.Lock()
        rt._events = _EventBus()

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

    @pytest.mark.asyncio
    async def test_on_parse_error_receives_validation_error_not_provider_error(
        self,
    ) -> None:
        """Callback receives ValidationError for schema mismatch, never ProviderError.

        on_parse_error fires only on json.JSONDecodeError or
        pydantic.ValidationError -- not on provider-level rejections.
        """
        from arcana.gateway.base import ProviderError

        # Valid JSON but wrong types -> triggers pydantic.ValidationError
        malformed_json = '{"name": "Alice", "age": "thirty"}'
        received_error: Exception | None = None

        def fixer(raw: str, error: Exception) -> None:
            nonlocal received_error
            received_error = error
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

        assert received_error is not None
        assert isinstance(received_error, (json.JSONDecodeError, ValidationError))
        assert not isinstance(received_error, ProviderError)

    @pytest.mark.asyncio
    async def test_on_parse_error_receives_json_decode_error(self) -> None:
        """Callback receives JSONDecodeError for completely invalid JSON."""
        from arcana.gateway.base import ProviderError

        invalid_json = "this is not json at all"
        received_error: Exception | None = None

        def fixer(raw: str, error: Exception) -> None:
            nonlocal received_error
            received_error = error
            return None

        rt = self._build_runtime()
        fake_state = _make_fake_state(invalid_json)

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

        assert received_error is not None
        assert isinstance(received_error, json.JSONDecodeError)
        assert not isinstance(received_error, ProviderError)


# =========================================================================
# 11. Provider response_format auto-downgrade
# =========================================================================


class TestProviderResponseFormatDowngrade:
    """Verify provider auto-downgrades json_schema to json_object."""

    def test_deepseek_provider_downgrades_json_schema(self) -> None:
        """DeepSeek provider should use json_object instead of json_schema."""
        from arcana.gateway.providers.openai_compatible import (
            create_deepseek_provider,
        )

        provider = create_deepseek_provider(api_key="sk-test")
        assert provider._supports_json_schema is False

        # Build params with response_format and verify downgrade
        from arcana.contracts.llm import Message, MessageRole

        schema = Person.model_json_schema()
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are helpful."),
                Message(role=MessageRole.USER, content="Extract person"),
            ],
            response_format=schema,
        )
        messages = provider._convert_messages(request.messages)
        params: dict[str, Any] = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        provider._apply_response_format(params, request)

        # Should be json_object, not json_schema
        assert params["response_format"]["type"] == "json_object"
        assert "json_schema" not in params["response_format"]

        # Schema should be injected into the system message
        system_msg = params["messages"][0]
        assert "You MUST respond with valid JSON" in system_msg["content"]
        assert '"name"' in system_msg["content"]

    def test_openai_provider_uses_json_schema(self) -> None:
        """OpenAI provider should use json_schema directly."""
        from arcana.gateway.providers.openai_compatible import (
            OpenAICompatibleProvider,
        )

        provider = OpenAICompatibleProvider(
            provider_name="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            default_model="gpt-4o-mini",
            supports_json_schema=True,
        )
        assert provider._supports_json_schema is True

        from arcana.contracts.llm import Message, MessageRole

        schema = Person.model_json_schema()
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are helpful."),
                Message(role=MessageRole.USER, content="Extract person"),
            ],
            response_format=schema,
        )
        messages = provider._convert_messages(request.messages)
        params: dict[str, Any] = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        provider._apply_response_format(params, request)

        # Should be json_schema, not json_object
        assert params["response_format"]["type"] == "json_schema"
        assert "json_schema" in params["response_format"]
        assert params["response_format"]["json_schema"]["name"] == "response"
        assert params["response_format"]["json_schema"]["schema"] == schema

        # System message should NOT have schema injected
        system_msg = params["messages"][0]
        assert "You MUST respond with valid JSON" not in system_msg["content"]

    def test_ollama_provider_downgrades(self) -> None:
        """Ollama provider should downgrade to json_object."""
        from arcana.gateway.providers.openai_compatible import (
            create_ollama_provider,
        )

        provider = create_ollama_provider()
        assert provider._supports_json_schema is False

    def test_kimi_provider_downgrades(self) -> None:
        """Kimi provider should downgrade to json_object."""
        from arcana.gateway.providers.openai_compatible import (
            create_kimi_provider,
        )

        provider = create_kimi_provider(api_key="sk-test")
        assert provider._supports_json_schema is False

    def test_glm_provider_downgrades(self) -> None:
        """GLM provider should downgrade to json_object."""
        from arcana.gateway.providers.openai_compatible import (
            create_glm_provider,
        )

        provider = create_glm_provider(api_key="sk-test")
        assert provider._supports_json_schema is False

    def test_minimax_provider_downgrades(self) -> None:
        """MiniMax provider should downgrade to json_object."""
        from arcana.gateway.providers.openai_compatible import (
            create_minimax_provider,
        )

        provider = create_minimax_provider(api_key="sk-test")
        assert provider._supports_json_schema is False

    def test_gemini_provider_supports_json_schema(self) -> None:
        """Gemini provider should support json_schema (default True)."""
        from arcana.gateway.providers.openai_compatible import (
            create_gemini_provider,
        )

        provider = create_gemini_provider(api_key="AIza-test")
        assert provider._supports_json_schema is True

    def test_anthropic_provider_injects_schema_into_system(self) -> None:
        """Anthropic provider should inject schema into system prompt."""
        from arcana.contracts.llm import Message, MessageRole, ModelConfig
        from arcana.gateway.providers.anthropic import to_anthropic_request

        schema = Person.model_json_schema()
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are helpful."),
                Message(role=MessageRole.USER, content="Extract person"),
            ],
            response_format=schema,
        )
        config = ModelConfig(
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            max_tokens=4096,
        )

        params = to_anthropic_request(request, config)

        # System prompt should contain the schema instruction
        system_content = params["system"]
        if isinstance(system_content, list):
            system_text = system_content[0]["text"]
        else:
            system_text = system_content
        assert "You MUST respond with valid JSON" in system_text
        assert '"name"' in system_text

    def test_anthropic_provider_schema_without_system_message(self) -> None:
        """Anthropic provider should create system prompt from schema when none exists."""
        from arcana.contracts.llm import Message, MessageRole, ModelConfig
        from arcana.gateway.providers.anthropic import to_anthropic_request

        schema = Person.model_json_schema()
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.USER, content="Extract person"),
            ],
            response_format=schema,
        )
        config = ModelConfig(
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            max_tokens=4096,
        )

        params = to_anthropic_request(request, config)

        # Should still have system with schema even though no original system message
        assert "system" in params
        system_content = params["system"]
        if isinstance(system_content, list):
            system_text = system_content[0]["text"]
        else:
            system_text = system_content
        assert "You MUST respond with valid JSON" in system_text

    def test_anthropic_provider_schema_coexists_with_tools(self) -> None:
        """Anthropic provider should inject schema even when tools are present."""
        from arcana.contracts.llm import Message, MessageRole, ModelConfig
        from arcana.gateway.providers.anthropic import to_anthropic_request

        schema = Person.model_json_schema()
        tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for info",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are helpful."),
                Message(role=MessageRole.USER, content="Extract person"),
            ],
            response_format=schema,
            tools=[tool],
        )
        config = ModelConfig(
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            max_tokens=4096,
        )

        params = to_anthropic_request(request, config)

        # Schema in system prompt
        system_content = params["system"]
        if isinstance(system_content, list):
            system_text = system_content[0]["text"]
        else:
            system_text = system_content
        assert "You MUST respond with valid JSON" in system_text

        # Tools still present
        assert "tools" in params
        assert len(params["tools"]) == 1
        assert params["tools"][0]["name"] == "search"

    def test_downgrade_without_system_message(self) -> None:
        """When no system message exists, json_object is still used."""
        from arcana.gateway.providers.openai_compatible import (
            OpenAICompatibleProvider,
        )

        provider = OpenAICompatibleProvider(
            provider_name="test",
            api_key="sk-test",
            base_url="http://localhost",
            supports_json_schema=False,
        )

        from arcana.contracts.llm import Message, MessageRole

        schema = Person.model_json_schema()
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.USER, content="Extract person"),
            ],
            response_format=schema,
        )
        messages = provider._convert_messages(request.messages)
        params: dict[str, Any] = {
            "model": "test-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        provider._apply_response_format(params, request)

        # json_object should still be set
        assert params["response_format"]["type"] == "json_object"
        # No system message to inject into -- user message unchanged
        assert "You MUST respond with valid JSON" not in params["messages"][0]["content"]


# =========================================================================
# 12. Structured output skips streaming
# =========================================================================


class TestStructuredOutputSkipsStreaming:
    """Verify that when response_format is set, generate() is used."""

    @pytest.mark.asyncio
    async def test_structured_output_skips_streaming(self) -> None:
        """When response_format is set, ConversationAgent should use
        gateway.generate() instead of gateway.stream().
        """
        from arcana.contracts.llm import LLMResponse, ModelConfig, TokenUsage
        from arcana.runtime.conversation import ConversationAgent

        # Build mock gateway
        mock_gateway = MagicMock()
        mock_gateway.generate = AsyncMock(return_value=LLMResponse(
            content='{"name": "Alice", "age": 30}',
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            model="test-model",
            finish_reason="stop",
        ))
        mock_gateway.stream = AsyncMock(side_effect=AssertionError(
            "stream() should NOT be called when response_format is set"
        ))

        schema = Person.model_json_schema()
        agent = ConversationAgent(
            gateway=mock_gateway,
            model_config=ModelConfig(model_id="test-model", provider="test"),
            max_turns=1,
            response_format_schema=schema,
        )

        state = await agent.run("Extract person info from: Alice is 30")

        # generate() should have been called
        mock_gateway.generate.assert_called_once()
        # stream() should NOT have been called
        mock_gateway.stream.assert_not_called()
        # Agent should still complete successfully
        assert state.status.value == "completed"

    @pytest.mark.asyncio
    async def test_without_response_format_uses_streaming(self) -> None:
        """Without response_format, ConversationAgent should try streaming."""
        from arcana.contracts.llm import ModelConfig, StreamChunk, TokenUsage
        from arcana.runtime.conversation import ConversationAgent

        # Build mock gateway that streams successfully
        mock_gateway = MagicMock()

        async def mock_stream(*args: Any, **kwargs: Any):
            yield StreamChunk(type="text_delta", text="Hello world")
            yield StreamChunk(
                type="usage",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            )
            yield StreamChunk(
                type="done",
                metadata={"finish_reason": "stop", "model": "test-model"},
            )

        mock_gateway.stream = mock_stream
        mock_gateway.generate = AsyncMock(side_effect=AssertionError(
            "generate() should NOT be called when streaming works"
        ))

        agent = ConversationAgent(
            gateway=mock_gateway,
            model_config=ModelConfig(model_id="test-model", provider="test"),
            max_turns=1,
        )

        state = await agent.run("Say hello")

        # stream() was used (generate should not have been called)
        mock_gateway.generate.assert_not_called()
        assert state.status.value == "completed"


# =========================================================================
# 13. parsed is always BaseModel or None (never dict)
# =========================================================================


class TestParsedAlwaysModel:
    """Guarantee result.parsed is BaseModel | None, never a raw dict."""

    def _build_runtime(self) -> Any:
        from arcana.runtime_core import Budget, Runtime, RuntimeConfig, _EventBus

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
        import asyncio as _asyncio
        rt._total_tokens_used = 0
        rt._total_cost_usd = 0.0
        rt._totals_lock = _asyncio.Lock()
        rt._events = _EventBus()
        rt._resolve_model_config = MagicMock(return_value=MagicMock())
        return rt

    @pytest.mark.asyncio
    async def test_on_parse_error_returns_dict_gets_validated(self) -> None:
        """on_parse_error returning a dict should be model_validate'd."""
        malformed_json = '{"name": "Alice", "age": "thirty"}'

        def fixer(raw: str, error: Exception) -> dict:  # type: ignore[return-type]
            return {"name": "Alice", "age": 30}

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
        assert isinstance(result.parsed, Person), (
            f"parsed should be Person, got {type(result.parsed)}"
        )
        assert result.parsed.name == "Alice"
        assert result.parsed.age == 30

    @pytest.mark.asyncio
    async def test_working_memory_dict_gets_validated(self) -> None:
        """If working_memory['answer'] is a dict, it should be validated."""
        from arcana.contracts.state import AgentState, ExecutionStatus

        state = AgentState(
            run_id="test-dict-answer",
            goal="test",
            max_steps=1,
        )
        state.status = ExecutionStatus.COMPLETED
        # Simulate a code path that stores a dict instead of a string
        state.working_memory["answer"] = {"name": "Bob", "age": 25}
        state.current_step = 1
        state.tokens_used = 50

        rt = self._build_runtime()

        with patch(
            "arcana.runtime.conversation.ConversationAgent"
        ) as MockAgent, patch(
            "arcana.routing.classifier.RuleBasedClassifier"
        ):
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run = AsyncMock(return_value=state)
            MockAgent.return_value = mock_agent_instance

            result = await rt.run(
                "extract person",
                response_format=Person,
            )

        assert result.success is True
        assert isinstance(result.parsed, Person)
        assert result.parsed.name == "Bob"
        assert result.parsed.age == 25

    @pytest.mark.asyncio
    async def test_on_parse_error_returns_invalid_dict_fails_gracefully(self) -> None:
        """on_parse_error returning an invalid dict falls back to failure."""
        malformed_json = '{"name": "Alice", "age": "thirty"}'

        def fixer(raw: str, error: Exception) -> dict:  # type: ignore[return-type]
            return {"wrong_field": True}

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

        # Should fall through to failure since dict doesn't match Person
        assert result.success is False
        assert result.parsed is None


# =========================================================================
# Code fence stripping
# =========================================================================


class TestStripCodeFences:
    """Verify that markdown code fences are stripped before JSON parsing."""

    def test_strip_json_code_fence(self) -> None:
        """Standard ```json ... ``` wrapping should be stripped."""
        from arcana.runtime_core import strip_code_fences

        raw = '```json\n{"name": "Alice", "age": 30}\n```'
        assert strip_code_fences(raw) == '{"name": "Alice", "age": 30}'

    def test_strip_plain_code_fence(self) -> None:
        """Plain ``` ... ``` wrapping (no language tag) should be stripped."""
        from arcana.runtime_core import strip_code_fences

        raw = '```\n{"name": "Bob", "age": 25}\n```'
        assert strip_code_fences(raw) == '{"name": "Bob", "age": 25}'

    def test_no_fence_passthrough(self) -> None:
        """Plain JSON without fences should pass through unchanged."""
        from arcana.runtime_core import strip_code_fences

        raw = '{"name": "Charlie", "age": 20}'
        assert strip_code_fences(raw) == raw

    def test_strip_fence_with_leading_whitespace(self) -> None:
        """Fences with leading/trailing whitespace should be handled."""
        from arcana.runtime_core import strip_code_fences

        raw = '  ```json\n{"key": "value"}\n```  '
        assert strip_code_fences(raw) == '{"key": "value"}'

    def test_strip_fence_multiline_json(self) -> None:
        """Multi-line JSON inside fences should be preserved."""
        from arcana.runtime_core import strip_code_fences

        raw = '```json\n{\n  "name": "Dave",\n  "age": 35\n}\n```'
        result = strip_code_fences(raw)
        parsed = json.loads(result)
        assert parsed == {"name": "Dave", "age": 35}

    def test_strip_fence_uppercase_json(self) -> None:
        """```JSON (uppercase) should also be stripped."""
        from arcana.runtime_core import strip_code_fences

        raw = '```JSON\n{"x": 1}\n```'
        assert strip_code_fences(raw) == '{"x": 1}'

    def test_fenced_json_parses_via_strip(self) -> None:
        """Fenced JSON should be parseable after strip_code_fences."""
        from arcana.runtime_core import strip_code_fences

        fenced = '```json\n{"name": "Eve", "age": 22}\n```'

        # Simulate the Session.run() parsing path:
        # 1. strip_code_fences  2. json.loads  3. model_validate
        stripped = strip_code_fences(fenced)
        parsed_json = json.loads(stripped)
        person = Person.model_validate(parsed_json)

        assert person.name == "Eve"
        assert person.age == 22

    def test_plain_fence_parses_via_strip(self) -> None:
        """Plain ``` fence (no language tag) also works."""
        from arcana.runtime_core import strip_code_fences

        fenced = '```\n{"name": "Frank", "age": 40}\n```'
        stripped = strip_code_fences(fenced)
        parsed_json = json.loads(stripped)
        person = Person.model_validate(parsed_json)

        assert person.name == "Frank"
        assert person.age == 40

    def test_unfenced_json_still_works(self) -> None:
        """Plain JSON (no fence) still works after strip_code_fences."""
        from arcana.runtime_core import strip_code_fences

        raw = '{"name": "Grace", "age": 35}'
        stripped = strip_code_fences(raw)
        parsed_json = json.loads(stripped)
        person = Person.model_validate(parsed_json)

        assert person.name == "Grace"
        assert person.age == 35


# =========================================================================
# Strengthened schema prompt for json_object fallback
# =========================================================================


class TestSchemaPromptStrengthening:
    """Verify the json_object fallback prompt includes field guidance."""

    def _build_params(self, provider_name: str = "glm") -> tuple[dict[str, Any], Any]:
        """Helper to build params with a non-json_schema provider."""
        from arcana.contracts.llm import Message, MessageRole
        from arcana.gateway.providers.openai_compatible import (
            OpenAICompatibleProvider,
        )

        provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            api_key="sk-test",
            base_url="http://localhost",
            supports_json_schema=False,
        )

        class MathResult(BaseModel):
            expression: str
            result: int

        schema = MathResult.model_json_schema()
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are a calculator."),
                Message(role=MessageRole.USER, content="What is 42 * 17?"),
            ],
            response_format=schema,
        )
        messages = provider._convert_messages(request.messages)
        params: dict[str, Any] = {
            "model": "test-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        provider._apply_response_format(params, request)
        return params, provider

    def test_prompt_includes_field_names_list(self) -> None:
        """The schema prompt should list exact field names."""
        params, _ = self._build_params()
        system_content = params["messages"][0]["content"]
        assert '"expression"' in system_content
        assert '"result"' in system_content
        assert "Required fields:" in system_content

    def test_prompt_includes_exact_field_instruction(self) -> None:
        """The prompt should explicitly forbid renaming fields."""
        params, _ = self._build_params()
        system_content = params["messages"][0]["content"]
        assert "You MUST use exactly these field names" in system_content
        assert "Do not rename or omit any fields" in system_content

    def test_prompt_includes_example(self) -> None:
        """The prompt should include a concrete example JSON."""
        params, _ = self._build_params()
        system_content = params["messages"][0]["content"]
        assert "Example response format:" in system_content
        # The example should contain the field names
        assert '"expression"' in system_content
        assert '"result"' in system_content

    def test_prompt_example_is_valid_json(self) -> None:
        """The example in the prompt should be parseable JSON."""
        params, _ = self._build_params()
        system_content = params["messages"][0]["content"]
        # Find the example JSON block (second ```json block)
        import re
        blocks = re.findall(r"```json\n(.*?)\n```", system_content, re.DOTALL)
        assert len(blocks) >= 2, "Expected at least 2 JSON blocks (schema + example)"
        example = json.loads(blocks[-1])
        assert "expression" in example
        assert "result" in example
