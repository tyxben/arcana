"""Tests for the §3.5a tool-calling hint slot.

The hint is rendered as a separate system block when:
  - tools are bound to the request
  - a hint resolves non-empty (per-provider > global default)

The framework ships no defaults; the user supplies the content. The
user's authored system prompt is never mutated. The injection is
auditable via PromptSnapshot (no special trace event needed).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from arcana.contracts.llm import Message, MessageRole, ModelConfig
from arcana.runtime.conversation import ConversationAgent


def _make_agent(
    *,
    tool_calling_hint: str | None = None,
    tool_calling_hints: dict[str, str] | None = None,
) -> ConversationAgent:
    gateway = MagicMock()
    gateway.default_provider = "glm"
    gateway.get = MagicMock(return_value=None)
    return ConversationAgent(
        gateway=gateway,
        model_config=ModelConfig(provider="glm", model_id="glm-4-flash"),
        max_turns=1,
        tool_calling_hint=tool_calling_hint,
        tool_calling_hints=tool_calling_hints,
    )


def _system(content: str) -> Message:
    return Message(role=MessageRole.SYSTEM, content=content)


def _user(content: str) -> Message:
    return Message(role=MessageRole.USER, content=content)


def _assistant(content: str) -> Message:
    return Message(role=MessageRole.ASSISTANT, content=content)


# -- No-op cases ---------------------------------------------------------

class TestNoOpCases:
    def test_no_active_tools_means_no_injection(self) -> None:
        agent = _make_agent(tool_calling_hint="this is the hint")
        msgs = [_system("you are X"), _user("hi")]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")

        result = agent._maybe_inject_tool_calling_hint(msgs, [], config)
        assert result == msgs  # unchanged

    def test_no_active_tools_means_no_injection_even_with_per_provider(self) -> None:
        agent = _make_agent(tool_calling_hints={"glm": "GLM hint"})
        msgs = [_system("you are X")]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")

        result = agent._maybe_inject_tool_calling_hint(msgs, None, config)
        assert result == msgs

    def test_no_hint_configured_means_no_injection(self) -> None:
        agent = _make_agent()  # both None / empty
        msgs = [_system("X"), _user("hi")]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")

        result = agent._maybe_inject_tool_calling_hint(
            msgs, [{"name": "search", "description": "..."}], config,
        )
        assert result == msgs

    def test_empty_string_hint_is_treated_as_no_hint(self) -> None:
        """Empty string hint should be a no-op, matching None semantics."""
        agent = _make_agent(tool_calling_hint="")
        msgs = [_system("X")]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")
        result = agent._maybe_inject_tool_calling_hint(
            msgs, [{"name": "x"}], config,
        )
        assert result == msgs

    def test_per_provider_for_different_provider_is_no_op(self) -> None:
        """A hint for kimi must not apply to a glm request when no global is set."""
        agent = _make_agent(tool_calling_hints={"kimi": "kimi hint"})
        msgs = [_system("X")]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")

        result = agent._maybe_inject_tool_calling_hint(
            msgs, [{"name": "x"}], config,
        )
        assert result == msgs


# -- Injection cases -----------------------------------------------------

class TestInjection:
    def test_global_hint_injected_when_tools_bound(self) -> None:
        agent = _make_agent(tool_calling_hint="GLOBAL HINT")
        msgs = [_system("you are X"), _user("hi")]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")

        result = agent._maybe_inject_tool_calling_hint(
            msgs, [{"name": "search"}], config,
        )

        assert len(result) == 3
        assert result[0] == msgs[0]  # original system unchanged
        assert result[1].role == MessageRole.SYSTEM
        assert result[1].content == "GLOBAL HINT"
        assert result[2] == msgs[1]  # user message preserved

    def test_per_provider_hint_injected(self) -> None:
        agent = _make_agent(tool_calling_hints={"glm": "GLM SPECIFIC"})
        msgs = [_system("X"), _user("hi")]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")

        result = agent._maybe_inject_tool_calling_hint(
            msgs, [{"name": "x"}], config,
        )
        assert any(
            m.role == MessageRole.SYSTEM and m.content == "GLM SPECIFIC"
            for m in result
        )

    def test_per_provider_overrides_global(self) -> None:
        agent = _make_agent(
            tool_calling_hint="GLOBAL",
            tool_calling_hints={"glm": "GLM ONLY"},
        )
        msgs = [_system("X")]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")

        result = agent._maybe_inject_tool_calling_hint(
            msgs, [{"name": "x"}], config,
        )
        # Exactly one hint, the per-provider value
        hint_msgs = [m for m in result if m.role == MessageRole.SYSTEM and m.content == "GLM ONLY"]
        global_msgs = [m for m in result if m.role == MessageRole.SYSTEM and m.content == "GLOBAL"]
        assert len(hint_msgs) == 1
        assert len(global_msgs) == 0

    def test_global_used_when_provider_not_in_hints_dict(self) -> None:
        """If per-provider has entries but not for the active provider, global wins."""
        agent = _make_agent(
            tool_calling_hint="GLOBAL",
            tool_calling_hints={"kimi": "KIMI"},  # not glm
        )
        msgs = [_system("X")]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")
        result = agent._maybe_inject_tool_calling_hint(
            msgs, [{"name": "x"}], config,
        )
        assert any(m.content == "GLOBAL" for m in result if m.role == MessageRole.SYSTEM)


# -- Insertion order -----------------------------------------------------

class TestInsertionOrder:
    def test_inserted_after_leading_system_messages(self) -> None:
        """Hint goes after existing system blocks, before user/assistant turns."""
        agent = _make_agent(tool_calling_hint="HINT")
        msgs = [
            _system("identity"),
            _system("memory context"),  # multiple leading system messages
            _user("hi"),
            _assistant("hello"),
        ]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")

        result = agent._maybe_inject_tool_calling_hint(
            msgs, [{"name": "x"}], config,
        )

        # Expected ordering: identity / memory / HINT / user / assistant
        assert result[0].content == "identity"
        assert result[1].content == "memory context"
        assert result[2].content == "HINT"
        assert result[2].role == MessageRole.SYSTEM
        assert result[3].content == "hi"
        assert result[4].content == "hello"

    def test_inserted_at_start_when_no_leading_system(self) -> None:
        """If conversation starts with a non-system message, hint goes first."""
        agent = _make_agent(tool_calling_hint="HINT")
        msgs = [_user("hi"), _assistant("hello")]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")

        result = agent._maybe_inject_tool_calling_hint(
            msgs, [{"name": "x"}], config,
        )
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "HINT"
        assert result[1].content == "hi"

    def test_user_authored_system_prompt_is_not_mutated(self) -> None:
        """The injected hint is a SEPARATE message — original system content unchanged."""
        agent = _make_agent(tool_calling_hint="ADDITIONAL")
        original = _system("ORIGINAL USER PROMPT")
        msgs = [original]
        config = ModelConfig(provider="glm", model_id="glm-4-flash")

        result = agent._maybe_inject_tool_calling_hint(
            msgs, [{"name": "x"}], config,
        )
        # The original Message instance must still be in the list, untouched
        assert result[0] is original
        assert result[0].content == "ORIGINAL USER PROMPT"

    def test_no_framework_default_for_any_provider(self) -> None:
        """An agent constructed with no hint config injects nothing for ANY provider."""
        agent = _make_agent()
        msgs = [_system("X")]
        for provider in ("glm", "kimi", "openai", "deepseek", "anthropic", "minimax"):
            config = ModelConfig(provider=provider, model_id="x")
            result = agent._maybe_inject_tool_calling_hint(
                msgs, [{"name": "x"}], config,
            )
            assert result == msgs, f"Framework leaked a default for {provider}"


# -- Integration: RuntimeConfig plumbing --------------------------------

class TestRuntimeConfigPlumbing:
    def test_runtime_config_defaults_are_empty(self) -> None:
        from arcana.runtime_core import RuntimeConfig

        cfg = RuntimeConfig()
        assert cfg.tool_calling_hint is None
        assert cfg.tool_calling_hints == {}

    def test_runtime_config_accepts_per_provider(self) -> None:
        from arcana.runtime_core import RuntimeConfig

        cfg = RuntimeConfig(
            tool_calling_hint="global",
            tool_calling_hints={"glm": "glm-specific"},
        )
        assert cfg.tool_calling_hint == "global"
        assert cfg.tool_calling_hints == {"glm": "glm-specific"}

    def test_chatsession_picks_up_runtime_config_hint(self) -> None:
        """The hint flows from RuntimeConfig → Runtime → ChatSession's agent."""
        import arcana

        rt = arcana.Runtime(
            providers={"ollama": ""},
            config=arcana.RuntimeConfig(
                default_provider="ollama",
                tool_calling_hint="GLOBAL",
                tool_calling_hints={"ollama": "OLLAMA SPECIFIC"},
            ),
        )

        # Build the agent the same way ChatSession would
        session = arcana.ChatSession(runtime=rt)
        agent = session._build_agent("test goal")
        assert agent._tool_calling_hint == "GLOBAL"
        assert agent._tool_calling_hints == {"ollama": "OLLAMA SPECIFIC"}
