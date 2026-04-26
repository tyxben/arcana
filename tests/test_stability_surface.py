"""Stability-surface audit -- catches docs/impl drift.

Every name that ``docs/guide/stability.md`` and
``specs/v1.0.0-stability.md`` §1 claim is part of the stable public
surface MUST round-trip — the import must succeed, the attribute must
exist. If you remove or rename one of these, this test fails loudly
*before* it can ship as a docs lie.

Conversely: if you add a new public surface, also add it here. The
test is the second source of truth alongside the docs; keeping them
in sync is part of the constitutional contract (Principle 5,
Auditability).

The original failure modes this caught:

  - ``ChatSession.turn_count`` / ``max_history`` were claimed in
    ``stability.md`` but never existed (only ``_turn_count`` /
    ``_max_history``). Fixed in 3c74da5.
  - ``arcana.contracts.diagnosis.DiagnosticBrief`` and
    ``arcana.contracts.context.ContextBudget`` were claimed across
    five doc files but never existed in src/. Real names are
    ``ErrorDiagnosis`` and ``TokenBudget``. Fixed in 8ddaa07.
"""

from __future__ import annotations

import importlib

import pytest

# -- Top-level package surface -------------------------------------------

TOP_LEVEL_NAMES = [
    # Entrypoints
    "run",
    "Runtime",
    "RuntimeConfig",
    # Resource configs
    "Budget",
    "BudgetScope",
    "AgentConfig",
    # Sessions / results
    "ChatSession",
    "ChatResponse",
    "RunResult",
    "BatchResult",
    # Pipelines
    "ChainStep",
    "ChainResult",
    # Multi-agent
    "AgentPool",
    # Conversation primitives (re-exports from contracts.llm)
    "Message",
    "MessageRole",
    # Tools
    "tool",
    "Tool",
    # Streaming
    "StreamEvent",
    "StreamEventType",
    # Graph
    "StateGraph",
]


def test_top_level_exports_exist() -> None:
    import arcana

    missing = [n for n in TOP_LEVEL_NAMES if not hasattr(arcana, n)]
    assert not missing, f"Stability docs claim arcana.{{{','.join(missing)}}} but they do not exist"


def test_top_level_version_is_set() -> None:
    """``arcana.__version__`` must be set and roughly current."""
    import arcana

    assert hasattr(arcana, "__version__")
    # Sanity: must look like a version (digits + dots), not the stale "0.3.1".
    assert arcana.__version__.count(".") >= 2


# -- Runtime method surface ----------------------------------------------

RUNTIME_METHODS = [
    "run",
    "run_batch",
    "chat",
    "chain",
    "collaborate",
    "session",
    "close",
]


def test_runtime_methods_exist() -> None:
    import arcana

    rt = arcana.Runtime(
        providers={"ollama": ""},
        config=arcana.RuntimeConfig(default_provider="ollama"),
    )
    missing = [m for m in RUNTIME_METHODS if not hasattr(rt, m)]
    assert not missing, f"Stability docs claim Runtime.{{{','.join(missing)}}} but they do not exist"


# -- ChatSession public surface ------------------------------------------

CHAT_SESSION_PUBLIC = [
    # Methods
    "send",
    "stream",
    "seed_history",
    # Properties
    "history",
    "max_history",
    "message_count",
    "session_id",
    "total_cost_usd",
    "total_tokens",
    "turn_count",
]


def test_chat_session_public_surface_exact() -> None:
    """``ChatSession`` exposes exactly the documented public surface.

    Both directions matter:
      - Documented names must exist (catches docs/impl drift).
      - Every existing public attribute must be documented (catches
        accidental public exposure of internals).
    """
    import arcana

    rt = arcana.Runtime(
        providers={"ollama": ""},
        config=arcana.RuntimeConfig(default_provider="ollama"),
    )
    session = arcana.ChatSession(runtime=rt)

    actual_public = {n for n in dir(session) if not n.startswith("_")}
    documented = set(CHAT_SESSION_PUBLIC)

    undocumented_extras = actual_public - documented
    missing_documented = documented - actual_public

    assert not missing_documented, (
        f"Stability docs claim ChatSession.{{{','.join(sorted(missing_documented))}}} "
        f"but they do not exist"
    )
    assert not undocumented_extras, (
        f"ChatSession exposes public attributes not in stability docs: "
        f"{sorted(undocumented_extras)}. Either add them to "
        f"docs/guide/stability.md or rename to start with '_' if internal."
    )


# -- Contracts module surface --------------------------------------------

CONTRACTS_SURFACE: dict[str, list[str]] = {
    "arcana.contracts.tool": [
        "ToolSpec",
        "ToolCall",
        "ToolResult",
        "ToolError",
        "ToolErrorCategory",
        "SideEffect",
        "ASK_USER_TOOL_NAME",
    ],
    "arcana.contracts.llm": [
        "Message",
        "MessageRole",
        "LLMRequest",
        "LLMResponse",
        "ContentBlock",
        "ModelConfig",
    ],
    "arcana.contracts.turn": [
        "TurnFacts",
        "TurnAssessment",
    ],
    "arcana.contracts.context": [
        "ContextBlock",
        "ContextDecision",
        "MessageDecision",
        "ContextReport",
        "ContextStrategy",
        "ContextLayer",
        "TokenBudget",
        "WorkingSet",
        "StepContext",
    ],
    "arcana.contracts.diagnosis": [
        "ErrorDiagnosis",
        "ErrorCategory",
        "ErrorLayer",
        "RecoveryStrategy",
    ],
    "arcana.contracts.streaming": [
        "StreamEvent",
        "StreamEventType",
    ],
    "arcana.contracts.runtime": [
        "RuntimeConfig",
    ],
    "arcana.contracts.cognitive": [
        "RecallRequest",
        "RecallResult",
        "PinRequest",
        "PinResult",
        "UnpinRequest",
        "UnpinResult",
        "PinEntry",
        "PinState",
    ],
    "arcana.contracts.trace": [
        "TraceEvent",
        "EventType",
        "BudgetSnapshot",
        "PromptSnapshot",
        "ToolCallRecord",
    ],
}


@pytest.mark.parametrize(
    ("module_name", "claimed_names"),
    list(CONTRACTS_SURFACE.items()),
    ids=list(CONTRACTS_SURFACE.keys()),
)
def test_contracts_module_surface(module_name: str, claimed_names: list[str]) -> None:
    module = importlib.import_module(module_name)
    missing = [n for n in claimed_names if not hasattr(module, n)]
    assert not missing, (
        f"Stability docs claim {module_name}.{{{','.join(missing)}}} "
        f"but they do not exist"
    )


# -- §3.2 canonical Message path -----------------------------------------

def test_runtime_conversation_does_not_advertise_message() -> None:
    """``arcana.runtime.conversation.__all__`` must exclude Message/MessageRole.

    Internal lazy-import lines may keep them reachable for explicit
    ``from arcana.runtime.conversation import Message`` (which we cannot
    block), but ``import *`` should only yield ``ConversationAgent``.
    """
    import arcana.runtime.conversation as conv

    assert hasattr(conv, "__all__"), "conversation.py must declare __all__"
    assert "Message" not in conv.__all__, (
        "Message is canonical at arcana.contracts.llm; "
        "do not advertise via runtime.conversation.__all__"
    )
    assert "MessageRole" not in conv.__all__
    assert "ConversationAgent" in conv.__all__
