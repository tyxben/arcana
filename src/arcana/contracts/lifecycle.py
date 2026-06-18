"""Typed payloads for V2 lifecycle (observer) hook events.

These are the ``event=`` payloads delivered to callbacks registered via
``Runtime.on(name, callback)`` at the V2 ``ConversationAgent`` boundaries
(``turn_start`` / ``turn_end`` / ``tool_start`` / ``tool_end``).

Lifecycle hooks are **observers**: they are notified, they cannot block,
rewrite, or redirect the run, and a raising hook is swallowed (observer hooks
fail open). The only blocking boundary is the explicit tool-call guardrail
(``contracts/guardrail.py``). Keeping observation and control on separate
surfaces is what prevents an observer hook from becoming a hidden planner
(Constitution v3.6 -- guardrails are boundaries, not hidden workflows).

The run-level ``run_start`` / ``run_end`` / ``error`` events keep their
historical keyword-argument payloads (``run_id`` / ``goal`` / ``result`` /
``error``) for backward compatibility; only the new turn/tool events carry
these typed objects.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _LifecycleEvent(BaseModel):
    """Base for immutable lifecycle event payloads."""

    model_config = ConfigDict(frozen=True)


class TurnStartEvent(_LifecycleEvent):
    """Emitted at the start of each conversation turn."""

    run_id: str
    turn: int  # 1-indexed turn number


class TurnEndEvent(_LifecycleEvent):
    """Emitted at the end of each conversation turn."""

    run_id: str
    turn: int
    turn_tokens: int = 0
    turn_cost_usd: float = 0.0
    tool_calls_made: int = 0
    completed: bool = False
    failed: bool = False


class ToolStartEvent(_LifecycleEvent):
    """Emitted before a tool call executes."""

    run_id: str
    turn: int
    tool_name: str
    tool_call_id: str = ""


class ToolEndEvent(_LifecycleEvent):
    """Emitted after a tool call result is available."""

    run_id: str
    turn: int
    tool_name: str
    tool_call_id: str = ""
    success: bool = True
    error: str | None = None
