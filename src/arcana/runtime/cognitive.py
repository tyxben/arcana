"""Cognitive handler — services recall / pin / unpin tool invocations.

This is the runtime-side counterpart of ``arcana.contracts.cognitive``. The
handler is owned by a ``ConversationAgent`` (session-local) and is invoked
from ``_execute_tools`` when the LLM calls one of the cognitive primitive
tool names. No handler work goes through ``ToolGateway`` — same interception
pattern as ``ask_user``.

Design notes:

- Recall delegates to ``TraceReader`` (or an in-memory conversation log
  fallback) — no parallel trace parsing.
- Pin is idempotent by SHA-256 of content. The framework never auto-unpins
  and never truncates existing pins; pin requests that would exceed the
  budget cap are rejected with a structured ``PinResult`` the LLM can act on.
- The handler never calls a primitive on the LLM's behalf, and never inserts
  hint text into prompts. Primitives are a door the LLM may open.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from arcana.contracts.cognitive import (
    PinEntry,
    PinRequest,
    PinResult,
    PinState,
    RecallRequest,
    RecallResult,
    UnpinRequest,
    UnpinResult,
    _content_hash,
)
from arcana.contracts.tool import SideEffect, ToolSpec

if TYPE_CHECKING:
    from arcana.trace.reader import TraceReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool specs (shown to the LLM when the primitive is enabled)
# ---------------------------------------------------------------------------

RECALL_TOOL_NAME = "recall"
PIN_TOOL_NAME = "pin"
UNPIN_TOOL_NAME = "unpin"


RECALL_SPEC = ToolSpec(
    name=RECALL_TOOL_NAME,
    description=(
        "Retrieve the full original content of a past turn, bypassing any "
        "working-set compression. Use when you need exact wording of an "
        "earlier message that may have been compressed or summarized."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "turn": {
                "type": "integer",
                "description": "Turn number (1-indexed) to recall.",
            },
            "include": {
                "type": "string",
                "enum": ["all", "assistant_only", "tool_calls"],
                "description": (
                    "Filter which messages to return from that turn. "
                    "Default 'all'."
                ),
            },
        },
        "required": ["turn"],
    },
    when_to_use=(
        "When you need the exact wording of an earlier message that may "
        "have been compressed or dropped from the working set. Useful "
        "for verifying specific details of a plan, assumption, or "
        "conclusion stated earlier."
    ),
    what_to_expect=(
        "A structured result with `found`, `turn`, `messages` (list of "
        "role/content dicts), and optional `note`. When `found` is false, "
        "`note` explains why (e.g. turn out of range)."
    ),
    failure_meaning=(
        "Turn is out of range, invalid, or the trace is unavailable. "
        "Proceed with the working-set content you already have, or ask "
        "the user to re-supply the needed detail."
    ),
    side_effect=SideEffect.READ,
)


PIN_SPEC = ToolSpec(
    name=PIN_TOOL_NAME,
    description=(
        "Protect specific content from compression in future working sets. "
        "Use for critical conclusions or facts you will need at full "
        "fidelity in subsequent reasoning. Returns a pin_id you can use "
        "with unpin later."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Exact content to pin at full fidelity.",
            },
            "label": {
                "type": "string",
                "description": "Optional readable label for the pin.",
            },
            "until_turn": {
                "type": "integer",
                "description": (
                    "Auto-expire the pin after this turn number. Omit for "
                    "a session-long pin."
                ),
            },
        },
        "required": ["content"],
    },
    when_to_use=(
        "After deriving a conclusion, assumption, or fact you will need "
        "intact several turns later. Pin sparingly — the pin budget is "
        "capped, and over-pinning starves the working set."
    ),
    what_to_expect=(
        "A PinResult. On success, `pinned=True` with a `pin_id`. If the "
        "same content is already pinned, `already_pinned=True` and the "
        "existing pin_id is returned. If the pin would exceed the budget "
        "cap, `pinned=False` with a diagnostic `reason` and `suggestion`."
    ),
    failure_meaning=(
        "Pin budget exhausted. Consider unpinning older content, pinning "
        "a shorter excerpt, or proceeding without pinning."
    ),
    side_effect=SideEffect.WRITE,
)


UNPIN_SPEC = ToolSpec(
    name=UNPIN_TOOL_NAME,
    description=(
        "Remove a previously created pin by its pin_id. The content will "
        "be subject to normal compression on subsequent turns."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pin_id": {
                "type": "string",
                "description": "The pin_id returned from a prior pin call.",
            },
        },
        "required": ["pin_id"],
    },
    when_to_use=(
        "When a pinned piece of content is no longer load-bearing for "
        "upcoming reasoning, or to free budget for a new pin."
    ),
    what_to_expect=(
        "An UnpinResult with `unpinned=True` on success, or `unpinned=False` "
        "with a `note` if the pin_id was unknown."
    ),
    failure_meaning=(
        "The pin_id did not exist. Either it was already unpinned, or the "
        "id is wrong. No action needed."
    ),
    side_effect=SideEffect.WRITE,
)


# Mapping from primitive name → spec, used by ConversationAgent when
# injecting cognitive tool schemas into the tool list.
COGNITIVE_TOOL_SPECS: dict[str, ToolSpec] = {
    RECALL_TOOL_NAME: RECALL_SPEC,
    PIN_TOOL_NAME: PIN_SPEC,
    # unpin is implicitly enabled whenever pin is.
}


COGNITIVE_TOOL_NAMES: frozenset[str] = frozenset(
    {RECALL_TOOL_NAME, PIN_TOOL_NAME, UNPIN_TOOL_NAME}
)


def is_cognitive_tool(name: str) -> bool:
    """Return True if ``name`` is a cognitive primitive tool name."""
    return name in COGNITIVE_TOOL_NAMES


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Reuse the WorkingSetBuilder token estimator."""
    from arcana.context.builder import estimate_tokens

    return estimate_tokens(text)


class CognitiveHandler:
    """Runtime handler for cognitive primitive tool calls.

    One handler per session. The handler:

    - Knows which primitives are enabled (``enabled``).
    - Owns the session's ``PinState``.
    - Services ``recall`` using in-memory conversation history (preferred)
      or a ``TraceReader`` as fallback.
    - Enforces the pin budget cap (``pin_budget_fraction * total_window``).

    The handler never invokes a primitive on its own — only when
    ``ConversationAgent._execute_tools`` dispatches here because the LLM
    made an explicit tool call.
    """

    def __init__(
        self,
        *,
        enabled: set[str] | None = None,
        pin_budget_fraction: float = 0.5,
        total_token_window: int = 128_000,
        trace_reader: TraceReader | None = None,
        run_id: str | None = None,
    ) -> None:
        self.enabled: set[str] = set(enabled or set())
        self.pin_budget_fraction = pin_budget_fraction
        self.total_token_window = total_token_window
        self._trace_reader = trace_reader
        self._run_id = run_id
        self.pin_state = PinState()

        # Conversation log: a list where index [i] holds the assistant /
        # tool messages produced during the 1-indexed turn (i+1). This is
        # the primary source for recall; the trace reader is the fallback.
        # Populated by ConversationAgent via record_turn().
        self._turn_log: list[list[dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def recall_enabled(self) -> bool:
        return RECALL_TOOL_NAME in self.enabled

    @property
    def pin_enabled(self) -> bool:
        return PIN_TOOL_NAME in self.enabled

    @property
    def pin_budget_cap_tokens(self) -> int:
        return int(self.total_token_window * self.pin_budget_fraction)

    # ------------------------------------------------------------------
    # Turn log — populated by the agent as the conversation progresses
    # ------------------------------------------------------------------

    def record_turn(self, turn: int, messages: list[dict[str, Any]]) -> None:
        """Store messages produced during ``turn`` (1-indexed).

        Called by the ``ConversationAgent`` after each turn so that
        ``recall`` can serve content from the live session without
        depending on the trace writer being enabled.
        """
        # turn is 1-indexed; ensure the list is long enough.
        while len(self._turn_log) < turn:
            self._turn_log.append([])
        self._turn_log[turn - 1] = list(messages)

    def max_turn(self) -> int:
        return len(self._turn_log)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def handle_recall(self, req: RecallRequest) -> RecallResult:
        """Service a recall tool call.

        All failure modes return structured results, never raise.
        """
        if req.turn <= 0:
            return RecallResult(
                turn=req.turn,
                found=False,
                note=(
                    "turn must be 1 or greater (turns are 1-indexed). "
                    f"received: {req.turn}"
                ),
            )

        max_turn = self.max_turn()
        # Prefer in-memory turn log for mid-session calls.
        if max_turn > 0 and req.turn > max_turn:
            return RecallResult(
                turn=req.turn,
                found=False,
                note=f"turn out of range: {req.turn}, max={max_turn}",
            )

        messages: list[dict[str, Any]] = []
        if max_turn > 0:
            messages = list(self._turn_log[req.turn - 1])
        elif self._trace_reader is not None and self._run_id is not None:
            # Fallback: try to pull from a completed trace on disk.
            messages = self._recall_from_trace(req.turn)
        else:
            return RecallResult(
                turn=req.turn,
                found=False,
                note=(
                    "trace not available for this run; no in-memory "
                    "history has been recorded yet."
                ),
            )

        if not messages:
            return RecallResult(
                turn=req.turn,
                found=False,
                note=f"no messages recorded for turn {req.turn}",
            )

        filtered = self._filter_messages(messages, req.include)
        return RecallResult(turn=req.turn, found=True, messages=filtered)

    def _recall_from_trace(self, turn: int) -> list[dict[str, Any]]:
        """Pull messages for ``turn`` from the prompt-snapshot trace event.

        Requires ``trace_include_prompt_snapshots`` to have been enabled
        when the run was recorded; otherwise returns an empty list.
        """
        if self._trace_reader is None or self._run_id is None:
            return []
        try:
            replay = self._trace_reader.replay_prompt(
                self._run_id, turn=turn - 1,
            )
        except Exception:  # noqa: BLE001 — trace read is best-effort
            return []
        if replay is None or replay.prompt_snapshot is None:
            return []
        return list(replay.prompt_snapshot.messages)

    @staticmethod
    def _filter_messages(
        messages: list[dict[str, Any]],
        include: str,
    ) -> list[dict[str, Any]]:
        if include == "all":
            return messages
        if include == "assistant_only":
            return [m for m in messages if m.get("role") == "assistant"]
        if include == "tool_calls":
            out: list[dict[str, Any]] = []
            for m in messages:
                if m.get("role") == "tool":
                    out.append(m)
                    continue
                # Assistant messages that contain tool_calls
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    out.append(m)
            return out
        # Unknown include value — return everything rather than failing.
        return messages

    # ------------------------------------------------------------------
    # Pin
    # ------------------------------------------------------------------

    def handle_pin(self, req: PinRequest, *, current_turn: int) -> PinResult:
        """Service a pin tool call.

        - Idempotent: same content returns the existing pin_id.
        - Hard budget cap: total pinned tokens <= cap.
        - Never auto-unpins; never truncates existing pins.
        """
        if not req.content:
            return PinResult(
                pinned=False,
                reason="empty_content",
                suggestion="pass non-empty content to pin.",
            )

        content_hash = _content_hash(req.content)

        # Idempotency: same content → return existing pin as no-op.
        existing = self.pin_state.find_by_hash(content_hash)
        if existing is not None:
            return PinResult(
                pinned=True,
                pin_id=existing.pin_id,
                label=existing.label,
                until_turn=existing.until_turn,
                already_pinned=True,
            )

        requested_tokens = _estimate_tokens(req.content)
        current_tokens = self.pin_state.total_tokens()
        cap = self.pin_budget_cap_tokens

        if current_tokens + requested_tokens > cap:
            return PinResult(
                pinned=False,
                reason="pin_budget_exceeded",
                current_pin_tokens=current_tokens,
                requested_tokens=requested_tokens,
                cap=cap,
                suggestion=(
                    "unpin older content (see active pins listed in prior "
                    "working-set decisions) or pin a shorter excerpt."
                ),
            )

        pin_id = f"p_{uuid4().hex[:8]}"
        entry = PinEntry(
            pin_id=pin_id,
            content=req.content,
            content_hash=content_hash,
            label=req.label,
            until_turn=req.until_turn,
            created_turn=current_turn,
            token_count=requested_tokens,
        )
        self.pin_state.add(entry)

        return PinResult(
            pinned=True,
            pin_id=pin_id,
            label=req.label,
            until_turn=req.until_turn,
            already_pinned=False,
        )

    def handle_unpin(self, req: UnpinRequest) -> UnpinResult:
        """Service an unpin tool call.

        Returns a structured result whether or not the pin existed.
        """
        removed = self.pin_state.remove(req.pin_id)
        if removed:
            return UnpinResult(unpinned=True, pin_id=req.pin_id)
        return UnpinResult(
            unpinned=False,
            pin_id=req.pin_id,
            note="unknown pin_id (already removed or never existed)",
        )

    # ------------------------------------------------------------------
    # Rendering pins into context blocks
    # ------------------------------------------------------------------

    def active_pin_blocks(self, *, current_turn: int) -> list[Any]:
        """Return a list of ``ContextBlock`` for pins active at ``current_turn``.

        Late import to avoid a circular dependency with ``contracts.context``
        (which is imported elsewhere in the runtime).
        """
        from arcana.contracts.context import ContextBlock, ContextLayer

        blocks: list[ContextBlock] = []
        for entry in self.pin_state.active_at(current_turn):
            key = f"pin:{entry.pin_id}"
            label = entry.label or "pinned content"
            # Include the label in the rendered block so the LLM sees
            # why it was pinned.
            content = f"[pinned: {label}]\n{entry.content}"
            blocks.append(
                ContextBlock(
                    layer=ContextLayer.WORKING,
                    key=key,
                    content=content,
                    token_count=entry.token_count
                    + _estimate_tokens(f"[pinned: {label}]\n"),
                    priority=1.0,
                    compressible=False,
                    source="cognitive_pin",
                    pinned=True,
                )
            )
        return blocks


__all__ = [
    "COGNITIVE_TOOL_NAMES",
    "COGNITIVE_TOOL_SPECS",
    "CognitiveHandler",
    "PIN_SPEC",
    "PIN_TOOL_NAME",
    "RECALL_SPEC",
    "RECALL_TOOL_NAME",
    "UNPIN_SPEC",
    "UNPIN_TOOL_NAME",
    "is_cognitive_tool",
]
