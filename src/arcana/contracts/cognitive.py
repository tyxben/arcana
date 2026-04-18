"""Cognitive primitive contracts (v0.7.0).

The cognitive primitives are intercepted tools the LLM may call to operate
on its own reasoning state. See ``specs/v0.7.0-cognitive-primitives.md`` and
``CONSTITUTION.md`` Principle 9 for the constitutional argument.

The runtime services these tools without going through ``ToolGateway`` (same
pattern as ``ask_user``). The framework never calls them on the LLM's behalf.

Contracts in this module:

- ``RecallRequest`` / ``RecallResult`` — retrieve full-fidelity content of an
  earlier turn, bypassing any working-set compression.
- ``PinRequest`` / ``PinResult`` — protect specified content from compression
  in future working sets.
- ``UnpinRequest`` / ``UnpinResult`` — remove a previously created pin.
- ``PinEntry`` — a single active pin tracked by the runtime.
- ``PinState`` — session-local registry of active pins.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class RecallRequest(BaseModel):
    """Request to recall an earlier turn.

    ``turn`` is 1-indexed — the first user-visible turn is ``1``.
    """

    turn: int
    include: Literal["all", "assistant_only", "tool_calls"] = "all"


class RecallResult(BaseModel):
    """Structured recall result.

    On success, ``found=True`` and ``messages`` contains the recovered
    messages in trace order (each a role/content dict, optionally with
    ``tool_calls``). On failure, ``found=False`` and ``note`` carries a
    human-readable, actionable explanation the LLM can reason about.

    Errors are never raised as exceptions — they are always returned as
    structured tool results (Principle 5).
    """

    turn: int
    found: bool
    messages: list[dict[str, Any]] = Field(default_factory=list)
    note: str | None = None


# ---------------------------------------------------------------------------
# Pin / Unpin
# ---------------------------------------------------------------------------


class PinRequest(BaseModel):
    """Request to pin content against future compression."""

    content: str
    label: str | None = None
    until_turn: int | None = None


class PinResult(BaseModel):
    """Structured pin result.

    On success: ``pinned=True`` and ``pin_id`` is the opaque handle the LLM
    uses with ``unpin``. ``already_pinned`` is true when the same content was
    already pinned (idempotent no-op).

    On refusal: ``pinned=False`` and the remaining fields explain why. The
    framework never auto-unpins; the LLM decides whether to unpin something
    else, shrink the content, or proceed without pinning.
    """

    pinned: bool
    pin_id: str | None = None
    label: str | None = None
    until_turn: int | None = None
    already_pinned: bool = False

    # Populated on refusal
    reason: str | None = None
    current_pin_tokens: int | None = None
    requested_tokens: int | None = None
    cap: int | None = None
    suggestion: str | None = None


class UnpinRequest(BaseModel):
    """Request to remove a pin by pin_id."""

    pin_id: str


class UnpinResult(BaseModel):
    """Structured unpin result."""

    unpinned: bool
    pin_id: str
    note: str | None = None


# ---------------------------------------------------------------------------
# Pin state (session-local)
# ---------------------------------------------------------------------------


def _content_hash(content: str) -> str:
    """SHA-256 hash of a pin's content, used for idempotency."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class PinEntry(BaseModel):
    """A single active pin.

    ``content_hash`` is the canonical identity of the pin for idempotency
    checks. ``pin_id`` is a stable opaque handle exposed to the LLM for
    ``unpin`` calls.
    """

    pin_id: str
    content: str
    content_hash: str
    label: str | None = None
    until_turn: int | None = None
    created_turn: int = 0
    token_count: int = 0


class PinState(BaseModel):
    """Session-local registry of active pins.

    Pins are scoped to a single session / ``ChatSession``. Runtime pools do
    not share pins across agents. This model is mutable — the cognitive
    handler mutates ``entries`` directly.

    Methods here are pure bookkeeping. Budget enforcement and ID minting
    live in ``runtime/cognitive.py`` so this contract stays lightweight.
    """

    entries: list[PinEntry] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find_by_hash(self, content_hash: str) -> PinEntry | None:
        """Return the pin with this content hash, or None."""
        for entry in self.entries:
            if entry.content_hash == content_hash:
                return entry
        return None

    def find_by_id(self, pin_id: str) -> PinEntry | None:
        """Return the pin with this pin_id, or None."""
        for entry in self.entries:
            if entry.pin_id == pin_id:
                return entry
        return None

    def active_at(self, turn: int) -> list[PinEntry]:
        """Pins that are still active at the given turn.

        A pin is active if ``until_turn`` is None or ``until_turn >= turn``.
        The framework never auto-unpins; expired pins are simply excluded
        from rendering, but remain in ``entries`` for trace visibility.
        """
        out: list[PinEntry] = []
        for entry in self.entries:
            if entry.until_turn is None or entry.until_turn >= turn:
                out.append(entry)
        return out

    def total_tokens(self) -> int:
        """Total tokens across all pins currently in the registry."""
        return sum(e.token_count for e in self.entries)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, entry: PinEntry) -> None:
        self.entries.append(entry)

    def remove(self, pin_id: str) -> bool:
        """Remove the pin with this id. Returns True if something was removed."""
        for i, entry in enumerate(self.entries):
            if entry.pin_id == pin_id:
                self.entries.pop(i)
                return True
        return False


__all__ = [
    "PinEntry",
    "PinRequest",
    "PinResult",
    "PinState",
    "RecallRequest",
    "RecallResult",
    "UnpinRequest",
    "UnpinResult",
]
