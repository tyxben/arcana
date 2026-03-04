"""Field-level reducers for merging node outputs into graph state."""

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel


def replace_reducer(old: Any, new: Any) -> Any:
    """Replace old value with new value (default reducer)."""
    return new


def append_reducer(old: Any, new: Any) -> Any:
    """Append new value(s) to a list."""
    old_list = old if isinstance(old, list) else ([] if old is None else [old])
    if isinstance(new, list):
        return old_list + new
    return old_list + [new]


def add_reducer(old: Any, new: Any) -> Any:
    """Add numeric values."""
    return (old or 0) + new


def merge_reducer(old: Any, new: Any) -> Any:
    """Merge dictionaries."""
    return {**(old or {}), **(new or {})}


def add_messages(existing: list[Any], new: list[Any] | Any) -> list[Any]:
    """
    Message reducer that appends new messages, deduplicating by id if present.

    Similar to LangGraph's add_messages reducer.
    """
    existing = existing or []
    if not isinstance(new, list):
        new = [new]

    # Build index of existing messages by id for dedup
    existing_by_id: dict[str, int] = {}
    for i, msg in enumerate(existing):
        msg_id = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
        if msg_id is not None:
            existing_by_id[msg_id] = i

    result = list(existing)
    for msg in new:
        msg_id = msg.get("id") if isinstance(msg, dict) else getattr(msg, "id", None)
        if msg_id is not None and msg_id in existing_by_id:
            # Update existing message
            result[existing_by_id[msg_id]] = msg
        else:
            result.append(msg)

    return result


def extract_reducers(schema: type[BaseModel] | None) -> dict[str, Callable[..., Any]]:
    """
    Extract field-level reducers from a Pydantic model's type annotations.

    Supports ``Annotated[type, reducer_fn]`` syntax::

        class MyState(BaseModel):
            messages: Annotated[list, add_messages] = []
            counter: Annotated[int, add_reducer] = 0
            result: str = ""  # defaults to replace_reducer
    """
    if schema is None:
        return {}

    reducers: dict[str, Callable[..., Any]] = {}
    hints = get_type_hints(schema, include_extras=True)

    for field_name, hint in hints.items():
        if get_origin(hint) is typing.Annotated:
            args = get_args(hint)
            # Look for a callable in the Annotated metadata
            for arg in args[1:]:
                if callable(arg) and not isinstance(arg, type):
                    reducers[field_name] = arg
                    break

    return reducers


def apply_reducers(
    state: dict[str, Any],
    output: dict[str, Any],
    reducers: dict[str, Callable[..., Any]],
) -> dict[str, Any]:
    """Apply reducers to merge node output into state."""
    new_state = dict(state)
    for key, value in output.items():
        if key in reducers:
            new_state[key] = reducers[key](state.get(key), value)
        else:
            # Default: replace
            new_state[key] = value
    return new_state
