"""SharedContext -- thread-safe key-value store for agent collaboration."""

from __future__ import annotations

import threading
from typing import Any

_SENTINEL = object()


class SharedContext:
    """Concurrent-safe shared state for collaborating agents.

    Any agent in a collaboration pool can read/write keys.
    No access control -- all agents see everything (Constitution P8:
    "every agent can see what others have said").
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value. Returns *default* if key doesn't exist."""
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value."""
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> bool:
        """Delete a key. Returns ``True`` if existed."""
        with self._lock:
            return self._data.pop(key, _SENTINEL) is not _SENTINEL

    def keys(self) -> list[str]:
        """List all keys."""
        with self._lock:
            return list(self._data.keys())

    def snapshot(self) -> dict[str, Any]:
        """Get a shallow copy of all data."""
        with self._lock:
            return dict(self._data)

    def clear(self) -> None:
        """Remove all data."""
        with self._lock:
            self._data.clear()
