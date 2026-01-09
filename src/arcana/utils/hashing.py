"""Canonical JSON hashing utilities for consistent digests."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any

from pydantic import BaseModel


def _normalize_value(value: Any) -> Any:
    """Normalize a value for canonical JSON serialization."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        # Normalize floats to 6 decimal places to avoid precision issues
        if value != value:  # NaN check
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return round(value, 6)
    if isinstance(value, Decimal):
        return float(round(value, 6))
    if isinstance(value, BaseModel):
        return _normalize_value(value.model_dump())
    if isinstance(value, dict):
        # Sort keys for consistent ordering
        return {k: _normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    # For other types, convert to string
    return str(value)


def canonical_json(obj: Any) -> str:
    """
    Convert an object to canonical JSON string.

    Properties:
    - Keys are sorted alphabetically
    - No extra whitespace
    - UTF-8 encoding
    - Floats normalized to 6 decimal places
    - Consistent handling of special values (NaN, Infinity)
    """
    normalized = _normalize_value(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def canonical_hash(obj: Any, length: int = 16) -> str:
    """
    Generate a SHA-256 hash of an object's canonical JSON representation.

    Args:
        obj: The object to hash (dict, list, Pydantic model, etc.)
        length: Number of characters to include from the hash (default: 16)

    Returns:
        Truncated hex digest of the SHA-256 hash
    """
    json_str = canonical_json(obj)
    hash_bytes = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    return hash_bytes[:length]


def verify_hash(obj: Any, expected_hash: str, length: int = 16) -> bool:
    """
    Verify that an object matches an expected hash.

    Args:
        obj: The object to verify
        expected_hash: The expected hash value
        length: Number of characters in the hash

    Returns:
        True if the hash matches, False otherwise
    """
    actual_hash = canonical_hash(obj, length)
    return actual_hash == expected_hash
