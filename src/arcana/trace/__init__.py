"""Trace system for Arcana - JSONL-based event logging."""

from arcana.trace.reader import TraceReader
from arcana.trace.writer import TraceWriter

__all__ = ["TraceWriter", "TraceReader"]
