"""
Memory system for Arcana.

Two tiers:

  Runtime memory (built-in)
  ─────────────────────────
  RunMemoryStore — lightweight cross-run fact retrieval.
  Enabled via ``Runtime(memory=True)``. Lives and dies with the process.

  Advanced memory (composable)
  ────────────────────────────
  MemoryManager + WorkingMemoryStore / LongTermMemoryStore / EpisodicMemoryStore
  Application-layer modules you compose yourself when you need governed,
  multi-tier memory with revocation and episodic trace.

  See ``examples/11_advanced_memory.py`` for usage.
"""

# ── Runtime memory (default) ────────────────────────────────────
# ── Advanced memory (composable) ────────────────────────────────
from arcana.memory.episodic import EpisodicMemoryStore
from arcana.memory.governance import WritePolicy
from arcana.memory.long_term import LongTermMemoryStore
from arcana.memory.manager import MemoryManager
from arcana.memory.run_memory import RunMemoryStore
from arcana.memory.working import WorkingMemoryStore

__all__ = [
    # Runtime
    "RunMemoryStore",
    # Advanced
    "EpisodicMemoryStore",
    "LongTermMemoryStore",
    "MemoryManager",
    "WorkingMemoryStore",
    "WritePolicy",
]
