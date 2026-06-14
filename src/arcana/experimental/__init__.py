"""Experimental, unstable Arcana surfaces.

Everything exported here is **incubating**: the API shape and semantics may
change in any minor release until it graduates to the stable public surface
(``specs/v1.0.0-stability.md`` §1). Import paths under ``arcana.experimental``
are deliberately separate from the top-level ``arcana`` namespace so that
nothing here is mistaken for a semver-committed contract.

Current residents:

- :mod:`arcana.experimental.subagents` -- optional, user-directed subtask
  isolation (``SubagentService`` / ``SubagentResult`` / :func:`subagents`).
  Phase 3 of ``specs/kimi-code-response-roadmap.md``: an explicit facade over
  the existing :class:`~arcana.multi_agent.agent_pool.AgentPool` /
  ``ChatSession`` machinery. There is **no** default main-agent/subagent
  topology, scheduler, or supervisor -- the caller decides who is asked and
  when.
"""

from __future__ import annotations

from arcana.experimental.subagents import (
    SubagentRecursionError,
    SubagentResult,
    SubagentService,
    subagents,
)

__all__ = [
    "SubagentRecursionError",
    "SubagentResult",
    "SubagentService",
    "subagents",
]
