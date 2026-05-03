"""Multi-agent — name-addressed collaboration primitives.

Post-Amendment-3 cleanup (2026-05-03): the role-addressed
``TeamOrchestrator`` / ``RoleConfig`` / ``MessageBus`` were removed in
the same release that deprecated them, on the grounds that
``arcana.multi_agent.*`` is internal-not-stable per
``specs/v1.0.0-stability.md`` §2 and the deprecated classes had not yet
been published to PyPI in any release. Use ``runtime.collaborate()``
plus ``Channel`` for name-addressed pub/sub.
"""

from arcana.contracts.multi_agent import (
    ChannelMessage,
    MessageType,
)
from arcana.multi_agent.agent_pool import AgentPool
from arcana.multi_agent.channel import Channel
from arcana.multi_agent.shared_context import SharedContext

__all__ = [
    # Contracts
    "ChannelMessage",
    "MessageType",
    # Primitives
    "AgentPool",
    "Channel",
    "SharedContext",
]
