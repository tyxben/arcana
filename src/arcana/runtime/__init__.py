"""Agent Runtime - execution engine for Arcana agents."""

from arcana.contracts.runtime import (
    PolicyDecision,
    RuntimeConfig,
    StepResult,
    StepType,
)
from arcana.runtime.agent import Agent
from arcana.runtime.exceptions import (
    CheckpointError,
    HashVerificationError,
    PolicyError,
    ProgressStallError,
    RuntimeError,
    StateTransitionError,
    StepExecutionError,
)
from arcana.runtime.factory import create_agent, create_react_agent
from arcana.runtime.hooks import RuntimeHook
from arcana.runtime.policies import BasePolicy, ReActPolicy
from arcana.runtime.progress import ProgressDetector
from arcana.runtime.reducers import BaseReducer, DefaultReducer
from arcana.runtime.state_manager import StateManager
from arcana.runtime.step import StepExecutor

__all__ = [
    # Core
    "Agent",
    "StepExecutor",
    "StateManager",
    "ProgressDetector",
    # Contracts
    "RuntimeConfig",
    "PolicyDecision",
    "StepResult",
    "StepType",
    # Policies
    "BasePolicy",
    "ReActPolicy",
    # Reducers
    "BaseReducer",
    "DefaultReducer",
    # Hooks
    "RuntimeHook",
    # Exceptions
    "RuntimeError",
    "StateTransitionError",
    "CheckpointError",
    "HashVerificationError",
    "StepExecutionError",
    "PolicyError",
    "ProgressStallError",
    # Factory
    "create_agent",
    "create_react_agent",
]
