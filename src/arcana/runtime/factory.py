"""Factory functions for convenient agent setup."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arcana.contracts.llm import Budget
from arcana.contracts.runtime import RuntimeConfig
from arcana.runtime.agent import Agent
from arcana.runtime.policies.react import ReActPolicy
from arcana.runtime.reducers.default import DefaultReducer

if TYPE_CHECKING:
    from arcana.gateway.budget import BudgetTracker
    from arcana.gateway.registry import ModelGatewayRegistry
    from arcana.runtime.hooks.base import RuntimeHook
    from arcana.runtime.policies.base import BasePolicy
    from arcana.runtime.reducers.base import BaseReducer
    from arcana.trace.writer import TraceWriter


def create_agent(
    *,
    gateway: ModelGatewayRegistry,
    policy: BasePolicy | None = None,
    reducer: BaseReducer | None = None,
    trace_writer: TraceWriter | None = None,
    budget: Budget | None = None,
    budget_tracker: BudgetTracker | None = None,
    config: RuntimeConfig | None = None,
    hooks: list[RuntimeHook] | None = None,
) -> Agent:
    """
    Create an agent with sensible defaults.

    Args:
        gateway: Model gateway (required)
        policy: Policy to use (default: ReActPolicy)
        reducer: Reducer to use (default: DefaultReducer)
        trace_writer: Optional trace writer
        budget: Optional budget constraints (creates BudgetTracker)
        budget_tracker: Optional existing budget tracker (overrides budget)
        config: Optional runtime configuration
        hooks: Optional runtime hooks

    Returns:
        Configured Agent instance
    """
    # Create budget tracker if budget provided and no tracker given
    tracker = budget_tracker
    if tracker is None and budget is not None:
        from arcana.gateway.budget import BudgetTracker

        tracker = BudgetTracker.from_budget(budget)

    return Agent(
        policy=policy or ReActPolicy(),
        reducer=reducer or DefaultReducer(),
        gateway=gateway,
        config=config or RuntimeConfig(),
        trace_writer=trace_writer,
        budget_tracker=tracker,
        hooks=hooks or [],
    )


def create_react_agent(
    gateway: ModelGatewayRegistry,
    **kwargs: object,
) -> Agent:
    """
    Create an agent using ReAct policy.

    Args:
        gateway: Model gateway (required)
        **kwargs: Additional arguments passed to create_agent

    Returns:
        Agent configured with ReAct policy
    """
    return create_agent(
        gateway=gateway,
        policy=ReActPolicy(),
        **kwargs,  # type: ignore[arg-type]
    )
