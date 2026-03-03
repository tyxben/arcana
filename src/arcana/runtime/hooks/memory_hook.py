"""MemoryHook — integrates the memory system with the Agent Runtime."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from arcana.contracts.memory import MemoryType, MemoryWriteRequest

if TYPE_CHECKING:
    from arcana.contracts.runtime import StepResult
    from arcana.contracts.state import AgentState
    from arcana.contracts.trace import TraceContext
    from arcana.memory.manager import MemoryManager


class MemoryHook:
    """
    RuntimeHook that syncs working_memory with the MemoryManager.

    - on_run_start: loads persisted working memory into AgentState
    - on_step_complete: persists memory_updates from StepResult
    - on_run_end: promotes flagged entries to long-term on success
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory_manager = memory_manager

    async def on_run_start(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
    ) -> None:
        """Load persisted working memory into state."""
        entries = await self.memory_manager.working.get_all(state.run_id)
        for key, entry in entries.items():
            if key not in state.working_memory:
                state.working_memory[key] = entry.content

    async def on_step_complete(
        self,
        state: AgentState,
        step_result: StepResult,
        trace_ctx: TraceContext,
    ) -> None:
        """Persist memory_updates from the step result."""
        for key, value in step_result.memory_updates.items():
            if value is None:
                await self.memory_manager.working.delete(state.run_id, key)
                continue

            content = value if isinstance(value, str) else json.dumps(value)
            request = MemoryWriteRequest(
                memory_type=MemoryType.WORKING,
                key=key,
                content=content,
                confidence=1.0,  # Step results are trusted
                source="step_result",
                run_id=state.run_id,
                step_id=step_result.step_id,
            )
            await self.memory_manager.write(request)

    async def on_run_end(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
    ) -> None:
        """Promote flagged working memory to long-term on successful completion."""
        if state.status.value != "completed":
            return

        entries = await self.memory_manager.working.get_all(state.run_id)
        for _key, entry in entries.items():
            if entry.metadata.get("promote_to_long_term"):
                lt_request = MemoryWriteRequest(
                    memory_type=MemoryType.LONG_TERM,
                    key=entry.key,
                    content=entry.content,
                    confidence=entry.confidence,
                    source=f"promoted_from_working:{state.run_id}",
                    run_id=state.run_id,
                    tags=[*entry.tags, "promoted"],
                )
                await self.memory_manager.write(lt_request)

    async def on_checkpoint(
        self,
        state: AgentState,
        trace_ctx: TraceContext,
    ) -> None:
        """No-op — working memory is already persisted per-step."""

    async def on_error(
        self,
        state: AgentState,
        error: Exception,
        trace_ctx: TraceContext,
    ) -> None:
        """No-op for now."""
