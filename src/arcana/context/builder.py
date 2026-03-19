"""Working Set Context Builder -- assembles minimal context for each LLM call."""

from __future__ import annotations

from arcana.contracts.context import (
    ContextBlock,
    ContextLayer,
    StepContext,
    TokenBudget,
    WorkingSet,
)
from arcana.contracts.llm import Message, MessageRole
from arcana.contracts.state import AgentState


def estimate_tokens(text: str) -> int:
    """Rough token estimation. ~4 chars per token for English, ~2 for CJK."""
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_count = len(text) - cjk_count
    return (cjk_count // 2) + (other_count // 4) + 1


class WorkingSetBuilder:
    """
    Builds the minimal context for each LLM call.

    The builder follows the four-layer model:
    - Identity: fixed system prompt (always included)
    - Task: current goal (always included)
    - Working: step-specific content (dynamic, priority-sorted)
    - External: on-demand tools/memory (pulled into Working when needed)

    Supports two modes:
    - **Policy mode** (V1): tool descriptions, memory, history as context blocks
    - **Conversation mode** (V2): real message history with budget-aware compression
    """

    def __init__(
        self,
        identity: str,
        token_budget: TokenBudget | None = None,
    ) -> None:
        self._identity = identity
        self._budget = token_budget or TokenBudget()

        # Pre-compute identity block
        self._identity_block = ContextBlock(
            layer=ContextLayer.IDENTITY,
            key="identity",
            content=identity,
            token_count=estimate_tokens(identity),
            priority=1.0,
            compressible=False,
        )

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    # ------------------------------------------------------------------
    # V2 Conversation mode — for ConversationAgent
    # ------------------------------------------------------------------

    def build_conversation_context(
        self,
        messages: list[Message],
        *,
        memory_context: str | None = None,
        tool_token_estimate: int = 0,
    ) -> list[Message]:
        """Build curated message list for a conversation turn.

        Takes the full message history and returns a version that fits within
        the token budget. Strategy:

        1. System message (identity + memory) — always kept
        2. Recent messages (tail) — kept verbatim for coherence
        3. Old messages (middle) — compressed into a summary if over budget

        Args:
            messages: Full conversation history (system + user/assistant/tool)
            memory_context: Optional memory retrieval text to inject into system prompt
            tool_token_estimate: Estimated tokens consumed by tool schemas (not in messages
                but counted against the context window)

        Returns:
            Curated message list fitting within context window.
        """
        # Reserve space for tool schemas — they count against the context window
        budget = self._budget.total_window - self._budget.response_reserve - tool_token_estimate
        total = sum(estimate_tokens(m.content or "") for m in messages)

        # Inject memory into system prompt if provided and not already there
        if memory_context and messages and messages[0].role in (MessageRole.SYSTEM, "system"):
            sys_content = messages[0].content or ""
            if memory_context not in sys_content:
                messages = [
                    Message(role=MessageRole.SYSTEM, content=sys_content + "\n\n" + memory_context),
                    *messages[1:],
                ]
                total = sum(estimate_tokens(m.content or "") for m in messages)

        # Under budget — pass through
        if total <= budget:
            return messages

        # Over budget — compress middle, keep head + tail
        keep_head = 1  # system prompt
        keep_tail = min(len(messages) - keep_head, 6)

        head = messages[:keep_head]
        tail = messages[-keep_tail:] if keep_tail > 0 else []
        middle = messages[keep_head: len(messages) - keep_tail] if keep_tail > 0 else messages[keep_head:]

        head_tokens = sum(estimate_tokens(m.content or "") for m in head)
        tail_tokens = sum(estimate_tokens(m.content or "") for m in tail)
        summary_budget = budget - head_tokens - tail_tokens - 100  # margin

        if summary_budget <= 0 or not middle:
            return head + tail

        # Compress middle into summary
        summary_lines = []
        for msg in middle:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            content = msg.content or ""
            if len(content) > 150:
                content = content[:150] + "..."
            summary_lines.append(f"[{role}] {content}")

        summary_text = "[Earlier conversation summary]\n" + "\n".join(summary_lines)

        # Truncate summary if too long
        while estimate_tokens(summary_text) > summary_budget and len(summary_lines) > 1:
            summary_lines.pop(0)
            summary_text = "[Earlier conversation summary]\n" + "\n".join(summary_lines)

        summary_msg = Message(role=MessageRole.USER, content=summary_text)
        return head + [summary_msg] + tail

    # ------------------------------------------------------------------
    # V1 Policy mode — for Agent + AdaptivePolicy
    # ------------------------------------------------------------------

    def build(
        self,
        state: AgentState,
        step_context: StepContext,
        *,
        tool_descriptions: str | None = None,
        memory_results: str | None = None,
        recent_history: list[dict] | None = None,  # type: ignore[type-arg]
    ) -> WorkingSet:
        """Build the working set for a single LLM call (V1 policy mode)."""

        # Layer 0: Identity (always)
        identity = self._identity_block

        # Layer 1: Task
        task_content = f"Goal: {state.goal or 'No goal specified'}"
        if step_context.focus_instruction:
            task_content += f"\nFocus: {step_context.focus_instruction}"
        task = ContextBlock(
            layer=ContextLayer.TASK,
            key="task",
            content=task_content,
            token_count=estimate_tokens(task_content),
            priority=1.0,
            compressible=False,
        )

        # Layer 2: Working (dynamic)
        working_blocks: list[ContextBlock] = []
        remaining = self._budget.working_budget

        # Error context (highest priority)
        if step_context.previous_error:
            error_content = step_context.previous_error.get(
                "recovery_prompt", str(step_context.previous_error)
            )
            block = ContextBlock(
                layer=ContextLayer.WORKING,
                key="error_context",
                content=error_content,
                token_count=estimate_tokens(error_content),
                priority=0.95,
            )
            working_blocks.append(block)

        # Tool descriptions
        if step_context.needs_tools and tool_descriptions:
            block = ContextBlock(
                layer=ContextLayer.WORKING,
                key="tools",
                content=tool_descriptions,
                token_count=estimate_tokens(tool_descriptions),
                priority=0.8,
            )
            working_blocks.append(block)

        # Recent history (compressed)
        if recent_history:
            history_content = self._format_history(recent_history, max_entries=5)
            block = ContextBlock(
                layer=ContextLayer.WORKING,
                key="history",
                content=history_content,
                token_count=estimate_tokens(history_content),
                priority=0.6,
                compressible=True,
            )
            working_blocks.append(block)

        # Memory results
        if step_context.needs_memory and memory_results:
            block = ContextBlock(
                layer=ContextLayer.WORKING,
                key="memory",
                content=memory_results,
                token_count=estimate_tokens(memory_results),
                priority=0.5,
                compressible=True,
            )
            working_blocks.append(block)

        # Pack by priority (highest first), drop what doesn't fit
        working_blocks.sort(key=lambda b: b.priority, reverse=True)
        final_blocks = []
        dropped = []

        for block in working_blocks:
            if remaining >= block.token_count:
                final_blocks.append(block)
                remaining -= block.token_count
            else:
                dropped.append(block.key)

        total_tokens = (
            identity.token_count
            + task.token_count
            + sum(b.token_count for b in final_blocks)
        )

        return WorkingSet(
            identity=identity,
            task=task,
            working_blocks=final_blocks,
            total_tokens=total_tokens,
            dropped_keys=dropped,
        )

    def to_messages(self, working_set: WorkingSet) -> list[Message]:
        """Convert WorkingSet to LLM messages (V1 policy mode)."""
        messages = []

        # System message: identity + task
        system_content = working_set.identity.content
        if working_set.task.content:
            system_content += "\n\n" + working_set.task.content

        messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=system_content,
            )
        )

        # Working blocks as user context
        for block in working_set.working_blocks:
            if block.key == "history":
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=f"[Context: {block.key}]\n{block.content}",
                    )
                )
            else:
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=f"[{block.key}]\n{block.content}",
                    )
                )

        return messages

    def _format_history(
        self,
        history: list[dict],  # type: ignore[type-arg]
        max_entries: int = 5,
    ) -> str:
        """Format recent history into a compact string."""
        recent = history[-max_entries:] if len(history) > max_entries else history
        lines = []
        for entry in recent:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
