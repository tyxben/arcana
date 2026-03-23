"""Working Set Context Builder -- assembles minimal context for each LLM call."""

from __future__ import annotations

import logging
import re
from typing import Any

from arcana.contracts.context import (
    ContextBlock,
    ContextDecision,
    ContextLayer,
    StepContext,
    TokenBudget,
    WorkingSet,
)
from arcana.contracts.llm import Message, MessageRole
from arcana.contracts.state import AgentState

logger = logging.getLogger(__name__)


def _content_text(content: str | list[Any] | None) -> str:
    """Extract plain text from Message.content (handles str, list[ContentBlock], None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # list[ContentBlock] — join text blocks
    return " ".join(
        block.text for block in content if hasattr(block, "text") and block.text
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimation. ~4 chars per token for English, ~2 for CJK."""
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_count = len(text) - cjk_count
    return (cjk_count // 2) + (other_count // 4) + 1


def _extract_keywords(text: str) -> set[str]:
    """Extract simple keywords from text for relevance matching."""
    words = re.findall(r"[a-zA-Z_]\w{2,}", text.lower())
    return set(words)


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

    Every call to build_conversation_context() produces a ContextDecision
    accessible via `last_decision`, explaining why context was composed this way.
    """

    def __init__(
        self,
        identity: str,
        token_budget: TokenBudget | None = None,
        goal: str | None = None,
    ) -> None:
        self._identity = identity
        self._budget = token_budget or TokenBudget()
        self._goal_keywords = _extract_keywords(goal) if goal else set()

        # Pre-compute identity block
        self._identity_block = ContextBlock(
            layer=ContextLayer.IDENTITY,
            key="identity",
            content=identity,
            token_count=estimate_tokens(identity),
            priority=1.0,
            compressible=False,
        )

        # Last context decision — set after each build_conversation_context()
        self.last_decision: ContextDecision | None = None

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    def set_goal(self, goal: str) -> None:
        """Update goal keywords for relevance scoring."""
        self._goal_keywords = _extract_keywords(goal)

    # ------------------------------------------------------------------
    # V2 Conversation mode — for ConversationAgent
    # ------------------------------------------------------------------

    def build_conversation_context(
        self,
        messages: list[Message],
        *,
        memory_context: str | None = None,
        tool_token_estimate: int = 0,
        turn: int = 0,
    ) -> list[Message]:
        """Build curated message list for a conversation turn.

        Strategy:
        1. System message (identity + memory) — always kept
        2. Score middle messages by relevance to goal
        3. Recent messages (tail) — kept verbatim for coherence
        4. Old messages — compress with relevance-aware detail levels

        Produces a ContextDecision accessible via self.last_decision.
        """
        # Apply per-layer budget caps
        effective_tool_tokens = tool_token_estimate
        if self._budget.tool_budget is not None:
            effective_tool_tokens = min(tool_token_estimate, self._budget.tool_budget)

        budget = self._budget.total_window - self._budget.response_reserve - effective_tool_tokens

        # Apply memory budget cap
        memory_injected = False
        if memory_context:
            mem_tokens = estimate_tokens(memory_context)
            if self._budget.memory_budget is not None and mem_tokens > self._budget.memory_budget:
                # Truncate memory to fit budget
                ratio = self._budget.memory_budget / max(mem_tokens, 1)
                char_limit = int(len(memory_context) * ratio)
                memory_context = memory_context[:char_limit] + "\n[memory truncated]"

        total = sum(estimate_tokens(_content_text(m.content)) for m in messages)
        messages_in = len(messages)

        # Inject memory into system prompt if provided and not already there
        if memory_context and messages and messages[0].role in (MessageRole.SYSTEM, "system"):
            sys_content = _content_text(messages[0].content)
            if memory_context not in sys_content:
                messages = [
                    Message(role=MessageRole.SYSTEM, content=sys_content + "\n\n" + memory_context),
                    *messages[1:],
                ]
                total = sum(estimate_tokens(_content_text(m.content)) for m in messages)
                memory_injected = True

        # Check if history budget cap forces compression even if under total budget
        effective_budget = budget
        if self._budget.history_budget is not None and messages:
            sys_tokens = estimate_tokens(_content_text(messages[0].content))
            history_tokens = total - sys_tokens
            if history_tokens > self._budget.history_budget:
                effective_budget = sys_tokens + self._budget.history_budget

        # Under budget — pass through
        if total <= effective_budget:
            self.last_decision = ContextDecision(
                turn=turn,
                budget_total=budget,
                budget_used=total,
                budget_tools=effective_tool_tokens,
                budget_reserve=self._budget.response_reserve,
                messages_in=messages_in,
                messages_out=len(messages),
                memory_injected=memory_injected,
                explanation=f"Under budget ({total}/{budget} tokens), all {len(messages)} messages kept",
            )
            return messages

        # Over budget — compress with relevance awareness
        return self._compress_with_relevance(
            messages,
            budget=effective_budget,
            turn=turn,
            messages_in=messages_in,
            effective_tool_tokens=effective_tool_tokens,
            memory_injected=memory_injected,
        )

    def _compress_with_relevance(
        self,
        messages: list[Message],
        *,
        budget: int,
        turn: int,
        messages_in: int,
        effective_tool_tokens: int,
        memory_injected: bool,
    ) -> list[Message]:
        """Compress messages using relevance-aware strategy.

        Instead of blindly truncating middle messages to 150 chars each,
        we score them and give more detail to relevant ones.
        """
        keep_head = 1  # system prompt
        keep_tail = min(len(messages) - keep_head, 6)

        head = messages[:keep_head]
        tail = messages[-keep_tail:] if keep_tail > 0 else []
        middle = messages[keep_head: len(messages) - keep_tail] if keep_tail > 0 else messages[keep_head:]

        head_tokens = sum(estimate_tokens(_content_text(m.content)) for m in head)
        tail_tokens = sum(estimate_tokens(_content_text(m.content)) for m in tail)
        summary_budget = budget - head_tokens - tail_tokens - 100  # margin

        compressed_descs: list[str] = []
        for msg in middle:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            content = _content_text(msg.content)
            trunc = content[:80] + "..." if len(content) > 80 else content
            compressed_descs.append(f"{role}:{trunc}")

        if summary_budget <= 0 or not middle:
            self.last_decision = ContextDecision(
                turn=turn,
                budget_total=budget,
                budget_used=head_tokens + tail_tokens,
                budget_tools=effective_tool_tokens,
                budget_reserve=self._budget.response_reserve,
                messages_in=messages_in,
                messages_out=len(head) + len(tail),
                compressed_count=len(middle),
                memory_injected=memory_injected,
                history_compressed=True,
                compressed_messages=compressed_descs,
                explanation=f"No budget for summary, {len(middle)} messages dropped",
            )
            return head + tail

        # Score middle messages for relevance
        scored = [(msg, self._relevance_score(msg)) for msg in middle]

        # Build relevance-aware summary
        summary_lines = []
        for msg, score in scored:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            content = _content_text(msg.content)

            # High relevance: keep more detail
            if score >= 0.6:
                char_limit = 300
            elif score >= 0.3:
                char_limit = 150
            else:
                char_limit = 60

            if len(content) > char_limit:
                content = content[:char_limit] + "..."
            summary_lines.append(f"[{role}] {content}")

        summary_text = "[Earlier conversation — relevance-compressed]\n" + "\n".join(summary_lines)

        # Truncate lowest-relevance lines first if still over budget
        while estimate_tokens(summary_text) > summary_budget and len(summary_lines) > 1:
            # Remove the line corresponding to the lowest-scored message
            min_idx = 0
            min_score = 999.0
            for i, (_, s) in enumerate(scored[:len(summary_lines)]):
                if s < min_score:
                    min_score = s
                    min_idx = i
            summary_lines.pop(min_idx)
            scored.pop(min_idx)
            summary_text = "[Earlier conversation — relevance-compressed]\n" + "\n".join(summary_lines)

        summary_msg = Message(role=MessageRole.USER, content=summary_text)
        result = head + [summary_msg] + tail

        used_tokens = sum(estimate_tokens(_content_text(m.content)) for m in result)
        original_middle_tokens = sum(estimate_tokens(_content_text(m.content)) for m in middle)
        compressed_tokens = estimate_tokens(summary_text)
        ratio = original_middle_tokens / max(compressed_tokens, 1)

        self.last_decision = ContextDecision(
            turn=turn,
            budget_total=budget,
            budget_used=used_tokens,
            budget_tools=effective_tool_tokens,
            budget_reserve=self._budget.response_reserve,
            messages_in=messages_in,
            messages_out=len(result),
            compressed_count=len(middle),
            memory_injected=memory_injected,
            history_compressed=True,
            compressed_messages=compressed_descs,
            explanation=(
                f"{len(middle)} messages compressed ({ratio:.1f}x), "
                f"budget {used_tokens}/{budget} tokens ({used_tokens * 100 // budget}% full)"
                if budget > 0
                else f"{len(middle)} messages compressed ({ratio:.1f}x), budget exhausted"
            ),
        )
        return result

    def _relevance_score(self, msg: Message) -> float:
        """Score a message's relevance to the current goal.

        Higher score = more detail preserved during compression.
        """
        content = _content_text(msg.content).lower()
        role = msg.role if isinstance(msg.role, str) else msg.role.value
        score = 0.0

        # Role-based base score
        if role == "tool":
            score += 0.5  # Tool results often contain key data
        elif role == "assistant":
            score += 0.3
        elif role == "user":
            score += 0.2

        # Keyword overlap with goal
        if self._goal_keywords:
            msg_keywords = _extract_keywords(content)
            overlap = len(self._goal_keywords & msg_keywords)
            score += min(overlap * 0.1, 0.4)

        # Error/diagnosis content is always important
        if "error" in content or "failed" in content or "recovery" in content:
            score += 0.3

        return min(score, 1.0)

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
