"""Working Set Context Builder -- assembles minimal context for each LLM call."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from arcana.contracts.context import (
    ContextBlock,
    ContextDecision,
    ContextLayer,
    ContextReport,
    ContextStrategy,
    StepContext,
    TokenBudget,
    WorkingSet,
)
from arcana.contracts.llm import Message, MessageRole, ModelConfig
from arcana.contracts.state import AgentState

if TYPE_CHECKING:
    from arcana.gateway.registry import ModelGatewayRegistry

logger = logging.getLogger(__name__)

# Minimum token count in the middle section to justify an LLM compression call.
# Below this threshold, keyword-based compression is sufficient.
_LLM_COMPRESSION_THRESHOLD = 2000


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
        gateway: ModelGatewayRegistry | None = None,
        compression_model: ModelConfig | None = None,
        strategy: ContextStrategy | None = None,
    ) -> None:
        self._identity = identity
        self._budget = token_budget or TokenBudget()
        self._goal = goal
        self._goal_keywords = _extract_keywords(goal) if goal else set()
        self._gateway = gateway
        self._compression_model = compression_model
        self._strategy = strategy or ContextStrategy()

        # Track whether memory has been injected into system prompt
        self._memory_injected: bool = False

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

        # Last context report — richer than ContextDecision, for v0.3.0 visibility
        self.last_report: ContextReport | None = None

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    def set_goal(self, goal: str) -> None:
        """Update goal keywords for relevance scoring."""
        self._goal = goal
        self._goal_keywords = _extract_keywords(goal)
        self._memory_injected = False

    # ------------------------------------------------------------------
    # Phase 0: Zero-cost tool result pruning
    # ------------------------------------------------------------------

    def _prune_stale_tool_results(
        self,
        messages: list[Message],
        *,
        current_turn: int,
    ) -> tuple[list[Message], int, int]:
        """Zero-cost pruning: replace old tool results with summary placeholders.

        Rules:
        - Only prune messages with role "tool"
        - Only prune if the message is in the "stale" region (before the recent tail)
        - The recent tail is defined as the last ``staleness_turns * 3`` messages
          (each turn typically has ~3 messages: user/assistant/tool)
        - Do NOT prune tool messages containing "error" or "failed"
        - Replace pruned messages with a shorter version preserving:
          - First N chars of content (configurable via tool_result_prune_max_chars)
          - A header: "[tool result pruned, was ~{original_tokens} tokens]"

        Returns:
            (pruned_messages, count_pruned, tokens_saved)
        """
        staleness_turns = self._strategy.tool_result_staleness_turns
        max_chars = self._strategy.tool_result_prune_max_chars

        # Recent tail boundary: messages within this index range are "recent"
        recent_count = staleness_turns * 3
        stale_boundary = len(messages) - recent_count

        if stale_boundary <= 0:
            # All messages are recent — nothing to prune
            return messages, 0, 0

        result: list[Message] = []
        count_pruned = 0
        tokens_saved = 0

        for i, msg in enumerate(messages):
            # Only prune messages in the stale region
            if i >= stale_boundary:
                result.append(msg)
                continue

            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role != "tool":
                result.append(msg)
                continue

            content = _content_text(msg.content)
            content_lower = content.lower()

            # Never prune error/failure results
            if "error" in content_lower or "failed" in content_lower:
                result.append(msg)
                continue

            # Prune: replace with summary placeholder
            original_tokens = msg.token_count
            preview = content[:max_chars]
            pruned_content = (
                f"[tool result pruned, was ~{original_tokens} tokens]\n{preview}"
            )
            pruned_msg = Message(
                role=msg.role,
                content=pruned_content,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
            )
            pruned_tokens = pruned_msg.token_count
            saved = max(0, original_tokens - pruned_tokens)

            count_pruned += 1
            tokens_saved += saved
            result.append(pruned_msg)

        return result, count_pruned, tokens_saved

    # ------------------------------------------------------------------
    # V2 Conversation mode — for ConversationAgent
    # ------------------------------------------------------------------

    def _resolve_strategy_name(self, utilization: float) -> str:
        """Resolve the compression strategy name based on mode and utilization."""
        mode = self._strategy.mode
        if mode == "off":
            return "passthrough"
        if mode == "always_compress":
            return "tail_preserve"
        # adaptive mode (default)
        if utilization < self._strategy.passthrough_threshold:
            return "passthrough"
        if utilization < self._strategy.tail_preserve_threshold:
            return "tail_preserve"
        if utilization < self._strategy.llm_summarize_threshold:
            return "llm_summarize"
        return "aggressive_truncate"

    def _build_context_report(
        self,
        *,
        turn: int,
        strategy_name: str,
        curated: list[Message],
        messages: list[Message],
        tool_token_estimate: int,
        memory_context: str | None,
        memory_injected: bool,
        messages_in: int,
    ) -> ContextReport:
        """Build a ContextReport from the curated messages."""
        identity_tokens = curated[0].token_count if curated else 0
        history_tokens = sum(m.token_count for m in curated[1:])
        memory_tok = estimate_tokens(memory_context) if memory_context and memory_injected else 0
        original_total = sum(m.token_count for m in messages)
        curated_total = sum(m.token_count for m in curated)

        # Count fidelity distribution from curated messages
        fidelity_dist: dict[str, int] = {}
        for m in curated:
            level = getattr(m, "_fidelity", None)
            if level is not None:
                key = f"L{level}"
                fidelity_dist[key] = fidelity_dist.get(key, 0) + 1

        return ContextReport(
            turn=turn,
            strategy_used=strategy_name,
            total_tokens=curated_total + tool_token_estimate,
            identity_tokens=identity_tokens,
            task_tokens=0,
            tools_tokens=tool_token_estimate,
            history_tokens=history_tokens,
            memory_tokens=memory_tok,
            compression_applied=(strategy_name != "passthrough"),
            compression_savings=max(0, original_total - curated_total),
            messages_compressed=messages_in - len(curated),
            window_size=self._budget.total_window,
            utilization=(curated_total + tool_token_estimate) / max(self._budget.total_window, 1),
            fidelity_distribution=fidelity_dist,
        )

    def _aggressive_truncate(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Keep system message + last N user/assistant turn pairs only."""
        keep_turns = self._strategy.aggressive_keep_turns
        if not messages:
            return messages

        # Always keep system message (first)
        head = messages[:1]
        rest = messages[1:]

        # Collect recent turn pairs from the end
        # A "turn" is a user message + assistant response (possibly with tool messages)
        # Walk backwards and keep the last keep_turns worth of messages
        kept: list[Message] = []
        turns_seen = 0
        for msg in reversed(rest):
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            kept.append(msg)
            if role == "user":
                turns_seen += 1
                if turns_seen >= keep_turns:
                    break

        kept.reverse()
        return head + kept

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

        Produces a ContextDecision accessible via self.last_decision and
        a ContextReport accessible via self.last_report.
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

        total = sum(m.token_count for m in messages)
        messages_in = len(messages)

        # Inject memory into the first user message (keeps system prompt stable for cache,
        # avoids back-to-back user messages that Anthropic/DeepSeek reject)
        if memory_context and not self._memory_injected and messages:
            for _mi, _mm in enumerate(messages):
                if _mm.role in (MessageRole.USER, "user"):
                    merged = f"[Run context]\n{memory_context}\n\n{_content_text(_mm.content)}"
                    messages = [*messages[:_mi], Message(role=MessageRole.USER, content=merged), *messages[_mi + 1:]]
                    break
            else:
                # No user message yet — insert after system
                messages = [messages[0], Message(role=MessageRole.USER, content=f"[Run context]\n{memory_context}"), *messages[1:]]
            total = sum(m.token_count for m in messages)
            memory_injected = True
            self._memory_injected = True

        # Phase 0: prune stale tool results (zero-cost, before compression)
        messages, p0_pruned, p0_saved = self._prune_stale_tool_results(
            messages, current_turn=turn,
        )
        if p0_pruned > 0:
            total = sum(m.token_count for m in messages)

        # Check if history budget cap forces compression even if under total budget
        effective_budget = budget
        if self._budget.history_budget is not None and messages:
            sys_tokens = messages[0].token_count
            history_tokens = total - sys_tokens
            if history_tokens > self._budget.history_budget:
                effective_budget = sys_tokens + self._budget.history_budget

        # Compute utilization for strategy selection
        utilization = total / max(self._budget.total_window, 1)
        strategy_name = self._resolve_strategy_name(utilization)

        # Under budget — pass through (unless strategy forces compression)
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
            report = self._build_context_report(
                turn=turn,
                strategy_name="passthrough",
                curated=messages,
                messages=messages,
                tool_token_estimate=effective_tool_tokens,
                memory_context=memory_context,
                memory_injected=memory_injected,
                messages_in=messages_in,
            )
            report.tool_results_pruned = p0_pruned
            report.tool_results_tokens_saved = p0_saved
            self.last_report = report
            return messages

        # Over budget — try tail_preserve first (relevance compression),
        # then fall back to aggressive_truncate if result is still too large.
        effective_strategy = strategy_name if strategy_name != "passthrough" else "tail_preserve"
        result = self._compress_with_relevance(
            messages,
            budget=effective_budget,
            turn=turn,
            messages_in=messages_in,
            effective_tool_tokens=effective_tool_tokens,
            memory_injected=memory_injected,
        )

        # If strategy says aggressive_truncate and relevance compression is still over budget,
        # fall back to aggressive truncation
        if strategy_name == "aggressive_truncate":
            result_total = sum(m.token_count for m in result)
            if result_total > effective_budget:
                curated = self._aggressive_truncate(messages)
                self.last_decision = ContextDecision(
                    turn=turn,
                    budget_total=budget,
                    budget_used=sum(m.token_count for m in curated),
                    budget_tools=effective_tool_tokens,
                    budget_reserve=self._budget.response_reserve,
                    messages_in=messages_in,
                    messages_out=len(curated),
                    memory_injected=memory_injected,
                    history_compressed=True,
                    explanation=f"Aggressive truncate: kept system + last {self._strategy.aggressive_keep_turns} turns",
                )
                report = self._build_context_report(
                    turn=turn,
                    strategy_name="aggressive_truncate",
                    curated=curated,
                    messages=messages,
                    tool_token_estimate=effective_tool_tokens,
                    memory_context=memory_context,
                    memory_injected=memory_injected,
                    messages_in=messages_in,
                )
                report.tool_results_pruned = p0_pruned
                report.tool_results_tokens_saved = p0_saved
                self.last_report = report
                return curated

        report = self._build_context_report(
            turn=turn,
            strategy_name=effective_strategy,
            curated=result,
            messages=messages,
            tool_token_estimate=effective_tool_tokens,
            memory_context=memory_context,
            memory_injected=memory_injected,
            messages_in=messages_in,
        )
        report.tool_results_pruned = p0_pruned
        report.tool_results_tokens_saved = p0_saved
        self.last_report = report
        return result

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
        keep_tail = min(len(messages) - keep_head, self._strategy.tail_preserve_keep_recent)

        head = messages[:keep_head]
        tail = messages[-keep_tail:] if keep_tail > 0 else []
        middle = messages[keep_head: len(messages) - keep_tail] if keep_tail > 0 else messages[keep_head:]

        head_tokens = sum(m.token_count for m in head)
        tail_tokens = sum(m.token_count for m in tail)
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

        # Build fidelity-graded messages
        fidelity_msgs: list[Message] = []
        fidelity_counts: dict[str, int] = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}

        for msg, score in scored:
            role = msg.role
            content = _content_text(msg.content)
            role_str = role if isinstance(role, str) else role.value

            if score >= 0.7:
                # L0: keep original (copy to avoid mutating caller's message)
                new_msg = msg.model_copy()
                new_msg._fidelity = 0
                fidelity_counts["L0"] += 1
            elif score >= 0.4:
                # L1: truncate to 300 chars, keep role
                trunc = content[:300] + "..." if len(content) > 300 else content
                new_msg = Message(role=role, content=trunc)
                new_msg._fidelity = 1
                fidelity_counts["L1"] += 1
            elif score >= 0.2:
                # L2: one-line summary
                trunc = content[:80] + "..." if len(content) > 80 else content
                new_msg = Message(role=role, content=f"[compressed] {trunc}")
                new_msg._fidelity = 2
                fidelity_counts["L2"] += 1
            else:
                # L3: tag only
                new_msg = Message(role=MessageRole.USER, content=f"[{role_str}] (earlier message)")
                new_msg._fidelity = 3
                fidelity_counts["L3"] += 1

            fidelity_msgs.append(new_msg)

        # Budget enforcement: demote lowest-scored messages until under budget
        result = head + fidelity_msgs + tail
        while sum(m.token_count for m in result) > budget and fidelity_msgs:
            # Find lowest-scored message that can still be demoted
            min_idx = -1
            min_score = 999.0
            for i, (_, s) in enumerate(scored[:len(fidelity_msgs)]):
                if s < min_score and fidelity_msgs[i]._fidelity is not None and fidelity_msgs[i]._fidelity < 3:
                    min_score = s
                    min_idx = i

            if min_idx == -1:
                # All at L3, drop the lowest-scored one entirely
                min_idx = 0
                min_score = 999.0
                for i, (_, s) in enumerate(scored[:len(fidelity_msgs)]):
                    if s < min_score:
                        min_score = s
                        min_idx = i
                fidelity_msgs.pop(min_idx)
                scored.pop(min_idx)
            else:
                # Demote: increase fidelity level
                fmsg = fidelity_msgs[min_idx]
                orig_msg, _score = scored[min_idx]
                orig_role = orig_msg.role
                orig_role_str = orig_role if isinstance(orig_role, str) else orig_role.value
                orig_content = _content_text(orig_msg.content)

                current_level = fmsg._fidelity or 0
                if current_level == 0:
                    trunc = orig_content[:300] + "..." if len(orig_content) > 300 else orig_content
                    demoted = Message(role=orig_role, content=trunc)
                    demoted._fidelity = 1
                elif current_level == 1:
                    trunc = orig_content[:80] + "..." if len(orig_content) > 80 else orig_content
                    demoted = Message(role=orig_role, content=f"[compressed] {trunc}")
                    demoted._fidelity = 2
                else:  # current_level == 2
                    demoted = Message(role=MessageRole.USER, content=f"[{orig_role_str}] (earlier message)")
                    demoted._fidelity = 3
                fidelity_msgs[min_idx] = demoted

            result = head + fidelity_msgs + tail

        used_tokens = sum(m.token_count for m in result)
        original_middle_tokens = sum(m.token_count for m in middle)
        compressed_tokens = sum(m.token_count for m in fidelity_msgs)
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
                f"{len(middle)} messages fidelity-compressed ({ratio:.1f}x), "
                f"budget {used_tokens}/{budget} tokens ({used_tokens * 100 // budget}% full)"
                if budget > 0
                else f"{len(middle)} messages fidelity-compressed ({ratio:.1f}x), budget exhausted"
            ),
        )
        return result

    # ------------------------------------------------------------------
    # Async conversation mode — uses LLM compression when available
    # ------------------------------------------------------------------

    async def abuild_conversation_context(
        self,
        messages: list[Message],
        *,
        memory_context: str | None = None,
        tool_token_estimate: int = 0,
        turn: int = 0,
    ) -> list[Message]:
        """Async version of build_conversation_context.

        When a gateway is configured and the middle section is large enough,
        uses LLM-based semantic compression instead of keyword truncation.
        Falls back to sync compression if no gateway is set or on failure.
        """
        # Apply per-layer budget caps (identical to sync version)
        effective_tool_tokens = tool_token_estimate
        if self._budget.tool_budget is not None:
            effective_tool_tokens = min(tool_token_estimate, self._budget.tool_budget)

        budget = self._budget.total_window - self._budget.response_reserve - effective_tool_tokens

        # Apply memory budget cap
        memory_injected = False
        if memory_context:
            mem_tokens = estimate_tokens(memory_context)
            if self._budget.memory_budget is not None and mem_tokens > self._budget.memory_budget:
                ratio = self._budget.memory_budget / max(mem_tokens, 1)
                char_limit = int(len(memory_context) * ratio)
                memory_context = memory_context[:char_limit] + "\n[memory truncated]"

        total = sum(m.token_count for m in messages)
        messages_in = len(messages)

        # Inject memory into the first user message (keeps system prompt stable for cache,
        # avoids back-to-back user messages that Anthropic/DeepSeek reject)
        if memory_context and not self._memory_injected and messages:
            for _mi, _mm in enumerate(messages):
                if _mm.role in (MessageRole.USER, "user"):
                    merged = f"[Run context]\n{memory_context}\n\n{_content_text(_mm.content)}"
                    messages = [*messages[:_mi], Message(role=MessageRole.USER, content=merged), *messages[_mi + 1:]]
                    break
            else:
                # No user message yet — insert after system
                messages = [messages[0], Message(role=MessageRole.USER, content=f"[Run context]\n{memory_context}"), *messages[1:]]
            total = sum(m.token_count for m in messages)
            memory_injected = True
            self._memory_injected = True

        # Phase 0: prune stale tool results (zero-cost, before compression)
        messages, p0_pruned, p0_saved = self._prune_stale_tool_results(
            messages, current_turn=turn,
        )
        if p0_pruned > 0:
            total = sum(m.token_count for m in messages)

        # Check if history budget cap forces compression
        effective_budget = budget
        if self._budget.history_budget is not None and messages:
            sys_tokens = messages[0].token_count
            history_tokens = total - sys_tokens
            if history_tokens > self._budget.history_budget:
                effective_budget = sys_tokens + self._budget.history_budget

        # Compute utilization for strategy selection
        utilization = total / max(self._budget.total_window, 1)
        strategy_name = self._resolve_strategy_name(utilization)

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
            report = self._build_context_report(
                turn=turn,
                strategy_name="passthrough",
                curated=messages,
                messages=messages,
                tool_token_estimate=effective_tool_tokens,
                memory_context=memory_context,
                memory_injected=memory_injected,
                messages_in=messages_in,
            )
            report.tool_results_pruned = p0_pruned
            report.tool_results_tokens_saved = p0_saved
            self.last_report = report
            return messages

        # Strategy: llm_summarize or aggressive_truncate with gateway available —
        # try LLM compression first (gateway-assisted), fall back to aggressive truncate
        if (strategy_name in ("llm_summarize", "aggressive_truncate")) and self._gateway is not None:
            result = await self._acompress_with_relevance(
                messages,
                budget=effective_budget,
                turn=turn,
                messages_in=messages_in,
                effective_tool_tokens=effective_tool_tokens,
                memory_injected=memory_injected,
            )
            # If aggressive was requested and LLM compression is still over budget,
            # fall back to aggressive truncation
            if strategy_name == "aggressive_truncate":
                result_total = sum(m.token_count for m in result)
                if result_total > effective_budget:
                    curated = self._aggressive_truncate(messages)
                    self.last_decision = ContextDecision(
                        turn=turn,
                        budget_total=budget,
                        budget_used=sum(m.token_count for m in curated),
                        budget_tools=effective_tool_tokens,
                        budget_reserve=self._budget.response_reserve,
                        messages_in=messages_in,
                        messages_out=len(curated),
                        memory_injected=memory_injected,
                        history_compressed=True,
                        explanation=f"Aggressive truncate: kept system + last {self._strategy.aggressive_keep_turns} turns",
                    )
                    report = self._build_context_report(
                        turn=turn,
                        strategy_name="aggressive_truncate",
                        curated=curated,
                        messages=messages,
                        tool_token_estimate=effective_tool_tokens,
                        memory_context=memory_context,
                        memory_injected=memory_injected,
                        messages_in=messages_in,
                    )
                    report.tool_results_pruned = p0_pruned
                    report.tool_results_tokens_saved = p0_saved
                    self.last_report = report
                    return curated
            report = self._build_context_report(
                turn=turn,
                strategy_name="llm_summarize",
                curated=result,
                messages=messages,
                tool_token_estimate=effective_tool_tokens,
                memory_context=memory_context,
                memory_injected=memory_injected,
                messages_in=messages_in,
            )
            report.tool_results_pruned = p0_pruned
            report.tool_results_tokens_saved = p0_saved
            self.last_report = report
            return result

        # Default: tail_preserve (or aggressive without gateway)
        effective_strategy = strategy_name if strategy_name not in ("passthrough",) else "tail_preserve"
        result = self._compress_with_relevance(
            messages,
            budget=effective_budget,
            turn=turn,
            messages_in=messages_in,
            effective_tool_tokens=effective_tool_tokens,
            memory_injected=memory_injected,
        )
        # If aggressive was requested and relevance compression is still over budget
        if strategy_name == "aggressive_truncate":
            result_total = sum(m.token_count for m in result)
            if result_total > effective_budget:
                curated = self._aggressive_truncate(messages)
                self.last_decision = ContextDecision(
                    turn=turn,
                    budget_total=budget,
                    budget_used=sum(m.token_count for m in curated),
                    budget_tools=effective_tool_tokens,
                    budget_reserve=self._budget.response_reserve,
                    messages_in=messages_in,
                    messages_out=len(curated),
                    memory_injected=memory_injected,
                    history_compressed=True,
                    explanation=f"Aggressive truncate: kept system + last {self._strategy.aggressive_keep_turns} turns",
                )
                report = self._build_context_report(
                    turn=turn,
                    strategy_name="aggressive_truncate",
                    curated=curated,
                    messages=messages,
                    tool_token_estimate=effective_tool_tokens,
                    memory_context=memory_context,
                    memory_injected=memory_injected,
                    messages_in=messages_in,
                )
                report.tool_results_pruned = p0_pruned
                report.tool_results_tokens_saved = p0_saved
                self.last_report = report
                return curated
        report = self._build_context_report(
            turn=turn,
            strategy_name=effective_strategy,
            curated=result,
            messages=messages,
            tool_token_estimate=effective_tool_tokens,
            memory_context=memory_context,
            memory_injected=memory_injected,
            messages_in=messages_in,
        )
        report.tool_results_pruned = p0_pruned
        report.tool_results_tokens_saved = p0_saved
        self.last_report = report
        return result

    async def _acompress_with_relevance(
        self,
        messages: list[Message],
        *,
        budget: int,
        turn: int,
        messages_in: int,
        effective_tool_tokens: int,
        memory_injected: bool,
    ) -> list[Message]:
        """Async compression: uses LLM when available, falls back to keyword-based."""
        keep_head = 1
        keep_tail = min(len(messages) - keep_head, self._strategy.tail_preserve_keep_recent)

        head = messages[:keep_head]
        tail = messages[-keep_tail:] if keep_tail > 0 else []
        middle = messages[keep_head: len(messages) - keep_tail] if keep_tail > 0 else messages[keep_head:]

        head_tokens = sum(m.token_count for m in head)
        tail_tokens = sum(m.token_count for m in tail)
        summary_budget = budget - head_tokens - tail_tokens - 100

        middle_tokens = sum(m.token_count for m in middle)

        # Try LLM compression if gateway is available and middle section is large enough
        if (
            self._gateway is not None
            and middle
            and middle_tokens > _LLM_COMPRESSION_THRESHOLD
            and summary_budget > 0
        ):
            try:
                summary_text = await self._compress_with_llm(middle, summary_budget)
                summary_msg = Message(role=MessageRole.USER, content=summary_text)
                result = head + [summary_msg] + tail

                used_tokens = sum(m.token_count for m in result)
                compressed_tokens = estimate_tokens(summary_text)
                ratio = middle_tokens / max(compressed_tokens, 1)

                compressed_descs: list[str] = []
                for msg in middle:
                    role = msg.role if isinstance(msg.role, str) else msg.role.value
                    content = _content_text(msg.content)
                    trunc = content[:80] + "..." if len(content) > 80 else content
                    compressed_descs.append(f"{role}:{trunc}")

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
                        f"{len(middle)} messages LLM-compressed ({ratio:.1f}x), "
                        f"budget {used_tokens}/{budget} tokens ({used_tokens * 100 // budget}% full)"
                        if budget > 0
                        else f"{len(middle)} messages LLM-compressed ({ratio:.1f}x), budget exhausted"
                    ),
                )
                return result
            except Exception:
                logger.warning(
                    "LLM compression failed, falling back to keyword-based compression",
                    exc_info=True,
                )

        # Fall back to sync keyword-based compression
        return self._compress_with_relevance(
            messages,
            budget=budget,
            turn=turn,
            messages_in=messages_in,
            effective_tool_tokens=effective_tool_tokens,
            memory_injected=memory_injected,
        )

    async def _compress_with_llm(self, messages: list[Message], budget_tokens: int) -> str:
        """Use a cheap LLM to summarize middle messages semantically.

        Args:
            messages: The middle messages to compress.
            budget_tokens: Maximum token count for the resulting summary.

        Returns:
            A summary string that fits within budget_tokens.
        """
        assert self._gateway is not None  # Caller must check

        # Build conversation text for the summarizer
        conversation_lines: list[str] = []
        for msg in messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            content = _content_text(msg.content)
            # Truncate very long individual messages to avoid blowing up the summarizer input
            if len(content) > 1000:
                content = content[:1000] + "..."
            conversation_lines.append(f"[{role}] {content}")
        conversation_text = "\n".join(conversation_lines)

        # Estimate max_tokens for the summary response (~4 chars per token)
        max_summary_tokens = min(budget_tokens, 500)

        goal_instruction = ""
        if self._goal:
            goal_instruction = f"\nThe user's current goal is: {self._goal}\nPreserve details relevant to this goal."

        from arcana.contracts.llm import LLMRequest

        prompt = (
            "Summarize the following conversation excerpt concisely. "
            "Keep key facts, decisions, tool results, and errors. "
            "Drop filler and repetition. "
            f"Use at most {max_summary_tokens * 4} characters."
            f"{goal_instruction}\n\n"
            f"---\n{conversation_text}\n---"
        )

        request = LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are a conversation summarizer. Be concise."),
                Message(role=MessageRole.USER, content=prompt),
            ],
        )

        # Use compression_model if configured, otherwise build a cheap default
        config = self._compression_model
        if config is None:
            # Derive a cheap config from the gateway's default provider
            provider_name = self._gateway.default_provider
            if provider_name is None:
                providers = self._gateway.list_providers()
                if not providers:
                    raise RuntimeError("No providers registered in gateway")
                provider_name = providers[0]
            config = ModelConfig(
                provider=provider_name,
                model_id="",  # Will be resolved by provider's default_model
                temperature=0.0,
                max_tokens=max_summary_tokens,
            )
            # Try to get the provider's default model
            provider = self._gateway.get(provider_name)
            if provider and hasattr(provider, "default_model"):
                dm = provider.default_model
                if isinstance(dm, str) and dm:
                    config = config.model_copy(update={"model_id": dm})

        response = await self._gateway.generate(request, config)
        summary = response.content or ""

        # Truncate if it somehow exceeds budget
        while estimate_tokens(summary) > budget_tokens and len(summary) > 10:
            summary = summary[: int(len(summary) * 0.8)]

        return f"[Earlier conversation — LLM summary]\n{summary}"

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
