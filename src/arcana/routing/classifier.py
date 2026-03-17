"""Intent classifiers: rule-based, LLM-based, and hybrid."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from arcana.contracts.intent import IntentClassification, IntentType
from arcana.contracts.llm import LLMRequest, Message, MessageRole, ModelConfig
from arcana.gateway.registry import ModelGatewayRegistry

# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class IntentClassifier(ABC):
    """Base class for intent classifiers."""

    @abstractmethod
    async def classify(
        self,
        goal: str,
        *,
        available_tools: list[str] | None = None,
        context: dict | None = None,
    ) -> IntentClassification:
        """Classify user intent into an execution path.

        Args:
            goal: The user's request or goal.
            available_tools: Names of available tools (for tool-matching).
            context: Optional additional context (conversation history, etc.).

        Returns:
            IntentClassification with route and metadata.
        """
        ...


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------


class RuleBasedClassifier(IntentClassifier):
    """Fast, heuristic-based intent classifier.

    Trades accuracy for speed and zero cost.
    Use as first pass; fall back to LLM classifier for ambiguous cases.
    """

    # Patterns that suggest direct answer
    DIRECT_PATTERNS = [
        r"^(what|who|when|where|how much|how many)\b.*\?$",
        r"^(explain|describe|define|summarize)\b",
        r"^(yes|no|true|false)\b",
        r"^(translate|convert)\b.*\bto\b",
    ]

    # Patterns that suggest tool usage
    TOOL_TRIGGER_WORDS: dict[str, list[str]] = {
        "search": ["web_search", "search"],
        "read file": ["file_read", "read_file"],
        "write file": ["file_write", "write_file"],
        "run": ["shell_exec", "code_exec"],
        "execute": ["shell_exec", "code_exec"],
        "fetch": ["web_fetch", "http_get"],
        "calculate": ["calculator", "math"],
    }

    # Patterns that suggest complexity
    COMPLEX_INDICATORS = [
        r"\b(and then|after that|finally|step \d)\b",
        r"\b(refactor|migrate|redesign|implement|build)\b",
        r"\b(analyze|compare|evaluate).*(and|then)\b",
        r"\b(create a plan|make a plan|plan out)\b",
    ]

    async def classify(
        self,
        goal: str,
        *,
        available_tools: list[str] | None = None,
        context: dict | None = None,
    ) -> IntentClassification:
        goal_lower = goal.lower().strip()

        # Check for direct answer patterns
        for pattern in self.DIRECT_PATTERNS:
            if re.search(pattern, goal_lower):
                return IntentClassification(
                    intent=IntentType.DIRECT_ANSWER,
                    confidence=0.7,
                    reasoning=f"Matched direct pattern: {pattern}",
                )

        # Check for single tool triggers
        matched_tools: list[str] = []
        for trigger, tools in self.TOOL_TRIGGER_WORDS.items():
            if trigger in goal_lower:
                matched_tools.extend(tools)

        if matched_tools and not self._has_complexity_indicators(goal_lower):
            # Filter to available tools if provided
            if available_tools:
                matched_tools = [t for t in matched_tools if t in available_tools]
            if matched_tools:
                return IntentClassification(
                    intent=IntentType.SINGLE_TOOL,
                    confidence=0.6,
                    reasoning=f"Tool trigger detected: {matched_tools[0]}",
                    suggested_tools=matched_tools[:1],
                )

        # Check for complexity indicators
        complexity = self._estimate_complexity(goal_lower)
        if complexity >= 4:
            return IntentClassification(
                intent=IntentType.COMPLEX_PLAN,
                confidence=0.5,
                reasoning="High complexity indicators detected",
                complexity_estimate=complexity,
            )

        if complexity >= 2 or matched_tools:
            return IntentClassification(
                intent=IntentType.AGENT_LOOP,
                confidence=0.5,
                reasoning="Moderate complexity or tool usage needed",
                suggested_tools=matched_tools[:3],
                complexity_estimate=complexity,
            )

        # Default: direct answer (most things are simple)
        return IntentClassification(
            intent=IntentType.DIRECT_ANSWER,
            confidence=0.4,
            reasoning="No complexity signals; defaulting to direct",
        )

    def _has_complexity_indicators(self, text: str) -> bool:
        return any(re.search(p, text) for p in self.COMPLEX_INDICATORS)

    def _estimate_complexity(self, text: str) -> int:
        score = 1
        if len(text) > 200:
            score += 1
        if self._has_complexity_indicators(text):
            score += 2
        if text.count(" and ") >= 2:
            score += 1
        return min(score, 5)


# ---------------------------------------------------------------------------
# LLM-based classifier
# ---------------------------------------------------------------------------

CLASSIFICATION_PROMPT = """Classify the user's request into one of four categories:

1. direct_answer - Can be answered in a single response without tools
2. single_tool - Needs exactly one tool call (e.g., file read, web search)
3. agent_loop - Needs multiple steps and/or tools, approach uncertain
4. complex_plan - High-stakes, multi-phase task requiring explicit planning

Available tools: {tools}

Respond with ONLY a JSON object:
{{"intent": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief>", "suggested_tools": [<tool names if applicable>]}}

User request: {goal}"""


class LLMClassifier(IntentClassifier):
    """LLM-based intent classifier.

    Uses a small, fast model for accurate classification.
    """

    def __init__(
        self,
        gateway: ModelGatewayRegistry,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.gateway = gateway
        if model_config is not None:
            self.model_config = model_config
        else:
            # Resolve from gateway's default provider instead of hardcoding
            provider_name = gateway.default_provider or "deepseek"
            provider = gateway.get(provider_name)
            default_model_id: str | None = None
            if provider and hasattr(provider, "default_model"):
                dm = provider.default_model
                if isinstance(dm, str) and dm:
                    default_model_id = dm
            if not default_model_id:
                msg = (
                    f"No default model configured for provider '{provider_name}'. "
                    "Pass model_config explicitly or register a provider with a default_model."
                )
                raise ValueError(msg)
            self.model_config = ModelConfig(
                provider=provider_name,
                model_id=default_model_id,
                temperature=0.0,
                max_tokens=200,
            )

    async def classify(
        self,
        goal: str,
        *,
        available_tools: list[str] | None = None,
        context: dict | None = None,
    ) -> IntentClassification:
        tools_str = ", ".join(available_tools) if available_tools else "none"
        prompt = CLASSIFICATION_PROMPT.format(goal=goal, tools=tools_str)

        request = LLMRequest(
            messages=[Message(role=MessageRole.USER, content=prompt)]
        )
        response = await self.gateway.generate(
            request=request, config=self.model_config
        )

        # Parse response
        try:
            data = json.loads(response.content or "{}")
            return IntentClassification(
                intent=IntentType(data.get("intent", "direct_answer")),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning"),
                suggested_tools=data.get("suggested_tools", []),
            )
        except (json.JSONDecodeError, ValueError):
            # Parse failure: default to agent_loop (safe fallback)
            return IntentClassification(
                intent=IntentType.AGENT_LOOP,
                confidence=0.3,
                reasoning="Classification parse failed; defaulting to agent_loop",
            )


# ---------------------------------------------------------------------------
# Hybrid classifier
# ---------------------------------------------------------------------------


class HybridClassifier(IntentClassifier):
    """Rule-based first pass, LLM fallback for ambiguous cases.

    Confidence threshold determines when to escalate to LLM.
    """

    def __init__(
        self,
        gateway: ModelGatewayRegistry,
        *,
        confidence_threshold: float = 0.6,
        llm_model_config: ModelConfig | None = None,
    ) -> None:
        self.rule_classifier = RuleBasedClassifier()
        self.llm_classifier = LLMClassifier(gateway, llm_model_config)
        self.confidence_threshold = confidence_threshold

    async def classify(
        self,
        goal: str,
        *,
        available_tools: list[str] | None = None,
        context: dict | None = None,
    ) -> IntentClassification:
        # Try rule-based first (free)
        result = await self.rule_classifier.classify(
            goal, available_tools=available_tools, context=context
        )

        # If confident enough, use rule-based result
        if result.confidence >= self.confidence_threshold:
            return result

        # Otherwise, ask the LLM
        return await self.llm_classifier.classify(
            goal, available_tools=available_tools, context=context
        )
