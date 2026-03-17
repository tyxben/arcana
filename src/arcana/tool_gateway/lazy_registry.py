"""Lazy tool registry that exposes tools incrementally to the LLM."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from arcana.contracts.tool import ToolSpec
from arcana.tool_gateway.formatter import format_tool_for_llm

if TYPE_CHECKING:
    from arcana.tool_gateway.registry import ToolRegistry


class ToolExpansionEvent(BaseModel):
    """Record of a working set expansion for tracing."""

    trigger: str  # "initial_selection", "on_demand_expansion", "explicit_request"
    context: str  # The goal, request text, or tool name that triggered expansion
    tools_added: list[str]
    working_set_size: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class ToolMatcher(Protocol):
    """Protocol for ranking tools by relevance to a query."""

    def rank(self, query: str, candidates: list[ToolSpec]) -> list[ToolSpec]:
        """
        Return candidates sorted by relevance to query (most relevant first).

        Must be deterministic for the same inputs.
        """
        ...


class KeywordToolMatcher:
    """
    Matches tools by keyword overlap between query and tool metadata.

    Scoring:
    - +3 for tool name substring match in query
    - +2 for each keyword in tool description found in query
    - +2 for each when_to_use keyword match
    - +1 for category match
    - Ties broken by tool name alphabetically (deterministic)
    """

    CATEGORY_KEYWORDS: dict[str, list[str]] = {
        "search": ["search", "find", "look up", "query", "lookup", "google"],
        "file": [
            "file",
            "read",
            "write",
            "create",
            "edit",
            "delete",
            "path",
            "directory",
        ],
        "code": ["code", "execute", "run", "script", "compile", "test", "debug"],
        "web": ["web", "http", "url", "fetch", "download", "api", "request"],
        "data": ["data", "database", "sql", "table", "csv", "json", "parse"],
        "shell": ["shell", "command", "terminal", "bash", "process", "system"],
    }

    def rank(self, query: str, candidates: list[ToolSpec]) -> list[ToolSpec]:
        """Rank candidates by keyword relevance to query."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: list[tuple[float, str, ToolSpec]] = []

        for spec in candidates:
            score = 0.0

            # Name match: +3 if tool name appears as substring in query
            name_normalized = spec.name.replace("_", " ")
            if name_normalized in query_lower or spec.name in query_lower:
                score += 3.0

            # Description keyword overlap: +2 per overlapping word
            desc_words = set(spec.description.lower().split())
            overlap = desc_words & query_words
            score += len(overlap) * 2.0

            # when_to_use match: +2 per overlapping word
            if spec.when_to_use:
                wtu_words = set(spec.when_to_use.lower().split())
                wtu_overlap = wtu_words & query_words
                score += len(wtu_overlap) * 2.0

            # Category match: +1 if query and tool share category keywords
            for _category, keywords in self.CATEGORY_KEYWORDS.items():
                if any(kw in query_lower for kw in keywords):
                    if any(
                        kw in spec.name.lower() or kw in spec.description.lower()
                        for kw in keywords
                    ):
                        score += 1.0

            scored.append((score, spec.name, spec))

        # Sort by score descending, then name ascending for determinism
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [spec for _, _, spec in scored]


class LazyToolRegistry:
    """
    Wraps a full ToolRegistry and exposes tools incrementally.

    The LLM sees only a working set of tools. Tools are added to the
    working set based on goal analysis, explicit requests, or error
    recovery suggestions.
    """

    def __init__(
        self,
        full_registry: ToolRegistry,
        *,
        max_initial_tools: int = 5,
        max_working_set: int = 15,
        matcher: ToolMatcher | None = None,
    ) -> None:
        self._full = full_registry
        self._working_set: dict[str, ToolSpec] = {}
        self._max_initial = max_initial_tools
        self._max_working = max_working_set
        self._matcher: ToolMatcher = matcher or KeywordToolMatcher()
        self._expansion_log: list[ToolExpansionEvent] = []

    @property
    def working_set(self) -> list[ToolSpec]:
        """Currently exposed tools, ordered by insertion."""
        return list(self._working_set.values())

    @property
    def available_but_hidden(self) -> list[str]:
        """Tool names in full registry but not in working set."""
        return [
            name for name in self._full.list_tools() if name not in self._working_set
        ]

    @property
    def expansion_log(self) -> list[ToolExpansionEvent]:
        """History of working set expansions."""
        return list(self._expansion_log)

    def select_initial_tools(self, goal: str) -> list[ToolSpec]:
        """
        Analyze the goal and select the most relevant initial tools.

        Called once at the start of a run. Returns up to max_initial_tools.
        The returned tools are added to the working set.
        """
        all_specs = self._full.get_specs()
        ranked = self._matcher.rank(goal, all_specs)
        selected = ranked[: self._max_initial]

        for spec in selected:
            self._working_set[spec.name] = spec

        self._log_expansion("initial_selection", goal, [s.name for s in selected])
        return selected

    def expand(self, request: str) -> list[ToolSpec]:
        """
        Expand the working set based on an explicit need.

        Called when the LLM says "I need a tool for X" or the diagnostic
        system suggests SWITCH_TOOL.

        Returns newly added tools (not previously in working set).
        """
        all_specs = self._full.get_specs()
        hidden_specs = [s for s in all_specs if s.name not in self._working_set]
        ranked = self._matcher.rank(request, hidden_specs)

        room = self._max_working - len(self._working_set)
        to_add = ranked[: max(room, 1)]  # Always add at least 1

        new_tools: list[ToolSpec] = []
        for spec in to_add:
            if spec.name not in self._working_set:
                self._working_set[spec.name] = spec
                new_tools.append(spec)

        self._log_expansion(
            "on_demand_expansion", request, [s.name for s in new_tools]
        )
        return new_tools

    def get_tool_on_demand(self, tool_name: str) -> ToolSpec | None:
        """
        Load a specific tool by name into the working set.

        Called when the LLM explicitly names a tool that is not in
        the working set but exists in the full registry.

        Returns the ToolSpec if found, None if the tool does not exist.
        """
        if tool_name in self._working_set:
            return self._working_set[tool_name]

        provider = self._full.get(tool_name)
        if provider is None:
            return None

        spec = provider.spec
        self._working_set[tool_name] = spec
        self._log_expansion("explicit_request", tool_name, [tool_name])
        return spec

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert working set (not full registry) to OpenAI format."""
        tools: list[dict[str, Any]] = []
        for spec in self._working_set.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": format_tool_for_llm(spec),
                        "parameters": spec.input_schema,
                    },
                }
            )
        return tools

    def reset(self) -> None:
        """Clear the working set (call at the start of a new run)."""
        self._working_set.clear()
        self._expansion_log.clear()

    def _log_expansion(
        self, trigger: str, context: str, tools: list[str]
    ) -> None:
        self._expansion_log.append(
            ToolExpansionEvent(
                trigger=trigger,
                context=context,
                tools_added=tools,
                working_set_size=len(self._working_set),
            )
        )
