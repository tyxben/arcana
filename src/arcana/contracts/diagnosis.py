"""Diagnostic error recovery contracts and data models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    """What kind of mistake was made.

    This is the semantic-layer classification: what went wrong and why.
    It drives the recovery prompt shown to the LLM, complementing the
    transport-layer ``ErrorType`` which drives the automatic retry loop.
    """

    FACT_ERROR = "fact_error"
    # LLM hallucinated a tool name, parameter value, or assumption.
    # Example: called "search_web" when the tool is "web_search".

    FORMAT_ERROR = "format_error"
    # LLM output was structurally wrong: bad JSON, missing fields, wrong types.
    # Example: passed {"query": 123} when a string was required.

    CONSTRAINT_VIOLATION = "constraint_violation"
    # Arguments were syntactically valid but violated semantic constraints.
    # Example: date range spans 10 years when max is 1 year.

    TOOL_MISMATCH = "tool_mismatch"
    # The LLM chose the wrong tool for its actual intent.
    # Example: used file_read to query a database.

    SCOPE_TOO_BROAD = "scope_too_broad"
    # The request was valid but too large to process.
    # Example: "summarize all files in the repository" with 10k files.

    PERMISSION_DENIED = "permission_denied"
    # The agent lacks required capabilities.
    # Example: tried to write a file without write capability.

    RESOURCE_UNAVAILABLE = "resource_unavailable"
    # External dependency is down, rate-limited, or unreachable.
    # Example: API returned 503, database connection refused.


class ErrorLayer(str, Enum):
    """Where in the pipeline the error originated."""

    LLM_REASONING = "llm_reasoning"
    # The LLM's decision was flawed before any tool was invoked.
    # Diagnosed from: tool name not found, obviously wrong arguments.

    TOOL_EXECUTION = "tool_execution"
    # The tool itself threw an error during execution.
    # Diagnosed from: ToolExecutionError, provider exceptions.

    VALIDATION = "validation"
    # Schema or constraint validation caught the problem.
    # Diagnosed from: validate_arguments() returning ToolError.

    EXTERNAL_SERVICE = "external_service"
    # An external API, database, or service failed.
    # Diagnosed from: HTTP status codes, connection errors.


class RecoveryStrategy(str, Enum):
    """What the system should do about the error.

    Escalation curve (defaults):
      Attempt 1: RETRY_WITH_MODIFICATION
      Attempt 2: RETRY_WITH_MODIFICATION
      Attempt 3: SWITCH_TOOL or NARROW_SCOPE
      Attempt 4: ESCALATE
      Attempt 5: ABORT
    """

    RETRY_WITH_MODIFICATION = "retry_with_modification"
    # Fix the specific problem and try the same tool again.

    SWITCH_TOOL = "switch_tool"
    # Abandon this tool and suggest an alternative.

    NARROW_SCOPE = "narrow_scope"
    # Reduce the ambition of the request.

    ESCALATE = "escalate"
    # Ask the human for help or clarification.

    ABORT = "abort"
    # Stop trying. This path is a dead end.


class ErrorDiagnosis(BaseModel):
    """Complete diagnosis of what went wrong and what to do about it."""

    error_category: ErrorCategory
    error_layer: ErrorLayer
    root_cause: str
    # Human-readable explanation of the root cause.

    actionable_suggestions: list[str]
    # Concrete suggestions the LLM can act on in the next turn.

    recommended_strategy: RecoveryStrategy
    # The best recovery strategy based on diagnosis.

    confidence: float = 1.0
    # How confident the diagnoser is (0.0 to 1.0).

    previous_attempts: int = 0
    # How many times we have already tried to recover from this category.

    original_error: str | None = None
    # Raw error message, preserved for debugging.

    related_tool: str | None = None
    # The tool that was involved, if any.

    suggested_tool: str | None = None
    # Alternative tool to try, if strategy is SWITCH_TOOL.

    suggested_arguments: dict[str, Any] | None = None
    # Modified arguments to try, if strategy is RETRY_WITH_MODIFICATION.

    metadata: dict[str, Any] = Field(default_factory=dict)
    # Extensible metadata for diagnoser-specific details.

    def to_recovery_prompt(self) -> str:
        """Generate a structured recovery prompt for the LLM.

        Pure method -- no side effects. Produces a human-readable summary
        that the LLM can use to adjust its next action.
        """
        suggestions_block = "\n".join(f"  - {s}" for s in self.actionable_suggestions)

        strategy_hints: dict[RecoveryStrategy, str] = {
            RecoveryStrategy.RETRY_WITH_MODIFICATION: (
                "Modify your previous attempt based on the suggestions above."
            ),
            RecoveryStrategy.SWITCH_TOOL: (
                f"Consider using '{self.suggested_tool}' instead."
                if self.suggested_tool
                else "Look for an alternative tool that better matches your intent."
            ),
            RecoveryStrategy.NARROW_SCOPE: (
                "Break your request into smaller pieces or add constraints to reduce scope."
            ),
            RecoveryStrategy.ESCALATE: (
                "This requires human assistance. Explain what you need help with."
            ),
            RecoveryStrategy.ABORT: (
                "This approach is not viable. Reconsider your overall strategy."
            ),
        }

        return (
            "Your previous action encountered a problem. Here is the diagnosis:\n"
            "\n"
            f"**Error type**: {self.error_category.value}\n"
            f"**Where it happened**: {self.error_layer.value}\n"
            f"**Root cause**: {self.root_cause}\n"
            "\n"
            "**Suggested adjustments**:\n"
            f"{suggestions_block}\n"
            "\n"
            f"**Recommended strategy**: {strategy_hints[self.recommended_strategy]}\n"
            "\n"
            "Based on this diagnosis, choose one of the following approaches:\n"
            "1. Fix the specific issue and retry (if the suggestion is clear)\n"
            "2. Use a different tool that better fits your intent\n"
            "3. Reduce the scope of what you're trying to do\n"
            "4. Indicate that you need human input to proceed\n"
            "\n"
            "Do NOT simply repeat the same action. Address the root cause."
        )
