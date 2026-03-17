"""Pure-function diagnosers for tool, LLM, and validation errors.

Every function in this module is deterministic, side-effect-free, and easy to
unit test.  The only stateful component lives in ``tracker.py``.
"""

from __future__ import annotations

import json
from typing import Any

from arcana.contracts.diagnosis import (
    ErrorCategory,
    ErrorDiagnosis,
    ErrorLayer,
    RecoveryStrategy,
)
from arcana.contracts.tool import ToolError, ToolSpec

# ---------------------------------------------------------------------------
# Escalation tables (category -> strategy per attempt index)
# ---------------------------------------------------------------------------

_ESCALATION_TABLE: dict[ErrorCategory, list[RecoveryStrategy]] = {
    ErrorCategory.FACT_ERROR: [
        RecoveryStrategy.RETRY_WITH_MODIFICATION,
        RecoveryStrategy.SWITCH_TOOL,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.FORMAT_ERROR: [
        RecoveryStrategy.RETRY_WITH_MODIFICATION,
        RecoveryStrategy.RETRY_WITH_MODIFICATION,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.CONSTRAINT_VIOLATION: [
        RecoveryStrategy.RETRY_WITH_MODIFICATION,
        RecoveryStrategy.NARROW_SCOPE,
        RecoveryStrategy.ESCALATE,
    ],
    ErrorCategory.TOOL_MISMATCH: [
        RecoveryStrategy.SWITCH_TOOL,
        RecoveryStrategy.SWITCH_TOOL,
        RecoveryStrategy.ESCALATE,
    ],
    ErrorCategory.SCOPE_TOO_BROAD: [
        RecoveryStrategy.NARROW_SCOPE,
        RecoveryStrategy.NARROW_SCOPE,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.PERMISSION_DENIED: [
        RecoveryStrategy.SWITCH_TOOL,
        RecoveryStrategy.ESCALATE,
        RecoveryStrategy.ABORT,
    ],
    ErrorCategory.RESOURCE_UNAVAILABLE: [
        RecoveryStrategy.RETRY_WITH_MODIFICATION,
        RecoveryStrategy.NARROW_SCOPE,
        RecoveryStrategy.ABORT,
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers (pure)
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


def _fuzzy_match_tool_name(name: str, available: list[str]) -> str | None:
    """Find the closest tool name using edit distance.

    Returns the match if edit distance <= 3, else ``None``.
    """
    best: str | None = None
    best_dist = 4  # threshold + 1
    for candidate in available:
        dist = _levenshtein(name.lower(), candidate.lower())
        if dist < best_dist:
            best_dist = dist
            best = candidate
    return best


def _suggest_field_fix(
    field_name: str,
    field_schema: dict[str, Any],
    actual_value: Any,
) -> str:
    """Generate a human-readable suggestion for fixing a field value."""
    expected_type = field_schema.get("type", "unknown")
    actual_type = type(actual_value).__name__
    if expected_type != "unknown" and actual_type != expected_type:
        return (
            f"Field '{field_name}' should be {expected_type} "
            f"(e.g., {_example_for_type(expected_type)}), "
            f"not {actual_type} ({actual_value!r})."
        )
    return f"Field '{field_name}' has an invalid value: {actual_value!r}."


def _example_for_type(json_type: str) -> str:
    """Return a short example literal for a JSON Schema type."""
    examples: dict[str, str] = {
        "string": '"example"',
        "integer": "10",
        "number": "3.14",
        "boolean": "true",
        "array": "[]",
        "object": "{}",
    }
    return examples.get(json_type, "...")


def _escalation_needed(attempt_history: list[ErrorDiagnosis]) -> bool:
    """Determine if we should escalate based on recovery history.

    Rules:
    - Same ``ErrorCategory`` failed 3+ times -> escalate
    - Total recovery attempts > 5 -> escalate
    - ``ABORT`` already recommended once -> do not override
    """
    if len(attempt_history) > 5:
        return True
    from collections import Counter

    counts = Counter(d.error_category for d in attempt_history)
    return any(c >= 3 for c in counts.values())


def _pick_strategy(
    category: ErrorCategory,
    attempt_history: list[ErrorDiagnosis],
) -> RecoveryStrategy:
    """Pick a recovery strategy from the escalation table."""
    # If any prior diagnosis already recommended ABORT, respect it.
    if any(d.recommended_strategy == RecoveryStrategy.ABORT for d in attempt_history):
        return RecoveryStrategy.ABORT

    if _escalation_needed(attempt_history):
        return RecoveryStrategy.ABORT

    # Count same-category prior attempts
    same_cat_count = sum(1 for d in attempt_history if d.error_category == category)
    table = _ESCALATION_TABLE.get(category, [RecoveryStrategy.ABORT])
    idx = min(same_cat_count, len(table) - 1)
    return table[idx]


# ---------------------------------------------------------------------------
# Public diagnosers
# ---------------------------------------------------------------------------


def diagnose_tool_error(
    tool_name: str,
    tool_error: ToolError,
    available_tools: list[str],
    attempt_history: list[ErrorDiagnosis] | None = None,
) -> ErrorDiagnosis:
    """Diagnose a tool execution failure.  Pure function.

    Decision tree:
    1. ``TOOL_NOT_FOUND`` + close name match -> FACT_ERROR + suggest correct name
    2. ``TOOL_NOT_FOUND`` + no match         -> TOOL_MISMATCH + SWITCH_TOOL
    3. ``UNAUTHORIZED``                      -> PERMISSION_DENIED
    4. ``TIMEOUT``                           -> RESOURCE_UNAVAILABLE
    5. ``VALIDATION_ERROR``                  -> FORMAT_ERROR or CONSTRAINT_VIOLATION
    6. ``CONFIRMATION_REQUIRED/REJECTED``    -> PERMISSION_DENIED + ESCALATE
    7. Fallback                              -> RESOURCE_UNAVAILABLE + ABORT
    """
    history = attempt_history or []
    code = (tool_error.code or "").upper()

    # 1/2. Tool not found
    if code == "TOOL_NOT_FOUND" or "not found" in tool_error.message.lower():
        match = _fuzzy_match_tool_name(tool_name, available_tools)
        if match:
            category = ErrorCategory.FACT_ERROR
            strategy = _pick_strategy(category, history)
            return ErrorDiagnosis(
                error_category=category,
                error_layer=ErrorLayer.LLM_REASONING,
                root_cause=f"Tool '{tool_name}' does not exist. "
                f"Did you mean '{match}'?",
                actionable_suggestions=[
                    f"Use '{match}' instead of '{tool_name}'.",
                    f"Available tools: {', '.join(available_tools)}",
                ],
                recommended_strategy=strategy,
                original_error=tool_error.message,
                related_tool=tool_name,
                suggested_tool=match,
                previous_attempts=len(history),
            )
        # No close match
        category = ErrorCategory.TOOL_MISMATCH
        strategy = _pick_strategy(category, history)
        return ErrorDiagnosis(
            error_category=category,
            error_layer=ErrorLayer.LLM_REASONING,
            root_cause=f"Tool '{tool_name}' does not exist and no similar tool was found.",
            actionable_suggestions=[
                f"Available tools: {', '.join(available_tools)}",
                "Choose a tool from the available list that matches your intent.",
            ],
            recommended_strategy=strategy,
            original_error=tool_error.message,
            related_tool=tool_name,
            previous_attempts=len(history),
        )

    # 3. Unauthorized / permission denied
    if code == "UNAUTHORIZED" or "unauthorized" in tool_error.message.lower():
        category = ErrorCategory.PERMISSION_DENIED
        strategy = _pick_strategy(category, history)
        capability = tool_error.details.get("required_capability", "unknown")
        return ErrorDiagnosis(
            error_category=category,
            error_layer=ErrorLayer.TOOL_EXECUTION,
            root_cause=f"Missing required capability '{capability}' for tool '{tool_name}'.",
            actionable_suggestions=[
                f"You lack the '{capability}' capability for this tool.",
                "Try a different tool that does not require this capability.",
            ],
            recommended_strategy=strategy,
            original_error=tool_error.message,
            related_tool=tool_name,
            previous_attempts=len(history),
        )

    # 4. Timeout
    if code == "TIMEOUT" or "timeout" in tool_error.message.lower():
        category = ErrorCategory.RESOURCE_UNAVAILABLE
        strategy = _pick_strategy(category, history)
        return ErrorDiagnosis(
            error_category=category,
            error_layer=ErrorLayer.EXTERNAL_SERVICE,
            root_cause=f"Tool '{tool_name}' timed out. The service may be slow or overloaded.",
            actionable_suggestions=[
                "Try reducing the scope of your request.",
                "If the service is temporarily unavailable, wait and retry.",
            ],
            recommended_strategy=strategy,
            original_error=tool_error.message,
            related_tool=tool_name,
            previous_attempts=len(history),
        )

    # 5. Validation error
    if code == "VALIDATION_ERROR" or "validation" in tool_error.message.lower():
        # Delegate to the more specific validation diagnoser when possible,
        # but handle the case where we only have a ToolError.
        if "constraint" in tool_error.message.lower():
            category = ErrorCategory.CONSTRAINT_VIOLATION
        else:
            category = ErrorCategory.FORMAT_ERROR
        strategy = _pick_strategy(category, history)
        return ErrorDiagnosis(
            error_category=category,
            error_layer=ErrorLayer.VALIDATION,
            root_cause=tool_error.message,
            actionable_suggestions=[
                "Review the tool's input schema and fix the invalid arguments.",
            ],
            recommended_strategy=strategy,
            original_error=tool_error.message,
            related_tool=tool_name,
            previous_attempts=len(history),
        )

    # 6. Confirmation required / rejected
    if code in ("CONFIRMATION_REQUIRED", "CONFIRMATION_REJECTED"):
        return ErrorDiagnosis(
            error_category=ErrorCategory.PERMISSION_DENIED,
            error_layer=ErrorLayer.TOOL_EXECUTION,
            root_cause=f"Tool '{tool_name}' requires user confirmation which was "
            f"{'not provided' if code == 'CONFIRMATION_REQUIRED' else 'rejected'}.",
            actionable_suggestions=[
                "This action requires explicit user approval.",
                "Explain what you need to do and ask the user for permission.",
            ],
            recommended_strategy=RecoveryStrategy.ESCALATE,
            original_error=tool_error.message,
            related_tool=tool_name,
            previous_attempts=len(history),
        )

    # 7. Fallback -- unknown error
    return ErrorDiagnosis(
        error_category=ErrorCategory.RESOURCE_UNAVAILABLE,
        error_layer=ErrorLayer.TOOL_EXECUTION,
        root_cause=f"Unexpected error from tool '{tool_name}': {tool_error.message}",
        actionable_suggestions=[
            "This is an unexpected error. Consider trying a different approach.",
        ],
        recommended_strategy=RecoveryStrategy.ABORT,
        original_error=tool_error.message,
        related_tool=tool_name,
        confidence=0.5,
        previous_attempts=len(history),
    )


def diagnose_llm_error(
    expected_format: str,
    actual_output: str,
    error_message: str,
    attempt_history: list[ErrorDiagnosis] | None = None,
) -> ErrorDiagnosis:
    """Diagnose an LLM output error.  Pure function.

    Decision tree:
    1. JSON parse failure           -> FORMAT_ERROR
    2. Missing required fields      -> FORMAT_ERROR
    3. Tool name does not exist     -> FACT_ERROR
    4. Repeated format failures     -> ABORT
    """
    history = attempt_history or []

    # Short-circuit: repeated format failures -> ABORT
    format_failures = sum(1 for d in history if d.error_category == ErrorCategory.FORMAT_ERROR)
    if format_failures >= 3:
        return ErrorDiagnosis(
            error_category=ErrorCategory.FORMAT_ERROR,
            error_layer=ErrorLayer.LLM_REASONING,
            root_cause="Repeated format failures. The prompt or expected format may be broken.",
            actionable_suggestions=[
                "The LLM has failed to produce valid output multiple times.",
                "The expected format or prompt itself may need revision.",
            ],
            recommended_strategy=RecoveryStrategy.ABORT,
            original_error=error_message,
            previous_attempts=len(history),
        )

    # JSON parse failure
    if expected_format.lower() in ("json", "json_object"):
        try:
            json.loads(actual_output)
        except (json.JSONDecodeError, ValueError):
            category = ErrorCategory.FORMAT_ERROR
            strategy = _pick_strategy(category, history)
            return ErrorDiagnosis(
                error_category=category,
                error_layer=ErrorLayer.LLM_REASONING,
                root_cause="Output is not valid JSON.",
                actionable_suggestions=[
                    "Respond with valid JSON only.",
                    "Ensure no trailing commas, unquoted keys, or extra text outside the JSON.",
                ],
                recommended_strategy=strategy,
                original_error=error_message,
                previous_attempts=len(history),
            )

    # Missing fields (heuristic: error message mentions "missing" or "required")
    lower_err = error_message.lower()
    if "missing" in lower_err or "required" in lower_err:
        category = ErrorCategory.FORMAT_ERROR
        strategy = _pick_strategy(category, history)
        return ErrorDiagnosis(
            error_category=category,
            error_layer=ErrorLayer.LLM_REASONING,
            root_cause=f"Output is missing required content: {error_message}",
            actionable_suggestions=[
                f"Ensure your response includes all required fields. Error: {error_message}",
                f"Expected format: {expected_format}",
            ],
            recommended_strategy=strategy,
            original_error=error_message,
            previous_attempts=len(history),
        )

    # Tool name does not exist (heuristic)
    if "tool" in lower_err and ("not found" in lower_err or "not exist" in lower_err):
        category = ErrorCategory.FACT_ERROR
        strategy = _pick_strategy(category, history)
        return ErrorDiagnosis(
            error_category=category,
            error_layer=ErrorLayer.LLM_REASONING,
            root_cause=f"Referenced a non-existent tool: {error_message}",
            actionable_suggestions=[
                "Use only tools from the provided tool list.",
                "Double-check the tool name for typos.",
            ],
            recommended_strategy=strategy,
            original_error=error_message,
            previous_attempts=len(history),
        )

    # Generic fallback
    category = ErrorCategory.FORMAT_ERROR
    strategy = _pick_strategy(category, history)
    return ErrorDiagnosis(
        error_category=category,
        error_layer=ErrorLayer.LLM_REASONING,
        root_cause=error_message,
        actionable_suggestions=[
            f"Ensure your output matches the expected format: {expected_format}",
        ],
        recommended_strategy=strategy,
        original_error=error_message,
        previous_attempts=len(history),
    )


def diagnose_validation_error(
    tool_spec: ToolSpec,
    arguments: dict[str, Any],
    validation_error: ToolError,
    attempt_history: list[ErrorDiagnosis] | None = None,
) -> ErrorDiagnosis:
    """Diagnose an argument validation failure.  Pure function.

    Uses the full ``ToolSpec`` (including ``input_schema``) to provide
    precise, field-level suggestions.
    """
    history = attempt_history or []
    schema = tool_spec.input_schema
    required_fields: list[str] = schema.get("required", [])
    properties: dict[str, Any] = schema.get("properties", {})
    msg = validation_error.message.lower()
    suggestions: list[str] = []

    # Missing required fields
    missing = [f for f in required_fields if f not in arguments]
    if missing:
        for field in missing:
            field_schema = properties.get(field, {})
            desc = field_schema.get("description", "")
            hint = f" ({desc})" if desc else ""
            suggestions.append(f"Add the required field '{field}'{hint}.")
        category = ErrorCategory.FORMAT_ERROR
        strategy = _pick_strategy(category, history)
        return ErrorDiagnosis(
            error_category=category,
            error_layer=ErrorLayer.VALIDATION,
            root_cause=f"Missing required fields: {', '.join(missing)}",
            actionable_suggestions=suggestions,
            recommended_strategy=strategy,
            original_error=validation_error.message,
            related_tool=tool_spec.name,
            previous_attempts=len(history),
        )

    # Unknown fields (possibly typos)
    known_fields = set(properties.keys())
    unknown = [k for k in arguments if k not in known_fields]
    if unknown:
        for field in unknown:
            match = _fuzzy_match_tool_name(field, list(known_fields))
            if match:
                suggestions.append(f"Rename '{field}' to '{match}'.")
            else:
                suggestions.append(
                    f"Remove unknown field '{field}'. "
                    f"Valid fields: {', '.join(sorted(known_fields))}."
                )
        category = ErrorCategory.FORMAT_ERROR
        strategy = _pick_strategy(category, history)
        return ErrorDiagnosis(
            error_category=category,
            error_layer=ErrorLayer.VALIDATION,
            root_cause=f"Unknown fields in arguments: {', '.join(unknown)}",
            actionable_suggestions=suggestions,
            recommended_strategy=strategy,
            original_error=validation_error.message,
            related_tool=tool_spec.name,
            previous_attempts=len(history),
        )

    # Type mismatches
    for field_name, field_schema in properties.items():
        if field_name in arguments:
            expected_type = field_schema.get("type")
            actual = arguments[field_name]
            if expected_type and not _type_matches(expected_type, actual):
                suggestions.append(_suggest_field_fix(field_name, field_schema, actual))

    if suggestions:
        category = ErrorCategory.FORMAT_ERROR
        strategy = _pick_strategy(category, history)
        return ErrorDiagnosis(
            error_category=category,
            error_layer=ErrorLayer.VALIDATION,
            root_cause="One or more arguments have the wrong type.",
            actionable_suggestions=suggestions,
            recommended_strategy=strategy,
            original_error=validation_error.message,
            related_tool=tool_spec.name,
            previous_attempts=len(history),
        )

    # Constraint violation (catch-all for valid types but invalid values)
    if "constraint" in msg or "range" in msg or "maximum" in msg or "minimum" in msg:
        category = ErrorCategory.CONSTRAINT_VIOLATION
    else:
        category = ErrorCategory.FORMAT_ERROR
    strategy = _pick_strategy(category, history)
    return ErrorDiagnosis(
        error_category=category,
        error_layer=ErrorLayer.VALIDATION,
        root_cause=validation_error.message,
        actionable_suggestions=[
            f"Review the schema for tool '{tool_spec.name}' and fix the arguments.",
            f"Error detail: {validation_error.message}",
        ],
        recommended_strategy=strategy,
        original_error=validation_error.message,
        related_tool=tool_spec.name,
        previous_attempts=len(history),
    )


def _type_matches(expected_json_type: str, value: Any) -> bool:
    """Check whether a Python value matches a JSON Schema type string."""
    type_map: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list,),
        "object": (dict,),
    }
    expected_types = type_map.get(expected_json_type)
    if expected_types is None:
        return True  # unknown type -> assume ok
    # In Python, bool is a subclass of int; treat booleans strictly.
    if expected_json_type == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, expected_types)


# ---------------------------------------------------------------------------
# Recovery prompt builder
# ---------------------------------------------------------------------------


def build_recovery_prompt(
    diagnosis: ErrorDiagnosis,
    original_goal: str,
    step_context: str | None = None,
) -> str:
    """Build a full recovery prompt for the LLM.  Pure function.

    Combines the diagnosis's own ``to_recovery_prompt()`` output with the
    original goal and optional step context to give the LLM maximum
    information for course-correction.
    """
    parts: list[str] = [diagnosis.to_recovery_prompt()]

    if step_context:
        parts.append(f"\n**Step context**: {step_context}")

    parts.append(f"\n**Your original goal**: {original_goal}")
    parts.append("Please adjust your approach accordingly.")

    return "\n".join(parts)
