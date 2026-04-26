"""Argument validation for tool calls against JSON Schema."""

from __future__ import annotations

from typing import Any

from arcana.contracts.tool import ToolError, ToolErrorCategory, ToolSpec

# JSON Schema type to Python type mapping
_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> ToolError | None:
    """
    Validate tool call arguments against the ToolSpec's input_schema.

    Checks:
    - Required fields are present
    - Top-level field types match schema
    - Arguments is a dict

    Returns:
        None if valid, ToolError with VALIDATION category if invalid.
    """
    schema = spec.input_schema
    if not schema:
        return None

    errors: list[str] = []

    # Check required fields
    missing = check_required_fields(schema, arguments)
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")

    # Check types
    type_errors = check_types(schema, arguments)
    errors.extend(type_errors)

    if errors:
        return ToolError(
            category=ToolErrorCategory.VALIDATION,
            message=f"Validation failed for tool '{spec.name}': {'; '.join(errors)}",
            code="VALIDATION_ERROR",
            details={"errors": errors},
        )

    return None


def check_required_fields(schema: dict[str, Any], data: dict[str, Any]) -> list[str]:
    """Return list of missing required fields."""
    required = schema.get("required", [])
    return [field for field in required if field not in data]


def check_types(schema: dict[str, Any], data: dict[str, Any]) -> list[str]:
    """Return list of type mismatch descriptions."""
    properties = schema.get("properties", {})
    errors: list[str] = []

    for field_name, field_schema in properties.items():
        if field_name not in data:
            continue

        expected_type = field_schema.get("type")
        if not expected_type:
            continue

        value = data[field_name]
        python_type = _TYPE_MAP.get(expected_type)
        if python_type is None:
            continue

        if not isinstance(value, python_type):
            errors.append(
                f"Field '{field_name}': expected {expected_type}, "
                f"got {type(value).__name__}"
            )

    return errors
