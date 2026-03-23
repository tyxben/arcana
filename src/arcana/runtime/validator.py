"""Schema validation for LLM outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from arcana.contracts.llm import LLMResponse


class ValidationResult(BaseModel):
    """Result of schema validation."""

    valid: bool
    data: dict[str, Any] | None = None
    errors: list[str] = []
    raw_content: str | None = None


class OutputValidator:
    """
    Validates LLM outputs against expected schemas.

    Supports:
    - JSON schema validation
    - Structured output parsing
    - Retry logic for invalid outputs
    """

    def __init__(self, max_retry_attempts: int = 2) -> None:
        """
        Initialize the validator.

        Args:
            max_retry_attempts: Max attempts to get valid output
        """
        self.max_retry_attempts = max_retry_attempts

    def validate_json(
        self,
        response: LLMResponse,
        schema: type[BaseModel] | None = None,
    ) -> ValidationResult:
        """
        Validate LLM response as JSON.

        Args:
            response: LLM response to validate
            schema: Optional Pydantic model to validate against

        Returns:
            ValidationResult with validation outcome
        """
        import json

        content = response.content
        if not content:
            return ValidationResult(
                valid=False,
                errors=["Empty response content"],
                raw_content=content,
            )

        # Try to parse as JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code block
            data = self._extract_json_from_markdown(content)
            if data is None:
                return ValidationResult(
                    valid=False,
                    errors=[f"Invalid JSON: {e}"],
                    raw_content=content,
                )

        # If schema provided, validate against it
        if schema:
            try:
                validated = schema.model_validate(data)
                return ValidationResult(
                    valid=True,
                    data=validated.model_dump(),
                    raw_content=content,
                )
            except ValidationError as e:
                errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                return ValidationResult(
                    valid=False,
                    errors=errors,
                    data=data,
                    raw_content=content,
                )

        # No schema, just check if valid JSON
        return ValidationResult(
            valid=True,
            data=data,
            raw_content=content,
        )

    def validate_structured_format(
        self,
        response: LLMResponse,
        required_fields: list[str],
    ) -> ValidationResult:
        """
        Validate that response contains required structured fields.

        Expected format:
        Field1: value
        Field2: value

        Args:
            response: LLM response
            required_fields: List of required field names

        Returns:
            ValidationResult
        """
        content = response.content
        if not content:
            return ValidationResult(
                valid=False,
                errors=["Empty response content"],
                raw_content=content,
            )

        # Parse fields
        parsed = self._parse_structured_text(content)

        # Check required fields
        missing = [f for f in required_fields if f not in parsed]
        if missing:
            return ValidationResult(
                valid=False,
                errors=[f"Missing required fields: {', '.join(missing)}"],
                data=parsed,
                raw_content=content,
            )

        return ValidationResult(
            valid=True,
            data=parsed,
            raw_content=content,
        )

    def create_retry_prompt(
        self,
        validation_result: ValidationResult,
        schema_description: str | None = None,
    ) -> str:
        """
        Create a prompt to retry with when validation fails.

        Args:
            validation_result: Failed validation result
            schema_description: Description of expected schema

        Returns:
            Retry prompt message
        """
        errors_str = "\n".join(f"- {err}" for err in validation_result.errors)

        prompt = f"""Your previous response was invalid. Please try again.

Errors found:
{errors_str}

Your previous response:
{validation_result.raw_content}
"""

        if schema_description:
            prompt += f"\n\nExpected format:\n{schema_description}"

        prompt += "\n\nPlease provide a valid response."

        return prompt

    def _extract_json_from_markdown(self, content: str) -> dict[str, Any] | None:
        """
        Try to extract JSON from markdown code block.

        Handles:
        ```json
        {...}
        ```

        or

        ```
        {...}
        ```
        """
        import json
        import re

        # Try to find JSON in code blocks
        patterns = [
            r"```json\s*\n(.*?)\n```",
            r"```\s*\n(\{.*?\})\n```",
            r"```\s*\n(\[.*?\])\n```",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    result: dict[str, Any] | None = json.loads(match.group(1))
                    return result
                except json.JSONDecodeError:
                    continue

        # Try to find any JSON object in text
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                result = json.loads(match)
                return result
            except json.JSONDecodeError:
                continue

        return None

    def _parse_structured_text(self, content: str) -> dict[str, str]:
        """
        Parse structured text format.

        Format:
        Field1: value1
        Field2: value2
        """
        import re

        parsed = {}
        lines = content.split("\n")

        for line in lines:
            # Match "Field: value" pattern
            match = re.match(r"^([A-Za-z_]\w*):\s*(.*)$", line.strip())
            if match:
                key = match.group(1).lower()
                value = match.group(2).strip()
                parsed[key] = value

        return parsed
