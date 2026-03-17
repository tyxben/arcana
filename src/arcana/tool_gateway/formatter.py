"""Tool description formatting for LLM consumption."""

from __future__ import annotations

from arcana.contracts.tool import SideEffect, ToolSpec


def format_tool_for_llm(spec: ToolSpec) -> str:
    """
    Generate an LLM-optimized description of a tool.

    This is NOT the description sent to OpenAI's function calling API.
    This is the description injected into the system prompt or tool
    preamble that helps the LLM understand when and how to use the tool.

    If affordance fields are present, generates a rich semantic description.
    If affordance fields are absent, falls back to a traditional one-line
    description.

    Pure function, no side effects.
    """
    parts: list[str] = []

    # Core description (always present)
    parts.append(f"**{spec.name}**: {spec.description}")

    # When to use (affordance)
    if spec.when_to_use:
        parts.append(f"  Use when: {spec.when_to_use}")

    # What to expect (affordance)
    if spec.what_to_expect:
        parts.append(f"  Expect: {spec.what_to_expect}")

    # Failure guidance (affordance)
    if spec.failure_meaning:
        parts.append(f"  If it fails: {spec.failure_meaning}")

    # Success guidance (affordance)
    if spec.success_next_step:
        parts.append(f"  After success: {spec.success_next_step}")

    # Side effect warning
    if spec.side_effect == SideEffect.WRITE:
        parts.append("  [WRITE] This tool modifies external state. Use with care.")

    return "\n".join(parts)


def format_tool_list_for_llm(specs: list[ToolSpec]) -> str:
    """
    Format multiple tools into a system prompt section.

    Includes:
    - Tool count and category summary
    - Individual tool descriptions with affordances
    - Note about requesting additional tools

    Pure function, no side effects.
    """
    if not specs:
        return "No tools are currently available."

    sections: list[str] = []
    sections.append(f"You have access to {len(specs)} tools:\n")

    for spec in specs:
        sections.append(format_tool_for_llm(spec))

    sections.append(
        "\nIf you need a capability not covered by these tools, "
        "describe what you need and additional tools may be made available."
    )

    return "\n\n".join(sections)
