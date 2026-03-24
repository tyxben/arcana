"""Example 15: Code Review Assistant

A practical demo showing chat + tools + ask_user in a real scenario.
The assistant reviews Python code, can read files, and asks clarifying
questions when needed.

Features demonstrated:
- runtime.chat() for multi-turn review conversation
- @arcana.tool for file reading and code analysis
- ask_user (via input_handler) for clarification
- Budget tracking across the review session

Usage:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python examples/15_code_review_assistant.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import arcana

# Sandbox: all file access is restricted to the project directory.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _safe_path(path: str) -> str:
    """Resolve *path* relative to PROJECT_DIR and reject traversal."""
    resolved = os.path.normpath(os.path.join(PROJECT_DIR, path))
    if not resolved.startswith(PROJECT_DIR):
        raise PermissionError(f"Access denied: {path} is outside the project directory")
    return resolved


# --- Tools ---


@arcana.tool(
    when_to_use="When you need to read source code for review",
    what_to_expect="The full file contents as a string with line numbers",
    failure_meaning="File not found or access denied",
    side_effect="read",
)
def read_file(path: str) -> str:
    """Read a file from the project directory. Returns numbered lines."""
    safe = _safe_path(path)
    if not os.path.isfile(safe):
        return f"Error: '{path}' not found"
    with open(safe) as f:
        lines = f.readlines()
    numbered = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
    return "".join(numbered)


@arcana.tool(
    when_to_use="When you need to find which files exist for review",
    what_to_expect="A newline-separated list of Python file paths",
    failure_meaning="Directory not found or access denied",
    side_effect="read",
)
def list_files(directory: str) -> str:
    """List Python files in a directory (non-recursive)."""
    safe = _safe_path(directory)
    if not os.path.isdir(safe):
        return f"Error: '{directory}' is not a directory"
    entries = sorted(
        f
        for f in os.listdir(safe)
        if f.endswith(".py") and os.path.isfile(os.path.join(safe, f))
    )
    if not entries:
        return "(no Python files found)"
    return "\n".join(entries)


@arcana.tool(
    when_to_use="When you need basic metrics about a file",
    what_to_expect="A summary of total, code, blank, and comment line counts",
    failure_meaning="File not found or access denied",
    side_effect="read",
)
def count_lines(path: str) -> str:
    """Count lines in a file: total, code, blank, and comment lines."""
    safe = _safe_path(path)
    if not os.path.isfile(safe):
        return f"Error: '{path}' not found"
    with open(safe) as f:
        lines = f.readlines()
    total = len(lines)
    blank = sum(1 for line in lines if not line.strip())
    comment = sum(1 for line in lines if line.strip().startswith("#"))
    code = total - blank - comment
    return (
        f"File: {path}\n"
        f"  Total lines:   {total}\n"
        f"  Code lines:    {code}\n"
        f"  Blank lines:   {blank}\n"
        f"  Comment lines: {comment}"
    )


# --- Main ---

SYSTEM_PROMPT = (
    "You are an expert Python code reviewer. When reviewing code:\n"
    "1. Read the file using the read_file tool\n"
    "2. Analyze for bugs, security issues, performance, and style\n"
    "3. If the code's purpose is unclear, ask the user for context\n"
    "4. Provide specific, actionable feedback with line references\n"
    "5. Rate the code quality (1-10) with justification\n\n"
    "Available files are relative to the examples/ directory.\n"
    "Try: read_file('sample_code/calculator.py') to start."
)


async def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Set DEEPSEEK_API_KEY to run this demo.")
        sys.exit(1)

    runtime = arcana.Runtime(
        providers={"deepseek": api_key},
        tools=[read_file, list_files, count_lines],
        budget=arcana.Budget(max_cost_usd=1.0),
        trace=True,
    )

    print("=" * 60)
    print("  Code Review Assistant  (powered by Arcana)")
    print("=" * 60)
    print()
    print("Ask me to review a file. For example:")
    print("  'Review sample_code/calculator.py'")
    print("  'List files in sample_code'")
    print("  'How many lines in sample_code/calculator.py?'")
    print()
    print("Type 'exit' to quit.")
    print()

    async with runtime.chat(
        system_prompt=SYSTEM_PROMPT,
        input_handler=lambda q: input(f"\n[Agent asks] {q}\n> "),
    ) as chat:
        while True:
            try:
                user_input = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user_input.strip().lower() in ("exit", "quit", "q"):
                break

            if not user_input.strip():
                continue

            response = await chat.send(user_input)
            print(f"\nReviewer: {response.content}")
            print(f"  [{response.tokens_used} tokens, ${response.cost_usd:.4f}]")
            print()

        print("-" * 60)
        print(
            f"Session total: {chat.total_tokens} tokens, "
            f"${chat.total_cost_usd:.4f}"
        )
        print("-" * 60)

    await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
