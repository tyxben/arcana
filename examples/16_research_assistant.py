"""Example 16: Research Assistant with Multi-Agent Team

A practical demo showing tool use + team collaboration + structured output.
Three phases work together to research a topic:

Phase 1 - Research: A single agent uses search_web and save_note tools
          to gather raw findings (demonstrates runtime.run() + tools)
Phase 2 - Team Analysis: Three agents collaborate in a team discussion
          to analyze, critique, and synthesize the findings
          (demonstrates runtime.team() multi-agent collaboration)
Phase 3 - Structure: The final output is parsed into a typed report
          (demonstrates arcana.run() with response_format)

Features demonstrated:
- @arcana.tool for simulated web search and note-taking
- runtime.run() for tool-using agent execution
- runtime.team() for multi-agent collaboration
- arcana.run() with response_format for structured output
- Budget tracking across all phases

Usage:
    export DEEPSEEK_API_KEY=sk-xxx
    uv run python examples/16_research_assistant.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from pydantic import BaseModel

import arcana

# ---------------------------------------------------------------------------
# Simulated search database -- deterministic results for demo reliability
# ---------------------------------------------------------------------------

_SEARCH_DB: dict[str, list[str]] = {
    "agent": [
        "Agent frameworks in 2026 have shifted toward runtime-governed architectures "
        "where the framework controls budget, tool access, and safety while the LLM "
        "focuses purely on reasoning. This separation of concerns has become the "
        "dominant design pattern.",
        "Multi-agent collaboration is now standard in production systems. Teams of "
        "specialized agents (researcher, coder, reviewer) coordinate through shared "
        "conversation histories managed by the runtime, replacing fragile hand-coded "
        "orchestration pipelines.",
        "The 'agentic coding' wave has matured. Most agent frameworks now provide "
        "built-in tool gateways with authentication, rate limiting, and audit logging "
        "rather than raw function calling.",
    ],
    "llm": [
        "Large language models in 2026 have largely converged on the 'reasoning + "
        "tool use' paradigm. Extended thinking capabilities (chain-of-thought visible "
        "to the runtime) allow frameworks to assess confidence without prompt hacking.",
        "Cost of inference has dropped 10x since 2024, making multi-agent workflows "
        "economically viable for production use cases. DeepSeek and open-weight models "
        "have driven much of this cost reduction.",
        "Prompt caching and context compression have become essential runtime features. "
        "Frameworks that manage context windows intelligently see 40-60% cost savings "
        "on long-running agent tasks.",
    ],
    "framework": [
        "The framework landscape has consolidated around three approaches: (1) graph-based "
        "orchestration (LangGraph), (2) runtime-governed agents (Arcana), and "
        "(3) code-generation agents (OpenAI Codex, Devin). Each serves different use cases.",
        "Key differentiators in 2026 frameworks include: context management strategy, "
        "budget enforcement, tool safety guarantees, and multi-agent coordination "
        "primitives. Raw LLM wrapper libraries have largely been replaced.",
        "Evaluation and reproducibility have become first-class concerns. Production "
        "frameworks now include built-in trace logging, replay capabilities, and "
        "automated evaluation suites.",
    ],
    "trend": [
        "Three major trends define AI agent development in 2026: (1) runtime-level "
        "safety controls replacing prompt-based guardrails, (2) structured output as "
        "a standard capability rather than an afterthought, and (3) multi-modal agents "
        "that process text, images, and code in unified workflows.",
        "The 'contracts-first' design philosophy has gained traction. Frameworks define "
        "data schemas before implementation, enabling cross-language portability -- "
        "agents designed in Python can be deployed in Rust or Go runtimes.",
        "Enterprise adoption has shifted focus from 'can agents do X' to 'can we audit, "
        "budget, and govern agents doing X.' Observability and cost control are now the "
        "primary selection criteria for agent platforms.",
    ],
}

_DEFAULT_RESULTS = [
    "AI research continues to advance rapidly in 2026, with particular focus on "
    "practical deployment of agent systems in enterprise environments.",
    "The open-source AI ecosystem has matured significantly, with standardized "
    "interfaces and interoperability between different frameworks and providers.",
]

# Shared note storage
_notes: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@arcana.tool(
    when_to_use="When you need to find information about a topic",
    what_to_expect="Returns search result snippets relevant to the query",
)
def search_web(query: str) -> str:
    """Search the web for information on a topic."""
    query_lower = query.lower()
    results: list[str] = []

    for keyword, snippets in _SEARCH_DB.items():
        if keyword in query_lower:
            results.extend(snippets)

    if not results:
        results = _DEFAULT_RESULTS

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for r in results:
        if r not in seen:
            seen.add(r)
            unique.append(r)

    formatted = "\n\n".join(
        f"[Result {i + 1}] {snippet}" for i, snippet in enumerate(unique[:5])
    )
    return f"Search results for '{query}':\n\n{formatted}"


@arcana.tool(
    when_to_use="When you want to record a finding for later use",
    what_to_expect="Confirms the note was saved",
)
def save_note(title: str, content: str) -> str:
    """Save a research note for later reference."""
    _notes[title] = content
    return f"Note saved: '{title}' ({len(content)} chars)"


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------


class ResearchReport(BaseModel):
    """Structured research report."""

    title: str
    key_findings: list[str]
    trends: list[str]
    conclusion: str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Set DEEPSEEK_API_KEY to run this example.")
        sys.exit(1)

    topic = "How are LLM agent frameworks evolving in 2026?"

    print("=" * 60)
    print("  Research Assistant Demo")
    print("=" * 60)
    print(f"\nTopic: {topic}\n")

    runtime = arcana.Runtime(
        providers={"deepseek": api_key},
        tools=[search_web, save_note],
        budget=arcana.Budget(max_cost_usd=2.0),
    )

    # ------------------------------------------------------------------
    # Phase 1: Gather raw research with tool-using agent
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Phase 1: Gathering research (agent + tools)")
    print("-" * 60)

    research_result = await runtime.run(
        f"Research this topic thoroughly: {topic}\n\n"
        "Instructions:\n"
        "1. Search for information about LLM agents, frameworks, and 2026 trends\n"
        "2. Make at least 3 different searches to cover multiple angles\n"
        "3. Save your key findings as notes using save_note\n"
        "4. End with a summary of everything you found",
        max_turns=15,
    )

    print(f"  Steps: {research_result.steps}")
    print(f"  Cost:  ${research_result.cost_usd:.4f}")
    print(f"  Notes saved: {len(_notes)}")
    for title in _notes:
        print(f"    - {title}")

    raw_findings = research_result.output or "No findings gathered."
    print(f"\n  Raw findings preview: {str(raw_findings)[:200]}...")

    # ------------------------------------------------------------------
    # Phase 2: Team analysis -- three agents discuss the findings
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print("Phase 2: Team analysis (multi-agent collaboration)")
    print("-" * 60)

    # Build context from Phase 1 for the team
    notes_text = "\n".join(f"- {t}: {c}" for t, c in _notes.items())
    team_context = (
        f"Research findings on '{topic}':\n\n"
        f"{raw_findings}\n\n"
        f"Saved notes:\n{notes_text}"
    )

    team_result = await runtime.team(
        goal=(
            f"Analyze these research findings and produce a comprehensive summary.\n\n"
            f"{team_context}"
        ),
        agents=[
            arcana.AgentConfig(
                name="researcher",
                prompt=(
                    "You are a thorough researcher. Review the findings provided. "
                    "Highlight the most important data points and flag anything that "
                    "seems incomplete or needs deeper investigation. Share your "
                    "assessment with the team."
                ),
            ),
            arcana.AgentConfig(
                name="analyst",
                prompt=(
                    "You are a critical analyst. Review the researcher's assessment. "
                    "Identify the top 3-5 key trends, note any contradictions or gaps "
                    "in the findings, and provide a structured analysis. Organize your "
                    "thoughts clearly for the writer."
                ),
            ),
            arcana.AgentConfig(
                name="writer",
                prompt=(
                    "You are a concise technical writer. Synthesize the researcher's "
                    "findings and the analyst's structured analysis into a clear, "
                    "well-organized summary. Include: key findings, major trends, and "
                    "a brief conclusion. End with [DONE] when your summary is complete."
                ),
            ),
        ],
        max_rounds=3,
    )

    print(f"  Success: {team_result.success}")
    print(f"  Rounds:  {team_result.rounds}")
    print(f"  Cost:    ${team_result.total_cost_usd:.4f}")

    print("\n  Conversation log:")
    for entry in team_result.conversation_log:
        preview = entry["content"][:150].replace("\n", " ")
        print(f"    [Round {entry['round']}] {entry['agent']}: {preview}...")

    # ------------------------------------------------------------------
    # Phase 3: Structure the output with response_format
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print("Phase 3: Structuring report (structured output)")
    print("-" * 60)

    final_text = team_result.output or str(raw_findings)

    structured = await arcana.run(
        f"Structure the following research into a report with exactly these fields:\n"
        f"- title: A concise title for the report\n"
        f"- key_findings: A list of 3-5 key findings (one sentence each)\n"
        f"- trends: A list of 3-5 major trends (one sentence each)\n"
        f"- conclusion: A 2-3 sentence conclusion\n\n"
        f"Research to structure:\n{final_text}",
        response_format=ResearchReport,
        provider="deepseek",
        api_key=api_key,
        max_cost_usd=0.50,
    )

    print(f"  Cost: ${structured.cost_usd:.4f}")

    if isinstance(structured.output, ResearchReport):
        report = structured.output
        print()
        print("=" * 60)
        print(f"  {report.title}")
        print("=" * 60)
        print("\n  Key Findings:")
        for finding in report.key_findings:
            print(f"    * {finding}")
        print("\n  Trends:")
        for trend in report.trends:
            print(f"    * {trend}")
        print(f"\n  Conclusion:\n    {report.conclusion}")
    else:
        print(f"\n  Raw output: {structured.output}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_cost = (
        research_result.cost_usd
        + team_result.total_cost_usd
        + structured.cost_usd
    )
    print()
    print("=" * 60)
    print("  Cost Summary")
    print("=" * 60)
    print(f"  Phase 1 (research):   ${research_result.cost_usd:.4f}")
    print(f"  Phase 2 (team):       ${team_result.total_cost_usd:.4f}")
    print(f"  Phase 3 (structure):  ${structured.cost_usd:.4f}")
    print(f"  Total:                ${total_cost:.4f}")

    await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
