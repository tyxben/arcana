"""Arcana CLI — Agent Runtime for Production."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="arcana",
    help="Arcana — Agent Runtime for Production",
    no_args_is_help=True,
)
console = Console()


def _load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML agent config file."""
    import yaml  # type: ignore[import-untyped]

    p = Path(path)
    if not p.exists():
        typer.echo(f"Error: Config file not found: {path}", err=False)
        raise typer.Exit(1)
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        typer.echo(f"Error: Invalid config: expected a YAML mapping, got {type(data).__name__}", err=False)
        raise typer.Exit(1)
    return data


@app.command()
def run(
    goal: str = typer.Argument(..., help="Goal string or path to .yaml/.yml config file"),
    provider: str = typer.Option(None, "--provider", "-p", help="LLM provider"),
    model: str = typer.Option(None, "--model", "-m", help="Model ID"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="API key (or set env var)"),
    max_turns: int = typer.Option(None, "--max-turns", help="Maximum turns"),
    max_cost: float = typer.Option(None, "--max-cost", help="Maximum cost in USD"),
    engine: str = typer.Option(None, "--engine", "-e", help="Engine: conversation or adaptive"),
    override: str = typer.Option(None, "--override", help="Override goal from config file"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run an agent task.

    Usage:
        arcana run "What is 2+2?"
        arcana run agent.yaml
        arcana run agent.yaml --override "Override this goal"
        arcana run agent.yaml --provider openai
    """
    # ── Load YAML config if goal is a config file ────────────────
    yaml_cfg: dict[str, Any] = {}
    effective_goal = goal

    if goal.endswith((".yaml", ".yml")):
        yaml_cfg = _load_yaml_config(goal)
        effective_goal = override or yaml_cfg.get("goal", "")
        if not effective_goal:
            typer.echo("Error: No goal specified in config or command line.", err=False)
            raise typer.Exit(1)

    # ── Merge: CLI flags override YAML, YAML overrides defaults ──
    effective_provider = provider or yaml_cfg.get("provider", "deepseek")
    effective_model = model or yaml_cfg.get("model")
    effective_max_turns = max_turns if max_turns is not None else yaml_cfg.get("max_turns", 20)
    effective_max_cost = max_cost if max_cost is not None else yaml_cfg.get("max_cost", 1.0)
    effective_engine = engine or yaml_cfg.get("engine", "conversation")
    effective_system_prompt = yaml_cfg.get("system_prompt")
    effective_memory = yaml_cfg.get("memory", False)
    effective_trace = yaml_cfg.get("trace", False)

    # ── Resolve API key ──────────────────────────────────────────
    resolved_key = api_key or yaml_cfg.get("api_key", "")
    if not resolved_key:
        env_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "kimi": "KIMI_API_KEY",
            "glm": "GLM_API_KEY",
            "minimax": "MINIMAX_API_KEY",
        }
        env_var = env_map.get(effective_provider, f"{effective_provider.upper()}_API_KEY")
        resolved_key = os.environ.get(env_var, "")
        if not resolved_key and effective_provider != "ollama":
            console.print("[red]Error: API key required.[/red]")
            console.print(f"Pass --api-key or set {env_var}")
            raise typer.Exit(1)

    asyncio.run(
        _run_agent(
            goal=effective_goal,
            provider=effective_provider,
            model=effective_model,
            api_key=resolved_key,
            max_turns=effective_max_turns,
            max_cost=effective_max_cost,
            engine=effective_engine,
            json_output=json_output,
            system_prompt=effective_system_prompt,
            memory=effective_memory,
            trace=effective_trace,
        )
    )


async def _run_agent(
    goal: str,
    provider: str,
    model: str | None,
    api_key: str,
    max_turns: int,
    max_cost: float,
    engine: str,
    json_output: bool,
    system_prompt: str | None = None,
    memory: bool = False,
    trace: bool = False,
) -> None:
    """Execute the agent task."""
    from arcana.runtime_core import Budget, Runtime, RuntimeConfig

    if not json_output:
        console.print(
            Panel(f"[bold]{goal}[/bold]", title="Arcana", subtitle=f"{provider} | {engine}")
        )

    rt = Runtime(
        providers={provider: api_key},
        budget=Budget(max_cost_usd=max_cost),
        config=RuntimeConfig(
            default_provider=provider,
            default_model=model,
            max_turns=max_turns,
            system_prompt=system_prompt,
        ),
        memory=memory,
        trace=trace,
    )

    try:
        result = await rt.run(goal, engine=engine, max_turns=max_turns)
    except Exception as e:
        if json_output:
            print(json.dumps({"error": str(e), "success": False}))
        else:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None

    if json_output:
        print(
            json.dumps(
                {
                    "output": result.output,
                    "success": result.success,
                    "steps": result.steps,
                    "tokens": result.tokens_used,
                    "cost_usd": result.cost_usd,
                    "run_id": result.run_id,
                },
                ensure_ascii=False,
            )
        )
    else:
        console.print()
        if result.success:
            console.print(
                Panel(
                    str(result.output),
                    title="[green]Result[/green]",
                    subtitle=f"{result.steps} steps | {result.tokens_used} tokens | ${result.cost_usd:.4f}",
                )
            )
        else:
            console.print(
                Panel(
                    str(result.output) or "No output",
                    title="[yellow]Incomplete[/yellow]",
                    subtitle=f"{result.steps} steps | {result.tokens_used} tokens",
                )
            )


@app.command()
def trace(
    action: str = typer.Argument(..., help="Action: list, show, summary, serve"),
    run_id: str = typer.Argument(None, help="Run ID for show"),
    trace_dir: str = typer.Option("./traces", "--dir", help="Trace directory"),
    port: int = typer.Option(8100, "--port", help="Port for serve"),
    last: int = typer.Option(0, "--last", help="Limit to last N traces (summary)"),
    errors: bool = typer.Option(False, "--errors", help="Show only error events"),
    tools: bool = typer.Option(False, "--tools", help="Show only tool call events"),
    llm: bool = typer.Option(False, "--llm", help="Show only LLM call events"),
    context: bool = typer.Option(False, "--context", help="Show only context decision events"),
) -> None:
    """View agent execution traces."""
    import datetime
    from pathlib import Path

    trace_path = Path(trace_dir)

    if action == "serve":
        try:
            from arcana.trace.web import serve_traces
        except ImportError:
            console.print("[red]Error: UI dependencies not installed.[/red]")
            console.print("Install with: pip install arcana-agent[ui]")
            raise typer.Exit(1) from None
        console.print(f"[bold]Starting Trace Viewer[/bold] on http://127.0.0.1:{port}")
        console.print(f"Trace dir: {trace_dir}")
        serve_traces(trace_dir=trace_dir, port=port)
        return

    if action == "list":
        if not trace_path.exists():
            console.print(f"[dim]No traces found in {trace_dir}[/dim]")
            return

        files = sorted(trace_path.glob("*.jsonl"), reverse=True)
        if not files:
            console.print("[dim]No trace files found[/dim]")
            return

        table = Table(title="Traces")
        table.add_column("Run ID")
        table.add_column("File")
        table.add_column("Size")
        table.add_column("Modified")

        for f in files[:20]:
            mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
            table.add_row(
                f.stem,
                f.name,
                f"{f.stat().st_size:,} bytes",
                mtime.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(table)

    elif action == "summary":
        _trace_summary(trace_path, last)

    elif action == "show" and run_id:
        trace_file = trace_path / f"{run_id}.jsonl"
        if not trace_file.exists():
            console.print(f"[red]Trace not found: {trace_file}[/red]")
            raise typer.Exit(1)

        all_events = []
        with open(trace_file) as fh:
            for line in fh:
                if line.strip():
                    all_events.append(json.loads(line))

        # Apply filters
        filter_types: set[str] | None = None
        if errors or tools or llm or context:
            filter_types = set()
            if errors:
                filter_types.add("error")
            if tools:
                filter_types.add("tool_call")
            if llm:
                filter_types.add("llm_call")
            if context:
                filter_types.add("context_decision")

        events = all_events
        if filter_types is not None:
            events = [
                e for e in all_events
                if e.get("event_type", "") in filter_types
            ]

        console.print(f"[bold]Trace: {run_id}[/bold]")
        console.print(f"Events: {len(events)}" + (f" (filtered from {len(all_events)})" if filter_types else ""))
        console.print()

        for i, event in enumerate(events):
            event_type = event.get("event_type", "unknown")
            timestamp = event.get("timestamp", "")[:19]
            model = event.get("model", "")

            if "complete" in event_type:
                style = "green"
            elif "error" in event_type:
                style = "red"
            elif "context_decision" in event_type:
                style = "magenta"
            elif "llm" in event_type:
                style = "cyan"
            elif "tool" in event_type:
                style = "yellow"
            else:
                style = "dim"
            console.print(
                f"  [{style}]{i + 1:3d}. {event_type:20s}[/{style}] {timestamp} {model}"
            )

            # Show context decision details
            if event_type == "context_decision" and context:
                meta = event.get("metadata", {})
                explanation = meta.get("explanation", "")
                compressed = meta.get("compressed_count", 0)
                console.print(f"       [dim]{explanation}[/dim]")
                if compressed > 0:
                    msgs_in = meta.get("messages_in", 0)
                    msgs_out = meta.get("messages_out", 0)
                    console.print(f"       [dim]messages: {msgs_in} → {msgs_out} ({compressed} compressed)[/dim]")

    else:
        console.print("[red]Usage: arcana trace list | show <run_id> | summary | serve[/red]")


def _trace_summary(trace_path: Path, last: int) -> None:
    """Display aggregate metrics from trace files."""
    from arcana.contracts.trace import TraceEvent
    from arcana.observability.metrics import MetricsCollector

    if not trace_path.exists():
        console.print(f"[dim]No traces found in {trace_path}[/dim]")
        return

    files = sorted(trace_path.glob("*.jsonl"), reverse=True)
    if last > 0:
        files = files[:last]
    if not files:
        console.print("[dim]No trace files found[/dim]")
        return

    summaries = []
    for f in files:
        events = []
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    events.append(TraceEvent.model_validate_json(line))
                except Exception:
                    pass  # Skip malformed lines
        if events:
            summaries.append(MetricsCollector.summarize_run(events))

    if not summaries:
        console.print("[dim]No valid traces found[/dim]")
        return

    agg = MetricsCollector.aggregate(summaries)

    console.print(f"[bold]Trace Summary[/bold] ({agg.count} runs)")
    console.print()

    table = Table(show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total runs", str(agg.count))
    table.add_row("Avg tokens", f"{agg.avg_tokens:,.0f}")
    table.add_row("Avg cost", f"${agg.avg_cost_usd:.4f}")
    table.add_row("Avg duration", f"{agg.avg_duration_ms:,.0f} ms")
    table.add_row("P95 duration", f"{agg.p95_duration_ms:,.0f} ms")
    table.add_row("Success rate", f"{agg.success_rate:.0%}")
    table.add_row("Error rate", f"{agg.error_rate:.0%}")

    if agg.error_type_counts:
        table.add_row("Error types", ", ".join(
            f"{k}: {v}" for k, v in agg.error_type_counts.items()
        ))

    console.print(table)


@app.command()
def chat(
    provider: str = typer.Option("deepseek", "--provider", "-p", help="LLM provider"),
    model: str = typer.Option(None, "--model", "-m", help="Model override"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="API key (or set env var)"),
    system: str = typer.Option(None, "--system", "-s", help="System prompt"),
    budget: float = typer.Option(10.0, "--budget", help="Max cost in USD"),
    trace: bool = typer.Option(False, "--trace", help="Enable tracing"),
) -> None:
    """Start an interactive chat session with an Arcana agent.

    Usage:
        arcana chat
        arcana chat --provider openai --model gpt-4o
        arcana chat --system "You are a math tutor"
        arcana chat --budget 5.0 --trace
    """
    asyncio.run(_chat_session(
        provider=provider,
        model=model,
        api_key=api_key,
        system_prompt=system,
        budget=budget,
        trace=trace,
    ))


async def _chat_session(
    provider: str,
    model: str | None,
    api_key: str | None,
    system_prompt: str | None,
    budget: float,
    trace: bool,
) -> None:
    """Run an interactive chat loop."""
    from arcana.runtime_core import Budget, Runtime, RuntimeConfig

    # Resolve API key
    resolved_key = api_key or ""
    if not resolved_key:
        env_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "kimi": "KIMI_API_KEY",
            "glm": "GLM_API_KEY",
            "minimax": "MINIMAX_API_KEY",
        }
        env_var = env_map.get(provider, f"{provider.upper()}_API_KEY")
        resolved_key = os.environ.get(env_var, "")
        if not resolved_key and provider != "ollama":
            console.print("[red]Error: API key required.[/red]")
            console.print(f"Pass --api-key or set {env_var}")
            raise typer.Exit(1)

    rt = Runtime(
        providers={provider: resolved_key},
        budget=Budget(max_cost_usd=budget),
        config=RuntimeConfig(
            default_provider=provider,
            default_model=model,
        ),
        trace=trace,
    )

    console.print(Panel(
        f"Provider: [bold]{provider}[/bold] | Budget: [bold]${budget:.2f}[/bold]"
        + (f" | System: [dim]{system_prompt[:60]}...[/dim]" if system_prompt and len(system_prompt) > 60 else f" | System: [dim]{system_prompt}[/dim]" if system_prompt else ""),
        title="[bold]Arcana Chat[/bold]",
        subtitle="Type 'exit' or Ctrl-C to end",
    ))
    console.print()

    # Chat state: accumulate messages for multi-turn context
    from arcana.contracts.llm import (
        LLMRequest,
        Message,
        MessageRole,
    )

    model_config = rt._resolve_model_config()
    messages: list[Message] = []
    if system_prompt:
        messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))

    total_tokens = 0
    total_cost = 0.0
    turn_count = 0

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if user_input.strip().lower() in ("exit", "quit", "q"):
            break

        if not user_input.strip():
            continue

        messages.append(Message(role=MessageRole.USER, content=user_input))

        try:
            request = LLMRequest(messages=messages)
            response = await rt._gateway.generate(
                request=request,
                config=model_config,
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            # Remove the failed user message so conversation stays consistent
            messages.pop()
            continue

        assistant_text = response.content or ""
        messages.append(Message(role=MessageRole.ASSISTANT, content=assistant_text))

        tokens = response.usage.total_tokens if response.usage else 0
        cost = response.usage.cost_estimate if response.usage else 0.0
        total_tokens += tokens
        total_cost += cost
        turn_count += 1

        console.print(f"[bold green]Agent:[/bold green] {assistant_text}")
        console.print(f"[dim]({tokens} tokens, ${cost:.4f})[/dim]")
        console.print()

        # Budget check
        if total_cost >= budget:
            console.print("[yellow]Budget limit reached.[/yellow]")
            break

    console.print(Panel(
        f"Turns: {turn_count} | Tokens: {total_tokens:,} | Cost: ${total_cost:.4f}",
        title="Session Summary",
    ))

    await rt.close()


@app.command()
def version() -> None:
    """Show Arcana version."""
    from importlib.metadata import version as get_version

    try:
        v = get_version("arcana-agent")
    except Exception:
        v = "dev"
    console.print(f"arcana {v}")


@app.command()
def providers() -> None:
    """List supported providers."""
    data = [
        ("deepseek", "DEEPSEEK_API_KEY", "deepseek-chat", "Verified"),
        ("openai", "OPENAI_API_KEY", "gpt-4o-mini", "Verified"),
        ("anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-20250514", "Verified"),
        ("kimi", "KIMI_API_KEY", "moonshot-v1-8k", "Supported"),
        ("glm", "GLM_API_KEY", "glm-4-flash", "Supported"),
        ("minimax", "MINIMAX_API_KEY", "abab6.5s-chat", "Supported"),
        ("gemini", "GEMINI_API_KEY", "gemini-2.0-flash", "Supported"),
        ("ollama", "(none)", "llama3.2", "Supported"),
    ]
    table = Table(title="Supported Providers")
    table.add_column("Provider")
    table.add_column("Env Variable")
    table.add_column("Default Model")
    table.add_column("Status")
    for name, env, default_model, status in data:
        style = "green" if status == "Verified" else "dim"
        table.add_row(name, env, default_model, f"[{style}]{status}[/{style}]")
    console.print(table)


@app.command()
def init(
    directory: str = typer.Argument(".", help="Directory to initialize (default: current)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
) -> None:
    """Scaffold a starter Arcana agent project.

    Usage:
        arcana init
        arcana init my-agent
        arcana init --force
    """
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {
        "main.py": _TEMPLATE_MAIN,
        ".env.example": _TEMPLATE_ENV,
        "agent.yaml": _TEMPLATE_AGENT_YAML,
    }

    created: list[str] = []
    skipped: list[str] = []

    for filename, content in files.items():
        filepath = target / filename
        if filepath.exists() and not force:
            skipped.append(filename)
            continue
        filepath.write_text(content)
        created.append(filename)

    console.print()
    if created:
        for f in created:
            console.print(f"  [green]\u2713[/green] {f}")
    if skipped:
        for f in skipped:
            console.print(f"  [yellow]\u2013[/yellow] {f} [dim](exists, use --force to overwrite)[/dim]")

    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Copy .env.example to .env and add your API key")
    console.print("  2. Run your agent:")
    console.print("     [dim]arcana run agent.yaml[/dim]")
    console.print("     [dim]python main.py[/dim]")


_TEMPLATE_MAIN = '''\
"""My Arcana Agent."""
import arcana


@arcana.tool(
    when_to_use="When you need to do math calculations",
    what_to_expect="Returns the numeric result",
)
def calc(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))


async def main():
    runtime = arcana.Runtime(
        providers={"deepseek": "your-api-key-here"},
        tools=[calc],
        budget=arcana.Budget(max_cost_usd=1.0),
    )

    result = await runtime.run("What is 42 * 17 + 5?")
    print(result.output)
    print(f"Cost: ${result.cost_usd:.4f}")

    await runtime.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
'''

_TEMPLATE_ENV = '''\
# LLM Provider API Keys
DEEPSEEK_API_KEY=your-key-here
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
'''

_TEMPLATE_AGENT_YAML = '''\
# Arcana Agent Configuration
# Run with: arcana run agent.yaml

goal: "Hello! What can you help me with?"
provider: deepseek
max_turns: 10
max_cost: 1.0
engine: conversation
# system_prompt: "You are a helpful assistant."
# trace: true
'''


@app.command(name="eval")
def eval_cmd(
    action: str = typer.Argument("run", help="Action: run"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run the eval gate suite.

    Usage:
        arcana eval run
        arcana eval run --category tool_use
        arcana eval run --json
    """
    if action != "run":
        console.print("[red]Usage: arcana eval run [--category CAT][/red]")
        raise typer.Exit(1)

    asyncio.run(_run_eval(category=category, json_output=json_output))


async def _run_eval(
    *,
    category: str | None = None,
    json_output: bool = False,
) -> None:
    """Execute the eval suite with mock provider."""
    from arcana.eval.baseline import EvalGate, build_baseline_cases

    gate = EvalGate()
    for case in build_baseline_cases():
        gate.register(case)

    if not json_output:
        console.print("[bold]Running eval suite...[/bold]")
        if category:
            console.print(f"  Category filter: {category}")
        console.print()

    report = await gate.run_all(category=category)

    if json_output:
        import json as json_mod

        print(json_mod.dumps({
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "score": report.score,
            "by_category": report.by_category,
            "results": [
                {"name": name, "passed": r.passed, "score": r.score,
                 "detail": r.detail, "duration_ms": r.duration_ms}
                for name, r in report.results
            ],
        }, ensure_ascii=False, indent=2))
    else:
        # Results table
        table = Table(title="Eval Results")
        table.add_column("Case", style="bold")
        table.add_column("Status")
        table.add_column("Score")
        table.add_column("Duration")
        table.add_column("Detail")

        for name, r in report.results:
            status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            table.add_row(
                name,
                status,
                f"{r.score:.1%}",
                f"{r.duration_ms}ms",
                r.detail[:60],
            )
        console.print(table)
        console.print()

        # Category summary
        for cat, rate in report.by_category.items():
            style = "green" if rate == 1.0 else "yellow" if rate >= 0.5 else "red"
            console.print(f"  [{style}]{cat}: {rate:.0%}[/{style}]")

        console.print()
        style = "green" if report.score >= 0.9 else "yellow" if report.score >= 0.5 else "red"
        console.print(
            f"[{style}]Overall: {report.passed}/{report.total} passed "
            f"({report.score:.0%})[/{style}]"
        )

        if report.failed > 0:
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
