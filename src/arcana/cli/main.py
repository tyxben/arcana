"""Arcana CLI — Agent Runtime for Production."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

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


def _load_yaml_config(path: str) -> dict:
    """Load a YAML agent config file."""
    import yaml

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
    yaml_cfg: dict = {}
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
    action: str = typer.Argument(..., help="Action: list, show, serve"),
    run_id: str = typer.Argument(None, help="Run ID for show"),
    trace_dir: str = typer.Option("./traces", "--dir", help="Trace directory"),
    port: int = typer.Option(8100, "--port", help="Port for serve"),
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

    elif action == "show" and run_id:
        trace_file = trace_path / f"{run_id}.jsonl"
        if not trace_file.exists():
            console.print(f"[red]Trace not found: {trace_file}[/red]")
            raise typer.Exit(1)

        events = []
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))

        console.print(f"[bold]Trace: {run_id}[/bold]")
        console.print(f"Events: {len(events)}")
        console.print()

        for i, event in enumerate(events):
            event_type = event.get("event_type", "unknown")
            timestamp = event.get("timestamp", "")[:19]
            model = event.get("model", "")

            if "complete" in event_type:
                style = "green"
            elif "llm" in event_type:
                style = "cyan"
            elif "tool" in event_type:
                style = "yellow"
            else:
                style = "dim"
            console.print(
                f"  [{style}]{i + 1:3d}. {event_type:20s}[/{style}] {timestamp} {model}"
            )

    else:
        console.print("[red]Usage: arcana trace list  OR  arcana trace show <run_id>[/red]")


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


if __name__ == "__main__":
    app()
