"""Arcana CLI — Agent Runtime for Production."""
from __future__ import annotations

import asyncio
import json
import os

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


@app.command()
def run(
    goal: str = typer.Argument(..., help="What you want the agent to accomplish"),
    provider: str = typer.Option("deepseek", "--provider", "-p", help="LLM provider"),
    model: str = typer.Option(None, "--model", "-m", help="Model ID"),
    api_key: str = typer.Option(None, "--api-key", "-k", help="API key (or set env var)"),
    max_turns: int = typer.Option(20, "--max-turns", help="Maximum turns"),
    max_cost: float = typer.Option(1.0, "--max-cost", help="Maximum cost in USD"),
    engine: str = typer.Option(
        "conversation", "--engine", "-e", help="Engine: conversation or adaptive"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run an agent task."""
    # Resolve API key
    resolved_key = api_key
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

    asyncio.run(
        _run_agent(
            goal=goal,
            provider=provider,
            model=model,
            api_key=resolved_key,
            max_turns=max_turns,
            max_cost=max_cost,
            engine=engine,
            json_output=json_output,
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
        ),
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
