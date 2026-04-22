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
    action: str = typer.Argument(..., help="Action: list, show, summary, serve, replay, pool-replay, explain, flow"),
    run_id: str = typer.Argument(None, help="Run ID for show / replay / pool-replay / explain / flow"),
    trace_dir: str = typer.Option("./traces", "--dir", help="Trace directory"),
    port: int = typer.Option(8100, "--port", help="Port for serve"),
    last: int = typer.Option(0, "--last", help="Limit to last N traces (summary)"),
    errors: bool = typer.Option(False, "--errors", help="Show only error events"),
    tools: bool = typer.Option(False, "--tools", help="Show only tool call events"),
    llm: bool = typer.Option(False, "--llm", help="Show only LLM call events"),
    context: bool = typer.Option(False, "--context", help="Show only context decision events"),
    cognitive: bool = typer.Option(False, "--cognitive", help="Show only cognitive primitive events (recall/pin/unpin)"),
    turn: int = typer.Option(None, "--turn", help="Turn number to replay / explain"),
    prompt_only: bool = typer.Option(False, "--prompt-only", help="Replay: only show the prompt snapshot"),
    decision_only: bool = typer.Option(False, "--decision-only", help="Replay: only show the context decision"),
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON instead of formatted output"),
    agent: str = typer.Option(None, "--agent", help="Filter events by pool agent name (metadata.source_agent)"),
    explain: bool = typer.Option(False, "--explain", help="With show --errors: auto-unfold explain view for each error's turn"),
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
        if errors or tools or llm or context or cognitive:
            filter_types = set()
            if errors:
                filter_types.add("error")
            if tools:
                filter_types.add("tool_call")
            if llm:
                filter_types.add("llm_call")
            if context:
                filter_types.add("context_decision")
            if cognitive:
                filter_types.add("cognitive_primitive")

        events = all_events
        if filter_types is not None:
            events = [
                e for e in all_events
                if e.get("event_type", "") in filter_types
            ]
        # v0.8.0 — optional pool-agent scoping via metadata.source_agent
        if agent is not None:
            events = [
                e for e in events
                if (e.get("metadata", {}) or {}).get("source_agent") == agent
            ]

        console.print(f"[bold]Trace: {run_id}[/bold]" + (f" [dim](agent={agent})[/dim]" if agent else ""))
        console.print(f"Events: {len(events)}" + (f" (filtered from {len(all_events)})" if filter_types or agent else ""))
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
            elif "cognitive_primitive" in event_type:
                style = "bright_magenta"
            elif "llm" in event_type:
                style = "cyan"
            elif "tool" in event_type:
                style = "yellow"
            else:
                style = "dim"
            # v0.8.0 — show [agent] tag when the event came from a pool member
            source_agent = (event.get("metadata", {}) or {}).get("source_agent")
            agent_tag = f" [dim]\\[{source_agent}][/dim]" if source_agent else ""
            console.print(
                f"  [{style}]{i + 1:3d}. {event_type:20s}[/{style}]{agent_tag} {timestamp} {model}"
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
                # v0.7.0 — flag pinned entries with [PIN]
                decision = meta.get("context_decision", {})
                if isinstance(decision, dict):
                    for d in decision.get("decisions", []):
                        if d.get("reason") == "pinned":
                            role = d.get("role", "?")
                            before = d.get("token_count_before", 0)
                            console.print(
                                f"       [bright_magenta][PIN][/bright_magenta] "
                                f"[{d.get('index', '?')}] {role} "
                                f"{before} tokens (kept at L0)"
                            )

            # Show cognitive primitive details
            if event_type == "cognitive_primitive" and cognitive:
                meta = event.get("metadata", {})
                primitive = meta.get("primitive", "?")
                args = meta.get("args", {}) or {}
                result = meta.get("result", {}) or {}
                if primitive == "recall":
                    turn_val = args.get("turn")
                    found = result.get("found")
                    msg_count = len(result.get("messages", []) or [])
                    note = result.get("note") or ""
                    status = f"→ {msg_count} messages" if found else f"→ {note}"
                    console.print(
                        f"       [bright_magenta]recall[/bright_magenta] "
                        f"turn={turn_val} {status}"
                    )
                elif primitive == "pin":
                    pid = result.get("pin_id")
                    label = result.get("label") or "(no label)"
                    pinned = result.get("pinned")
                    if pinned:
                        console.print(
                            f"       [bright_magenta]pin[/bright_magenta] "
                            f"pin_id={pid} label={label!r}"
                        )
                    else:
                        reason = result.get("reason", "?")
                        console.print(
                            f"       [red]pin REJECTED[/red] reason={reason}"
                        )
                elif primitive == "unpin":
                    pid = args.get("pin_id")
                    unpinned = result.get("unpinned")
                    console.print(
                        f"       [bright_magenta]unpin[/bright_magenta] "
                        f"pin_id={pid} {'ok' if unpinned else '(unknown id)'}"
                    )

        # --errors --explain: auto-unfold explain view for each error's turn
        if errors and explain:
            from arcana.trace.reader import TraceReader

            reader = TraceReader(trace_dir=trace_path)
            all_typed = reader.read_events(run_id)
            # Map turn step_id → turn number
            turn_by_step = {
                ev.step_id: (ev.metadata or {}).get("step")
                for ev in all_typed if ev.event_type.value == "turn"
            }
            # For each error event, resolve its turn via parent_step_id
            error_events = [
                ev for ev in all_typed if ev.event_type.value == "error"
            ]
            seen_turns: set[int] = set()
            for ev in error_events:
                t_num = turn_by_step.get(ev.parent_step_id or "")
                if not isinstance(t_num, int) or t_num in seen_turns:
                    continue
                seen_turns.add(t_num)
                console.print()
                console.print(f"[bold red]━━ Error at turn {t_num} ━━[/bold red]")
                _trace_explain(
                    trace_path=trace_path,
                    run_id=run_id,
                    turn=t_num,
                    as_json=False,
                )
            if error_events and not seen_turns:
                console.print(
                    "[dim]No error events have parent_step_id linking them to a turn.[/dim]"
                )

    elif action == "replay" and run_id:
        _trace_replay(
            trace_path=trace_path,
            run_id=run_id,
            turn=turn,
            prompt_only=prompt_only,
            decision_only=decision_only,
            as_json=as_json,
            agent=agent,
        )

    elif action == "pool-replay" and run_id:
        _trace_pool_replay(
            trace_path=trace_path,
            run_id=run_id,
            turn=turn,
            agent=agent,
            prompt_only=prompt_only,
            decision_only=decision_only,
            as_json=as_json,
        )

    elif action == "explain" and run_id:
        if turn is None:
            console.print("[red]--turn N is required for explain[/red]")
            raise typer.Exit(1)
        _trace_explain(
            trace_path=trace_path,
            run_id=run_id,
            turn=turn,
            as_json=as_json,
        )

    elif action == "flow" and run_id:
        _trace_flow(
            trace_path=trace_path,
            run_id=run_id,
            as_json=as_json,
        )

    else:
        console.print(
            "[red]Usage: arcana trace list | show <run_id> [--agent NAME] [--errors --explain] | "
            "summary | serve | replay <run_id> --turn N [--agent NAME] | "
            "pool-replay <run_id> [--agent NAME --turn N] | "
            "explain <run_id> --turn N | flow <run_id>[/red]"
        )


def _trace_replay(
    *,
    trace_path: Path,
    run_id: str,
    turn: int | None,
    prompt_only: bool,
    decision_only: bool,
    as_json: bool,
    agent: str | None = None,
) -> None:
    """Print the reconstructed prompt composition for a given turn.

    When ``agent`` is given, only CONTEXT_DECISION / PROMPT_SNAPSHOT
    events whose metadata.source_agent matches are considered — use this
    to replay a specific pool member's turn in an interleaved pool trace.
    """
    from arcana.trace.reader import TraceReader

    reader = TraceReader(trace_dir=trace_path)
    if not reader.exists(run_id):
        console.print(f"[red]Trace not found: {trace_path / (run_id + '.jsonl')}[/red]")
        raise typer.Exit(1)

    if turn is None:
        turns = _list_turns(reader, run_id, agent=agent)
        if not turns:
            scope = f" for agent {agent!r}" if agent else ""
            console.print(
                f"[yellow]No replay evidence in this run{scope} "
                "(no CONTEXT_DECISION or PROMPT_SNAPSHOT events).[/yellow]"
            )
            raise typer.Exit(1)
        console.print(
            "[yellow]--turn is required. Available turns:[/yellow] "
            + ", ".join(str(t) for t in turns)
        )
        raise typer.Exit(2)

    replay = _replay_prompt_scoped(reader, run_id, turn=turn, agent=agent)
    if replay is None:
        scope = f" for agent {agent!r}" if agent else ""
        console.print(f"[red]No replay events for turn {turn}{scope}[/red]")
        raise typer.Exit(1)

    if as_json:
        console.print_json(data=replay.model_dump(mode="json"))
        return

    decision = replay.context_decision
    report = replay.context_report
    snapshot = replay.prompt_snapshot

    header_bits: list[str] = [f"run {run_id}", f"turn {turn}"]
    if agent:
        header_bits.append(f"agent={agent}")
    if decision is not None:
        header_bits.append(f"strategy={decision.strategy}")
    if report is not None:
        header_bits.append(f"utilization={report.utilization:.2%}")
    console.print("[bold]" + " | ".join(header_bits) + "[/bold]")
    console.print()

    if not prompt_only and decision is not None:
        console.print("[cyan]Context decision[/cyan]")
        console.print(f"  {decision.explanation}")
        console.print(
            f"  messages: {decision.messages_in} → {decision.messages_out}, "
            f"compressed={decision.compressed_count}, "
            f"budget={decision.budget_used}/{decision.budget_total}"
        )
        if decision.decisions:
            console.print("  per-message:")
            for d in decision.decisions:
                fid = f" [{d.fidelity}]" if d.fidelity else ""
                score = f" score={d.relevance_score:.2f}" if d.relevance_score is not None else ""
                console.print(
                    f"    [{d.index:>3}] {d.role:<9} {d.outcome:<10}"
                    f" {d.token_count_before:>5}→{d.token_count_after:<5}"
                    f" {d.reason}{fid}{score}"
                )
        console.print()

    if not decision_only and snapshot is not None:
        console.print(f"[cyan]Prompt snapshot[/cyan] (model={snapshot.model})")
        console.print(f"  messages ({len(snapshot.messages)}):")
        for i, msg in enumerate(snapshot.messages):
            role = msg.get("role", "?")
            content = msg.get("content")
            if isinstance(content, str):
                preview = content[:120] + ("..." if len(content) > 120 else "")
            else:
                preview = f"<{type(content).__name__}>"
            console.print(f"    [{i:>3}] {role:<9} {preview}")
        if snapshot.tools:
            console.print(f"  tools ({len(snapshot.tools)}):")
            for t in snapshot.tools:
                console.print(f"    - {t.get('name', t)}")
    elif not decision_only and snapshot is None:
        console.print(
            "[dim]No prompt snapshot recorded. "
            "Enable with RuntimeConfig.trace_include_prompt_snapshots=True.[/dim]"
        )

    # v0.7.0 — derive and show active pin state at this turn from
    # COGNITIVE_PRIMITIVE trace events.
    active_pins = _active_pins_at_turn(reader, run_id, turn, agent=agent)
    if active_pins:
        console.print()
        console.print(f"[cyan]Active pins at turn {turn}[/cyan]")
        for p in active_pins:
            label = p.get("label") or "(no label)"
            pid = p.get("pin_id", "?")
            tokens = p.get("token_count", 0)
            created = p.get("created_turn", "?")
            console.print(
                f"  {pid}  label={label!r}  {tokens} tokens  pinned at turn {created}"
            )


def _active_pins_at_turn(
    reader: Any,  # arcana.trace.reader.TraceReader
    run_id: str,
    turn: int,
    agent: str | None = None,
) -> list[dict[str, Any]]:
    """Replay COGNITIVE_PRIMITIVE events to derive pin state at ``turn``.

    Returns a list of dicts {pin_id, label, token_count, created_turn}.
    Pins whose ``until_turn`` has passed are filtered out. Pins with a
    later unpin event are excluded. Best-effort — malformed events are
    skipped silently. When ``agent`` is given (v0.8.0 pool replay), only
    events whose metadata.source_agent matches are considered — pool
    agents have independent ``PinState``s.
    """
    pins: dict[str, dict] = {}
    try:
        events = reader.read_events(run_id)
    except Exception:  # pragma: no cover — best effort
        return []
    for event in events:
        if event.event_type.value != "cognitive_primitive":
            continue
        meta = event.metadata or {}
        if agent is not None and meta.get("source_agent") != agent:
            continue
        primitive = meta.get("primitive")
        args = meta.get("args") or {}
        result = meta.get("result") or {}
        if primitive == "pin" and result.get("pinned") and not result.get("already_pinned"):
            pid = result.get("pin_id")
            if pid:
                created = result.get("created_turn") or meta.get("turn") or 0
                # Best-effort: derive token count from args content
                content = args.get("content", "") or ""
                from arcana.context.builder import estimate_tokens

                pins[pid] = {
                    "pin_id": pid,
                    "label": result.get("label"),
                    "token_count": estimate_tokens(content),
                    "created_turn": created,
                    "until_turn": args.get("until_turn"),
                }
        elif primitive == "unpin" and result.get("unpinned"):
            pid = args.get("pin_id")
            if pid and pid in pins:
                pins.pop(pid)

    # Filter out expired pins
    active = []
    for pin in pins.values():
        until = pin.get("until_turn")
        if until is not None and until < turn:
            continue
        active.append(pin)
    return active


def _event_agent(event: Any) -> str | None:
    """Best-effort extraction of the pool source_agent from a TraceEvent."""
    meta = getattr(event, "metadata", None) or {}
    value = meta.get("source_agent")
    return value if isinstance(value, str) else None


def _list_turns(reader: Any, run_id: str, agent: str | None = None) -> list[int]:
    """Like ``TraceReader.list_turns`` but scoped to a pool agent.

    ``agent=None`` delegates to the reader unchanged (v0.7.0 behaviour).
    """
    if agent is None:
        return list(reader.list_turns(run_id))

    from arcana.contracts.trace import EventType

    turns: set[int] = set()
    try:
        events = reader.iter_events(run_id)
    except Exception:  # pragma: no cover — best effort
        return []
    for event in events:
        if event.event_type not in (
            EventType.CONTEXT_DECISION,
            EventType.PROMPT_SNAPSHOT,
        ):
            continue
        if _event_agent(event) != agent:
            continue
        meta = event.metadata or {}
        turn = meta.get("turn")
        if turn is None:
            decision = meta.get("context_decision")
            if isinstance(decision, dict):
                turn = decision.get("turn")
        if isinstance(turn, int):
            turns.add(turn)
    return sorted(turns)


def _replay_prompt_scoped(
    reader: Any,
    run_id: str,
    *,
    turn: int,
    agent: str | None,
) -> Any:
    """Like ``TraceReader.replay_prompt`` but scoped to a pool agent.

    When ``agent`` is None, delegates unchanged. When set, only events
    matching ``metadata.source_agent == agent`` participate in replay
    reconstruction.
    """
    if agent is None:
        return reader.replay_prompt(run_id, turn=turn)

    from arcana.contracts.context import ContextDecision, ContextReport
    from arcana.contracts.llm import BudgetSnapshot, PromptSnapshot
    from arcana.contracts.trace import EventType
    from arcana.trace.reader import PromptReplay

    # Local reconstruction mirroring TraceReader.replay_prompt, but with
    # the source_agent filter. Any future field additions to PromptReplay
    # should be mirrored here or (better) pushed into the reader as an
    # optional filter kwarg.

    snapshot: PromptSnapshot | None = None
    decision: ContextDecision | None = None
    report: ContextReport | None = None
    budget: BudgetSnapshot | None = None
    seen = False

    for event in reader.iter_events(run_id):
        if event.event_type not in (
            EventType.CONTEXT_DECISION,
            EventType.PROMPT_SNAPSHOT,
        ):
            continue
        if _event_agent(event) != agent:
            continue
        meta = event.metadata or {}
        event_turn = meta.get("turn")
        if event_turn is None:
            decision_dump = meta.get("context_decision")
            if isinstance(decision_dump, dict):
                event_turn = decision_dump.get("turn")
        if event_turn != turn:
            continue
        seen = True

        if event.event_type == EventType.PROMPT_SNAPSHOT:
            snap = meta.get("prompt_snapshot")
            if isinstance(snap, dict):
                try:
                    snapshot = PromptSnapshot.model_validate(snap)
                except ValueError:
                    snapshot = None
                if snapshot is not None and snapshot.budget_snapshot is not None:
                    budget = snapshot.budget_snapshot
        elif event.event_type == EventType.CONTEXT_DECISION:
            dec = meta.get("context_decision")
            if isinstance(dec, dict):
                try:
                    decision = ContextDecision.model_validate(dec)
                except ValueError:
                    decision = None
            rep = meta.get("context_report")
            if isinstance(rep, dict):
                try:
                    report = ContextReport.model_validate(rep)
                except ValueError:
                    report = None

    if not seen:
        return None
    return PromptReplay(
        run_id=run_id,
        turn=turn,
        prompt_snapshot=snapshot,
        context_decision=decision,
        context_report=report,
        budget_snapshot=budget,
    )


def _pool_agents_in_trace(reader: Any, run_id: str) -> dict[str, int]:
    """Count events per pool agent found in ``run_id``'s trace.

    Agent attribution is via ``metadata.source_agent`` (set by
    :class:`_PoolTaggedTraceWriter` in v0.8.0). Returns an
    agent-name → event-count dict. Events without ``source_agent`` are
    bucketed under ``"(no-agent)"`` — typically present only when pool
    traces are mixed with non-pool events.
    """
    counts: dict[str, int] = {}
    try:
        events = reader.iter_events(run_id)
    except Exception:  # pragma: no cover
        return {}
    for event in events:
        name = _event_agent(event) or "(no-agent)"
        counts[name] = counts.get(name, 0) + 1
    return counts


def _trace_pool_replay(
    *,
    trace_path: Path,
    run_id: str,
    turn: int | None,
    agent: str | None,
    prompt_only: bool,
    decision_only: bool,
    as_json: bool,
) -> None:
    """v0.8.0 — pool-aware replay.

    Without ``--agent``: lists which pool members participated and how
    many events each emitted. With ``--agent``: defers to
    :func:`_trace_replay` scoped to that agent.
    """
    from arcana.trace.reader import TraceReader

    reader = TraceReader(trace_dir=trace_path)
    if not reader.exists(run_id):
        console.print(f"[red]Trace not found: {trace_path / (run_id + '.jsonl')}[/red]")
        raise typer.Exit(1)

    if agent is None:
        counts = _pool_agents_in_trace(reader, run_id)
        if not counts or (len(counts) == 1 and "(no-agent)" in counts):
            console.print(
                "[yellow]No pool events in this trace "
                "(no metadata.source_agent found). "
                "Was the run produced by runtime.collaborate()?[/yellow]"
            )
            raise typer.Exit(1)

        console.print(f"[bold]Pool trace: {run_id}[/bold]")
        console.print()
        table = Table(title="Pool agents")
        table.add_column("Agent")
        table.add_column("Events", justify="right")
        table.add_column("Replayable turns")
        for name in sorted(counts):
            turns = _list_turns(reader, run_id, agent=name if name != "(no-agent)" else None)
            turn_list = ", ".join(str(t) for t in turns) if turns else "(none)"
            table.add_row(name, str(counts[name]), turn_list)
        console.print(table)
        console.print()
        console.print(
            "[dim]Next: arcana trace pool-replay "
            f"{run_id} --agent <name> --turn <N>[/dim]"
        )
        return

    # With --agent → delegate
    _trace_replay(
        trace_path=trace_path,
        run_id=run_id,
        turn=turn,
        prompt_only=prompt_only,
        decision_only=decision_only,
        as_json=as_json,
        agent=agent,
    )


def _trace_explain(
    *,
    trace_path: Path,
    run_id: str,
    turn: int,
    as_json: bool,
) -> None:
    """Single-turn full-story view.

    Prints everything needed to debug one turn: inputs summary, context
    decision, LLM thinking + assistant text, tool calls + results,
    TurnAssessment verdict, budget delta. Degrades gracefully when
    PROMPT_SNAPSHOT is absent (i.e. when ``trace_include_prompt_snapshots``
    was off during the run).
    """
    from arcana.trace.reader import TraceReader

    reader = TraceReader(trace_dir=trace_path)
    if not reader.exists(run_id):
        console.print(f"[red]Trace not found: {trace_path / (run_id + '.jsonl')}[/red]")
        raise typer.Exit(1)

    bundle = reader.collect_turn(run_id, turn)
    turn_event = bundle["turn_event"]
    if turn_event is None:
        available = reader.list_turns(run_id)
        console.print(
            f"[red]No TURN event for turn={turn} in run {run_id}[/red]"
        )
        if available:
            console.print(f"[dim]Turns with events: {available}[/dim]")
        raise typer.Exit(1)

    if as_json:
        console.print(json.dumps(
            {
                "run_id": run_id,
                "turn": turn,
                "turn_event": turn_event.model_dump(mode="json"),
                "context_decision": (
                    bundle["context_decision"].model_dump(mode="json")
                    if bundle["context_decision"] else None
                ),
                "prompt_snapshot": (
                    bundle["prompt_snapshot"].model_dump(mode="json")
                    if bundle["prompt_snapshot"] else None
                ),
                "tool_calls": [e.model_dump(mode="json") for e in bundle["tool_calls"]],
                "cognitive": [e.model_dump(mode="json") for e in bundle["cognitive"]],
                "errors": [e.model_dump(mode="json") for e in bundle["errors"]],
            },
            indent=2,
            default=str,
        ))
        return

    # Header
    meta = turn_event.metadata or {}
    facts = meta.get("facts", {}) or {}
    assessment = meta.get("assessment", {}) or {}
    model = turn_event.model or facts.get("model") or (
        (bundle["prompt_snapshot"].metadata or {}).get("prompt_snapshot", {}).get("model")
        if bundle["prompt_snapshot"] else None
    ) or "?"

    source_agent = (meta.get("source_agent") or "") if meta else ""
    agent_label = f" [dim]\\[{source_agent}][/dim]" if source_agent else ""
    console.print(
        f"[bold]Turn {turn}[/bold] — run [dim]{run_id}[/dim] · model [cyan]{model}[/cyan]{agent_label}"
    )
    console.print(f"[dim]step_id={turn_event.step_id}[/dim]")
    console.print()

    # Inputs (context decision + prompt snapshot)
    console.print("[bold yellow]Inputs[/bold yellow]")
    cd = bundle["context_decision"]
    if cd is not None:
        cdm = cd.metadata or {}
        rep = cdm.get("context_report") or {}
        msgs_in = rep.get("messages_in", "?")
        msgs_out = rep.get("messages_out", "?")
        compressed = rep.get("compressed_count", 0)
        tokens_in = rep.get("input_tokens")
        tokens_out = rep.get("output_tokens")
        line = f"  messages: {msgs_in} → {msgs_out}"
        if compressed:
            line += f"  ({compressed} compressed)"
        if tokens_in is not None and tokens_out is not None:
            line += f"  · tokens: {tokens_in} → {tokens_out}"
        console.print(line)
        explanation = cdm.get("explanation")
        if explanation:
            console.print(f"  [dim]{explanation}[/dim]")
        decision_dump = cdm.get("context_decision") or {}
        if isinstance(decision_dump, dict):
            pinned = [
                d for d in decision_dump.get("decisions", [])
                if d.get("reason") == "pinned"
            ]
            for d in pinned:
                console.print(
                    f"  [bright_magenta][PIN][/bright_magenta] "
                    f"[{d.get('index', '?')}] {d.get('role', '?')} "
                    f"{d.get('token_count_before', 0)} tokens (kept at L0)"
                )
    else:
        console.print("  [dim]<no context decision recorded>[/dim]")

    ps = bundle["prompt_snapshot"]
    if ps is not None:
        snap = (ps.metadata or {}).get("prompt_snapshot") or {}
        messages = snap.get("messages", [])
        tools = snap.get("tools", [])
        console.print(
            f"  prompt: {len(messages)} messages, {len(tools)} tools available"
        )
        for i, m in enumerate(messages[-3:]):  # last 3 messages preview
            role = m.get("role", "?")
            content = m.get("content") or ""
            if isinstance(content, list):
                parts = [
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                content = " ".join(parts)
            preview = (content or "").strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "…"
            console.print(f"    [dim][-{len(messages) - (len(messages)-3) - i}][/dim] [cyan]{role}[/cyan]: {preview}")
    else:
        console.print(
            "  [dim]<prompt snapshot not recorded — enable dev_mode=True or "
            "trace_include_prompt_snapshots=True for full inputs>[/dim]"
        )
    console.print()

    # LLM output
    console.print("[bold cyan]LLM output[/bold cyan]")
    thinking = facts.get("thinking") or ""
    if thinking:
        preview = thinking.strip().replace("\n", " ")
        if len(preview) > 500:
            preview = preview[:500] + "…"
        console.print(f"  [italic dim]thinking:[/italic dim] {preview}")
    assistant_text = facts.get("assistant_text") or ""
    if assistant_text:
        preview = assistant_text.strip().replace("\n", " ")
        if len(preview) > 500:
            preview = preview[:500] + "…"
        console.print(f"  text: {preview}")
    tool_call_requests = facts.get("tool_calls") or []
    if tool_call_requests:
        console.print(f"  tool_calls: {len(tool_call_requests)}")
        for tc in tool_call_requests:
            name = tc.get("name", "?")
            args = tc.get("arguments") or ""
            if len(args) > 120:
                args = args[:120] + "…"
            console.print(f"    [yellow]→ {name}[/yellow]({args})")
    if not thinking and not assistant_text and not tool_call_requests:
        console.print("  [dim]<empty>[/dim]")
    console.print()

    # Tool results
    tool_events = bundle["tool_calls"]
    if tool_events:
        console.print(f"[bold yellow]Tool results[/bold yellow] ({len(tool_events)})")
        for ev in tool_events:
            tc = ev.tool_call
            if tc is None:
                continue
            mark = "[red]✗[/red]" if tc.error else "[green]✓[/green]"
            dur = f"{tc.duration_ms}ms" if tc.duration_ms is not None else ""
            detail = ""
            if tc.error:
                detail = f"  [red]error:[/red] {tc.error}"
            elif tc.result_digest:
                detail = f"  digest={tc.result_digest}"
            console.print(f"  {mark} [yellow]{tc.name}[/yellow] {dur}{detail}")
        console.print()

    # Cognitive
    cog_events = bundle["cognitive"]
    if cog_events:
        console.print("[bold bright_magenta]Cognitive[/bold bright_magenta]")
        for ev in cog_events:
            em = ev.metadata or {}
            prim = em.get("primitive", "?")
            args = em.get("args") or {}
            result = em.get("result") or {}
            if prim == "recall":
                tgt = args.get("turn")
                n = len(result.get("messages", []) or [])
                status = f"→ {n} messages" if result.get("found") else "→ not found"
                console.print(f"  recall turn={tgt} {status}")
            elif prim == "pin":
                if result.get("pinned"):
                    console.print(f"  pin pin_id={result.get('pin_id')} label={result.get('label')!r}")
                else:
                    console.print(f"  [red]pin REJECTED[/red] reason={result.get('reason')}")
            elif prim == "unpin":
                ok = result.get("unpinned")
                console.print(f"  unpin pin_id={args.get('pin_id')} {'ok' if ok else '(unknown id)'}")
        console.print()

    # Verdict
    console.print("[bold green]Runtime verdict[/bold green]")
    completed = assessment.get("completed", False)
    failed = assessment.get("failed", False)
    confidence = assessment.get("confidence")
    reason = assessment.get("completion_reason")
    console.print(f"  completed: {completed}   failed: {failed}")
    if confidence is not None:
        console.print(f"  confidence: {confidence}")
    if reason:
        console.print(f"  completion_reason: {reason}")

    # Errors attached to this turn
    err_events = bundle["errors"]
    if err_events:
        console.print()
        console.print(f"[bold red]Errors[/bold red] ({len(err_events)})")
        for ev in err_events:
            em = ev.metadata or {}
            msg = em.get("message") or em.get("error") or str(em)
            console.print(f"  [red]{msg}[/red]")


def _trace_flow(
    *,
    trace_path: Path,
    run_id: str,
    as_json: bool,
) -> None:
    """ASCII DAG of a run — turn → tools → turn chain.

    Walks the causal chain via parent_step_id. Falls back to turn number
    ordering when parent links are missing (legacy traces).
    """
    from arcana.contracts.trace import EventType
    from arcana.trace.reader import TraceReader

    reader = TraceReader(trace_dir=trace_path)
    if not reader.exists(run_id):
        console.print(f"[red]Trace not found: {trace_path / (run_id + '.jsonl')}[/red]")
        raise typer.Exit(1)

    events = reader.read_events(run_id)
    if not events:
        console.print("[dim]<empty trace>[/dim]")
        return

    turn_events = [e for e in events if e.event_type == EventType.TURN]
    turn_events.sort(key=lambda e: (e.metadata or {}).get("step", 0))

    tools_by_parent: dict[str, list[Any]] = {}
    for e in events:
        if e.event_type == EventType.TOOL_CALL and e.parent_step_id:
            tools_by_parent.setdefault(e.parent_step_id, []).append(e)

    stop_reason = None
    for e in reversed(events):
        if e.stop_reason:
            stop_reason = e.stop_reason.value
            break
    if stop_reason is None and turn_events:
        last_assessment = (turn_events[-1].metadata or {}).get("assessment") or {}
        if last_assessment.get("completed"):
            stop_reason = "completed"
        elif last_assessment.get("failed"):
            stop_reason = "failed"

    if as_json:
        payload = {
            "run_id": run_id,
            "turns": [
                {
                    "turn": (e.metadata or {}).get("step"),
                    "step_id": e.step_id,
                    "parent_step_id": e.parent_step_id,
                    "model": e.model,
                    "tool_calls": [
                        {
                            "name": te.tool_call.name if te.tool_call else None,
                            "error": te.tool_call.error if te.tool_call else None,
                            "duration_ms": te.tool_call.duration_ms if te.tool_call else None,
                        }
                        for te in tools_by_parent.get(e.step_id, [])
                    ],
                }
                for e in turn_events
            ],
            "stop": stop_reason,
        }
        console.print(json.dumps(payload, indent=2, default=str))
        return

    console.print(f"[bold]Flow[/bold] — run [dim]{run_id}[/dim] · {len(turn_events)} turns")
    console.print()
    for i, e in enumerate(turn_events):
        step = (e.metadata or {}).get("step", i + 1)
        assessment = (e.metadata or {}).get("assessment") or {}
        facts = (e.metadata or {}).get("facts") or {}
        n_tool_calls = len(facts.get("tool_calls") or [])
        status = ""
        if assessment.get("completed"):
            status = " [green](completed)[/green]"
        elif assessment.get("failed"):
            status = " [red](failed)[/red]"
        console.print(
            f"  [cyan]Turn {step}[/cyan]{status}  [dim]{e.model or '?'}[/dim]"
        )
        tool_events = tools_by_parent.get(e.step_id, [])
        for te in tool_events:
            tc = te.tool_call
            if tc is None:
                continue
            mark = "[red]✗[/red]" if tc.error else "[green]✓[/green]"
            dur = f" {tc.duration_ms}ms" if tc.duration_ms is not None else ""
            console.print(f"    ├─ {mark} [yellow]{tc.name}[/yellow]{dur}")
        # Legacy fallback: show claimed tool call count when no parent_step_id links
        if not tool_events and n_tool_calls:
            console.print(
                f"    ├─ [dim]{n_tool_calls} tool calls (legacy trace, no parent_step_id)[/dim]"
            )
        if i < len(turn_events) - 1:
            console.print("    │")
    if stop_reason:
        console.print(f"  [dim]→ stop: {stop_reason}[/dim]")


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
