"""Trace Web UI — FastAPI app for visual trace inspection."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from arcana.trace.reader import TraceReader

app = FastAPI(title="Arcana Trace Viewer")

# Will be set by serve_traces()
_trace_dir: Path = Path("./traces")


def _reader() -> TraceReader:
    return TraceReader(trace_dir=_trace_dir)


def _esc(value: Any) -> str:
    """HTML-escape a value."""
    return html.escape(str(value))


# ── Shared CSS ──────────────────────────────────────────────────────

_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: #0d1117; color: #c9d1d9; padding: 24px;
    max-width: 1200px; margin: 0 auto;
}
h1 { color: #58a6ff; margin-bottom: 8px; font-size: 1.5rem; }
h2 { color: #58a6ff; margin-bottom: 12px; font-size: 1.2rem; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
.subtitle { color: #8b949e; margin-bottom: 24px; font-size: 0.85rem; }
table { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
th {
    background: #161b22; color: #8b949e; text-align: left;
    padding: 10px 12px; border-bottom: 1px solid #30363d;
    font-weight: 600; font-size: 0.8rem; text-transform: uppercase;
}
td {
    padding: 10px 12px; border-bottom: 1px solid #21262d;
    font-size: 0.85rem;
}
tr:hover { background: #161b22; }
.badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600;
}
.badge-llm { background: #1f3a5f; color: #58a6ff; }
.badge-tool { background: #3b2e00; color: #d29922; }
.badge-error { background: #3d1418; color: #f85149; }
.badge-state { background: #1b3a2d; color: #3fb950; }
.badge-checkpoint { background: #2d1b3a; color: #bc8cff; }
.badge-default { background: #21262d; color: #8b949e; }
.summary-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin-bottom: 24px;
}
.summary-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px; text-align: center;
}
.summary-card .value { font-size: 1.5rem; color: #f0f6fc; font-weight: 700; }
.summary-card .label { font-size: 0.75rem; color: #8b949e; margin-top: 4px; }
.meta { color: #8b949e; font-size: 0.8rem; }
.event-detail { color: #8b949e; font-size: 0.8rem; max-width: 500px; }
.event-detail code {
    background: #161b22; padding: 1px 4px; border-radius: 3px; color: #c9d1d9;
}
.back-link { margin-bottom: 16px; display: inline-block; }
.empty { color: #8b949e; text-align: center; padding: 48px; }
"""


def _layout(title: str, body: str) -> str:
    """Wrap body in the HTML layout."""
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{_esc(title)}</title>
    <style>{_CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def _badge_for_event(event_type: str) -> str:
    """Return a styled badge for an event type."""
    if "llm" in event_type:
        cls = "badge-llm"
    elif "tool" in event_type:
        cls = "badge-tool"
    elif "error" in event_type:
        cls = "badge-error"
    elif "state" in event_type:
        cls = "badge-state"
    elif "checkpoint" in event_type or "complete" in event_type:
        cls = "badge-checkpoint"
    else:
        cls = "badge-default"
    return f'<span class="badge {cls}">{_esc(event_type)}</span>'


def _format_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ── Routes ──────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> str:
    """List all trace files."""
    trace_path = _trace_dir
    if not trace_path.exists():
        return _layout("Arcana Traces", '<p class="empty">No trace directory found.</p>')

    files = sorted(trace_path.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return _layout("Arcana Traces", '<p class="empty">No trace files found.</p>')

    rows = []
    for f in files:
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        run_id = f.stem
        reader = _reader()
        summary = reader.get_summary(run_id)
        events_count = summary.get("total_events", 0)
        llm_count = summary.get("llm_calls", 0)
        tool_count = summary.get("tool_calls", 0)
        tokens = summary.get("total_tokens", 0)
        cost = summary.get("total_cost_usd", 0.0)

        rows.append(f"""\
<tr>
    <td><a href="/trace/{_esc(run_id)}">{_esc(run_id)}</a></td>
    <td>{events_count}</td>
    <td>{llm_count}</td>
    <td>{tool_count}</td>
    <td>{tokens:,}</td>
    <td>${cost:.4f}</td>
    <td>{_format_size(stat.st_size)}</td>
    <td class="meta">{mtime.strftime("%Y-%m-%d %H:%M:%S")}</td>
</tr>""")

    body = f"""\
<h1>Arcana Trace Viewer</h1>
<p class="subtitle">{len(files)} trace(s) in {_esc(str(trace_path))}</p>
<table>
<thead>
<tr>
    <th>Run ID</th><th>Events</th><th>LLM</th><th>Tools</th>
    <th>Tokens</th><th>Cost</th><th>Size</th><th>Modified</th>
</tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>"""

    return _layout("Arcana Traces", body)


@app.get("/trace/{run_id}", response_class=HTMLResponse)
async def trace_detail(run_id: str) -> str:
    """Show trace detail for a specific run."""
    reader = _reader()

    if not reader.exists(run_id):
        return _layout(
            "Trace Not Found",
            f"""\
<a href="/" class="back-link">&larr; Back to traces</a>
<p class="empty">Trace not found: {_esc(run_id)}</p>""",
        )

    summary = reader.get_summary(run_id)
    events = reader.read_events(run_id)

    # Summary cards
    cards = [
        ("Events", summary.get("total_events", 0)),
        ("LLM Calls", summary.get("llm_calls", 0)),
        ("Tool Calls", summary.get("tool_calls", 0)),
        ("Errors", summary.get("errors", 0)),
        ("Steps", summary.get("unique_steps", 0)),
        ("Tokens", f'{summary.get("total_tokens", 0):,}'),
        ("Cost", f'${summary.get("total_cost_usd", 0.0):.4f}'),
    ]
    if summary.get("stop_reason"):
        cards.append(("Stop Reason", summary["stop_reason"]))

    cards_html = "".join(
        f'<div class="summary-card"><div class="value">{_esc(str(v))}</div>'
        f'<div class="label">{_esc(label)}</div></div>'
        for label, v in cards
    )

    # Time range
    time_range = ""
    if summary.get("start_time") and summary.get("end_time"):
        time_range = (
            f'<p class="subtitle">{_esc(summary["start_time"])} &rarr; '
            f'{_esc(summary["end_time"])}</p>'
        )

    # Event timeline rows
    event_rows = []
    for i, event in enumerate(events):
        et = event.event_type.value
        ts = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        model = _esc(event.model or "")

        # Build detail column
        detail_parts: list[str] = []
        if event.tool_call:
            tc = event.tool_call
            detail_parts.append(f"<code>{_esc(tc.name)}</code>")
            if tc.error:
                detail_parts.append(' <span class="badge badge-error">error</span>')
            if tc.duration_ms is not None:
                detail_parts.append(f" ({tc.duration_ms}ms)")
        if event.stop_reason:
            detail_parts.append(f"{_esc(event.stop_reason.value)}")
            if event.stop_detail:
                detail_parts.append(f": {_esc(event.stop_detail)}")
        if event.budgets:
            b = event.budgets
            if b.tokens_used:
                detail_parts.append(f" [{b.tokens_used:,} tok / ${b.cost_usd:.4f}]")

        detail = "".join(detail_parts) if detail_parts else ""

        event_rows.append(f"""\
<tr>
    <td class="meta">{i + 1}</td>
    <td class="meta">{ts}</td>
    <td>{_badge_for_event(et)}</td>
    <td>{model}</td>
    <td class="event-detail">{detail}</td>
</tr>""")

    body = f"""\
<a href="/" class="back-link">&larr; Back to traces</a>
<h1>Trace: {_esc(run_id)}</h1>
{time_range}
<div class="summary-grid">
{cards_html}
</div>
<h2>Event Timeline</h2>
<table>
<thead>
<tr><th>#</th><th>Time</th><th>Type</th><th>Model</th><th>Detail</th></tr>
</thead>
<tbody>
{"".join(event_rows)}
</tbody>
</table>"""

    return _layout(f"Trace: {run_id}", body)


def create_app(trace_dir: str | Path = "./traces") -> FastAPI:
    """Create and configure the trace viewer app.

    Args:
        trace_dir: Directory containing trace JSONL files.

    Returns:
        Configured FastAPI application.
    """
    global _trace_dir  # noqa: PLW0603
    _trace_dir = Path(trace_dir)
    return app


def serve_traces(trace_dir: str | Path = "./traces", port: int = 8100) -> None:
    """Start the trace viewer web server.

    Args:
        trace_dir: Directory containing trace JSONL files.
        port: Port to serve on.
    """
    import uvicorn

    create_app(trace_dir)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
