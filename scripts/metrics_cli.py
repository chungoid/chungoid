"""metrics_cli – local helper to inspect execution metrics.

Example::

    metrics query --since 1h --status error
    metrics tail --limit 10  # follow newest events (one-shot)

The CLI simply wraps `chungoid.utils.metrics_store.MetricsStore`.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer

from chungoid.utils.metrics_store import MetricsStore

app = typer.Typer(add_help_option=True, no_args_is_help=True)

# Allow overriding project root so CLI can be used from anywhere
_DEFAULT_ROOT = Path(os.getenv("CHUNGOID_PROJECT_ROOT", Path.cwd()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_since(value: str | None) -> Optional[datetime]:
    if value is None:
        return None
    value = value.strip()
    # duration format like "1h", "30m", "10s"
    if value[-1] in {"h", "m", "s"} and value[:-1].isdigit():
        amount = int(value[:-1])
        unit = value[-1]
        if unit == "h":
            return datetime.now(timezone.utc) - timedelta(hours=amount)
        if unit == "m":
            return datetime.now(timezone.utc) - timedelta(minutes=amount)
        if unit == "s":
            return datetime.now(timezone.utc) - timedelta(seconds=amount)
    # Try ISO 8601
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        typer.echo("[error] Invalid --since value. Use e.g. '1h' or ISO timestamp.", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def query(
    run_id: Optional[str] = typer.Option(None, help="Filter by run_id"),
    stage_id: Optional[str] = typer.Option(None, help="Filter by stage_id"),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    since: Optional[str] = typer.Option(None, help="e.g. 1h, 30m, ISO timestamp"),
    limit: int = typer.Option(100, help="Maximum number of rows to return"),
    project_root: Path = typer.Option(_DEFAULT_ROOT, exists=False, help="Project root (auto-detected)"),
):
    """Query metric events with simple filters and print JSON to stdout."""

    since_dt = _parse_since(since)
    store = MetricsStore(project_root=project_root, chroma_mode="persistent")
    rows = store.query(run_id=run_id, stage_id=stage_id, status=status, since=since_dt, limit=limit)
    typer.echo(json.dumps([r.model_dump() for r in rows], indent=2, default=str))


@app.command()
def tail(
    limit: int = typer.Option(20, help="Show N most recent events"),
    project_root: Path = typer.Option(_DEFAULT_ROOT, exists=False, help="Project root (auto-detected)"),
):
    """Print the last *limit* metric events ordered by timestamp descending."""

    store = MetricsStore(project_root=project_root, chroma_mode="persistent")
    # naive: peek bigger than limit and sort; good enough for CLI
    rows = store.query(limit=limit * 5)
    rows.sort(key=lambda e: e.timestamp, reverse=True)
    typer.echo(json.dumps([r.model_dump() for r in rows[:limit]], indent=2, default=str))


if __name__ == "__main__":
    app() 