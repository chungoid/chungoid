from __future__ import annotations

"""Command-line interface for Chungoid-core.

This Click-based CLI provides a friendlier wrapper around the existing
`chungoid.mcp` entry-point, surfacing a few high-level sub-commands that
map to the most common actions developers perform when interacting with
Chungoid from the shell.

Examples
--------
$ chungoid run .            # run next stage for current dir
$ chungoid init my_proj     # initialise a fresh project skeleton
$ chungoid status           # print project status (pretty or JSON)

Auto-completion can be installed via `chungoid --install-completion` once
this script is on your `$PATH` (Click handles the plumbing).
"""

from pathlib import Path
import json
import logging
import sys
from typing import Any

import click

from chungoid.engine import ChungoidEngine
from chungoid.utils.state_manager import StateManager, StatusFileError

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# ---------------------------------------------------------------------------
# Root Click group
# ---------------------------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(_LOG_LEVELS, case_sensitive=False),
    help="Logging verbosity (default: INFO)",
    show_default=True,
)
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:  # noqa: D401 – imperative mood fine
    """Chungoid command-line interface."""
    _configure_logging(log_level.upper())
    # Expose common values to sub-commands via click context obj
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level.upper()


# ---------------------------------------------------------------------------
# Sub-command: run (execute next stage)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument(
    "project_dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=False, path_type=Path),
    default=".",
)
@click.option("--json", "as_json", is_flag=True, help="Emit result as JSON")
@click.pass_context
def run(ctx: click.Context, project_dir: Path, as_json: bool) -> None:  # noqa: D401
    """Execute the next stage for *PROJECT_DIR* (default: current directory)."""
    logger = logging.getLogger("chungoid.cli.run")
    project_path = project_dir.expanduser().resolve()
    logger.info("Running next stage for project: %s", project_path)

    try:
        engine = ChungoidEngine(str(project_path))
        result: dict[str, Any] = engine.run_next_stage()
    except Exception as exc:  # pragma: no cover – surfacing
        logger.exception("Failed to run next stage: %s", exc)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        # Pretty print using click's echo and json dumps for deterministic output
        click.echo(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Sub-command: init (bootstrap project skeleton)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument(
    "project_dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=False, path_type=Path),
    required=True,
)
@click.pass_context
def init(ctx: click.Context, project_dir: Path) -> None:  # noqa: D401
    """Initialise a new Chungoid project in *PROJECT_DIR*."""
    logger = logging.getLogger("chungoid.cli.init")
    project_path = project_dir.expanduser().resolve()

    if project_path.exists() and any(project_path.iterdir()):  # directory not empty
        click.confirm(
            f"Directory {project_path} is not empty. Continue and create .chungoid/ anyway?",
            abort=True,
        )
    project_path.mkdir(parents=True, exist_ok=True)

    chungoid_dir = project_path / ".chungoid"
    chungoid_dir.mkdir(exist_ok=True)

    status_file = chungoid_dir / "project_status.json"
    if not status_file.exists():
        status_file.write_text(json.dumps({"runs": []}, indent=2))
        logger.debug("Created fresh status file at %s", status_file)

    click.echo(f"✓ Initialised Chungoid project at {project_path}")
    click.echo("You can now run `chungoid run` to start Stage 0")


# ---------------------------------------------------------------------------
# Sub-command: status (print project status)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument(
    "project_dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    default=".",
)
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON status")
@click.pass_context
def status(ctx: click.Context, project_dir: Path, as_json: bool) -> None:  # noqa: D401
    """Show full status for *PROJECT_DIR* (default: current directory)."""
    logger = logging.getLogger("chungoid.cli.status")
    project_path = project_dir.expanduser().resolve()

    # Locate stages directory relative to package root
    core_root = Path(__file__).resolve().parent
    stages_dir = core_root / "server_prompts" / "stages"

    try:
        sm = StateManager(
            target_directory=str(project_path),
            server_stages_dir=str(stages_dir),
            use_locking=False,
        )
        status_data = sm.get_full_status()
    except (StatusFileError, ValueError) as exc:
        logger.error("Unable to read status: %s", exc)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(status_data, indent=2))
    else:
        # Simple human-readable summary
        runs = status_data.get("runs", [])
        if not runs:
            click.echo("(no runs recorded yet)")
            return
        click.echo("Runs (newest first):")
        for run in reversed(runs):
            click.echo(
                f"  • Stage {run.get('stage')} – {run.get('status')} – {run.get('timestamp', '')}"
            )


# ---------------------------------------------------------------------------
# Entrypoint for `python -m chungoid.cli` (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    cli() 