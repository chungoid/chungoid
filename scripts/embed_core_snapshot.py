#!/usr/bin/env python
"""Generate a semantic snapshot of the *current* chungoid-core state and embed it to Chroma.

This implements CP3 of the *Core Preparation & Snapshot Infrastructure* roadmap.

The snapshot captures:
• git commit SHA (short)
• `pyproject.toml` version (fallback: importlib.metadata)
• List of MCP tool specs (as returned by `engine.get_mcp_tools()`)
• List of stage YAML files under `server_prompts/stages/`
• Timestamp and author (via git config user.name/email if available)

The structured document is embedded via `dev/scripts/embed_meta.py` into the
`meta_core_snapshot_history` collection.  A synthetic document with
`alias: core_latest` replaces any previous alias entry to provide fast access
to the most recent snapshot.

Usage:

    python dev/scripts/embed_core_snapshot.py --dry-run   # prints YAML only
    python dev/scripts/embed_core_snapshot.py             # embeds to Chroma
"""
from __future__ import annotations

import subprocess
import sys
import json
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List
import textwrap
import os
import typer
import yaml
import importlib
import importlib.metadata as importlib_metadata
from datetime import timezone

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent  # <repo root>
CORE_DIR = WORKSPACE_ROOT / "chungoid-core"

app = typer.Typer(add_help_option=True, no_args_is_help=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(*args: str, cwd: Path = CORE_DIR) -> str:
    """Run a git command and return stdout (strip newlines)."""
    try:
        out = subprocess.check_output(["git", *args], cwd=cwd).decode().strip()
        return out
    except Exception as exc:  # pylint: disable=broad-except
        typer.secho(f"[WARN] git command failed: git {' '.join(args)} -> {exc}", fg=typer.colors.YELLOW)
        return "UNKNOWN"


def _get_core_version() -> str:
    # Prefer pyproject.toml [project] version field to avoid import issues
    pyproject = CORE_DIR / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            if line.strip().startswith("version"):
                return line.split("=")[-1].strip().strip('"')
    # fallback to installed package metadata
    try:
        return importlib_metadata.version("chungoid")
    except importlib_metadata.PackageNotFoundError:
        return "UNKNOWN"


def _get_tool_specs() -> List[Dict[str, Any]]:
    """Import engine and collect tool specs; safe-guard for import errors."""
    try:
        engine_mod = importlib.import_module("chungoid.engine")
        Engine = getattr(engine_mod, "ChungoidEngine")  # type: ignore[attr-defined]
        engine = Engine(project_directory=str(Path.cwd()))
        return engine.get_mcp_tools()  # type: ignore[attr-defined]
    except Exception as exc:  # pylint: disable=broad-except
        typer.secho(f"[WARN] Could not import engine to retrieve tools: {exc}", fg=typer.colors.YELLOW)
        return []


def _get_stage_files() -> List[str]:
    stage_dir = CORE_DIR / "server_prompts" / "stages"
    if not stage_dir.exists():
        return []
    return sorted(str(p.relative_to(CORE_DIR)) for p in stage_dir.glob("*.yaml"))


def _build_snapshot() -> Dict[str, Any]:
    commit = _git("rev-parse", "--short", "HEAD")
    version = _get_core_version()
    author = _git("config", "user.name") or None
    email = _git("config", "user.email") or None

    snapshot: Dict[str, Any] = {
        "type": "core_snapshot",
        "core_commit": commit,
        "core_version": version,
        "created": _dt.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "author": author,
        "author_email": email,
        "stage_files": _get_stage_files(),
        "tool_specs": _get_tool_specs(),
    }
    return snapshot


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def run(
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not embed, just print YAML"),
    tarball: Path | None = typer.Option(None, "--tarball", exists=True, file_okay=True, dir_okay=False, help="Path to a core snapshot tarball (not supported in mono-repo variant)."),
    chroma_host: str = typer.Option("localhost", help="ChromaDB host"),
    chroma_port: int = typer.Option(8000, help="ChromaDB port"),
):
    """Generate snapshot (and optionally embed it)."""
    if tarball:
        typer.secho("[WARN] --tarball ignored in mono-repo embed script", fg=typer.colors.YELLOW)

    snapshot = _build_snapshot()

    yaml_doc = yaml.safe_dump(snapshot, sort_keys=False)

    if dry_run:
        typer.echo("# --- snapshot (dry-run) ---\n" + yaml_doc)
        raise typer.Exit()

    # -------------------------------------------------------------------
    # CI gating – if Chroma connection details are *not* supplied via env
    # (e.g., secrets not configured), perform a silent no-op so the build
    # still passes.  Users can still review the printed YAML via artifact.
    # -------------------------------------------------------------------
    env_chroma_host = os.getenv("CHROMA_HOST")
    env_chroma_port = os.getenv("CHROMA_PORT")

    if not env_chroma_host or not env_chroma_port:
        typer.secho(
            "[INFO] CHROMA secrets missing – skipping embed and printing snapshot only.",
            fg=typer.colors.YELLOW,
        )
        typer.echo(yaml_doc)
        raise typer.Exit()

    # Write to temp file then call embed_meta.py programmatically
    import tempfile
    from importlib import import_module

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_path = Path(tmpd) / f"core_snapshot_{snapshot['core_commit']}.yaml"
        tmp_path.write_text(yaml_doc)
        typer.secho(f"[INFO] Snapshot YAML written to {tmp_path}")

        # Import embed_meta CLI module
        embed_meta = import_module("dev.scripts.embed_meta")  # type: ignore

        # The embed_meta Typer app exposes its entrypoint as `app()`. We can
        # invoke programmatically by building args list.
        args = [
            "add",
            str(tmp_path),
            "--type", "documentation",
            "--collection", "meta_core_snapshot_history",
            "--tags", "core_snapshot,version:{0},commit:{1}".format(snapshot["core_version"], snapshot["core_commit"]),
            "--chroma-host", chroma_host,
            "--chroma-port", str(chroma_port),
        ]
        typer.secho("[INFO] Embedding snapshot via embed_meta.py", fg=typer.colors.GREEN)
        embed_meta.app(args)  # type: ignore[attr-defined]

        # Embed alias doc "core_latest" (small stub referencing current commit)
        alias_doc = {
            "type": "core_snapshot_alias",
            "alias": "core_latest",
            "points_to_commit": snapshot["core_commit"],
            "updated": snapshot["created"],
        }
        alias_yaml = yaml.safe_dump(alias_doc)
        alias_path = Path(tmpd) / "core_latest_alias.yaml"
        alias_path.write_text(alias_yaml)
        embed_meta.app([
            "add",
            str(alias_path),
            "--type", "documentation",
            "--collection", "meta_core_snapshot_history",
            "--tags", "core_snapshot,alias,latest",
            "--force",
            "--chroma-host", chroma_host,
            "--chroma-port", str(chroma_port),
        ])

    typer.secho("[SUCCESS] Core snapshot embedded.", fg=typer.colors.GREEN)


# Provide a root-level wrapper so that the script works both as:
# - `python embed_core_snapshot.py run ...` (explicit `run` subcommand)
# - `python embed_core_snapshot.py ...` (implicit `run` via callback)
@app.callback(invoke_without_command=True)
def _root_callback(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not embed, just print YAML"),
    tarball: Path | None = typer.Option(None, "--tarball", exists=True, file_okay=True, dir_okay=False, help="Path to a core snapshot tarball."),
    chroma_host: str = typer.Option("localhost", "--chroma-host", help="ChromaDB host"),
    chroma_port: int = typer.Option(8000, "--chroma-port", help="ChromaDB port"),
) -> None:  # pragma: no cover
    """Main entrypoint; forwards to `run()` if no subcommand was given."""
    if ctx.invoked_subcommand is None:
        run(
            dry_run=dry_run,
            tarball=tarball,
            chroma_host=chroma_host,
            chroma_port=chroma_port,
        )

if __name__ == "__main__":
    app() 