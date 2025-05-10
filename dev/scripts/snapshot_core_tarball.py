"""CLI to package chungoid-core into a versioned tarball.

This is *Phase-4, Task P4.1* – initial skeleton that supports `--dry-run`
and minimal tarball creation logic.  Functionality will be expanded in
subsequent commits but the current version is sufficient for unit tests
and CI smoke-runs.

Usage (from repository root):

    python chungoid-core/dev/scripts/snapshot_core_tarball.py --dry-run
"""
from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import os
import sys
import tarfile
from pathlib import Path
from typing import List, Set

import typer

app = typer.Typer(add_help_option=True)

# Default patterns/directories to exclude from the snapshot
_DEFAULT_EXCLUDES: Set[str] = {
    ".git",
    ".venv",
    "dev_chroma_db",
    "tests",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".env",
}

_MAX_SIZE_MB_DEFAULT = 1500


def _should_include(path: Path, exclude_patterns: Set[str]) -> bool:
    """Return *True* if *path* should be included based on *exclude_patterns*."""
    rel = str(path)
    for pattern in exclude_patterns:
        # Glob-style match (filename) or prefix check (directory)
        if path.match(pattern) or rel.startswith(f"{pattern}{os.sep}"):
            return False
    return True


def _create_tarball(
    root: Path, output: Path, exclude_patterns: Set[str], gzip: bool = True
) -> None:
    """Create a tarball from *root* at *output* honouring *exclude_patterns*."""
    mode = "w:gz" if gzip else "w"
    with tarfile.open(output, mode) as tar:
        for fs_path in root.rglob("*"):
            if not _should_include(fs_path.relative_to(root), exclude_patterns):
                continue
            tar.add(fs_path, arcname=str(fs_path.relative_to(root)))


@app.command()
def _cli(
    dry_run: bool = typer.Option(
        False, help="Preview files to be archived without creating the tarball."
    ),
    output_dir: Path = typer.Option(
        Path("dist"), exists=False, help="Directory for the generated tarball."
    ),
    include_hash: bool = typer.Option(
        False, help="Write a <tarball>.sha256 file alongside the archive."
    ),
    gzip: bool = typer.Option(True, help="Compress the archive using gzip."),
    max_size_mb: float = typer.Option(
        _MAX_SIZE_MB_DEFAULT,
        help="Fail if the resulting archive exceeds this size (MB). Accepts fractions like 0.01.",
    ),
    exclude: List[str] | None = typer.Option(
        None, help="Additional glob patterns or directories to exclude."
    ),
) -> None:  # pragma: no cover – Typer entry-point
    """Package **chungoid-core** into a reproducible tarball.

    The snapshot captures the *current working directory* by default. Run this
    script from the repository root for predictable results.
    """

    repo_root = Path.cwd()
    if (repo_root / "pyproject.toml").exists() is False:
        typer.secho(
            "[warning] Running outside repository root – tarball contents may be unexpected.",
            fg=typer.colors.YELLOW,
        )

    excludes: Set[str] = set(_DEFAULT_EXCLUDES)
    if exclude:
        excludes.update(exclude)

    date_tag = _dt.datetime.utcnow().strftime("%Y%m%d")
    tar_name = (
        f"core_snapshot_{date_tag}.tar.gz" if gzip else f"core_snapshot_{date_tag}.tar"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / tar_name

    if dry_run:
        typer.echo(
            f"[dry-run] Would create {tar_path} with excludes: {sorted(excludes)}"
        )
        sys.exit(0)

    _create_tarball(repo_root, tar_path, excludes, gzip=gzip)

    # Size check
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        tar_path.unlink(missing_ok=True)
        typer.secho(
            f"[ERROR] Archive size {size_mb:.1f} MB exceeds limit of {max_size_mb} MB.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    if include_hash:
        sha256 = _hashlib.sha256(tar_path.read_bytes()).hexdigest()
        hash_path = tar_path.with_suffix(tar_path.suffix + ".sha256")
        hash_path.write_text(sha256 + "\n")
        typer.echo(f"Wrote SHA-256 digest → {hash_path}")

    typer.secho(f"Snapshot created: {tar_path}", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# Root callback (so tests can call CLI without subcommand)
# ---------------------------------------------------------------------------

# The tests expect to invoke the Typer application directly, e.g. `app --dry-run`.
# Register a root-level callback that simply forwards to the real implementation
# above so that no explicit sub-command is required.


@app.callback(invoke_without_command=True)
def _root_callback(
    ctx: typer.Context,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview files to be archived without creating the tarball.")
    ,
    output_dir: Path = typer.Option(
        Path("dist"), "--output-dir", exists=False, help="Directory for the generated tarball.",
    ),
    include_hash: bool = typer.Option(
        False, "--include-hash", help="Write a <tarball>.sha256 file alongside the archive.",
    ),
    gzip: bool = typer.Option(True, "--gzip/--no-gzip", help="Compress the archive using gzip."),
    max_size_mb: float = typer.Option(
        _MAX_SIZE_MB_DEFAULT, "--max-size-mb", help="Fail if the resulting archive exceeds this size (MB)."
    ),
    exclude: List[str] | None = typer.Option(
        None, "--exclude", help="Additional glob patterns or directories to exclude."
    ),
):
    """Root command wrapper – delegates to *_cli* to preserve test expectations."""

    # When Typer is invoked without subcommands, *ctx.invoked_subcommand* is None.
    # We only forward to the real implementation in that case.
    if ctx.invoked_subcommand is None:
        _cli(  # type: ignore[arg-types]
            dry_run=dry_run,
            output_dir=output_dir,
            include_hash=include_hash,
            gzip=gzip,
            max_size_mb=max_size_mb,
            exclude=exclude,
        )


if __name__ == "__main__":  # pragma: no cover
    app() 