"""CLI to package chungoid-core into a versioned tarball.

This is *Phase-4, Task P4.1* – initial skeleton that supports `--dry-run`
and minimal tarball creation logic.  Functionality will be expanded in
subsequent commits but the current version is sufficient for unit tests
and CI smoke-runs.

Usage (from repository root):

    python chungoid-core/scripts/snapshot_core_tarball.py --dry-run
"""
from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import os
import sys
import tarfile
from pathlib import Path
from typing import List, Set, Optional

import typer

app = typer.Typer(add_help_option=True)

# Default patterns/directories to exclude from the snapshot
_DEFAULT_EXCLUDES: Set[str] = {
    ".git",
    ".venv",
    "dev_chroma_db", # This might be relevant if chungoid-core has its own dev_chroma_db inside its folder
    "tests", # Usually tests are excluded from distributable tarballs, but sometimes included for sdist
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".env",
    ".pytest_cache",
    "htmlcov",
    "*.egg-info", # Exclude build artifacts
    "build",
    "dist", # Exclude output directory itself from being included recursively if script run from core root
    ".chungoid" # Exclude local chungoid data like chroma_db within the core project
}

_MAX_SIZE_MB_DEFAULT = 1500


def _should_include(path: Path, exclude_patterns: Set[str]) -> bool:
    """Return *True* if *path* should be included based on *exclude_patterns*."""
    rel_path_str = str(path)
    for pattern in exclude_patterns:
        if path.match(pattern): # Glob-style match for files/dirs at current level
            return False
        # Check if path is inside an excluded directory (e.g. pattern is "foo", path is "foo/bar/baz.txt")
        if rel_path_str.startswith(pattern + os.sep) or rel_path_str == pattern:
            return False
    return True


def _create_tarball(
    root: Path, output: Path, exclude_patterns: Set[str], gzip: bool = True
) -> None:
    """Create a tarball from *root* at *output* honouring *exclude_patterns*."""
    mode = "w:gz" if gzip else "w"
    with tarfile.open(output, mode) as tar:
        for fs_path in root.rglob("*"):
            relative_path = fs_path.relative_to(root)
            if not _should_include(relative_path, exclude_patterns):
                # print(f"Excluding: {relative_path}") # For debugging
                continue
            # print(f"Including: {relative_path}") # For debugging
            tar.add(fs_path, arcname=str(relative_path))


def find_repo_root(start_path: Path) -> Optional[Path]:
    """Search upwards from start_path to find the metachungoid repository root."""
    current = start_path.resolve()
    while current != current.parent:
        if (
            (current / ".git").is_dir()
            and (current / "pyproject.toml").is_file() # Meta-project pyproject.toml
            and (current / "chungoid-core").is_dir()
        ):
            return current
        current = current.parent
    # Check the last path (filesystem root)
    if (
        (current / ".git").is_dir()
        and (current / "pyproject.toml").is_file()
        and (current / "chungoid-core").is_dir()
    ):
        return current
    return None


@app.command()
def _cli(
    dry_run: bool = typer.Option(
        False, help="Preview files to be archived without creating the tarball."
    ),
    output_dir_override: Optional[Path] = typer.Option(
        None, "--output-dir", help="Override default output directory (repo_root/dist)."
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
        None, help="Additional glob patterns or directories to exclude (appends to defaults)."
    ),
    # New option to specify the core project directory if not automatically found or different
    core_dir_override: Optional[Path] = typer.Option(
        None, "--core-dir", help="Explicitly set the chungoid-core directory to package."
    )
) -> None:  # pragma: no cover – Typer entry-point
    """Package **chungoid-core** into a reproducible tarball.

    The script attempts to find the monorepo root and then targets the `chungoid-core`
    subdirectory within it for packaging.
    """

    initial_cwd = Path.cwd()
    repo_root_found = find_repo_root(initial_cwd)

    if repo_root_found is None:
        typer.secho(
            f"[WARNING] Could not determine the metachungoid repository root from {initial_cwd}. Assuming CWD is the monorepo root or inside it.",
            fg=typer.colors.YELLOW,
        )
        # Fallback: assume current working directory or its parent might be repo root if .git exists
        # This part is tricky; for a script *inside* chungoid-core/scripts, it might be better to
        # target its own parent (chungoid-core) directly unless explicitly told otherwise.
        # For now, we require this script to be run from a context where find_repo_root works, or core_dir_override is used.
        repo_root_found = initial_cwd # Default to CWD if find_repo_root fails, and hope for the best or rely on core_dir_override

    core_project_root: Path
    if core_dir_override:
        core_project_root = core_dir_override.resolve()
        if not core_project_root.is_dir():
            typer.secho(f"[ERROR] Explicitly provided --core-dir not found or not a directory: {core_project_root}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        # If core_dir_override is given, repo_root_found might not be relevant for actual packaging path.
        # But it is still used for default output_dir. Let's try to set repo_root_found to parent of core_project_root if possible.
        if (core_project_root.parent / "pyproject.toml").exists() and (core_project_root.parent / ".git").exists():
            repo_root_found = core_project_root.parent

    elif repo_root_found and (repo_root_found / "chungoid-core").is_dir():
        core_project_root = repo_root_found / "chungoid-core"
    else:
        # If find_repo_root failed and no override, and we are in chungoid-core/scripts, then core_project_root is parent.
        # Path(__file__) is chungoid-core/scripts/snapshot_core_tarball.py
        # Path(__file__).parent is chungoid-core/scripts/
        # Path(__file__).parent.parent is chungoid-core/
        script_parent_dir = Path(__file__).resolve().parent
        assumed_core_root = script_parent_dir.parent
        if assumed_core_root.name == "chungoid-core" and (assumed_core_root / "src").is_dir():
            core_project_root = assumed_core_root
            typer.secho(f"[INFO] Assuming chungoid-core directory is: {core_project_root}", fg=typer.colors.BLUE)
            if repo_root_found == initial_cwd: # If find_repo_root defaulted to CWD, try to set repo_root_found more reliably
                if (core_project_root.parent / "pyproject.toml").exists() and (core_project_root.parent / ".git").exists():
                     repo_root_found = core_project_root.parent
        else:
            typer.secho(
                f"[ERROR] chungoid-core directory could not be determined. Neither found via monorepo structure from {initial_cwd}, nor explicitly provided via --core-dir, nor inferred from script location. Please run from within the monorepo or use --core-dir.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    final_output_dir = output_dir_override if output_dir_override else (repo_root_found / "dist" if repo_root_found else initial_cwd / "dist")

    excludes_set: Set[str] = set(_DEFAULT_EXCLUDES)
    if exclude:
        excludes_set.update(exclude)

    date_tag = _dt.datetime.utcnow().strftime("%Y%m%d")
    # Use core_project_root.name for the tarball name prefix, e.g. "chungoid-core"
    tar_name_prefix = core_project_root.name 
    tar_name = (
        f"{tar_name_prefix}_snapshot_{date_tag}.tar.gz" if gzip else f"{tar_name_prefix}_snapshot_{date_tag}.tar"
    )
    final_output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = final_output_dir / tar_name

    typer.echo(f"Preparing to package: {core_project_root}")
    typer.echo(f"Output target: {tar_path}")
    typer.echo(f"Exclusion patterns: {sorted(excludes_set)}")

    if dry_run:
        typer.echo("[dry-run] Files that would be included:")
        item_count = 0
        for fs_path in core_project_root.rglob("*"):
            relative_path = fs_path.relative_to(core_project_root)
            if _should_include(relative_path, excludes_set):
                typer.echo(f"  + {relative_path}")
                item_count +=1
        typer.echo(f"[dry-run] Total items to include: {item_count}")
        typer.echo(f"[dry-run] Tarball creation skipped.")
        sys.exit(0)

    _create_tarball(core_project_root, tar_path, excludes_set, gzip=gzip)

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

    typer.secho(f"Snapshot created: {tar_path} ({size_mb:.2f} MB)", fg=typer.colors.GREEN)


@app.callback(invoke_without_command=True)
def _root_callback(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_dir_override: Optional[Path] = typer.Option(None, "--output-dir"),
    include_hash: bool = typer.Option(False, "--include-hash"),
    gzip: bool = typer.Option(True, "--gzip/--no-gzip"), # Allow --no-gzip
    max_size_mb: float = typer.Option(_MAX_SIZE_MB_DEFAULT, "--max-size-mb"),
    exclude: List[str] | None = typer.Option(None, "--exclude"),
    core_dir_override: Optional[Path] = typer.Option(None, "--core-dir")
):
    """Root command wrapper for packaging chungoid-core."""
    if ctx.invoked_subcommand is None:
        _cli(
            dry_run=dry_run,
            output_dir_override=output_dir_override,
            include_hash=include_hash,
            gzip=gzip,
            max_size_mb=max_size_mb,
            exclude=exclude,
            core_dir_override=core_dir_override,
        )


if __name__ == "__main__":
    app() 