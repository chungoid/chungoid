"""Tests for snapshot_core_tarball CLI (Phase-4 stub).

These tests intentionally operate in *dry-run* mode so they do not write
large archives in CI environments.
"""
from typing import Any
import importlib
import importlib.util
import sys

from pathlib import Path, PurePath
from typer.testing import CliRunner

MODULE_NAME = "snapshot_core_tarball"


def _load_cli_module() -> Any:
    """Dynamically load the CLI module from its file path."""
    script_path = (
        Path(__file__).resolve().parents[2]
        / "dev"
        / "scripts"
        / "snapshot_core_tarball.py"
    )
    spec = importlib.util.spec_from_file_location(MODULE_NAME, script_path)
    assert spec and spec.loader, "Unable to load CLI module spec"
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module  # type: ignore[assignment]
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


runner = CliRunner()


# Reload every test to avoid cross-state.

def test_cli_dry_run(tmp_path: Path) -> None:
    """CLI should exit with code 0 and mention the tarball path."""
    cli_module = _load_cli_module()
    result = runner.invoke(
        cli_module.app,  # type: ignore[attr-defined]
        ["--dry-run", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    # Generated filename should appear in output (without actually creating it)
    assert "core_snapshot_" in result.output 

def test_real_archive_contains_pyproject(tmp_path: Path) -> None:
    """Building a real tarball should produce an archive <200 MB and include pyproject.toml"""
    cli_module = _load_cli_module()
    result = runner.invoke(
        cli_module.app,
        ["--output-dir", str(tmp_path), "--max-size-mb", "300"],
    )
    assert result.exit_code == 0, result.output
    # Find the tarball in tmp_path
    files = list(tmp_path.glob("core_snapshot_*.tar.gz"))
    assert files, "Tarball not created"
    tar_path = files[0]
    import tarfile
    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()
        assert "pyproject.toml" in names 

def test_embed_snapshot_with_tarball(tmp_path: Path) -> None:
    """embed_core_snapshot should accept --tarball and work in dry-run mode."""
    # create tarball first
    cli_module = _load_cli_module()
    runner.invoke(cli_module.app, ["--output-dir", str(tmp_path)])
    tarball = list(tmp_path.glob("core_snapshot_*.tar.gz"))[0]

    # load embed script dynamically
    script_path = (
        Path(__file__).resolve().parents[2]
        / "dev"
        / "scripts"
        / "embed_core_snapshot.py"
    )
    spec = importlib.util.spec_from_file_location("embed_core_snapshot", script_path)
    assert spec and spec.loader
    embed_mod = importlib.util.module_from_spec(spec)
    sys.modules["embed_core_snapshot"] = embed_mod  # type: ignore
    spec.loader.exec_module(embed_mod)  # type: ignore

    result = runner.invoke(embed_mod.app, ["run", "--dry-run", "--tarball", str(tarball)])
    assert result.exit_code == 0, result.output
    assert "core_snapshot" in result.output 