from __future__ import annotations

import subprocess
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List
import importlib
import importlib.metadata as importlib_metadata

CORE_DIR = Path(__file__).resolve().parents[3] / "chungoid-core"

__all__ = ["build_snapshot"]

# ---------------------------------------------------------------------------
# Helper functions (adapted from embed_core_snapshot.py)
# ---------------------------------------------------------------------------

def _git(*args: str, cwd: Path = CORE_DIR) -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=cwd).decode().strip()
        return out
    except Exception:
        return "UNKNOWN"


def _get_core_version() -> str:
    pyproject = CORE_DIR / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            if line.strip().startswith("version"):
                return line.split("=")[-1].strip().strip('"')
    try:
        return importlib_metadata.version("chungoid")
    except importlib_metadata.PackageNotFoundError:
        return "UNKNOWN"


def _get_tool_specs() -> List[Dict[str, Any]]:
    try:
        engine_mod = importlib.import_module("chungoid.engine")
        Engine = getattr(engine_mod, "ChungoidEngine")  # type: ignore[attr-defined]
        engine = Engine(project_directory=str(Path.cwd()))
        return engine.get_mcp_tools()  # type: ignore[attr-defined]
    except Exception:
        return []


def _get_stage_files() -> List[str]:
    stage_dir = CORE_DIR / "server_prompts" / "stages"
    if not stage_dir.exists():
        return []
    return sorted(str(p.relative_to(CORE_DIR)) for p in stage_dir.glob("*.yaml"))


def build_snapshot() -> Dict[str, Any]:
    """Return a dict representing the latest core snapshot (same schema as embed script)."""
    return {
        "type": "core_snapshot",
        "core_commit": _git("rev-parse", "--short", "HEAD"),
        "core_version": _get_core_version(),
        "created": _dt.datetime.utcnow().isoformat() + "Z",
        "stage_files": _get_stage_files(),
        "tool_specs": _get_tool_specs(),
    } 