from __future__ import annotations

"""FastAPI wrapper that exposes chungoid-core tools over HTTP.

Implements CP4 (FastMCP wrapper with auth) of the Core Snapshot roadmap.

Key endpoints
-------------
GET  /metadata               → current snapshot (commit, version, tools, stage files)
POST /invoke                 → invoke a registered tool (json: {"tool_name": str, "args": {…}})

Security: simple API-Key via `X-API-Key` header.  The expected key is read from
`MCP_API_KEY` env-var at startup (default "dev-key" in testing branch).
"""

import os
import json
from pathlib import Path
from typing import Any, Dict
import datetime as _dt

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR
import importlib
import subprocess
import yaml

# ---------------------------------------------------------------------------
# Helpers (lightweight – mirror logic from dev/scripts/embed_core_snapshot.py)
# ---------------------------------------------------------------------------
_CORE_DIR = Path(__file__).resolve().parent.parent.parent  # <repo>/chungoid-core


def _git(*args: str) -> str:
    try:
        return (
            subprocess.check_output(["git", *args], cwd=_CORE_DIR).decode().strip()
        )
    except Exception:  # pylint: disable=broad-except
        return "UNKNOWN"


# Try importing the engine lazily so server still boots even if something is
# broken in core modules – endpoints that need it will raise later.
try:
    _engine_mod = importlib.import_module("chungoid.engine")
    _EngineCls = getattr(_engine_mod, "ChungoidEngine")  # type: ignore[attr-defined]
    _engine = _EngineCls(project_directory=str(Path.cwd()))
except Exception as exc:  # pylint: disable=broad-except
    _engine = None  # type: ignore
    _ENGINE_IMPORT_ERR = exc
else:
    _ENGINE_IMPORT_ERR = None


# Build once on startup; refresh on /metadata if engine is still None
_def_stage_dir = _CORE_DIR / "server_prompts" / "stages"
_stage_files = (
    sorted(str(p.relative_to(_CORE_DIR)) for p in _def_stage_dir.glob("*.yaml"))
    if _def_stage_dir.exists()
    else []
)


# ---------------------------------------------------------------------------
# API-Key auth dependency
# ---------------------------------------------------------------------------
_EXPECTED_KEY = os.getenv("MCP_API_KEY", "dev-key")


def _check_api_key(x_api_key: str | None):  # noqa: N802 (FastAPI header style)
    if _EXPECTED_KEY and x_api_key != _EXPECTED_KEY:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid API key")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Chungoid FastMCP", version="0.1.0-dev")


@app.get("/metadata")
async def metadata(x_api_key: str | None = Header(None, alias="X-API-Key")):
    _check_api_key(x_api_key)

    if _engine is None:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Engine import failed at startup: {_ENGINE_IMPORT_ERR}",
        )

    snapshot = {
        "type": "core_snapshot_live",
        "core_commit": _git("rev-parse", "--short", "HEAD"),
        "core_version": _git("describe", "--tags", "--abbrev=0"),
        "generated": _dt.datetime.utcnow().isoformat() + "Z",
        "tool_specs": _engine.get_mcp_tools(),  # type: ignore[attr-defined]
        "stage_files": _stage_files,
    }
    return JSONResponse(snapshot)


@app.post("/invoke")
async def invoke(
    payload: Dict[str, Any],
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)

    if _engine is None:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Engine import failed: {_ENGINE_IMPORT_ERR}",
        )

    tool_name: str = payload.get("tool_name")  # type: ignore[assignment]
    tool_args: Dict[str, Any] = payload.get("args", {})  # type: ignore[assignment]
    if not tool_name:
        raise HTTPException(status_code=422, detail="'tool_name' missing in payload")

    try:
        result = _engine.execute_mcp_tool(tool_name, tool_args)  # type: ignore[arg-type]
    except AttributeError:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Tool not found") from None
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return JSONResponse({"status": "ok", "result": result})


# ---------------------------------------------------------------------------
# Entrypoint helper
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 9000):  # pragma: no cover
    """Launch with `python -m chungoid.utils.mcp_server` for quick dev."""
    import uvicorn  # local import to avoid mandatory dep for non-server users

    uvicorn.run("chungoid.utils.mcp_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main() 