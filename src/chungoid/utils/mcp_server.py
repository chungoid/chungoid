from __future__ import annotations

"""FastAPI wrapper that exposes chungoid-core tools over HTTP.

Implements CP4 (FastMCP wrapper with auth) of the Core Snapshot roadmap.

Key endpoints
-------------
GET  /metadata               → current snapshot (commit, version, tools, stage files)
POST /invoke                 → invoke a registered tool (json: {"tool_name": str, "args": {…}})
GET  /tools                  → list all tools (no auth)

Security: simple API-Key via `X-API-Key` header.  The expected key is read from
`MCP_API_KEY` env-var at startup (default "dev-key" in testing branch).
"""

import os
import json
from pathlib import Path
from typing import Any, Dict
import datetime as _dt

from fastapi import FastAPI, Header, HTTPException, Body, Depends
from fastapi.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR
import importlib
import subprocess
import yaml
from .core_snapshot_utils import build_snapshot
from .core_snapshot_utils import _get_tool_specs  # internal helper
from .flow_registry import FlowRegistry, FlowCard  # NEW
from .flow_api import get_router as get_flow_router
from chungoid.runtime.orchestrator import ExecutionPlan, AsyncOrchestrator
from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.state_manager import StateManager

# Reflection model & store (for /reflection endpoint)
from typing import TYPE_CHECKING

try:
    from .reflection_store import Reflection, ReflectionStore
except Exception:  # pragma: no cover – reflection store may fail if chromadb missing
    if TYPE_CHECKING:  # Provide type hints during static analysis
        from typing import Any as Reflection  # type: ignore
        from typing import Any as ReflectionStore  # type: ignore

    Reflection = None  # type: ignore # noqa: N816
    ReflectionStore = None  # type: ignore # noqa: N816

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

app = FastAPI(title="Chungoid MCP", version="0.1.0")

# Register sub-routers (keeps main file slim)
app.include_router(get_flow_router(_check_api_key))

# Expose API key check for submodules
def get_api_key_checker():
    return _check_api_key


@app.get("/metadata", tags=["meta"])
async def get_metadata():
    """Return the latest core snapshot JSON."""
    return build_snapshot()


@app.get("/tools", tags=["meta"])
async def list_tools():
    """Return short spec of all registered MCP tools (no auth)."""
    return _get_tool_specs()


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
# Reflection endpoint
# ---------------------------------------------------------------------------

@app.post("/reflection", tags=["reflection"])
async def add_reflection(
    reflection: "Reflection",  # type: ignore[name-defined]
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    """Persist one Reflection document to Chroma.

    Body must follow the `Reflection` Pydantic model (see JSON schema).
    """

    _check_api_key(x_api_key)

    if ReflectionStore is None or Reflection is None:  # chromadb not installed or import failed
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Reflection subsystem unavailable (chromadb missing)",
        )

    try:
        store = ReflectionStore(project_root=Path.cwd())
        store.add(reflection)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return JSONResponse({"status": "ok", "message_id": reflection.message_id})


# ---------------------------------------------------------------------------
# Reflection query endpoint (read-only)
# ---------------------------------------------------------------------------

@app.get("/reflection", tags=["reflection"])
async def query_reflections(
    conversation_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 100,
):
    """Return reflections matching optional filters.

    No authentication enforced for reads (subject to change).
    """

    if ReflectionStore is None:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Reflection subsystem unavailable",
        )

    try:
        store = ReflectionStore(project_root=Path.cwd())
        results = store.query(conversation_id=conversation_id, agent_id=agent_id, limit=limit)
        return [r.dict() for r in results]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Flow Registry helper & endpoints
# ---------------------------------------------------------------------------

# Choose in-memory store automatically when running under pytest to avoid
# touching the developer's persistent DB.
_FLOW_REG_MODE = os.getenv("FLOW_REGISTRY_MODE", "persistent")

if _FLOW_REG_MODE not in {"persistent", "http", "memory"}:
    _FLOW_REG_MODE = "memory"

# Default to in-memory when running under pytest or when Chroma is HTTP-only
if "PYTEST_CURRENT_TEST" in os.environ or os.getenv("CHROMA_API_IMPL") == "http" or _FLOW_REG_MODE == "persistent" and os.getenv("CHROMA_SERVER_HOST"):
    _FLOW_REG_MODE = "memory"

_flow_registry = FlowRegistry(project_root=Path.cwd(), chroma_mode=_FLOW_REG_MODE)


@app.get("/flows", tags=["flow"])
async def list_flows(limit: int = 100):
    """Return brief metadata for recent flows (no auth)."""
    cards = _flow_registry.list(limit=limit)
    return [c.model_dump(exclude={"yaml_text"}) for c in cards]


@app.get("/flows/{flow_id}", tags=["flow"])
async def get_flow(flow_id: str):
    """Return full FlowCard for given *flow_id*."""
    card = _flow_registry.get(flow_id)
    if card is None:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Flow not found")
    return card.model_dump()


# ---------------------------------------------------------------------------
# Flow execution endpoint (Phase-6)
# ---------------------------------------------------------------------------

# The @app.post("/run/{flow_id}", tags=["flow"]) endpoint defined directly in this file is removed.
# Its functionality is now handled by the router from flow_api.py.

@app.post("/flows", tags=["flow"])
async def add_flow(
    payload: Dict[str, Any],
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    """Add a new flow definition to the registry (API-key protected)."""
    _check_api_key(x_api_key)

    required = {"flow_id", "name", "yaml_text"}
    missing = required - payload.keys()
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields: {', '.join(sorted(missing))}")

    card = FlowCard(
        flow_id=payload["flow_id"],
        name=payload["name"],
        yaml_text=payload["yaml_text"],
        description=payload.get("description"),
        version=payload.get("version", "0.1"),
        tags=payload.get("tags", []),
        owner=payload.get("owner"),
    )

    try:
        _flow_registry.add(card, overwrite=bool(payload.get("overwrite", False)))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return JSONResponse({"status": "ok", "flow_id": card.flow_id})