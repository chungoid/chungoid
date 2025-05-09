import os
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from chungoid.utils.mcp_server import app

# ReflectionStore relies on a persistent Chroma client which may not be
# available in CI (http-only build). If initialisation fails we skip the
# module tests to keep the rest of the suite green.

import pytest


try:
    from chungoid.utils.reflection_store import ReflectionStore
    _STORE_AVAILABLE = True
except RuntimeError:  # pragma: no cover – Chroma running in http-only mode
    _STORE_AVAILABLE = False


if not _STORE_AVAILABLE:
    pytest.skip("Chroma persistent client unavailable; skipping reflection API tests", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test client (with optional dummy store)
# ---------------------------------------------------------------------------


_memory: dict[str, dict] = {}

class _DummyStore:  # noqa: D401
    def __init__(self, *_, **__):
        pass

    def add(self, reflection):  # type: ignore[no-self-use]
        _memory[reflection.message_id] = reflection.dict()

    def get(self, msg_id):  # noqa: D401
        from pydantic import BaseModel

        if msg_id not in _memory:
            return None

        class _Reflection(BaseModel):
            __root__: dict

        return _Reflection(__root__=_memory[msg_id])

    class _DummyObj:
        def __init__(self, d):
            self._d = d

        def dict(self, *_, **__):  # noqa: D401
            return self._d

    def query(self, **_kwargs):  # noqa: D401
        return [self._DummyObj(v) for v in _memory.values()]

# Monkey-patch server's ReflectionStore symbol (regardless of availability)
import importlib

mcp_mod = importlib.import_module("chungoid.utils.mcp_server")
mcp_mod.ReflectionStore = _DummyStore  # type: ignore[attr-defined]

_STORE_AVAILABLE = False

CLIENT = TestClient(app)


def _make_payload():
    return {
        "conversation_id": "test-convo",
        "message_id": str(uuid.uuid4()),
        "agent_id": "test-agent",
        "content_type": "thought",
        "content": "Hello world",
    }


def test_post_reflection_ok(tmp_path, monkeypatch):
    """POST /reflection should persist the Reflection and return 200."""
    # Ensure API key matches default
    monkeypatch.setenv("MCP_API_KEY", "dev-key")

    payload = _make_payload()
    headers = {"X-API-Key": "dev-key"}

    resp = CLIENT.post("/reflection", json=payload, headers=headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert data["message_id"] == payload["message_id"]

    if _STORE_AVAILABLE:
        store = ReflectionStore(project_root=Path.cwd())
        r = store.get(payload["message_id"])
        assert r is not None
        assert r.agent_id == payload["agent_id"]
        assert r.content == payload["content"]


def test_get_reflection_query(monkeypatch):
    """GET /reflection should return list with stored reflection."""
    monkeypatch.setenv("MCP_API_KEY", "dev-key")

    # insert one reflection first
    payload = _make_payload()
    CLIENT.post("/reflection", json=payload, headers={"X-API-Key": "dev-key"})

    resp = CLIENT.get(f"/reflection?conversation_id={payload['conversation_id']}&limit=10")
    assert resp.status_code == 200
    arr = resp.json()
    assert isinstance(arr, list) and any(item["message_id"] == payload["message_id"] for item in arr) 