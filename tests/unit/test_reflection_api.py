import os
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from chungoid.utils.mcp_server import app
from chungoid.utils.reflection_store import ReflectionStore


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

    # Verify reflection stored in Chroma (persistent mode to tmp_path)
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