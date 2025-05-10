from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure in-memory chroma during tests
os.environ["FLOW_REGISTRY_MODE"] = "memory"

from chungoid.utils import mcp_server  # noqa: E402  # import after env var set

client = TestClient(mcp_server.app)


@pytest.fixture(autouse=True)
def _reset_registry(tmp_path: Path, monkeypatch):
    """Patch FlowRegistry to use fresh tmp_path for each test."""
    from chungoid.utils.flow_registry import FlowRegistry

    reg = FlowRegistry(project_root=tmp_path, chroma_mode="memory")
    monkeypatch.setattr(mcp_server, "_flow_registry", reg, raising=True)
    yield


def _sample_flow_payload(flow_id: str = "demo-flow") -> dict[str, object]:
    return {
        "flow_id": flow_id,
        "name": "Demo Flow",
        "yaml_text": "start_stage: greet\nstages:\n  greet:\n    agent_id: greeter\n    next: null\n",
    }


def test_list_empty():
    response = client.get("/flows")
    assert response.status_code == 200
    assert response.json() == []


def test_add_and_get_flow():
    payload = _sample_flow_payload("flow-123")
    response = client.post("/flows", json=payload, headers={"X-API-Key": "dev-key"})
    assert response.status_code == 200
    flow_id = response.json()["flow_id"]

    resp_get = client.get(f"/flows/{flow_id}")
    data = resp_get.json()
    assert data["flow_id"] == flow_id
    assert "yaml_text" in data

    resp_list = client.get("/flows")
    ids = {f["flow_id"] for f in resp_list.json()}
    assert flow_id in ids 