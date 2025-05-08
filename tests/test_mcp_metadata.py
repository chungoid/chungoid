from fastapi.testclient import TestClient
from chungoid.utils import mcp_server


def test_metadata_endpoint():
    client = TestClient(mcp_server.app)
    response = client.get("/metadata", headers={"X-API-Key": "dev-key"})
    assert response.status_code == 200
    data = response.json()
    assert "core_commit" in data
    assert "tool_specs" in data 