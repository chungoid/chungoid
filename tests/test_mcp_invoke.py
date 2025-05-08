from fastapi.testclient import TestClient
from chungoid.utils import mcp_server


def test_invoke_endpoint_no_args():
    client = TestClient(mcp_server.app)
    payload = {
        "tool_name": "load_pending_reflection",
        "args": {},
    }
    response = client.post("/invoke", json=payload, headers={"X-API-Key": "dev-key"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "result" in data 