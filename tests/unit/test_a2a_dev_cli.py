import json
from pathlib import Path
from typing import Any

import typer.testing
import httpx
from fastapi.testclient import TestClient

from chungoid.utils.mcp_server import app  # FastAPI app

# Import CLI app
import importlib
import pytest

try:
    cli = importlib.import_module("dev.scripts.a2a_dev_cli")  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – meta package not present
    pytest.skip("meta-layer 'dev' package not installed – skipping CLI tests", allow_module_level=True)

runner = typer.testing.CliRunner()


class _LocalTransport(httpx.BaseTransport):
    """Route httpx calls to FastAPI TestClient (no network)."""

    def __init__(self, test_client: TestClient):
        self._client = test_client

    def handle_request(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        # Convert httpx request to ASGI via TestClient
        method = request.method
        url = request.url
        path = url.raw_path.decode()
        query = url.raw_query.decode()
        headers = dict(request.headers)
        body = request.read()
        response = self._client.request(method, path + ("?" + query if query else ""), headers=headers, content=body)
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )


def test_cli_send_and_pull(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_API_KEY", "dev-key")

    test_client = TestClient(app)
    transport = _LocalTransport(test_client)
    monkeypatch.setattr(cli, "httpx", httpx.Client)

    # Patch _API to use our custom client
    def _patched_api(base_url: str, api_key: str):
        return httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, transport=transport)

    monkeypatch.setattr(cli, "httpx", httpx)  # ensure module imported
    monkeypatch.setattr(cli.httpx, "Client", lambda *a, **kw: _patched_api(*a, **kw))

    # Write reflection JSON file
    payload = {
        "conversation_id": "cli-test",
        "agent_id": "tester",
        "content_type": "note",
        "content": "hello",
    }
    json_file = tmp_path / "r.json"
    json.dump(payload, json_file.open("w"))

    res = runner.invoke(cli.app, ["send", str(json_file), "--url", "http://localhost:9000"])
    assert res.exit_code == 0, res.stderr

    res2 = runner.invoke(cli.app, ["pull", "--conversation-id", "cli-test", "--limit", "5", "--url", "http://localhost:9000"])
    assert res2.exit_code == 0
    data = json.loads(res2.stdout)
    assert isinstance(data, list) and data 