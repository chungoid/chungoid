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

        # Get decoded path (e.g. "/my/path")
        path_only_str = url.path
        
        # Get raw query bytes (e.g. b"a=1&b=2") and decode to string
        query_bytes = url.query
        query_str = query_bytes.decode() if query_bytes else ""

        # Construct the target URL path for TestClient (e.g. "/my/path?a=1&b=2")
        target_url_for_client = path_only_str
        if query_str:
            target_url_for_client += "?" + query_str

        headers = dict(request.headers)
        body_content = request.read() # Use a different name to avoid conflict with TestClient params
        
        response = self._client.request(
            method,
            target_url_for_client,
            headers=headers,
            content=body_content # Pass content/body correctly
        )
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

    # Ensure cli.httpx points to the global httpx from the test file
    monkeypatch.setattr(cli, "httpx", httpx)

    # Store the original httpx.Client constructor from the (now global) httpx module
    original_httpx_client_constructor = httpx.Client 

    # Define the replacement constructor that will be used by the patched code
    def patched_client_constructor(*a, **kw):
        kw.pop('transport', None) # Remove any transport passed by original code
        # Use the original constructor to create the client, but with our special transport
        return original_httpx_client_constructor(*a, transport=transport, **kw)

    # Patch httpx.Client (which cli.httpx.Client now refers to) to use our replacement
    monkeypatch.setattr(cli.httpx, "Client", patched_client_constructor)

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