import asyncio, importlib.util, json, sys
from pathlib import Path

import httpx
from httpx import Response, Request

import pytest

# Dynamically load the client module because it lives in dev/scripts (not a package)
CLIENT_PATH = Path(__file__).resolve().parents[3] / "dev" / "scripts" / "core_mcp_client.py"

# In the standalone *public* chungoid-core repo the helper script may be absent.
if not CLIENT_PATH.exists():  # pragma: no cover
    pytest.skip("core_mcp_client helper script not present", allow_module_level=True)

spec = importlib.util.spec_from_file_location("core_mcp_client", CLIENT_PATH)
module = importlib.util.module_from_spec(spec)  # type: ignore
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore
CoreMCPClient = module.CoreMCPClient  # type: ignore


@pytest.mark.asyncio
async def test_discover_and_invoke():
    """CoreMCPClient should hit /metadata then /invoke with correct payload."""

    def handler(request: Request):  # type: ignore
        if request.url.path == "/metadata":
            data = {
                "core_version": "0.1.0",
                "tool_specs": [
                    {"name": "echo", "description": "return payload"},
                ],
            }
            return Response(200, json=data)
        elif request.url.path == "/invoke":
            body = json.loads(request.content)
            assert body["tool_name"] == "echo"
            return Response(200, json={"result": body["args"].get("msg")})
        return Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        client = CoreMCPClient("http://testserver", client=async_client)
        tools = await client.discover_tools()
        assert tools[0]["name"] == "echo"
        res = await client.invoke_tool("echo", msg="hi")
        assert res["result"] == "hi" 