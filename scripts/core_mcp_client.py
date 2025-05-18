from __future__ import annotations

"""Async wrapper around the Chungoid FastMCP HTTP server.

Implements CP2 task *Meta Layer MCPClient Integration* from the snapshot roadmap.

Usage example:

>>> import asyncio, core_mcp_client as client
>>> async def demo():
...     mcp = client.CoreMCPClient("http://localhost:9000", api_key="dev-key")
...     meta = await mcp.get_metadata()
...     print(meta["core_version"])
...     tools = await mcp.discover_tools()
...     result = await mcp.invoke_tool("get_project_status", {})
...
>>> asyncio.run(demo())
"""

import asyncio
import httpx
from typing import Any, Dict, List


class CoreMCPClient:
    """Thin async wrapper for FastMCP endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 10,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "dev-key"
        self._timeout = timeout
        # Allow dependency injection for tests
        self._client: httpx.AsyncClient | None = client

    # ---------------------------------------------------------------------
    # Context manager helpers
    # ---------------------------------------------------------------------
    async def __aenter__(self):  # noqa: D401
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: D401
        if self._client:
            await self._client.aclose()

    # ---------------------------------------------------------------------
    # Internal helper
    # ---------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key}

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def get_metadata(self) -> Dict[str, Any]:
        await self._ensure_client()
        resp = await self._client.get("/metadata", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    async def discover_tools(self) -> List[Dict[str, Any]]:
        # Prefer dedicated endpoint if available (introduced May-2025)
        await self._ensure_client()
        try:
            resp = await self._client.get("/tools", headers=self._headers())
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            # Fallback to metadata path
            pass

        meta = await self.get_metadata()
        return meta.get("tool_specs", [])

    async def invoke_tool(self, tool_name: str, **params) -> Dict[str, Any]:
        await self._ensure_client()
        payload = {"tool_name": tool_name, "args": params}
        resp = await self._client.post("/invoke", headers=self._headers(), json=payload)
        resp.raise_for_status()
        return resp.json()

# -------------------------------------------------------------------------
# Quick CLI for ad-hoc usage
# -------------------------------------------------------------------------

async def _cli():
    import argparse, json
    parser = argparse.ArgumentParser(description="Invoke tools on a running FastMCP server.")
    parser.add_argument("tool", help="Tool name (or 'meta' for /metadata)")
    parser.add_argument("--server", default="http://localhost:9000", help="Base URL for MCP server")
    parser.add_argument("--api-key", default="dev-key", help="X-API-Key header value")
    parser.add_argument("--args", default="{}", help="JSON string of tool args")
    args = parser.parse_args()

    async with CoreMCPClient(args.server, api_key=args.api_key) as mcp:
        if args.tool.lower() in {"meta", "metadata"}:
            data = await mcp.get_metadata()
        else:
            data = await mcp.invoke_tool(args.tool, **json.loads(args.args))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    asyncio.run(_cli()) 