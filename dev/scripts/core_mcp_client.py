from __future__ import annotations

"""Async wrapper around the Chungoid FastMCP HTTP server.

Copied from meta-layer scripts so unit tests can import it without reaching
outside the repository.
"""

import asyncio
from typing import Any, Dict, List

import httpx


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
        self._client: httpx.AsyncClient | None = client

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self._timeout
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self._timeout
            )

    def _headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key}

    # ------------------------------------------------------------------
    async def get_metadata(self) -> Dict[str, Any]:
        await self._ensure_client()
        resp = await self._client.get("/metadata", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    async def discover_tools(self) -> List[Dict[str, Any]]:
        meta = await self.get_metadata()
        return meta.get("tool_specs", [])

    async def invoke_tool(self, tool_name: str, **params):
        await self._ensure_client()
        payload = {"tool_name": tool_name, "args": params}
        resp = await self._client.post(
            "/invoke", headers=self._headers(), json=payload
        )
        resp.raise_for_status()
        return resp.json()


async def _cli():  # pragma: no cover
    import argparse, json

    p = argparse.ArgumentParser("Invoke tools on FastMCP server")
    p.add_argument("tool", help="Tool name or 'meta'")
    p.add_argument("--server", default="http://localhost:9000")
    p.add_argument("--api-key", default="dev-key")
    p.add_argument("--args", default="{}", help="JSON string of tool args")
    args = p.parse_args()

    async with CoreMCPClient(args.server, api_key=args.api_key) as mcp:
        if args.tool.lower() in {"meta", "metadata"}:
            data = await mcp.get_metadata()
        else:
            data = await mcp.invoke_tool(args.tool, **json.loads(args.args))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_cli()) 