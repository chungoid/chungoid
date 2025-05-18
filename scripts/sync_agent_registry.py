#!/usr/bin/env python
"""Synchronise Agent Registry with tools exposed by the MCP server.

• Fetches `/tools` list via CoreMCPClient (or snapshot fallback).
• Creates/updates an AgentCard per tool (agent_id == tool_name).
• Stores in `a2a_agent_registry` collection.

Run examples:
    python dev/scripts/sync_agent_registry.py --server http://localhost:9000 --memory
    python dev/scripts/sync_agent_registry.py  # uses defaults
"""
from __future__ import annotations

import asyncio
import typer
from pathlib import Path
from typing import List

from chungoid.utils.agent_registry import AgentRegistry, AgentCard
from .core_mcp_client import CoreMCPClient  # type: ignore

app = typer.Typer(add_help_option=True)


async def _fetch_tools(server: str, api_key: str) -> List[dict]:
    async with CoreMCPClient(server, api_key=api_key) as mcp:
        return await mcp.discover_tools()


@app.command()
def run(
    server: str = typer.Option("http://localhost:9000", help="Base URL of MCP server"),
    api_key: str = typer.Option("dev-key", help="X-API-Key"),
    memory: bool = typer.Option(False, help="Use in-memory Chroma (no persistence)"),
):
    """Synchronise registry entries with current tool list."""

    tools = asyncio.run(_fetch_tools(server, api_key))
    typer.echo(f"[sync] retrieved {len(tools)} tool specs")

    reg = AgentRegistry(project_root=Path.cwd(), chroma_mode="memory" if memory else "persistent")

    for spec in tools:
        agent_id = spec.get("name")
        if not agent_id:
            continue
        card = AgentCard(
            agent_id=agent_id,
            name=spec.get("name", agent_id),
            description=spec.get("description", "Auto-generated from /tools"),
            capabilities=["tool"],
            tool_names=[agent_id],
        )
        reg.add(card, overwrite=True)
        typer.echo(f"[sync] upserted agent card '{agent_id}'")

    typer.echo("[sync] complete")


if __name__ == "__main__":
    run() 