#!/usr/bin/env python
"""CLI helper around AgentRegistry (add/list/get)."""
from __future__ import annotations

import json
import typer
from pathlib import Path
from chungoid.utils.agent_registry import AgentRegistry, AgentCard

app = typer.Typer(add_help_option=True)


@app.command()
def add(
    agent_id: str = typer.Option(...),
    name: str = typer.Option(...),
    description: str = typer.Option("", help="Free-text description"),
    stage_focus: str | None = typer.Option(None),
    capabilities: str = typer.Option("", help="Comma-separated list"),
    tools: str = typer.Option("", help="Comma-separated tool names"),
    memory: bool = typer.Option(False, help="Use in-memory Chroma"),
):
    """Add a new agent card."""
    reg = AgentRegistry(project_root=Path.cwd(), chroma_mode="memory" if memory else "persistent")
    card = AgentCard(
        agent_id=agent_id,
        name=name,
        description=description,
        stage_focus=stage_focus,
        capabilities=[c.strip() for c in capabilities.split(",") if c.strip()],
        tool_names=[t.strip() for t in tools.split(",") if t.strip()],
    )
    reg.add(card, overwrite=True)
    typer.echo("[OK] added/updated agent card")


@app.command()
def list(memory: bool = typer.Option(False)):
    """List agent cards (JSON)."""
    reg = AgentRegistry(project_root=Path.cwd(), chroma_mode="memory" if memory else "persistent")
    cards = reg.list()
    typer.echo(json.dumps([c.model_dump(mode='json') for c in cards], indent=2))


@app.command()
def get(agent_id: str, memory: bool = typer.Option(False)):
    reg = AgentRegistry(project_root=Path.cwd(), chroma_mode="memory" if memory else "persistent")
    card = reg.get(agent_id)
    if not card:
        typer.echo("not found", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(card.model_dump(mode='json'), indent=2))


if __name__ == "__main__":
    app() 