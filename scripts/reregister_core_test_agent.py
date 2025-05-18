#!/usr/bin/env python
"""Script to explicitly re-register the CoreTestGeneratorAgent_v1 with the AgentRegistry."""
from __future__ import annotations

from pathlib import Path
import typer

from chungoid.utils.agent_registry import AgentRegistry
from chungoid.runtime.agents.core_test_generator_agent import get_agent_card_static as get_core_test_generator_agent_card

app = typer.Typer()

@app.command()
def main(
    project_root_str: str = typer.Option(".", help="Path to the project root containing the .chungoid directory."),
    memory: bool = typer.Option(False, help="Use in-memory Chroma (no persistence). Default is persistent.")
):
    """Re-registers the CoreTestGeneratorAgent_v1."""
    project_root = Path(project_root_str).resolve()
    chroma_mode = "memory" if memory else "persistent"
    
    typer.echo(f"Initializing AgentRegistry with project_root: {project_root} and chroma_mode: {chroma_mode}")
    
    try:
        registry = AgentRegistry(project_root=project_root, chroma_mode=chroma_mode)
        typer.echo("AgentRegistry initialized.")
    except Exception as e:
        typer.echo(f"Error initializing AgentRegistry: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Fetching AgentCard for CoreTestGeneratorAgent_v1...")
    try:
        agent_card = get_core_test_generator_agent_card()
        typer.echo(f"Successfully fetched agent card for ID: {agent_card.agent_id}")
        typer.echo(f"  Name: {agent_card.name}")
        typer.echo(f"  Categories: {agent_card.categories}")
        typer.echo(f"  Capability Profile: {agent_card.capability_profile}")

    except Exception as e:
        typer.echo(f"Error fetching agent card: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Attempting to add/update agent card for {agent_card.agent_id} in the registry...")
    try:
        registry.add(agent_card, overwrite=True)
        typer.echo(f"Successfully added/updated agent card for {agent_card.agent_id} in the registry.")
    except Exception as e:
        typer.echo(f"Error adding/updating agent card in registry: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("CoreTestGeneratorAgent_v1 re-registration process complete.")

if __name__ == "__main__":
    app() 