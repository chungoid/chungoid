"""Flow Registry CLI (Phase-5 stub).

Allows basic *add*, *show*, *list*, and *remove* operations on the Chroma-backed
`FlowRegistry`.  It is intentionally lightweight: the first iteration only
supports adding flows from a YAML file and printing summaries.

Example usage (from repo root):

    python chungoid-core/dev/scripts/flow_registry_cli.py add path/to/flow.yaml \
        --flow-id my_flow_v1 --name "My flow" --tags bootstrap,example

"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich import print as rprint  # nicer output

# Local import via repo root path manipulation when running from editable checkout
PROJ_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJ_ROOT / "chungoid-core" / "src"))

from chungoid.utils.flow_registry import FlowCard, FlowRegistry  # noqa: E402

app = typer.Typer(add_help_option=True, no_args_is_help=True)


# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _get_registry() -> FlowRegistry:
    return FlowRegistry(project_root=PROJ_ROOT, chroma_mode="persistent")


# ---------------------------------------------------------------------------
# Commands ------------------------------------------------------------------
# ---------------------------------------------------------------------------


@app.command()
def add(
    yaml_file: Path = typer.Argument(..., exists=True, readable=True),
    flow_id: str = typer.Option(..., help="Unique identifier for the flow"),
    name: str = typer.Option(..., help="Human readable name"),
    description: Optional[str] = typer.Option(None, help="Short description"),
    version: str = typer.Option("0.1", help="Semantic version"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated tags"),
    owner: Optional[str] = typer.Option(None, help="Maintainer handle"),
    overwrite: bool = typer.Option(False, help="Replace existing flow if already present"),
):
    """Add a new flow to the registry from *YAML_FILE*."""
    yaml_text = yaml_file.read_text(encoding="utf-8")
    card = FlowCard(
        flow_id=flow_id,
        name=name,
        yaml_text=yaml_text,
        description=description,
        version=version,
        tags=(tags.split(",") if tags else []),
        owner=owner,
    )
    reg = _get_registry()
    reg.add(card, overwrite=overwrite)
    typer.secho(f"✅ Flow '{flow_id}' added to registry", fg=typer.colors.GREEN)


@app.command()
def show(flow_id: str = typer.Argument(..., help="Flow identifier")):
    """Print details about a stored flow."""
    reg = _get_registry()
    card = reg.get(flow_id)
    if card is None:
        raise typer.Exit(code=1)
    rprint(card.model_dump())


@app.command(name="list")
def _list(limit: int = typer.Option(20, min=1, help="Number of flows to display")):
    """List recent flows (peek)."""
    reg = _get_registry()
    cards = reg.list(limit=limit)
    for card in cards:
        typer.echo(f"• {card.flow_id}\t{card.name}\t[{card.version}]")


@app.command()
def remove(flow_id: str = typer.Argument(..., help="Flow identifier")):
    """Delete a flow from the registry."""
    reg = _get_registry()
    reg.remove(flow_id)
    typer.secho(f"🗑️  Flow '{flow_id}' removed", fg=typer.colors.YELLOW)


# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    app() 