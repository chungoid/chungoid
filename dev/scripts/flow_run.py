"""CLI to execute a Flow YAML via the Chungoid Execution Runtime.

Usage examples::

    python flow_run.py sample.yaml               # sync run
    python flow_run.py sample.yaml --async-run   # async run (placeholder)

The command prints the list of visited stage IDs on success.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

import typer
import yaml

from chungoid.runtime.orchestrator import ExecutionPlan, SyncOrchestrator, AsyncOrchestrator

app = typer.Typer(help="Execute a Stage-Flow YAML using the built-in orchestrator.")


@app.command()
def run(
    yaml_file: Path = typer.Argument(..., exists=True, help="Path to the Flow YAML file"),
    async_run: bool = typer.Option(
        False, "--async-run", help="Use the asynchronous orchestrator (experimental)"
    ),
    max_hops: int = typer.Option(64, help="Safety limit for stage traversal"),
    input_value: str = typer.Option(None, "--input", help="Input value for condition evaluation (used in next_if)")
):
    """Run *yaml_file* and print the visited stage names."""

    yaml_text = yaml_file.read_text(encoding="utf-8")
    plan = ExecutionPlan.from_yaml(yaml_text, flow_id=yaml_file.stem)

    context = {"input": input_value} if input_value is not None else {}

    if async_run:
        orch = AsyncOrchestrator(plan)

        async def _go() -> List[str]:
            return await orch.run(max_hops=max_hops)

        visited = asyncio.run(_go())
    else:
        orch = SyncOrchestrator(plan)
        visited = orch.run(max_hops=max_hops, context=context)

    typer.secho(" → Executed stages: " + ", ".join(visited), fg=typer.colors.GREEN) 