"""CLI to execute a Flow YAML via the Chungoid Execution Runtime.

Usage examples (run from chungoid-core/scripts/):

    python flow_run.py ../server_prompts/stages/stage0.yaml
    python flow_run.py ../server_prompts/stages/stage0.yaml --async-run

The command prints the list of visited stage IDs on success.

Note: Ensure 'chungoid-core' is installed (e.g., `pip install -e .` from `chungoid-core` directory)
      or that PYTHONPATH includes `chungoid-core/src`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List
import sys

import typer
import yaml

# Ensure src is in path if running script directly from chungoid-core/scripts
# and chungoid-core is not installed.
# chungoid-core/scripts -> chungoid-core -> chungoid-mcp
PROJ_ROOT = Path(__file__).resolve().parents[2]
CHUNGOID_CORE_SRC = PROJ_ROOT / "chungoid-core" / "src"
if str(CHUNGOID_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CHUNGOID_CORE_SRC))

from chungoid.runtime.orchestrator import ExecutionPlan, SyncOrchestrator, AsyncOrchestrator # noqa: E402

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
        # AsyncOrchestrator might need project_config or similar initialization
        # For now, assuming its constructor is parameterless or takes plan directly
        orch = AsyncOrchestrator(plan) # Original script had this

        async def _go() -> List[str]:
            # Async run method might also require context, adapting from sync
            return await orch.run(max_hops=max_hops, context=context) # Added context for consistency

        visited = asyncio.run(_go())
    else:
        # SyncOrchestrator in original script took project_config={} - assuming this is still valid
        orch = SyncOrchestrator(project_config={}) 
        visited = orch.run(plan=plan, context=context)

    typer.secho(" → Executed stages: " + ", ".join(visited), fg=typer.colors.GREEN)

if __name__ == "__main__":
    app() 