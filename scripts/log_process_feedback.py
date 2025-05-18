#!/usr/bin/env python
"""Quick helper to log a ProcessFeedback entry from the CLI.

Usage example:
    python scripts/log_process_feedback.py \
        --agent core_dev --stage utils_refactor \
        --sentiment 👍 \
        --comment "FeedbackStore integration was straightforward."

By default it uses in-memory Chroma when run with ``--memory`` (useful for tests).
It will use the .chungoid/chroma_db in the current working directory for persistent storage.
"""
from __future__ import annotations

import typer
from pathlib import Path

# Assuming chungoid.utils is in PYTHONPATH, which it should be if chungoid-core is installed
# or if running from a context where chungoid-core/src is in sys.path.
from chungoid.utils.feedback_store import FeedbackStore, ProcessFeedback

app = typer.Typer(add_help_option=True)


@app.command()
def add(
    agent: str = typer.Option(..., help="Agent or user id (e.g., core_developer)"),
    comment: str = typer.Option(..., help="Free-text feedback comment"),
    sentiment: str = typer.Option("👍", help="Emoji or short tag summarising sentiment"),
    stage: str | None = typer.Option(None, help="Stage, component, or task ID (optional)"),
    conversation_id: str = typer.Option("core_dev_session", help="Logical thread/session id (optional)"),
    memory: bool = typer.Option(False, help="Use in-memory Chroma (for local tests)"),
    project_root_dir: str = typer.Option(".", help="Project root directory containing .chungoid folder. Defaults to current directory.")
):
    """Add a feedback entry to the a2a_process_feedback collection for the specified project root."""

    mode = "memory" if memory else "persistent"
    
    # Use the provided project_root_dir. Path.cwd() is the fallback if Typer option default is used.
    # FeedbackStore expects project_root to be the directory where .chungoid/chroma_db might exist.
    resolved_project_root = Path(project_root_dir).resolve()

    store = FeedbackStore(project_root=resolved_project_root, chroma_mode=mode)

    fb = ProcessFeedback(
        conversation_id=conversation_id,
        agent_id=agent,
        stage=stage,
        sentiment=sentiment,
        comment=comment,
    )
    store.add(fb)
    typer.secho(f"[SUCCESS] Feedback logged to ChromaDB at {resolved_project_root / '.chungoid' / 'chroma_db' if mode == 'persistent' else 'in-memory'}.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app() 