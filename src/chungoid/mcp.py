"""CLI entry-point for the Chungoid MCP server.

This tiny wrapper exists so that users can simply run the console script
`chungoid-server` (or `python -m chungoid.mcp`) instead of invoking the
old `chungoidmcp.py` script directly.  All heavy lifting is delegated to
:class:`chungoid.engine.ChungoidEngine`.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from chungoid.engine import ChungoidEngine  # type: ignore  # local import

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chungoid-server",
        description="Start the Chungoid MCP engine for a given project directory.",
    )
    parser.add_argument(
        "project_dir",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Path to the project directory (defaults to current working directory).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the engine result as JSON (default is pretty-printed).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # Entry-point for console-script
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    project_path: Path = args.project_dir.expanduser().resolve()
    logger.info("Starting Chungoid engine for project: %s", project_path)

    try:
        engine = ChungoidEngine(str(project_path))
        result = engine.run_next_stage()
    except Exception as exc:
        logger.exception("Engine failed to start: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Pretty print basic result
        from pprint import pprint

        pprint(result)


if __name__ == "__main__":  # pragma: no cover — executed only when run as module
    main() 