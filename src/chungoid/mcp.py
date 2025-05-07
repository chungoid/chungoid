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
import os
import sys
from pathlib import Path

# --- HACK: Temporarily add src to sys.path ---
try:
    from chungoid.utils.log_utils import setup_logging
except ModuleNotFoundError:
    # This assumes mcp.py is in src/chungoid/mcp.py
    # So, two levels up is the 'src' directory.
    src_dir = Path(__file__).resolve().parent.parent.parent
    if (src_dir / "chungoid" / "utils").is_dir(): # Check if 'chungoid/utils' exists in the guessed src_dir
        sys.path.insert(0, str(src_dir))
        from chungoid.utils.log_utils import setup_logging # Retry import
    else:
        # If the structure isn't as expected, re-raise to see the original error
        # or provide a more specific error about the hack failing.
        print(f"Temporary sys.path hack in mcp.py failed. Guessed src_dir: {src_dir}", file=sys.stderr)
        raise
# --- END HACK ---

from chungoid.engine import ChungoidEngine  # type: ignore  # local import

__version__ = "0.1.0"  # Example version

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chungoid-server",
        description="Start the Chungoid MCP engine for a given project directory.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=os.environ.get("CHUNGOID_PROJECT_DIR", "."),
        help="Directory of the project to operate on. Defaults to CHUNGOID_PROJECT_DIR env var or current dir.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (currently affects placeholder message).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("CHUNGOID_LOG_LEVEL", "INFO"),
        help="Set the logging level (e.g., DEBUG, INFO, WARNING). Defaults to CHUNGOID_LOG_LEVEL env var or INFO.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # Entry-point for console-script
    args = _parse_args(argv)

    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(
        f"Chungoid MCP Server starting. Version: {__version__}, Project Dir: {args.project_dir}, Log Level: {args.log_level}"
    )

    # Basic MCP server loop (simplified for stdio)
    # In a real MCP server, you'd handle message framing and parsing here.
    # For now, we assume the IDE sends JSON messages line by line.

    # Placeholder: Initialize engine or other core components if needed immediately
    # try:
    #     engine = ChungoidEngine(project_dir=args.project_dir)
    #     logger.info(f"ChungoidEngine initialized for project: {engine.project_dir}")
    #     logger.info(f"Chroma client: {engine.chroma_client}")
    # except Exception as e:
    #     logger.error(f"Failed to initialize ChungoidEngine: {e}", exc_info=True)
    #     # Depending on severity, you might exit or try to operate in a limited mode
    #     # For now, we'll log and continue to allow MCP communication for diagnostics.
    #     # sys.exit(1) # Uncomment if engine initialization is critical for any operation

    if args.json:
        print(
            json.dumps(
                {
                    "message": "Chungoid server alive and awaiting MCP messages.",
                    "project_dir": args.project_dir,
                }
            )
        )
    else:
        # This is primarily for direct CLI testing, not for MCP interaction
        print(
            f"Chungoid server alive. Project: {args.project_dir}. Awaiting MCP messages on stdin..."
        )
        print("Use --json for structured output if testing directly.")
        print("This executable is intended to be run by an MCP client (e.g., an IDE).")


    # Example: Basic echo server for testing MCP communication (replace with actual MCP logic)
    # try:
    #     for line in sys.stdin:
    #         line = line.strip()
    #         if not line:
    #             continue # Skip empty lines, though well-behaved clients shouldn't send them

    #         logger.debug(f"Received raw line: {line}")
    #         try:
    #             request = json.loads(line)
    #             logger.info(f"Received MCP request: {request}")

    #             # Replace with actual request handling and response generation
    #             response = {"status": "received", "original_request": request}
                
    #             json_response = json.dumps(response)
    #             print(json_response)
    #             sys.stdout.flush() # Ensure the client receives the message promptly
    #             logger.info(f"Sent MCP response: {json_response}")

    #         except json.JSONDecodeError:
    #             logger.error(f"Failed to decode JSON from line: {line}", exc_info=True)
    #             # Send an error response if possible, or log and continue
    #             error_response = {"error": "Invalid JSON received", "received_line": line}
    #             json_error_response = json.dumps(error_response)
    #             print(json_error_response)
    #             sys.stdout.flush()
    #         except Exception as e:
    #             logger.error(f"Error processing message: {line}: {e}", exc_info=True)
    #             # Send a generic error response
    #             error_response = {"error": "Internal server error", "details": str(e)}
    #             json_error_response = json.dumps(error_response)
    #             print(json_error_response)
    #             sys.stdout.flush()


    # except KeyboardInterrupt:
    #     logger.info("Chungoid server shutting down due to KeyboardInterrupt.")
    # except Exception as e:
    #     logger.error(f"Chungoid server crashed: {e}", exc_info=True)
    # finally:
    #     logger.info("Chungoid server exited.")


if __name__ == "__main__":  # pragma: no cover — executed only when run as module
    main() 