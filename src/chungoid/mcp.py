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

# --- ALL DIAGNOSTIC PRINT STATEMENTS REMOVED FROM HERE ---

from chungoid.utils.logger_setup import setup_logging
from chungoid.engine import ChungoidEngine  # type: ignore  # local import

__version__ = "0.1.0"  # Example version

# logger will be set up in main after setup_logging

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
        help="Output results as JSON (placeholder, may not affect MCP operation).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("CHUNGOID_LOG_LEVEL", "INFO"),
        help="Set the logging level (e.g., DEBUG, INFO, WARNING). Defaults to CHUNGOID_LOG_LEVEL env var or INFO.",
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(
        f"Chungoid MCP Server starting. Version: {__version__}, Project Dir: {args.project_dir}, Log Level: {args.log_level}"
    )
    logger.info("Chungoid MCP Server now listening on stdin for MCP messages...") # This log is fine

    # MCP Server Loop (from previous step)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                logger.debug("Received empty line, skipping.")
                continue

            logger.debug(f"Received raw line: {line}")
            try:
                request = json.loads(line)
                logger.info(f"Received MCP request: {request}")

                response = {"status": "received_but_not_processed", "original_request": request, "message": "Implement actual tool/command handling"}
                
                json_response = json.dumps(response)
                print(json_response) # This print IS for MCP communication
                sys.stdout.flush()
                logger.info(f"Sent MCP response: {json_response}")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from line: {line}", exc_info=True)
                error_response = {"error": "Invalid JSON received", "received_line": line}
                json_error_response = json.dumps(error_response)
                print(json_error_response) # This print IS for MCP communication
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Error processing message: {line}: {e}", exc_info=True)
                error_response = {"error": "Internal server error processing message", "details": str(e)}
                json_error_response = json.dumps(error_response)
                print(json_error_response) # This print IS for MCP communication
                sys.stdout.flush()

    except KeyboardInterrupt:
        logger.info("Chungoid MCP Server shutting down due to KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"Chungoid MCP Server crashed in main loop: {e}", exc_info=True)
    finally:
        logger.info("Chungoid MCP Server exited.")

if __name__ == "__main__":
    main() 