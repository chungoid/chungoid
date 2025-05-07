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
    logger.info("Chungoid MCP Server now listening on stdin for MCP messages...")

    # Store CHUNGOID_PROJECT_DIR for potential use in tool handlers
    project_directory = Path(args.project_dir).resolve()
    logger.info(f"Effective project directory for MCP operations: {project_directory}")

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                logger.debug("Received empty line, skipping.")
                continue

            logger.debug(f"Received raw line: {line}")
            request_id = None # Initialize request_id
            try:
                request = json.loads(line)
                request_id = request.get("id") # Capture the request ID
                method = request.get("method")

                logger.info(f"Received MCP request (ID: {request_id}): {request}")

                response_payload = None

                if method == "listOfferings":
                    response_payload = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": [], # No tools defined yet
                            "resources": [],
                            "resourceTemplates": []
                        }
                    }
                    logger.info("Responding to listOfferings.")
                # TODO: Add handlers for "executeTool" and other MCP methods
                # elif method == "executeTool":
                #     tool_name = request.get("params", {}).get("name")
                #     tool_args = request.get("params", {}).get("arguments")
                #     # result = run_tool(tool_name, tool_args, project_directory)
                #     # response_payload = { "jsonrpc": "2.0", "id": request_id, "result": result }
                #     pass # Placeholder for tool execution

                else:
                    # Unhandled method
                    logger.warning(f"Unhandled MCP method: {method}")
                    response_payload = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,  # JSON-RPC standard error code for Method not found
                            "message": f"Method not found: {method}"
                        }
                    }
                
                if response_payload:
                    json_response = json.dumps(response_payload)
                    print(json_response)
                    sys.stdout.flush()
                    logger.info(f"Sent MCP response (ID: {request_id}): {json_response}")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from line: {line}", exc_info=True)
                # Cannot determine request_id if JSON is invalid
                error_response_payload = {
                    "jsonrpc": "2.0",
                    "id": None, # ID is unknown
                    "error": {"code": -32700, "message": "Parse error: Invalid JSON received"}
                }
                print(json.dumps(error_response_payload))
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Error processing message (ID: {request_id}): {line} - {e}", exc_info=True)
                # Use captured request_id if available
                error_response_payload = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": f"Internal server error: {str(e)}"}
                }
                print(json.dumps(error_response_payload))
                sys.stdout.flush()

    except KeyboardInterrupt:
        logger.info("Chungoid MCP Server shutting down due to KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"Chungoid MCP Server crashed in main loop: {e}", exc_info=True)
    finally:
        logger.info("Chungoid MCP Server exited.")

if __name__ == "__main__":
    main() 