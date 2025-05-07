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
    
    project_directory_path = Path(args.project_dir).resolve()
    logger.info(f"Effective project directory for MCP operations: {project_directory_path}")

    # Instantiate the ChungoidEngine
    try:
        engine = ChungoidEngine(str(project_directory_path))
        logger.info("ChungoidEngine instantiated successfully.")
    except Exception as e:
        logger.error(f"Failed to instantiate ChungoidEngine: {e}", exc_info=True)
        engine = None 

    logger.info("Chungoid MCP Server now listening on stdin for MCP messages...")

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                logger.debug("Received empty line, skipping.")
                continue

            logger.debug(f"Received raw line: {line}")
            request_id = None # Initialize request_id
            response_payload = None
            request = None

            try:
                request = json.loads(line)
                request_id = request.get("id")
                method = request.get("method")
                params = request.get("params", {})

                logger.info(f"Received MCP request (ID: {request_id}, Method: {method}): {request}")

                if not engine and method not in ["shutdown", "exit"]: # Allow shutdown/exit even if engine failed
                    response_payload = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32000, "message": "Server error: ChungoidEngine failed to initialize."}
                    }
                elif method == "initialize":
                    # Respond to initialize request
                    # Based on MCP spec, clientInfo and capabilities are in params
                    # client_info = params.get("clientInfo", {})
                    # client_capabilities = params.get("capabilities", {})
                    # logger.info(f"Initialize request from {client_info.get('name')} v{client_info.get('version')}")
                    
                    server_capabilities = {
                        "tools": True,  # We will offer tools
                        "prompts": False, # Not offering prompts for now
                        "resources": True, # We might offer resources
                        "logging": False,  # Not offering custom logging via MCP for now
                        "roots": {"listChanged": False} # Not supporting dynamic root changes for now
                    }
                    # ChungoidEngine might have its own capabilities to merge/report
                    if hasattr(engine, "get_server_capabilities"):
                        server_capabilities = engine.get_server_capabilities(params)

                    response_payload = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2024-11-05", # Align with client if possible, or state our version
                            "capabilities": server_capabilities,
                            "serverInfo": {
                                "name": "chungoid-mcp-server",
                                "version": __version__,
                                # "documentationUrl": "Optional URL to server docs"
                            }
                        }
                    }
                    logger.info(f"Responded to initialize (ID: {request_id}).")

                elif method == "listOfferings":
                    tools = []
                    resources = []
                    resource_templates = []
                    
                    if hasattr(engine, "get_mcp_tools"):
                        tools = engine.get_mcp_tools()
                    if hasattr(engine, "get_mcp_resources"):
                        resources = engine.get_mcp_resources()
                    if hasattr(engine, "get_mcp_resource_templates"):
                        resource_templates = engine.get_mcp_resource_templates()
                        
                    response_payload = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": tools,
                            "resources": resources,
                            "resourceTemplates": resource_templates
                        }
                    }
                    logger.info(f"Responded to listOfferings (ID: {request_id}) with {len(tools)} tools.")

                elif method == "executeTool":
                    tool_call_id = params.get("toolCallId") # MCP spec often uses toolCallId
                    tool_name = params.get("name")
                    tool_arguments = params.get("arguments")
                    
                    if hasattr(engine, "execute_mcp_tool"):
                        try:
                            # The engine method should handle the execution and return a result or raise an error
                            tool_result = engine.execute_mcp_tool(tool_name, tool_arguments, tool_call_id=tool_call_id, project_dir=project_directory_path)
                            response_payload = {
                                "jsonrpc": "2.0",
                                "id": request_id, # This is the MCP request ID
                                "result": tool_result # The result structure is tool-specific, often {"toolCallId": ..., "output": ...}
                            }
                            logger.info(f"Tool {tool_name} executed successfully (ID: {request_id}, ToolCallID: {tool_call_id}).")
                        except Exception as tool_exec_error:
                            logger.error(f"Error executing tool {tool_name} (ID: {request_id}, ToolCallID: {tool_call_id}): {tool_exec_error}", exc_info=True)
                            response_payload = {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {"code": -32001, "message": f"Tool execution error: {str(tool_exec_error)}", "data": {"toolCallId": tool_call_id}}
                            }
                    else:
                        logger.warning(f"Engine has no method execute_mcp_tool for tool: {tool_name}")
                        response_payload = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32601, "message": f"Tool not found or engine cannot execute: {tool_name}", "data": {"toolCallId": tool_call_id}}
                        }
                
                elif method == "shutdown" or method == "exit":
                    logger.info(f"Received {method} request. Server will shut down. (ID: {request_id})")
                    # Acknowledge shutdown if an ID was provided, then break loop
                    if request_id is not None: # 'exit' is a notification, might not have an id
                         response_payload = {"jsonrpc": "2.0", "id": request_id, "result": None} # Acknowledge if it's not a notification
                    if response_payload:
                         print(json.dumps(response_payload))
                         sys.stdout.flush()
                    break # Exit the loop

                else:
                    logger.warning(f"Unhandled MCP method: {method} (ID: {request_id})")
                    response_payload = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"Method not found: {method}"}
                    }
                
                if response_payload:
                    json_response = json.dumps(response_payload)
                    print(json_response)
                    sys.stdout.flush()
                    logger.info(f"Sent MCP response (ID: {request_id}): {json_response}")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from line: {line}", exc_info=True)
                error_response_payload = {
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "Parse error: Invalid JSON received"}
                }
                print(json.dumps(error_response_payload))
                sys.stdout.flush()
            except Exception as e:
                # Catch any other unexpected error during request processing
                # Use request_id if it was parsed, otherwise it's None
                current_request_id = request.get("id") if request else None
                logger.error(f"Generic error processing message (ID: {current_request_id}): {line} - {e}", exc_info=True)
                error_response_payload = {
                    "jsonrpc": "2.0", "id": current_request_id,
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