"""Main MCP Server implementation for Chungoid Bootstrapper."""

import os
import logging
import shutil  # <<< UNCOMMENTED
import uuid
import inspect
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional, Any
import json

# <<< ADDED DIAGNOSTICS >>>
import sys
import pprint

# --- MCP and Project Imports ---
from fastmcp import FastMCP, Context
from utils.logger_setup import setup_logging
from utils.state_manager import StateManager, StatusFileError, ChromaOperationError
from stage_executor import StageExecutor, StageExecutionError  # <<< UNCOMMENTED
from utils.chroma_utils import ChromaOperationError
from utils.exceptions import StageExecutionError, ToolExecutionError, PromptRenderError
from utils.prompt_manager import PromptManager

# <<< EARLY EXECUTION LOGGING >>>
EARLY_PID = os.getpid()
EARLY_NAME = __name__
# Use basic print first, as logging might not be configured yet
print(f"[EARLY CHECK {EARLY_PID}] Script executing. __name__ = '{EARLY_NAME}'")
# <<< END EARLY EXECUTION LOGGING >>>

# <<< ADDED DIAGNOSTICS >>>
print("--- Python Search Path (sys.path) ---")
pprint.pprint(sys.path)
print("-------------------------------------")
# <<< END DIAGNOSTICS >>>


# --- Custom Exceptions ---
class SecurityError(Exception):
    """Custom exception for security-related errors, like path traversal."""

    pass

# Load environment variables from .env file
load_dotenv()

# Define the env var name *before* using it
CHUNGOID_PROJECT_DIR_ENV_VAR = "CHUNGOID_PROJECT_DIR"

# --- <<< NEW: Read Project Directory from Environment Variable >>> ---
GLOBAL_PROJECT_DIR: Optional[str] = os.getenv(CHUNGOID_PROJECT_DIR_ENV_VAR)
if GLOBAL_PROJECT_DIR:
    try:
        # Resolve and validate the path immediately
        resolved_path = Path(GLOBAL_PROJECT_DIR).resolve()
        if resolved_path.is_dir():
            GLOBAL_PROJECT_DIR = str(resolved_path)
            logging.info(
                f"Using project directory from env var {CHUNGOID_PROJECT_DIR_ENV_VAR}: {GLOBAL_PROJECT_DIR}"
            )
        else:
            logging.error(
                f"Path from env var {CHUNGOID_PROJECT_DIR_ENV_VAR} ('{GLOBAL_PROJECT_DIR}') is not a valid directory. Path resolution: {resolved_path}. Server may fail."
            )
            GLOBAL_PROJECT_DIR = None  # Invalidate if not a dir
    except Exception as e:
        logging.error(
            f"Error resolving path from env var {CHUNGOID_PROJECT_DIR_ENV_VAR} ('{GLOBAL_PROJECT_DIR}'): {e}. Server may fail."
        )
        GLOBAL_PROJECT_DIR = None  # Invalidate on error
else:
    logging.error(
        f"Environment variable {CHUNGOID_PROJECT_DIR_ENV_VAR} is not set. Server cannot determine project directory and will likely fail."
    )
    raise ValueError(
        f"Required environment variable {CHUNGOID_PROJECT_DIR_ENV_VAR} is not set."
    )  # <<< RESTORED RAISE >>>
# --- <<< END NEW >>> ---

# Determine the base directory of the Chungoid MCP server installation FIRST
# This assumes chungoidmcp.py is at the root of the installation
CHUNGOID_BASE_DIR = Path(__file__).parent.resolve()

# Configure logging to use the base directory
setup_logging()
logger = logging.getLogger(__name__)
# <<< START ADDED LOGGING >>>
# Log again after logging is set up
logger.info(f"[EARLY CHECK {EARLY_PID}] Logging configured. __name__ = '{EARLY_NAME}'")
# <<< END ADDED LOGGING >>>

# Also set FastMCP's internal logger to DEBUG if possible (optional, setup_logging might cover it)
try:
    logging.getLogger("FastMCP").setLevel(logging.DEBUG)
    logging.getLogger("mcp").setLevel(logging.DEBUG)
except Exception:
    logger.warning("Could not explicitly set FastMCP/MCP logger level.")

# --- Module-Level Initialization ---

logger.info("Initializing Chungoid MCP Server components at module level...")
logger.info(f"Chungoid base directory determined as: {CHUNGOID_BASE_DIR}")

# Initialize FastMCP instance with STDIO transport FIRST
mcp = FastMCP(
    name="ChungoidMCP",  # Restored name
    version="0.1.0",
    transport="stdio",
)
# <<< START ADDED LOGGING >>>
logger.info(f"Root logger handlers AFTER FastMCP init: {logging.getLogger().handlers}")
# <<< END ADDED LOGGING >>>

# <<< COMMENTED OUT Module-Level StateManager Initialization >>>
# try:
#     # Determine the absolute path to the server stages directory
#     SERVER_STAGES_RELATIVE_PATH = "server_prompts/stages" # Standard location
#     server_stages_absolute_path = CHUNGOID_BASE_DIR / SERVER_STAGES_RELATIVE_PATH
#     logger.info(f"Server stages directory determined as: {server_stages_absolute_path}")
#
#     # --- <<< FIX: Construct absolute path for status file >>> ---
#     if GLOBAL_PROJECT_DIR:
#         chungoid_dir = Path(GLOBAL_PROJECT_DIR) / ".chungoid"
#         status_file_absolute_path = chungoid_dir / "project_status.json"
#         logger.info("Derived absolute status file path: %s", status_file_absolute_path)
#     else:
#         # This path should not be reached anymore due to the raise above
#         logger.error("Critical Error: GLOBAL_PROJECT_DIR check failed unexpectedly after initial validation.")
#         raise ValueError("StateManager initialization failed: Project directory validation failed.")
#     # --- <<< END FIX >>> ---
#
#     # Initialize StateManager with the required arguments
#     state_manager = StateManager(
#         target_directory=str(CHUNGOID_BASE_DIR), # Pass the project root directory
#         server_stages_dir=str(server_stages_absolute_path)
#     )
#
#     # Pass necessary ABSOLUTE paths/configs to StageExecutor
#     # REMOVED module-level instance, as handlers now create context-specific ones.
#
#     # Removed instantiation of non-existent ApplicationSecurity
#
#     logger.info("Chungoid MCP Server components initialized successfully.")
#
# except FileNotFoundError as fnf_error:
#     logger.exception("Initialization failed: Required file not found: %s", fnf_error)
#     raise
# except Exception as e:
#     # Use % formatting for logging exceptions
#     logger.exception("Unexpected error during module-level initialization: %s", e)
#     raise
# <<< End COMMENTED OUT Initialization Block >>>

logger.info("FastMCP instance created. StateManager initialization deferred to tool calls.")


# --- ASGI Application Entry Point ---
# No ASGI app needed for stdio transport.

# --- Session Management (Simple Global Dictionary) ---
client_sessions: Dict[str, Dict[str, Any]] = {}
stdio_client_id: Optional[str] = None


# --- Helper function to get target directory ---
def _get_target_directory(
    target_directory_arg: Optional[str] = None, ctx: Optional[Context] = None
) -> Optional[str]:
    """Gets the effective target directory, prioritizing:
    1. Explicit argument (if provided)
    2. Session context (if ctx or stdio_client_id exists)
    3. Global environment variable fallback.
    """
    # 1. Prioritize the explicit argument if it's valid
    if target_directory_arg:
        logger.debug(f"Explicit target_directory argument provided: {target_directory_arg}")
        try:
            resolved_path = Path(target_directory_arg).resolve()
            if resolved_path.is_dir():
                logger.info(f"Using valid directory provided via argument: {resolved_path}")
                return str(resolved_path)  # <<< Return the validated explicit path
            else:
                # If explicit argument is invalid, log error but continue to check session/global
                logger.error(
                    f"Provided target_directory argument '{target_directory_arg}' resolves to '{resolved_path}' which is not a directory. Checking session/global..."
                )
        except Exception as e:
            # If resolving fails, log error but continue to check session/global
            logger.error(
                f"Error resolving provided target_directory argument '{target_directory_arg}': {e}. Checking session/global..."
            )

    # 2. Check Session Context (only if explicit argument was NOT provided or was invalid)
    logger.debug("Checking session context for target directory.")
    session_target_dir: Optional[str] = None
    effective_client_id: Optional[str] = None

    # Determine the effective client ID (handle stdio case)
    original_client_id = ctx.client_id if ctx else None
    if not original_client_id:
        # Likely stdio transport, use the stored stdio_client_id
        if stdio_client_id:
            effective_client_id = stdio_client_id
            logger.debug(f"Using stored stdio_client_id for session lookup: {stdio_client_id}")
        else:
            logger.debug(
                "No original client_id and no stored stdio_client_id, cannot check session."
            )
    else:
        effective_client_id = original_client_id
        logger.debug(f"Using original client_id for session lookup: {original_client_id}")

    # Look up the directory in the session if we have an ID
    if effective_client_id and effective_client_id in client_sessions:
        session_target_dir = client_sessions[effective_client_id].get("target_directory")
        if session_target_dir:
            # Validate the path stored in the session
            if Path(session_target_dir).is_dir():
                logger.info(
                    f"Using project directory from session context for client {effective_client_id}: {session_target_dir}"
                )
                return session_target_dir
            else:
                logger.warning(
                    f"Path from session context for client {effective_client_id} ('{session_target_dir}') is not a valid directory. Ignoring."
                )
        else:
            logger.debug(
                f"No 'target_directory' found in session context for client {effective_client_id}."
            )
    elif effective_client_id:
        logger.debug(f"No session found in client_sessions for client {effective_client_id}.")
    # End Session Check

    # 3. Fallback to the global environment variable if no valid argument or session context found
    logger.debug("Checking environment variable fallback for target directory.")
    if GLOBAL_PROJECT_DIR:
        logger.info(
            f"Using project directory from environment variable fallback: {GLOBAL_PROJECT_DIR}"
        )
        return GLOBAL_PROJECT_DIR
    else:
        # 4. Error if neither argument, session, nor env var yields a valid directory
        logger.error(
            "Could not determine project directory from argument, session, or environment variable. Check setup."
        )
        return None


# --- Helper function to initialize StateManager within a tool call --- <<< UNCOMMENTED >>>
def _initialize_state_manager_for_target(target_dir_path_str: str) -> StateManager:
    """Initializes StateManager for a given target directory path. Raises exceptions on failure."""
    logger.debug(f"Attempting to initialize StateManager for target: {target_dir_path_str}")
    try:
        target_dir_path = Path(target_dir_path_str)
        if not target_dir_path.is_dir():
            raise ValueError(f"Target directory '{target_dir_path_str}' is not a valid directory.")

        # Define paths needed for StateManager initialization
        SERVER_STAGES_RELATIVE_PATH = "server_prompts/stages"  # Standard location
        server_stages_absolute_path = CHUNGOID_BASE_DIR / SERVER_STAGES_RELATIVE_PATH

        # Initialize StateManager
        state_manager_instance = StateManager(
            target_directory=str(target_dir_path),  # Use the provided target path
            server_stages_dir=str(server_stages_absolute_path),
        )
        # <<< ADDED CODE LOADING CHECK >>>
        try:
            sm_file = inspect.getfile(StateManager)
            logger.debug(f"StateManager class loaded from: {sm_file}")
        except Exception as inspect_err:
            logger.error(f"Could not inspect StateManager file path: {inspect_err}")
        # <<< END ADDED CODE LOADING CHECK >>>

        logger.info(f"StateManager initialized successfully for target: {target_dir_path_str}")
        return state_manager_instance
    except (StatusFileError, ValueError, FileNotFoundError, Exception) as e:
        # logger.exception(f"Failed to initialize StateManager for target '{target_dir_path_str}': {e}")
        # <<< REPLACED LOGGING WITH DIRECT PRINT TO STDERR FOR DIAGNOSIS >>>
        import sys

        print(
            f"!!! STDERR: EXCEPTION CAUGHT in _initialize_state_manager_for_target: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        import traceback

        traceback.print_exc(file=sys.stderr)
        # <<< END REPLACEMENT >>>
        raise  # Re-raise the exception to be caught by the calling tool handler


# Define a function to handle common error types and return JSONResponse
def handle_errors(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (StateManager.ProjectNotFoundError, FileNotFoundError) as e:
            logger.warning(f"Resource not found error: {e}", exc_info=True)
            raise HTTPException(status_code=404, detail=str(e))
        except ChromaOperationError as e:
            logger.error(f"ChromaDB operation failed: {e}", exc_info=True)
            # Return 503 Service Unavailable as the DB backend has issues
            raise HTTPException(status_code=503, detail=f"Database operation failed: {e}")
        except (PromptLoadError, PromptRenderError) as e:
            logger.error(f"Prompt loading/rendering failed: {e}", exc_info=True)
            # Return 500 as it's a server configuration/internal error
            raise HTTPException(status_code=500, detail=f"Prompt configuration error: {e}")
        except StageExecutionError as e:
            logger.error(f"Stage execution failed: {e}", exc_info=True)
            # Return 500 as the core logic failed
            raise HTTPException(status_code=500, detail=f"Stage execution error: {e}")
        except ToolExecutionError as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            # Use 400 Bad Request if it's likely due to bad input (e.g., invalid path),
            # 404 for not found, or 500 if it's an internal server issue (e.g., permissions).
            status_code = 500 # Default to internal server error
            err_str = str(e).lower() # Lowercase for case-insensitive check

            if "access denied" in err_str or "outside the project boundary" in err_str:
                 status_code = 400 # Bad request due to invalid path
            elif "corrupted or not valid json" in err_str:
                status_code = 500 # Indicate server-side data corruption issue
            elif "not found" in err_str or "could not find" in err_str:
                 status_code = 404 # Resource not found

            # Keep 500 for filesystem errors like permissions, disk full etc.
            # The error message `e` provides specifics.

            raise HTTPException(status_code=status_code, detail=f"Tool execution error: {e}")
        except ValueError as e: # Catch general value errors (e.g., invalid JSON in request body)
            logger.warning(f"Value error during request processing: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException: # Re-raise HTTPExceptions
            raise
        except Exception as e:
            logger.exception(f"Unhandled exception during request: {e}") # Log full traceback
            # Return 500 for any other unexpected errors
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {type(e).__name__}")
    # Preserve original signature for FastAPI documentation
    import functools
    functools.update_wrapper(wrapper, func)
    return wrapper


# --- MCP Tool Handlers (Top-Level Async Functions) ---
# <<< Restore original handlers with deferred StateManager init >>>
@mcp.tool(
    name="set_project_context",
    description="Sets the target project directory for the current client session, removing the need to specify it in subsequent calls.",
)
async def set_project_context(target_directory: str, ctx: Context) -> dict:
    """Sets the target_directory in the server's session state for the calling client."""
    global stdio_client_id  # <<< Declare global to modify it
    original_client_id = ctx.client_id  # <<< Use original_client_id
    effective_client_id: Optional[str] = None  # <<< Define effective_client_id

    logger.info(
        f"set_project_context called for original_client_id: {original_client_id}, target_directory: '{target_directory}'"
    )

    if not original_client_id:
        # Client ID missing from context (likely stdio transport)
        logger.info("Original client_id missing from context. Handling as potential stdio session.")
        if stdio_client_id is None:
            # Generate and store a persistent ID for this stdio server instance
            stdio_client_id = str(uuid.uuid4())
            logger.info(f"Generated and stored new stdio_client_id: {stdio_client_id}")
        else:
            logger.info(f"Using existing stdio_client_id: {stdio_client_id}")
        effective_client_id = stdio_client_id
    else:
        # Client ID provided by context (e.g., websocket transport)
        effective_client_id = original_client_id

    # Now, proceed using the effective_client_id
    if not effective_client_id:
        # This case should theoretically not happen if logic above is correct
        logger.error("Critical error: Effective client ID is still missing after handling.")
        return {
            "status": "error",
            "message": "Internal server error: Could not determine client identifier.",
        }

    # Basic validation of the directory path (check if it's a non-empty string)
    if (
        not target_directory
        or not isinstance(target_directory, str)
        or not target_directory.strip()
    ):
        logger.error(
            f"Invalid target_directory provided for client {effective_client_id}: '{target_directory}'"
        )
        return {
            "status": "error",
            "message": "Invalid 'target_directory' provided. It must be a non-empty string.",
        }

    # Simple path check (does not guarantee it's a valid *project* dir)
    # More robust checks could be added if needed (e.g., check for .chungoid)
    target_path = Path(target_directory)
    if not target_path.exists() or not target_path.is_dir():
        logger.warning(
            f"Target directory '{target_directory}' provided for client {effective_client_id} does not exist or is not a directory."
        )
        # Allow setting anyway, but log a warning. Initialization/other tools will fail later if invalid.

    if effective_client_id not in client_sessions:  # <<< Use effective_client_id
        client_sessions[effective_client_id] = {}  # <<< Use effective_client_id
        logger.info(
            f"Initialized new session for client_id: {effective_client_id}"
        )  # <<< Use effective_client_id

    client_sessions[effective_client_id]["target_directory"] = str(
        target_path.resolve()
    )  # Store resolved absolute path # <<< Use effective_client_id
    stored_path = client_sessions[effective_client_id][
        "target_directory"
    ]  # <<< Use effective_client_id
    logger.info(
        f"Set target_directory for client {effective_client_id} to: {stored_path}"
    )  # <<< Use effective_client_id

    return {
        "status": "success",
        "message": f"Project context set to {stored_path} for this session.",
    }


@mcp.tool(
    name="initialize_project",
    description="Copies necessary template files and directories into the REQUIRED target directory to set up a new project for Chungoid.",
)
async def handle_initialize_project(target_directory: str, ctx: Optional[Context] = None) -> dict:
    """Handler for the 'initialize_project' tool. Copies templates to the specified directory.

    On success, also sets the target directory context for the current session.
    """
    logger.info(f"handle_initialize_project called. Target_directory: '{target_directory}'")

    # --- Validation (Simpler: Only check if it's a non-empty string) ---
    if (
        not target_directory
        or not isinstance(target_directory, str)
        or not target_directory.strip()
    ):
        logger.error("Missing or invalid 'target_directory' argument.")
        error_message = (
            "Error: Missing or invalid 'target_directory' argument. "
            "ACTION REQUIRED: Agent must determine the current working directory (CWD) by running `pwd`. "
            "Then, recall this tool ('initialize_project') with the CWD path provided as the 'target_directory' argument. "
            'Example: `@chungoid initialize_project target_directory="/path/from/pwd"`'  # Simplified example
        )
        return {"status": "error", "message": error_message}
    # --- End Validation ---

    logger.info(f"Proceeding with initialization in target_directory: '{target_directory}'.")
    try:
        # Use the provided target_directory argument directly
        target_dir = Path(target_directory).resolve()
        logger.info(f"Resolved target directory: {target_dir}")

        # Ensure target directory exists or create it
        if not target_dir.is_dir():
            if target_dir.exists():
                logger.error(f"Target directory path exists but is a file: {target_dir}")
                return {
                    "status": "error",
                    "message": f"Target directory path exists but is a file: {target_dir}",
                }
            else:
                logger.info(f"Target directory does not exist, creating: {target_dir}")
                try:
                    target_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logger.error(f"Failed to create target directory {target_dir}: {e}")
                    return {"status": "error", "message": f"Failed to create target directory: {e}"}

        # Define source paths relative to the Chungoid installation base
        source_server_prompts_dir = CHUNGOID_BASE_DIR / "server_prompts"
        # source_templates_dir = CHUNGOID_BASE_DIR / "templates" # REMOVED

        # Define specific source files
        # source_build_prompt = source_templates_dir / "build-prompt.txt" # REMOVED
        source_initial_status = source_server_prompts_dir / "initial_status.json"
        # source_goal_template = source_templates_dir / "drop-in-goal.txt" # No longer copied

        # Define destination paths in the target directory (using the argument)
        dest_chungoid_dir = target_dir / ".chungoid"
        # dest_build_prompt = dest_chungoid_dir / "build-prompt.txt" # REMOVED
        dest_status_file = dest_chungoid_dir / "project_status.json"
        # dest_goal_file = target_dir / "goal.txt" # No longer created here
        # dest_stages_dir = dest_chungoid_dir / "templates" / "stages" # REMOVED destination mapping
        # Destination for the auto-start rule - REMOVED
        # dest_rules_dir = target_dir / ".cursor" / "rules" / "global-rules"
        # dest_auto_start_rule = dest_rules_dir / "chungoid_auto_start_stage0.mdc"

        # Define source for the auto-start rule - REMOVED
        # source_rules_template_dir = CHUNGOID_BASE_DIR / "templates" / "rules"
        # source_auto_start_rule = source_rules_template_dir / "auto_start_stage0.mdc"

        # Overwrite is no longer an option via kwargs
        overwrite = False  # Default to False, or decide a fixed behavior

        copied_items = []
        skipped_items = []
        errors = []

        logger.info("Ensuring target directory structure exists...")
        # Ensure the main .chungoid directory and stages dir exist
        # dest_stages_dir.mkdir(parents=True, exist_ok=True) # REMOVED creation of old stages dir
        dest_chungoid_dir.mkdir(parents=True, exist_ok=True)  # Ensure .chungoid still gets created
        # dest_rules_dir.mkdir(parents=True, exist_ok=True) # REMOVED rules dir creation

        # --- Copy individual files ---
        files_to_copy = [
            # (source_build_prompt, dest_build_prompt), # REMOVED
            (source_initial_status, dest_status_file),
            # (source_auto_start_rule, dest_auto_start_rule), # REMOVED rule file copy
        ]

        for src, dest in files_to_copy:
            if not src.exists():
                msg = f"Source file missing: {src}"
                logger.error(msg)
                errors.append(msg)
                continue  # Skip this file

            if dest.exists() and not overwrite:
                msg = (
                    f"Skipping copy: Destination file {dest} already exists and overwrite is False."
                )
                logger.warning(msg)
                skipped_items.append(str(dest))
                continue  # Skip copying if destination exists and overwrite is false

            try:
                shutil.copy2(src, dest)  # copy2 preserves metadata
                logger.info(f"Copied {src} to {dest}")
                copied_items.append(str(dest))
            except Exception as copy_err:
                msg = f"Error copying {src} to {dest}: {copy_err}"
                logger.exception(msg)
                errors.append(msg)

        # --- Copy stages directory --- # REMOVED ENTIRE BLOCK

        # --- Construct response ---
        if errors:
            return {
                "status": "error",
                "message": "Project initialization encountered errors.",
                "copied": copied_items,
                "skipped": skipped_items,
                "errors": errors,
            }
        elif skipped_items and not copied_items:
            return {
                "status": "skipped",
                "message": "Project initialization skipped. All target files/dirs already exist.",
                "skipped": skipped_items,
            }
        else:
            # <<< SUCCESS: Also set session context >>>
            success_message = "Project initialized successfully."
            context_set_message = ""
            if ctx:  # Only try to set context if ctx is available
                try:
                    # Resolve path once more just to be safe
                    resolved_target_dir = str(target_dir.resolve())

                    # --- Replicate logic from set_project_context --- #
                    global stdio_client_id
                    original_client_id = ctx.client_id
                    effective_client_id: Optional[str] = None

                    if not original_client_id:
                        if stdio_client_id is None:
                            stdio_client_id = str(uuid.uuid4())
                        effective_client_id = stdio_client_id
                    else:
                        effective_client_id = original_client_id

                    if effective_client_id:
                        if effective_client_id not in client_sessions:
                            client_sessions[effective_client_id] = {}
                        client_sessions[effective_client_id]["target_directory"] = (
                            resolved_target_dir
                        )
                        context_set_message = (
                            f" Session context automatically set to {resolved_target_dir}."
                        )
                        logger.info(
                            f"Automatically set session context for client {effective_client_id} to {resolved_target_dir} after successful initialization."
                        )
                    else:
                        logger.warning(
                            "Could not determine effective client ID during initialization context setting."
                        )
                        context_set_message = (
                            " Could not automatically set session context (client ID missing)."
                        )
                    # --- End replicated logic --- #
                except Exception as context_err:
                    logger.error(
                        f"Error automatically setting session context after initialization: {context_err}"
                    )
                    context_set_message = f" Project initialized, but failed to automatically set session context: {context_err}"

            return {
                "status": "success",
                "message": success_message + context_set_message,
                "copied": copied_items,
                "skipped": skipped_items,
                "next_action_hint": "execute_next_stage",  # <<< Update hint
            }

    except Exception as e:
        # Catch unexpected errors during path resolution or copying
        logger.exception(
            f"Unexpected error during project initialization for target '{target_directory}': {e}"
        )
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}


@mcp.tool(
    name="get_initialization_context",
    description="Provides instructions and a template for calling 'initialize_project' when the target directory needs to be the current working directory.",
)
async def handle_get_initialization_context(**kwargs) -> dict:
    """Handler for the 'get_initialization_context' tool. Guides the agent to determine CWD and call initialize_project."""
    logger.info("Received 'get_initialization_context' request.")

    # Construct response with the extremely simplified template
    response = {
        "status": "ACTION_REQUIRED",
        "agent_instruction": "Run `pwd` to get your current working directory (CWD).",
        "next_tool_call_template": {
            "tool_name": "initialize_project",
            "arguments": {
                # Only target_directory now
                "target_directory": "<REPLACE_WITH_CWD_OUTPUT>"
            },
        },
        "next_step_description": (
            "After running `pwd`, call the 'initialize_project' tool using the arguments template above, replacing '<REPLACE_WITH_CWD_OUTPUT>' with the actual path returned by `pwd`."  # Simplified description
        ),
    }
    return response


@mcp.tool(
    name="execute_next_stage",
    description="Executes the next pending stage. Uses environment variable for project dir if target_directory not provided.",
)
async def handle_execute_next_stage(
    target_directory: Optional[str] = None, ctx: Optional[Context] = None
) -> dict:
    """Handler for the 'execute_next_stage' tool."""
    logger.info(f"Received 'execute_next_stage' request (explicit target: '{target_directory}')")
    # Pass target_directory_arg explicitly now
    effective_target_directory = _get_target_directory(
        target_directory_arg=target_directory, ctx=ctx
    )
    if not effective_target_directory:
        return {
            "status": "error",
            "message": "Could not determine project directory. Ensure client is configured with CHUNGOID_PROJECT_DIR or set session context.",
        }

    logger.info(f"Executing next stage for effective target: '{effective_target_directory}'")

    try:
        # <<< Initialize StateManager within the handler >>>
        context_state_manager = _initialize_state_manager_for_target(effective_target_directory)

        # --- Get Next Stage --- #
        next_stage_num = context_state_manager.get_next_stage()
        last_status = context_state_manager.get_last_status()  # Get last status for context

        # --- Prepare arguments for StageExecutor --- #
        decision_config = {}
        # <<< REMOVE context retrieval/formatting block >>>
        # reflections_context_str = ...
        # artifacts_context_str = ...
        # try...except block for context retrieval... REMOVE
        # <<< END REMOVE >>>

        # <<< Calculate server prompt paths >>>
        script_dir = Path(__file__).parent
        server_stages_dir = str((script_dir / "server_prompts" / "stages").resolve())
        common_template_path = str(
            (script_dir / "server_prompts" / "common_template.yaml").resolve()
        )
        logger.debug(f"Calculated server_stages_dir: {server_stages_dir}")
        logger.debug(f"Calculated common_template_path: {common_template_path}")

        # --- Stage Loading & Execution --- #
        stage_executor = StageExecutor(
            state_manager=context_state_manager,
            decision_logic_config=decision_config,
            project_root_dir=effective_target_directory,
            reflections_summary="",  # <<< Pass empty string for now
            server_stages_dir=server_stages_dir,
            common_template_path=common_template_path,
        )

        # <<< ADDED DEBUGGING >>>
        logger.debug(f"StageExecutor prompt_manager initialized: {hasattr(stage_executor, 'prompt_manager')}")
        if hasattr(stage_executor, 'prompt_manager') and stage_executor.prompt_manager:
            logger.debug(f"PromptManager instance: {stage_executor.prompt_manager}")
            logger.debug(f"PromptManager stage_definitions keys: {list(stage_executor.prompt_manager.stage_definitions.keys())}")
            # <<< Simplified check for key type >>>
            stage_1_key = next((k for k in stage_executor.prompt_manager.stage_definitions if k == 1 or k == 1.0), None)
            if stage_1_key is not None:
                 logger.debug(f"Type of stage 1 key found: {type(stage_1_key).__name__}")
            else:
                 logger.debug("Stage 1 key (1 or 1.0) not found in stage_definitions.")
            # <<< End Simplified check >>>
        else:
            logger.error("PromptManager not found or not initialized on StageExecutor!")
        # <<< END ADDED DEBUGGING >>>

        # Pass EMPTY context_data to the prompt rendering initially.
        # Agent is now responsible for using tools to get context.
        prompt_content = stage_executor.prompt_manager.get_rendered_prompt(
            stage_number=next_stage_num,
            context_data={},  # <<< Pass empty dictionary
        )

        logger.info(f"Returning prompt for stage {next_stage_num}")
        return {"status": "success", "next_stage": next_stage_num, "prompt": prompt_content}

    # <<< Catch StateManager init errors specifically >>>
    except (StatusFileError, ValueError, FileNotFoundError) as sm_init_error:
        err_msg = f"Failed to initialize project state for next stage: {sm_init_error}. Project might be uninitialized. Try running 'initialize_project' first."
        logger.error(
            f"StateManager initialization failed for next stage execution in '{effective_target_directory}': {sm_init_error}"
        )
        return {"status": "error", "message": err_msg}
    except StageExecutionError as e:
        logger.error(f"Stage execution error in '{effective_target_directory}': {e}")
        return {"status": "error", "message": str(e)}
    except RuntimeError as e:
        # Catch specific errors from StateManager like cannot determine next stage
        logger.error(
            f"Runtime error during next stage execution in '{effective_target_directory}': {e}"
        )
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception(
            f"Unexpected error during next stage execution in '{effective_target_directory}': {e}"
        )
        return {"status": "error", "message": f"An unexpected server error occurred: {e}"}


@mcp.tool(
    name="get_project_status",
    description="Retrieves project status. Uses environment variable for project dir if target_directory not provided.",
)
async def handle_get_project_status(
    target_directory: Optional[str] = None, ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """Retrieves the project status from .chungoid/project_status.json."""
    logger.info(f"Getting project status for: {target_directory}")
    target_path = Path(target_directory).resolve()
    project_status_file = target_path / ".chungoid" / "project_status.json"

    try:
        if not target_path.is_dir():
             raise FileNotFoundError(f"Target project directory not found: {target_directory}")
        if not project_status_file.is_file():
            # If the status file specifically is missing, maybe return a default or indicate initialization needed?
            # For now, treat as an error requiring explicit initialization.
            # Consider if `.chungoid` dir missing vs only the file missing warrants different handling.
            raise FileNotFoundError(f"Project status file not found: {project_status_file}. Project may need initialization.")

        with open(project_status_file, "r", encoding="utf-8") as f:
            status_data = json.load(f)
        logger.info(f"Successfully retrieved project status from {project_status_file}")
        return status_data
    except FileNotFoundError as e:
        logger.error(f"Error getting project status: {e}", exc_info=False) # Don't need full trace for FileNotFoundError usually
        raise ToolExecutionError(f"Could not find project directory or status file: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding project status JSON in {project_status_file}: {e}", exc_info=True)
        raise ToolExecutionError(f"Project status file ({project_status_file}) is corrupted or not valid JSON: {e}") from e
    except (OSError, IOError) as e:
        logger.error(f"Filesystem error reading project status file {project_status_file}: {e}", exc_info=True)
        raise ToolExecutionError(f"Could not read project status file {project_status_file}: {e}") from e
    except Exception as e: # Catch any other unexpected error
        logger.error(f"Unexpected error getting project status for {target_directory}: {e}", exc_info=True)
        raise ToolExecutionError(f"An unexpected error occurred while getting project status: {e}") from e


@mcp.tool(
    name="submit_stage_artifacts",
    description="Submits artifacts. Uses environment variable for project dir if target_directory not provided.",
)
async def handle_submit_stage_artifacts(
    target_directory: Optional[str] = None,
    stage_number: float = -1.0,
    generated_artifacts: Dict[str, str] = {},
    stage_result_status: str = "UNKNOWN",
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Handler for the 'submit_stage_artifacts' tool."""
    logger.info(
        f"Received 'submit_stage_artifacts' (explicit target: '{target_directory}') for stage='{stage_number}', status='{stage_result_status}'"
    )
    effective_target_directory = _get_target_directory(target_directory, ctx)
    if not effective_target_directory:
        return {
            "status": "error",
            "message": "Could not determine project directory. Provide valid 'target_directory' argument or set CHUNGOID_PROJECT_DIR env var and restart server.",
        }

    logger.info(
        f"Submitting artifacts for effective target: '{effective_target_directory}', stage: {stage_number}"
    )
    target_dir_path = Path(effective_target_directory)

    # --- Input Validation --- #
    if not isinstance(stage_number, (int, float)) or stage_number < 0:
        return {
            "status": "error",
            "message": "Invalid stage_number. Must be a non-negative number.",
        }
    if not isinstance(generated_artifacts, dict):
        return {"status": "error", "message": "Invalid generated_artifacts. Must be a dictionary."}
    if not target_dir_path.is_dir():
        return {
            "status": "error",
            "message": f"Target directory not found or invalid: {target_dir_path}",
        }
    valid_statuses = ["DONE", "FAIL", "PASS", "UNKNOWN"]
    if stage_result_status.upper() not in valid_statuses:
        return {
            "status": "error",
            "message": f"Invalid stage_result_status '{stage_result_status}'. Use {valid_statuses}.",
        }
    final_status = stage_result_status.upper()
    # --- End Input Validation --- #

    # --- Determine Artifacts to Process --- #
    artifacts_to_process: Dict[str, Optional[str]] = {}
    if generated_artifacts:
        logger.info("Processing artifacts explicitly provided in the tool call.")
        # Validate provided content is string
        for path, content in generated_artifacts.items():
            if not isinstance(content, str):
                logger.error(
                    f"Invalid content type for provided artifact '{path}'. Expected string, got {type(content)}. Skipping artifact writing."
                )
                return {
                    "status": "error",
                    "message": f"Invalid content type for provided artifact '{path}'.",
                }
        artifacts_to_process = generated_artifacts  # Use provided dict
    else:
        # Infer artifacts from expected stage output (simple hardcoded example for Stage 0 for now)
        # TODO: Load this dynamically from stage prompt yaml or previous status
        logger.warning(
            "No artifacts provided in tool call. Inferring artifacts based on hardcoded Stage 0 expectation."
        )
        expected_artifacts = []
        if stage_number == 0:
            expected_artifacts = [
                "dev-docs/design/requirements.md",
                "dev-docs/design/research_notes.md",
                "dev-docs/design/blueprint.md",
            ]
        elif stage_number == 1:
            expected_artifacts = ["dev-docs/design/validation_report.json"]
        # Add other stages as needed
        else:
            logger.warning(
                f"No hardcoded expected artifacts found for stage {stage_number}. Will only update status."
            )

        if expected_artifacts:
            logger.info(f"Inferred artifacts for stage {stage_number}: {expected_artifacts}")
            for path in expected_artifacts:
                artifacts_to_process[path] = None  # Mark for reading, content is None initially
        else:
            logger.info(
                f"No artifacts inferred for stage {stage_number}. Only status will be updated."
            )
    # --- End Determine Artifacts --- #

    written_files = []
    errors = []
    # --- Artifact Handling Loop --- #
    for rel_path_str, content_or_none in artifacts_to_process.items():
        if not isinstance(rel_path_str, str) or not rel_path_str:
            msg = f"Invalid artifact path key (must be non-empty string): {rel_path_str}"
            logger.error(msg)
            errors.append(msg)
            continue

        if ".." in rel_path_str or rel_path_str.startswith("/"):
            msg = f"Security risk: Invalid relative path '{rel_path_str}'"
            logger.error(msg)
            errors.append(msg)
            continue

        try:
            rel_path = Path(rel_path_str)
            abs_path = (target_dir_path / rel_path).resolve()
            # Security Check: Ensure path is within target_dir_path
            if not abs_path.is_relative_to(target_dir_path):
                msg = f"Security risk: Resolved path '{abs_path}' is outside target directory '{target_dir_path}'"
                logger.error(msg)
                errors.append(msg)
                continue

            if content_or_none is not None:
                # Content was provided explicitly, write it
                if not isinstance(content_or_none, str):
                    msg = f"Invalid explicit content for '{rel_path_str}' (must be string). Type: {type(content_or_none)}"
                    logger.error(msg)
                    errors.append(msg)
                    continue
                logger.info(f"Writing explicitly provided artifact content to: {abs_path}")
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(content_or_none, encoding="utf-8")
                written_files.append(rel_path_str)  # Mark as written
            else:
                # Content not provided, assume file exists (was generated by agent)
                logger.info(f"Artifact '{rel_path_str}' was inferred. Assuming it exists on disk.")
                if not abs_path.is_file():
                    msg = f"Inferred artifact file not found: {abs_path}"
                    logger.error(msg)
                    errors.append(msg)  # Report error if inferred file doesn't exist
                    continue
                else:
                    # File exists, mark it as 'written' (or processed) for status update/ChromaDB
                    written_files.append(rel_path_str)

        except OSError as e:
            msg = f"OS Error handling artifact '{rel_path_str}' at '{abs_path}': {e}"
            logger.exception(msg)
            errors.append(msg)
        except Exception as e:
            msg = f"Unexpected error handling artifact '{rel_path_str}' at '{abs_path}': {e}"
            logger.exception(msg)
            errors.append(msg)
    # --- End Artifact Handling Loop --- #

    status_updated = False
    status_update_error = None
    # --- Status Update (includes ChromaDB persistence) --- #
    if not errors:
        try:
            # <<< Initialize StateManager within the handler >>>
            context_state_manager = _initialize_state_manager_for_target(effective_target_directory)

            # Store reflections if provided
            reflection_data = getattr(ctx, "reflection_data", None) if ctx else None
            if isinstance(reflection_data, dict):
                logger.info(f"Storing reflection data for stage {stage_number}...")
                reflection_stored = context_state_manager.store_reflection(reflection_data)
                if not reflection_stored:
                    logger.warning("Failed to store reflection data.")
            else:
                if reflection_data is not None:  # Log if present but not a dict
                    logger.warning(
                        f"Received reflection_data but it was not a dictionary (type: {type(reflection_data)}). Skipping storage."
                    )

            # <<< NEW: Store written artifacts in ChromaDB >>>
            if written_files:
                # <<< ADDED DIAGNOSTIC LOGGING >>>
                logger.info(
                    f"Identified {len(written_files)} artifacts for ChromaDB storage: {written_files}"
                )
                # <<< END DIAGNOSTIC LOGGING >>>
                logger.info(
                    f"Attempting to store {len(written_files)} written artifacts in ChromaDB..."
                )
                for rel_path_str in written_files:
                    try:
                        # Read the content back from the file that was just written
                        abs_path = (target_dir_path / rel_path_str).resolve()
                        content = abs_path.read_text(encoding="utf-8")

                        # Determine artifact type (simple heuristic based on path for now)
                        artifact_type_guess = "unknown"
                        if Path(rel_path_str).suffix in [
                            ".py",
                            ".js",
                            ".ts",
                            ".java",
                            ".c",
                            ".cpp",
                            ".go",
                        ]:
                            artifact_type_guess = "code"
                        elif Path(rel_path_str).suffix in [".md", ".txt", ".rst"]:
                            artifact_type_guess = "document"
                        elif Path(rel_path_str).suffix in [".json", ".yaml", ".xml"]:
                            artifact_type_guess = "config"
                        # Add more specific types based on path (e.g., dev-docs/design -> design)
                        if "dev-docs/design" in rel_path_str:
                            artifact_type_guess = "design"
                        if "dev-docs/planning" in rel_path_str:
                            artifact_type_guess = "planning"
                        if "dev-docs/reports" in rel_path_str:
                            artifact_type_guess = "report"
                        if "dev-docs/validation" in rel_path_str:
                            artifact_type_guess = "validation"
                        if "dev-docs/release" in rel_path_str:
                            artifact_type_guess = "release"
                        if "dev-docs/reflections" in rel_path_str:
                            artifact_type_guess = (
                                "reflection_doc"  # Separate from structured reflection
                            )

                        logger.info(
                            f"Storing artifact '{rel_path_str}' (type: {artifact_type_guess}) in ChromaDB for stage {stage_number}"
                        )
                        # --- Wrap ChromaDB call in try...except --- #
                        try:
                            stored_ok = context_state_manager.store_artifact_context_in_chroma(
                                rel_path=rel_path_str,
                                content=content,
                                stage_number=float(stage_number),
                                artifact_type=artifact_type_guess,
                            )
                            if not stored_ok:
                                logger.warning(
                                    f"StateManager reported failure storing artifact context for {rel_path_str}. Check StateManager logs."
                                )
                                # Optionally add to errors list if this should halt the overall submission:
                                errors.append(f"StateManager failed to store {rel_path_str} context in ChromaDB (returned False).")
                        except ChromaOperationError as chroma_e:
                            logger.error(
                                f"ChromaDB Error storing artifact context for '{rel_path_str}': {chroma_e}"
                            )
                            errors.append(f"ChromaDB Error storing {rel_path_str}: {chroma_e}")
                        # --- End wrap ChromaDB call --- #

                    except FileNotFoundError:
                        logger.error(
                            f"ChromaDB Store Error: File '{rel_path_str}' not found after writing, cannot store context."
                        )
                        errors.append(
                            f"File not found for ChromaDB storage: {rel_path_str}"
                        )  # Add to errors
                    except Exception as chroma_err:
                        logger.exception(
                            f"ChromaDB Store Error: Failed to store artifact '{rel_path_str}' context: {chroma_err}"
                        )
                        # Decide if this should prevent status update
                        # Adding to errors for now to make it visible
                        errors.append(f"Error storing {rel_path_str} context: {chroma_err}")

            # Update the main status file
            status_updated = context_state_manager.update_status(
                stage=float(stage_number),
                status=final_status,  # Use validated final_status
                artifacts=written_files,  # Pass list of successfully written files
            )
            if not status_updated:
                status_update_error = "StateManager reported status update failed."
                logger.error(status_update_error)

        # <<< Catch StateManager init errors specifically >>>
        except (StatusFileError, ValueError, FileNotFoundError) as sm_init_error:
            status_update_error = (
                f"Failed to initialize project state for status update: {sm_init_error}"
            )
            logger.exception(status_update_error)
        except Exception as e:
            logger.exception(f"Unexpected error during status update for stage {stage_number}")
            status_update_error = f"Unexpected error during status update: {type(e).__name__}"
    else:
        status_update_error = "Status update skipped due to artifact writing errors."
        logger.warning(status_update_error)
    # --- End Status Update --- #

    # --- Construct Response --- #
    if errors:
        return {
            "status": "error",
            "message": "Errors occurred during artifact writing.",
            "errors": errors,
            "written_files": written_files,
        }
    elif status_update_error:
        # Artifacts might have been written even if status update failed
        return {
            "status": "error",
            "message": f"Artifacts partially/fully written, but status update failed: {status_update_error}",
            "written_files": written_files,
            "errors": errors,
        }
    elif not status_updated:
        # Should ideally be covered by status_update_error, but as a fallback
        return {
            "status": "error",
            "message": "Artifacts written, but status update failed for an unknown reason.",
            "written_files": written_files,
        }
    else:
        return {
            "status": "success",
            "message": f"Stage {stage_number} artifacts submitted and status updated to {final_status}.",
            "written_files": written_files,
            "next_action_hint": "execute_next_stage",
        }


@mcp.tool(
    name="load_reflections",
    description="Loads past reflections. Uses environment variable for project dir if target_directory not provided.",
)
async def handle_load_reflections(
    target_directory: Optional[str] = None, ctx: Optional[Context] = None
) -> dict:
    """Handler for the 'load_reflections' tool."""
    logger.info(
        f"Received 'load_reflections' request (explicit target: '{target_directory}') with args: {ctx}"
    )
    effective_target_directory = _get_target_directory(target_directory, ctx)
    if not effective_target_directory:
        return {
            "status": "error",
            "message": "Could not determine project directory. Provide valid 'target_directory' argument or set CHUNGOID_PROJECT_DIR env var and restart server.",
            "summary": "",
        }

    logger.info(f"Loading reflections for effective target: '{effective_target_directory}'")
    try:
        # <<< Initialize StateManager within the handler >>>
        context_state_manager = _initialize_state_manager_for_target(effective_target_directory)

        # TODO: Refine reflection loading...
        # query_string = "reflection" # No longer needed
        reflections_data = None
        limit_to_load = 20 # Set a reasonable default limit
        # --- Wrap ChromaDB call in try...except --- #
        try:
            # <<< CALL NEW METHOD >>>
            logger.info(f"Calling get_all_reflections with limit={limit_to_load}")
            reflections_data = context_state_manager.get_all_reflections(limit=limit_to_load)
        except ChromaOperationError as chroma_e:
            logger.error(f"ChromaDB operation failed while loading reflections: {chroma_e}")
            return {
                "status": "error",
                "message": f"Failed to load reflections from database: {chroma_e}",
                "summary": "",
            }
        # --- End wrap ChromaDB call --- #
        summary = ""

        # --- Processing and Summarization --- #
        if reflections_data: # Now check reflections_data (the list of dicts)
            # The data is already in the desired list format from get_all_reflections
            # Format: [{'id': ..., 'metadata': {...}, 'document': ..., 'timestamp': ...}, ...]
            processed_reflections = reflections_data # Use the data directly

            # --- Summarization --- #
            if processed_reflections:
                num_to_summarize = len(processed_reflections)
                # Update summary message to reflect non-query based retrieval
                summary_parts = [f"Summary of last {num_to_summarize} reflections (up to limit {limit_to_load}):"]
                for reflection_entry in processed_reflections:
                    # Extract info directly from the reflection entry dictionary
                    stage = reflection_entry.get("metadata", {}).get("stage_number", "Unknown Stage")
                    timestamp = reflection_entry.get("timestamp", "No Timestamp")
                    # Simple summary: Stage and Timestamp
                    part = f"  - Stage {stage} (Timestamp: {timestamp}): Retrieved." # Adjust summary
                    # TODO: Enhance summary based on reflection document content if needed
                    summary_parts.append(part)
                summary = "\n".join(summary_parts)
                logger.info(
                    f"Successfully loaded and summarized {num_to_summarize} reflections via StateManager (get_all_reflections)."
                )
            # This case should ideally not be hit if reflections_data was non-empty
            # but processed_reflections became empty (e.g., due to future filtering)
            elif not summary:
                summary = "No reflections found."
                logger.info("No reflections found via StateManager (get_all_reflections).")
        # --- End Processing and Summarization --- #
        else: # Handles case where get_all_reflections returned empty list
             if not summary:
                summary = f"No reflections found (limit: {limit_to_load})."
                logger.info(f"No reflections found via StateManager (get_all_reflections, limit: {limit_to_load}).")

        return {"status": "success", "summary": summary}

    # <<< Catch StateManager init errors specifically >>>
    except (StatusFileError, ValueError, FileNotFoundError) as sm_init_error:
        logger.error(
            f"StateManager initialization failed retrieving reflections for target '{effective_target_directory}': {sm_init_error}"
        )
        return {
            "status": "error",
            "message": f"Failed to initialize project state for reflection retrieval: {sm_init_error}",
            "summary": "",
        }
    except Exception as e:
        logger.exception(
            f"Error retrieving reflections via ChromaDB for target '{effective_target_directory}': {e}"
        )
        return {
            "status": "error",
            "message": f"An unexpected error occurred while retrieving reflections: {e}",
            "summary": "",
        }


@mcp.tool(
    name="get_file",
    description="Reads the content of a specific file within the project directory. Path must be relative to the project root.",
)
async def handle_get_file(target_directory: str, relative_path: str) -> Dict[str, str]:
    """Reads the content of a specific file within the project directory."""
    logger.info(f"Getting file content for: {relative_path} in {target_directory}")
    target_path = Path(target_directory).resolve()

    try:
        # Resolve the potential file path *before* checking directory existence
        # This helps prevent race conditions where the dir exists but the path escapes it.
        potential_file_path = (target_path / relative_path).resolve()

        # Ensure target directory exists *after* resolving potential path to avoid errors
        if not target_path.is_dir():
            raise FileNotFoundError(f"Target project directory not found: {target_directory}")

        # Security Check: Ensure the resolved path is strictly within the target directory
        # Use Path.is_relative_to() available in Python 3.9+
        # For older Python, a more manual check might be needed.
        if not potential_file_path.is_relative_to(target_path):
             logger.error(f"Security Alert: Path traversal attempt blocked. Requested: '{relative_path}', Resolved outside target: '{potential_file_path}', Target: '{target_path}'")
             raise ToolExecutionError(f"Access denied: Path '{relative_path}' points outside the project directory.")

        # Now that we know it's safe, assign the validated path
        file_path = potential_file_path

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found at resolved path: {file_path}")

        # Read as bytes to avoid encoding issues initially
        with open(file_path, "rb") as f:
            content_bytes = f.read()

        # Attempt to decode as UTF-8, fallback to lossy representation for safety
        try:
            content_str = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(f"Could not decode file {file_path} as UTF-8. Returning lossy representation.")
            content_str = content_bytes.decode("utf-8", errors="replace") # Replace invalid chars

        logger.info(f"Successfully read file content from {file_path}")
        return {"status": "success", "content": content_str}

    except FileNotFoundError as e:
        logger.error(f"Error getting file: {e}", exc_info=False)
        raise ToolExecutionError(f"Could not find the requested file: {relative_path} (resolved path: {potential_file_path})") from e
    except (OSError, IOError) as e:
        logger.error(f"Filesystem error reading file {potential_file_path}: {e}", exc_info=True)
        raise ToolExecutionError(f"Could not read file {relative_path}: {e}") from e
    except ToolExecutionError: # Re-raise the security/validation error
        raise
    except Exception as e: # Catch any other unexpected error
        logger.error(f"Unexpected error getting file {relative_path} in {target_directory}: {e}", exc_info=True)
        raise ToolExecutionError(f"An unexpected error occurred while reading the file: {e}") from e


@mcp.tool(
    name="retrieve_reflections",
    description="Retrieves relevant reflections from the project's ChromaDB.",
)
async def handle_retrieve_reflections_tool(
    query: str,
    n_results: int = 3,
    filter_stage_min: str | None = None,
    target_directory: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> dict:
    logger.info(
        f"Received 'retrieve_reflections' request for query: '{query}' (explicit target: '{target_directory}')"
    )
    effective_target_directory = _get_target_directory(target_directory, ctx)
    if not effective_target_directory:
        return {
            "status": "error",
            "message": "Could not determine project directory.",
            "reflections": [],
        }

    logger.info(f"Retrieving reflections for effective target: '{effective_target_directory}'")
    try:
        # Initialize StateManager within the handler
        context_state_manager = _initialize_state_manager_for_target(effective_target_directory)

        # Convert filter_stage_min to float if provided
        stage_filter_float: Optional[float] = None
        if filter_stage_min:
            try:
                stage_filter_float = float(filter_stage_min)
            except ValueError:
                logger.warning(f"Invalid filter_stage_min '{filter_stage_min}', expected number. Ignoring filter.")

        # --- Wrap ChromaDB call in try...except --- #
        results = None
        try:
            results = context_state_manager.get_reflection_context_from_chroma(
                query=query,
                n_results=n_results,
                filter_stage_min=stage_filter_float, # Pass converted float
            )
        except ChromaOperationError as chroma_e:
            logger.error(f"ChromaDB operation failed while retrieving reflections: {chroma_e}")
            return {
                "status": "error",
                "message": f"Failed to retrieve reflections from database: {chroma_e}",
                "reflections": [],
            }
        # --- End wrap ChromaDB call --- #

        # --- Processing (only if results were retrieved) --- #
        reflections_list = []
        if results:
            # ... (Existing processing logic remains the same) ...
            # Structure the results for the client
            # Assuming results is a list of tuples/dicts like: [{'document': text, 'metadata': {...}}, ...]
            for item in results:
                # Adapt based on actual structure returned by get_reflection_context_from_chroma
                doc = item.get("document", "No document found")
                meta = item.get("metadata", {})
                reflections_list.append({"metadata": meta, "document": doc})
        # --- End Processing --- #

        logger.info(
            f"Successfully retrieved {len(reflections_list)} reflections using ChromaDB query."
        )
        return {"status": "success", "reflections": reflections_list}

    # <<< Catch StateManager init errors specifically >>>
    except (StatusFileError, ValueError, FileNotFoundError) as sm_init_error:
        logger.error(
            f"StateManager initialization failed retrieving reflections for target '{effective_target_directory}': {sm_init_error}"
        )
        return {
            "status": "error",
            "message": f"Failed to initialize project state for reflection retrieval: {sm_init_error}",
            "reflections": [],
        }
    except Exception as e:
        logger.exception(
            f"Error retrieving reflections via ChromaDB for target '{effective_target_directory}': {e}"
        )
        return {
            "status": "error",
            "message": f"An unexpected error occurred while retrieving reflections: {e}",
            "reflections": [],
        }


# --- Log Registered Tools (DEBUG) ---
try:
    logger.info("--- Inspecting Registered MCP Tools ---")
    registered_tools = getattr(mcp, "tools", None)
    if registered_tools and isinstance(registered_tools, dict):
        logger.info(f"mcp.tools type: {type(registered_tools)}")
        for name, tool_info in registered_tools.items():
            # Assuming tool_info might be an object with a 'handler' attribute or similar
            handler = getattr(tool_info, "handler", "N/A")
            logger.info(f"  Tool: '{name}' -> Handler: {handler} (Info: {tool_info})")
    else:
        logger.warning("Could not access or inspect mcp.tools registry.")
    logger.info("-------------------------------------")
except Exception as inspect_err:
    logger.error(f"Error inspecting mcp.tools: {inspect_err}")
# --- End Log Registered Tools ---

# --- Main Execution Block ---

# Run the MCP server using the stdio transport
if __name__ == "__main__":
    logger.info(
        "Starting Chungoid MCP Server via STDIO (Restored Full Script / Renamed Tool Mode)..."
    )  # Updated message
    mcp.run(transport="stdio")
    logger.info("Chungoid MCP Server finished.")
