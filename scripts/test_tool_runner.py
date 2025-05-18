import asyncio
import sys
import os
from pathlib import Path
import logging
from pprint import pprint
import json
import importlib

# The helper script previously inserted <repo>/chungoid-core on sys.path.  Now
# that the core package must be installed in editable mode (`pip install -e
# ./chungoid-core`), we can import it directly.  The fallback below keeps the
# script usable if the user forgets to install – it prints a clear message and
# exits instead of failing with an opaque ImportError.

# --- Imports from chungoid-core (installed) ---
try:
    # Import the module that contains the handlers
    chungoidmcp_module = importlib.import_module("chungoidmcp")
    from chungoid.utils.logger_setup import setup_logging  # pylint: disable=import-error
except ImportError as e:
    print("\n[ERROR] Could not import 'chungoid-core'.\n", file=sys.stderr)
    print("• Ensure you ran 'pip install -e ./chungoid-core[test]' as per ONBOARDING.md", file=sys.stderr)
    print("• Then re-activate your virtualenv or restart your shell.\n", file=sys.stderr)
    sys.exit(1)

# --- Configure Logging (Mimic Server Setup) ---
setup_logging()
logger = logging.getLogger("test_tool_runner")
logger.setLevel(logging.INFO)
print("Logging configured.")

# --- Tool Execution Logic ---
async def main(tool_name_arg: str, tool_args_json: str):
    # Define the target directory for the tool call
    target_dir = "." # Default target is current dir when running script
    absolute_target_dir = str(Path(target_dir).resolve())

    # Find the handler function dynamically
    try:
        # Construct expected handler name (e.g., get_project_status -> handle_get_project_status)
        handler_name = f"handle_{tool_name_arg}"
        tool_to_test = getattr(chungoidmcp_module, handler_name)
    except AttributeError:
        logger.error(f"Could not find handler function '{handler_name}' in chungoidmcp module.")
        sys.exit(1)

    logger.info(f"Testing tool handler: {handler_name}")

    # Parse arguments from JSON string
    try:
        tool_args = json.loads(tool_args_json)
        if not isinstance(tool_args, dict):
            raise ValueError("Arguments JSON must decode to a dictionary.")
        logger.info(f"Parsed arguments: {tool_args}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON arguments string: {e}")
        logger.error(f"Input string: {tool_args_json}")
        sys.exit(1)
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    # Ensure the CHUNGOID_PROJECT_DIR env var is set, as _get_target_directory relies on it as fallback
    # Setting it here mimics the client environment (using the script's CWD as the project dir)
    os.environ['CHUNGOID_PROJECT_DIR'] = absolute_target_dir
    logger.info(f"Set temporary environment variable CHUNGOID_PROJECT_DIR={absolute_target_dir}")

    logger.warning("Ensuring development ChromaDB server (localhost:8000) is stopped might be needed for some tools!")

    try:
        # Call the tool handler function with parsed arguments
        # Pass ctx=None as we are not simulating a specific client session
        result = await tool_to_test(**tool_args, ctx=None)

        # Print the result nicely
        logger.info("Tool execution finished. Result:")
        pprint(result)

    except Exception as e:
        logger.exception(f"An error occurred during tool execution: {e}")

if __name__ == "__main__":
    # --- Argument Parsing --- #
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <tool_name> '<json_arguments_string>'")
        print("Example: python dev/scripts/test_tool_runner.py get_project_status '{}'")
        print("Example: python dev/scripts/test_tool_runner.py submit_stage_artifacts '{ \"stage_number\": 1, \"generated_artifacts\": {}, \"stage_result_status\": \"PASS\" }'")
        sys.exit(1)

    tool_name_arg = sys.argv[1]
    tool_args_json = sys.argv[2]
    # --- End Argument Parsing --- #

    # Ensure we are in the project root directory for consistency
    project_root = Path(__file__).parent.parent.parent
    expected_cwd = project_root
    actual_cwd = Path.cwd()
    if actual_cwd != expected_cwd:
        print(f"Warning: Script is being run from {actual_cwd}, not the expected project root {expected_cwd}. Relative paths might be affected.", file=sys.stderr)

    asyncio.run(main(tool_name_arg, tool_args_json)) 