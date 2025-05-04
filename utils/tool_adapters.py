"""Adapter functions for executing external tools and MCP commands."""

import logging
import subprocess
import shlex
from typing import Dict, Any, Optional


# Custom Exception for Tool Execution Errors
class ToolExecutionError(Exception):
    """Custom exception for errors encountered during tool execution."""

    def __init__(self, message: str, stdout: Optional[str] = None, stderr: Optional[str] = None):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        msg = super().__str__()
        if self.stderr:
            msg += f"\nStderr: {self.stderr}"
        if self.stdout:
            msg += f"\nStdout: {self.stdout}"
        return msg


logger = logging.getLogger(__name__)

# --- Core Tool Execution Logic ---


def execute_tool(tool_name: str, tool_def: Dict[str, Any], arguments: Dict[str, Any]) -> str:
    """Executes a defined tool (CLI command) with provided arguments.

    Args:
        tool_name: The name of the tool being executed.
        tool_def: The tool definition dictionary (containing 'command_template', etc.).
        arguments: A dictionary of arguments provided for the tool execution.

    Returns:
        The standard output of the executed command as a string.

    Raises:
        ToolExecutionError: If the tool definition is invalid, arguments are missing,
                            the command fails, times out, or produces errors.
    """
    logger.info("Executing tool: %s with args: %s", tool_name, arguments)

    command_template = tool_def.get("command_template")
    if not command_template:
        raise ToolExecutionError(f"Tool '{tool_name}' definition is missing 'command_template'.")

    # Validate arguments based on tool_def (optional but recommended)
    # TODO: Implement validation based on tool_def['arguments'] if schema is available.

    # Format the command string safely
    try:
        # Using f-string formatting, ensure arguments are sanitized if coming from untrusted sources
        # For simple CLI tools where args are controlled, direct formatting might be okay.
        # Consider more robust templating if complex/untrusted inputs are possible.
        formatted_command = command_template.format(**arguments)
    except KeyError as e:
        logger.error("Missing argument '%s' required by command template for %s", e, tool_name)
        # Raise with specific error message, chain original exception
        raise ToolExecutionError(
            f"Missing argument '{e}' required by command template for {tool_name}"
        ) from e
    except Exception as e:
        logger.exception("Error formatting command for %s: %s", tool_name, e)
        # Raise with specific error message, chain original exception
        raise ToolExecutionError(f"Error formatting command for {tool_name}: {e}") from e

    logger.debug("Formatted command: %s", formatted_command)

    # Execute the command using subprocess
    try:
        # shlex.split helps handle quoted arguments correctly
        cmd_parts = shlex.split(formatted_command)
        if not cmd_parts:
            raise ToolExecutionError(
                f"Formatted command for '{tool_name}' resulted in empty parts."
            )

        # Execute the command
        timeout_seconds = tool_def.get("timeout", 60)  # Default timeout 60 seconds
        process = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            check=False,  # Don't raise CalledProcessError automatically
            timeout=timeout_seconds,
            encoding="utf-8",  # Explicitly set encoding
        )

        # Check return code
        if process.returncode != 0:
            logger.error(
                "Tool '%s' failed with return code %d. Stderr: %s",
                tool_name,
                process.returncode,
                process.stderr.strip(),
            )
            raise ToolExecutionError(
                f"Tool '{tool_name}' failed with exit code {process.returncode}.",
                stdout=process.stdout.strip(),
                stderr=process.stderr.strip(),
            )

        # Success
        logger.info("Tool '%s' executed successfully.", tool_name)
        logger.debug("Tool '%s' stdout: %s", tool_name, process.stdout.strip())
        return process.stdout.strip()

    except FileNotFoundError as exc:
        # Handle case where the command itself is not found
        cmd_name = cmd_parts[0] if cmd_parts else "command"
        logger.error(
            "Command '%s' not found for tool '%s'. Ensure it's in PATH.",
            cmd_name,
            tool_name,
        )
        raise ToolExecutionError(f"Command not found: {cmd_name}") from exc
    except subprocess.TimeoutExpired as e:
        logger.error("Tool '%s' timed out after %s seconds.", tool_name, timeout_seconds)
        raise ToolExecutionError(
            f"Tool '{tool_name}' timed out.", stdout=e.stdout, stderr=e.stderr
        ) from e
    except Exception as e:
        logger.exception("Unexpected error executing tool '%s': %s", tool_name, e)
        raise ToolExecutionError(f"Unexpected error executing tool '{tool_name}': {e}") from e


# --- Placeholder/Example MCP Tool Adapters ---
# These would interact with an MCP client library if this module
# needed to CALL other MCP tools.


async def call_mcp_tool(
    mcp_client: Any, tool_name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Placeholder for calling another MCP tool via a client instance.

    Args:
        mcp_client: An initialized MCP client object (e.g., from fastmcp).
        tool_name: The name of the MCP tool to call.
        arguments: The arguments for the MCP tool.

    Returns:
        The result dictionary from the MCP tool call.

    Raises:
        ToolExecutionError: If the MCP call fails.
    """
    logger.info("Calling MCP tool: %s with args: %s", tool_name, arguments)
    if not hasattr(mcp_client, "call_tool"):
        raise ToolExecutionError("Provided mcp_client does not have a 'call_tool' method.")

    try:
        # Assuming the client has an async call_tool method
        result = await mcp_client.call_tool(tool_name, **arguments)
        logger.info("MCP tool '%s' call successful.", tool_name)
        logger.debug("MCP tool '%s' result: %s", tool_name, result)
        return result
    except Exception as e:
        logger.exception("Failed to call MCP tool '%s': %s", tool_name, e)
        # More specific error handling based on the MCP client library would be better
        raise ToolExecutionError(f"Failed to execute MCP tool '{tool_name}': {e}") from e


# --- Example Tool Definition Loading ---
# In a real application, this might load from a JSON/YAML file
# defined in dev-docs/tools/tools_schema.json

EXAMPLE_TOOL_DEFINITIONS = {
    "list_directory": {
        "description": "Lists the contents of a specified directory.",
        "command_template": "ls -1a {directory_path}",  # Example template
        "arguments": {
            "directory_path": {
                "type": "string",
                "required": True,
                "description": "Path of the directory to list.",
            }
        },
        "timeout": 10,  # Seconds
    },
    "show_file": {
        "description": "Displays the content of a specified file.",
        "command_template": "cat {file_path}",  # Example template
        "arguments": {
            "file_path": {
                "type": "string",
                "required": True,
                "description": "Path of the file to display.",
            }
        },
        "timeout": 5,
    },
    # Add more tool definitions here
}

# --- Main Execution / Test ---

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("--- Testing Tool Adapters --- ")

    # Test listing current directory
    try:
        logger.info("Testing 'list_directory' on '.'")
        output = execute_tool(
            "list_directory",
            EXAMPLE_TOOL_DEFINITIONS["list_directory"],
            {"directory_path": "."},
        )
        logger.info("'list_directory' successful. Output:\n%s", output)
    except ToolExecutionError as e:
        logger.error("'list_directory' failed: %s", e)

    logger.info("-" * 20)

    # Test showing a file (e.g., requirements.txt)
    test_file = "requirements.txt"
    try:
        logger.info("Testing 'show_file' on '%s'", test_file)
        output = execute_tool(
            "show_file", EXAMPLE_TOOL_DEFINITIONS["show_file"], {"file_path": test_file}
        )
        logger.info("'show_file' successful. Output:\n%s", output)
    except ToolExecutionError as e:
        logger.error("'show_file' failed: %s", e)

    logger.info("-" * 20)

    # Test non-existent command
    try:
        logger.info("Testing non-existent command")
        bad_tool_def = {"command_template": "non_existent_command_123", "timeout": 2}
        execute_tool("bad_command", bad_tool_def, {})
    except ToolExecutionError as e:
        logger.warning("Caught expected error for non-existent command: %s", e)

    logger.info("-" * 20)

    # Test command failure (e.g., cat non-existent file)
    try:
        logger.info("Testing command failure (cat non-existent file)")
        execute_tool(
            "show_file",
            EXAMPLE_TOOL_DEFINITIONS["show_file"],
            {"file_path": "./non_existent_file_xyz.abc"},
        )
    except ToolExecutionError as e:
        logger.warning("Caught expected error for command failure: %s", e)

    logger.info("--- Tool Adapter Tests Complete --- ")
