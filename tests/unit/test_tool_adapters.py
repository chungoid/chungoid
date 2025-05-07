import pytest
import subprocess
from unittest.mock import patch, MagicMock

from chungoid.utils.tool_adapters import execute_tool, ToolExecutionError, EXAMPLE_TOOL_DEFINITIONS

# Tests for ToolExecutionError
def test_tool_execution_error_str_representation():
    error_no_std = ToolExecutionError("Basic error")
    assert str(error_no_std) == "Basic error"

    error_with_stderr = ToolExecutionError("Error with stderr", stderr="some error output")
    assert str(error_with_stderr) == "Error with stderr\nStderr: some error output"

    error_with_stdout = ToolExecutionError("Error with stdout", stdout="some std output")
    assert str(error_with_stdout) == "Error with stdout\nStdout: some std output"

    error_with_all = ToolExecutionError("Error with all", stdout="out", stderr="err")
    assert str(error_with_all) == "Error with all\nStderr: err\nStdout: out"


# Tests for execute_tool
def test_execute_tool_success():
    tool_name = "echo_test"
    tool_def = {"command_template": "echo {message}", "timeout": 5}
    arguments = {"message": "Hello Adapter"}
    
    result = execute_tool(tool_name, tool_def, arguments)
    assert result == "Hello Adapter"

def test_execute_tool_missing_command_template():
    tool_name = "no_template_tool"
    tool_def = {"timeout": 5} # Missing command_template
    arguments = {}
    
    with pytest.raises(ToolExecutionError, match=f"Tool '{tool_name}' definition is missing 'command_template'."):
        execute_tool(tool_name, tool_def, arguments)

def test_execute_tool_missing_argument_for_template():
    tool_name = "missing_arg_tool"
    tool_def = {"command_template": "echo {message}", "timeout": 5}
    arguments = {} # Missing 'message' argument
    
    with pytest.raises(ToolExecutionError, match=f"Missing argument ''message'' required by command template for {tool_name}"):
        execute_tool(tool_name, tool_def, arguments)

@patch('subprocess.run')
def test_execute_tool_command_not_found(mock_subprocess_run):
    mock_subprocess_run.side_effect = FileNotFoundError("[Errno 2] No such file or directory: 'non_existent_cmd'")
    
    tool_name = "not_found_cmd_tool"
    # Use a simple template that shlex.split can process
    tool_def = {"command_template": "non_existent_cmd arg1", "timeout": 5}
    arguments = {} # No arguments needed for this specific template format

    with pytest.raises(ToolExecutionError, match="Command not found: non_existent_cmd"):
        execute_tool(tool_name, tool_def, arguments)

@patch('subprocess.run')
def test_execute_tool_command_timeout(mock_subprocess_run):
    # Configure the mock to simulate TimeoutExpired
    # The TimeoutExpired exception needs stdout and stderr attributes, which can be bytes
    mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="timeout_cmd", timeout=5, output=b"partial_out", stderr=b"partial_err")

    tool_name = "timeout_cmd_tool"
    tool_def = {"command_template": "timeout_cmd", "timeout": 5}
    arguments = {}

    with pytest.raises(ToolExecutionError, match=f"Tool '{tool_name}' timed out.") as excinfo:
        execute_tool(tool_name, tool_def, arguments)
    # Check if stdout and stderr from the exception are included in the ToolExecutionError instance
    assert excinfo.value.stdout == b"partial_out"
    assert excinfo.value.stderr == b"partial_err"


@patch('subprocess.run')
def test_execute_tool_command_non_zero_exit(mock_subprocess_run):
    # Configure the mock to simulate a failed command
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = "some output"
    mock_process.stderr = "some error"
    mock_subprocess_run.return_value = mock_process
    
    tool_name = "fail_cmd_tool"
    tool_def = {"command_template": "failing_command", "timeout": 5}
    arguments = {}
    
    with pytest.raises(ToolExecutionError, match=f"Tool '{tool_name}' failed with exit code 1.") as excinfo:
        execute_tool(tool_name, tool_def, arguments)
    assert excinfo.value.stdout == "some output"
    assert excinfo.value.stderr == "some error"

def test_execute_tool_empty_formatted_command():
    tool_name = "empty_cmd_tool"
    tool_def = {"command_template": "{placeholder}", "timeout": 5} # Template expects placeholder
    arguments = {"placeholder": "      "} # Argument that results in an empty command after shlex.split
    
    # This tests the scenario where formatted_command might be non-empty
    # but shlex.split(formatted_command) results in an empty list.
    # e.g. if formatted_command was just whitespace.
    
    with pytest.raises(ToolExecutionError, match=f"Formatted command for '{tool_name}' resulted in empty parts."):
        execute_tool(tool_name, tool_def, arguments)

def test_execute_tool_with_example_definitions_list_directory():
    # This test uses one of the EXAMPLE_TOOL_DEFINITIONS
    # It requires a live 'ls' command to work.
    tool_name = "list_directory"
    tool_def = EXAMPLE_TOOL_DEFINITIONS[tool_name]
    arguments = {"directory_path": "."} # Test with current directory
    
    # We expect this to succeed and return a list of files/dirs
    result = execute_tool(tool_name, tool_def, arguments)
    assert isinstance(result, str)
    assert "." in result # Current directory listing should contain '.'

def test_execute_tool_with_example_definitions_show_file_not_found():
    # This test uses 'cat' which should fail if file not found
    tool_name = "show_file"
    tool_def = EXAMPLE_TOOL_DEFINITIONS[tool_name]
    arguments = {"file_path": "a_very_unlikely_file_to_exist_in_test_env.xyz"}
    
    with pytest.raises(ToolExecutionError) as excinfo:
        execute_tool(tool_name, tool_def, arguments)
    assert f"Tool '{tool_name}' failed with exit code" in str(excinfo.value) # Cat usually exits with 1
    assert "No such file or directory" in excinfo.value.stderr # Check stderr from cat 