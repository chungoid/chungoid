#!/bin/bash

# Get the directory where the script resides
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the path to the virtual environment relative to the script
VENV_PATH="$SCRIPT_DIR/.venv"

# Activate the virtual environment
echo "Activating venv: $VENV_PATH/bin/activate" >&2 # Log to stderr
source "$VENV_PATH/bin/activate"

# Check if activation succeeded (optional but good practice)
if ! command -v python &> /dev/null || [[ "$(which python)" != "$VENV_PATH/bin/python" ]]; then
    echo "ERROR: Failed to activate virtual environment at $VENV_PATH" >&2
    exit 1
fi

# Define the path to the python script relative to the script dir
PYTHON_SCRIPT="$SCRIPT_DIR/chungoidmcp.py"

# Echo the command being run to stderr for debugging
echo "Executing: python $PYTHON_SCRIPT $@" >&2

# Execute the python script, passing along any arguments received by this wrapper
python "$PYTHON_SCRIPT" "$@"

# Deactivate venv (optional, happens on script exit anyway)
# deactivate 