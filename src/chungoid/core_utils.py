from __future__ import annotations

import json
from pathlib import Path
import logging

from chungoid.constants import PROJECT_CHUNGOID_DIR # Assuming this will be needed

logger = logging.getLogger(__name__)

def init_project_structure(project_path: Path) -> None:
    """
    Initializes the basic Chungoid project structure at the given path.
    Creates the .chungoid directory and a default project_status.json file.
    """
    logger.info(f"Initializing Chungoid project structure at: {project_path}")

    if project_path.exists() and any(project_path.iterdir()) and not (project_path / PROJECT_CHUNGOID_DIR).exists():
        # If directory exists, is not empty, but .chungoid doesn't exist, we might be in a pre-existing project.
        # The CLI's click.confirm handles the user prompt for non-empty dirs.
        # Here, we just ensure the base path exists if we are to create .chungoid inside it.
        pass # Handled by CLI user confirmation or direct call where appropriate
    
    project_path.mkdir(parents=True, exist_ok=True)

    chungoid_dir = project_path / PROJECT_CHUNGOID_DIR
    chungoid_dir.mkdir(exist_ok=True)
    logger.debug(f"Ensured {PROJECT_CHUNGOID_DIR} directory exists at {chungoid_dir}")

    # Default status file, matching what was in cli.py's init command
    # This might be better handled by StateManager's initialization in a more mature setup.
    status_file = chungoid_dir / "project_status.json" # Consider using STATE_FILE_NAME from constants
    if not status_file.exists():
        status_file.write_text(json.dumps({"runs": []}, indent=2))
        logger.debug(f"Created fresh status file at {status_file}")
    
    # Create default master_flows and server_stages directories inside .chungoid
    # These are read by MasterFlowRegistry and StateManager respectively
    # (DEFAULT_MASTER_FLOWS_DIR and DEFAULT_SERVER_STAGES_DIR are relative paths from constants)
    
    # from chungoid.constants import DEFAULT_MASTER_FLOWS_DIR, DEFAULT_SERVER_STAGES_DIR
    # (master_flows_dir_path = chungoid_dir / DEFAULT_MASTER_FLOWS_DIR).mkdir(exist_ok=True)
    # (server_stages_dir_path = chungoid_dir / DEFAULT_SERVER_STAGES_DIR).mkdir(exist_ok=True)
    # logger.debug(f"Ensured default directories '{DEFAULT_MASTER_FLOWS_DIR}' and '{DEFAULT_SERVER_STAGES_DIR}' exist in {chungoid_dir}")
    # Commenting out the creation of master_flows and server_stages as they might not be universally needed by init_project_structure
    # and could be the responsibility of specific components like registries.

    logger.info(f"Chungoid project structure initialized at {project_path}")


def get_project_root_or_raise(start_path: Path) -> Path:
    """
    Searches upward from start_path to find a Chungoid project root.
    A project root is identified by the presence of a '.chungoid' directory
    or a 'pyproject.toml' file.
    Raises FileNotFoundError if no project root is found.
    """
    current_path = Path(start_path).resolve()
    while True:
        if (current_path / PROJECT_CHUNGOID_DIR).is_dir():
            logger.debug(f"Found project root at {current_path} (marker: {PROJECT_CHUNGOID_DIR})")
            return current_path
        if (current_path / "pyproject.toml").is_file(): # A common Python project marker
            logger.debug(f"Found project root at {current_path} (marker: pyproject.toml)")
            return current_path
        
        parent = current_path.parent
        if parent == current_path: # Reached filesystem root
            break
        current_path = parent
            
    raise FileNotFoundError(
        f"Could not find Chungoid project root. Searched upwards from '{start_path}'. "
        f"Looked for '{PROJECT_CHUNGOID_DIR}/' or 'pyproject.toml'."
    ) 