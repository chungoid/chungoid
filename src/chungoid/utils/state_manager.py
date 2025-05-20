"""Manages the project status file (project_status.json) with file locking."""

import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, cast
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from logging.handlers import RotatingFileHandler
import hashlib
import uuid
import contextlib

import filelock  # Ensure filelock is installed
from . import chroma_utils
from ..schemas.errors import AgentErrorDetails # Correct relative import
from ..schemas.flows import PausedRunDetails # <<< Import PausedRunDetails
from ..schemas.common_enums import StageStatus # ADDED
import json # Add for JSONDecodeError
from pydantic import ValidationError # Add for Pydantic validation error

# NEW IMPORTS for ProjectStateV2
from ..schemas.project_state import ProjectStateV2, CycleHistoryItem, RunRecord, StageRecord # ENSURE RunRecord and StageRecord are also imported if they come from here
from ..constants import STATE_FILE_NAME # Using constant for state file name
from ..schemas.project_status_schema import ProjectOverallStatus # ADDED ProjectOverallStatus import

# from chungoid.schemas.common_enums import FileStatus, ProjectStatus # Removed unused
from chungoid.schemas.master_flow import MasterExecutionPlan # Added import
# from chungoid.schemas.file_schemas import FileProcessResult, StatusEntry # Removed unused
# from .file_ops import find_project_root, load_json_from_file, write_json_to_file # Changed to relative import - NOW REMOVING AS UNUSED

from pydantic import BaseModel, Field

# from chungoid.schemas.orchestration import PausedRunDetails # Not directly used here, but for context
# from chungoid.schemas.agent_errors import AgentErrorDetails # Not directly used, context

from chungoid.schemas.chroma_agent_io_schemas import StoreArtifactInput, RetrieveArtifactOutput # Import from new location
# REMOVED ProjectStateV2 from this import, kept others
from chungoid.schemas.project_status_schema import CycleInfo, CycleStatus, ProjectOverallStatus, ArtifactLink, KeyDecision, HumanReviewRecord


class StatusFileError(Exception):
    """Custom exception for errors related to status file operations."""

    pass


class ChromaOperationError(Exception):
    """Custom exception for errors during ChromaDB operations within StateManager."""

    pass


class DummyFileLock:
    """A dummy lock object that does nothing, for testing or when locking is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def acquire(self, timeout=None):
        pass

    def release(self):
        pass


class StateManager:
    """Handles reading, updating, and validating the project status file.

    Uses file locking to prevent race conditions during updates.
    """

    _CONTEXT_COLLECTION_NAME = "chungoid_context"
    _REFLECTIONS_COLLECTION_NAME = "chungoid_reflections"
    _project_state: ProjectStateV2 # Type hint for the main state object
    _current_cycle_info: Optional[CycleInfo] = None # To hold details of the cycle currently in progress

    def __init__(self, target_directory: str, server_stages_dir: str, use_locking: bool = True):
        """Initializes the StateManager.

        Args:
            target_directory: The root directory of the Chungoid project.
            server_stages_dir: The path to the directory containing stage*.yaml files.
            use_locking: Whether to use file locking (default: True).
        """
        # <<< Add Logging Point >>>
        logging.getLogger(__name__).debug(f"StateManager.__init__ called for target: {target_directory}")
        # <<< End Logging Point >>>

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # <<< START ADDED CODE >>>
        # Attempt to add root file handler directly to this logger
        try:
            root_logger = logging.getLogger()
            file_handler = None
            for handler in root_logger.handlers:
                if isinstance(handler, RotatingFileHandler):
                    file_handler = handler
                    break
            if file_handler:
                if file_handler not in self.logger.handlers:
                    self.logger.addHandler(file_handler)
                    self.logger.info(
                        f"Successfully added root RotatingFileHandler to {__name__} logger."
                    )
                else:
                    self.logger.debug(
                        f"Root RotatingFileHandler already present on {__name__} logger."
                    )
            else:
                self.logger.warning(
                    f"Could not find RotatingFileHandler on root logger to add to {__name__} logger."
                )
            # <<< START ADDED LOGGING >>>
            self.logger.info(f"Checking {__name__} logger propagation: {self.logger.propagate}")
            self.logger.info(f"Checking {__name__} logger handlers: {self.logger.handlers}")
            # <<< END ADDED LOGGING >>>
        except Exception as log_add_err:
            self.logger.error(
                f"Error trying to add root file handler to {__name__} logger: {log_add_err}"
            )
        # <<< END ADDED CODE >>>
        self.target_dir_path = Path(target_directory).resolve()
        self.server_stages_dir = Path(server_stages_dir).resolve()

        # Directory existence checks
        if not self.target_dir_path.is_dir():
            self.logger.error(f"Target project directory not found or not a directory: {self.target_dir_path}")
            raise ValueError(f"Target project directory not found or not a directory: {self.target_dir_path}")
        if not self.server_stages_dir.is_dir():
             self.logger.warning(f"Server stages directory not found or not a directory: {self.server_stages_dir}. Operations requiring it may fail.")
             # Do not raise ValueError here, allow initialization to continue.

        self.chungoid_dir = self.target_dir_path / ".chungoid"
        # Use STATE_FILE_NAME from constants
        self.status_file_path: Path = self.chungoid_dir / STATE_FILE_NAME
        self.lock_file_path = f"{str(self.status_file_path)}.lock"
        self.use_locking = use_locking

        # --- Initialize Lock EARLY --- 
        if self.use_locking:
            self._lock = filelock.FileLock(self.lock_file_path)
            self.logger.debug("Using real file lock: %s", self.lock_file_path)
        else:
            self._lock = DummyFileLock()
            self.logger.debug("File locking is disabled.")
        # --- End Lock Init ---

        # --- Set ChromaDB Project Context ---
        # This must be called before the first call to get_chroma_client() in this context
        chroma_utils.set_chroma_project_context(self.target_dir_path)
        # --- End Set Context ---

        self.chroma_client: Optional[chromadb.ClientAPI] = (
            None  # Keep Optional for type hinting if init fails
        )
        try:
            self.logger.info("Attempting synchronous ChromaDB client initialization...")
            # Now get_chroma_client() can use the context we just set
            self.chroma_client = chroma_utils.get_chroma_client()
            # --- Modified Logging --- #
            if self.chroma_client:
                # Attempt a simple operation to confirm client validity (optional)
                try:
                    # List collections as a basic check that the client is operational
                    self.chroma_client.list_collections()
                    self.logger.info("ChromaDB client initialization successful and confirmed operational.")
                except Exception as conn_err:
                    self.logger.error(f"ChromaDB client initialized but failed basic operation check: {conn_err}", exc_info=True)
                    # Optionally set client back to None if basic check fails?
                    # self.chroma_client = None
            else:
                # This case means get_chroma_client() returned None
                self.logger.error("ChromaDB client initialization failed (get_chroma_client returned None). Check chroma_utils logs for details.")
            # --- End Modified Logging --- #
        except Exception as e:
            # Catch errors during the call to get_chroma_client itself
            self.logger.error(
                f"Exception during ChromaDB client retrieval: {e}", exc_info=True
            )
            self.chroma_client = None  # Ensure it's None on error

        # Ensure the .chungoid directory exists
        try:
            # This should only be called AFTER confirming target_dir_path exists and is a dir
            self.chungoid_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug("Ensured .chungoid directory exists: %s", self.chungoid_dir)
        except OSError as e:
            self.logger.error(
                "Failed to create .chungoid directory at %s: %s", self.chungoid_dir, e
            )
            raise StatusFileError(f"Failed to create .chungoid directory: {e}") from e

        # Initial read of the status file.
        # _load_or_initialize_project_state will set self._project_state
        # either by reading an existing file or creating an in-memory placeholder.
        # It handles logging and potential errors during this process.
        self._load_or_initialize_project_state()
        self.logger.info("StateManager initialized for file: %s", self.status_file_path)

        # The problematic try-except block that was here has been removed.
        # _load_or_initialize_project_state is now solely responsible for
        # setting the initial self._project_state.

        self._current_cycle_info = None # Explicitly initialize in __init__

    def _create_default_project_state_v2(self, project_id: str, project_name: Optional[str] = None, initial_user_goal_summary: str = "Default goal summary") -> ProjectStateV2:
        """Creates a default ProjectStateV2 object for a new project."""
        self.logger.info(f"Creating new ProjectStateV2 with project_id: {project_id}")
        now = datetime.now(timezone.utc)
        return ProjectStateV2(
            project_id=project_id,
            project_name=project_name,
            initial_user_goal_summary=initial_user_goal_summary,
            overall_project_status=ProjectOverallStatus.NOT_STARTED.value,
            schema_version="2.0.0", 
            # created_at_utc=now, # REMOVED - field does not exist in target ProjectStateV2
            last_updated=now, # CHANGED from last_updated_utc
            run_history={}
        )

    def _load_or_initialize_project_state(self) -> None:
        """Loads project state from file or initializes a new one if not found or invalid."""
        self.logger.info(f"UNIQUE_LOG_MARKER_V3: Entering _load_or_initialize_project_state")
        loaded_successfully = False
        if self.status_file_path.exists() and self.status_file_path.stat().st_size > 0:
            try:
                with open(self.status_file_path, "r") as f:
                    data = json.load(f)
                
                # Ensure 'run_history' key exists in the loaded data before creating the model instance.
                # This is crucial because if it's missing, Pydantic V2 might not consider it a valid field
                # for direct assignment later, even if the class schema has it.
                if 'run_history' not in data or data['run_history'] is None:
                    self.logger.warning(
                        f"Loaded data from {self.status_file_path} is missing 'run_history' or it is None. "
                        "Adding an empty 'run_history': {} before Pydantic validation."
                    )
                    data['run_history'] = {}

                # Attempt to parse with ProjectStateV2 using the (potentially modified) data dict
                self._project_state = ProjectStateV2(**data)
                self.logger.info(f"Successfully loaded and validated project state for {self._project_state.project_id} from {self.status_file_path}")
                loaded_successfully = True

            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                self.logger.warning(
                    f"Failed to load or validate existing status file at {self.status_file_path} (Error: {e}). "
                    "Backing up and creating a new default state.",
                    exc_info=True
                )
                self._backup_existing_status_file()
                # new_project_id = f"proj_{uuid.uuid4().hex[:12]}" # Project ID should ideally be stable or re-used if backup has it
                # Try to get project ID from backup if possible, else generate new
                new_project_id = data.get("project_id", f"proj_{uuid.uuid4().hex[:12]}") if 'data' in locals() and isinstance(data, dict) else f"proj_{uuid.uuid4().hex[:12]}"
                project_name = data.get("project_name") if 'data' in locals() and isinstance(data, dict) else None

                self._project_state = self._create_default_project_state_v2(
                    project_id=new_project_id,
                    project_name=project_name
                )
                # Ensure run_history is present even in this newly created default state
                if not hasattr(self._project_state, 'run_history') or self._project_state.run_history is None:
                    self._project_state.run_history = {}
                self._write_status_file(self._project_state)
                loaded_successfully = True # Technically successful as a new valid state is now in memory and saved
        
        if not loaded_successfully: # If file didn't exist, was empty, or initial load failed and wasn't recovered above
            self.logger.info(f"No existing status file found at {self.status_file_path}, file was empty, or initial load failed. Creating a new default state.")
            # Generate a new project ID if one couldn't be salvaged from a failed load.
            # This path is usually for a completely fresh setup.
            new_project_id = f"proj_{uuid.uuid4().hex[:12]}"
            self._project_state = self._create_default_project_state_v2(project_id=new_project_id)
            # Ensure run_history is present in this brand new default state
            #if not hasattr(self._project_state, 'run_history') or self._project_state.run_history is None:
            #     self._project_state.run_history = {}
            self._write_status_file(self._project_state)

        # Final check to ensure _project_state itself is not None and has run_history
        if not hasattr(self, '_project_state') or not self._project_state:
            self.logger.critical(f"CRITICAL: _project_state is None after _load_or_initialize_project_state. Forcing placeholder.")
            self._project_state = self._create_placeholder_project_state() 
            if not hasattr(self._project_state, 'run_history') or self._project_state.run_history is None:
                self._project_state.run_history = {}
            self._write_status_file(self._project_state)
        elif not hasattr(self._project_state, 'run_history') or self._project_state.run_history is None:
            # This case means _project_state exists but somehow run_history is still not there (e.g. placeholder didn't get it)
            # This should be very rare if _create_default_project_state_v2 and _create_placeholder_project_state ensure it.
            self.logger.warning(
                f"Post-load/init: _project_state for {self._project_state.project_id} still missing 'run_history'. "
                "Attempting to forcefully re-create the model with run_history."
            )
            try:
                current_data = self._project_state.model_dump()
                current_data['run_history'] = {} # Ensure it's there
                self._project_state = ProjectStateV2(**current_data)
                self.logger.info(f"Forcefully re-created ProjectStateV2 for {self._project_state.project_id} to include run_history.")
                self._save_project_state() 
            except Exception as e_force_assign:
                self.logger.error(f"Failed to forcefully re-create _project_state with run_history: {e_force_assign}. ", exc_info=True)
                                  
        self.logger.info(f"StateManager._project_state initialized for project: {self._project_state.project_id}, Name: {self._project_state.project_name}")

    def _create_placeholder_project_state(self) -> ProjectStateV2:
        """Creates a minimal, temporary ProjectStateV2 object when actual state cannot be loaded."""
        return ProjectStateV2(
            project_id=f"placeholder_proj_{uuid.uuid4().hex[:8]}",
            initial_user_goal_summary="Project state could not be loaded or initialized.",
            overall_status=ProjectOverallStatus.PROJECT_FAILED # Indicate an issue
        )

    def _read_status_file(self) -> ProjectStateV2:
        """Reads and validates the status file, returning a ProjectStateV2 object."""
        if not self.status_file_path.exists():
            self.logger.error(f"Status file does not exist at {self.status_file_path} during read attempt.")
            raise StatusFileError(f"Status file not found: {self.status_file_path}")
        try:
            raw_content = self.status_file_path.read_text(encoding="utf-8")
            if not raw_content.strip(): # Handle empty file
                self.logger.warning(f"Status file is empty: {self.status_file_path}. Will attempt to re-initialize.")
                raise StatusFileError(f"Status file is empty: {self.status_file_path}")
            
            data = json.loads(raw_content)
            
            # Basic check for V2 schema before full parsing
            if data.get("schema_version", "").startswith("2."):
                 # Attempt to parse as ProjectStateV2
                return ProjectStateV2.model_validate(data)
            else:
                # Handle migration or error for older/unknown versions
                self.logger.warning(
                    f"Old or unknown schema version detected in {self.status_file_path}. "
                    f"Schema version: {data.get('schema_version', 'Not specified')}. "
                    "Attempting to create a new V2 state. Old data will be lost if not backed up."
                )
                # This will lead to re-initialization by the calling function if StatusFileError is raised
                raise StatusFileError("Old or incompatible schema version detected.")

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in status file {self.status_file_path}: {e}", exc_info=True)
            raise StatusFileError(f"Invalid JSON in status file: {e}") from e
        except ValidationError as e:
            self.logger.error(f"Schema validation error for ProjectStateV2 in {self.status_file_path}: {e}", exc_info=True)
            raise StatusFileError(f"ProjectStateV2 schema validation error: {e}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error reading status file {self.status_file_path}: {e}", exc_info=True)
            raise StatusFileError(f"Unexpected error reading status file: {e}") from e

    def _write_status_file(self, project_state: ProjectStateV2) -> None:
        """Writes the ProjectStateV2 object to the status file in JSON format."""
        try:
            project_state.last_updated = datetime.now(timezone.utc) # CHANGED from last_updated_utc
            # model_dump_json handles datetime serialization correctly by default with Pydantic v2
            json_data = project_state.model_dump_json(indent=2)
            self.status_file_path.write_text(json_data, encoding="utf-8")
            self.logger.debug(f"Successfully wrote ProjectStateV2 to {self.status_file_path}")
        except TypeError as e: # Catch issues if model_dump_json fails for some reason
            self.logger.error(f"Serialization error writing ProjectStateV2 to {self.status_file_path}: {e}", exc_info=True)
            raise StatusFileError(f"Serialization error: {e}") from e
        except Exception as e:
            self.logger.error(f"Failed to write status file {self.status_file_path}: {e}", exc_info=True)
            raise StatusFileError(f"Failed to write status file: {e}") from e

    # Custom JSON Encoder for datetime objects (if not using Pydantic's built-in serialization fully)
    # Pydantic v2 model_dump_json should handle datetime correctly.
    # This might be needed if we were manually calling json.dumps with non-Pydantic models.
    # For Pydantic ProjectStateV2, this custom handler in json.dumps is less critical.
    # Keeping it for now in case parts of the system still rely on manual json.dumps.
    def _datetime_serializer(self, obj: Any) -> Union[str, Any]:
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Let the default serializer handle other types
        # Or raise a TypeError if you want to be strict
        # raise TypeError(f"Type {type(obj)} not serializable") 
        return obj

    def get_project_state(self) -> ProjectStateV2:
        """Returns the current project state."""
        with self._get_lock(): # Ensure read is also consistent if needed, though usually for writes
            # Re-read from disk to ensure freshness if multiple StateManager instances could exist (unlikely for typical CLI use)
            # For now, assume self._project_state is the single source of truth in this instance.
            # If strict consistency across processes is needed, every access might need a read.
            # However, typical use involves one orchestrator with one StateManager.
            return self._project_state.model_copy(deep=True) # Return a copy to prevent external modification

    def _save_project_state(self) -> None:
        """Saves the current in-memory project state to the file with locking."""
        if not hasattr(self, '_project_state'):
            self.logger.error("Attempted to save project state, but StateManager is not properly initialized.")
            raise StatusFileError("StateManager not properly initialized for saving.")
        with self._get_lock():
            self._write_status_file(self._project_state)
            self.logger.debug(f"Project state for {self._project_state.project_id} saved successfully.")

    def start_new_cycle(self, cycle_objective: str) -> str:
        """
        Starts a new autonomous cycle.
        Updates project state with new cycle information and persists it.
        Returns the ID of the newly started cycle.
        """
        if not hasattr(self, '_project_state') or self._project_state.overall_status == ProjectOverallStatus.PROJECT_FAILED:
            self.logger.error("Cannot start new cycle: Project state not loaded or project failed.")
            raise StatusFileError("Project state not loaded or project failed, cannot start new cycle.")
        
        if self._current_cycle_info is not None and self._current_cycle_info.status == CycleStatus.IN_PROGRESS:
            self.logger.warning(f"Cannot start new cycle: Cycle {self._current_cycle_info.cycle_id} is already in progress.")
            # Or raise an error, depending on desired behavior. For now, let\'s prevent starting a new one.
            raise StatusFileError(f"A cycle ({self._current_cycle_info.cycle_id}) is already in progress.")

        with self._get_lock():
            self._project_state.current_cycle_number += 1
            new_cycle_id = f"cycle_{self._project_state.project_id}_{self._project_state.current_cycle_number}_{uuid.uuid4().hex[:8]}"
            self._project_state.current_cycle_id = new_cycle_id

            self._current_cycle_info = CycleInfo(
                cycle_id=new_cycle_id,
                cycle_number=self._project_state.current_cycle_number,
                cycle_objective=cycle_objective,
                status=CycleStatus.IN_PROGRESS,
                start_time_utc=datetime.now(timezone.utc)
                # cycle_produced_artifacts and key_decisions_in_cycle will be populated as the cycle progresses
            )
            
            if self._project_state.current_cycle_number == 1: # Assuming first user-facing cycle is 1
                 self._project_state.overall_status = ProjectOverallStatus.ANALYZING_GOAL # Or more specific initial phase
            else:
                self._project_state.overall_status = ProjectOverallStatus.REFINEMENT_CYCLE_IN_PROGRESS
            
            self.logger.info(f"Starting new cycle: {new_cycle_id} (Objective: {cycle_objective}) for project {self._project_state.project_id}")
            self._save_project_state()
            return new_cycle_id

    def add_artifact_to_current_cycle(self, artifact_link: ArtifactLink) -> None:
        """Adds a produced artifact to the current active cycle."""
        if self._current_cycle_info is None or self._current_cycle_info.status != CycleStatus.IN_PROGRESS:
            self.logger.error("Cannot add artifact: No cycle is currently in progress.")
            raise StatusFileError("No cycle in progress to add artifact to.")
        
        with self._get_lock():
            self._current_cycle_info.cycle_produced_artifacts.append(artifact_link)
            self.logger.info(f"Added artifact {artifact_link.artifact_doc_id} (type: {artifact_link.artifact_type}) to cycle {self._current_cycle_info.cycle_id}")
            # Project state itself (current_cycle_id etc.) doesn\'t change here, but cycle content does.
            # We need to decide if _current_cycle_info is directly part of _project_state or if we persist it separately
            # For now, assume _save_project_state will handle persisting _project_state which might include current cycle details
            # For more robust saving of ongoing cycle details, _current_cycle_info would need to be regularly synced or part of _project_state
            # The ProjectStateV2 schema itself doesn't have a 'current_cycle_details' field, only historical_cycles.
            # This implies that _current_cycle_info is managed in memory by StateManager and added to historical_cycles upon completion.
            # So, just updating _current_cycle_info is enough for now. It gets saved when the cycle completes.
            # However, for recovery, it's better if _current_cycle_info's state is also persisted somehow if it represents the "active" part of project_state.
            # Let's update a temporary field in _project_state or a specific mechanism.
            # For now, the save is implicit when the cycle completes.
            # A save might be desired here for resilience.
            # Let's try saving the whole state for now, assuming ProjectStateV2 might be extended or this is good practice.
            self._project_state.last_updated = datetime.now(timezone.utc) # Mark update
            self._save_project_state()

    def _backup_existing_status_file(self) -> None:
        """Backs up the existing status file by renaming it with a .bak extension and timestamp."""
        if self.status_file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file_path = self.status_file_path.with_suffix(f".{timestamp}.bak")
            try:
                self.status_file_path.rename(backup_file_path)
                self.logger.info(f"Backed up existing status file to: {backup_file_path}")
            except OSError as e:
                self.logger.error(f"Failed to backup status file from {self.status_file_path} to {backup_file_path}: {e}", exc_info=True)
        else:
            self.logger.info(f"No status file found at {self.status_file_path} to back up.")

    def add_key_decision_to_current_cycle(self, decision: KeyDecision) -> None:
        """Adds a key decision to the current active cycle."""
        if self._current_cycle_info is None or self._current_cycle_info.status != CycleStatus.IN_PROGRESS:
            self.logger.error("Cannot add key decision: No cycle is currently in progress.")
            raise StatusFileError("No cycle in progress to add key decision to.")

        with self._get_lock():
            self._current_cycle_info.key_decisions_in_cycle.append(decision)
            self.logger.info(f"Added key decision {decision.decision_id} to cycle {self._current_cycle_info.cycle_id}")
            self._project_state.last_updated = datetime.now(timezone.utc)
            self._save_project_state() # Save for resilience

    def complete_current_cycle(
        self, 
        final_status: CycleStatus, 
        arca_summary_doc_id: Optional[str] = None, 
        arca_decision: Optional[str] = None,
        issues_flagged_by_arca: Optional[List[str]] = None
    ) -> None:
        """
        Completes the current cycle, updates its status and details,
        moves it to historical_cycles, and updates the overall project status.
        """
        if self._current_cycle_info is None:
            self.logger.error("Cannot complete cycle: No cycle is currently active.")
            raise StatusFileError("No active cycle to complete.")
        
        if self._project_state.current_cycle_id != self._current_cycle_info.cycle_id:
            self.logger.error(f"Mismatch! current_cycle_id in project_state ({self._project_state.current_cycle_id}) != active _current_cycle_info.cycle_id ({self._current_cycle_info.cycle_id})")
            # This indicates a potential state inconsistency.
            # For now, we'll trust _current_cycle_info as the one being completed.
            # This scenario should ideally not happen with proper locking and state flow.

        with self._get_lock():
            cycle_to_complete = self._current_cycle_info
            cycle_to_complete.status = final_status
            cycle_to_complete.end_time_utc = datetime.now(timezone.utc)
            cycle_to_complete.arca_cycle_summary_doc_id = arca_summary_doc_id
            cycle_to_complete.arca_decision_at_cycle_end = arca_decision
            cycle_to_complete.issues_flagged_by_arca_in_cycle = issues_flagged_by_arca

            self._project_state.historical_cycles.append(cycle_to_complete.model_copy(deep=True))
            
            self.logger.info(f"Completed cycle {cycle_to_complete.cycle_id} with status {final_status}.")

            # Update overall project status based on cycle outcome
            if final_status == CycleStatus.COMPLETED_SUCCESS:
                self._project_state.overall_status = ProjectOverallStatus.CYCLE_COMPLETED_PENDING_REVIEW
                self._project_state.next_action_determined_by = "AUTONOMOUS_AGENT" # ARCA finished
            elif final_status == CycleStatus.COMPLETED_WITH_ISSUES_FOR_REVIEW:
                self._project_state.overall_status = ProjectOverallStatus.CYCLE_COMPLETED_PENDING_REVIEW
                self._project_state.next_action_determined_by = "AUTONOMOUS_AGENT" # ARCA finished, but with flags
            elif final_status in [CycleStatus.FAILED_INTERNAL_ERROR, CycleStatus.FAILED_UNRESOLVABLE_ISSUES]:
                self._project_state.overall_status = ProjectOverallStatus.PROJECT_FAILED # Or a more specific "CYCLE_FAILED_NEEDS_REVIEW"
                # Error details should be captured elsewhere, e.g., in ARCA logs or cycle summary
            elif final_status == CycleStatus.TERMINATED_BY_USER:
                 self._project_state.overall_status = ProjectOverallStatus.PROJECT_PAUSED_BY_USER # Or specific terminated status

            self._project_state.current_cycle_id = None
            self._current_cycle_info = None
            
            self._save_project_state()

    def update_master_artifact_ids(
        self,
        loprd_id: Optional[str] = None,
        blueprint_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        codebase_snapshot_link: Optional[str] = None,
    ) -> None:
        """Updates the master artifact IDs in the project state."""
        with self._get_lock():
            updated = False
            if loprd_id is not None:
                self._project_state.master_loprd_id = loprd_id
                self.logger.debug(f"Updated master_loprd_id to {loprd_id}")
                updated = True
            if blueprint_id is not None:
                self._project_state.master_blueprint_id = blueprint_id
                self.logger.debug(f"Updated master_blueprint_id to {blueprint_id}")
                updated = True
            if plan_id is not None:
                self._project_state.master_execution_plan_id = plan_id
                self.logger.debug(f"Updated master_execution_plan_id to {plan_id}")
                updated = True
            if codebase_snapshot_link is not None:
                self._project_state.link_to_live_codebase_collection_snapshot = codebase_snapshot_link
                self.logger.debug(f"Updated link_to_live_codebase_collection_snapshot to {codebase_snapshot_link}")
                updated = True
            
            if updated:
                self._save_project_state()
                self.logger.info("Master artifact IDs updated and state saved.")
            else:
                self.logger.debug("No master artifact IDs provided for update.")

    # --- Methods to be refactored or removed as they relate to the old schema ---
    # For now, these will likely break or need significant adaptation if ProjectStateV2
    # is the sole structure. The `runs` and `master_plans` attributes are not directly
    # on ProjectStateV2 in the current V2 schema draft.

    def get_last_status(self) -> Optional[Dict[str, Any]]:
        """Gets the status of the last executed stage from the latest run.
        
        ADAPTATION NOTE: This needs to be re-thought for ProjectStateV2. 
        Perhaps it should get the status of the last *cycle* or the overall project status.
        For now, it will look for a 'runs' attribute if it were part of ProjectStateV2,
        which it currently isn't. This method will likely be deprecated or heavily modified.
        """
        self.logger.warning("get_last_status() is based on the old 'runs' structure and may not be compatible with ProjectStateV2 as is.")
        # Assuming 'runs' might be reintroduced or this logic moved elsewhere.
        # This is a placeholder for the old logic:
        # if hasattr(self._project_state, 'runs') and self._project_state.runs:
        #     latest_run = self._project_state.runs[-1]
        #     if latest_run.get("stages"):
        #         return latest_run["stages"][-1]
        # return None
        if self._project_state.cycle_history:
            last_cycle = self._project_state.cycle_history[-1]
            return {
                "cycle_id": last_cycle.cycle_id,
                "status": self._project_state.overall_project_status, # Or a more specific cycle status if added
                "summary": last_cycle.arca_summary_of_cycle_outcome,
                "end_time": last_cycle.end_time
            }
        return None

    def get_full_status(self) -> Dict[str, Any]:
        """Returns the entire status structure (ProjectStateV2), project directory, and counts.
        ADAPTATION NOTE: This method now directly returns the model_dump of ProjectStateV2,
        plus the project_directory. Other counts might need to be derived from V2 fields.
        """
        self.logger.debug("get_full_status: Reading entire ProjectStateV2 structure.")
        # data = self._read_status_file() # _project_state is now ProjectStateV2
        data_dict = self._project_state.model_dump(mode='json') # Serialize to dict, handling datetime
        
        # Add project_directory to the returned data
        data_dict["project_directory"] = str(self.target_dir_path)
        # Add total_master_plans - this concept needs to be re-evaluated for V2.
        # For now, returning 0 or count if master_execution_plan_id exists.
        data_dict["total_master_plans"] = 1 if self._project_state.master_execution_plan_id else 0
        
        # Add total_runs - this concept maps to total_cycles in V2.
        data_dict["total_runs"] = len(self._project_state.cycle_history) # Or "total_cycles"
        
        self.logger.debug("get_full_status: Returning V2 data: %s", data_dict)
        return data_dict

    def get_next_stage(self) -> Optional[Union[int, float]]:
        """Determines the next stage to be executed based on the *latest run*.

        ADAPTATION NOTE: This method is highly dependent on the old 'runs' and 'stages'
        structure and stage definition files. It's unclear how 'stages' map directly
        to the new 'cycle_history' in ProjectStateV2. This will likely be deprecated
        or require a new component to manage stage execution *within* a cycle
        if that level of granularity is still driven by StateManager.
        For now, returning None as stage progression logic needs redesign for V2 cycles.
        """
        self.logger.warning("get_next_stage() is based on the old 'runs' and 'stages' structure "
                           "and is not directly compatible with ProjectStateV2's cycle model. "
                           "Returning None. Stage progression within cycles needs redesign.")
        return None
        # Old logic commented out:
        # self.logger.info("Determining next stage based on available definitions and latest run...")
        # highest_completed_stage = -1
        # try:
        #     status_data = self._read_status_file()
        #     if "runs" not in status_data or not status_data["runs"]:
        #         self.logger.info("No runs found. Defaulting to stage 0.")
        #         return 0

        #     latest_run = status_data["runs"][-1]
        #     if "stages" not in latest_run or not latest_run["stages"]:
        #         self.logger.info("No stages found in the latest run. Defaulting to stage 0.")
        #         return 0

        #     for stage_entry in latest_run["stages"]:
        #         stage_num = stage_entry.get("stage")
        #         stage_status = stage_entry.get("status")
        #         if stage_status in ["DONE", "PASS"] and stage_num is not None:
        #             try:
        #                 current_stage_float = float(stage_num)
        #                 if current_stage_float > highest_completed_stage:
        #                     highest_completed_stage = current_stage_float
        #             except ValueError:
        #                 self.logger.warning(f"Could not parse stage number '{stage_num}' as float.")

        # except StatusFileError as e:
        #     self.logger.error(f"Error reading status file for get_next_stage: {e}")
        #     return None # Or raise, or return a default

        # self.logger.info(f"Highest completed stage in latest run: {highest_completed_stage}")

        # # List available stage definition files (e.g., stage0.yaml, stage1.yaml, etc.)
        # available_stages = []
        # try:
        #     for f_path in self.server_stages_dir.glob("stage*.yaml"):
        #         try:
        #             stage_num_str = f_path.stem.replace("stage", "")
        #             available_stages.append(float(stage_num_str))
        #         except ValueError:
        #             self.logger.warning(f"Could not parse stage number from filename: {f_path.name}")
        # except Exception as e:
        #     self.logger.error(f"Error listing stage definition files: {e}")
        #     return None # Cannot determine next stage if definitions can't be listed
        
        # if not available_stages:
        #     self.logger.warning("No stage definition files found. Cannot determine next stage.")
        #     return None # Or 0 if we assume stage 0 always exists implicitly

        # available_stages.sort()
        # self.logger.debug(f"Available stages from definitions: {available_stages}")

        # if highest_completed_stage == -1: # No stages completed yet
        #     next_s = available_stages[0] if available_stages else 0
        #     self.logger.info(f"No completed stages, next stage is: {next_s}")
        #     return next_s

        # for stage_num_float in available_stages:
        #     if stage_num_float > highest_completed_stage:
        #         self.logger.info(f"Next stage to execute: {stage_num_float}")
        #         return stage_num_float
        
        # # If all defined stages are completed or highest_completed_stage is beyond defined stages
        # self.logger.info("All defined stages seem to be completed or no next stage found based on current definitions.")
        # return None # Or a special value indicating completion or error

    def update_status(
        self,
        stage: Union[int, float],
        status: str,
        artifacts: List[str],
        reason: Optional[str] = None,
        reflection_text: Optional[str] = None, # This is for direct passing
        error_details: Optional[AgentErrorDetails] = None, # <<< New argument
    ) -> bool:
        """Updates the status file by adding a new entry to the *latest run*.

        Creates the first run (run_id: 0) if no runs exist.
        Validates input status.
        Adds a timestamp to the new entry.
        Writes the updated list back to the file.
        Uses and clears any pending reflection text if reflection_text argument is None.

        Args:
            stage: The stage number (e.g., 1.0, 2).
            status: The result status (e.g., "PASS", "FAIL", "DONE").
            artifacts: A list of relative paths to generated artifacts.
            reason: Optional reason, e.g., for FAIL status.
            reflection_text: Optional reflection text for this stage update. If None,
                             will try to use and clear _current_cycle_info.
            error_details: Optional structured details about an agent error. # <<< Updated docstring

        Returns:
            True if the update was successful, False otherwise.
        """
        self.logger.info(f"Attempting to update status for Stage: {stage}, Status: {status}")
        valid_statuses = ["PASS", "FAIL", "DONE"]
        if status.upper() not in valid_statuses:
            self.logger.error(f"Invalid status provided: {status}. Must be one of {valid_statuses}")
            return False

        # Determine the reflection text to use
        final_reflection_text = reflection_text
        if final_reflection_text is None and self._current_cycle_info is not None:
            final_reflection_text = self._current_cycle_info.arca_summary_of_cycle_outcome
            self.logger.info(f"Using pending reflection text for stage {stage} update.")
            self._current_cycle_info = None # Clear after use
            self.logger.info("Cleared pending reflection text after incorporating into status update.")
        elif self._current_cycle_info is not None and reflection_text is not None:
            self.logger.warning(
                f"Both a direct reflection_text and a pending_reflection_text were present for stage {stage}. "
                f"Prioritizing the directly passed reflection_text. Pending reflection was NOT cleared."
            )
            # In this case, we prioritize the explicitly passed `reflection_text`.
            # The `_current_cycle_info` remains uncleared, which might be unexpected.
            # Consider if _current_cycle_info should be cleared anyway or if this state is an error.
            # For now, it persists if a direct one is also provided.

        # Create the new status entry
        new_entry: Dict[str, Any] = {
            "stage": float(stage),
            "status": status.upper(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts if isinstance(artifacts, list) else [],
        }
        if reason:
            new_entry["reason"] = reason
        if final_reflection_text is not None: # Use the determined reflection text
            new_entry["reflection_text"] = final_reflection_text
        if error_details is not None: # <<< Add error details if provided
            # Store as JSON string to ensure serializability
            new_entry["error_details"] = error_details.model_dump_json(indent=2)

        try:
            # Read the current state
            status_data = self._read_status_file()
            runs = status_data.get("runs", [])

            # Determine the target run_id
            target_run_id: int = -1
            target_run_index: int = -1

            if not runs:
                # First run ever, create it with run_id 0
                target_run_id = 0
                new_run = {
                    "run_id": target_run_id, # Ensure run_id is explicitly set to 0
                    "start_timestamp": datetime.now(timezone.utc).isoformat(),
                    "status_updates": [new_entry],
                }
                runs.append(new_run)
                self.logger.info(
                    f"Creating first run (id: {target_run_id}) and adding status update."
                )
            else:
                # Find the latest run based on highest existing run_id
                try:
                    # Ensure key function always returns comparable types (int)
                    latest_run = max(runs, key=lambda r: r.get("run_id") if isinstance(r.get("run_id"), int) else -1)
                    current_run_id = latest_run.get("run_id")
                    
                    # Validate the run_id
                    if current_run_id is None or not isinstance(current_run_id, int) or current_run_id < 0:
                        self.logger.error(f"Found invalid run_id '{current_run_id}' (type: {type(current_run_id)}) in latest run: {latest_run}")
                        return False
                        
                    # Find the index of the latest run to modify it
                    for i, run in enumerate(runs):
                        if run.get("run_id") == current_run_id:
                            target_run_index = i
                            break

                    if target_run_index == -1:
                        # Should not happen if target_run_id was found
                        raise StatusFileError(
                            f"Internal error: Could not find index for run_id {current_run_id}."
                        )

                    # Append the new status to the latest run's updates
                    if "status_updates" not in runs[target_run_index]:
                        runs[target_run_index][
                            "status_updates"
                        ] = []  # Initialize if missing (defensive)
                    runs[target_run_index]["status_updates"].append(new_entry)
                    self.logger.info(f"Appending status update to latest run (id: {current_run_id}).")

                except (ValueError, TypeError) as e:
                    # Error if run_id is missing, not an int, or list is empty/malformed
                    self.logger.error(f"Error processing run list to find latest run_id: {e}")
                    return False

            # Write the modified structure back
            self._write_status_file({"runs": runs})
            self.logger.info(f"Status update for Stage {stage} completed successfully.")
            return True

        except StatusFileError as e:
            self.logger.error(f"Failed to update status file: {e}")
            return False
        except Exception as e:
            self.logger.exception(f"Unexpected error during status update: {e}")
            return False

    def _get_chroma_client(self) -> Optional[chromadb.ClientAPI]:
        """Internal helper to get the initialized client or log error."""
        if not self.chroma_client:
            self.logger.error(
                "ChromaDB client was not initialized successfully. Cannot perform DB operations."
            )
        return self.chroma_client

    def store_artifact_context_in_chroma(
        self,
        stage_number: float,
        rel_path: str,
        content: str,
        artifact_type: str = "unknown",
    ) -> bool:
        """Stores artifact content as context in ChromaDB.

        Returns:
            True if successful.
        Raises:
            ChromaOperationError: If the ChromaDB operation fails.
        """
        client = self._get_chroma_client()
        if not client:
            # Raise an error if client is unavailable, as operation cannot succeed
            raise ChromaOperationError("ChromaDB client is not available.")

        collection = client.get_or_create_collection(name=self._CONTEXT_COLLECTION_NAME)
        if not collection:
            # Raise error if collection cannot be accessed
            raise ChromaOperationError(f"Could not get or create collection '{self._CONTEXT_COLLECTION_NAME}'.")

        # Generate a stable ID based on relative path and potentially stage?
        # Using hash for now, consider if stage needs to be included for uniqueness across stages
        doc_id = hashlib.sha1(rel_path.encode()).hexdigest()

        metadata = {
            "stage_number": stage_number,
            "relative_path": rel_path,
            "artifact_type": artifact_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "artifact_context", # Added source type
        }
        self.logger.debug(
            f"Attempting to upsert artifact context to Chroma '{self._CONTEXT_COLLECTION_NAME}': ID={doc_id}, Path={rel_path[:50]}..."
        )
        try:
            # <<< Use chroma_utils wrapper instead of direct collection call >>>
            # Using add_or_update_document which handles upsert logic and collection access
            success = chroma_utils.add_or_update_document(
                collection_name=self._CONTEXT_COLLECTION_NAME,
                doc_id=doc_id,
                document_content=content,
                metadata=metadata
            )
            # <<< Check success from utility function >>>
            if success:
                self.logger.info(
                    f"Successfully stored artifact context for '{rel_path}' (Stage {stage_number}) in Chroma."
                )
                return True
            else:
                # Error should have been logged in chroma_utils
                raise ChromaOperationError(
                    f"chroma_utils.add_or_update_document failed for artifact '{rel_path}' in collection '{self._CONTEXT_COLLECTION_NAME}'"
                )

        except Exception as e:
            self.logger.exception(
                f"Failed to store artifact context for '{rel_path}' in Chroma collection '{self._CONTEXT_COLLECTION_NAME}': {e}"
            )
            # Re-raise as specific error, ensuring original exception context is kept if applicable
            if not isinstance(e, ChromaOperationError):
                raise ChromaOperationError(
                    f"Failed to store artifact context for '{rel_path}' in collection '{self._CONTEXT_COLLECTION_NAME}'"
                ) from e
            else:
                raise e # Re-raise original ChromaOperationError

    def get_artifact_context_from_chroma(
        self,
        query: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves relevant artifact context from ChromaDB based on a query.

        Returns:
            A list of results (dictionaries with document, metadata, distance).
        Raises:
            ChromaOperationError: If the ChromaDB query operation fails.
        """
        # <<< REMOVED client/collection getting - Handled by chroma_utils.query_documents >>>
        # client = self._get_chroma_client()
        # if not client:
        #     raise ChromaOperationError("ChromaDB client is not available.")
        # collection = client.get_or_create_collection(name=self._CONTEXT_COLLECTION_NAME)
        # if not collection:
        #      raise ChromaOperationError(f"Could not get or create collection '{self._CONTEXT_COLLECTION_NAME}'.")

        # Handle empty filter case for ChromaDB
        final_where = where_filter if where_filter else None

        self.logger.debug(
            f"Querying Chroma collection '{self._CONTEXT_COLLECTION_NAME}' for artifact context: '{query[:50]}...' (n={n_results}, filter={final_where})"
        )
        try:
            # <<< Use the chroma_utils.query_documents function >>>
            # This function handles client/collection access and formats the results
            results = chroma_utils.query_documents(
                collection_name=self._CONTEXT_COLLECTION_NAME,
                query_texts=[query],
                n_results=n_results,
                where_filter=final_where,
                include=["metadatas", "documents", "distances"], # Ensure required fields included
            )

            # Check if the utility function returned None (indicating an error)
            if results is None:
                 # Error should have been logged in chroma_utils
                 raise ChromaOperationError(f"ChromaDB query operation failed for collection '{self._CONTEXT_COLLECTION_NAME}'. Check logs for details.")

            # <<< Results are already formatted correctly by chroma_utils.query_documents >>>
            self.logger.info(
                f"Retrieved {len(results)} artifact context results from Chroma for query '{query[:50]}...'"
            )
            return results

        except Exception as e:
            # Catch any unexpected errors here or re-raised errors from query_documents
            self.logger.exception(
                f"Failed to get artifact context from Chroma collection '{self._CONTEXT_COLLECTION_NAME}': {e}"
            )
            # Re-raise as specific error, ensuring original context
            if not isinstance(e, ChromaOperationError):
                 raise ChromaOperationError(
                     f"Failed to query artifact context from collection '{self._CONTEXT_COLLECTION_NAME}'"
                 ) from e
            else:
                 raise e # Re-raise original ChromaOperationError

    def persist_reflections_to_chroma(
        self, run_id: int | None, stage_number: float, reflections: str
    ) -> bool:
        """Persists reflection text to ChromaDB.

        Returns:
            True if successful.
        Raises:
            ChromaOperationError: If the ChromaDB operation fails.
        """
        # Validate provided run_id; if invalid, attempt to infer from status file
        if run_id is None or not isinstance(run_id, int) or run_id < 0:
            self.logger.debug(
                "Supplied run_id '%s' is invalid. Falling back to the run_id of the latest run.",
                run_id,
            )
            try:
                status_data = self._read_status_file()
                runs = status_data.get("runs", [])
                if runs:
                    latest_run = max(runs, key=lambda r: r.get("run_id", -1))
                    run_id = latest_run.get("run_id")
                else:
                    run_id = 0  # initialise first run
            except Exception as e:
                self.logger.error(
                    "Failed to derive run_id from status file: %s", e, exc_info=True
                )
                raise ChromaOperationError("Unable to determine run_id for reflection persistence") from e

        # Final validation after fallback
        if run_id is None or not isinstance(run_id, int):
            err_msg = (
                f"Invalid or missing run_id ({run_id}, type: {type(run_id)}) after fallback. Cannot persist reflection."
            )
            self.logger.error(err_msg)
            raise ChromaOperationError(err_msg)

        # Create a unique ID for this reflection entry
        doc_id = f"run{run_id}_stage{stage_number}_{uuid.uuid4()}"
        metadata = {
            "run_id": run_id,
            "stage_number": stage_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "reflection", # Added source type
        }
        self.logger.debug(
            f"Attempting to add reflection to Chroma '{self._REFLECTIONS_COLLECTION_NAME}': ID={doc_id}, Stage={stage_number}"
        )
        try:
            # <<< Use chroma_utils.add_documents >>>
            success = chroma_utils.add_documents(
                collection_name=self._REFLECTIONS_COLLECTION_NAME,
                ids=[doc_id],
                documents=[reflections],
                metadatas=[metadata]
            )
            # <<< Check success from utility function >>>
            if success:
                self.logger.info(
                    f"Successfully persisted reflection for Run {run_id}, Stage {stage_number} in Chroma."
                )
                return True
            else:
                 # Error logged in chroma_utils
                 raise ChromaOperationError(
                    f"chroma_utils.add_documents failed for reflection Run {run_id}, Stage {stage_number} in collection '{self._REFLECTIONS_COLLECTION_NAME}'"
                 )

        except Exception as e:
            self.logger.exception(
                f"Failed to persist reflection for Run {run_id}, Stage {stage_number} in Chroma collection '{self._REFLECTIONS_COLLECTION_NAME}': {e}"
            )
            # Re-raise as specific error
            if not isinstance(e, ChromaOperationError):
                raise ChromaOperationError(
                    f"Failed to add reflection to collection '{self._REFLECTIONS_COLLECTION_NAME}'"
                ) from e
            else:
                raise e # Re-raise original ChromaOperationError

    def get_reflection_context_from_chroma(
        self,
        query: str,
        n_results: int = 3,
        filter_stage_min: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves relevant reflection context from ChromaDB based on a query.

        Returns:
            A list of results (dictionaries with document, metadata, distance).
        Raises:
            ChromaOperationError: If the ChromaDB query operation fails.
        """
        # <<< REMOVED client/collection getting - Handled by chroma_utils.query_documents >>>
        # client = self._get_chroma_client()
        # if not client:
        #      raise ChromaOperationError("ChromaDB client is not available.")
        # collection = client.get_or_create_collection(name=self._REFLECTIONS_COLLECTION_NAME)
        # if not collection:
        #     raise ChromaOperationError(f"Could not get or create collection '{self._REFLECTIONS_COLLECTION_NAME}'.")

        where_filter = {}
        if filter_stage_min is not None:
            # Assuming stage_number is stored as float in metadata
            where_filter = {"stage_number": {"$gte": filter_stage_min}}

        # Handle empty filter case for ChromaDB
        final_where = where_filter if where_filter else None

        self.logger.debug(
            f"Querying Chroma collection '{self._REFLECTIONS_COLLECTION_NAME}' for reflections: '{query[:50]}...' (n={n_results}, filter={final_where})"
        )
        try:
            # <<< Use the chroma_utils.query_documents function >>>
            results = chroma_utils.query_documents(
                 collection_name=self._REFLECTIONS_COLLECTION_NAME,
                 query_texts=[query],
                 n_results=n_results,
                 where_filter=final_where,
                 include=["metadatas", "documents", "distances"],
             )

            # Check if the utility function returned None (indicating an error)
            if results is None:
                 # Error logged in chroma_utils
                 raise ChromaOperationError(f"ChromaDB query operation failed for collection '{self._REFLECTIONS_COLLECTION_NAME}'. Check logs for details.")

            # <<< Results are already formatted correctly by chroma_utils.query_documents >>>
            self.logger.info(
                f"Retrieved {len(results)} reflection context results from Chroma for query '{query[:50]}...'"
            )
            return results

        except Exception as e:
            self.logger.exception(
                f"Failed to get reflection context from Chroma collection '{self._REFLECTIONS_COLLECTION_NAME}': {e}"
            )
            # Re-raise as specific error
            if not isinstance(e, ChromaOperationError):
                 raise ChromaOperationError(
                     f"Failed to query reflection context from collection '{self._REFLECTIONS_COLLECTION_NAME}'"
                 ) from e
            else:
                 raise e # Re-raise original ChromaOperationError

    def list_artifact_metadata(
        self,
        stage_filter: Optional[float] = None,
        artifact_type_filter: Optional[str] = None,
        limit: Optional[int] = None,  # Optional limit on number of results
    ) -> List[Dict[str, Any]]:
        """Lists metadata of stored artifacts, optionally filtering.

        Returns:
            List of metadata dictionaries for matching artifacts.
        Raises:
            ChromaOperationError: If the ChromaDB get operation fails.
        """
        client = self._get_chroma_client()
        if not client:
            raise ChromaOperationError("ChromaDB client is not available.")

        collection = client.get_or_create_collection(name=self._CONTEXT_COLLECTION_NAME)
        if not collection:
            raise ChromaOperationError(f"Could not get or create collection '{self._CONTEXT_COLLECTION_NAME}'.")

        where_filter = {}
        if stage_filter is not None:
            where_filter["stage_number"] = stage_filter
        if artifact_type_filter:
            where_filter["artifact_type"] = artifact_type_filter

        # Handle empty filter case for ChromaDB
        final_where = where_filter if where_filter else None

        self.logger.debug(
            f"Listing artifact metadata from '{self._CONTEXT_COLLECTION_NAME}' with filter: {final_where}, limit: {limit}"
        )

        try:
            results = collection.get(
                where=final_where,
                limit=limit,
                include=["metadatas"] # Only fetch metadata
            )
            metadata_list = results.get("metadatas", [])
            self.logger.info(
                f"Retrieved {len(metadata_list)} artifact metadata entries matching filter {final_where}"
            )
            return metadata_list
        except Exception as e:
            self.logger.exception(
                f"Failed to list artifact metadata from collection '{self._CONTEXT_COLLECTION_NAME}': {e}"
            )
            raise ChromaOperationError(
                f"Failed to list artifact metadata from collection '{self._CONTEXT_COLLECTION_NAME}'"
            ) from e

    def get_all_reflections(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves all reflections, up to a limit, without semantic search.

        Uses collection.get() which retrieves based on internal order or filters (not used here),
        not semantic similarity.

        Args:
            limit: The maximum number of reflections to return.

        Returns:
            A list of reflection dictionaries (metadata and document).

        Raises:
            ChromaOperationError: If the ChromaDB get operation fails.
        """
        client = self._get_chroma_client() # <<< Keep client check here for get() >>>
        if not client:
            raise ChromaOperationError("ChromaDB client is not available.")

        collection = client.get_or_create_collection(name=self._REFLECTIONS_COLLECTION_NAME) # <<< Keep collection getting here for get() >>>
        if not collection:
            raise ChromaOperationError(f"Could not get or create collection '{self._REFLECTIONS_COLLECTION_NAME}'.")

        self.logger.debug(
            f"Getting all reflections from '{self._REFLECTIONS_COLLECTION_NAME}' (limit: {limit})"
        )

        try:
            # Use collection.get() to retrieve documents without a query
            results = collection.get(
                limit=limit,
                include=["metadatas", "documents"] # Only fetch metadata and documents
            )

            # Reformat results into a simpler list of dictionaries
            formatted_results = []
            if results and results.get("ids"):
                num_results = len(results["ids"])
                ids = results["ids"]
                metadatas = results.get("metadatas", [None] * num_results)
                documents = results.get("documents", [None] * num_results)

                for i in range(num_results):
                    formatted_results.append(
                        {
                            "id": ids[i],
                            "metadata": metadatas[i],
                            "document": documents[i],
                            # Add other relevant fields if needed, like timestamp from metadata
                            "timestamp": metadatas[i].get("timestamp") if metadatas[i] else None
                        }
                    )
                # Sort by timestamp descending if possible (best effort)
                try:
                    formatted_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                except Exception as sort_err:
                    self.logger.warning(f"Could not sort reflections by timestamp: {sort_err}")

            self.logger.info(
                f"Retrieved {len(formatted_results)} reflections from collection '{self._REFLECTIONS_COLLECTION_NAME}' (limit: {limit})."
            )
            return formatted_results

        except Exception as e:
            self.logger.exception(
                f"Failed to get all reflections from collection '{self._REFLECTIONS_COLLECTION_NAME}': {e}"
            )
            raise ChromaOperationError(
                f"Failed to get all reflections from collection '{self._REFLECTIONS_COLLECTION_NAME}'"
            ) from e

    def initialize_project(self, project_id: str, project_name: Optional[str], initial_user_goal_summary: str, initial_user_goal_doc_id: Optional[str] = None) -> ProjectStateV2:
        """
        Initializes a new project. Creates project_status.json if it doesn't exist,
        or errors if it exists and is already a V2 project.
        """
        with self._get_lock():
            if self.status_file_path.exists():
                try:
                    # Try to read to see if it's a V2 project already
                    existing_state = self._read_status_file()
                    if existing_state.project_id == project_id:
                        self.logger.warning(f"Project {project_id} already initialized at {self.status_file_path}. Loading existing state.")
                        self._project_state = existing_state
                        # Try to set current_cycle_info if a cycle was in progress
                        if self._project_state.current_cycle_id:
                             found_cycle = next((c for c in self._project_state.historical_cycles if c.cycle_id == self._project_state.current_cycle_id and c.status == CycleStatus.IN_PROGRESS), None)
                             if found_cycle: self._current_cycle_info = found_cycle
                        return self._project_state
                    else:
                        self.logger.error(f"Status file {self.status_file_path} exists but for a different project_id ({existing_state.project_id}). Cannot initialize {project_id}.")
                        raise StatusFileError(f"Status file exists for a different project: {existing_state.project_id}")
                except StatusFileError: 
                    # File exists but is not a valid V2 schema or unreadable, potentially allow overwrite if forced.
                    # For now, error out.
                    self.logger.error(f"Status file {self.status_file_path} exists but is invalid. Cannot initialize. Please backup/remove manually.")
                    raise StatusFileError(f"Existing invalid status file at {self.status_file_path}. Manual intervention required.")

            # Create new project state
            self.logger.info(f"Initializing new project with ID: {project_id} at {self.target_dir_path}")
            now = datetime.now(timezone.utc)
            self._project_state = ProjectStateV2(
                project_id=project_id,
                project_name=project_name,
                initial_user_goal_summary=initial_user_goal_summary,
                initial_user_goal_doc_id=initial_user_goal_doc_id,
                schema_version="2.0.0",
                created_at_utc=now,
                last_updated_utc=now,
                overall_status=ProjectOverallStatus.NOT_STARTED
            )
            self._current_cycle_info = None # No active cycle yet
            self._write_status_file(self._project_state)
            self.logger.info(f"Project {project_id} initialized successfully. Status file: {self.status_file_path}")
            return self._project_state.model_copy(deep=True)

    def _get_lock(self) -> Union[filelock.FileLock, DummyFileLock]:
        """Returns the appropriate lock object based on configuration."""
        return self._lock

    def set_current_cycle_info(self, cycle_info: CycleInfo) -> None:
        """Stores the current cycle information."""
        self._current_cycle_info = cycle_info
        self.logger.info(f"Current cycle info set. Cycle ID: {cycle_info.cycle_id}")

    def get_current_cycle_info(self) -> Optional[CycleInfo]:
        """Retrieves the current cycle information."""
        if self._current_cycle_info is not None:
            self.logger.info(f"Retrieved current cycle info. Cycle ID: {self._current_cycle_info.cycle_id}")
        else:
            self.logger.info("No current cycle info to retrieve.")
        return self._current_cycle_info

    def export_cursor_rule(self) -> Optional[str]:
        """Exports the chungoid.mdc rule to ~/.cursor/rules/chungoid.mdc.

        Returns:
            The absolute path to the exported rule file if successful, None otherwise.
        """
        rule_filename = "chungoid.mdc"
        target_dir = Path.home() / ".cursor" / "rules"
        rule_file_path = target_dir / rule_filename

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            
            rule_content = """---
description: Chungoid AI development assistant rule providing core commands.
globs: []
alwaysApply: false
---
# Chungoid Bootstrap Rule for Cursor
# Version: 0.1.0
# This rule helps initialize a Cursor IDE environment for working with Chungoid.

# On project open, suggest setting the Chungoid project context if not already done.
# (This part is more of a conceptual placeholder as direct IDE event hooking is complex)

# Provides a command to initialize the current folder as a Chungoid project.
@command(
    {
        "name": "chungoid.initializeProjectHere",
        "title": "Chungoid: Initialize Project in Current Folder",
        "description": "Sets up the current folder as a Chungoid project by creating .chungoid/project_status.json.",
        "category": "Chungoid"
    }
)
async def initialize_chungoid_project_here(ide_services):
    project_path = await ide_services.get_workspace_path()
    if not project_path:
        await ide_services.show_error_message("Could not determine workspace path.")
        return
    
    response = await ide_services.call_mcp_tool(
        tool_name="initialize_project",
        tool_arguments={"project_directory": project_path}
    )
    if response and not response.get('isError'):
        await ide_services.show_information_message(f"Chungoid project initialized at {project_path}")
    else:
        error_message = response.get('error', {}).get('message', "Unknown error during initialization.")
        await ide_services.show_error_message(f"Failed to initialize Chungoid project: {error_message}")

# Provides a command to set the active Chungoid project context.
@command(
    {
        "name": "chungoid.setProjectContext",
        "title": "Chungoid: Set Active Project Context",
        "description": "Prompts for a directory and sets it as the active Chungoid project context.",
        "category": "Chungoid"
    }
)
async def set_chungoid_project_context(ide_services):
    project_path = await ide_services.show_input_box(
        title="Set Chungoid Project Directory",
        prompt="Enter the absolute path to your Chungoid project directory",
        placeholder=await ide_services.get_workspace_path() or "/path/to/your/project"
    )
    if not project_path:
        return

    response = await ide_services.call_mcp_tool(
        tool_name="set_project_context",
        tool_arguments={"project_directory": project_path}
    )
    if response and not response.get('isError'):
        await ide_services.show_information_message(f"Chungoid project context set to: {project_path}")
    else:
        error_message = response.get('error', {}).get('message', "Unknown error setting context.")
        await ide_services.show_error_message(f"Failed to set Chungoid project context: {error_message}")

# Add more commands and context providers as Chungoid evolves.
# Example: A command to show current project status via MCP tool.
@command(
    {
        "name": "chungoid.getCurrentStatus",
        "title": "Chungoid: Get Current Project Status",
        "description": "Retrieves and displays the current status of the active Chungoid project.",
        "category": "Chungoid"
    }
)
async def get_chungoid_project_status(ide_services):
    response = await ide_services.call_mcp_tool(tool_name="get_project_status", tool_arguments={})
    if response and not response.get('isError') and response.get('content'):
        status_text = response['content'][0].get('text', "Failed to parse status.")
        # Attempt to pretty-print if it's JSON
        try:
            import json
            status_obj = json.loads(status_text)
            pretty_status = json.dumps(status_obj, indent=2)
            await ide_services.show_information_message(f"Chungoid Project Status:\n{pretty_status}", use_modal=True)
        except:
            await ide_services.show_information_message(f"Chungoid Project Status:\n{status_text}", use_modal=True)
"""
            with open(rule_file_path, "w", encoding="utf-8") as f:
                f.write(rule_content)
            
            self.logger.info(f"Exported Cursor rule to: {rule_file_path}")
            return str(rule_file_path)
        
        except Exception as e:
            self.logger.error(f"Failed to export Cursor rule to {rule_file_path}: {e}", exc_info=True)
            return None

    def get_or_create_current_run_id(self) -> Optional[int]:
        """Gets the run_id of the latest run, or determines the next one (0 if none exist).

        This method reads the status file to find the highest existing run_id.
        It does NOT modify the status file.

        Returns:
            The integer run_id of the latest run (or 0 if no runs exist), 
            or None if an error occurs or run_ids are invalid.
        """
        self.logger.debug("Attempting to get current run_id.")
        try:
            # Read the current state using the locking read method
            status_data = self._read_status_file()
            runs = status_data.get("runs", [])

            if not runs:
                # No runs exist yet, the next run will be 0
                self.logger.info("No existing runs found, next run_id will be 0.")
                return 0
            else:
                # Find the latest run based on highest existing run_id
                try:
                    # Ensure key function always returns comparable types (int)
                    latest_run = max(runs, key=lambda r: r.get("run_id") if isinstance(r.get("run_id"), int) else -1)
                    current_run_id = latest_run.get("run_id")
                    
                    # Validate the run_id
                    if current_run_id is None or not isinstance(current_run_id, int) or current_run_id < 0:
                        self.logger.error(f"Found invalid run_id '{current_run_id}' (type: {type(current_run_id)}) in latest run: {latest_run}")
                        return None # Indicate error
                        
                    self.logger.info(f"Determined current run_id: {current_run_id}")
                    return current_run_id
                except (ValueError, TypeError) as e:
                    # This except block should now only catch errors from max() if runs is empty after filtering,
                    # or other unexpected TypeErrors during comparison if the key func has issues.
                    self.logger.error(f"Error processing run list to find latest run_id: {e}")
                    return None # Indicate error

        except StatusFileError as e:
            self.logger.error(f"Failed to get current run_id due to StatusFileError: {e}")
            return None # Indicate error reading status file
        except Exception as e:
            self.logger.exception(f"Unexpected error getting current run_id: {e}")
            return None # Indicate unexpected error

    def save_master_execution_plan(self, plan: MasterExecutionPlan) -> bool:
        """Saves a MasterExecutionPlan to the project_status.json file.

        Args:
            plan: The MasterExecutionPlan object to save.

        Returns:
            True if successful, False otherwise.
        """
        lock = self._get_lock()
        try:
            with lock:
                self.logger.info(f"Attempting to save MasterExecutionPlan ID: {plan.id}")
                current_status_data = self._read_status_file()
                
                # Ensure 'master_plans' key exists and is a list
                if "master_plans" not in current_status_data or not isinstance(current_status_data.get("master_plans"), list):
                    current_status_data["master_plans"] = []
                
                # Pydantic model_dump() converts to dict, including datetime to ISO strings etc.
                current_status_data["master_plans"].append(plan.model_dump())
                
                self._write_status_file(current_status_data)
                self.logger.info(f"Successfully saved MasterExecutionPlan ID: {plan.id}")
                return True
        except StatusFileError as e:
            self.logger.error(f"Failed to save MasterExecutionPlan due to status file error: {e}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while saving MasterExecutionPlan: {e}", exc_info=True)
            return False

    def get_all_master_execution_plans(self) -> List[MasterExecutionPlan]:
        """Retrieves all saved MasterExecutionPlans from the status file.

        Returns:
            A list of MasterExecutionPlan objects.
        """
        try:
            current_status_data = self._read_status_file()
            plan_dicts = current_status_data.get("master_plans", [])
            plans = [MasterExecutionPlan(**plan_data) for plan_data in plan_dicts]
            return plans
        except StatusFileError as e:
            self.logger.error(f"Failed to retrieve master execution plans due to status file error: {e}", exc_info=True)
            return []
        except ValidationError as ve: # Assuming ValidationError is Pydantic's
            self.logger.error(f"Failed to parse stored master execution plans: {ve}", exc_info=True)
            return [] 
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while retrieving master execution plans: {e}", exc_info=True)
            return []

    def _get_paused_flow_file_path(self, run_id: str) -> Path:
        """Constructs the Path object for a specific paused flow state file."""
        paused_runs_dir = self.status_file_path.parent / "paused_runs"
        paused_runs_dir.mkdir(parents=True, exist_ok=True)
        return paused_runs_dir / f"paused_run_{run_id}.json" # Ensure 'paused_run_' prefix

    def save_paused_flow_state(self, paused_details: PausedRunDetails) -> bool:
        """Saves the state of a paused run to a dedicated JSON file.

        Args:
            paused_details: The PausedRunDetails object containing state to save.

        Returns:
            True if saving was successful, False otherwise.
        """
        paused_runs_dir = self.status_file_path.parent / "paused_runs"
        try:
            paused_runs_dir.mkdir(parents=True, exist_ok=True)
            file_path = paused_runs_dir / f"paused_run_{paused_details.run_id}.json"
            
            self.logger.info(f"Saving paused flow state for run_id '{paused_details.run_id}' to {file_path}")
            
            with file_path.open("w", encoding="utf-8") as f:
                f.write(paused_details.model_dump_json(indent=2))
            
            self.logger.info(f"Successfully saved paused state for run_id '{paused_details.run_id}'.")
            return True
        except IOError as e:
            self.logger.error(f"IOError saving paused state for run '{paused_details.run_id}' to {file_path}: {e}")
            return False
        except Exception as e:
            self.logger.exception(f"Unexpected error saving paused state for run '{paused_details.run_id}': {e}")
            return False

    def load_paused_flow_state(self, run_id: str) -> Optional[PausedRunDetails]:
        """Loads the state of a paused flow run.
        """
        paused_file_path = self._get_paused_flow_file_path(run_id)
        self.logger.info(f"Attempting to load paused flow state for run_id '{run_id}' from {paused_file_path}")

        if not paused_file_path.exists() or not paused_file_path.is_file():
            self.logger.warning(f"Paused flow state file not found for run_id '{run_id}' at {paused_file_path}")
            return None
        try:
            content = paused_file_path.read_text(encoding="utf-8")
            # The crucial change from P10.7 was here:
            details = PausedRunDetails.model_validate_json(content) 

            self.logger.info(f"Successfully loaded paused flow state for run_id '{run_id}'")
            return details
        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError loading paused state for run '{run_id}' from {paused_file_path}: {e}")
            return None
        except ValidationError as e: # pydantic.ValidationError
            self.logger.error(f"Pydantic ValidationError loading paused state for run '{run_id}' from {paused_file_path}: {e}")
            return None
        except IOError as e:
            self.logger.error(f"IOError loading paused state for run '{run_id}' from {paused_file_path}: {e}")
            return None
        except Exception as e:
            self.logger.exception(f"Unexpected error loading paused state for run '{run_id}' from {paused_file_path}: {e}")
            return None

    def delete_paused_flow_state(self, run_id: str) -> bool:
        """Deletes the saved state file for a given paused run.

        Args:
            run_id: The unique ID of the paused run whose state file should be deleted.

        Returns:
            True if the file was deleted or did not exist, False on error.
        """
        paused_file_path = self._get_paused_flow_file_path(run_id)
        self.logger.debug(
            f"Attempting to delete paused flow state file for run_id '{run_id}' at {paused_file_path}"
        )

        try:
            if paused_file_path.exists() and paused_file_path.is_file():
                self.logger.info(f"Deleting paused flow state file for run_id '{run_id}' at {paused_file_path}")
                paused_file_path.unlink() # Use unlink() to delete the file
                self.logger.info(f"Successfully deleted paused state file for run_id '{run_id}'.")
                return True
            elif not paused_file_path.exists():
                self.logger.info(f"Paused state file for run_id '{run_id}' not found at {paused_file_path}. No deletion needed.")
                return True # Consider not found as success in terms of the state being gone
            else:
                # Path exists but is not a file (e.g., a directory)
                self.logger.warning(f"Path for paused state run_id '{run_id}' exists but is not a file: {paused_file_path}. Cannot delete.")
                return False
        except OSError as e:
            self.logger.error(f"OSError deleting paused state file for run '{run_id}' at {paused_file_path}: {e}")
            return False
        except Exception as e:
            self.logger.exception(f"Unexpected error deleting paused state file for run '{run_id}' from {paused_file_path}: {e}")
            return False

    def record_human_feedback_for_cycle(
        self,
        cycle_id_to_update: str,
        human_feedback_summary_or_id: str,
        next_cycle_status: str = "ready_for_next_cycle" # Allows flexibility, e.g., "paused_pending_directives"
    ) -> bool:
        """
        Records human feedback for a completed cycle and updates the project status.

        Args:
            cycle_id_to_update: The ID of the cycle for which feedback is being provided.
            human_feedback_summary_or_id: The summary of human feedback or an artifact ID 
                                            pointing to detailed feedback in ChromaDB.
            next_cycle_status: The overall_project_status to set after recording feedback.
        
        Returns:
            True if feedback was recorded and state saved, False otherwise.
        """
        with self._get_lock():
            project_state = self._project_state # self._project_state is ProjectStateV2
            
            cycle_to_update = next((c for c in project_state.cycle_history if c.cycle_id == cycle_id_to_update), None)

            if not cycle_to_update:
                self.logger.error(f"Cannot record human feedback: Cycle ID '{cycle_id_to_update}' not found in history.")
                return False

            if cycle_to_update.end_time is None:
                self.logger.warning(f"Recording human feedback for cycle '{cycle_id_to_update}' which has no end_time. This might indicate it wasn't properly concluded by ARCA.")
            
            cycle_to_update.human_feedback_and_directives_for_next_cycle = human_feedback_summary_or_id
            project_state.overall_project_status = next_cycle_status
            # current_cycle_id should be None if a cycle just ended and is awaiting next cycle's start
            # Or, it might remain if the human review is considered part of the *same* logical cycle number, but a new phase.
            # For now, let's assume the cycle pending review is no longer "current" in terms of active processing.
            # The start_new_cycle method will set the new current_cycle_id.
            # If overall_project_status is just updated, current_cycle_id can remain as is, referring to the cycle that was just reviewed.
            # Let's clear current_cycle_id if we are truly between autonomous cycles.
            if next_cycle_status == "ready_for_next_cycle":
                 project_state.current_cycle_id = None # No cycle is actively running

            self._save_project_state()
            self.logger.info(f"Human feedback recorded for cycle '{cycle_id_to_update}'. Project status set to '{next_cycle_status}'.")
            return True

    def update_project_state_after_arca_review(
        self,
        arca_best_state_summary_doc_id: Optional[str],
        arca_overall_confidence: Optional[float], # Should be 0.0 to 1.0
        arca_issues_pending_review: List[Dict[str, Any]]
    ) -> None:
        """Updates project state with ARCA's end-of-cycle assessment before human review."""
        if not hasattr(self, '_project_state'):
            raise StatusFileError("Project state not initialized.")

        with self._get_lock():
            self._project_state.arca_best_state_summary_doc_id = arca_best_state_summary_doc_id
            if arca_overall_confidence is not None:
                if not (0.0 <= arca_overall_confidence <= 1.0):
                    self.logger.warning(f"ARCA overall confidence {arca_overall_confidence} out of range [0,1]. Clamping.")
                    arca_overall_confidence = max(0.0, min(1.0, arca_overall_confidence))
                self._project_state.arca_overall_confidence_in_best_state = arca_overall_confidence
            else:
                self._project_state.arca_overall_confidence_in_best_state = None # Explicitly set if None

            self._project_state.arca_issues_pending_human_review = arca_issues_pending_review
            
            self.logger.info(f"Project state updated after ARCA review for project {self._project_state.project_id}.")
            self._save_project_state()

    def record_human_review(self, review_record: HumanReviewRecord) -> None:
        """
        Records human review feedback and updates the project state accordingly.
        This method directly uses the logic defined in ProjectStateV2.record_human_review if available,
        or implements similar logic.
        """
        if not hasattr(self, '_project_state'):
            raise StatusFileError("Project state not initialized.")

        with self._get_lock():
            # Use the method from ProjectStateV2 instance if it exists and is robust
            # For now, let's replicate the logic here for clarity on StateManager's role
            
            self._project_state.last_human_review = review_record
            # Initial status during review
            # self._project_state.overall_status = ProjectOverallStatus.HUMAN_REVIEW_IN_PROGRESS 
            # This is handled by the calling orchestrator potentially. Let's set final status based on decision.

            decision = review_record.decision_for_next_step
            if decision == "PROCEED_TO_NEXT_AUTONOMOUS_PHASE":
                # This implies the current main phase (e.g. LOPRD gen) is done, and we move to next (e.g. Architecting)
                # The orchestrator will need to determine what "next phase" means and potentially call start_new_cycle with new objective
                self._project_state.overall_status = ProjectOverallStatus.AWAITING_NEXT_CYCLE_START 
            elif decision == "INITIATE_REFINEMENT_CYCLE":
                self._project_state.overall_status = ProjectOverallStatus.AWAITING_NEXT_CYCLE_START
                if review_record.next_cycle_objective_override:
                    self.logger.info(f"Next cycle objective overridden by human: {review_record.next_cycle_objective_override}")
                    # This objective will be used by the orchestrator when calling start_new_cycle
            elif decision == "MODIFY_PROJECT_GOAL":
                self._project_state.overall_status = ProjectOverallStatus.ANALYZING_GOAL # Or similar, needs re-analysis
                # This implies a significant reset. Orchestrator needs to handle this.
            elif decision == "PAUSE_PROJECT":
                self._project_state.overall_status = ProjectOverallStatus.PROJECT_PAUSED_BY_USER
            elif decision == "ARCHIVE_PROJECT_SUCCESS":
                self._project_state.overall_status = ProjectOverallStatus.PROJECT_COMPLETED_SUCCESSFULLY
            elif decision == "ARCHIVE_PROJECT_FAILURE":
                self._project_state.overall_status = ProjectOverallStatus.PROJECT_FAILED # Or a specific "ARCHIVED_FAILURE"

            self._project_state.next_action_determined_by = "HUMAN_REVIEWER"
            
            # Update accepted artifacts if implied by review (e.g. "LOPRD v2 approved")
            # This logic needs to be more sophisticated, potentially part of HumanReviewRecord
            # or handled by the orchestrator based on review content.
            # For example, if review_record.decision_for_next_step == "PROCEED..." and it refers to a specific LOPRD version.

            self.logger.info(f"Human review {review_record.review_id} recorded for project {self._project_state.project_id}. Decision: {decision}")
            self._save_project_state()

    # --- Utility methods to update latest accepted artifacts ---
    def update_latest_accepted_loprd(self, doc_id: str) -> None:
        with self._get_lock():
            self._project_state.latest_accepted_loprd_doc_id = doc_id
            self._save_project_state()
            self.logger.info(f"Updated latest_accepted_loprd_doc_id to {doc_id}")

    def update_latest_accepted_blueprint(self, doc_id: str) -> None:
        with self._get_lock():
            self._project_state.latest_accepted_blueprint_doc_id = doc_id
            self._save_project_state()
            self.logger.info(f"Updated latest_accepted_blueprint_doc_id to {doc_id}")

    def update_latest_accepted_master_plan(self, doc_id: str) -> None:
        with self._get_lock():
            self._project_state.latest_accepted_master_plan_doc_id = doc_id
            self._save_project_state()
            self.logger.info(f"Updated latest_accepted_master_plan_doc_id to {doc_id}")

    def update_latest_accepted_code_snapshot(self, snapshot_id: str) -> None:
        with self._get_lock():
            self._project_state.latest_accepted_code_snapshot_id = snapshot_id
            self._save_project_state()
            self.logger.info(f"Updated latest_accepted_code_snapshot_id to {snapshot_id}")
            
    def update_latest_test_report_summary(self, doc_id: str) -> None:
        with self._get_lock():
            self._project_state.latest_test_report_summary_doc_id = doc_id
            self._save_project_state()
            self.logger.info(f"Updated latest_test_report_summary_doc_id to {doc_id}")

    def update_latest_project_readme(self, doc_id: str) -> None:
        with self._get_lock():
            self._project_state.latest_project_readme_doc_id = doc_id
            self._save_project_state()
            self.logger.info(f"Updated latest_project_readme_doc_id to {doc_id}")

    # <<< ADDED FLOW AND STAGE RECORDING METHODS >>>

    def record_flow_start(
        self, 
        run_id: str, 
        flow_id: str, 
        initial_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Records the start of a flow execution."""
        with self._get_lock():
            if not hasattr(self, '_project_state'):
                self.logger.error("Cannot record flow start: Project state not initialized.")
                return

            # --- DEBUGGING: Log type and attributes of _project_state --- #
            self.logger.info(f"DEBUG: Type of self._project_state in record_flow_start: {type(self._project_state)}")
            try:
                self.logger.info(f"DEBUG: dir(self._project_state) in record_flow_start: {dir(self._project_state)}")
                self.logger.info(f"DEBUG: self._project_state.model_fields.keys(): {self._project_state.model_fields.keys()}")
            except Exception as e_debug:
                self.logger.error(f"DEBUG: Error inspecting self._project_state: {e_debug}")
            # --- END DEBUGGING --- #

            now = datetime.now(timezone.utc)
            # Ensure run_history exists
            if self._project_state.run_history is None: # THIS IS THE LINE CAUSING AttributeError
                self._project_state.run_history = {}
            
            # Create a RunRecord instance
            new_run_record = RunRecord(
                run_id=run_id,
                flow_id=flow_id,
                start_time=now,
                status=StageStatus.RUNNING, # Use the enum member
                initial_context_summary={k: str(type(v)) for k, v in initial_context.items()} if initial_context else {},
                stages=[] # Initialize with an empty list of StageRecord
            )
            
            self._project_state.run_history[run_id] = new_run_record # Assign the model instance
            self._project_state.last_updated = now
            self._save_project_state()
            self.logger.info(f"Flow run {run_id} (flow: {flow_id}) started.")

    def record_flow_end(
        self, 
        run_id: str, 
        flow_id: str, # flow_id added for consistency, though run_id is primary key
        final_status: str, # e.g., "COMPLETED_SUCCESS", "COMPLETED_FAILURE" from StageStatus enum .value
        error_message: Optional[str] = None,
        final_outputs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Records the end of a flow execution."""
        with self._get_lock():
            # Defensive check and potential re-initialization of run_history
            if not hasattr(self, '_project_state') or not self._project_state: # Ensure _project_state itself exists
                self.logger.error(f"Project state not loaded. Cannot record flow end for run_id {run_id}.")
                # Attempt to load/init to prevent further errors, though this particular call might still fail to find the run.
                self._load_or_initialize_project_state() # Ensure _project_state is loaded
                if not hasattr(self, '_project_state') or not self._project_state: # Still not there?
                     self.logger.critical(f"CRITICAL: _project_state is None even after _load_or_initialize_project_state in record_flow_end for {run_id}.")
                     return # Cannot proceed

            if not hasattr(self._project_state, 'run_history') or self._project_state.run_history is None:
                self.logger.warning(f"_project_state.run_history is missing or None for run_id {run_id} during record_flow_end. Initializing to empty dict.")
                self._project_state.run_history = {} # Initialize if missing

            run_record_obj = self._project_state.run_history.get(run_id)
            if not run_record_obj:
                self.logger.error(f"Cannot record flow end for run_id {run_id}: Run ID not found in run_history. State might be inconsistent.")
                return
            
            # Ensure run_record_obj is a RunRecord instance
            if not isinstance(run_record_obj, RunRecord):
                self.logger.warning(f"Run record for {run_id} is not a RunRecord instance (type: {type(run_record_obj)}). Attempting to re-validate.")
                try:
                    run_record_obj = RunRecord(**run_record_obj) # If it was a dict, try to convert
                except Exception as e_revalidate:
                    self.logger.error(f"Failed to re-validate run record {run_id} to RunRecord: {e_revalidate}")
                    return
            
            now = datetime.now(timezone.utc)
            run_record_obj.status = StageStatus(final_status) # Convert string to StageStatus enum
            run_record_obj.end_time = now
            run_record_obj.error_message = error_message
            run_record_obj.final_outputs_summary = {k: str(type(v)) for k, v in final_outputs.items()} if final_outputs else {}
            
            self._project_state.last_updated = now
            self._save_project_state()
            self.logger.info(f"Flow run {run_id} (flow: {flow_id}) ended with status: {final_status}.")

    def record_stage_start(
        self, 
        run_id: str, 
        flow_id: str, # For context
        stage_id: str, 
        agent_id: str
    ) -> None:
        """Records the start of a stage execution within a flow run."""
        with self._get_lock():
            run_record_obj = self._project_state.run_history.get(run_id) if hasattr(self, '_project_state') and self._project_state.run_history else None
            if not run_record_obj:
                self.logger.error(f"Cannot record stage start for run_id {run_id}, stage {stage_id}: Flow run not found.")
                return
            
            if not isinstance(run_record_obj, RunRecord):
                self.logger.warning(f"Run record for {run_id} is not a RunRecord instance (type: {type(run_record_obj)}) when recording stage start. Attempting re-validation.")
                try:
                    run_record_obj = RunRecord(**run_record_obj)
                    self._project_state.run_history[run_id] = run_record_obj # Update with validated model
                except Exception as e_revalidate:
                    self.logger.error(f"Failed to re-validate run record {run_id} to RunRecord for stage start: {e_revalidate}")
                    return

            now = datetime.now(timezone.utc)
            
            # Find if stage already exists (for retries, we update, otherwise append)
            existing_stage_index = -1
            for i, sr in enumerate(run_record_obj.stages):
                if sr.stage_id == stage_id:
                    existing_stage_index = i
                    break
            
            if existing_stage_index != -1:
                # This is likely a retry, update existing stage record
                stage_to_update = run_record_obj.stages[existing_stage_index]
                stage_to_update.status = StageStatus.RUNNING
                stage_to_update.start_time = now # Update start time for the retry
                stage_to_update.end_time = None
                stage_to_update.error_details = None
                # stage_to_update.attempt_count +=1 # attempt_count is not in StageRecord, was in old dict structure
                self.logger.info(f"Updating existing stage record for {stage_id} (retry) in run {run_id}.")
            else:
                # New stage execution
                new_stage_record = StageRecord(
                    stage_id=stage_id,
                    agent_id=agent_id,
                    start_time=now,
                    status=StageStatus.RUNNING
                )
                run_record_obj.stages.append(new_stage_record)

            self._project_state.last_updated = now
            self._save_project_state()
            self.logger.info(f"Stage {stage_id} (agent: {agent_id}) started for run {run_id}.")

    def record_stage_end(
        self,
        run_id: str,
        flow_id: str, # For context
        stage_id: str,
        status: str, # e.g., "COMPLETED_SUCCESS", "COMPLETED_FAILURE" from StageStatus enum .value
        outputs: Optional[Any] = None,
        error_details: Optional[Dict[str, Any]] = None # From AgentErrorDetails.model_dump()
    ) -> None:
        """Records the end of a stage execution."""
        with self._get_lock():
            run_record_obj = self._project_state.run_history.get(run_id) if hasattr(self, '_project_state') and self._project_state.run_history else None

            if not run_record_obj: # Check if run_record_obj is None first
                self.logger.error(f"Cannot record stage end for run_id {run_id}, stage {stage_id}: Flow run not found.")
                return

            # Ensure it is a RunRecord instance
            if not isinstance(run_record_obj, RunRecord):
                self.logger.warning(f"Run record for {run_id} is not a RunRecord instance (type: {type(run_record_obj)}) when recording stage end. Attempting re-validation.")
                try:
                    run_record_obj = RunRecord(**run_record_obj) # Try to convert if it's a dict
                    self._project_state.run_history[run_id] = run_record_obj # Update with validated model
                except Exception as e_revalidate:
                    self.logger.error(f"Failed to re-validate run record {run_id} to RunRecord for stage end: {e_revalidate}")
                    return

            # Find the stage record in the list
            target_stage_record: Optional[StageRecord] = None
            for sr in run_record_obj.stages:
                if sr.stage_id == stage_id:
                    target_stage_record = sr
                    break
            
            if not target_stage_record:
                self.logger.error(f"Cannot record stage end for run_id {run_id}, stage {stage_id}: Stage not found in run's stage list.")
                return

            now = datetime.now(timezone.utc)
            
            target_stage_record.status = StageStatus(status) # Convert string to StageStatus enum
            target_stage_record.end_time = now
            if outputs is not None:
                 target_stage_record.outputs_summary = {"type": str(type(outputs)), "value_preview": str(outputs)[:200]}
            target_stage_record.error_details = error_details # This is already a Dict[str, Any]
            
            self._project_state.last_updated = now
            self._save_project_state()
            self.logger.info(f"Stage {stage_id} for run {run_id} ended with status: {status}.")

    def update_run_context(self, run_id: str, current_context_outputs: Dict[str, Any], current_artifact_references: Dict[str, List[str]]) -> None:
        """Updates the persisted context snapshot for an ongoing run."""
        # This is a placeholder for more sophisticated context snapshotting.
        # For now, it might not write to ProjectStateV2 directly unless specific fields are added.
        # It could write to a separate run-specific file or log.
        self.logger.debug(f"Run context update for {run_id}: {len(current_context_outputs)} output keys, {len(current_artifact_references)} artifact stages.")
        # If ProjectStateV2.run_history[run_id] is to store the full context, it needs a field for it.
        # For now, this method serves as a hook.

# Example usage (for testing or demonstration)
if __name__ == "__main__":
    # Setup basic logging for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_proj_dir = Path("./test_project")
    test_proj_dir.mkdir(exist_ok=True)
    # Create a dummy stages dir for init
    dummy_stages = Path("./dummy_stages")
    dummy_stages.mkdir(exist_ok=True)

    sm = StateManager(str(test_proj_dir), str(dummy_stages), use_locking=True)

    # Test basic status update
    print("\n--- Testing Status Update ---")
    update_success = sm.update_status(0, "PASS", ["artifact1.txt"], "Initial setup", "Initial reflection text")
    print(f"Update 1 successful: {update_success}")
    update_success = sm.update_status(1, "FAIL", ["artifact2.py"], "Failed validation", "Failed reflection text")
    print(f"Update 2 successful: {update_success}")

    # Test status retrieval
    print("\n--- Testing Status Retrieval ---")
    last_status = sm.get_last_status()
    print(f"Last Status: {last_status}")
    full_status = sm.get_full_status()
    print(f"Full Status: {json.dumps(full_status, indent=2)}")

    # Test next stage logic
    print("\n--- Testing Next Stage Logic ---")
    next_stage = sm.get_next_stage()
    print(f"Next Stage: {next_stage}") # Expected: 1 (retry after fail)

    # Clean up dummy files/dirs if needed (example)
    # import shutil
    # shutil.rmtree(test_proj_dir)
    # shutil.rmtree(dummy_stages)

    # Test ChromaDB interactions (requires running ChromaDB server)
    print("\n--- Testing ChromaDB Interactions (Ensure Chroma Server is running) ---")
    try:
        # Store artifact
        store_success = sm.store_artifact_context_in_chroma(
            stage_number=0.0,
            rel_path="src/main.py",
            content="print('Hello World')",
            artifact_type="code"
        )
        print(f"Store Artifact Success: {store_success}")

        # Store reflection
        reflection_to_process = "Stage 1 validation failed due to unclear requirements."
        reflection_stored = sm.persist_reflections_to_chroma(
            run_id=None,
            stage_number=float(1.0),
            reflections=reflection_to_process,
        )
        print(f"Persist Reflection Success: {reflection_stored}")

        # Get artifact context
        print("\nQuerying Artifact Context:")
        context_results = sm.get_artifact_context_from_chroma(query="hello world code")
        pprint(context_results)

        # Get reflection context
        print("\nQuerying Reflection Context:")
        reflection_results = sm.get_reflection_context_from_chroma(query="validation failure")
        pprint(reflection_results)

        # List metadata
        print("\nListing Artifact Metadata:")
        metadata_list = sm.list_artifact_metadata(artifact_type_filter="code")
        pprint(metadata_list)

    except ChromaOperationError as coe:
        print(f"ChromaDB Operation Error: {coe}")
    except Exception as e:
        print(f"General error during ChromaDB test: {e}")

    # Clean up
    try:
        import shutil
        shutil.rmtree(test_proj_dir)
        shutil.rmtree(dummy_stages)
        print("\nCleaned up test directories.")
    except Exception as e:
        print(f"Error during cleanup: {e}")
