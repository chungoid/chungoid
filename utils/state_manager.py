"""Manages the project status file (project_status.json) with file locking."""

import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from logging.handlers import RotatingFileHandler
import hashlib
import uuid

import filelock  # Ensure filelock is installed
from . import chroma_utils


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
        self.chungoid_dir = self.target_dir_path / ".chungoid"
        self.status_file_path = str(self.chungoid_dir / "project_status.json")
        self.lock_file_path = f"{self.status_file_path}.lock"
        self.use_locking = use_locking
        self.server_stages_dir = Path(server_stages_dir).resolve()  # Ensure resolved Path

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
            self.chungoid_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug("Ensured .chungoid directory exists: %s", self.chungoid_dir)
        except OSError as e:
            self.logger.error(
                "Failed to create .chungoid directory at %s: %s", self.chungoid_dir, e
            )
            raise StatusFileError(f"Failed to create .chungoid directory: {e}") from e

        # Verify server_stages_dir exists during init
        if not self.server_stages_dir.is_dir():
            self.logger.error(
                "Server stages directory not found or not a directory: %s", self.server_stages_dir
            )
            # This is critical for finding stages, raise error
            raise ValueError(
                f"Server stages directory not found or not a directory: {server_stages_dir}"
            )

        # Setup lock
        if self.use_locking:
            self._lock = filelock.FileLock(self.lock_file_path, timeout=10)
            self.logger.debug("Using real file lock: %s", self.lock_file_path)
        else:
            self._lock = DummyFileLock()
            self.logger.debug("File locking is disabled.")

        self.logger.info("StateManager initialized for file: %s", self.status_file_path)

        # Validate status file on init, but don't attempt ChromaDB connection here
        try:
            _ = self._read_status_file()  # Read to check format
            self.logger.debug("Initial status file read successful.")
        except StatusFileError as e:
            self.logger.warning(
                "Initial status file validation failed: %s. Assuming empty/will be created.", e
            )
        except Exception as e:
            self.logger.exception("Unexpected error during StateManager init read: %s", e)
            raise StatusFileError(f"Unexpected error reading status file on init: {e}") from e

        # Remove or comment out the deferred connection log message
        # self.logger.debug("ChromaDB client connection deferred to first use.")

    def _get_lock(self) -> Union[filelock.FileLock, DummyFileLock]:
        """Returns the appropriate lock object based on configuration."""
        return self._lock

    def _read_status_file(self) -> Dict[str, List[Dict[str, Any]]]:
        """Reads the status file content (expecting {'runs': [...]}).

        Acquires a lock if locking is enabled.

        Returns:
            A dictionary containing the 'runs' list. Returns {'runs': []} if file empty/missing.

        Raises:
            StatusFileError: If the file is unreadable or contains invalid JSON not matching the expected structure.
        """
        lock = self._get_lock()
        try:
            with lock:
                self.logger.debug("Attempting to read status file: %s", self.status_file_path)
                if not os.path.exists(self.status_file_path):
                    self.logger.warning(
                        "Status file not found at %s, returning empty structure {'runs': []}.",
                        self.status_file_path,
                    )
                    return {"runs": []}  # Return default structure

                with open(self.status_file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        self.logger.warning(
                            "Status file is empty, returning empty structure {'runs': []}."
                        )
                        return {"runs": []}  # Return default structure

                    data = json.loads(content)
                    # Validate new structure
                    if (
                        not isinstance(data, dict)
                        or "runs" not in data
                        or not isinstance(data["runs"], list)
                    ):
                        raise StatusFileError(
                            f"Status file content is not a valid JSON object with a 'runs' list. Found: {type(data)}"
                        )
                    # Optional: Validate structure of each run/status entry here if needed
                    self.logger.debug(
                        "Successfully read and parsed status file with {'runs': ...} structure."
                    )
                    return data
        except filelock.Timeout:
            self.logger.error("Timeout occurred while waiting for status file lock.")
            raise StatusFileError("Could not acquire lock to read status file.")
        except json.JSONDecodeError as e:
            self.logger.error("Failed to decode JSON from status file: %s", e)
            raise StatusFileError(f"Invalid JSON in status file: {e}") from e
        except IOError as e:
            self.logger.error("Failed to read status file: %s", e)
            raise StatusFileError(f"Could not read status file: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during read
            self.logger.exception("Unexpected error reading status file: %s", e)
            raise StatusFileError(f"Unexpected error reading status file: {e}") from e

    def _write_status_file(self, data: Dict[str, List[Dict[str, Any]]]):
        """Writes the status data (expecting {'runs': [...]}) to the file.

        Requires acquiring the file lock.

        Args:
            data: The dictionary containing the 'runs' list to write.

        Raises:
            StatusFileError: If the lock cannot be acquired or writing fails.
        """
        lock = self._get_lock()
        try:
            with lock:
                self.logger.debug("Attempting to write to status file: %s", self.status_file_path)
                # Validate structure before writing
                if (
                    not isinstance(data, dict)
                    or "runs" not in data
                    or not isinstance(data["runs"], list)
                ):
                    err_msg = f"Invalid data structure provided to _write_status_file. Expected {{'runs': [...]}}, got: {type(data)}"
                    self.logger.error(err_msg)
                    raise StatusFileError(err_msg)

                with open(self.status_file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)  # Use indent for readability
                    f.flush()
                    os.fsync(f.fileno())
                self.logger.debug("Successfully wrote to status file (post-dump, pre-release).")
            self.logger.debug("Lock released after write.")

            try:
                with open(self.status_file_path, "r", encoding="utf-8") as f_verify:
                    content_after_write = f_verify.read()
                self.logger.debug(
                    "Read file back immediately after write. Content: %s", content_after_write
                )
            except Exception as read_err:
                self.logger.error("Error reading file back immediately after write: %s", read_err)

        except filelock.Timeout:
            self.logger.error("Timeout occurred while waiting for status file lock.")
            raise StatusFileError("Could not acquire lock to write status file.")
        except IOError as e:
            self.logger.error("Failed to write to status file: %s", e)
            raise StatusFileError(f"Could not write status file: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during write
            self.logger.exception("Unexpected error writing status file: %s", e)
            raise StatusFileError(f"Unexpected error writing status file: {e}") from e

    def get_last_status(self) -> Optional[Dict[str, Any]]:
        """Gets the status entry for the highest stage in the *latest run*.

        Finds the run with the highest `run_id` and returns the last status update
        within that run's `status_updates` list.

        Returns:
            The last status dictionary from the latest run, or None if no runs or statuses exist.
        """
        try:
            status_data = self._read_status_file()  # Reads {'runs': [...]} structure
            runs = status_data.get("runs", [])
            if not runs:
                self.logger.info("get_last_status: No runs found in status file.")
                return None

            # Find the latest run (highest run_id)
            latest_run = max(
                runs, key=lambda r: r.get("run_id", -1)
            )  # Handle missing run_id defensively
            latest_run_id = latest_run.get("run_id", "unknown")

            status_updates = latest_run.get("status_updates", [])
            if not status_updates:
                self.logger.info(
                    f"get_last_status: No status updates found in latest run (id: {latest_run_id})."
                )
                return None  # No statuses within the latest run

            # Assuming updates are appended, the last one is the latest status for this run
            last_status_in_run = status_updates[-1]
            self.logger.debug(
                f"get_last_status: Found last status in run {latest_run_id}: {last_status_in_run}"
            )
            return last_status_in_run

        except StatusFileError as e:
            self.logger.error("Could not get last status due to StatusFileError: %s", e)
            return None
        except Exception as e:
            self.logger.exception("Unexpected error in get_last_status: %s", e)
            return None

    def get_full_status(self) -> Dict[str, List[Dict[str, Any]]]:
        """Gets the entire status structure (including all runs).

        Returns:
            The full status dictionary { 'runs': [...] }.

        Raises:
            StatusFileError: If the status file cannot be read or parsed.
        """
        # This re-raises StatusFileError if reading fails
        self.logger.debug("get_full_status: Reading entire status structure.")
        status_data = self._read_status_file()
        self.logger.debug(f"get_full_status: Returning data: {status_data}")
        return status_data

    def get_next_stage(self) -> Optional[Union[int, float]]:
        """Determines the next stage to be executed based on the *latest run*.

        Logic:
        - Reads the status file and finds the latest run.
        - Finds the highest stage number marked as DONE or PASS in that run.
        - Lists available stage definition files (stage*.yaml) in server_stages_dir.
        - Returns the lowest available stage number greater than the highest completed stage.
        - Returns 0 if no runs or no completed stages exist in the latest run.

        Returns:
            The next stage number (int or float), or None if an error occurs.
        """
        self.logger.info("Determining next stage based on available definitions and latest run...")
        highest_completed_stage = -1  # Start before stage 0

        try:
            status_data = self._read_status_file()  # Reads {'runs': [...]} structure
            runs = status_data.get("runs", [])

            if not runs:
                self.logger.info("get_next_stage: No runs found. Starting with Stage 0.")
                highest_completed_stage = -1  # Ensures stage 0 is next
            else:
                # Find the latest run
                latest_run = max(runs, key=lambda r: r.get("run_id", -1))
                latest_run_id = latest_run.get("run_id", "unknown")
                self.logger.debug(f"get_next_stage: Examining latest run (id: {latest_run_id}).")
                status_updates = latest_run.get("status_updates", [])

                # Find highest completed stage *within the latest run*
                for update in status_updates:
                    stage = update.get("stage")
                    status = update.get("status")
                    if isinstance(stage, (int, float)) and status in ["DONE", "PASS"]:
                        highest_completed_stage = max(highest_completed_stage, stage)
                self.logger.info(
                    f"get_next_stage: Highest completed stage in run {latest_run_id} is {highest_completed_stage}."
                )

        except StatusFileError as e:
            self.logger.error("Could not read status file to determine next stage: %s", e)
            # If we can't read status, we can't proceed.
            raise RuntimeError(
                f"Failed to determine next stage due to status file error: {e}"
            ) from e
        except Exception as e:
            self.logger.exception("Unexpected error reading status for next stage: %s", e)
            raise RuntimeError(f"Unexpected error reading status for next stage: {e}") from e

        # List available stage files
        try:
            available_stages = []
            # server_stages_dir should be validated during __init__
            for item in self.server_stages_dir.glob("stage*.yaml"):
                if item.is_file():
                    try:
                        # Extract stage number from filename (e.g., stage0.yaml, stage3.5.yaml)
                        stage_str = item.stem.replace("stage", "")
                        stage_num = float(stage_str)
                        available_stages.append(stage_num)
                    except ValueError:
                        self.logger.warning(
                            f"Could not parse stage number from filename: {item.name}"
                        )

            if not available_stages:
                self.logger.error(f"No stage definition files found in {self.server_stages_dir}")
                raise RuntimeError("No stage definition files (stage*.yaml) found.")

            available_stages.sort()
            self.logger.debug(f"Available stage definitions found: {available_stages}")

        except Exception as e:
            self.logger.exception("Error listing available stage files: %s", e)
            raise RuntimeError(f"Error listing available stage files: {e}") from e

        # Find the lowest available stage greater than the highest completed
        next_stage = None
        for stage in available_stages:
            if stage > highest_completed_stage:
                next_stage = stage
                break  # Found the first applicable next stage

        if next_stage is not None:
            self.logger.info(f"Next stage determined: {next_stage}")
            return next_stage
        else:
            # This condition means all defined stages are completed or no stages are defined > highest_completed
            self.logger.info(
                f"No available stage found greater than highest completed stage ({highest_completed_stage}). Project may be complete."
            )
            # We should signal completion or an issue here. Raising error is safer for now.
            raise RuntimeError(
                "Cannot determine next stage: All defined stages appear complete or no higher stage definition exists."
            )

    def update_status(
        self,
        stage: Union[int, float],
        status: str,
        artifacts: List[str],
        reason: Optional[str] = None,
    ) -> bool:
        """Updates the status file by adding a new entry to the *latest run*.

        Creates the first run (run_id: 0) if no runs exist.
        Validates input status.
        Adds a timestamp to the new entry.
        Writes the updated list back to the file.

        Args:
            stage: The stage number (e.g., 1.0, 2).
            status: The result status (e.g., "PASS", "FAIL", "DONE").
            artifacts: A list of relative paths to generated artifacts.
            reason: Optional reason, e.g., for FAIL status.

        Returns:
            True if the update was successful, False otherwise.
        """
        self.logger.info(f"Attempting to update status for Stage: {stage}, Status: {status}")
        valid_statuses = ["PASS", "FAIL", "DONE"]
        if status.upper() not in valid_statuses:
            self.logger.error(f"Invalid status provided: {status}. Must be one of {valid_statuses}")
            return False

        # Create the new status entry
        new_entry: Dict[str, Any] = {
            "stage": float(stage),
            "status": status.upper(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts if isinstance(artifacts, list) else [],
        }
        if reason:
            new_entry["reason"] = reason

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
                latest_run = max(runs, key=lambda r: r.get("run_id", -1))
                target_run_id = latest_run.get("run_id", -1) # Get the ID of the latest run

                # --- Add validation for existing target_run_id --- #
                if target_run_id == -1 or not isinstance(target_run_id, int):
                    # This indicates a malformed status file if runs exist but no valid run_id
                    err_msg = f"Could not determine a valid integer run_id for the latest run in status file. Found: {target_run_id}"
                    self.logger.error(err_msg)
                    raise StatusFileError(err_msg)
                # --- End validation ---

                # Find the index of the latest run to modify it
                for i, run in enumerate(runs):
                    if run.get("run_id") == target_run_id:
                        target_run_index = i
                        break

                if target_run_index == -1:
                    # Should not happen if target_run_id was found
                    raise StatusFileError(
                        f"Internal error: Could not find index for run_id {target_run_id}."
                    )

                # Append the new status to the latest run's updates
                if "status_updates" not in runs[target_run_index]:
                    runs[target_run_index][
                        "status_updates"
                    ] = []  # Initialize if missing (defensive)
                runs[target_run_index]["status_updates"].append(new_entry)
                self.logger.info(f"Appending status update to latest run (id: {target_run_id}).")

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
    update_success = sm.update_status(0, "PASS", ["artifact1.txt"], "Initial setup")
    print(f"Update 1 successful: {update_success}")
    update_success = sm.update_status(1, "FAIL", ["artifact2.py"], "Failed validation")
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
