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

import filelock  # Ensure filelock is installed
from . import chroma_utils


class StatusFileError(Exception):
    """Custom exception for errors related to status file operations."""

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
        self.chroma_client: Optional[chromadb.ClientAPI] = (
            None  # Keep Optional for type hinting if init fails
        )
        try:
            self.logger.info("Attempting synchronous ChromaDB client initialization...")
            self.chroma_client = chroma_utils.get_chroma_client()
            if self.chroma_client:
                self.logger.info("Synchronous ChromaDB client initialization successful.")
            else:
                self.logger.error("Synchronous ChromaDB client initialization returned None.")
        except Exception as e:
            self.logger.error(
                f"Exception during synchronous ChromaDB client initialization: {e}", exc_info=True
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
                # First run ever, create it
                target_run_id = 0
                new_run = {
                    "run_id": target_run_id,
                    "start_timestamp": datetime.now(timezone.utc).isoformat(),  # Record start time
                    "status_updates": [new_entry],  # Add first entry
                }
                runs.append(new_run)
                self.logger.info(
                    f"Creating first run (id: {target_run_id}) and adding status update."
                )
            else:
                # Find the latest run
                latest_run = max(runs, key=lambda r: r.get("run_id", -1))
                target_run_id = latest_run.get("run_id", -1)
                if target_run_id == -1:
                    # Should not happen if runs is not empty and runs have run_id
                    raise StatusFileError("Could not determine target run_id from existing runs.")

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
        """Internal helper to get the *already initialized* ChromaDB client synchronously."""
        # This method might now just return self.chroma_client if init always runs
        # Or it could retain the logic as a fallback/check, but it shouldn't re-initialize.
        # For now, let's just return the instance variable.
        if self.chroma_client is None:
            self.logger.warning(
                "_get_chroma_client called, but self.chroma_client is None (initialization might have failed)."
            )
        return self.chroma_client
        # Original logic commented out, as init should handle it now.
        # try:
        #     client = chroma_utils.get_chroma_client() # <--- Removed await
        #     if client is None:
        #         self.logger.error("Failed to get ChromaDB client from chroma_utils.")
        #         return None
        #     return client
        # except Exception as e:
        #     self.logger.error(f"Exception while getting ChromaDB client: {e}", exc_info=True)
        #     return None

    def store_artifact_context_in_chroma(
        self,
        stage_number: float,
        rel_path: str,
        content: str,
        artifact_type: str = "unknown",
    ) -> bool:
        """Stores a single artifact's content and metadata in ChromaDB synchronously."""
        self.logger.debug(
            f"Storing artifact context. Stage: {stage_number}, Path: {rel_path}, Type: {artifact_type}"
        )
        try:
            # Get client synchronously first
            client = self._get_chroma_client()
            if not client:
                self.logger.error("ChromaDB client not available. Cannot store artifact context.")
                return False

            self.logger.info(
                f"Getting or creating collection: {self._CONTEXT_COLLECTION_NAME} with default embedding function."
            )
            # Use the default embedding function from sentence-transformers
            embed_func = embedding_functions.DefaultEmbeddingFunction()
            collection = client.get_or_create_collection(
                name=self._CONTEXT_COLLECTION_NAME,
                embedding_function=embed_func,
            )
            self.logger.info("Collection obtained.")

            doc_id = f"artifact_{stage_number}_{rel_path}"

            # --- Generate Description and Keywords --- #
            description = (
                f"Artifact of type '{artifact_type}' for stage {stage_number}. Path: {rel_path}"
            )
            keywords = ["artifact", artifact_type, f"stage_{stage_number}"]
            # Add keywords based on path components if desired
            try:
                keywords.extend(Path(rel_path).parts)
                keywords.append(Path(rel_path).name)
            except Exception:  # Catch potential path parsing errors
                pass
            # --- End Generation --- #

            metadata = {
                "type": "artifact",  # Core type for filtering
                "stage": stage_number,
                "path": rel_path,
                "artifact_type": artifact_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": description,  # Added field
                # Convert keywords list to space-separated string for storage
                "keywords": " ".join(sorted(list(set(kw.lower() for kw in keywords if kw)))),
            }
            self.logger.debug(f"Prepared doc_id: {doc_id}, metadata keys: {list(metadata.keys())}")

            # <<< ADDED DIAGNOSTIC LOGGING >>>
            self.logger.debug(f"Preparing to call collection.add() for doc_id: {doc_id}")
            self.logger.debug(f"  Metadata keys: {list(metadata.keys())}")
            # Log content snippet or length for verification
            content_log = content[:100] + "..." if len(content) > 100 else content
            self.logger.debug(f"  Content length: {len(content)}, Snippet: '{content_log}'")
            # <<< END DIAGNOSTIC LOGGING >>>

            # --- Specific logging and error handling for collection.add() ---
            add_successful = False
            try:
                self.logger.info(f"Attempting collection.add() for doc_id: {doc_id}")
                # The actual add call - Pass single values
                collection.add(
                    documents=content,  # Pass single string
                    metadatas=metadata,  # Pass single dict
                    ids=doc_id,  # Pass single string
                )
                self.logger.info(f"collection.add() completed for doc_id: {doc_id}")
                add_successful = True  # Mark success
            except Exception as add_err:
                # Log the specific error from collection.add()
                self.logger.error(
                    f"Error during collection.add() for doc_id {doc_id}: {add_err}",
                    exc_info=True,
                )
                # Do not proceed, failure will be returned below
            # --- End specific handling ---

            if add_successful:
                self.logger.info(f"Successfully stored artifact context for {doc_id}")
                return True
            else:
                # Error already logged in the inner except block
                self.logger.warning(
                    f"Returning False from store_artifact_context_in_chroma due to collection.add() error for {doc_id}."
                )
                return False

        except Exception as e:
            # Catch other errors (getting client, getting collection, etc.)
            self.logger.error(
                f"Outer exception in store_artifact_context_in_chroma for {rel_path}: {e}",
                exc_info=True,
            )
            return False

    def get_artifact_context_from_chroma(
        self,
        query: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves artifact context from ChromaDB based on a semantic query (synchronous)."""
        self.logger.debug(
            f"Attempting to retrieve artifact context from ChromaDB. Query: '{query}', n_results: {n_results}, filter: {where_filter}"
        )
        results = []
        # Get client synchronously first
        client = self._get_chroma_client()
        if not client:
            self.logger.warning("ChromaDB client not available. Cannot retrieve artifact context.")
            return results

        final_where_filter = {"type": "artifact"}  # Always filter by type=artifact
        if where_filter:
            final_where_filter.update(where_filter)
        self.logger.debug(f"Using where filter: {final_where_filter}")

        try:
            self.logger.info("Getting collection with embedding function for query...")
            embed_func = embedding_functions.DefaultEmbeddingFunction()
            collection = client.get_or_create_collection(
                name=self._CONTEXT_COLLECTION_NAME, embedding_function=embed_func
            )
            self.logger.info(f"Querying collection '{self._CONTEXT_COLLECTION_NAME}'...")
            query_results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=final_where_filter,
                include=["metadatas", "documents", "distances"],
            )
            self.logger.info(
                f"Query completed. Found {len(query_results.get('ids', [[]])[0])} results."
            )

            # Process results
            ids = query_results.get("ids", [[]])[0]
            documents = query_results.get("documents", [[]])[0]
            metadatas = query_results.get("metadatas", [[]])[0]
            distances = query_results.get("distances", [[]])[0]

            for i, doc_id in enumerate(ids):
                results.append(
                    {
                        "id": doc_id,
                        "content": documents[i],
                        "metadata": metadatas[i],
                        "distance": distances[i],
                    }
                )

        except Exception as e:
            self.logger.error(
                f"Failed to retrieve artifact context from ChromaDB: {e}", exc_info=True
            )
            # Return empty list on error

        return results

    def persist_reflections_to_chroma(
        self, run_id: int, stage_number: float, reflections: str
    ) -> bool:
        """Persists reflections for a given stage to ChromaDB synchronously."""
        self.logger.debug(
            f"Attempting to persist reflections for run {run_id}, stage {stage_number}"
        )
        client = self._get_chroma_client()
        if not client:
            self.logger.warning("ChromaDB client not available. Cannot persist reflections.")
            return False

        try:
            self.logger.info("Getting or creating reflection collection with embedding function...")
            embed_func = embedding_functions.DefaultEmbeddingFunction()
            collection = client.get_or_create_collection(
                name=self._REFLECTIONS_COLLECTION_NAME,  # Use dedicated collection
                embedding_function=embed_func,
            )
            self.logger.info("Reflection collection obtained.")

            doc_id = f"reflection_{run_id}_{stage_number}"
            metadata = {
                "type": "reflection",
                "run_id": run_id,
                "stage": stage_number,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self.logger.info(f"Adding reflection document: {doc_id}")
            collection.add(documents=[reflections], metadatas=[metadata], ids=[doc_id])
            self.logger.info(f"Successfully persisted reflection for {doc_id}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to persist reflections to ChromaDB for run {run_id}, stage {stage_number}: {e}",
                exc_info=True,
            )
            return False

    def get_reflection_context_from_chroma(
        self,
        query: str,
        n_results: int = 3,
        filter_stage_min: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves relevant reflections from ChromaDB based on a semantic query (synchronous)."""
        self.logger.debug(
            f"Attempting to retrieve reflection context. Query: '{query}', n_results: {n_results}, stage_min: {filter_stage_min}"
        )
        results = []
        client = self._get_chroma_client()
        if not client:
            self.logger.warning("ChromaDB client not available. Cannot retrieve reflections.")
            return results

        where_filter = {"type": "reflection"}  # Always filter by type=reflection
        if filter_stage_min is not None:
            where_filter["stage"] = {"$gte": filter_stage_min}
        self.logger.debug(f"Using where filter for reflections: {where_filter}")

        try:
            self.logger.info("Getting reflection collection with embedding function for query...")
            embed_func = embedding_functions.DefaultEmbeddingFunction()
            collection = client.get_or_create_collection(
                name=self._REFLECTIONS_COLLECTION_NAME, embedding_function=embed_func
            )
            self.logger.info(
                f"Querying reflection collection '{self._REFLECTIONS_COLLECTION_NAME}'..."
            )
            query_results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["metadatas", "documents", "distances"],
            )
            self.logger.info(
                f"Reflection query completed. Found {len(query_results.get('ids', [[]])[0])} results."
            )

            # Process results
            ids = query_results.get("ids", [[]])[0]
            documents = query_results.get("documents", [[]])[0]
            metadatas = query_results.get("metadatas", [[]])[0]
            distances = query_results.get("distances", [[]])[0]

            for i, doc_id in enumerate(ids):
                results.append(
                    {
                        "id": doc_id,
                        "reflection": documents[i],
                        "metadata": metadatas[i],
                        "distance": distances[i],
                    }
                )

        except Exception as e:
            self.logger.error(f"Failed to retrieve reflections from ChromaDB: {e}", exc_info=True)
            # Return empty list on error

        return results

    def list_artifact_metadata(
        self,
        stage_filter: Optional[float] = None,
        artifact_type_filter: Optional[str] = None,
        limit: Optional[int] = None,  # Optional limit on number of results
    ) -> List[Dict[str, Any]]:
        """Retrieves artifact metadata based on filters, without semantic search."""
        self.logger.debug(
            f"Attempting to list artifact metadata. Stage Filter: {stage_filter}, Type Filter: {artifact_type_filter}, Limit: {limit}"
        )
        results_metadata = []
        client = self._get_chroma_client()
        if not client:
            self.logger.warning("ChromaDB client not available. Cannot list artifact metadata.")
            return results_metadata

        where_filter = {"type": "artifact"}  # Base filter
        if stage_filter is not None:
            where_filter["stage"] = stage_filter
        if artifact_type_filter:
            where_filter["artifact_type"] = artifact_type_filter

        # <<< REMOVING DIAGNOSTIC LOGGING >>>
        # self.logger.debug(f"Constructed where filter for collection.get(): {where_filter}")
        self.logger.debug(f"Using where filter for listing: {where_filter}")

        try:
            collection = client.get_collection(name=self._CONTEXT_COLLECTION_NAME)
            # <<< REMOVING DIAGNOSTIC LOGGING >>>
            # self.logger.info(f"Attempting collection.get(where={where_filter}, limit={limit}, include=['metadatas']) for collection '{self._CONTEXT_COLLECTION_NAME}'...")
            self.logger.info(
                f"Attempting collection.get() for collection '{self._CONTEXT_COLLECTION_NAME}'..."
            )

            get_results = collection.get(
                where=where_filter,
                limit=limit,
                include=["metadatas"],  # Fetch only metadata
            )
            # <<< REMOVING DIAGNOSTIC LOGGING >>>
            # self.logger.info(f"Raw results from collection.get(): {get_results}")
            self.logger.info(
                f"Raw get_results from collection.get(): {get_results}"
            )  # <<< KEEPING ONE >>>

            self.logger.info(
                f"collection.get() completed. Found {len(get_results.get('ids', []))} matching artifacts."
            )

            # Process results more carefully
            ids = get_results.get("ids", [])
            metadatas = get_results.get("metadatas", [])

            for i, doc_id in enumerate(ids):
                # Start with the full metadata dictionary if available
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                # Ensure base fields are present even if metadata retrieval was partial/empty
                meta["_id"] = doc_id  # Add the internal ID for reference
                meta["type"] = meta.get("type", "artifact")  # Default if missing
                meta["path"] = meta.get("path", "unknown_path")
                meta["stage"] = meta.get("stage", None)
                # Explicitly add description and keywords if they exist in the source metadata
                meta["description"] = meta.get("description", "")  # Add default empty string
                meta["keywords"] = meta.get("keywords", [])  # Add default empty list

                self.logger.info(f"Processed metadata for doc_id '{doc_id}': {meta}")

                results_metadata.append(meta)

        except ValueError as e:  # <<< REVERTED TO ValueError
            # Handle case where collection doesn't exist (expected behavior for this version)
            if "does not exist" in str(e).lower():
                self.logger.info(
                    f"Collection '{self._CONTEXT_COLLECTION_NAME}' not found during list_artifact_metadata (ValueError check). Returning empty list."
                )
            else:
                # Log other ValueErrors differently
                self.logger.warning(
                    f"ValueError occurred while accessing collection '{self._CONTEXT_COLLECTION_NAME}': {e}. Returning empty list."
                )
            # Return empty list if collection doesn't exist or other ValueError
        except Exception as e:
            self.logger.error(f"Failed to list artifact metadata from ChromaDB: {e}", exc_info=True)
            # Return empty list on other errors

        return results_metadata


# Example usage (for testing purposes)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    test_file = "./temp_project_status.json"

    # Clean up previous test file if exists
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists(f"{test_file}.lock"):
        os.remove(f"{test_file}.lock")

    # --- Test Initialization ---
    sm = StateManager(test_file)
    print(f"Initial next stage: {sm.get_next_stage()}")  # Should be 0

    # --- Test Update ---
    success = sm.update_status(stage=0, status="DONE", artifacts=["goal.txt"])
    print(f"Update Stage 0 success: {success}")
    print(f"Next stage after 0: {sm.get_next_stage()}")  # Should be 0.5

    success = sm.update_status(stage=0.5, status="DONE", artifacts=["doc.txt"])
    print(f"Update Stage 0.5 success: {success}")
    print(f"Next stage after 0.5: {sm.get_next_stage()}")  # Should be 1

    success = sm.update_status(stage=1, status="FAIL", artifacts=[], reason="Blueprint invalid")
    print(f"Update Stage 1 success: {success}")
    print(f"Next stage after 1 (FAIL): {sm.get_next_stage()}")  # Should be None

    # --- Test Reading ---
    full_status = sm.get_full_status()
    print("\nFull Status:")
    print(json.dumps(full_status, indent=2))

    last = sm.get_last_status()
    print("\nLast Status:")
    print(json.dumps(last, indent=2))

    # --- Test Overwrite ---
    success = sm.update_status(stage=1, status="DONE", artifacts=["bp.txt"])
    print(f"\nOverwrite Stage 1 success: {success}")
    print(f"Next stage after 1 (DONE): {sm.get_next_stage()}")  # Should be 2

    full_status = sm.get_full_status()
    print("\nFull Status after overwrite:")
    print(json.dumps(full_status, indent=2))

    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists(f"{test_file}.lock"):
        os.remove(f"{test_file}.lock")
    print("\nCleanup complete.")
