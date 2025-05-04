import chromadb
import logging
import os
from typing import Any, Dict


from chromadb.config import Settings


class StatusFileError(Exception):
    """Custom exception for errors related to the status file."""

    pass


class StateManager:
    """Manages the state of the MCP server, including run ID, stages, and ChromaDB interactions."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initializes the StateManager, sets up logging and ChromaDB client."""
        self.logger = logger
        self.config = config
        self.client = None
        self.collection_name = config.get("chroma_collection", "chungoid_artifacts")  # Default name

        try:
            chroma_config = config.get("chromadb", {})
            client_type = chroma_config.get("client_type", "persistent")  # Default to persistent
            chroma_path = chroma_config.get("path", "./chroma_db")
            chroma_host = chroma_config.get("host")
            chroma_port = chroma_config.get("port")
            # Ensure path is absolute if provided relative to project
            # Assuming config path might be relative to project root where config is loaded
            # This might need adjustment based on how config path is handled elsewhere
            if client_type == "persistent" and not os.path.isabs(chroma_path):
                # This assumes the CWD or a base path needs to be known.
                # For simplicity, let's default to making it relative to CWD if not absolute.
                # A better approach would be to pass the project root.
                chroma_path = os.path.abspath(chroma_path)
                self.logger.warning(f"Chroma path was relative, resolved to: {chroma_path}")

            self.logger.info(
                f"Initializing ChromaDB client. Type: {client_type}, Path: {chroma_path}, Host: {chroma_host}, Port: {chroma_port}"
            )

            if client_type == "http":
                if not chroma_host or not chroma_port:
                    raise ValueError(
                        "ChromaDB client type is http, but host or port is missing in config."
                    )
                self.client = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=Settings(allow_reset=chroma_config.get("allow_reset", False)),
                )
            elif client_type == "persistent":
                self.client = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=Settings(allow_reset=chroma_config.get("allow_reset", False)),
                )
            else:
                raise ValueError(f"Unsupported ChromaDB client_type in config: {client_type}")

            # Optional: Verify connection by listing collections or heartbeat
            self.client.heartbeat()  # Raises exception on failure
            self.logger.info(
                f"ChromaDB client initialized successfully ({client_type}) and connection verified."
            )

            # Ensure the collection exists (optional, depends on desired behavior)
            # self.client.get_or_create_collection(self.collection_name)
            # self.logger.info(f"Ensured ChromaDB collection '{self.collection_name}' exists.")

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            self.client = None  # Ensure client is None if init fails

    def list_artifact_metadata(
        self, stage_filter: str | None = None, limit: int | None = None
    ) -> list[dict]:
        """Lists artifact metadata, optionally filtered by stage and limited.

        Args:
            stage_filter: Optional stage number (as str) to filter artifacts by.
            limit: Optional maximum number of artifacts to return.

        Returns:
            A list of metadata dictionaries for matching artifacts.
        """
        self.logger.info(
            f"Listing artifact metadata with stage_filter='{stage_filter}', limit={limit}"
        )
        if not self.client:
            self.logger.error("Chroma client not initialized.")
            return []

        try:
            # Attempt to get the collection
            self.logger.info(f"Attempting to get collection: {self.collection_name}")
            collection = self.client.get_collection(self.collection_name)
            self.logger.info(f"Successfully got collection: {self.collection_name}")

            # Fetch all metadata
            self.logger.info("Attempting to fetch all metadata...")
            # Using get() without filters fetches all. Handle potential inefficiency.
            results = collection.get(include=["metadatas"])
            metadata_list = results.get("metadatas", [])
            self.logger.info(f"Raw metadata fetched ({len(metadata_list)} items): {metadata_list}")

            filtered_metadata = []
            if metadata_list:
                for meta in metadata_list:
                    if not isinstance(meta, dict):
                        self.logger.warning(f"Skipping non-dict metadata item: {meta}")
                        continue

                    # Apply stage filter
                    stage_match = True
                    if stage_filter is not None:
                        stage_val = meta.get("stage")
                        if stage_val is not None:
                            # Robust comparison as strings
                            stage_match = str(stage_val) == str(stage_filter)
                        else:
                            stage_match = False  # Stage key missing

                    if stage_match:
                        # Ensure required keys exist before adding
                        required_keys = [
                            "artifact_type",
                            "description",
                            "keywords",
                            "relative_path",
                            "stage",
                        ]
                        if all(k in meta for k in required_keys):
                            filtered_metadata.append(meta)
                        else:
                            missing_keys = [k for k in required_keys if k not in meta]
                            self.logger.warning(
                                f"Skipping metadata due to missing keys ({missing_keys}): {meta}"
                            )
            else:
                self.logger.info("Metadata list is empty or None.")

            # Apply limit
            self.logger.info(
                f"Applying limit ({limit}) to filtered metadata ({len(filtered_metadata)} items)."
            )
            if limit is not None and limit >= 0:
                final_list = filtered_metadata[:limit]
            else:
                final_list = filtered_metadata

            self.logger.info(
                f"Returning final metadata list ({len(final_list)} items): {final_list}"
            )
            return final_list

        except ValueError as e:
            # Handle case where collection doesn't exist (expected behavior)
            # This error is specifically caught when get_collection fails.
            if "does not exist" in str(e).lower():
                self.logger.info(
                    f"Collection '{self.collection_name}' not found. Returning empty list."
                )
            else:
                # Log other ValueErrors differently
                self.logger.warning(
                    f"ValueError occurred while accessing collection '{self.collection_name}': {e}. Returning empty list."
                )
            return []
        except Exception as e:
            # Catch any other unexpected errors during metadata retrieval
            self.logger.error(
                f"Unexpected error retrieving artifact metadata from '{self.collection_name}': {e}",
                exc_info=True,
            )
            return []

    # ... (rest of the class methods)
