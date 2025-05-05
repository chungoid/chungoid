"""Utilities for interacting with ChromaDB for context storage."""

import logging
from typing import List, Dict, Any, Optional
# import asyncio # No longer needed for this module directly
import chromadb
from .config_loader import get_config  # <<< Import config loader
from pathlib import Path
import os
import subprocess
import time
import socket
import shutil

logger = logging.getLogger(__name__)

# --- Project Context Management ---
_current_project_directory: Optional[Path] = None

def set_chroma_project_context(project_directory: Path):
    """
    Sets the project directory context for initializing the persistent Chroma client.
    This should be called once per project context initialization.
    """
    global _current_project_directory
    logger.info(f"Setting Chroma project context directory: {project_directory}")
    _current_project_directory = project_directory.resolve()

def clear_chroma_project_context():
    """
    Clears the project directory context.
    """
    global _current_project_directory
    logger.info("Clearing Chroma project context directory.")
    _current_project_directory = None
# --- End Project Context Management ---

# Define Custom Exception for Chroma Operations
class ChromaOperationError(Exception):
    """Custom exception for errors during ChromaDB operations."""
    pass

# Configuration Defaults (Read from environment or fallback) - Handled by get_config now

# Singleton ChromaDB client instance
_client: Optional[chromadb.ClientAPI] = None
_client_project_context: Optional[Path] = None # Store context used for init

# --- Public ChromaDB Client Accessor ---


def get_chroma_client() -> Optional[chromadb.ClientAPI]:
    """
    Provides a singleton ChromaDB client instance based on configuration.
    Reads configuration using utils.config_loader.
    Uses the project context set by `set_chroma_project_context` for persistent clients.

    Returns:
        An initialized ChromaDB client (HttpClient or PersistentClient) or None if config fails.
    """
    global _client, _client_project_context, _current_project_directory

    # Check if the context has changed since the client was initialized
    if _client is not None and _client_project_context != _current_project_directory:
        logger.warning(
            f"Chroma project context changed from '{_client_project_context}' "
            f"to '{_current_project_directory}' since client initialization. "
            f"Re-initializing client. This might indicate improper context management."
        )
        _client = None # Force re-initialization

    if _client is None:
        from .chroma_client_factory import get_client as _factory_get_client
        try:
            cfg = get_config().get("chromadb", {})
            mode = cfg.get("mode", "auto")
            if mode == "auto":
                mode = "http" if os.getenv("CHROMA_API_IMPL", "").lower() == "http" else "persistent"

            server_url = cfg.get("server_url", "")

            if mode == "persistent" and _current_project_directory is None:
                logger.error("Persistent mode requested but project context not set via set_chroma_project_context().")
                return None

            project_dir = _current_project_directory or Path.cwd()

            try:
                _client = _factory_get_client(mode, project_dir, server_url=server_url or None)
                _client_project_context = _current_project_directory
                logger.info(f"ChromaDB client initialised in {mode} mode.")
            except RuntimeError as rt_err:
                # Detect the common http-only build error when asking for persistent mode
                if (
                    mode == "persistent"
                    and "http-only client mode" in str(rt_err).lower()
                ):
                    fallback_url = server_url or cfg.get("server_url") or "http://localhost:8000"
                    logger.warning(
                        "Chroma library is compiled http-only – falling back to HttpClient (%s).", fallback_url
                    )
                    try:
                        _client = _factory_get_client("http", project_dir, server_url=fallback_url)
                        _client_project_context = _current_project_directory
                        logger.info("ChromaDB HttpClient initialised as fallback.")
                    except Exception as http_err:
                        logger.error(
                            "Fallback HttpClient initialisation failed: %s", http_err, exc_info=True
                        )
                        _client = None
                else:
                    raise
        except Exception as e:
            logger.error(f"Failed to initialise Chroma client: {e}", exc_info=True)
            _client = None

    if _client:
        logger.debug("Returning ChromaDB client instance.")
    else:
        logger.warning("Returning None for ChromaDB client.")

    return _client


# --- Core Operations (modified to handle potential None client and be synchronous) ---


def get_or_create_collection(collection_name: str) -> Optional[chromadb.Collection]:
    """Helper to get or create a collection, returning the collection object or None."""
    client = get_chroma_client() # Removed await
    if not client:
        logger.error(
            f"Cannot get/create collection '{collection_name}': ChromaDB client not available."
        )
        return None
    try:
        collection = client.get_or_create_collection(name=collection_name)
        logger.debug(f"Ensured collection exists: {collection_name}")
        return collection
    except Exception as e:
        logger.exception(f"Error getting/creating collection '{collection_name}': {e}")
        return None


def add_documents(
    collection_name: str,
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
) -> bool:
    """Adds multiple documents to a specified Chroma collection."""
    collection = get_or_create_collection(collection_name) # Removed await
    if not collection:
        return False
    try:
        # Direct call (underlying add is sync)
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Added/updated {len(documents)} documents in collection '{collection_name}'.")
        return True
    except Exception as e:
        logger.exception(f"Error adding documents to collection '{collection_name}': {e}")
        return False


def get_documents(
    collection_name: str,
    doc_ids: List[str],
    include: Optional[List[str]] = ["metadatas", "documents"],
) -> Optional[Dict[str, Any]]:
    """Retrieves documents from a collection by their IDs."""
    collection = get_or_create_collection(collection_name) # Removed await
    if not collection:
        return None
    try:
        # Direct call (underlying get is sync)
        results = collection.get(ids=doc_ids, include=include)
        logger.info(
            f"Retrieved {len(results.get('ids', []))} documents for {len(doc_ids)} requested IDs from '{collection_name}'."
        )
        return results
    except Exception as e:
        logger.exception(f"Error getting documents by ID from '{collection_name}': {e}")
        return None


def query_documents(
    collection_name: str,
    query_texts: Optional[List[str]] = None,
    query_embeddings: Optional[List[List[float]]] = None,
    n_results: int = 5,
    where_filter: Optional[Dict[str, Any]] = None,
    where_document_filter: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = ["metadatas", "documents", "distances"],
) -> Optional[List[Dict[str, Any]]]:  # Changed return format for consistency
    """Queries a Chroma collection by text or embeddings."""
    collection = get_or_create_collection(collection_name) # Removed await
    if not collection:
        return None
    try:
        # Direct call (underlying query is sync)
        results = collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where_filter,
            where_document=where_document_filter,
            include=include,
        )

        # Reformat results into a simpler list of dictionaries
        formatted_results = []
        # Check if results are valid and contain IDs before processing
        if results and results.get("ids") and results["ids"][0]:
             num_results = len(results["ids"][0])
             ids = results["ids"][0]
             distances = results.get("distances", [[None] * num_results])[0]
             metadatas = results.get("metadatas", [[None] * num_results])[0]
             documents = results.get("documents", [[None] * num_results])[0]
             embeddings = results.get("embeddings")
             embeddings_list = embeddings[0] if embeddings else [None] * num_results

             for i in range(num_results):
                formatted_results.append(
                     {
                         "id": ids[i],
                         "distance": distances[i],
                         "metadata": metadatas[i],
                         "document": documents[i],
                         "embedding": embeddings_list[i] if embeddings else None,
                     }
                 )
             logger.info(f"Query returned {len(formatted_results)} results from '{collection_name}'.")
        else:
            logger.info(f"Query returned no results from '{collection_name}'.")

        return formatted_results

    except Exception as e:
        logger.exception(f"Error querying documents from '{collection_name}': {e}")
        return None


def get_document_by_id(
    collection_name: str, doc_id: str, include: Optional[List[str]] = ["metadatas", "documents"]
) -> Optional[Dict[str, Any]]:
    """Retrieves a single document by its ID. Returns the document object or None."""
    results = get_documents(collection_name, doc_ids=[doc_id], include=include) # Removed await
    if results and results.get("ids") and results["ids"]:
        # Reformat the result from collection.get format to a single dict
        doc_index = 0 # Should only be one result
        single_result = {
            "id": results["ids"][doc_index],
            "metadata": results.get("metadatas", [None])[doc_index],
            "document": results.get("documents", [None])[doc_index],
            "embedding": results.get("embeddings", [None])[doc_index] if results.get("embeddings") else None
        }
        logger.debug(f"Retrieved document ID {doc_id} from '{collection_name}'.")
        return single_result
    else:
        logger.warning(f"Document ID {doc_id} not found in collection '{collection_name}'.")
        return None


def add_or_update_document(
    collection_name: str, doc_id: str, document_content: str, metadata: Dict[str, Any]
) -> bool:
    """Adds or updates (upserts) a single document in a collection."""
    collection = get_or_create_collection(collection_name) # Removed await
    if not collection:
        return False
    try:
        # Use upsert for add-or-update behavior
        collection.upsert(
            ids=[doc_id], metadatas=[metadata], documents=[document_content]
        )
        logger.info(f"Upserted document ID {doc_id} in collection '{collection_name}'.")
        return True
    except Exception as e:
        logger.exception(f"Error upserting document ID {doc_id} in '{collection_name}': {e}")
        return False


# --- Example Metadata Structure ---
# {
#   "source": "file_path_or_type", e.g., "chungoidmcp.py" or "planning_doc"
#   "type": "code" | "state" | "history" | "reflection" | "planning_doc"
#   "timestamp": "ISO-8601 string",
#   "stage": 2.0, # Stage number when added/relevant
#   # other relevant keys...
# }

# Add other utility functions as needed, e.g., delete_document, update_document, list_collections...

def get_persistent_chroma_client(project_directory: Path) -> chromadb.ClientAPI:
    """
    Initializes and returns a persistent ChromaDB client scoped to the project directory.

    Args:
        project_directory: The root directory of the user's project.

    Returns:
        An initialized ChromaDB client API instance.

    Raises:
        OSError: If the persistence directory cannot be created due to permissions.
        Exception: If the ChromaDB client fails to initialize for other reasons.
    """
    try:
        persist_path = project_directory.resolve() / ".chungoid" / "chroma_db"
        logger.info(f"Ensuring ChromaDB persistence directory exists: {persist_path}")
        # exist_ok=True prevents errors if the directory already exists
        os.makedirs(persist_path, exist_ok=True)

        logger.info(f"Initializing PersistentClient at: {persist_path}")
        # <<< Explicitly create Settings object >>>
        # By creating a default settings object, we might override any
        # problematic implicit settings the library might load.
        # <<< MODIFICATION: Explicitly set is_persistent=True >>>
        explicit_settings = chromadb.Settings(is_persistent=True)
        # We are intentionally *not* setting chroma_api_impl here,
        # hoping the library defaults correctly for PersistentClient.
        logger.debug(f"Using explicit chromadb.Settings: {explicit_settings}")

        # Pass the explicit settings object
        client = chromadb.PersistentClient(path=str(persist_path), settings=explicit_settings)
        logger.info("PersistentClient initialized successfully.")
        return client
    except OSError as e:
        logger.error(f"Failed to create ChromaDB persistence directory {persist_path}: {e}", exc_info=True)
        # Consider more specific error handling or user feedback depending on context
        raise
    except Exception as e:
        # Catch potential chromadb initialization errors
        logger.error(f"Failed to initialize PersistentClient at {persist_path}: {e}", exc_info=True)
        raise
