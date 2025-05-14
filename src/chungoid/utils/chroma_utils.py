"""Utilities for interacting with ChromaDB for context storage."""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings # Ensure Settings is imported
from .config_loader import get_config
from pathlib import Path
import os
import subprocess
import time
import socket
import shutil
from urllib.parse import urlparse

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

class ChromaOperationError(Exception):
    """Custom exception for errors during ChromaDB operations."""
    pass

# Singleton ChromaDB client instance
_client: Optional[chromadb.ClientAPI] = None
_client_project_context: Optional[Path] = None  # Store context used for init
_client_mode: Optional[str] = None  # Track mode used during init

# --- Public ChromaDB Client Accessor ---
def get_chroma_client() -> chromadb.ClientAPI:
    """
    Creates and returns a ChromaDB client based on the application config.
    Uses a singleton pattern to reuse the client once initialized for a given context.
    Handles different modes (in-memory, http, persistent) and project context.
    """
    global _client, _client_project_context, _client_mode, _current_project_directory

    # Core configuration loading directly using the imported get_config
    app_config = get_config() 
    chroma_config = app_config.get("chromadb", {})
    mode = chroma_config.get("mode", "in-memory") # Default to in-memory

    # Determine the effective project directory for persistent mode
    # This is crucial for the singleton logic: if project context changes, client needs re-init
    effective_project_dir = _current_project_directory # Can be None

    # Singleton Re-initialization Logic:
    # Re-initialize if:
    # 1. Client doesn't exist OR
    # 2. Mode has changed OR
    # 3. For persistent mode, if the effective_project_dir has changed from what was used for init.
    if (
        _client is None or 
        _client_mode != mode or 
        (mode == "persistent" and _client_project_context != effective_project_dir)
    ):
        logger.info(f"Chroma client needs initialization/re-initialization. Current mode: '{mode}'.")
        if mode == "in-memory":
            logger.debug("Initializing in-memory ChromaDB client.")
            _client = chromadb.Client(Settings(is_persistent=False)) # Explicitly non-persistent
        elif mode == "http":
            server_url = chroma_config.get("url")
            if not server_url:
                logger.error("ChromaDB mode is 'http' but no URL is configured.")
                raise ChromaOperationError("ChromaDB HTTP URL not configured.")
            logger.debug(f"Initializing HTTP ChromaDB client for URL: {server_url}")
            _client = _factory_get_client(mode="http", server_url=server_url, project_dir=Path(".")) # project_dir is not strictly used for http
        elif mode == "persistent":
            if not effective_project_dir:
                logger.error("ChromaDB mode is 'persistent' but no project directory context is set.")
                raise ChromaOperationError("ChromaDB project directory context not set for persistent mode.")
            logger.debug(f"Initializing persistent ChromaDB client for project: {effective_project_dir}")
            # _factory_get_client handles path construction using effective_project_dir
            _client = _factory_get_client(mode="persistent", project_dir=effective_project_dir)
        else:
            logger.error(f"Unsupported ChromaDB mode: '{mode}'")
            raise ChromaOperationError(f"Unsupported ChromaDB mode: '{mode}'")
        
        _client_mode = mode
        _client_project_context = effective_project_dir if mode == "persistent" else None # Store context for persistent
        logger.info(f"ChromaDB client initialized in '{mode}' mode.")
    else:
        logger.debug("Reusing existing ChromaDB client.")

    if not _client: # Should be caught by earlier logic, but as a safeguard
        raise ChromaOperationError("ChromaDB client could not be initialized.")
    return _client

# --- Internal Factory --- 
def _factory_get_client(
    mode: str,
    project_dir: Path, # Used for persistent mode base path
    server_url: Optional[str] = None,
) -> chromadb.ClientAPI:
    """Internal factory to create and return a ChromaDB client based on mode."""
    # Standard chromadb.config.Settings is fine for HttpClient and PersistentClient
    # as they configure themselves further based on parameters.
    # Specific settings like auth can be layered here if complex config is needed.
    client_settings = Settings() 

    if mode == "http":
        logger.debug(f"Factory: Attempting to create HttpClient. Server URL: '{server_url}'")
        if not server_url:
            raise ValueError("server_url must be provided for http mode in _factory_get_client")
        
        parsed_url = urlparse(server_url)
        host = parsed_url.hostname
        port = parsed_url.port
        ssl_enabled = parsed_url.scheme == "https"

        if not host or not port:
            raise ValueError(
                f"Invalid server_url for http mode: '{server_url}'. Could not parse host/port."
            )
        
        headers: Dict[str, str] = {}
        # Example for auth if needed via app_config:
        # app_config = get_config()
        # auth_config = app_config.get("chromadb", {}).get("auth", {})
        # if auth_config.get("provider") == "token":
        #    client_settings.chroma_client_auth_provider = "chromadb.auth.token.TokenAuthClientProvider"
        #    client_settings.chroma_client_auth_credentials = auth_config.get("credentials")
        #    headers["Authorization"] = f"Bearer {auth_config.get('credentials')}" # Or however client expects it

        logger.debug(f"HttpClient params: host='{host}', port={port}, ssl={ssl_enabled}, headers={headers}, settings={client_settings}")
        return chromadb.HttpClient(
            host=host,
            port=port,
            ssl=ssl_enabled,
            settings=client_settings,
            headers=headers,
        )
    elif mode == "persistent":
        logger.debug(f"Factory: Attempting to create PersistentClient. Project dir: '{project_dir}'")
        if not project_dir: 
            raise ValueError("Project directory must be provided for persistent mode in _factory_get_client.")
        
        # SIMPLIFICATION: Use vanilla Settings() and let Chroma handle defaults / modern env var loading.
        # Remove all logic that populates settings_params from os.environ or get_config()
        # _env = os.environ
        # env_api_impl = _env.get("CHROMA_API_IMPL")
        # env_db_impl = _env.get("CHROMA_DB_IMPL")
        # env_is_persistent = _env.get("CHROMA_IS_PERSISTENT")
        # env_allow_reset = _env.get("CHROMA_ALLOW_RESET")

        # config_settings = get_config().get("chromadb", {})
        # final_api_impl = env_api_impl if env_api_impl is not None else config_settings.get("chroma_api_impl")
        # final_db_impl = env_db_impl if env_db_impl is not None else config_settings.get("chroma_db_impl")
        # ... and so on for other params ...
        # settings_params = {}
        # if final_api_impl is not None: settings_params['chroma_api_impl'] = final_api_impl
        # ... and so on ...
        # client_settings = Settings(**settings_params)

        client_settings = Settings() # Use vanilla settings
        logger.debug(f"Chroma Client Settings for PersistentClient (vanilla): {client_settings}")

        db_path = project_dir.resolve() / ".chungoid" / "chroma_db"
        logger.debug(f"PersistentClient db_path: '{db_path}'")
        os.makedirs(str(db_path), exist_ok=True)
        # For PersistentClient, is_persistent=True is implicit or handled by the client itself.
        # We can pass our client_settings which might include other global defaults if necessary.
        # client_settings.is_persistent = True # Can be set if desired for clarity, but PersistentClient implies it.
        return chromadb.PersistentClient(path=str(db_path), settings=client_settings)
    else:
        # This case should ideally be caught by get_chroma_client before calling factory
        raise ValueError(f"Unsupported ChromaDB mode in _factory_get_client: '{mode}'")

# --- Core Operations (modified to handle potential None client and be synchronous) ---

def get_or_create_collection(collection_name: str) -> Optional[chromadb.Collection]:
    """Helper to get or create a collection, returning the collection object or None."""
    client = get_chroma_client()
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
    collection = get_or_create_collection(collection_name)
    if not collection:
        return False
    try:
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
    collection = get_or_create_collection(collection_name)
    if not collection:
        return None
    try:
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
) -> Optional[List[Dict[str, Any]]]:
    """Queries a Chroma collection by text or embeddings."""
    collection = get_or_create_collection(collection_name)
    if not collection:
        return None
    try:
        results = collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where_filter,
            where_document=where_document_filter,
            include=include,
        )

        formatted_results = []
        if results and results.get("ids") and results["ids"][0]:
             num_results = len(results["ids"][0])
             ids = results["ids"][0]
             distances = results.get("distances", [[None] * num_results])[0]
             metadatas = results.get("metadatas", [[None] * num_results])[0]
             documents = results.get("documents", [[None] * num_results])[0]
             embeddings_value = results.get("embeddings") # Changed variable name to avoid conflict
             embeddings_list = embeddings_value[0] if embeddings_value else [None] * num_results

             for i in range(num_results):
                formatted_results.append(
                     {
                         "id": ids[i],
                         "distance": distances[i],
                         "metadata": metadatas[i],
                         "document": documents[i],
                         "embedding": embeddings_list[i] if embeddings_value else None, # Use corrected variable
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
    results = get_documents(collection_name, doc_ids=[doc_id], include=include)
    if results and results.get("ids") and results["ids"] and results["ids"][0]: # check inner list too
        doc_index = 0 
        single_result = {
            "id": results["ids"][0][doc_index],
            "metadata": results.get("metadatas", [[None]])[0][doc_index],
            "document": results.get("documents", [[None]])[0][doc_index],
            "embedding": results.get("embeddings", [[None]])[0][doc_index] if results.get("embeddings") and results["embeddings"][0] else None
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
    collection = get_or_create_collection(collection_name)
    if not collection:
        return False
    try:
        collection.upsert(
            ids=[doc_id], metadatas=[metadata], documents=[document_content]
        )
        logger.info(f"Upserted document ID {doc_id} in collection '{collection_name}'.")
        return True
    except Exception as e:
        logger.exception(f"Error upserting document ID {doc_id} in '{collection_name}': {e}")
        return False

# --- Helper for Persistent Client (Example, may not be directly used by get_chroma_client if using factory)
# This function demonstrates direct persistent client creation if _factory_get_client wasn't used for it.
# It is kept for reference but the main get_chroma_client now uses _factory_get_client for persistent mode.
def get_persistent_chroma_client_direct_example(project_directory: Path) -> chromadb.ClientAPI:
    """
    (Example/Reference) Initializes and returns a persistent ChromaDB client scoped to the project directory.
    """
    try:
        persist_path = project_directory.resolve() / ".chungoid" / "chroma_db"
        logger.info(f"Ensuring ChromaDB persistence directory exists: {persist_path}")
        os.makedirs(persist_path, exist_ok=True)
        logger.info(f"Initializing PersistentClient at: {persist_path}")
        explicit_settings = chromadb.Settings(is_persistent=True)
        logger.debug(f"Using explicit chromadb.Settings: {explicit_settings}")
        client = chromadb.PersistentClient(path=str(persist_path), settings=explicit_settings)
        logger.info("PersistentClient initialized successfully via direct example.")
        return client
    except OSError as e:
        logger.error(f"Failed to create ChromaDB persistence directory {persist_path}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to initialize PersistentClient at {persist_path}: {e}", exc_info=True)
        raise
