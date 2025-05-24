"""Utilities for interacting with ChromaDB for context storage."""

import logging
from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings # Ensure Settings is imported
from .config_manager import get_config, ConfigurationError
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

    # Detailed logging at the beginning of the function
    logger.debug(f"Enter get_chroma_client. Current state before any logic: _client is {{'NOT None' if _client else 'None'}}, _client_mode='{_client_mode}', _client_project_context='{_client_project_context}', _current_project_directory='{_current_project_directory}'")

    # Core configuration loading using the new configuration manager
    try:
        system_config = get_config()
        chroma_config = system_config.chromadb
    except ConfigurationError as e:
        logger.error(f"Failed to load configuration for ChromaDB: {e}")
        # Fall back to in-memory mode
        chroma_config = None
        mode = "in-memory"
    else:
        mode = chroma_config.mode  # Default is handled by the Pydantic model
    
    logger.debug(f"  Determined mode from config: '{mode}'")

    # Determine the effective project directory for persistent mode
    effective_project_dir = _current_project_directory # Can be None
    logger.debug(f"  Effective project directory for this call: '{effective_project_dir}'")

    # Singleton Re-initialization Logic:
    # Re-initialize if:
    # 1. Client doesn't exist OR
    # 2. Mode has changed OR
    # 3. For persistent mode, if the effective_project_dir has changed from what was used for init.
    
    needs_reinit = False
    if _client is None:
        logger.debug("  Condition met: _client is None. Needs re-initialization.")
        needs_reinit = True
    elif _client_mode != mode:
        logger.debug(f"  Condition met: _client_mode ('{_client_mode}') != mode ('{mode}'). Needs re-initialization.")
        needs_reinit = True
    elif mode == "persistent" and _client_project_context != effective_project_dir:
        logger.debug(f"  Condition met: mode is 'persistent' AND _client_project_context ('{_client_project_context}') != effective_project_dir ('{effective_project_dir}'). Needs re-initialization.")
        needs_reinit = True

    if needs_reinit:
        logger.info(f"Chroma client needs initialization/re-initialization. Current mode: '{mode}'. Effective project dir: '{effective_project_dir}'")
        if mode == "in-memory":
            logger.debug("Initializing in-memory ChromaDB client.")
            _client = chromadb.Client(Settings(is_persistent=False)) # Explicitly non-persistent
        elif mode == "http":
            server_url = chroma_config.url if chroma_config else None
            if not server_url:
                logger.error("ChromaDB mode is 'http' but no URL is configured.")
                raise ChromaOperationError("ChromaDB HTTP URL not configured.")
            logger.debug(f"Initializing HTTP ChromaDB client for URL: {server_url}")
            # project_dir is not strictly used for http but pass something valid
            _client = _factory_get_client(mode="http", server_url=server_url, project_dir=Path(".") if effective_project_dir is None else effective_project_dir)
        elif mode == "persistent":
            if not effective_project_dir:
                logger.error("ChromaDB mode is 'persistent' but no project directory context is set (effective_project_dir is None).")
                raise ChromaOperationError("ChromaDB project directory context not set for persistent mode.")
            logger.debug(f"Initializing persistent ChromaDB client for project: {effective_project_dir}")
            # _factory_get_client handles path construction using effective_project_dir
            _client = _factory_get_client(mode="persistent", project_dir=effective_project_dir)
        else:
            logger.error(f"Unsupported ChromaDB mode: '{mode}'")
            raise ChromaOperationError(f"Unsupported ChromaDB mode: '{mode}'")
        
        _client_mode = mode
        _client_project_context = effective_project_dir if mode == "persistent" else None # Store context for persistent
        logger.info(f"ChromaDB client initialized in '{_client_mode}' mode. _client is now {{'NOT None' if _client else 'None'}}. _client_project_context set to '{_client_project_context}'.")
    else:
        logger.debug(f"Reusing existing ChromaDB client. Mode: '{_client_mode}', Project Context: '{_client_project_context}'")

    if not _client: # Should be caught by earlier logic, but as a safeguard
        logger.error("CRITICAL: ChromaDB client is None after get_chroma_client logic. THIS SHOULD NOT HAPPEN.")
        raise ChromaOperationError("ChromaDB client could not be initialized.")
    
    logger.debug(f"Exit get_chroma_client: Returning client. _client is {{'NOT None' if _client else 'None'}}, _client_mode='{_client_mode}', _client_project_context='{_client_project_context}'")
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
    collection = get_chroma_client().get_or_create_collection(name=collection_name)
    if not collection:
        logger.error(
            f"Cannot get/create collection '{collection_name}': ChromaDB client not available."
        )
        return None
    try:
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

    # Based on logs, for collection.get(ids=[doc_id]), results structure appears to be:
    # results['ids'] = [doc_id_found] (a flat list of strings)
    # results['metadatas'] = [metadata_dict_or_none] (a flat list)
    # results['documents'] = [document_content_or_none] (a flat list)
    # results['embeddings'] = [embedding_list_or_none] (a flat list)

    if (results and 
        results.get("ids") and 
        isinstance(results["ids"], list) and 
        doc_id in results["ids"]) : # Check if doc_id is in the list of returned IDs
        
        try:
            # Find the index of our doc_id in the returned list of IDs.
            # For a single queried ID, if found, this index should be 0 if results["ids"] is like [doc_id].
            idx = results["ids"].index(doc_id)

            meta_list = results.get("metadatas")
            doc_list = results.get("documents")
            embed_list = results.get("embeddings")

            single_result = {
                "id": results["ids"][idx],
                "metadata": meta_list[idx] if meta_list and isinstance(meta_list, list) and len(meta_list) > idx else None,
                "document": doc_list[idx] if doc_list and isinstance(doc_list, list) and len(doc_list) > idx else None,
                "embedding": embed_list[idx] if embed_list and isinstance(embed_list, list) and len(embed_list) > idx else None,
            }
            logger.debug(f"Retrieved document ID {doc_id} from '{collection_name}'.")
            return single_result
        except ValueError: # doc_id not in results["ids"] (should be caught by outer if, but for safety)
            logger.warning(f"Document ID {doc_id} was not found via index() in results.get('ids') though 'in' check passed. This is unexpected. Results: {results}")
            return None
        except IndexError: # Should not happen if ID is found and lists are parallel and non-empty for the found item
             logger.warning(f"IndexError accessing data for document ID {doc_id} even though ID was found and indexed. Potential list mismatch. Results: {results}")
             return None
    else:
        logger.warning(f"Document ID {doc_id} not found in collection '{collection_name}' (or results['ids'] empty/malformed). Results: {results}")
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

@classmethod
def get_client(cls, db_path: Optional[str] = None) -> Union[chromadb.HttpClient, chromadb.PersistentClient]:
    logger.debug(f"Enter ChromaUtils.get_client: db_path='{db_path}'. Current state before lock: _client is {'NOT None' if cls._client else 'None'}, _current_db_path='{cls._current_db_path}'")
    with cls._client_lock:
        logger.debug(f"  ChromaUtils.get_client inside lock: db_path='{db_path}'. Current state: _client is {'NOT None' if cls._client else 'None'}, _current_db_path='{cls._current_db_path}'")
        if db_path:  # Specific path requested
            logger.debug(f"    Specific path branch. Checking condition: (db_path ('{db_path}') != _current_db_path ('{cls._current_db_path}')) OR (_client is {'None' if cls._client is None else 'NOT None'}) OR (_current_db_path is None and db_path is not None)")
            if cls._client is None or cls._current_db_path != db_path:
                logger.info(f"Chroma client needs re-initialization for specific path: {db_path}. Previous path: {cls._current_db_path}, client was {'NOT None' if cls._client else 'None'}.")
                cls._current_db_path = db_path
                cls._client = chromadb.PersistentClient(path=db_path)
                logger.info(f"ChromaDB PersistentClient (NEW INSTANCE) initialized for specific path: {db_path}. _client is now {'NOT None' if cls._client else 'None'}.")
            else:  # db_path matches _current_db_path AND _client is set
                logger.info(f"Reusing existing ChromaDB PersistentClient for specific path: {db_path}. _client was already {'NOT None' if cls._client else 'None'}.")
        else:  # Default client requested (db_path is None)
            logger.debug(f"    Default client branch. Current state: _client is {'NOT None' if cls._client else 'None'}, _current_db_path='{cls._current_db_path}'")
            logger.debug(f"    Checking condition: (_client is None) OR (_current_db_path is {'NOT None' if cls._current_db_path else 'None'}) different from default path: {cls.DEFAULT_CHROMA_DB_PATH}")
            if cls._client is None or cls._current_db_path != cls.DEFAULT_CHROMA_DB_PATH:
                logger.info(f"Chroma client needs default initialization. Client was {'NOT None' if cls._client else 'None'}, _current_db_path was {cls._current_db_path}. Default path: {cls.DEFAULT_CHROMA_DB_PATH}")
                cls.initialize_client(force_reinit=True)  # force_reinit will ensure it sets _client and _current_db_path (to default)
                logger.info(f"Default client initialization complete. _client is now {'NOT None' if cls._client else 'None'}, _current_db_path='{cls._current_db_path}'")
            else:  # Default client already exists and was for the default path
                logger.info(f"Reusing existing default ChromaDB client for path {cls.DEFAULT_CHROMA_DB_PATH}. _client is {'NOT None' if cls._client else 'None'}")

        if cls._client is None:  # Should not happen if logic above is correct
            logger.error("CRITICAL: ChromaDB client is None after get_client logic. THIS SHOULD NOT HAPPEN.")
            raise ChromaOperationError("ChromaDB client is not initialized after get_client logic.")
        
        logger.debug(f"  Exit ChromaUtils.get_client inside lock: Returning client. _client is {'NOT None' if cls._client else 'None'}, _current_db_path='{cls._current_db_path}'")
    logger.debug(f"Exit ChromaUtils.get_client after lock: _client is {'NOT None' if cls._client else 'None'}, _current_db_path='{cls._current_db_path}'")
    return cls._client
