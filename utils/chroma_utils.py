"""Utilities for interacting with ChromaDB for context storage."""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import chromadb
from .config_loader import get_config  # <<< Import config loader

logger = logging.getLogger(__name__)

# Configuration Defaults (Read from environment or fallback)
# Default to HTTP client unless overridden
# CHROMA_CLIENT_TYPE = os.getenv("CHROMA_CLIENT_TYPE", "http").lower()
# CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
# CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
# CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./chroma")

# Singleton ChromaDB client instance
_client: Optional[chromadb.ClientAPI] = None
_client_lock = asyncio.Lock()

# --- Public ChromaDB Client Accessor ---


def get_chroma_client() -> Optional[chromadb.ClientAPI]:  # <--- Removed async
    """
    Provides a singleton ChromaDB client instance based on configuration.

    Reads configuration using utils.config_loader.

    Returns:
        An initialized ChromaDB client (HttpClient or PersistentClient) or None if config fails.
    """
    global _client

    # Use a lock to prevent race conditions during initialization
    # Note: This simplistic lock might not be fully robust in highly concurrent scenarios,
    # but it's sufficient for typical MCP server usage where client access is frequent
    # but initialization is rare.

    # Double-checked locking pattern (without async lock)
    if _client is None:
        # Acquire lock (conceptually, as this is now sync)
        if _client is None:
            try:
                # Get ChromaDB configuration from the central loader
                logger.debug("Fetching ChromaDB config...")
                config = get_config()
                chroma_config = config.get("chromadb", {})
                logger.debug(f"ChromaDB config fetched: {chroma_config}")

                client_type = chroma_config.get("client_type", "http").lower()
                persist_path = chroma_config.get(
                    "persist_path", "./chroma"
                )  # Get path regardless of type for logging

                logger.info(f"Determining client type: {client_type}, persist_path: {persist_path}")

                if client_type == "http":
                    host = chroma_config.get("host", "localhost")
                    port = chroma_config.get("port", 8000)
                    logger.info(f"Initializing HttpClient with host: {host}, port: {port}")
                    _client = chromadb.HttpClient(host=host, port=port)
                    logger.info("HttpClient initialized.")
                elif client_type == "persistent":
                    logger.info(f"Initializing PersistentClient with path: {persist_path}")
                    _client = chromadb.PersistentClient(path=persist_path)
                    logger.info(
                        f"PersistentClient initialized with path: {persist_path}"
                    )  # Added explicit path log
                else:
                    logger.error(f"Unsupported ChromaDB client_type: {client_type}")
                    _client = None  # Ensure client remains None on error

                if _client:
                    logger.info("ChromaDB client initialization successful.")
                else:
                    logger.error("ChromaDB client initialization failed.")

            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
                _client = None  # Ensure client is None on exception
            finally:
                # Release lock (conceptually)
                pass

    # Log whether we are returning an existing or newly created client
    if _client:
        logger.debug("Returning ChromaDB client instance.")
        # You could add a check here like `_client.heartbeat()` for HttpClient
        # but PersistentClient doesn't have an equivalent easy check.
    else:
        logger.warning("Returning None for ChromaDB client.")

    return _client


# --- Core Operations (modified to handle potential None client) ---


async def get_or_create_collection(collection_name: str) -> Optional[chromadb.Collection]:
    """Helper to get or create a collection, returning the collection object or None."""
    client = await get_chroma_client()
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


async def add_documents(
    collection_name: str,
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
) -> bool:
    """Adds multiple documents to a specified Chroma collection."""
    collection = await get_or_create_collection(collection_name)
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


async def get_documents(
    collection_name: str,
    doc_ids: List[str],
    include: Optional[List[str]] = ["metadatas", "documents"],
) -> Optional[Dict[str, Any]]:
    """Retrieves documents from a collection by their IDs."""
    collection = await get_or_create_collection(collection_name)
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


async def query_documents(
    collection_name: str,
    query_texts: Optional[List[str]] = None,
    query_embeddings: Optional[List[List[float]]] = None,
    n_results: int = 5,
    where_filter: Optional[Dict[str, Any]] = None,
    where_document_filter: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = ["metadatas", "documents", "distances"],
) -> Optional[List[Dict[str, Any]]]:  # Changed return format for consistency
    """Queries a Chroma collection by text or embeddings."""
    collection = await get_or_create_collection(collection_name)
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
        num_results = len(results.get("ids", [[]])[0]) if results.get("ids") else 0

        if num_results > 0:
            ids = results["ids"][0]
            distances = results.get("distances", [[None] * num_results])[0]
            metadatas = results.get("metadatas", [[None] * num_results])[0]
            documents = results.get("documents", [[None] * num_results])[0]
            embeddings = results.get("embeddings")  # Embeddings might be None or list of lists
            embeddings = embeddings[0] if embeddings else [None] * num_results

            for i in range(num_results):
                formatted_results.append(
                    {
                        "id": ids[i],
                        "distance": distances[i],
                        "metadata": metadatas[i],
                        "document": documents[i],
                        "embedding": embeddings[i],  # May be None if not included
                    }
                )

        query_summary = query_texts[0] if query_texts else "embedding query"
        logger.info(
            f"Queried '{collection_name}' with '{query_summary}...', got {num_results} results."
        )
        return formatted_results

    except Exception as e:
        logger.exception(f"Error querying collection '{collection_name}': {e}")
        return None


# Add other helper functions as needed, e.g.,
# - add_or_update_document (handle upsert logic)
# - get_document_by_id (simplify get_documents for single ID)
# - get_all_documents (use collection.get() without IDs)
# - delete_documents


# Example utility for get_document_by_id
async def get_document_by_id(
    collection_name: str, doc_id: str, include: Optional[List[str]] = ["metadatas", "documents"]
) -> Optional[Dict[str, Any]]:
    """Retrieves a single document by its ID."""
    results = await get_documents(
        collection_name=collection_name, doc_ids=[doc_id], include=include
    )
    if results and results.get("ids"):
        # Extract the single result (assuming ID is unique)
        doc_index = results["ids"].index(doc_id)
        single_result = {}
        if results.get("metadatas") and results["metadatas"][doc_index]:
            single_result["metadata"] = results["metadatas"][doc_index]
        if results.get("documents") and results["documents"][doc_index]:
            single_result["document"] = results["documents"][doc_index]
        # Add other included fields if needed
        single_result["id"] = doc_id
        return single_result
    return None


# Placeholder for add_or_update_document
async def add_or_update_document(
    collection_name: str, doc_id: str, document_content: str, metadata: Dict[str, Any]
) -> bool:
    """Adds or updates a single document (basic upsert)."""
    collection = await get_or_create_collection(collection_name)
    if not collection:
        return False
    try:
        collection.upsert(ids=[doc_id], documents=[document_content], metadatas=[metadata])
        logger.info(f"Upserted document ID '{doc_id}' in collection '{collection_name}'.")
        return True
    except Exception as e:
        logger.exception(
            f"Error upserting document '{doc_id}' in collection '{collection_name}': {e}"
        )
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
