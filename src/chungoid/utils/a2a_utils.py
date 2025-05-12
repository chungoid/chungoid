"""Utilities for Agent-to-Agent (A2A) communication and coordination."""

import uuid
from typing import List

# Forward reference for type hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .reflection_store import ReflectionStore, Reflection

def generate_correlation_id() -> str:
    """Generates a unique correlation ID (UUID v4) as a string.

    This ID is intended to be used to link related interactions across
    multiple agents, MCP tool calls, and ReflectionStore entries.

    Returns:
        str: A unique UUID v4 string.
    """
    return str(uuid.uuid4())

def get_reflections_by_correlation_id(
    store: "ReflectionStore", 
    correlation_id: str, 
    limit: int = 1000
) -> List["Reflection"]:
    """Retrieves reflections from the store that match a specific correlation_id.

    Note:
        This function currently performs client-side filtering. It retrieves
        up to 'limit' reflections using `store.peek()` and then filters them
        based on the `correlation_id` present in the `extra` metadata field.
        This might be inefficient for very large reflection stores.

    Args:
        store: An initialized ReflectionStore instance.
        correlation_id: The correlation ID to filter reflections by.
        limit: The maximum number of reflections to fetch from the store
               before filtering.

    Returns:
        A list of Reflection objects matching the correlation_id.
    """
    if not correlation_id:
        return []

    # Peek a number of records and filter client-side
    # Aligns with current ReflectionStore.query() pattern but could be optimized
    # later by pushing filter down to ChromaDB if needed.
    peek_results = store._coll.peek(limit=limit) 
    
    matched_reflections: List["Reflection"] = []
    retrieved_ids = peek_results.get("ids", [])
    retrieved_metadatas = peek_results.get("metadatas", [])
    retrieved_documents = peek_results.get("documents", [])

    for i in range(len(retrieved_ids)):
        meta = retrieved_metadatas[i]
        # The ReflectionStore.add method stores the raw reflection dict (excluding content)
        # in the metadata. The 'extra' field within that dict should contain the correlation_id.
        if meta and meta.get("extra", {}).get("correlation_id") == correlation_id:
            # Reconstruct the Reflection object
            doc_text = retrieved_documents[i]
            full_meta = meta.copy()
            full_meta["content"] = doc_text
            try:
                # Use the internal helper from ReflectionStore if available, 
                # otherwise reconstruct manually (assuming Reflection.parse_obj)
                if hasattr(store, '_reconstruct_single'):
                    # Note: _reconstruct_single expects the chroma get() payload format,
                    # we need to adapt the peek() result format slightly.
                    payload_for_reconstruct = {
                        "ids": [retrieved_ids[i]],
                        "metadatas": [meta],
                        "documents": [doc_text]
                    }
                    matched_reflections.append(store._reconstruct_single(payload_for_reconstruct))
                else:
                    # Fallback reconstruction if helper isn't accessible/changes
                    from .reflection_store import Reflection # Import locally if needed
                    matched_reflections.append(Reflection.parse_obj(full_meta))
            except Exception as e:
                 # Log error during reconstruction? For now, skip the problematic record.
                 print(f"Error reconstructing reflection {retrieved_ids[i]}: {e}") # Basic error feedback
                 continue
                 
    return matched_reflections 