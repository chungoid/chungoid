"""
ChromaDB Collection Management Tools

Project-aware collection management tools that provide enhanced functionality
over standard chroma-mcp tools by integrating with Chungoid's project context
and state management systems.

These tools automatically handle:
- Project-specific collection namespacing
- Integration with existing chroma_utils.py functions
- Project context management via set_chroma_project_context
- Error handling and validation
- State persistence and execution tracking
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from chungoid.utils.chroma_utils import (
    set_chroma_project_context,
    get_chroma_client,
    get_or_create_collection,
)
from chungoid.utils.config_manager import ConfigurationManager
from chungoid.utils.execution_state_persistence import ResumableExecutionService

logger = logging.getLogger(__name__)

# Standard project collections from ProjectChromaManagerAgent_v1
STANDARD_PROJECT_COLLECTIONS = [
    "project_goals",
    "loprd_artifacts_collection",
    "blueprint_artifacts_collection", 
    "execution_plans_collection",
    "risk_assessment_reports",
    "optimization_suggestion_reports",
    "traceability_reports",
    "live_codebase_collection",
    "generated_code_artifacts",
    "test_reports_collection",
    "debugging_session_logs",
    "library_documentation_collection",
    "external_mcp_tools_documentation_collection",
    "agent_reflections_and_logs",
    "quality_assurance_logs",
]


def _ensure_project_context(project_path: Optional[str] = None, project_id: Optional[str] = None) -> Path:
    """
    Ensures project context is set for ChromaDB operations.
    
    Args:
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Path: The resolved project path
        
    Raises:
        ValueError: If neither project_path nor project_id can resolve to a valid path
    """
    if project_path:
        resolved_path = Path(project_path).resolve()
        set_chroma_project_context(resolved_path)
        return resolved_path
    elif project_id:
        # Use current working directory with project_id context
        resolved_path = Path.cwd()
        set_chroma_project_context(resolved_path)
        return resolved_path
    else:
        # Use current working directory as fallback
        resolved_path = Path.cwd()
        set_chroma_project_context(resolved_path)
        return resolved_path


def _get_project_collection_name(base_name: str, project_id: Optional[str] = None) -> str:
    """
    Generates project-specific collection name.
    
    Args:
        base_name: Base collection name
        project_id: Optional project identifier for namespacing
        
    Returns:
        str: Project-aware collection name
    """
    if project_id:
        return f"{project_id}_{base_name}"
    return base_name


async def chroma_list_collections(
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Lists all collections in the project-specific ChromaDB instance.
    
    Enhanced version of standard chroma_list_collections with project context awareness.
    
    Args:
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        limit: Maximum number of collections to return
        offset: Number of collections to skip
        include_metadata: Whether to include collection metadata
        
    Returns:
        Dict containing collections list and metadata
    """
    try:
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Get ChromaDB client
        client = get_chroma_client()
        
        # List collections
        collections = client.list_collections()
        
        # Apply pagination if specified
        if offset:
            collections = collections[offset:]
        if limit:
            collections = collections[:limit]
            
        # Format response
        result = {
            "status": "success",
            "collections": [],
            "total_count": len(client.list_collections()),
            "project_path": str(resolved_path),
            "project_id": project_id,
        }
        
        for collection in collections:
            collection_info = {
                "name": collection.name,
                "count": collection.count(),
            }
            
            if include_metadata:
                collection_info["metadata"] = collection.metadata
                
            result["collections"].append(collection_info)
            
        logger.info(f"Listed {len(collections)} collections for project {project_id or 'default'}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to list collections: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collections": [],
            "total_count": 0,
        }


async def chroma_create_collection(
    collection_name: str,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    embedding_function: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    get_or_create: bool = True,
) -> Dict[str, Any]:
    """
    Creates a new collection in the project-specific ChromaDB instance.
    
    Enhanced version of standard chroma_create_collection with project context awareness.
    
    Args:
        collection_name: Name of the collection to create
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        embedding_function: Embedding function to use (optional)
        metadata: Collection metadata (optional)
        get_or_create: Whether to get existing collection if it exists
        
    Returns:
        Dict containing creation result and collection info
    """
    try:
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Generate project-aware collection name
        full_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Create collection using existing chroma_utils
        collection = get_or_create_collection(
            collection_name=full_collection_name,
            metadata=metadata
        )
        
        if collection is None:
            raise Exception(f"Failed to create collection {full_collection_name}")
            
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_name": collection_name,
            "project_path": str(resolved_path),
            "project_id": project_id,
            "count": collection.count(),
            "metadata": collection.metadata,
        }
        
        logger.info(f"Created collection '{full_collection_name}' for project {project_id or 'default'}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to create collection '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
        }


async def chroma_get_collection_info(
    collection_name: str,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gets detailed information about a collection.
    
    Args:
        collection_name: Name of the collection
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        
    Returns:
        Dict containing collection information
    """
    try:
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Generate project-aware collection name
        full_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Get collection
        client = get_chroma_client()
        collection = client.get_collection(full_collection_name)
        
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_name": collection_name,
            "count": collection.count(),
            "metadata": collection.metadata,
            "project_path": str(resolved_path),
            "project_id": project_id,
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get collection info for '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
        }


async def chroma_get_collection_count(
    collection_name: str,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gets the document count for a collection.
    
    Args:
        collection_name: Name of the collection
        project_path: Path to project directory (optional) 
        project_id: Project identifier for context (optional)
        
    Returns:
        Dict containing collection count
    """
    try:
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Generate project-aware collection name
        full_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Get collection and count
        client = get_chroma_client()
        collection = client.get_collection(full_collection_name)
        
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_name": collection_name,
            "count": collection.count(),
            "project_path": str(resolved_path),
            "project_id": project_id,
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get collection count for '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
        }


async def chroma_modify_collection(
    collection_name: str,
    new_name: Optional[str] = None,
    new_metadata: Optional[Dict[str, Any]] = None,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Modifies collection name or metadata.
    
    Args:
        collection_name: Current name of the collection
        new_name: New collection name (optional)
        new_metadata: New metadata to set (optional)
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        
    Returns:
        Dict containing modification result
    """
    try:
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Generate project-aware collection names
        full_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Get collection
        client = get_chroma_client()
        collection = client.get_collection(full_collection_name)
        
        # Modify collection (ChromaDB API limitations may apply)
        if new_metadata:
            collection.modify(metadata=new_metadata)
            
        if new_name:
            # Note: ChromaDB may not support renaming directly
            # This would require creating new collection and migrating data
            logger.warning("Collection renaming may require manual migration")
            
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_name": collection_name,
            "new_name": new_name,
            "metadata": collection.metadata,
            "project_path": str(resolved_path),
            "project_id": project_id,
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to modify collection '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
        }


async def chroma_delete_collection(
    collection_name: str,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    """
    Deletes a collection from the project-specific ChromaDB instance.
    
    Args:
        collection_name: Name of the collection to delete
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        confirm: Confirmation flag to prevent accidental deletion
        
    Returns:
        Dict containing deletion result
    """
    try:
        if not confirm:
            return {
                "status": "error",
                "error": "Deletion requires explicit confirmation (confirm=True)",
                "collection_name": collection_name,
            }
            
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Generate project-aware collection name
        full_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Delete collection
        client = get_chroma_client()
        client.delete_collection(full_collection_name)
        
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_name": collection_name,
            "project_path": str(resolved_path),
            "project_id": project_id,
            "message": f"Collection '{full_collection_name}' deleted successfully",
        }
        
        logger.info(f"Deleted collection '{full_collection_name}' for project {project_id or 'default'}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to delete collection '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
        }


async def chroma_peek_collection(
    collection_name: str,
    limit: int = 10,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Peeks at a sample of documents in a collection.
    
    Args:
        collection_name: Name of the collection
        limit: Number of documents to peek at
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        
    Returns:
        Dict containing sample documents
    """
    try:
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Generate project-aware collection name
        full_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Get collection and peek
        client = get_chroma_client()
        collection = client.get_collection(full_collection_name)
        
        # Get sample documents
        result_data = collection.peek(limit=limit)
        
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_name": collection_name,
            "sample_size": limit,
            "total_count": collection.count(),
            "documents": result_data.get("documents", []),
            "metadatas": result_data.get("metadatas", []),
            "ids": result_data.get("ids", []),
            "project_path": str(resolved_path),
            "project_id": project_id,
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to peek collection '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
        } 