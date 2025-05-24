"""
ChromaDB Project-Specific Tools

Project management tools that handle initialization and management of 
project-specific ChromaDB resources. These tools replace key functionality
from ProjectChromaManagerAgent_v1 with a more modular approach.

These tools handle:
- Project collection initialization with all standard collections
- Project context management and validation
- Project status tracking and health checks
- Integration with existing project management systems
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
from datetime import datetime

from chungoid.utils.chroma_utils import (
    set_chroma_project_context,
    get_chroma_client,
    get_or_create_collection,
)
from chungoid.utils.config_manager import ConfigurationManager
from chungoid.utils.execution_state_persistence import ResumableExecutionService
from .collection_tools import STANDARD_PROJECT_COLLECTIONS, _get_project_collection_name

logger = logging.getLogger(__name__)


async def chroma_set_project_context(
    project_path: str,
    project_id: Optional[str] = None,
    create_if_missing: bool = True,
) -> Dict[str, Any]:
    """
    Sets the project context for ChromaDB operations and validates setup.
    
    This tool establishes the project context that will be used by all subsequent
    ChromaDB operations, ensuring proper isolation and resource management.
    
    Args:
        project_path: Path to the project directory
        project_id: Optional project identifier for namespacing
        create_if_missing: Whether to create project directories if they don't exist
        
    Returns:
        Dict containing context setup result and validation info
    """
    try:
        # Resolve and validate project path
        resolved_path = Path(project_path).resolve()
        
        if not resolved_path.exists():
            if create_if_missing:
                resolved_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created project directory: {resolved_path}")
            else:
                raise FileNotFoundError(f"Project path does not exist: {resolved_path}")
        
        # Set ChromaDB project context
        set_chroma_project_context(resolved_path)
        
        # Validate ChromaDB connectivity
        try:
            client = get_chroma_client()
            collections = client.list_collections()
            client_status = "connected"
            total_collections = len(collections)
        except Exception as e:
            client_status = f"error: {str(e)}"
            total_collections = 0
            collections = []
        
        # Create project-specific directories if needed
        project_chroma_dir = resolved_path / ".project_chroma_dbs"
        if project_id:
            project_chroma_dir = project_chroma_dir / project_id
        
        if create_if_missing:
            project_chroma_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "status": "success",
            "project_path": str(resolved_path),
            "project_id": project_id,
            "project_chroma_dir": str(project_chroma_dir),
            "chromadb_status": client_status,
            "total_collections": total_collections,
            "collections": [c.name for c in collections] if collections else [],
            "context_set_at": datetime.now().isoformat(),
        }
        
        logger.info(f"Set project context for '{project_id or 'default'}' at {resolved_path}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to set project context: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "project_path": project_path,
            "project_id": project_id,
        }


async def chroma_initialize_project_collections(
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    collections: Optional[List[str]] = None,
    embedding_function: Optional[str] = None,
    force_recreate: bool = False,
) -> Dict[str, Any]:
    """
    Initializes all standard project collections for a Chungoid project.
    
    This tool creates all the standard collections needed for autonomous project
    operation, replacing the initialization functionality from ProjectChromaManagerAgent_v1.
    
    Args:
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        collections: Optional list of specific collections to initialize (defaults to all standard)
        embedding_function: Embedding function to use for collections (optional)
        force_recreate: Whether to recreate existing collections
        
    Returns:
        Dict containing initialization results for all collections
    """
    try:
        # Set project context if specified
        if project_path:
            resolved_path = Path(project_path).resolve()
            set_chroma_project_context(resolved_path)
        else:
            resolved_path = Path.cwd()
            
        # Use all standard collections if none specified
        if collections is None:
            collections = STANDARD_PROJECT_COLLECTIONS.copy()
            
        # Get configuration for default embedding function
        try:
            config = ConfigurationManager()
            default_embedding = config.get_config().get("chromadb", {}).get("default_embedding_function")
            if not embedding_function and default_embedding:
                embedding_function = default_embedding
        except Exception as e:
            logger.warning(f"Could not load config for embedding function: {e}")
        
        # Initialize collections
        results = {
            "status": "success",
            "project_path": str(resolved_path),
            "project_id": project_id,
            "embedding_function": embedding_function,
            "collections_initialized": [],
            "collections_skipped": [],
            "collections_failed": [],
            "total_requested": len(collections),
        }
        
        client = get_chroma_client()
        
        for collection_name in collections:
            try:
                # Generate project-aware collection name
                full_collection_name = _get_project_collection_name(collection_name, project_id)
                
                # Check if collection exists
                existing_collections = [c.name for c in client.list_collections()]
                collection_exists = full_collection_name in existing_collections
                
                if collection_exists and not force_recreate:
                    results["collections_skipped"].append({
                        "name": collection_name,
                        "full_name": full_collection_name,
                        "reason": "already_exists"
                    })
                    logger.info(f"Skipped existing collection: {full_collection_name}")
                    continue
                    
                if collection_exists and force_recreate:
                    # Delete existing collection
                    client.delete_collection(full_collection_name)
                    logger.info(f"Deleted existing collection for recreation: {full_collection_name}")
                
                # Create collection with metadata
                metadata = {
                    "project_id": project_id,
                    "collection_type": "standard_project",
                    "created_at": datetime.now().isoformat(),
                    "embedding_function": embedding_function,
                }
                
                collection = get_or_create_collection(
                    collection_name=full_collection_name,
                    metadata=metadata
                )
                
                if collection:
                    results["collections_initialized"].append({
                        "name": collection_name,
                        "full_name": full_collection_name,
                        "count": collection.count(),
                        "metadata": collection.metadata,
                    })
                    logger.info(f"Initialized collection: {full_collection_name}")
                else:
                    raise Exception(f"Failed to create collection {full_collection_name}")
                    
            except Exception as e:
                results["collections_failed"].append({
                    "name": collection_name,
                    "error": str(e)
                })
                logger.error(f"Failed to initialize collection '{collection_name}': {e}")
        
        # Update overall status based on results
        total_success = len(results["collections_initialized"]) + len(results["collections_skipped"])
        if len(results["collections_failed"]) > 0:
            if total_success > 0:
                results["status"] = "partial_success"
            else:
                results["status"] = "failed"
        
        summary_msg = f"Initialized {len(results['collections_initialized'])} collections, skipped {len(results['collections_skipped'])}, failed {len(results['collections_failed'])}"
        logger.info(f"Project collection initialization complete: {summary_msg}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to initialize project collections: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "project_path": project_path,
            "project_id": project_id,
            "collections_initialized": [],
            "collections_skipped": [],
            "collections_failed": [],
        }


async def chroma_get_project_status(
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    include_collection_details: bool = True,
    include_document_counts: bool = True,
) -> Dict[str, Any]:
    """
    Gets comprehensive status information for a project's ChromaDB setup.
    
    This tool provides detailed status information about a project's ChromaDB
    collections, document counts, and overall health.
    
    Args:
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        include_collection_details: Whether to include detailed collection information
        include_document_counts: Whether to include document counts (may be slower)
        
    Returns:
        Dict containing comprehensive project status information
    """
    try:
        # Set project context if specified
        if project_path:
            resolved_path = Path(project_path).resolve()
            set_chroma_project_context(resolved_path)
        else:
            resolved_path = Path.cwd()
        
        # Get ChromaDB client and basic info
        client = get_chroma_client()
        all_collections = client.list_collections()
        
        # Filter project-specific collections if project_id specified
        if project_id:
            project_collections = [
                c for c in all_collections 
                if c.name.startswith(f"{project_id}_")
            ]
        else:
            project_collections = all_collections
        
        # Build status response
        result = {
            "status": "success",
            "project_path": str(resolved_path),
            "project_id": project_id,
            "chromadb_connected": True,
            "total_collections": len(project_collections),
            "total_documents": 0,
            "collections": [],
            "standard_collections_status": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Analyze each collection
        for collection in project_collections:
            try:
                collection_info = {
                    "name": collection.name,
                    "count": collection.count() if include_document_counts else None,
                }
                
                if include_collection_details:
                    collection_info.update({
                        "metadata": collection.metadata,
                    })
                
                if include_document_counts:
                    result["total_documents"] += collection.count()
                
                result["collections"].append(collection_info)
                
            except Exception as e:
                logger.warning(f"Could not get details for collection {collection.name}: {e}")
                result["collections"].append({
                    "name": collection.name,
                    "error": str(e)
                })
        
        # Check status of standard project collections
        if project_id:
            for standard_collection in STANDARD_PROJECT_COLLECTIONS:
                expected_name = _get_project_collection_name(standard_collection, project_id)
                collection_exists = any(c.name == expected_name for c in project_collections)
                
                result["standard_collections_status"][standard_collection] = {
                    "expected_name": expected_name,
                    "exists": collection_exists,
                    "count": None
                }
                
                if collection_exists and include_document_counts:
                    try:
                        collection = client.get_collection(expected_name)
                        result["standard_collections_status"][standard_collection]["count"] = collection.count()
                    except Exception as e:
                        logger.warning(f"Could not get count for {expected_name}: {e}")
        
        # Calculate health score
        if project_id:
            existing_standard = sum(1 for status in result["standard_collections_status"].values() if status["exists"])
            total_standard = len(STANDARD_PROJECT_COLLECTIONS)
            health_score = (existing_standard / total_standard) * 100 if total_standard > 0 else 100
            result["health_score"] = round(health_score, 1)
            result["missing_standard_collections"] = [
                name for name, status in result["standard_collections_status"].items() 
                if not status["exists"]
            ]
        
        logger.info(f"Retrieved project status for '{project_id or 'default'}': {len(project_collections)} collections, {result['total_documents']} total documents")
        return result
        
    except Exception as e:
        logger.error(f"Failed to get project status: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "project_path": project_path,
            "project_id": project_id,
            "chromadb_connected": False,
        } 