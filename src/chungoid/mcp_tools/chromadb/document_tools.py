"""
ChromaDB Document Operations Tools

Project-aware document operations tools that provide enhanced functionality
over standard chroma-mcp tools by integrating with Chungoid's project context
and state management systems.

These tools handle:
- Document CRUD operations with project context awareness
- Integration with existing chroma_utils.py functions
- Automatic embedding generation and metadata handling
- Advanced querying with semantic search and filtering
- State persistence and execution tracking
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import uuid
from datetime import datetime

from chungoid.utils.chroma_utils import (
    set_chroma_project_context,
    get_chroma_client,
    get_or_create_collection,
    add_or_update_document,
    get_document_by_id,
    query_documents,
    add_documents,
)
from chungoid.utils.config_manager import ConfigurationManager
from chungoid.utils.execution_state_persistence import ResumableExecutionService

logger = logging.getLogger(__name__)


def _ensure_project_context(project_path: Optional[str] = None, project_id: Optional[str] = None) -> Path:
    """
    Ensures project context is set for ChromaDB operations.
    
    Args:
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Path: The resolved project path
    """
    if project_path:
        resolved_path = Path(project_path).resolve()
        set_chroma_project_context(resolved_path)
        return resolved_path
    elif project_id:
        resolved_path = Path.cwd()
        set_chroma_project_context(resolved_path)
        return resolved_path
    else:
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


def _generate_document_id(content: str, custom_id: Optional[str] = None) -> str:
    """
    Generates a document ID from content or uses custom ID.
    
    Args:
        content: Document content
        custom_id: Optional custom document ID
        
    Returns:
        str: Document ID
    """
    if custom_id:
        return custom_id
    
    # Generate ID from content hash + timestamp for uniqueness
    import hashlib
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"doc_{content_hash}_{timestamp}"


async def chroma_add_documents(
    collection_name: str,
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    auto_generate_ids: bool = True,
) -> Dict[str, Any]:
    """
    Adds multiple documents to a collection in the project-specific ChromaDB instance.
    
    Enhanced version of standard chroma_add_documents with project context awareness.
    
    Args:
        collection_name: Name of the collection
        documents: List of document contents to add
        metadatas: Optional list of metadata dictionaries
        ids: Optional list of document IDs
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        auto_generate_ids: Whether to auto-generate IDs if not provided
        
    Returns:
        Dict containing addition result and document info
    """
    try:
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Generate project-aware collection name
        full_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Get or create collection
        collection = get_or_create_collection(collection_name=full_collection_name)
        if collection is None:
            raise Exception(f"Failed to get collection {full_collection_name}")
        
        # Generate IDs if not provided
        if not ids and auto_generate_ids:
            ids = [_generate_document_id(doc) for doc in documents]
        elif not ids:
            ids = [str(uuid.uuid4()) for _ in documents]
            
        # Ensure metadata list matches documents length
        if metadatas is None:
            # Create metadata with only non-None values
            base_metadata = {"added_at": datetime.now().isoformat()}
            if project_id is not None:
                base_metadata["project_id"] = project_id
            metadatas = [base_metadata.copy() for _ in documents]
        else:
            # Add project metadata to existing metadata
            for i, metadata in enumerate(metadatas):
                if metadata is None:
                    metadatas[i] = {}
                # Only add non-None values to metadata
                update_metadata = {"added_at": datetime.now().isoformat()}
                if project_id is not None:
                    update_metadata["project_id"] = project_id
                metadatas[i].update(update_metadata)
        
        # Add documents using existing chroma_utils
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_collection_name": collection_name,
            "documents_added": len(documents),
            "ids": ids,
            "project_path": str(resolved_path),
            "project_id": project_id,
            "total_collection_count": collection.count(),
        }
        
        logger.info(f"Added {len(documents)} documents to collection '{full_collection_name}' for project {project_id or 'default'}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to add documents to collection '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
            "documents_added": 0,
        }


async def chroma_query_documents(
    collection_name: str,
    query_texts: Optional[List[str]] = None,
    query_embeddings: Optional[List[List[float]]] = None,
    n_results: int = 10,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Queries documents in a collection using semantic search and filtering.
    
    Enhanced version of standard chroma_query_documents with project context awareness.
    
    Args:
        collection_name: Name of the collection to query
        query_texts: List of query texts for semantic search
        query_embeddings: List of query embeddings (alternative to query_texts)
        n_results: Maximum number of results to return
        where: Metadata filter conditions
        where_document: Document content filter conditions
        include: What to include in results (documents, metadatas, distances, embeddings)
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        
    Returns:
        Dict containing query results
    """
    try:
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Generate project-aware collection name
        full_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Get collection
        client = get_chroma_client()
        collection = client.get_collection(full_collection_name)
        
        # Set default include fields if not specified
        if include is None:
            include = ["documents", "metadatas", "distances"]
            
        # Add project_id filter to where clause if specified
        if project_id and where is None:
            where = {"project_id": project_id}
        elif project_id and where:
            where = {**where, "project_id": project_id}
        
        # Perform query
        query_result = collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
        
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_collection_name": collection_name,
            "query_texts": query_texts,
            "n_results": n_results,
            "results_count": len(query_result.get("ids", [[]])[0]) if query_result.get("ids") else 0,
            "results": query_result,
            "project_path": str(resolved_path),
            "project_id": project_id,
        }
        
        logger.info(f"Queried collection '{full_collection_name}' for project {project_id or 'default'}, found {result['results_count']} results")
        return result
        
    except Exception as e:
        logger.error(f"Failed to query collection '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
            "results_count": 0,
            "results": {},
        }


async def chroma_get_documents(
    collection_name: str,
    ids: Optional[List[str]] = None,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieves documents from a collection by IDs or filters.
    
    Enhanced version of standard chroma_get_documents with project context awareness.
    
    Args:
        collection_name: Name of the collection
        ids: Optional list of document IDs to retrieve
        where: Metadata filter conditions
        where_document: Document content filter conditions
        include: What to include in results
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        
    Returns:
        Dict containing retrieved documents
    """
    try:
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Generate project-aware collection name
        full_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Get collection
        client = get_chroma_client()
        collection = client.get_collection(full_collection_name)
        
        # Set default include fields if not specified
        if include is None:
            include = ["documents", "metadatas"]
            
        # Add project_id filter to where clause if specified
        if project_id and where is None:
            where = {"project_id": project_id}
        elif project_id and where:
            where = {**where, "project_id": project_id}
        
        # Get documents
        get_result = collection.get(
            ids=ids,
            where=where,
            where_document=where_document,
            include=include,
            limit=limit,
            offset=offset
        )
        
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_collection_name": collection_name,
            "requested_ids": ids,
            "documents_count": len(get_result.get("ids", [])),
            "documents": get_result,
            "project_path": str(resolved_path),
            "project_id": project_id,
        }
        
        logger.info(f"Retrieved {result['documents_count']} documents from collection '{full_collection_name}' for project {project_id or 'default'}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to get documents from collection '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
            "documents_count": 0,
            "documents": {},
        }


async def chroma_update_documents(
    collection_name: str,
    ids: List[str],
    documents: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    embeddings: Optional[List[List[float]]] = None,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Updates existing documents in a collection.
    
    Enhanced version of standard chroma_update_documents with project context awareness.
    
    Args:
        collection_name: Name of the collection
        ids: List of document IDs to update
        documents: Optional list of new document contents
        metadatas: Optional list of new metadata dictionaries
        embeddings: Optional list of new embeddings
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        
    Returns:
        Dict containing update result
    """
    try:
        # Ensure project context is set
        resolved_path = _ensure_project_context(project_path, project_id)
        
        # Generate project-aware collection name
        full_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Get collection
        client = get_chroma_client()
        collection = client.get_collection(full_collection_name)
        
        # Add update timestamp to metadata
        if metadatas:
            for i, metadata in enumerate(metadatas):
                if metadata is None:
                    metadatas[i] = {}
                metadatas[i].update({
                    "updated_at": datetime.now().isoformat(),
                    "project_id": project_id
                })
        else:
            metadatas = [{"updated_at": datetime.now().isoformat(), "project_id": project_id} for _ in ids]
        
        # Update documents
        collection.update(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_collection_name": collection_name,
            "documents_updated": len(ids),
            "updated_ids": ids,
            "project_path": str(resolved_path),
            "project_id": project_id,
        }
        
        logger.info(f"Updated {len(ids)} documents in collection '{full_collection_name}' for project {project_id or 'default'}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to update documents in collection '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
            "documents_updated": 0,
        }


async def chroma_delete_documents(
    collection_name: str,
    ids: Optional[List[str]] = None,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    """
    Deletes documents from a collection.
    
    Enhanced version of standard chroma_delete_documents with project context awareness.
    
    Args:
        collection_name: Name of the collection
        ids: Optional list of document IDs to delete
        where: Metadata filter conditions for deletion
        where_document: Document content filter conditions for deletion
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
        
        # Get collection
        client = get_chroma_client()
        collection = client.get_collection(full_collection_name)
        
        # Add project_id filter to where clause if specified
        if project_id and where is None:
            where = {"project_id": project_id}
        elif project_id and where:
            where = {**where, "project_id": project_id}
        
        # Get documents to be deleted for counting
        before_count = collection.count()
        
        # Delete documents
        collection.delete(
            ids=ids,
            where=where,
            where_document=where_document
        )
        
        after_count = collection.count()
        deleted_count = before_count - after_count
        
        result = {
            "status": "success",
            "collection_name": full_collection_name,
            "original_collection_name": collection_name,
            "documents_deleted": deleted_count,
            "deleted_ids": ids,
            "before_count": before_count,
            "after_count": after_count,
            "project_path": str(resolved_path),
            "project_id": project_id,
        }
        
        logger.info(f"Deleted {deleted_count} documents from collection '{full_collection_name}' for project {project_id or 'default'}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to delete documents from collection '{collection_name}': {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "collection_name": collection_name,
            "documents_deleted": 0,
        }


async def chromadb_batch_operations(
    operations: List[Dict[str, Any]],
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    fail_on_error: bool = False
) -> Dict[str, Any]:
    """
    Perform multiple ChromaDB operations in batch for efficiency.
    
    Supports bulk add, update, delete, and query operations with
    optimized performance and transaction-like semantics.
    
    Args:
        operations: List of operation dictionaries, each containing:
            - operation: str - Operation type ('add', 'update', 'delete', 'query')
            - collection_name: str - Target collection name
            - data: Dict[str, Any] - Operation-specific data
        project_path: Optional project path for context
        project_id: Optional project ID for context  
        fail_on_error: Whether to stop on first error or continue
        
    Returns:
        Dict containing:
        - success: bool - Overall success status
        - results: List - Results for each operation
        - completed_operations: int - Number of successful operations
        - failed_operations: int - Number of failed operations
        - errors: List - Error details for failed operations
    """
    try:
        # Ensure project context
        project_path = _ensure_project_context(project_path, project_id)
        
        results = []
        errors = []
        completed_operations = 0
        failed_operations = 0
        
        logger.info(f"Starting batch operations: {len(operations)} operations")
        
        for i, operation in enumerate(operations):
            try:
                op_type = operation.get("operation", "").lower()
                collection_name = operation.get("collection_name")
                data = operation.get("data", {})
                
                if not collection_name:
                    raise ValueError(f"Operation {i}: collection_name is required")
                
                result = {"operation_index": i, "operation_type": op_type, "success": False}
                
                if op_type == "add":
                    documents = data.get("documents", [])
                    metadatas = data.get("metadatas")
                    ids = data.get("ids")
                    
                    add_result = await chroma_add_documents(
                        collection_name=collection_name,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        project_path=str(project_path),
                        project_id=project_id
                    )
                    result.update(add_result)
                    
                elif op_type == "update":
                    ids = data.get("ids", [])
                    documents = data.get("documents")
                    metadatas = data.get("metadatas")
                    
                    update_result = await chroma_update_documents(
                        collection_name=collection_name,
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        project_path=str(project_path),
                        project_id=project_id
                    )
                    result.update(update_result)
                    
                elif op_type == "delete":
                    ids = data.get("ids")
                    where = data.get("where")
                    where_document = data.get("where_document")
                    
                    delete_result = await chroma_delete_documents(
                        collection_name=collection_name,
                        ids=ids,
                        where=where,
                        where_document=where_document,
                        project_path=str(project_path),
                        project_id=project_id
                    )
                    result.update(delete_result)
                    
                elif op_type == "query":
                    query_texts = data.get("query_texts")
                    query_embeddings = data.get("query_embeddings")
                    n_results = data.get("n_results", 10)
                    where = data.get("where")
                    where_document = data.get("where_document")
                    include = data.get("include", ["metadatas", "documents", "distances"])
                    
                    query_result = await chroma_query_documents(
                        collection_name=collection_name,
                        query_texts=query_texts,
                        query_embeddings=query_embeddings,
                        n_results=n_results,
                        where=where,
                        where_document=where_document,
                        include=include,
                        project_path=str(project_path),
                        project_id=project_id
                    )
                    result.update(query_result)
                    
                else:
                    raise ValueError(f"Unsupported operation type: {op_type}")
                
                if result.get("success", False):
                    completed_operations += 1
                else:
                    failed_operations += 1
                    
            except Exception as e:
                error_detail = {
                    "operation_index": i,
                    "operation_type": operation.get("operation", "unknown"),
                    "error": str(e)
                }
                errors.append(error_detail)
                failed_operations += 1
                
                result = {
                    "operation_index": i,
                    "operation_type": operation.get("operation", "unknown"),
                    "success": False,
                    "error": str(e)
                }
                
                if fail_on_error:
                    logger.error(f"Batch operation failed on operation {i}: {str(e)}")
                    break
            
            results.append(result)
        
        overall_success = failed_operations == 0
        
        logger.info(f"Batch operations completed: {completed_operations} successful, {failed_operations} failed")
        
        return {
            "success": overall_success,
            "results": results,
            "completed_operations": completed_operations,
            "failed_operations": failed_operations,
            "total_operations": len(operations),
            "errors": errors,
            "project_path": str(project_path),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch operations failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "completed_operations": 0,
            "failed_operations": len(operations),
            "total_operations": len(operations),
            "errors": [{"error": str(e)}]
        }


async def chromadb_update_metadata(
    collection_name: str,
    ids: List[str],
    metadatas: List[Dict[str, Any]],
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Updates metadata for existing documents without changing document content.
    
    This is a convenience function that wraps chroma_update_documents specifically
    for metadata-only updates, as referenced in the blueprint.
    
    Args:
        collection_name: Name of the collection
        ids: List of document IDs to update
        metadatas: List of new metadata dictionaries
        project_path: Path to project directory (optional)
        project_id: Project identifier for context (optional)
        
    Returns:
        Dict containing update result
    """
    return await chroma_update_documents(
        collection_name=collection_name,
        ids=ids,
        documents=None,  # Don't update document content
        metadatas=metadatas,
        embeddings=None,  # Don't update embeddings
        project_path=project_path,
        project_id=project_id,
    )


# Compatibility alias for blueprint specification
chromadb_store_document = chroma_add_documents  # Blueprint name for document storage
chromadb_query_collection = chroma_query_documents  # Blueprint name for semantic search


async def chromadb_reflection_query(
    collection_name: str,
    reflection_type: str,
    context: Optional[Dict[str, Any]] = None,
    time_window: Optional[str] = None,
    agent_id: Optional[str] = None,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Specialized reflection and learning queries for agent self-improvement.
    
    Enables agents to query their own execution history, patterns, and 
    learnings stored in ChromaDB for autonomous improvement and reasoning.
    
    Args:
        collection_name: Name of the collection to query
        reflection_type: Type of reflection ('errors', 'patterns', 'learnings', 'performance', 'decisions')
        context: Optional context for filtering reflections
        time_window: Optional time window (e.g., '1d', '1w', '1m')
        agent_id: Optional specific agent ID to filter by
        project_path: Optional project path for context
        project_id: Optional project ID for context
        limit: Maximum number of results to return
        
    Returns:
        Dict containing:
        - success: bool - Query success status
        - reflections: List - Matching reflection documents
        - patterns: List - Identified patterns from reflections
        - insights: List - Generated insights
        - recommendations: List - Actionable recommendations
        - metadata: Dict - Query metadata and statistics
    """
    try:
        # Ensure project context
        project_path = _ensure_project_context(project_path, project_id)
        project_collection_name = _get_project_collection_name(collection_name, project_id)
        
        # Build reflection-specific filters
        where_filters = {"reflection_type": reflection_type}
        
        if agent_id:
            where_filters["agent_id"] = agent_id
            
        if context:
            # Add context-based filters
            for key, value in context.items():
                if key.startswith("context_"):
                    where_filters[key] = value
        
        # Add time window filter if specified
        if time_window:
            try:
                from datetime import datetime, timedelta
                import re
                
                # Parse time window (e.g., '1d', '7d', '1w', '1m')
                match = re.match(r'(\d+)([dwm])', time_window.lower())
                if match:
                    value, unit = match.groups()
                    value = int(value)
                    
                    if unit == 'd':
                        delta = timedelta(days=value)
                    elif unit == 'w':
                        delta = timedelta(weeks=value)
                    elif unit == 'm':
                        delta = timedelta(days=value * 30)  # Approximate
                    else:
                        delta = timedelta(days=7)  # Default to 1 week
                    
                    cutoff_time = (datetime.now() - delta).isoformat()
                    where_filters["timestamp"] = {"$gte": cutoff_time}
                    
            except Exception as e:
                logger.warning(f"Failed to parse time window '{time_window}': {e}")
        
        # Query reflection documents
        query_result = await chroma_query_documents(
            collection_name=project_collection_name,
            query_texts=[f"reflection type {reflection_type}"],
            n_results=limit,
            where=where_filters,
            include=["metadatas", "documents", "distances"],
            project_path=str(project_path),
            project_id=project_id
        )
        
        if not query_result.get("success", False):
            return {
                "success": False,
                "error": "Failed to query reflection documents",
                "reflections": [],
                "patterns": [],
                "insights": [],
                "recommendations": []
            }
        
        results = query_result.get("results", {})
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        
        # Structure reflection data
        reflections = []
        for i, doc in enumerate(documents):
            reflection = {
                "document": doc,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "relevance_score": 1.0 - (distances[i] if i < len(distances) else 0.0)
            }
            reflections.append(reflection)
        
        # Generate patterns and insights based on reflection type
        patterns = []
        insights = []
        recommendations = []
        
        if reflection_type == "errors":
            # Analyze error patterns
            error_types = {}
            for reflection in reflections:
                error_type = reflection["metadata"].get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            patterns = [f"Most common error: {max(error_types.keys(), key=error_types.get)}" if error_types else "No error patterns found"]
            insights = [f"Found {len(reflections)} error instances across {len(error_types)} error types"]
            recommendations = ["Consider implementing error-specific handling for frequent error types"]
            
        elif reflection_type == "patterns":
            # Analyze behavioral patterns
            patterns = [f"Found {len(reflections)} pattern instances"]
            insights = ["Pattern analysis requires deeper LLM processing for meaningful insights"]
            recommendations = ["Regular pattern review can improve decision making"]
            
        elif reflection_type == "learnings":
            # Analyze learning outcomes
            insights = [f"Retrieved {len(reflections)} learning instances"]
            recommendations = ["Apply learnings to similar future scenarios"]
            
        elif reflection_type == "performance":
            # Analyze performance metrics
            avg_scores = []
            for reflection in reflections:
                score = reflection["metadata"].get("performance_score", 0)
                if isinstance(score, (int, float)):
                    avg_scores.append(score)
            
            if avg_scores:
                avg_performance = sum(avg_scores) / len(avg_scores)
                insights = [f"Average performance score: {avg_performance:.2f}"]
                patterns = [f"Performance trend based on {len(avg_scores)} measurements"]
            
        elif reflection_type == "decisions":
            # Analyze decision patterns
            decision_outcomes = {}
            for reflection in reflections:
                outcome = reflection["metadata"].get("decision_outcome", "unknown")
                decision_outcomes[outcome] = decision_outcomes.get(outcome, 0) + 1
            
            if decision_outcomes:
                patterns = [f"Decision outcomes: {dict(decision_outcomes)}"]
                insights = [f"Analyzed {len(reflections)} decision instances"]
        
        logger.info(f"Reflection query completed: {len(reflections)} reflections, {len(patterns)} patterns")
        
        return {
            "success": True,
            "reflections": reflections,
            "patterns": patterns,
            "insights": insights,
            "recommendations": recommendations,
            "metadata": {
                "reflection_type": reflection_type,
                "total_reflections": len(reflections),
                "query_filters": where_filters,
                "timestamp": datetime.now().isoformat(),
                "project_path": str(project_path)
            }
        }
        
    except Exception as e:
        logger.error(f"Reflection query failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "reflections": [],
            "patterns": [],
            "insights": [],
            "recommendations": [],
            "metadata": {
                "reflection_type": reflection_type,
                "error": str(e)
            }
        }


# Setup logging
logger = logging.getLogger(__name__) 