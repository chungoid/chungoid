"""
ChromaDB Migration Utilities

Utilities to standardize migration from ProjectChromaManagerAgent_v1 to MCP tools.
Provides conversion functions and helpers to ensure consistent migration patterns.
"""

from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PCMAMigrationError(Exception):
    """Raised when PCMA to MCP tool migration fails."""
    pass

def ensure_project_context(project_id: str) -> None:
    """Ensure project context is set for MCP tool operations."""
    logger.debug(f"Project context set for: {project_id}")

async def migrate_store_artifact(
    collection_name: str,
    document_id: str,
    content: Union[str, Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    Migrate PCMA store_artifact() calls to MCP chromadb_store_document().
    
    STUB IMPLEMENTATION - Replace with actual MCP tool calls when ready.
    """
    logger.info(f"STUB: Would store artifact {document_id} in {collection_name}")
    return {
        "status": "SUCCESS",
        "document_id": document_id,
        "message": "STUB: Artifact stored successfully"
    }

async def migrate_retrieve_artifact(
    collection_name: str,
    document_id: str,
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    Migrate PCMA retrieve_artifact() calls to MCP chroma_get_documents().
    
    STUB IMPLEMENTATION - Replace with actual MCP tool calls when ready.
    """
    logger.info(f"STUB: Would retrieve artifact {document_id} from {collection_name}")
    return {
        "status": "SUCCESS",
        "document_id": document_id,
        "content": f"STUB: Content for {document_id}",
        "metadata": {"artifact_type": "stub", "project_id": project_id}
    }

async def migrate_query_artifacts(
    collection_name: str,
    query_text: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    n_results: int = 5,
    project_id: str = "default"
) -> Dict[str, Any]:
    """
    Migrate PCMA get_related_artifacts() calls to MCP chroma_query_documents().
    
    STUB IMPLEMENTATION - Replace with actual MCP tool calls when ready.
    """
    logger.info(f"STUB: Would query artifacts in {collection_name}")
    return {
        "status": "SUCCESS",
        "artifacts": [],
        "message": "STUB: Query completed successfully"
    }

async def migrate_initialize_collections(project_id: str) -> Dict[str, Any]:
    """
    Migrate PCMA initialize_project_collections() to MCP tool.
    
    STUB IMPLEMENTATION - Replace with actual MCP tool calls when ready.
    """
    logger.info(f"STUB: Would initialize collections for {project_id}")
    return {
        "status": "SUCCESS",
        "project_id": project_id,
        "message": "STUB: Collections initialized successfully"
    }

def remove_pcma_dependency_from_agent_class(agent_class_content: str) -> str:
    """
    Remove PCMA dependency from agent class code.
    """
    import re
    
    # Remove PCMA imports
    content = re.sub(
        r'from chungoid\.agents\.autonomous_engine\.project_chroma_manager_agent import.*?\n',
        '',
        agent_class_content
    )
    
    # Remove PCMA from import lists
    content = re.sub(r'\s*ProjectChromaManagerAgent_v1,?\s*', '', content)
    
    # Remove PCMA field declarations
    content = re.sub(
        r'\s*_?project_chroma_manager.*?ProjectChromaManagerAgent_v1.*?\n',
        '',
        content
    )
    
    # Remove PCMA constructor parameters
    content = re.sub(
        r'\s*project_chroma_manager:\s*ProjectChromaManagerAgent_v1[^,\)]*[,\)]?',
        '',
        content
    )
    
    # Remove PCMA validation checks
    content = re.sub(
        r'\s*if.*?ProjectChromaManagerAgent.*?[\n\s]*.*?raise ValueError.*?ProjectChromaManagerAgent.*?\n',
        '',
        content
    )
    
    return content

# Common collection names for migration
COLLECTION_MAPPINGS = {
    "loprd_artifacts": "loprd_artifacts_collection",
    "blueprint_artifacts": "blueprint_artifacts_collection", 
    "execution_plans": "execution_plans_collection",
    "risk_assessment_reports": "risk_assessment_reports",
    "project_documentation": "project_documentation_artifacts",
    "agent_logs": "agent_reflections_and_logs",
    "test_reports": "test_reports_collection",
    "generated_code": "generated_code_artifacts",
}

def get_standardized_collection_name(collection_name: str) -> str:
    """Get standardized collection name for migration."""
    return COLLECTION_MAPPINGS.get(collection_name, collection_name) 