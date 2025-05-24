"""
ChromaDB MCP Tools

Project-aware ChromaDB tools that enhance the standard chroma-mcp functionality
with Chungoid's project context management and state persistence systems.

These tools replace the ProjectChromaManagerAgent_v1 with a more modular,
standardized MCP tool interface while maintaining full project isolation
and context awareness.

Features:
- Project-aware collection management with automatic namespacing
- Integration with existing chroma_utils.py and project context system
- State persistence and execution tracking
- Enhanced error handling and validation
- Support for all standard ChromaDB operations
"""

from .collection_tools import (
    chroma_list_collections,
    chroma_create_collection,
    chroma_get_collection_info,
    chroma_get_collection_count,
    chroma_modify_collection,
    chroma_delete_collection,
    chroma_peek_collection,
)

from .document_tools import (
    chroma_add_documents,
    chroma_query_documents, 
    chroma_get_documents,
    chroma_update_documents,
    chroma_delete_documents,
    chromadb_batch_operations,
    chromadb_reflection_query,
    chromadb_update_metadata,
    chromadb_store_document,
    chromadb_query_collection,
)

from .project_tools import (
    chroma_initialize_project_collections,
    chroma_set_project_context,
    chroma_get_project_status,
)

__all__ = [
    # Collection Management Tools
    "chroma_list_collections",
    "chroma_create_collection", 
    "chroma_get_collection_info",
    "chroma_get_collection_count",
    "chroma_modify_collection",
    "chroma_delete_collection",
    "chroma_peek_collection",
    
    # Document Operations Tools
    "chroma_add_documents",
    "chroma_query_documents",
    "chroma_get_documents", 
    "chroma_update_documents",
    "chroma_delete_documents",
    
    # Advanced Operations Tools
    "chromadb_batch_operations",
    "chromadb_reflection_query",
    "chromadb_update_metadata",
    
    # Blueprint-Compatible Aliases
    "chromadb_store_document",
    "chromadb_query_collection", 
    
    # Project-Specific Tools
    "chroma_initialize_project_collections",
    "chroma_set_project_context",
    "chroma_get_project_status",
] 