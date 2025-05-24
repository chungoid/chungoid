"""
MCP Tools Module

This module contains all MCP (Model Context Protocol) tools for the Chungoid system.
These tools provide standardized interfaces for various operations that can be called
by agents and external systems through the MCP server.

Tool Categories:
- chromadb: Vector database operations with project context awareness
- filesystem: File system operations with project-aware path resolution
- terminal: Terminal command execution with security controls and sandboxing
- content: Dynamic content generation and web fetching with intelligence
"""

from .chromadb import *
from .filesystem import *
from .terminal import *
from .content import *
from .tool_manifest import (
    generate_tool_manifest,
    discover_tools,
    get_tool_composition_recommendations,
    get_tool_performance_analytics,
    tool_discovery,
)

# Phase 2: Adaptive Learning & Advanced Re-planning
try:
    from chungoid.utils.adaptive_learning_system import (
        adaptive_learning_analyze,
        create_strategy_experiment,
        apply_learning_recommendations,
    )
    from chungoid.utils.advanced_replanning_intelligence import (
        create_intelligent_recovery_plan,
        predict_potential_failures,
        analyze_historical_patterns,
    )
except ImportError:
    # Graceful degradation if dependencies not available
    pass

__all__ = [
    # ChromaDB tools
    "chroma_list_collections",
    "chroma_create_collection", 
    "chroma_get_collection_info",
    "chroma_get_collection_count",
    "chroma_modify_collection",
    "chroma_delete_collection",
    "chroma_peek_collection",
    "chroma_add_documents",
    "chroma_query_documents",
    "chroma_get_documents", 
    "chroma_update_documents",
    "chroma_delete_documents",
    "chromadb_batch_operations",
    "chromadb_reflection_query",
    "chromadb_update_metadata",
    "chromadb_store_document",
    "chromadb_query_collection",
    "chroma_initialize_project_collections",
    "chroma_set_project_context",
    "chroma_get_project_status",
    
    # File System tools
    "filesystem_read_file",
    "filesystem_write_file",
    "filesystem_copy_file",
    "filesystem_move_file",
    "filesystem_safe_delete",
    "filesystem_list_directory",
    "filesystem_create_directory",
    "filesystem_project_scan",
    "filesystem_sync_directories",
    "filesystem_batch_operations",
    "filesystem_backup_restore",
    "filesystem_template_expansion",
    
    # Terminal tools
    "tool_run_terminal_command",
    "terminal_execute_command", 
    "terminal_execute_batch",
    "terminal_get_environment",
    "terminal_set_working_directory",
    "terminal_classify_command",
    "terminal_check_permissions",
    "terminal_sandbox_status",
    
    # Content tools
    "mcptool_get_named_content",
    "content_generate_dynamic",
    "content_cache_management", 
    "content_version_control",
    "tool_fetch_web_content",
    "web_content_summarize",
    "web_content_extract",
    "web_content_validate",
    
    # Tool Discovery & Manifest tools
    "generate_tool_manifest",
    "discover_tools",
    "get_tool_composition_recommendations", 
    "get_tool_performance_analytics",
    "tool_discovery",
] 