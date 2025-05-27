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
    from chungoid.intelligence.adaptive_learning_system import (
        adaptive_learning_analyze,
        create_strategy_experiment,
        apply_learning_recommendations,
    )
    from chungoid.intelligence.advanced_replanning_intelligence import (
        create_intelligent_recovery_plan,
        predict_potential_failures,
        analyze_historical_patterns,
    )
    from chungoid.runtime.performance_optimizer import (
        get_real_time_performance_analysis,
        optimize_agent_resolution_mcp,
        generate_performance_recommendations,
    )
    INTELLIGENCE_FUNCTIONS_AVAILABLE = True
except ImportError:
    # Graceful degradation if dependencies not available
    INTELLIGENCE_FUNCTIONS_AVAILABLE = False

def get_mcp_tools_registry():
    """
    Get registry of available MCP tools for refinement capabilities.
    
    Returns a simple registry that agents can use to access MCP tools
    for analyzing project state, querying previous work, and enhancing
    context during refinement cycles.
    
    Returns:
        Dict containing available MCP tool functions organized by category
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Return a basic registry with available tool categories
        # This provides a foundation for refinement capabilities
        # without circular imports or heavy engine initialization
        
        registry = {
            "filesystem": [
                {
                    "name": "list_files",
                    "description": "List files in project directory",
                    "category": "filesystem"
                },
                {
                    "name": "read_file", 
                    "description": "Read file contents",
                    "category": "filesystem"
                }
            ],
            "chromadb": [
                {
                    "name": "query_collection",
                    "description": "Query ChromaDB collection",
                    "category": "chromadb"
                },
                {
                    "name": "store_document",
                    "description": "Store document in ChromaDB",
                    "category": "chromadb"
                }
            ],
            "project": [
                {
                    "name": "get_project_status",
                    "description": "Get current project status",
                    "category": "project"
                },
                {
                    "name": "analyze_project_structure",
                    "description": "Analyze project structure and dependencies",
                    "category": "project"
                }
            ],
            "available": True,
            "initialized": True
        }
        
        # Add intelligence tools if available
        if INTELLIGENCE_FUNCTIONS_AVAILABLE:
            registry["intelligence"] = [
                {
                    "name": "adaptive_learning_analyze",
                    "description": "Analyze execution patterns and generate learning insights",
                    "category": "intelligence",
                    "function": adaptive_learning_analyze
                },
                {
                    "name": "create_strategy_experiment",
                    "description": "Create A/B testing experiments for strategy optimization",
                    "category": "intelligence", 
                    "function": create_strategy_experiment
                },
                {
                    "name": "apply_learning_recommendations",
                    "description": "Apply learning recommendations from pattern analysis",
                    "category": "intelligence",
                    "function": apply_learning_recommendations
                },
                {
                    "name": "create_intelligent_recovery_plan",
                    "description": "Create comprehensive recovery plans for failures",
                    "category": "intelligence",
                    "function": create_intelligent_recovery_plan
                },
                {
                    "name": "predict_potential_failures", 
                    "description": "Predict potential failures based on current context",
                    "category": "intelligence",
                    "function": predict_potential_failures
                },
                {
                    "name": "analyze_historical_patterns",
                    "description": "Analyze historical patterns for planning insights",
                    "category": "intelligence",
                    "function": analyze_historical_patterns
                },
                {
                    "name": "get_real_time_performance_analysis",
                    "description": "Get real-time performance metrics and analysis",
                    "category": "intelligence",
                    "function": get_real_time_performance_analysis
                },
                {
                    "name": "optimize_agent_resolution",
                    "description": "Optimize agent resolution with performance monitoring",
                    "category": "intelligence",
                    "function": optimize_agent_resolution_mcp
                },
                {
                    "name": "generate_performance_recommendations",
                    "description": "Generate performance optimization recommendations",
                    "category": "intelligence",
                    "function": generate_performance_recommendations
                }
            ]
            logger.info("Intelligence tools registered in MCP registry")
        else:
            logger.warning("Intelligence functions not available - skipping intelligence tools registration")
        
        logger.info("Initialized MCP tools registry for refinement capabilities with intelligence integration")
        return registry
        
    except Exception as e:
        logger.warning(f"Failed to initialize MCP tools registry: {e}")
        
        return {
            "filesystem": [],
            "chromadb": [],
            "terminal": [],
            "project": [],
            "analysis": [],
            "intelligence": [],
            "available": False,
            "error": str(e)
        }

# Export the registry function
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
    # Registry function for refinement capabilities
    "get_mcp_tools_registry",
] 