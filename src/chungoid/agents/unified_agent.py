"""UnifiedAgent - UAEI Base Class (Phase 1)

Single interface for ALL agent execution - eliminates dual interface complexity.
According to enhanced_cycle.md Phase 1 implementation.
Enhanced with refinement capabilities for intelligent iteration cycles.
"""

from __future__ import annotations

import logging
import time
import os
import asyncio
from abc import ABC
from typing import Any, ClassVar, List, Optional, Dict

from pydantic import BaseModel, Field, ConfigDict

from ..schemas.unified_execution_schemas import (
    ExecutionContext,
    ExecutionConfig,
    AgentExecutionResult,
    ExecutionMode,
    ExecutionMetadata,
    CompletionReason,
    CompletionAssessment,
    IterationResult,
    ToolMode,
)
from ..utils.llm_provider import LLMProvider
from ..utils.prompt_manager import PromptManager

__all__ = ["UnifiedAgent"]


class UnifiedAgent(BaseModel, ABC):
    """
    Single interface for ALL agent execution - eliminates dual interface complexity.
    Replaces: invoke_async, execute_with_protocol, execute_with_protocols
    
    Phase 1: Basic unified interface with delegation to existing methods
    Phase 2: Direct implementation of agent logic
    Phase 3: Enhanced multi-iteration cycles
    Phase 4: Intelligent refinement with MCP tools and ChromaDB integration
    """
    
    # Required class metadata (enforced by validation)
    AGENT_ID: ClassVar[str]
    AGENT_VERSION: ClassVar[str] 
    PRIMARY_PROTOCOLS: ClassVar[List[str]]
    CAPABILITIES: ClassVar[List[str]]
    
    # Standard initialization
    llm_provider: LLMProvider = Field(..., description="LLM provider for AI capabilities")
    prompt_manager: PromptManager = Field(..., description="Prompt manager for templates")
    
    # Refinement capabilities (Phase 4 enhancement)
    enable_refinement: bool = Field(default=True, description="Enable intelligent refinement using MCP tools and ChromaDB")
    mcp_tools: Optional[Any] = Field(default=None, description="MCP tools registry for refinement capabilities")
    chroma_client: Optional[Any] = Field(default=None, description="ChromaDB client for storing/querying agent outputs")
    
    # Internal
    logger: Optional[logging.Logger] = Field(default=None)
    
    # Model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Initialize refinement capabilities if enabled
        if self.enable_refinement:
            self._initialize_refinement_capabilities()

    def _initialize_refinement_capabilities(self):
        """Initialize MCP tools and ChromaDB for refinement capabilities"""
        try:
            # Initialize MCP tools registry if not provided
            if self.mcp_tools is None:
                from chungoid.mcp_tools import get_mcp_tools_registry
                self.mcp_tools = get_mcp_tools_registry()
                self.logger.info(f"[Refinement] Initialized MCP tools registry for {self.AGENT_ID}")
            
            # Initialize ChromaDB client if not provided
            if self.chroma_client is None:
                import chromadb
                self.chroma_client = chromadb.Client()
                self.logger.info(f"[Refinement] Initialized ChromaDB client for {self.AGENT_ID}")
                
        except Exception as e:
            self.logger.warning(f"[Refinement] Failed to initialize refinement capabilities: {e}")
            self.enable_refinement = False

    # ========================================
    # PHASE 1: CRITICAL MISSING INFRASTRUCTURE
    # ========================================

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Universal MCP tool calling with intelligent parameter mapping
        
        Fixes critical parameter mapping errors that caused 100% filesystem tool failures.
        Maps generic parameters to tool-specific parameter names.
        """
        if not self.mcp_tools:
            return {"success": False, "error": "MCP tools not available"}
        
        try:
            # CRITICAL FIX: Map generic parameters to tool-specific parameters
            mapped_arguments = self._map_tool_parameters(tool_name, arguments)
            
            # Get tool function
            tool_func = getattr(self.mcp_tools, tool_name, None)
            if not tool_func:
                return {"success": False, "error": f"Tool {tool_name} not found"}
            
            # Call tool with mapped parameters
            result = await tool_func(**mapped_arguments)
            
            self.logger.info(f"[MCP] Successfully called tool {tool_name}")
            return {"success": True, "result": result, "tool_name": tool_name}
            
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                self.logger.error(f"[MCP] Parameter mapping error for {tool_name}: {e}")
                return {"success": False, "error": f"Parameter mapping error: {e}", "tool_name": tool_name}
            else:
                self.logger.error(f"[MCP] Tool call failed: {tool_name} - {e}", exc_info=True)
                return {"success": False, "error": str(e), "tool_name": tool_name}
        except Exception as e:
            self.logger.error(f"[MCP] Tool call failed: {tool_name} - {e}", exc_info=True)
            return {"success": False, "error": str(e), "tool_name": tool_name}

    def _map_tool_parameters(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        CRITICAL FIX: Map generic parameters to tool-specific parameter names
        
        This fixes the 100% failure rate for filesystem tools caused by incorrect parameter mapping.
        """
        # Tool-specific parameter mappings
        parameter_mappings = {
            # Filesystem tools
            "filesystem_project_scan": {
                "path": "scan_path",
                "project_path": "project_path",
                "project_id": "project_id"
            },
            "filesystem_list_directory": {
                "path": "directory_path",
                "project_path": "project_path", 
                "project_id": "project_id"
            },
            "filesystem_read_file": {
                "path": "file_path",
                "project_path": "project_path",
                "project_id": "project_id"
            },
            "filesystem_write_file": {
                "path": "file_path",
                "project_path": "project_path",
                "project_id": "project_id"
            },
            "filesystem_copy_file": {
                "source_path": "source_path",
                "destination_path": "destination_path",
                "project_path": "project_path",
                "project_id": "project_id"
            },
            "filesystem_move_file": {
                "source_path": "source_path", 
                "destination_path": "destination_path",
                "project_path": "project_path",
                "project_id": "project_id"
            },
            "filesystem_safe_delete": {
                "path": "file_path",
                "project_path": "project_path",
                "project_id": "project_id"
            },
            "filesystem_create_directory": {
                "path": "directory_path",
                "project_path": "project_path",
                "project_id": "project_id"
            },
            "filesystem_sync_directories": {
                "source_path": "source_path",
                "destination_path": "destination_path", 
                "project_path": "project_path",
                "project_id": "project_id"
            },
            "filesystem_batch_operations": {
                "project_path": "project_path",
                "project_id": "project_id"
            },
            "filesystem_backup_restore": {
                "project_path": "project_path",
                "project_id": "project_id"
            },
            "filesystem_template_expansion": {
                "project_path": "project_path",
                "project_id": "project_id"
            },
            
            # Content tools
            "content_generate_dynamic": {
                "context": "content_context",
                "content_type": "content_type"
            },
            "content_analyze_structure": {
                "content": "content_data"
            },
            "content_validate_format": {
                "content": "content_data"
            },
            "content_extract_metadata": {
                "content": "content_data"
            },
            
            # Intelligence tools - fix missing required parameters
            "get_tool_composition_recommendations": {
                "context": "analysis_context",
                "target_tools": "target_tools"  # Required parameter
            },
            "adaptive_learning_analyze": {
                "context": "learning_context",
                "domain": "domain"
            },
            "predict_potential_failures": {
                "context": "prediction_context"
            },
            "get_real_time_performance_analysis": {
                "agent_id": "agent_id",
                "context": "analysis_context"
            },
            
            # Terminal tools
            "terminal_get_environment": {},  # No parameter mapping needed
            "terminal_run_command": {
                "command": "command"
            },
            "terminal_sandbox_status": {},
            "terminal_check_permissions": {},
            
            # ChromaDB tools
            "chromadb_query_documents": {
                "query": "query_text"
            },
            "chromadb_store_document": {
                "document": "document_content"
            },
            "chroma_peek_collection": {},
            "chroma_get_documents": {}
        }
        
        # Get mapping for this tool
        tool_mapping = parameter_mappings.get(tool_name, {})
        
        # Apply parameter mapping
        mapped_args = {}
        for key, value in arguments.items():
            # Use mapped parameter name if available, otherwise use original
            mapped_key = tool_mapping.get(key, key)
            mapped_args[mapped_key] = value
        
        # Add required parameters for specific tools if missing
        if tool_name == "get_tool_composition_recommendations" and "target_tools" not in mapped_args:
            # Provide default target_tools if missing
            mapped_args["target_tools"] = list(arguments.keys()) if arguments else []
        
        return mapped_args

    async def _get_all_available_mcp_tools(self) -> Dict[str, Any]:
        """
        ENHANCED: Universal tool discovery with comprehensive categorization
        
        Implements complete tool discovery including the missing get_mcp_tools_registry method.
        """
        if not self.mcp_tools:
            self.logger.warning("[MCP] MCP tools not available")
            return {}
        
        try:
            # CRITICAL FIX: Implement missing get_mcp_tools_registry method
            registry_tools = await self._get_mcp_tools_registry()
            
            # Get all available tools by category
            all_tools = {}
            
            # Filesystem tools
            filesystem_tools = [
                "filesystem_read_file", "filesystem_write_file", "filesystem_copy_file",
                "filesystem_move_file", "filesystem_safe_delete", "filesystem_list_directory",
                "filesystem_create_directory", "filesystem_project_scan", "filesystem_sync_directories",
                "filesystem_batch_operations", "filesystem_backup_restore", "filesystem_template_expansion"
            ]
            
            # ChromaDB tools
            chromadb_tools = [
                "chromadb_store_document", "chromadb_query_documents", "chromadb_update_document",
                "chromadb_delete_document", "chromadb_batch_operations", "chromadb_collection_management",
                "chroma_create_collection", "chroma_delete_collection", "chroma_list_collections",
                "chroma_get_collection", "chroma_peek_collection", "chroma_count_documents",
                "chroma_add_documents", "chroma_update_documents", "chroma_upsert_documents",
                "chroma_get_documents", "chroma_delete_documents", "chroma_query_collection",
                "chroma_get_nearest_neighbors", "chroma_modify_collection", "chroma_reset_collection"
            ]
            
            # Terminal tools
            terminal_tools = [
                "terminal_run_command", "terminal_get_environment", "terminal_sandbox_status",
                "terminal_check_permissions", "terminal_run_secure_command", "terminal_batch_commands",
                "terminal_monitor_process", "terminal_kill_process"
            ]
            
            # Content tools
            content_tools = [
                "content_generate_dynamic", "content_analyze_structure", "content_validate_format",
                "content_extract_metadata", "content_transform_format", "content_merge_documents",
                "content_diff_analysis", "content_quality_assessment"
            ]
            
            # Intelligence tools
            intelligence_tools = [
                "adaptive_learning_analyze", "create_strategy_experiment", "performance_optimization",
                "predict_potential_failures", "get_real_time_performance_analysis", "analyze_historical_patterns",
                "get_tool_composition_recommendations", "optimize_agent_workflow", "generate_improvement_suggestions"
            ]
            
            # Tool discovery tools
            tool_discovery_tools = [
                "get_mcp_tools_registry", "discover_available_tools", "analyze_tool_capabilities",
                "recommend_tool_combinations"
            ]
            
            # Check availability and categorize
            for category, tools in [
                ("filesystem", filesystem_tools),
                ("chromadb", chromadb_tools), 
                ("terminal", terminal_tools),
                ("content", content_tools),
                ("intelligence", intelligence_tools),
                ("tool_discovery", tool_discovery_tools)
            ]:
                available_tools = {}
                for tool_name in tools:
                    if hasattr(self.mcp_tools, tool_name):
                        tool_func = getattr(self.mcp_tools, tool_name)
                        available_tools[tool_name] = {
                            "name": tool_name,
                            "category": category,
                            "callable": callable(tool_func),
                            "async": asyncio.iscoroutinefunction(tool_func) if callable(tool_func) else False
                        }
                
                if available_tools:
                    all_tools[category] = available_tools
            
            # Add registry tools if available
            if registry_tools:
                all_tools["registry"] = registry_tools
            
            # Log discovery results
            total_tools = sum(len(tools) for tools in all_tools.values())
            self.logger.info(f"[MCP] Discovered {total_tools} callable tools across {len(all_tools)} categories")
            
            for category, tools in all_tools.items():
                self.logger.info(f"[MCP] {category.title()}: {len(tools)} tools")
            
            return all_tools
            
        except Exception as e:
            self.logger.error(f"[MCP] Tool discovery failed: {e}", exc_info=True)
            return {}

    async def _get_mcp_tools_registry(self) -> Dict[str, Any]:
        """
        CRITICAL FIX: Implement missing get_mcp_tools_registry method
        
        This method was referenced throughout the codebase but didn't exist,
        causing tool discovery failures.
        """
        try:
            # Check if registry method exists on mcp_tools
            if hasattr(self.mcp_tools, 'get_mcp_tools_registry'):
                return await self.mcp_tools.get_mcp_tools_registry()
            
            # Fallback: Create registry from available tools
            registry = {}
            
            # Get all available tool categories
            categories = ["filesystem", "chromadb", "terminal", "content", "intelligence", "tool_discovery"]
            
            for category in categories:
                category_tools = {}
                
                # Get tools for this category using naming convention
                for attr_name in dir(self.mcp_tools):
                    if attr_name.startswith(category) and callable(getattr(self.mcp_tools, attr_name)):
                        tool_func = getattr(self.mcp_tools, attr_name)
                        category_tools[attr_name] = {
                            "name": attr_name,
                            "category": category,
                            "callable": True,
                            "async": asyncio.iscoroutinefunction(tool_func)
                        }
                
                if category_tools:
                    registry[category] = category_tools
            
            self.logger.info(f"[MCP] Built registry with {len(registry)} categories")
            return registry
            
        except Exception as e:
            self.logger.warning(f"[MCP] Tool get_mcp_tools_registry not available: {e}")
            return {}

    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tools based on name patterns"""
        if tool_name.startswith(("chroma", "chromadb")):
            return "chromadb"
        elif tool_name.startswith("filesystem"):
            return "filesystem"
        elif tool_name.startswith("terminal") or tool_name == "tool_run_terminal_command":
            return "terminal"
        elif tool_name.startswith(("content", "web_content", "mcptool_get_named_content", "tool_fetch_web_content")):
            return "content"
        elif tool_name.startswith(("adaptive_learning", "create_strategy", "apply_learning", "create_intelligent", "predict_potential", "analyze_historical", "get_real_time", "optimize_agent", "generate_performance")):
            return "intelligence"
        elif tool_name.startswith(("generate_tool", "discover_tools", "get_tool", "tool_discovery")):
            return "tool_discovery"
        else:
            return "uncategorized"

    def _validate_tool_availability(self, tool_names: List[str]) -> Dict[str, bool]:
        """Validate which tools are actually available and callable"""
        validation_results = {}
        
        for tool_name in tool_names:
            try:
                from chungoid.mcp_tools import __all__ as available_tools
                if tool_name in available_tools:
                    import chungoid.mcp_tools as mcp_tools_module
                    tool_func = getattr(mcp_tools_module, tool_name)
                    validation_results[tool_name] = callable(tool_func)
                else:
                    validation_results[tool_name] = False
            except Exception:
                validation_results[tool_name] = False
        
        return validation_results

    async def _safe_tool_call_with_fallback(self, tool_name: str, arguments: Dict[str, Any], fallback_tools: List[str] = None) -> Dict[str, Any]:
        """Call tool with fallback options if primary tool fails"""
        
        # Try primary tool
        result = await self._call_mcp_tool(tool_name, arguments)
        if result.get("success"):
            return result
        
        # Try fallback tools if provided
        if fallback_tools:
            for fallback_tool in fallback_tools:
                self.logger.info(f"[MCP] Trying fallback tool: {fallback_tool}")
                fallback_result = await self._call_mcp_tool(fallback_tool, arguments)
                if fallback_result.get("success"):
                    return fallback_result
        
        return result  # Return original error if all fallbacks fail

    def _intelligently_select_tools(self, all_tools: Dict[str, Any], inputs: Any, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED: Universal intelligent tool selection - agents choose which tools to use
        
        Implements the complete enhanced_mcp.md pattern:
        1. Universal tool access (no artificial filtering)
        2. Context-aware intelligent selection
        3. Dynamic tool composition based on task requirements
        4. Fallback tool recommendations
        """
        
        # 1. UNIVERSAL ACCESS: Start with ALL available tools
        selected_tools = {}
        
        # 2. CORE TOOLS: Essential tools every agent should consider
        core_tools = [
            "filesystem_project_scan",      # Project structure analysis
            "chromadb_query_documents",     # Historical context
            "terminal_get_environment",     # Environment validation
            "content_analyze_structure",    # Content analysis
            "adaptive_learning_analyze"     # Intelligence insights
        ]
        
        # Add core tools if available
        for tool_name in core_tools:
            if tool_name in all_tools:
                selected_tools[tool_name] = all_tools[tool_name]
        
        # 3. CAPABILITY-SPECIFIC TOOLS: Based on agent capabilities
        agent_capabilities = getattr(self, 'CAPABILITIES', [])
        
        if "code_generation" in agent_capabilities:
            code_tools = [
                "filesystem_read_file", "filesystem_write_file",
                "filesystem_template_expansion", "terminal_execute_command",
                "predict_potential_failures", "get_tool_composition_recommendations"
            ]
            for tool in code_tools:
                if tool in all_tools:
                    selected_tools[tool] = all_tools[tool]
        
        if "architecture_design" in agent_capabilities:
            arch_tools = [
                "content_generate_dynamic", "analyze_historical_patterns",
                "create_intelligent_recovery_plan", "chromadb_store_document"
            ]
            for tool in arch_tools:
                if tool in all_tools:
                    selected_tools[tool] = all_tools[tool]
        
        if "documentation" in agent_capabilities:
            doc_tools = [
                "web_content_extract", "content_cache_management",
                "filesystem_batch_operations", "generate_tool_manifest"
            ]
            for tool in doc_tools:
                if tool in all_tools:
                    selected_tools[tool] = all_tools[tool]
        
        if "risk_assessment" in agent_capabilities:
            risk_tools = [
                "predict_potential_failures", "analyze_historical_patterns",
                "get_real_time_performance_analysis", "chromadb_reflection_query"
            ]
            for tool in risk_tools:
                if tool in all_tools:
                    selected_tools[tool] = all_tools[tool]
        
        if "requirements_analysis" in agent_capabilities:
            req_tools = [
                "web_content_summarize", "content_version_control",
                "apply_learning_recommendations", "chromadb_batch_operations"
            ]
            for tool in req_tools:
                if tool in all_tools:
                    selected_tools[tool] = all_tools[tool]
        
        if "quality_assurance" in agent_capabilities:
            qa_tools = [
                "terminal_execute_batch", "filesystem_backup_restore",
                "optimize_agent_resolution_mcp", "get_tool_performance_analytics"
            ]
            for tool in qa_tools:
                if tool in all_tools:
                    selected_tools[tool] = all_tools[tool]
        
        if "dependency_management" in agent_capabilities:
            dep_tools = [
                "terminal_classify_command", "filesystem_sync_directories",
                "create_strategy_experiment", "terminal_check_permissions"
            ]
            for tool in dep_tools:
                if tool in all_tools:
                    selected_tools[tool] = all_tools[tool]
        
        if "environment_setup" in agent_capabilities:
            env_tools = [
                "terminal_sandbox_status", "filesystem_create_directory",
                "generate_performance_recommendations", "tool_discovery"
            ]
            for tool in env_tools:
                if tool in all_tools:
                    selected_tools[tool] = all_tools[tool]
        
        # 4. CONTEXT-AWARE SELECTION: Based on shared context
        project_path = shared_context.get("project_root_path", ".")
        
        # If project has specific characteristics, add relevant tools
        if "python" in str(project_path).lower() or shared_context.get("language") == "python":
            python_tools = ["terminal_execute_command", "filesystem_template_expansion"]
            for tool in python_tools:
                if tool in all_tools:
                    selected_tools[tool] = all_tools[tool]
        
        if "web" in str(project_path).lower() or shared_context.get("project_type") == "web":
            web_tools = ["tool_fetch_web_content", "web_content_validate"]
            for tool in web_tools:
                if tool in all_tools:
                    selected_tools[tool] = all_tools[tool]
        
        # 5. INTELLIGENT COMPOSITION: Add complementary tools
        if "filesystem_read_file" in selected_tools and "content_analyze_structure" not in selected_tools:
            if "content_analyze_structure" in all_tools:
                selected_tools["content_analyze_structure"] = all_tools["content_analyze_structure"]
        
        if "chromadb_query_documents" in selected_tools and "chromadb_store_document" not in selected_tools:
            if "chromadb_store_document" in all_tools:
                selected_tools["chromadb_store_document"] = all_tools["chromadb_store_document"]
        
        # 6. FALLBACK RECOMMENDATIONS: Suggest alternative tools
        fallback_mapping = {
            "filesystem_project_scan": ["filesystem_list_directory", "filesystem_read_file"],
            "chromadb_query_documents": ["chromadb_get_documents", "chroma_peek_collection"],
            "adaptive_learning_analyze": ["analyze_historical_patterns", "get_real_time_performance_analysis"],
            "content_analyze_structure": ["web_content_extract", "content_generate_dynamic"],
            "terminal_get_environment": ["terminal_execute_command", "terminal_sandbox_status"]
        }
        
        # Add fallback tools for missing core tools
        for primary_tool, fallbacks in fallback_mapping.items():
            if primary_tool not in selected_tools:
                for fallback in fallbacks:
                    if fallback in all_tools:
                        selected_tools[fallback] = all_tools[fallback]
                        break
        
        # 7. PERFORMANCE OPTIMIZATION: Limit selection to prevent overload
        max_tools = 15  # Reasonable limit for performance
        if len(selected_tools) > max_tools:
            # Prioritize by category importance
            priority_order = ["intelligence", "filesystem", "chromadb", "content", "terminal", "tool_discovery"]
            prioritized_tools = {}
            
            for category in priority_order:
                category_tools = {k: v for k, v in selected_tools.items() 
                                if v.get("category") == category}
                prioritized_tools.update(category_tools)
                if len(prioritized_tools) >= max_tools:
                    break
            
            selected_tools = dict(list(prioritized_tools.items())[:max_tools])
        
        self.logger.info(f"[MCP] Intelligently selected {len(selected_tools)} tools from {len(all_tools)} available")
        
        return selected_tools

    async def execute(
        self, 
        context: ExecutionContext,
        execution_mode: ExecutionMode = ExecutionMode.OPTIMAL
    ) -> AgentExecutionResult:
        """
        Universal execution interface - handles everything:
        
        Single-Pass Mode (max_iterations=1):
        - Replaces invoke_async() functionality
        - Quick task execution for simple operations
        
        Multi-Iteration Mode (max_iterations>1):
        - Enhanced cycle execution with quality optimization
        - Continues until completion criteria met or max iterations reached
        
        Refinement Mode (enable_refinement=True):
        - Uses MCP tools to query previous work and current state
        - Stores outputs in ChromaDB for future iterations
        - Builds context-aware refinement prompts
        
        Protocol Integration:
        - Uses protocols internally for structured execution
        - No external protocol management needed
        
        Tool Integration:
        - Comprehensive tool utilization across iterations
        - Built-in tool usage optimization
        """
        
        start_time = time.time()
        config = context.execution_config
        
        # Auto-determine execution strategy if OPTIMAL mode
        if execution_mode == ExecutionMode.OPTIMAL:
            execution_mode = self._determine_optimal_mode(context)
            
        # Set up execution parameters
        max_iterations = config.max_iterations if execution_mode == ExecutionMode.MULTI_ITERATION else 1
        current_context = context
        
        self.logger.info(f"[UAEI] Starting {execution_mode.value} execution with max_iterations={max_iterations}, refinement={'enabled' if self.enable_refinement else 'disabled'}")
        
        # Multi-iteration execution loop (Phase 3 implementation)
        best_result = None
        all_tools_used = []
        
        for iteration in range(max_iterations):
            self.logger.info(f"[UAEI] Starting iteration {iteration + 1}/{max_iterations}")
            
            # Phase 4: Enhance context with refinement data if enabled
            if self.enable_refinement and iteration > 0:
                current_context = await self._enhance_context_with_refinement_data(
                    current_context, iteration, best_result
                )
            
            # Execute single iteration using agent's core logic
            iteration_result = await self._execute_iteration(current_context, iteration)
            all_tools_used.extend(iteration_result.tools_used)
            
            # Phase 4: Store iteration output in ChromaDB if refinement enabled
            if self.enable_refinement:
                await self._store_iteration_output(iteration_result, current_context, iteration)
            
            # Evaluate completion criteria
            completion_assessment = await self._assess_completion(
                iteration_result, 
                config.completion_criteria,
                current_context
            )
            
            # Check if we've achieved sufficient quality/completeness
            if completion_assessment.is_complete or iteration_result.quality_score >= config.quality_threshold:
                self.logger.info(f"[UAEI] Execution complete after {iteration + 1} iterations: {completion_assessment.reason}")
                
                execution_time = time.time() - start_time
                return AgentExecutionResult(
                    output=iteration_result.output,
                    execution_metadata=ExecutionMetadata(
                        mode=execution_mode,
                        protocol_used=iteration_result.protocol_used,
                        execution_time=execution_time,
                        iterations_planned=max_iterations,
                        tools_utilized=list(set(all_tools_used))  # Deduplicate tools
                    ),
                    iterations_completed=iteration + 1,
                    completion_reason=completion_assessment.reason,
                    quality_score=iteration_result.quality_score,
                    protocol_used=iteration_result.protocol_used
                )
            
            # Store best result so far
            if best_result is None or iteration_result.quality_score > best_result.quality_score:
                best_result = iteration_result
            
            # Enhance context for next iteration based on results and gaps identified
            if iteration < max_iterations - 1:  # Don't enhance context for the last iteration
                current_context = await self._enhance_context_for_next_iteration(
                    current_context, 
                    iteration_result, 
                    completion_assessment
                )
        
        # Return best result after all iterations exhausted
        execution_time = time.time() - start_time
        return self._create_final_result_from_iterations(
            best_result, execution_mode, max_iterations, execution_time, all_tools_used
        )

    # ------------------------------------------------------------------
    # Refinement Capabilities (Phase 4 Enhancement)
    # ------------------------------------------------------------------
    
    async def _enhance_context_with_refinement_data(
        self, 
        context: ExecutionContext, 
        iteration: int, 
        previous_result: Optional[IterationResult]
    ) -> ExecutionContext:
        """Enhance context with refinement data from MCP tools and ChromaDB"""
        if not self.enable_refinement:
            return context
        
        try:
            enhanced_shared_context = context.shared_context.copy()
            
            # Query previous work from ChromaDB
            previous_outputs = await self._query_previous_work(
                context.shared_context.get("project_id", "unknown"),
                iteration
            )
            
            # Use MCP tools to analyze current project state
            current_state = await self._analyze_current_state_with_mcp_tools(context)
            
            # Add refinement context
            enhanced_shared_context.update({
                "refinement_context": {
                    "previous_outputs": previous_outputs,
                    "current_state": current_state,
                    "iteration": iteration,
                    "previous_quality_score": previous_result.quality_score if previous_result else 0.0,
                    "refinement_enabled": True
                }
            })
            
            self.logger.info(f"[Refinement] Enhanced context with {len(previous_outputs)} previous outputs and current state analysis")
            
            # Create new context with enhanced data
            return ExecutionContext(
                inputs=context.inputs,
                shared_context=enhanced_shared_context,
                stage_info=context.stage_info,
                execution_config=context.execution_config
            )
            
        except Exception as e:
            self.logger.warning(f"[Refinement] Failed to enhance context: {e}")
            return context
    
    async def _query_previous_work(self, project_id: str, current_iteration: int) -> List[Dict[str, Any]]:
        """Query previous outputs from ChromaDB for this agent and project"""
        if not self.enable_refinement or not self.chroma_client:
            return []
        
        try:
            # Get or create collection for this agent
            collection_name = f"{self.AGENT_ID}_outputs"
            collection = self.chroma_client.get_or_create_collection(collection_name)
            
            # Query previous outputs for this project
            results = collection.query(
                query_texts=[f"project_id:{project_id}"],
                where={"$and": [{"project_id": {"$eq": project_id}}, {"agent_id": {"$eq": self.AGENT_ID}}]},
                n_results=min(current_iteration, 5)  # Get up to 5 previous outputs
            )
            
            previous_outputs = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    previous_outputs.append({
                        "content": doc,
                        "metadata": metadata,
                        "iteration": metadata.get("iteration", 0),
                        "quality_score": metadata.get("quality_score", 0.0)
                    })
            
            self.logger.info(f"[Refinement] Retrieved {len(previous_outputs)} previous outputs from ChromaDB")
            
            # Add detailed logging of retrieved content for refinement debugging
            if previous_outputs:
                self.logger.info(f"[REFINEMENT DEBUG] ChromaDB Retrieved Content Summary:")
                for i, output in enumerate(previous_outputs):
                    metadata = output.get('metadata', {})
                    content_preview = str(output.get('content', ''))[:200]
                    self.logger.info(f"  Output {i+1}: iteration={metadata.get('iteration', 'unknown')}, "
                                   f"quality={metadata.get('quality_score', 'unknown')}, "
                                   f"content_preview='{content_preview}...'")
                    
                    # If full logging is enabled via environment variable, show full content
                    if os.getenv("CHUNGOID_FULL_LLM_LOGGING", "false").lower() == "true":
                        self.logger.info(f"[REFINEMENT DEBUG] Full Content for Output {i+1}:")
                        self.logger.info("=" * 60)
                        self.logger.info(str(output.get('content', '')))
                        self.logger.info("=" * 60)
            
            return previous_outputs
            
        except Exception as e:
            self.logger.warning(f"[Refinement] Failed to query previous work: {e}")
            return []
    
    async def _analyze_current_state_with_mcp_tools(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        ENHANCED: Universal tool access with intelligent orchestration
        
        Implements the complete enhanced_mcp.md pattern:
        1. Universal tool discovery (no filtering)
        2. Intelligent tool selection based on context
        3. Comprehensive multi-tool analysis
        4. Fallback strategies for failed tools
        """
        if not self.enable_refinement:
            return {}
        
        try:
            # 1. UNIVERSAL TOOL DISCOVERY: Get ALL available tools (no filtering)
            tool_discovery = await self._get_all_available_mcp_tools()
            
            # CRITICAL FIX: Handle NoneType returns that caused subscript errors
            if not tool_discovery or not isinstance(tool_discovery, dict):
                self.logger.warning("[MCP] Tool discovery returned None or invalid data")
                return {"error": "tool_discovery_failed", "fallback_analysis": True}
            
            # 2. INTELLIGENT TOOL SELECTION: Choose tools based on context
            selected_tools = self._intelligently_select_tools(
                tool_discovery, 
                context.inputs, 
                context.shared_context
            )
            
            # CRITICAL FIX: Validate selected_tools is not None
            if not selected_tools or not isinstance(selected_tools, dict):
                self.logger.warning("[MCP] Tool selection returned None or invalid data")
                return {"error": "tool_selection_failed", "fallback_analysis": True}
            
            # 3. COMPREHENSIVE ANALYSIS: Execute selected tools with fallback strategies
            current_state = {
                "tool_discovery_summary": {
                    "total_categories": len(tool_discovery),
                    "total_tools": sum(len(tools) for tools in tool_discovery.values() if isinstance(tools, dict)),
                    "selected_tools": len(selected_tools)
                }
            }
            
            # Get project context safely
            project_root = context.shared_context.get("project_root_path", ".")
            
            if "filesystem_project_scan" in selected_tools:
                self.logger.info("[MCP] Executing comprehensive project scan")
                project_structure = await self._safe_tool_call_with_fallback(
                    "filesystem_project_scan", 
                    {"path": project_root, "include_stats": True, "detect_project_type": True},
                    ["filesystem_list_directory", "filesystem_read_file"]
                )
                current_state["project_structure"] = project_structure
                
                # Enhanced project analysis if scan successful
                if project_structure and project_structure.get("success"):
                    # Use content tools for deeper structure analysis
                    if "content_analyze_structure" in selected_tools:
                        self.logger.info("[MCP] Analyzing project structure with content tools")
                        structure_analysis = await self._call_mcp_tool(
                            "content_analyze_structure",
                            {"content": project_structure.get("result", {})}
                        )
                        current_state["structure_analysis"] = structure_analysis
            
            # 4. INTELLIGENCE ANALYSIS: Use adaptive learning tools
            if "adaptive_learning_analyze" in selected_tools:
                self.logger.info("[MCP] Executing adaptive learning analysis")
                intelligence_context = {
                    "agent_id": self.AGENT_ID,
                    "project_structure": current_state.get("project_structure", {}),
                    "execution_context": {
                        "iteration": getattr(context, 'current_iteration', 0),
                        "mode": str(getattr(context, 'execution_mode', 'unknown'))
                    }
                }
                
                intelligence_analysis = await self._safe_tool_call_with_fallback(
                    "adaptive_learning_analyze",
                    {"context": intelligence_context, "domain": self.AGENT_ID},
                    ["analyze_historical_patterns", "get_real_time_performance_analysis"]
                )
                current_state["intelligence_analysis"] = intelligence_analysis
            
            # 5. HISTORICAL CONTEXT: Use ChromaDB tools
            if "chromadb_query_documents" in selected_tools:
                self.logger.info("[MCP] Querying historical context from ChromaDB")
                project_id = context.shared_context.get('project_id', 'unknown')
                historical_query = f"agent:{self.AGENT_ID} project:{project_id}"
                
                historical_context = await self._safe_tool_call_with_fallback(
                    "chromadb_query_documents",
                    {"query": historical_query, "limit": 5, "include_metadata": True},
                    ["chroma_peek_collection", "chroma_get_documents"]
                )
                current_state["historical_context"] = historical_context
            
            # 6. ENVIRONMENT VALIDATION: Use terminal tools
            if "terminal_get_environment" in selected_tools:
                self.logger.info("[MCP] Validating execution environment")
                environment_info = await self._safe_tool_call_with_fallback(
                    "terminal_get_environment",
                    {},
                    ["terminal_sandbox_status", "terminal_check_permissions"]
                )
                current_state["environment_info"] = environment_info
            
            # 7. PREDICTIVE ANALYSIS: Use intelligence tools for failure prediction
            if "predict_potential_failures" in selected_tools:
                self.logger.info("[MCP] Executing predictive failure analysis")
                prediction_context = {
                    "current_state": current_state,
                    "agent_capabilities": getattr(self, 'CAPABILITIES', []),
                    "execution_mode": str(getattr(context, 'execution_mode', 'unknown'))
                }
                
                failure_predictions = await self._call_mcp_tool(
                    "predict_potential_failures",
                    {"context": prediction_context}
                )
                current_state["failure_predictions"] = failure_predictions
            
            # 8. PERFORMANCE OPTIMIZATION: Use performance tools
            if "get_real_time_performance_analysis" in selected_tools:
                self.logger.info("[MCP] Executing real-time performance analysis")
                performance_analysis = await self._call_mcp_tool(
                    "get_real_time_performance_analysis",
                    {"agent_id": self.AGENT_ID, "context": current_state}
                )
                current_state["performance_analysis"] = performance_analysis
            
            # 9. TOOL COMPOSITION RECOMMENDATIONS: Use tool discovery
            if "get_tool_composition_recommendations" in selected_tools:
                self.logger.info("[MCP] Getting tool composition recommendations")
                composition_context = {
                    "agent_id": self.AGENT_ID,
                    "task_type": getattr(self, 'CATEGORY', 'unknown'),
                    "current_tools": list(selected_tools.keys())
                }
                
                # CRITICAL FIX: Provide required target_tools parameter
                tool_recommendations = await self._call_mcp_tool(
                    "get_tool_composition_recommendations",
                    {
                        "context": composition_context,
                        "target_tools": list(selected_tools.keys())  # Required parameter
                    }
                )
                current_state["tool_recommendations"] = tool_recommendations
            
            # 10. CONTENT GENERATION: Use content tools for dynamic insights
            if "content_generate_dynamic" in selected_tools:
                self.logger.info("[MCP] Generating dynamic content insights")
                content_context = {
                    "analysis_results": current_state,
                    "agent_context": {
                        "id": self.AGENT_ID,
                        "capabilities": getattr(self, 'CAPABILITIES', [])
                    }
                }
                
                # CRITICAL FIX: Use correct parameter name
                dynamic_insights = await self._call_mcp_tool(
                    "content_generate_dynamic",
                    {"content_context": content_context, "content_type": "analysis_insights"}
                )
                current_state["dynamic_insights"] = dynamic_insights
            
            # 11. ANALYSIS SUMMARY: Compile comprehensive results
            successful_analyses = sum(1 for key, value in current_state.items() 
                                    if isinstance(value, dict) and value.get("success"))
            
            current_state["analysis_summary"] = {
                "total_tools_used": len(selected_tools),
                "successful_analyses": successful_analyses,
                "analysis_coverage": successful_analyses / len(selected_tools) if selected_tools else 0,
                "analysis_quality": "comprehensive" if successful_analyses >= 5 else "partial"
            }
            
            self.logger.info(f"[MCP] Completed universal analysis: {successful_analyses}/{len(selected_tools)} tools successful")
            return current_state
            
        except Exception as e:
            self.logger.error(f"[MCP] Universal analysis failed: {e}", exc_info=True)
            return {"error": str(e), "fallback_analysis": True, "analysis_type": "error_fallback"}
    
    async def _analyze_code_with_mcp_tools(self, project_path: str) -> Dict[str, Any]:
        """
        FIXED: Replace phantom tool dependency with actual MCP tools
        
        OLD: hasattr(self.mcp_tools, 'analyze_code_ast') - DOESN'T EXIST
        NEW: Use actual filesystem and content tools
        """
        try:
            # Use actual MCP tools for code analysis
            code_analysis = {}
            
            # Scan for code files
            if hasattr(self, '_call_mcp_tool'):
                file_scan = await self._call_mcp_tool(
                    "filesystem_project_scan",
                    {"path": project_path, "include_patterns": ["*.py", "*.js", "*.ts", "*.java", "*.cpp"]}
                )
                code_analysis["file_scan"] = file_scan
                
                # Analyze code structure if files found
                if file_scan.get("success") and file_scan.get("result"):
                    structure_analysis = await self._call_mcp_tool(
                        "content_analyze_structure",
                        {"content": file_scan["result"]}
                    )
                    code_analysis["structure_analysis"] = structure_analysis
            
            return code_analysis if code_analysis else {"status": "no_mcp_tools_available"}
            
        except Exception as e:
            self.logger.warning(f"[Refinement] Code analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_project_structure_with_mcp_tools(self, project_path: str) -> Dict[str, Any]:
        """
        FIXED: Replace phantom tool dependency with actual MCP tools
        
        OLD: hasattr(self.mcp_tools, 'analyze_project_structure') - DOESN'T EXIST  
        NEW: Use actual filesystem_project_scan tool
        """
        try:
            # Use actual filesystem_project_scan tool
            if hasattr(self, '_call_mcp_tool'):
                return await self._call_mcp_tool(
                    "filesystem_project_scan",
                    {"path": project_path}
                )
            return {"status": "mcp_tool_method_not_available"}
            
        except Exception as e:
            self.logger.warning(f"[Refinement] Project structure analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_documentation_with_mcp_tools(self, project_path: str) -> Dict[str, Any]:
        """
        FIXED: Replace phantom tool dependency with actual MCP tools
        
        OLD: hasattr(self.mcp_tools, 'analyze_documentation') - DOESN'T EXIST
        NEW: Use actual content and filesystem tools
        """
        try:
            # Use actual MCP tools for documentation analysis
            if hasattr(self, '_call_mcp_tool'):
                # Scan for documentation files
                doc_scan = await self._call_mcp_tool(
                    "filesystem_project_scan",
                    {"path": project_path, "include_patterns": ["*.md", "*.rst", "*.txt", "*.doc*"]}
                )
                
                if doc_scan.get("success"):
                    # Extract and analyze documentation content
                    content_analysis = await self._call_mcp_tool(
                        "content_extract_text",
                        {"source": doc_scan["result"]}
                    )
                    return {"doc_scan": doc_scan, "content_analysis": content_analysis}
            
            return {"status": "mcp_tool_method_not_available"}
            
        except Exception as e:
            self.logger.warning(f"[Refinement] Documentation analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_file_structure_with_mcp_tools(self, project_path: str) -> Dict[str, Any]:
        """
        FIXED: Replace phantom tool dependency with actual MCP tools
        
        OLD: hasattr(self.mcp_tools, 'list_project_files') - DOESN'T EXIST
        NEW: Use actual filesystem_list_directory tool
        """
        try:
            # Use actual filesystem tools
            if hasattr(self, '_call_mcp_tool'):
                return await self._call_mcp_tool(
                    "filesystem_list_directory",
                    {"path": project_path, "recursive": True}
                )
            return {"status": "mcp_tool_method_not_available"}
            
        except Exception as e:
            self.logger.warning(f"[Refinement] File structure analysis failed: {e}")
            return {"error": str(e)}
    
    async def _store_iteration_output(
        self, 
        iteration_result: IterationResult, 
        context: ExecutionContext, 
        iteration: int
    ) -> None:
        """Store iteration output in ChromaDB for future refinement"""
        if not self.enable_refinement or not self.chroma_client:
            return
        
        try:
            # Get or create collection for this agent
            collection_name = f"{self.AGENT_ID}_outputs"
            collection = self.chroma_client.get_or_create_collection(collection_name)
            
            # Prepare document metadata
            project_id = context.shared_context.get("project_id", "unknown")
            timestamp = time.time()
            
            metadata = {
                "project_id": project_id,
                "agent_id": self.AGENT_ID,
                "iteration": iteration,
                "timestamp": timestamp,
                "quality_score": iteration_result.quality_score,
                "protocol_used": iteration_result.protocol_used,
                "tools_used": ",".join(iteration_result.tools_used),
                "stage_id": context.stage_info.stage_id if context.stage_info else "unknown"
            }
            
            # Store the output
            document_id = f"{project_id}_{self.AGENT_ID}_{iteration}_{int(timestamp)}"
            collection.add(
                documents=[str(iteration_result.output)],
                metadatas=[metadata],
                ids=[document_id]
            )
            
            self.logger.info(f"[Refinement] Stored iteration {iteration} output in ChromaDB with ID: {document_id}")
            
        except Exception as e:
            self.logger.warning(f"[Refinement] Failed to store iteration output: {e}")

    def _build_refinement_prompt(
        self, 
        original_inputs: Any, 
        refinement_context: Dict[str, Any]
    ) -> str:
        """Build a goal-oriented refinement prompt that compares current output to the actual goal"""
        if not self.enable_refinement or not refinement_context:
            return self._build_standard_prompt(original_inputs)
        
        try:
            previous_outputs = refinement_context.get("previous_outputs", [])
            current_state = refinement_context.get("current_state", {})
            iteration = refinement_context.get("iteration", 0)
            previous_quality = refinement_context.get("previous_quality_score", 0.0)
            
            # Add comprehensive debug logging for refinement debugging
            self.logger.debug(f"[Refinement] Building goal-oriented refinement prompt for iteration {iteration + 1}")
            self.logger.info(f"[Refinement Debug] Building goal-oriented refinement prompt for iteration {iteration + 1}")
            self.logger.debug(f"[Refinement] Previous outputs count: {len(previous_outputs)}")
            self.logger.info(f"[Refinement Debug] Previous outputs count: {len(previous_outputs)}")
            self.logger.debug(f"[Refinement] Previous quality score: {previous_quality}")
            self.logger.info(f"[Refinement Debug] Previous quality score: {previous_quality}")
            
            # Extract goal content from shared context or original inputs
            goal_content = self._extract_goal_content(original_inputs, current_state)
            
            for i, output in enumerate(previous_outputs):
                self.logger.debug(f"[Refinement] Previous output {i+1} metadata: {output.get('metadata', {})}")
                self.logger.info(f"[Refinement Debug] Previous output {i+1} quality: {output.get('quality_score', 'unknown')}")
                content_preview = str(output.get('content', ''))[:200]
                self.logger.debug(f"[Refinement] Previous output {i+1} content preview: {content_preview}...")
            
            # Build goal-oriented refinement prompt
            prompt_parts = [
                f"=== GOAL-ORIENTED REFINEMENT ITERATION {iteration + 1} ===",
                "",
                "ORIGINAL PROJECT GOAL:",
                goal_content,
                "",
                "CURRENT TASK:",
                str(original_inputs),
                "",
            ]
            
            if previous_outputs:
                # Get the most recent output for comparison
                latest_output = previous_outputs[-1]
                
                prompt_parts.extend([
                    "PREVIOUS ATTEMPT ANALYSIS:",
                    f"- Iteration {latest_output['iteration']} achieved quality score: {latest_output['quality_score']:.2f}",
                    f"- Output content: {str(latest_output['content'])[:300]}...",
                    "",
                    "GAP ANALYSIS:",
                    "Compare the previous output against the ORIGINAL PROJECT GOAL above.",
                    "Identify specific ways the output falls short of the goal requirements:",
                    ""
                ])
                
                # Add gap analysis for multiple previous outputs
                for i, output in enumerate(previous_outputs[-2:]):  # Last 2 outputs
                    prompt_parts.extend([
                        f"Previous Output {output['iteration']} gaps:",
                        f"- Quality: {output['quality_score']:.2f} (target: 0.95+)",
                        f"- Content analysis needed against goal requirements",
                        ""
                    ])
            
            if current_state:
                prompt_parts.extend([
                    "CURRENT PROJECT STATE:",
                    f"- File structure: {len(current_state.get('file_structure', {}))} files analyzed",
                    f"- Code analysis: {'Available' if 'code_analysis' in current_state else 'Not available'}",
                    f"- Architecture: {'Available' if 'project_structure' in current_state else 'Not available'}",
                    ""
                ])
            
            prompt_parts.extend([
                "GOAL-ORIENTED REFINEMENT INSTRUCTIONS:",
                "1. COMPARE your previous output directly against the ORIGINAL PROJECT GOAL",
                "2. IDENTIFY specific goal requirements that were missed or inadequately addressed",
                "3. ANALYZE how the current project state constrains or enables goal achievement",
                "4. GENERATE output that directly addresses the identified gaps in goal fulfillment",
                "5. ENSURE your output moves measurably closer to achieving the stated project goal",
                "",
                "Focus on GOAL ALIGNMENT rather than generic quality improvements.",
                "Your output should demonstrably better fulfill the original project goal.",
                "",
                "Generate your goal-aligned refined output now:"
            ])
            
            final_prompt = "\n".join(prompt_parts)
            
            # Add logging to show the full refinement prompt for debugging
            if os.getenv("CHUNGOID_FULL_LLM_LOGGING", "false").lower() == "true":
                self.logger.info(f"[GOAL-ORIENTED REFINEMENT DEBUG] Full Refinement Prompt for iteration {iteration + 1}:")
                self.logger.info("=" * 80)
                self.logger.info(final_prompt)
                self.logger.info("=" * 80)
            else:
                # Show a preview of the refinement prompt
                self.logger.info(f"[GOAL-ORIENTED REFINEMENT DEBUG] Refinement prompt preview (first 300 chars): {final_prompt[:300]}...")
            
            return final_prompt
            
        except Exception as e:
            self.logger.warning(f"[Refinement] Failed to build goal-oriented refinement prompt: {e}")
            return self._build_standard_prompt(original_inputs)

    def _extract_goal_content(self, original_inputs: Any, current_state: Dict[str, Any]) -> str:
        """Extract goal content from various sources in the execution context"""
        
        # Try to get goal from shared context first (most reliable)
        if hasattr(self, 'shared_context') and self.shared_context:
            if 'user_goal' in self.shared_context:
                return str(self.shared_context['user_goal'])
        
        # Try to get goal from current state
        if current_state and 'user_goal' in current_state:
            return str(current_state['user_goal'])
        
        # Try to extract from original inputs if it's a dict
        if isinstance(original_inputs, dict):
            if 'user_goal' in original_inputs:
                return str(original_inputs['user_goal'])
            if 'goal' in original_inputs:
                return str(original_inputs['goal'])
            if 'project_goal' in original_inputs:
                return str(original_inputs['project_goal'])
        
        # Fallback: use original inputs as goal content
        goal_content = str(original_inputs)
        
        # If it's very short, it might not be the actual goal
        if len(goal_content) < 50:
            return "Goal content not available - using task description as proxy for goal alignment"
        
        return goal_content

    def _detect_content_differentiation(self, current_output: Any, previous_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect if current output is meaningfully different from previous iterations"""
        
        if not previous_outputs:
            return {
                "is_different": True,
                "differentiation_score": 1.0,
                "similarity_analysis": "No previous outputs to compare",
                "content_changes": ["Initial generation"]
            }
        
        # Get the most recent previous output for comparison
        latest_previous = previous_outputs[-1]
        prev_content = latest_previous.get('content', {})
        
        # Convert current output to comparable format
        current_content = current_output
        if hasattr(current_output, 'dict'):
            current_content = current_output.dict()
        elif hasattr(current_output, '__dict__'):
            current_content = current_output.__dict__
        
        # Analyze different aspects of content differentiation
        differentiation_analysis = {
            "is_different": False,
            "differentiation_score": 0.0,
            "similarity_analysis": "",
            "content_changes": [],
            "identical_aspects": [],
            "improved_aspects": []
        }
        
        try:
            # 1. Structural differences
            structural_changes = self._analyze_structural_changes(current_content, prev_content)
            
            # 2. Content length differences
            length_changes = self._analyze_content_length_changes(current_content, prev_content)
            
            # 3. Semantic differences (basic text comparison)
            semantic_changes = self._analyze_semantic_changes(current_content, prev_content)
            
            # Calculate overall differentiation score
            total_changes = len(structural_changes) + len(length_changes) + len(semantic_changes)
            
            if total_changes == 0:
                differentiation_analysis.update({
                    "is_different": False,
                    "differentiation_score": 0.0,
                    "similarity_analysis": "Output appears identical to previous iteration",
                    "identical_aspects": ["structure", "content", "length"]
                })
            else:
                # Score based on types and magnitude of changes
                score = min(1.0, total_changes * 0.2)  # Each change type adds 0.2
                
                differentiation_analysis.update({
                    "is_different": score > 0.1,
                    "differentiation_score": score,
                    "similarity_analysis": f"Found {total_changes} types of changes",
                    "content_changes": structural_changes + length_changes + semantic_changes
                })
                
                if structural_changes:
                    differentiation_analysis["improved_aspects"].append("structure")
                if length_changes:
                    differentiation_analysis["improved_aspects"].append("content_length")
                if semantic_changes:
                    differentiation_analysis["improved_aspects"].append("content_semantics")
            
        except Exception as e:
            self.logger.warning(f"Content differentiation analysis failed: {e}")
            # Fallback: assume different if we can't analyze
            differentiation_analysis.update({
                "is_different": True,
                "differentiation_score": 0.5,
                "similarity_analysis": f"Analysis failed: {e}",
                "content_changes": ["Unable to analyze - assuming different"]
            })
        
        return differentiation_analysis

    def _analyze_structural_changes(self, current: Any, previous: Any) -> List[str]:
        """Analyze structural differences between outputs"""
        changes = []
        
        # Compare top-level keys if both are dicts
        if isinstance(current, dict) and isinstance(previous, dict):
            current_keys = set(current.keys())
            previous_keys = set(previous.keys())
            
            new_keys = current_keys - previous_keys
            removed_keys = previous_keys - current_keys
            
            if new_keys:
                changes.append(f"Added keys: {list(new_keys)}")
            if removed_keys:
                changes.append(f"Removed keys: {list(removed_keys)}")
            
            # Check for changes in list lengths
            for key in current_keys & previous_keys:
                if isinstance(current.get(key), list) and isinstance(previous.get(key), list):
                    if len(current[key]) != len(previous[key]):
                        changes.append(f"List length changed for {key}: {len(previous[key])} -> {len(current[key])}")
        
        return changes

    def _analyze_content_length_changes(self, current: Any, previous: Any) -> List[str]:
        """Analyze content length differences"""
        changes = []
        
        # Convert to strings for length comparison
        current_str = str(current)
        previous_str = str(previous)
        
        length_diff = len(current_str) - len(previous_str)
        
        if abs(length_diff) > 50:  # Significant length change
            if length_diff > 0:
                changes.append(f"Content expanded by {length_diff} characters")
            else:
                changes.append(f"Content reduced by {abs(length_diff)} characters")
        
        return changes

    def _analyze_semantic_changes(self, current: Any, previous: Any) -> List[str]:
        """Analyze semantic differences in content"""
        changes = []
        
        # Simple text-based comparison
        current_str = str(current).lower()
        previous_str = str(previous).lower()
        
        # Check for completely different content
        if current_str != previous_str:
            # Calculate rough similarity
            common_chars = sum(1 for c1, c2 in zip(current_str, previous_str) if c1 == c2)
            max_length = max(len(current_str), len(previous_str))
            
            if max_length > 0:
                similarity = common_chars / max_length
                if similarity < 0.8:  # Less than 80% similar
                    changes.append(f"Significant content changes detected (similarity: {similarity:.2f})")
                elif similarity < 0.95:  # Less than 95% similar
                    changes.append(f"Minor content changes detected (similarity: {similarity:.2f})")
        
        return changes

    def _build_standard_prompt(self, inputs: Any) -> str:
        """Build a standard prompt for non-refinement execution"""
        return f"Execute the following task:\n\n{inputs}"

    # ------------------------------------------------------------------
    # Existing Methods (Phase 1-3)
    # ------------------------------------------------------------------

    def _determine_optimal_mode(self, context: ExecutionContext) -> ExecutionMode:
        """Intelligent execution mode selection based on agent capabilities and task complexity"""
        # Phase 3: Enhanced logic for optimal mode selection
        
        # Simple agents get single-pass
        if "simple_operations" in self.CAPABILITIES:
            return ExecutionMode.SINGLE_PASS
        
        # Complex analysis agents benefit from multi-iteration
        elif "complex_analysis" in self.CAPABILITIES:
            # Enable multi-iteration for complex tasks
            context.execution_config.max_iterations = max(context.execution_config.max_iterations, 3)
            return ExecutionMode.MULTI_ITERATION
        
        # Code generation and architecture agents benefit from iteration
        elif any(cap in self.CAPABILITIES for cap in ["code_generation", "architecture_design", "project_analysis"]):
            context.execution_config.max_iterations = max(context.execution_config.max_iterations, 2)
            return ExecutionMode.MULTI_ITERATION
        
        else:
            return ExecutionMode.SINGLE_PASS  # Default for simple agents

    async def _execute_iteration(
        self, 
        context: ExecutionContext, 
        iteration: int
    ) -> IterationResult:
        """Execute a single iteration of the agent's core functionality
        
        Each agent must implement this method to define their core execution logic.
        This is called by the multi-iteration loop in execute().
        
        For refinement-capable agents, the context will include refinement_context
        with previous outputs and current state analysis.
        """
        
        self.logger.info(f"[UAEI] Executing iteration {iteration}")
        
        # Phase 3: Each agent implements this method for their specific logic
        raise NotImplementedError(
            f"Agent {self.__class__.__name__} must implement _execute_iteration() method "
            f"to define their core execution logic for multi-iteration support."
        )

    def _get_preferred_protocol(self, context: ExecutionContext) -> str:
        """Get the preferred protocol for this agent"""
        if context.execution_config.protocol_preference:
            return context.execution_config.protocol_preference
        elif self.PRIMARY_PROTOCOLS:
            return self.PRIMARY_PROTOCOLS[0]
        else:
            return "basic_protocol"

    async def _assess_completion(
        self, 
        iteration_result: IterationResult, 
        completion_criteria: Any,
        context: ExecutionContext
    ) -> CompletionAssessment:
        """Assess whether execution should complete or continue iterating with content differentiation detection"""
        
        # Add comprehensive debug logging for completion assessment
        self.logger.debug(f"[Completion Assessment] Assessing iteration result:")
        self.logger.info(f"[Completion Assessment Debug] Quality score: {iteration_result.quality_score}")
        self.logger.debug(f"[Completion Assessment] Quality score: {iteration_result.quality_score}")
        self.logger.debug(f"[Completion Assessment] Protocol used: {iteration_result.protocol_used}")
        self.logger.debug(f"[Completion Assessment] Tools used: {iteration_result.tools_used}")
        self.logger.debug(f"[Completion Assessment] Output preview: {str(iteration_result.output)[:200]}...")
        
        # Check quality threshold (default: 0.95 for better refinement)
        quality_threshold = getattr(completion_criteria, 'quality_threshold', 0.95)
        self.logger.debug(f"[Completion Assessment] Quality threshold: {quality_threshold}")
        self.logger.info(f"[Completion Assessment Debug] Quality threshold: {quality_threshold}")
        
        # ENHANCED: Content differentiation detection for refinement iterations
        gaps_identified = []
        recommendations = []
        
        # Check for refinement context to detect content stagnation
        refinement_context = context.shared_context.get("refinement_context")
        if self.enable_refinement and refinement_context:
            previous_outputs = refinement_context.get("previous_outputs", [])
            
            if previous_outputs:
                # Detect content differentiation
                differentiation_analysis = self._detect_content_differentiation(
                    iteration_result.output, previous_outputs
                )
                
                self.logger.info(f"[Content Differentiation] Analysis: {differentiation_analysis['similarity_analysis']}")
                self.logger.info(f"[Content Differentiation] Score: {differentiation_analysis['differentiation_score']:.3f}")
                
                # If content is identical or very similar, flag as problematic
                if not differentiation_analysis["is_different"]:
                    gaps_identified.append("Output identical to previous iteration - refinement not producing improvements")
                    recommendations.extend([
                        "Modify refinement prompts to enforce specific changes",
                        "Add explicit differentiation requirements",
                        "Consider alternative generation strategies"
                    ])
                    
                    # Lower effective quality score for identical content
                    effective_quality = iteration_result.quality_score * 0.7
                    self.logger.warning(f"[Content Differentiation] Identical content detected - reducing effective quality from {iteration_result.quality_score:.3f} to {effective_quality:.3f}")
                    
                    return CompletionAssessment(
                        is_complete=False,
                        quality_score=effective_quality,
                        reason=CompletionReason.QUALITY_THRESHOLD_NOT_MET,
                        gaps_identified=gaps_identified,
                        recommendations=recommendations
                    )
                
                elif differentiation_analysis["differentiation_score"] < 0.3:
                    gaps_identified.append("Minimal changes from previous iteration - refinement effectiveness low")
                    recommendations.append("Increase refinement specificity and goal alignment")
        
        # Standard quality threshold check
        if iteration_result.quality_score >= quality_threshold:
            self.logger.debug(f"[Completion Assessment] Quality threshold met ({iteration_result.quality_score} >= {quality_threshold})")
            self.logger.info(f"[Completion Assessment Debug] Quality threshold MET ({iteration_result.quality_score} >= {quality_threshold})")
            return CompletionAssessment(
                is_complete=True,
                quality_score=iteration_result.quality_score,
                reason=CompletionReason.QUALITY_THRESHOLD_MET,
                gaps_identified=[],
                recommendations=[]
            )
        
        self.logger.debug(f"[Completion Assessment] Quality threshold not met ({iteration_result.quality_score} < {quality_threshold})")
        self.logger.info(f"[Completion Assessment Debug] Quality threshold NOT MET ({iteration_result.quality_score} < {quality_threshold})")
        
        # Add standard gaps if none were identified from content differentiation
        if not gaps_identified:
            gaps_identified = ["Quality score below threshold"]
        if not recommendations:
            recommendations = ["Improve output quality", "Add more detail", "Enhance analysis"]
        
        return CompletionAssessment(
            is_complete=False,
            quality_score=iteration_result.quality_score,
            reason=CompletionReason.QUALITY_THRESHOLD_NOT_MET,
            gaps_identified=gaps_identified,
            recommendations=recommendations
        )

    async def _enhance_context_for_next_iteration(
        self,
        context: ExecutionContext, 
        iteration_result: IterationResult, 
        completion_assessment: CompletionAssessment
    ) -> ExecutionContext:
        """Enhance context for next iteration - Phase 3 implementation"""
        
        # Create enhanced context with additional information
        enhanced_shared_context = context.shared_context.copy()
        
        # Add iteration history and results
        if "iteration_history" not in enhanced_shared_context:
            enhanced_shared_context["iteration_history"] = []
        
        enhanced_shared_context["iteration_history"].append({
            "iteration": len(enhanced_shared_context["iteration_history"]) + 1,
            "quality_score": iteration_result.quality_score,
            "output_summary": str(iteration_result.output)[:200] + "..." if len(str(iteration_result.output)) > 200 else str(iteration_result.output),
            "tools_used": iteration_result.tools_used,
            "gaps_identified": completion_assessment.gaps_identified,
            "recommendations": completion_assessment.recommendations
        })
        
        # Add gap analysis and improvement suggestions
        enhanced_shared_context["identified_gaps"] = completion_assessment.gaps_identified
        enhanced_shared_context["improvement_recommendations"] = completion_assessment.recommendations
        enhanced_shared_context["previous_quality_score"] = iteration_result.quality_score
        
        # Enhance inputs with context from previous iteration
        enhanced_inputs = context.inputs
        if hasattr(enhanced_inputs, 'dict'):
            enhanced_inputs_dict = enhanced_inputs.dict()
            enhanced_inputs_dict["previous_iteration_context"] = {
                "quality_score": iteration_result.quality_score,
                "gaps_identified": completion_assessment.gaps_identified,
                "recommendations": completion_assessment.recommendations
            }
            # Note: This assumes inputs can be reconstructed from dict
            # Individual agents may need to handle this differently
        
        return ExecutionContext(
            inputs=enhanced_inputs,
            shared_context=enhanced_shared_context,
            stage_info=context.stage_info,
            execution_config=context.execution_config
        )
    
    def _create_final_result_from_iterations(
        self,
        best_result: IterationResult,
        execution_mode: ExecutionMode,
        max_iterations: int,
        execution_time: float,
        all_tools_used: List[str]
    ) -> AgentExecutionResult:
        """Create final result when max iterations reached"""
        return AgentExecutionResult(
            output=best_result.output,
            execution_metadata=ExecutionMetadata(
                mode=execution_mode,
                protocol_used=best_result.protocol_used,
                execution_time=execution_time,
                iterations_planned=max_iterations,
                tools_utilized=list(set(all_tools_used))  # Deduplicate tools
            ),
            iterations_completed=max_iterations,
            completion_reason=CompletionReason.MAX_ITERATIONS_REACHED,
            quality_score=best_result.quality_score,
            protocol_used=best_result.protocol_used
        )

    async def _enhanced_discovery_with_universal_tools(self, inputs: Any, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        UNIVERSAL TOOL ACCESS PATTERN: Complete implementation of enhanced_mcp.md
        
        This method provides a standardized interface for all agents to access
        the complete MCP tool ecosystem with intelligent selection and orchestration.
        
        Returns comprehensive analysis using:
        1. Universal tool discovery (no filtering)
        2. Intelligent tool selection based on agent capabilities
        3. Multi-tool orchestrated analysis
        4. Fallback strategies for resilience
        5. Performance optimization
        
        Args:
            inputs: Agent-specific input data
            shared_context: Shared execution context
            
        Returns:
            Dict containing comprehensive analysis results from multiple MCP tools
        """
        
        try:
            # 1. UNIVERSAL TOOL DISCOVERY: Get ALL available tools (no filtering)
            self.logger.info("[MCP] Starting universal tool discovery")
            tool_discovery = await self._get_all_available_mcp_tools()
            
            if not tool_discovery["discovery_successful"]:
                self.logger.error("[MCP] Tool discovery failed - falling back to limited functionality")
                return {"error": "Tool discovery failed", "limited_functionality": True}
            
            all_tools = tool_discovery["tools"]
            self.logger.info(f"[MCP] Discovered {len(all_tools)} tools across {len(tool_discovery['categories'])} categories")
            
            # 2. INTELLIGENT TOOL SELECTION: Based on context and capabilities
            selected_tools = self._intelligently_select_tools(all_tools, inputs, shared_context)
            self.logger.info(f"[MCP] Selected {len(selected_tools)} tools for enhanced discovery")
            
            # Initialize comprehensive analysis results
            analysis_results = {
                "discovery_metadata": {
                    "agent_id": self.AGENT_ID,
                    "timestamp": time.time(),
                    "total_tools_available": len(all_tools),
                    "tools_selected": len(selected_tools),
                    "selected_tool_names": list(selected_tools.keys()),
                    "analysis_type": "universal_enhanced_discovery"
                }
            }
            
            # 3. FILESYSTEM ANALYSIS: Comprehensive project structure analysis
            if "filesystem_project_scan" in selected_tools:
                self.logger.info("[MCP] Executing filesystem analysis")
                project_path = shared_context.get("project_root_path", getattr(inputs, 'project_path', '.'))
                
                project_analysis = await self._safe_tool_call_with_fallback(
                    "filesystem_project_scan",
                    {
                        "path": str(project_path),
                        "include_stats": True,
                        "detect_project_type": True,
                        "analyze_structure": True
                    },
                    ["filesystem_list_directory", "filesystem_read_file"]
                )
                analysis_results["project_analysis"] = project_analysis
            
            # 4. CONTENT ANALYSIS: Deep content structure analysis
            if "content_analyze_structure" in selected_tools and analysis_results.get("project_analysis", {}).get("success"):
                self.logger.info("[MCP] Executing content structure analysis")
                content_analysis = await self._call_mcp_tool(
                    "content_analyze_structure",
                    {"content": analysis_results["project_analysis"]["result"]}
                )
                analysis_results["content_analysis"] = content_analysis
            
            # 5. INTELLIGENCE ANALYSIS: Adaptive learning and pattern analysis
            if "adaptive_learning_analyze" in selected_tools:
                self.logger.info("[MCP] Executing intelligence analysis")
                intelligence_context = {
                    "agent_id": self.AGENT_ID,
                    "agent_capabilities": getattr(self, 'CAPABILITIES', []),
                    "project_analysis": analysis_results.get("project_analysis", {}),
                    "content_analysis": analysis_results.get("content_analysis", {}),
                    "inputs_summary": str(inputs)[:500]  # Truncated for safety
                }
                
                intelligence_analysis = await self._safe_tool_call_with_fallback(
                    "adaptive_learning_analyze",
                    {"context": intelligence_context, "domain": self.AGENT_ID},
                    ["analyze_historical_patterns", "get_real_time_performance_analysis"]
                )
                analysis_results["intelligence_analysis"] = intelligence_analysis
            
            # 6. HISTORICAL CONTEXT: ChromaDB query for previous work
            if "chromadb_query_documents" in selected_tools:
                self.logger.info("[MCP] Querying historical context")
                project_id = shared_context.get('project_id', getattr(inputs, 'project_id', 'unknown'))
                historical_query = f"agent:{self.AGENT_ID} project:{project_id}"
                
                historical_context = await self._safe_tool_call_with_fallback(
                    "chromadb_query_documents",
                    {
                        "query": historical_query,
                        "limit": 10,
                        "include_metadata": True,
                        "collection_name": f"project_{project_id}_artifacts"
                    },
                    ["chroma_peek_collection", "chroma_get_documents"]
                )
                analysis_results["historical_context"] = historical_context
            
            # 7. ENVIRONMENT VALIDATION: System environment analysis
            if "terminal_get_environment" in selected_tools:
                self.logger.info("[MCP] Validating environment")
                environment_info = await self._safe_tool_call_with_fallback(
                    "terminal_get_environment",
                    {},
                    ["terminal_sandbox_status", "terminal_check_permissions"]
                )
                analysis_results["environment_info"] = environment_info
            
            # 8. PREDICTIVE ANALYSIS: Failure prediction and risk assessment
            if "predict_potential_failures" in selected_tools:
                self.logger.info("[MCP] Executing predictive analysis")
                prediction_context = {
                    "agent_context": {
                        "id": self.AGENT_ID,
                        "capabilities": getattr(self, 'CAPABILITIES', []),
                        "category": getattr(self, 'CATEGORY', 'unknown')
                    },
                    "analysis_results": analysis_results,
                    "execution_context": shared_context
                }
                
                failure_predictions = await self._call_mcp_tool(
                    "predict_potential_failures",
                    {"context": prediction_context}
                )
                analysis_results["failure_predictions"] = failure_predictions
            
            # 9. PERFORMANCE OPTIMIZATION: Real-time performance analysis
            if "get_real_time_performance_analysis" in selected_tools:
                self.logger.info("[MCP] Executing performance analysis")
                performance_analysis = await self._call_mcp_tool(
                    "get_real_time_performance_analysis",
                    {
                        "agent_id": self.AGENT_ID,
                        "context": analysis_results,
                        "metrics": ["execution_time", "tool_usage", "success_rate"]
                    }
                )
                analysis_results["performance_analysis"] = performance_analysis
            
            # 10. TOOL COMPOSITION: Recommendations for optimal tool usage
            if "get_tool_composition_recommendations" in selected_tools:
                self.logger.info("[MCP] Getting tool composition recommendations")
                composition_context = {
                    "agent_id": self.AGENT_ID,
                    "task_type": getattr(self, 'CATEGORY', 'unknown'),
                    "current_tools": list(selected_tools.keys()),
                    "analysis_results": analysis_results
                }
                
                tool_recommendations = await self._call_mcp_tool(
                    "get_tool_composition_recommendations",
                    {"context": composition_context}
                )
                analysis_results["tool_recommendations"] = tool_recommendations
            
            # 11. DYNAMIC CONTENT GENERATION: Intelligent insights generation
            if "content_generate_dynamic" in selected_tools:
                self.logger.info("[MCP] Generating dynamic insights")
                content_context = {
                    "analysis_results": analysis_results,
                    "agent_context": {
                        "id": self.AGENT_ID,
                        "capabilities": getattr(self, 'CAPABILITIES', []),
                        "version": getattr(self, 'AGENT_VERSION', 'unknown')
                    },
                    "generation_type": "comprehensive_insights"
                }
                
                dynamic_insights = await self._call_mcp_tool(
                    "content_generate_dynamic",
                    {"context": content_context, "content_type": "analysis_insights"}
                )
                analysis_results["dynamic_insights"] = dynamic_insights
            
            # 12. COMPREHENSIVE ANALYSIS SUMMARY
            successful_analyses = sum(1 for key, value in analysis_results.items() 
                                    if isinstance(value, dict) and value.get("success"))
            
            analysis_results["comprehensive_summary"] = {
                "total_analyses_attempted": len([k for k in analysis_results.keys() if k != "discovery_metadata"]),
                "successful_analyses": successful_analyses,
                "analysis_coverage": successful_analyses / len(selected_tools) if selected_tools else 0,
                "analysis_quality": self._assess_analysis_quality(successful_analyses, len(selected_tools)),
                "recommended_next_steps": self._generate_next_step_recommendations(analysis_results),
                "tool_performance": self._assess_tool_performance(analysis_results)
            }
            
            self.logger.info(f"[MCP] Universal discovery completed: {successful_analyses} successful analyses")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"[MCP] Universal discovery failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "analysis_type": "universal_discovery_failed",
                "fallback_analysis": True,
                "timestamp": time.time()
            }
    
    def _assess_analysis_quality(self, successful_count: int, total_count: int) -> str:
        """Assess the quality of the analysis based on success rate"""
        if total_count == 0:
            return "no_tools_available"
        
        success_rate = successful_count / total_count
        if success_rate >= 0.8:
            return "excellent"
        elif success_rate >= 0.6:
            return "good"
        elif success_rate >= 0.4:
            return "partial"
        else:
            return "limited"
    
    def _generate_next_step_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate intelligent next step recommendations based on analysis"""
        recommendations = []
        
        # Check if project analysis was successful
        if analysis_results.get("project_analysis", {}).get("success"):
            recommendations.append("Project structure analysis completed - proceed with detailed planning")
        else:
            recommendations.append("Project analysis failed - consider manual project inspection")
        
        # Check intelligence analysis
        if analysis_results.get("intelligence_analysis", {}).get("success"):
            recommendations.append("Intelligence insights available - incorporate into decision making")
        
        # Check historical context
        if analysis_results.get("historical_context", {}).get("success"):
            recommendations.append("Historical context retrieved - leverage previous learnings")
        
        # Check failure predictions
        if analysis_results.get("failure_predictions", {}).get("success"):
            recommendations.append("Risk assessment completed - implement preventive measures")
        
        return recommendations if recommendations else ["Continue with standard execution flow"]
    
    def _assess_tool_performance(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the performance of tools used in the analysis"""
        tool_performance = {
            "high_performing_tools": [],
            "failed_tools": [],
            "performance_score": 0.0
        }
        
        successful_tools = 0
        total_tools = 0
        
        for key, value in analysis_results.items():
            if key in ["discovery_metadata", "comprehensive_summary"]:
                continue
                
            total_tools += 1
            if isinstance(value, dict) and value.get("success"):
                successful_tools += 1
                tool_performance["high_performing_tools"].append(key)
            else:
                tool_performance["failed_tools"].append(key)
        
        tool_performance["performance_score"] = successful_tools / total_tools if total_tools > 0 else 0.0
        return tool_performance 