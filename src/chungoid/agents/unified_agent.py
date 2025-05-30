"""UnifiedAgent - Optimized UAEI Base Class

Single interface for ALL agent execution with universal inheritance methods.
Eliminates dual interface complexity and code duplication across agents.
Enhanced with project-agnostic capabilities for ANY technology stack.

Optimizations:
- Consolidated tool management system
- Universal pattern matching and error handling  
- Unified discovery methods
- Intelligent enhancement system
- No duplicate code blocks
"""

from __future__ import annotations

import logging
import time
import os
import asyncio
import json
import re
import fnmatch
from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Optional, Dict, Type, Union, Callable
from enum import Enum
import inspect
from functools import wraps
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, ValidationError

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

__all__ = ["UnifiedAgent", "JsonValidationConfig", "JsonExtractionStrategy", "UniversalPatternMatcher", "universal_error_handler"]


# ========================================
# EFFICIENT DISCOVERY SERVICE - BIG BANG PERFORMANCE FIX
# ========================================

class EfficientDiscoveryService:
    """
    PERFORMANCE FIX: Single scan + in-memory pattern matching
    
    Replaces 52+ individual filesystem_glob_search calls with:
    1. ONE directory scan using filesystem_list_directory
    2. Cached file list in memory  
    3. Fast in-memory pattern matching
    4. Cache invalidation only when needed
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes TTL
    
    async def discover_files_by_patterns(
        self,
        project_path: str,
        patterns: List[str],
        recursive: bool = True,
        call_mcp_tool_func: Callable = None
    ) -> Dict[str, List[str]]:
        """Discover files matching patterns using cached directory scan."""
        
        if not call_mcp_tool_func:
            return {pattern: [] for pattern in patterns}
        
        # Performance optimization: single directory scan instead of 52+ individual searches
        file_list = await self._get_cached_file_list(project_path, call_mcp_tool_func, recursive)
        
        # Match patterns in memory using file list
        results = {}
        for pattern in patterns:
            matching_files = [f for f in file_list if self._match_pattern(f, pattern)]
            results[pattern] = matching_files
        
        return results
    
    async def _get_cached_file_list(
        self, 
        project_path: str, 
        call_mcp_tool_func: Callable,
        recursive: bool = True,
        force_refresh: bool = False
    ) -> List[str]:
        """Get file list from cache or scan directory once - FIXED VERSION."""
        # Get logger instance for this method since this is a standalone service class
        import logging
        logger_instance = logging.getLogger(__name__)
        
        cache_key = f"{project_path}:{recursive}"
        current_time = time.time()
        
        # CRITICAL FIX: Properly handle force_refresh by clearing cache FIRST
        if force_refresh:
            # Immediately clear cache when force_refresh is requested
            self.clear_cache(project_path)
            logger_instance.info(f"DISCOVERY_SERVICE: Force refresh - cleared cache for {project_path}")
        
        # Check cache validity AFTER potential clearing
        cache_valid = (cache_key in self._cache and 
                      cache_key in self._cache_timestamps and
                      current_time - self._cache_timestamps[cache_key] < self._cache_ttl)
        
        if cache_valid and not force_refresh:
            files = self._cache[cache_key].get("files", [])
            logger_instance.info(f"PERFORMANCE_BOOST: Cache HIT: {len(files)} files from cache for {project_path}")
            return files
        
        # Determine refresh reason for logging
        if force_refresh:
            reason = "FORCED_REFRESH"
        elif not cache_valid:
            reason = "CACHE_EXPIRED" if cache_key in self._cache else "CACHE_MISS"
        else:
            reason = "UNKNOWN"
        
        # Enhanced logic: Use cache coordination for proper refresh
        try:
            # Try to import cache coordinator
            try:
                from ..utils.cache_coordination_fix import coordinate_cache_refresh
                
                logger_instance.info(f"DISCOVERY_SERVICE: Using coordinated refresh - Reason: {reason}")
                
                # Use coordinated cache refresh instead of direct MCP call
                result = await coordinate_cache_refresh(
                    project_path,
                    call_mcp_tool_func,
                    discovery_service=self,
                    reason=f"discovery_{reason.lower()}"
                )
                
            except ImportError:
                # This should not happen anymore since we implemented the module
                logger_instance.error("DISCOVERY_SERVICE: Cache coordination module missing - this should not happen")
                result = await call_mcp_tool_func("filesystem_list_directory", {
                    "directory_path": project_path,
                    "recursive": recursive,
                    "include_files": True,
                    "include_directories": False,
                    "max_depth": 10,
                    "_force_refresh": force_refresh,
                    "_cache_bust": current_time
                })
            
            if result and result.get("success"):
                files = []
                items = result.get("items", [])
                
                for item in items:
                    if item.get("type") == "file":
                        # Store both absolute and relative paths
                        if "path" in item:
                            files.append(item["path"])
                        if "relative_path" in item:
                            files.append(item["relative_path"])
                
                # Cache the results
                self._cache[cache_key] = {"files": files}
                self._cache_timestamps[cache_key] = current_time
                
                logger_instance.info(f"DISCOVERY_SERVICE: Successfully cached {len(files)} files for {project_path}")
                return files
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger_instance.error(f"DISCOVERY_SERVICE: Directory scan failed for {project_path}: {error_msg}")
                
        except Exception as e:
            logger_instance.error(f"DISCOVERY_SERVICE: Directory scan exception for {project_path}: {e}")
        
        # Return empty list on any failure
        return []
    
    def _match_pattern(self, file_path: str, pattern: str) -> bool:
        """Fast in-memory pattern matching."""
        if not file_path or not pattern:
            return False
        
        # Extract filename for pattern matching
        filename = Path(file_path).name
        
        # Use fnmatch for glob patterns
        if "*" in pattern or "?" in pattern:
            return fnmatch.fnmatch(filename.lower(), pattern.lower())
        
        # Simple substring match for non-glob patterns  
        return pattern.lower() in filename.lower()
    
    def clear_cache(self, project_path: Optional[str] = None):
        """Clear cache for specific project or all projects."""
        if project_path:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(project_path)]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()

# Global service instance
_discovery_service = EfficientDiscoveryService()


# ========================================
# CONSOLIDATED INFRASTRUCTURE
# ========================================

class UniversalPatternMatcher:
    """Centralized pattern matching for all discovery operations."""
    
    @staticmethod
    async def find_files_by_patterns(
        project_path: str, 
        patterns: List[str], 
        recursive: bool = False,
        call_mcp_tool_func: Callable = None
    ) -> Dict[str, List[str]]:
        """
        PERFORMANCE FIX: Use efficient discovery service instead of individual searches.
        
        OLD: 52+ individual filesystem_glob_search calls
        NEW: 1 directory scan + in-memory pattern matching
        """
        return await _discovery_service.discover_files_by_patterns(
            project_path, patterns, recursive, call_mcp_tool_func
        )
    
    @staticmethod
    def match_pattern(file_path: str, pattern: str) -> bool:
        """Single pattern matching implementation."""
        return _discovery_service._match_pattern(file_path, pattern)
    
    @staticmethod
    def categorize_by_patterns(item_name: str, category_patterns: Dict[str, List[str]]) -> str:
        """Categorize items using pattern matching."""
        item_lower = item_name.lower()
        
        for category, patterns in category_patterns.items():
            if any(pattern.lower() in item_lower for pattern in patterns):
                return category
        
        return "unknown"


def universal_error_handler(operation_name: str, default_return: Any = None):
    """Decorator to standardize error handling across all methods."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.error(f"[{operation_name}] Failed: {e}")
                
                if operation_name in ["Context Retrieval", "Project Discovery", "Domain Analysis"]:
                    return default_return or {}
                elif operation_name in ["File Discovery", "Tool Discovery"]:
                    return default_return or {"success": False, "error": str(e)}
                else:
                    return default_return or {"success": False, "error": str(e), "operation": operation_name}
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.error(f"[{operation_name}] Failed: {e}")
                return default_return or {}
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class JsonExtractionStrategy(Enum):
    """Strategies for extracting JSON from LLM responses."""
    MARKDOWN_FIRST = "markdown_first"
    BRACKET_MATCHING = "bracket_matching"
    MULTI_STRATEGY = "multi_strategy"
    REPAIR_ENABLED = "repair_enabled"


class JsonValidationConfig(BaseModel):
    """Configuration for JSON validation in agents."""
    
    extraction_strategy: JsonExtractionStrategy = JsonExtractionStrategy.MULTI_STRATEGY
    enable_json_repair: bool = True
    max_extraction_retries: int = 3
    request_json_format: bool = True
    enable_json_mode: bool = True
    use_tool_calling: bool = True
    enable_schema_validation: bool = True
    strict_validation: bool = False
    allow_partial_validation: bool = True
    enable_llm_repair: bool = True
    max_repair_attempts: int = 2
    fallback_to_text: bool = True
    cache_extracted_json: bool = True
    validate_async: bool = False


# ========================================
# MAIN UNIFIED AGENT CLASS
# ========================================

class UnifiedAgent(BaseModel, ABC):
    """
    Optimized unified interface for ALL agent execution.
    Eliminates dual interface complexity and code duplication.
    Provides universal inheritance methods for all 9 agents.
    """
    
    # Required class metadata
    AGENT_ID: ClassVar[str]
    AGENT_VERSION: ClassVar[str] 
    PRIMARY_PROTOCOLS: ClassVar[List[str]]
    CAPABILITIES: ClassVar[List[str]]
    
    # Core components
    llm_provider: LLMProvider = Field(..., description="LLM provider for AI capabilities")
    prompt_manager: PromptManager = Field(..., description="Prompt manager for templates")
    
    # Enhanced capabilities
    enable_refinement: bool = Field(default=True, description="Enable intelligent refinement")
    mcp_tools: Optional[Any] = Field(default=None, description="MCP tools registry")
    chroma_client: Optional[Any] = Field(default=None, description="ChromaDB client")
    
    # JSON validation
    json_validation_config: JsonValidationConfig = Field(default_factory=JsonValidationConfig)
    json_cache: Dict[str, Any] = Field(default_factory=dict)
    
    # Internal
    logger: Optional[logging.Logger] = Field(default=None)
    
    # Model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Iteration tracking for performance optimization
    iteration_cache: Dict[str, Any] = Field(default_factory=dict, description="Per-execution iteration cache")
    current_execution_id: Optional[str] = Field(default=None, description="Current execution identifier")
    
    # CONSOLIDATED TOOL MANAGEMENT SYSTEM
    TOOL_CATEGORIES: ClassVar[Dict[str, Any]] = {
        "chromadb": {
            "keywords": ['chroma', 'database', 'collection', 'document', 'query'],
            "placeholder": {"collections": [], "documents": [], "metadata": {}},
            "aliases": {
                "chromadb_query_documents": "chroma_query_documents",
                "chromadb_get_document": "chroma_get_documents", 
                "chromadb_similarity_search": "chroma_query_documents",
            }
        },
        "filesystem": {
            "keywords": ['filesystem', 'file', 'directory', 'read', 'write'],
            "placeholder": {"files": [], "directories": [], "total_size": 0},
            "aliases": {}
        },
        "terminal": {
            "keywords": ['terminal', 'command', 'execute', 'environment'],
            "placeholder": {"output": "placeholder output", "exit_code": 0},
            "aliases": {
                "terminal_set_environment_variable": "terminal_execute_command",
                "terminal_run_script": "terminal_execute_command",
            }
        },
        "content": {
            "keywords": ['content', 'web', 'extract', 'generate'],
            "placeholder": {"content": "placeholder content", "type": "text"},
            "aliases": {
                "content_extract_text": "web_content_extract",
                "content_transform_format": "content_generate_dynamic",
            }
        },
        "intelligence": {
            "keywords": ['intelligence', 'learning', 'analyze', 'predict', 'performance'],
            "placeholder": {"analysis": "placeholder analysis", "recommendations": []},
            "aliases": {
                "optimize_execution_strategy": "optimize_agent_resolution_mcp",
                "recommend_tools_for_task": "discover_tools",
            }
        },
        "registry": {
            "keywords": ['registry'],
            "placeholder": {"tools": [], "metadata": {}},
            "aliases": {}
        }
    }

    # CONSOLIDATED DISCOVERY PATTERNS
    DISCOVERY_PATTERNS: ClassVar[Dict[str, Any]] = {
        "environment": ["*.env*", ".env*", "environment.*", "config.*", "docker*", "compose*", "Dockerfile*"],
        "dependencies": {
            "python": ["requirements*.txt", "pyproject.toml", "setup.py", "Pipfile"],
            "javascript": ["package.json", "yarn.lock", "package-lock.json"],
            "java": ["pom.xml", "build.gradle"],
            "csharp": ["*.csproj", "packages.config"],
            "ruby": ["Gemfile", "*.gemspec"],
            "go": ["go.mod", "go.sum"],
            "rust": ["Cargo.toml", "Cargo.lock"],
            "php": ["composer.json", "composer.lock"],
            "generic": ["*.lock", "*requirements*", "*dependencies*"]
        },
        "structure": ["src/*", "lib/*", "api/*", "frontend/*", "backend/*", "tests/*", "docs/*"]
    }

    # ENHANCEMENT STRATEGIES
    ENHANCEMENT_STRATEGIES: ClassVar[Dict[str, Any]] = {
        "environment": {"focus_areas": ["security", "performance"], "type": "additive"},
        "dependencies": {"focus_areas": ["security", "versions"], "type": "merge"},
        "documentation": {"focus_areas": ["completeness", "clarity"], "type": "additive"},
        "source_code": {"focus_areas": ["performance", "maintainability"], "type": "replacement"}
    }

    def __init__(self, **data):
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        if self.enable_refinement:
            self._initialize_refinement_capabilities()

    def get_id(self) -> str:
        """Get the agent's unique identifier"""
        return self.AGENT_ID

    def _initialize_refinement_capabilities(self):
        """Initialize MCP tools and ChromaDB for refinement capabilities"""
        try:
            if self.mcp_tools is None:
                from chungoid.mcp_tools import get_mcp_tools_registry
                self.mcp_tools = get_mcp_tools_registry()
            
            if self.chroma_client is None:
                import chromadb
                self.chroma_client = chromadb.Client()
                
        except Exception as e:
            self.logger.warning(f"[Refinement] Failed to initialize: {e}")
            self.enable_refinement = False

    # ========================================
    # CORE EXECUTION METHODS
    # ========================================

    async def execute(
        self, 
        context: ExecutionContext,
        mode: ExecutionMode = ExecutionMode.OPTIMAL
    ) -> AgentExecutionResult:
        """Optimized agent execution with unified iteration management."""
        
        start_time = time.time()
        iteration_results: List[IterationResult] = []
        
        # Initialize execution tracking to prevent repetitive discovery
        execution_id = f"{self.AGENT_ID}_{int(time.time() * 1000)}"
        self.current_execution_id = execution_id
        self.iteration_cache.clear()  # Fresh cache for new execution
        
        self.logger.info(f"[UNIFIED] Starting execution {execution_id} with mode {mode.value}")
        
        # PERFORMANCE FIX: Do discovery ONCE per execution, not per iteration
        if "discovery_results" not in self.iteration_cache:
            self.logger.info(f"[PERFORMANCE] Performing discovery for execution {execution_id}")
            self.iteration_cache["discovery_start_time"] = time.time()
        
        max_iterations = self._determine_max_iterations(context, mode)
        
        # Enhanced iteration loop with completion assessment
        for iteration in range(1, max_iterations + 1):
            try:
                self.logger.info(f"[UNIFIED] Iteration {iteration}/{max_iterations}")
                
                # Execute iteration with performance optimization
                result = await self._execute_iteration_optimized(context, iteration)
                iteration_results.append(result)
                
                # Enhanced completion assessment
                completion = self._assess_completion(iteration_results, context, iteration, max_iterations)
                
                if completion.is_complete:
                    self.logger.info(f"[UNIFIED] Early completion after {iteration} iterations: {completion.reason}")
                    break
                
                # Log iteration quality
                self.logger.info(f"[UNIFIED] Iteration {iteration} quality: {result.quality_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"[UNIFIED] Iteration {iteration} failed: {e}")
                
                # Create error result
                error_result = IterationResult(
                    output={"error": str(e), "iteration": iteration},
                    quality_score=0.0,
                    tools_used=[],
                    protocol_used="error_handling"
                )
                iteration_results.append(error_result)
                
                # Decide whether to continue or fail
                if iteration >= max_iterations * 0.8:  # Fail if we're near the end
                    break

        # Performance logging
        execution_time = time.time() - start_time
        discovery_time = self.iteration_cache.get("discovery_total_time", 0)
        
        self.logger.info(f"[PERFORMANCE] Execution {execution_id} completed:")
        self.logger.info(f"[PERFORMANCE] - Total time: {execution_time:.2f}s")
        self.logger.info(f"[PERFORMANCE] - Discovery time: {discovery_time:.2f}s")
        self.logger.info(f"[PERFORMANCE] - Iterations: {len(iteration_results)}")
        
        # Build final result
        final_result = self._build_final_result(iteration_results, context, execution_time)
        
        # Clear execution state
        self.current_execution_id = None
        self.iteration_cache.clear()
        
        return final_result

    async def _execute_iteration_optimized(self, context: ExecutionContext, iteration: int) -> IterationResult:
        """Execute iteration with discovery caching optimization."""
        
        # **CRITICAL FIX**: Check cache bypass flag from context
        cache_bypassed = getattr(context, 'cache_bypassed', False) or context.shared_context.get('cache_bypassed', False)
        force_refresh = cache_bypassed or (iteration > 1)  # Force refresh after first iteration
        
        # Check if we've already done discovery for this execution AND it's not bypassed
        if "discovery_results" not in self.iteration_cache or force_refresh:
            # Discovery needed - either first iteration or cache bypass requested
            discovery_start = time.time()
            
            if force_refresh:
                self.logger.info(f"[PERFORMANCE] Force refresh discovery for iteration {iteration} (cache_bypassed={cache_bypassed})")
            else:
                self.logger.info(f"[PERFORMANCE] Initial discovery for execution {self.current_execution_id}")
            
            # Clear existing cache entries to force fresh data
            if force_refresh and "discovery_results" in self.iteration_cache:
                self.iteration_cache.pop("discovery_results", None)
                self.iteration_cache.pop("technology_discovery", None)
                self.logger.info("[CACHE] Cleared existing discovery cache for fresh scan")
            
            # Cache the discovery results for the entire execution
            try:
                if hasattr(self, '_universal_discovery'):
                    discovery_results = await self._universal_discovery(
                        context.shared_context.get("project_root_path", "."),
                        ["environment", "dependencies", "structure", "patterns"]
                    )
                    self.iteration_cache["discovery_results"] = discovery_results
                    
                if hasattr(self, '_universal_technology_discovery'):
                    tech_discovery = await self._universal_technology_discovery(
                        context.shared_context.get("project_root_path", ".")
                    )
                    self.iteration_cache["technology_discovery"] = tech_discovery
                    
            except Exception as e:
                self.logger.warning(f"Discovery failed, using empty cache: {e}")
                self.iteration_cache["discovery_results"] = {}
                self.iteration_cache["technology_discovery"] = {}
            
            discovery_time = time.time() - discovery_start
            self.iteration_cache["discovery_total_time"] = discovery_time
            
            self.logger.info(f"[PERFORMANCE] Discovery completed in {discovery_time:.2f}s for execution {self.current_execution_id}")
        else:
            # Subsequent iterations - use cached discovery only if not bypassed
            self.logger.info(f"[PERFORMANCE] Using cached discovery for iteration {iteration}")
        
        # Now call the original _execute_iteration with cached context
        return await self._execute_iteration(context, iteration)

    def _build_final_result(self, iteration_results: List[IterationResult], context: ExecutionContext, execution_time: float) -> AgentExecutionResult:
        """Build the final result from all iterations."""
        
        if not iteration_results:
            return AgentExecutionResult(
                output={"error": "No iterations completed"},
                execution_metadata=ExecutionMetadata(
                    mode=ExecutionMode.OPTIMAL,
                    protocol_used="unified",
                    execution_time=execution_time,
                    iterations_planned=0,
                    tools_utilized=[]
                ),
                iterations_completed=0,
                completion_reason=CompletionReason.ERROR_OCCURRED,
                quality_score=0.0,
                protocol_used="unified",
                error_details="No iterations completed"
            )
        
        # Get the best result based on quality score
        best_result = max(iteration_results, key=lambda r: r.quality_score)
        
        # Aggregate tools used across iterations
        all_tools_used = []
        for result in iteration_results:
            all_tools_used.extend(result.tools_used)
        
        return AgentExecutionResult(
            output=best_result.output,
            execution_metadata=ExecutionMetadata(
                mode=ExecutionMode.OPTIMAL,
                protocol_used=best_result.protocol_used,
                execution_time=execution_time,
                iterations_planned=len(iteration_results),
                tools_utilized=list(set(all_tools_used))
            ),
            iterations_completed=len(iteration_results),
            completion_reason=CompletionReason.QUALITY_THRESHOLD_MET,
            quality_score=best_result.quality_score,
            protocol_used=best_result.protocol_used
        )

    def _determine_max_iterations(self, context: ExecutionContext, mode: ExecutionMode) -> int:
        """Determine maximum iterations based on execution mode and config."""
        config = context.execution_config
        
        if mode == ExecutionMode.SINGLE_PASS:
            return 1
        elif mode == ExecutionMode.MULTI_ITERATION:
            return config.max_iterations
        elif mode == ExecutionMode.OPTIMAL:
            if self.enable_refinement:
                return min(config.max_iterations * 2, 10)
            else:
                return config.max_iterations
        else:
            return config.max_iterations

    def _assess_completion(
        self, 
        iteration_results: List[IterationResult], 
        context: ExecutionContext, 
        current_iteration: int, 
        max_iterations: int
    ) -> CompletionAssessment:
        """Assess whether execution should complete based on results and criteria."""
        
        if not iteration_results:
            return CompletionAssessment(
                is_complete=False,
                reason=CompletionReason.ERROR_OCCURRED,
                quality_score=0.0
            )
        
        best_quality = max(r.quality_score for r in iteration_results)
        quality_threshold = context.execution_config.quality_threshold
        
        if best_quality >= quality_threshold:
            return CompletionAssessment(
                is_complete=True,
                reason=CompletionReason.QUALITY_THRESHOLD_MET,
                quality_score=best_quality
            )
        
        if current_iteration >= max_iterations:
            return CompletionAssessment(
                is_complete=True,
                reason=CompletionReason.MAX_ITERATIONS_REACHED,
                quality_score=best_quality
            )
        
        return CompletionAssessment(
            is_complete=False,
            reason=CompletionReason.QUALITY_THRESHOLD_NOT_MET,
            quality_score=best_quality
        )

    @abstractmethod
    async def _execute_iteration(
        self, 
        context: ExecutionContext, 
        iteration: int
    ) -> IterationResult:
        """Execute a single iteration of agent logic - implemented by each agent."""
        pass

    # ========================================
    # CONSOLIDATED TOOL MANAGEMENT
    # ========================================

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Universal MCP tool calling interface with unified management."""
        try:
            self.logger.info(f"[MCP] Calling tool {tool_name}")
            
            # Get tool info and handle aliases/placeholders
            tool_info = self._get_tool_category_and_info(tool_name)
            actual_tool_name = tool_info["actual_tool_name"]
            
            # Handle registry tools
            if tool_name.startswith('registry_'):
                return self._generate_registry_response(tool_name, arguments)
            
            # Check if tool is available and import
            from ..mcp_tools import __all__ as available_tools
            if actual_tool_name not in available_tools:
                # FAIL LOUDLY: Don't mask missing tools with placeholders
                error_msg = f"MCP tool '{actual_tool_name}' is not available in the tools registry. Available tools: {len(available_tools)} total. This indicates a missing tool implementation or import issue."
                self.logger.error(f"[MCP] Tool not available: {error_msg}")
                raise RuntimeError(f"MCP_TOOL_NOT_AVAILABLE: {error_msg}")
            
            import chungoid.mcp_tools as mcp_module
            
            # FAIL LOUDLY: Don't catch getattr failures
            try:
                tool_func = getattr(mcp_module, actual_tool_name)
            except AttributeError as e:
                error_msg = f"MCP tool '{actual_tool_name}' is listed in __all__ but not actually available in module. This indicates an import/export mismatch."
                self.logger.error(f"[MCP] Tool import failed: {error_msg}")
                raise RuntimeError(f"MCP_TOOL_IMPORT_FAILED: {error_msg}") from e
            
            # Convert arguments
            converted_args = self._convert_tool_arguments(tool_name, actual_tool_name, arguments)
            
            # FAIL LOUDLY: Don't mask tool execution failures
            try:
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**converted_args)
                else:
                    result = tool_func(**converted_args)
            except Exception as e:
                error_msg = f"MCP tool '{actual_tool_name}' execution failed with arguments {converted_args}. Error: {str(e)}"
                self.logger.error(f"[MCP] Tool execution failed: {error_msg}")
                raise RuntimeError(f"MCP_TOOL_EXECUTION_FAILED: {error_msg}") from e
            
            # Ensure consistent response format
            if isinstance(result, dict):
                if "success" not in result and "error" not in result:
                    result["success"] = True
                result["tool_name"] = tool_name
                return result
            else:
                return {"success": True, "result": result, "tool_name": tool_name}
                
        except Exception as e:
            # FAIL LOUDLY: Re-raise all exceptions instead of masking with placeholders
            if "MCP_TOOL_" in str(e):
                # These are our explicit MCP errors, re-raise as-is
                raise
            else:
                # Unexpected errors get wrapped with context
                error_msg = f"Unexpected error in MCP tool '{tool_name}' call: {str(e)}"
                self.logger.error(f"[MCP] Unexpected error: {error_msg}")
                raise RuntimeError(f"MCP_UNEXPECTED_ERROR: {error_msg}") from e

    def _get_tool_category_and_info(self, tool_name: str) -> Dict[str, Any]:
        """Get tool category and related information from centralized configuration."""
        category = UniversalPatternMatcher.categorize_by_patterns(
            tool_name, 
            {cat: info["keywords"] for cat, info in self.TOOL_CATEGORIES.items()}
        )
        
        category_info = self.TOOL_CATEGORIES.get(category, self.TOOL_CATEGORIES["registry"])
        
        return {
            "category": category,
            "placeholder_data": category_info["placeholder"],
            "aliases": category_info["aliases"],
            "actual_tool_name": category_info["aliases"].get(tool_name, tool_name)
        }

    def _generate_placeholder_response(self, tool_name: str, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        DEPRECATED: Placeholder responses are no longer used - system now FAILS LOUDLY.
        This method is kept for backwards compatibility but should not be called.
        """
        error_msg = f"Attempted to generate placeholder response for '{tool_name}' - this is deprecated. System should FAIL LOUDLY instead of using placeholders."
        self.logger.error(f"[MCP] Deprecated placeholder call: {error_msg}")
        raise RuntimeError(f"DEPRECATED_PLACEHOLDER_RESPONSE: {error_msg}")

    def _generate_registry_response(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate registry tool responses."""
        registry_responses = {
            "registry_get_tool_info": {
                "tool_info": {"name": arguments.get("tool_name", "unknown"), "category": "general"}
            },
            "registry_list_all_tools": {
                "tools": list(self.TOOL_CATEGORIES.keys()),
                "count": len(self.TOOL_CATEGORIES)
            },
            "registry_search_tools": {
                "results": [{"name": tool, "relevance": 0.9} for tool in self.TOOL_CATEGORIES.keys()]
            }
        }
        
        response = registry_responses.get(tool_name, {"message": "Registry operation completed"})
        response.update({"success": True, "tool_name": tool_name})
        return response

    def _convert_tool_arguments(self, original_tool_name: str, actual_tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Convert arguments based on tool aliases and parameter mappings."""
        converted_arguments = arguments.copy()
        
        # CRITICAL FIX: Filter out cache coordination arguments that are not meant for actual tools
        # These are internal parameters used by the cache system, not tool parameters
        cache_args_to_remove = [
            '_force_refresh', '_cache_bust', '_coordinated_refresh', 
            '_force_verification', '_cache_bypass', '_internal_call',
            'force_refresh', 'cache_bust', 'bypass_cache', 'timestamp'  # Also remove non-underscore versions
        ]
        for cache_arg in cache_args_to_remove:
            if cache_arg in converted_arguments:
                self.logger.debug(f"[MCP] Filtering cache argument {cache_arg} from tool call {actual_tool_name}")
                converted_arguments.pop(cache_arg)
        
        # Universal parameter mapping
        if "working_directory" in converted_arguments and "project_path" not in converted_arguments:
            converted_arguments["project_path"] = converted_arguments.pop("working_directory")
        
        # Specific conversions for common alias patterns
        if original_tool_name == "terminal_set_environment_variable":
            var_name = arguments.get("variable_name", "VAR")
            var_value = arguments.get("variable_value", "value")
            converted_arguments = {"command": f"export {var_name}={var_value}"}
        
        elif original_tool_name == "content_extract_text" and actual_tool_name == "web_content_extract":
            converted_arguments = {
                "content": str(arguments.get("source", "default text")), 
                "extraction_type": "text"
            }
        
        # Clean up None values
        return {k: v for k, v in converted_arguments.items() if v is not None}

    # ========================================
    # UNIVERSAL INHERITANCE METHODS
    # Project-agnostic methods for ALL 9 agents
    # ========================================

    def clear_discovery_cache(self, reason: str = "manual") -> None:
        """Enhanced cache clearing with coordination and detailed logging."""
        try:
            # Clear agent-level iteration cache
            cleared_iterations = len(self.iteration_cache)
            self.iteration_cache.clear()
            
            # Clear discovery service cache if available
            cleared_discovery = 0
            try:
                if hasattr(self, '_discovery_service') and self._discovery_service:
                    # Get cache size before clearing
                    cache_size = len(getattr(self._discovery_service, '_cache', {}))
                    self._discovery_service.clear_cache()
                    cleared_discovery = cache_size
                elif '_discovery_service' in globals():
                    cache_size = len(getattr(globals()['_discovery_service'], '_cache', {}))
                    globals()['_discovery_service'].clear_cache()
                    cleared_discovery = cache_size
            except Exception as e:
                self.logger.warning(f"CACHE_CLEAR: Failed to clear discovery service cache: {e}")
            
            # Clear any other caches if available
            try:
                from ..utils.cache_coordination_fix import clear_all_coordinated_caches
                clear_all_coordinated_caches(
                    discovery_service=getattr(self, '_discovery_service', None) or 
                                    globals().get('_discovery_service')
                )
            except ImportError:
                pass
            
            self.logger.info(f"CACHE_CLEAR: Cleared caches - iterations: {cleared_iterations}, "
                           f"discovery: {cleared_discovery}, reason: {reason}")
            
        except Exception as e:
            self.logger.error(f"CACHE_CLEAR: Failed to clear caches: {e}")

    async def verify_project_state_after_operations(self, project_path: str, expected_files: List[str] = None) -> Dict[str, Any]:
        """Enhanced project state verification with coordinated cache refresh."""
        try:
            # Use cache coordination for verification if available
            try:
                from ..utils.cache_coordination_fix import coordinate_cache_refresh
                
                self.logger.info("STATE_VERIFICATION: Starting coordinated verification")
                
                # Force coordinated cache refresh
                verification_result = await coordinate_cache_refresh(
                    project_path,
                    self._call_mcp_tool,
                    discovery_service=getattr(self, '_discovery_service', None) or 
                                    globals().get('_discovery_service'),
                    reason="post_operation_verification"
                )
                
            except ImportError:
                # Fallback to direct MCP call
                self.logger.warning("STATE_VERIFICATION: Cache coordination not available, using direct verification")
                verification_result = await self._call_mcp_tool("filesystem_list_directory", {
                    "directory_path": project_path,
                    "recursive": True,
                    "include_files": True,
                    "include_directories": True,
                    "_force_verification": True,
                    "_cache_bust": time.time()
                })
            
            if not verification_result.get("success"):
                error_msg = verification_result.get('error', 'Unknown error')
                self.logger.error(f"STATE_VERIFICATION: Failed to scan project: {error_msg}")
                return {"verified": False, "error": error_msg}
            
            # Extract file information
            found_files = []
            items = verification_result.get("items", [])
            for item in items:
                if item.get("type") == "file":
                    relative_path = item.get("relative_path", item.get("name", ""))
                    if relative_path:
                        found_files.append(relative_path)
            
            # Verify expected files if provided
            missing_files = []
            if expected_files:
                missing_files = [f for f in expected_files if not any(f in found for found in found_files)]
                if missing_files:
                    self.logger.warning(f"STATE_VERIFICATION: Missing expected files: {missing_files}")
                else:
                    self.logger.info(f"STATE_VERIFICATION: All expected files found: {len(expected_files)} files")
            
            verification_data = {
                "verified": True,
                "total_files": len(found_files),
                "found_files": found_files,
                "project_path": project_path,
                "timestamp": time.time(),
                "expected_files": expected_files or [],
                "missing_files": missing_files
            }
            
            self.logger.info(f"STATE_VERIFICATION: Complete - {len(found_files)} files detected, "
                           f"{len(missing_files)} missing")
            return verification_data
            
        except Exception as e:
            self.logger.error(f"STATE_VERIFICATION: Exception during verification: {e}")
            return {"verified": False, "error": str(e)}

    @universal_error_handler("Technology Discovery", {"primary_language": "unknown", "frameworks": [], "deployment": []})
    async def _universal_technology_discovery(self, project_path: str) -> Dict[str, Any]:
        """Universal project type detection that works for ANY technology."""
        # Use consolidated discovery
        discovery_result = await self._universal_discovery(project_path, ["dependencies", "structure"])
        
        # Extract characteristics
        dependencies = discovery_result.get("dependencies", {})
        structure = discovery_result.get("structure", {})
        
        # Determine primary language
        language_priority = ["python", "javascript", "java", "csharp", "go", "rust", "php", "ruby"]
        primary_language = "unknown"
        
        for lang in language_priority:
            if lang in dependencies and dependencies[lang]:
                primary_language = lang
                break
        
        # Extract frameworks from structure
        frameworks = []
        all_files = []
        for files in structure.values():
            all_files.extend(files)
        
        if any("package.json" in str(f) for f in all_files):
            frameworks.append("javascript")
        if any("requirements" in str(f) for f in all_files):
            frameworks.append("python")
        
        deployment = ["containerized"] if any("docker" in str(f).lower() for f in all_files) else []
        
        return {
            "primary_language": primary_language,
            "frameworks": frameworks,
            "deployment": deployment,
            "dependency_systems": list(dependencies.keys())
        }

    @universal_error_handler("Universal Discovery", {})
    async def _universal_discovery(self, project_path: str, discovery_types: List[str] = None) -> Dict[str, Any]:
        """Unified discovery system for all file types and patterns."""
        discovery_types = discovery_types or ["environment", "dependencies", "structure"]
        results = {}

        for discovery_type in discovery_types:
            if discovery_type == "environment":
                results[discovery_type] = await UniversalPatternMatcher.find_files_by_patterns(
                    project_path, 
                    self.DISCOVERY_PATTERNS["environment"],
                    recursive=False,
                    call_mcp_tool_func=self._call_mcp_tool
                )
            
            elif discovery_type == "dependencies":
                dependency_results = {}
                for language, patterns in self.DISCOVERY_PATTERNS["dependencies"].items():
                    language_results = await UniversalPatternMatcher.find_files_by_patterns(
                        project_path, patterns, recursive=True, call_mcp_tool_func=self._call_mcp_tool
                    )
                    if language_results:
                        dependency_results[language] = language_results
                results[discovery_type] = dependency_results
            
            elif discovery_type == "structure":
                results[discovery_type] = await UniversalPatternMatcher.find_files_by_patterns(
                    project_path, 
                    self.DISCOVERY_PATTERNS["structure"],
                    recursive=True,
                    call_mcp_tool_func=self._call_mcp_tool
                )

        return results

    @universal_error_handler("Content Enhancement", None)
    async def _enhance_existing_work_universally(
        self, 
        existing_content: str, 
        content_type: str, 
        project_characteristics: Dict[str, Any] = None,
        enhancement_context: Dict[str, Any] = None
    ) -> str:
        """Enhance existing work regardless of technology or format."""
        if not existing_content or not existing_content.strip():
            return existing_content
        
        project_characteristics = project_characteristics or {}
        enhancement_context = enhancement_context or {}
        
        # Get enhancement strategy
        strategy = self.ENHANCEMENT_STRATEGIES.get(
            content_type, 
            self.ENHANCEMENT_STRATEGIES["documentation"]
        )
        
        # Build enhancement prompt
        primary_language = project_characteristics.get("primary_language", "unknown")
        focus_areas = strategy.get("focus_areas", ["general"])
        
        enhancement_prompt = f"""Enhance this {content_type} content for a {primary_language} project.

Focus on: {', '.join(focus_areas)}

Current Content:
{existing_content}

Provide enhanced content following current best practices:"""
        
        try:
            enhanced_content = await self.llm_provider.generate_async(
                prompt=enhancement_prompt,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Apply enhancement strategy
            if strategy["type"] == "additive":
                return existing_content + "\n\n# Enhanced Sections\n" + enhanced_content
            elif strategy["type"] == "replacement":
                return enhanced_content if enhanced_content.strip() else existing_content
            else:  # merge
                return self._merge_content(existing_content, enhanced_content)
                
        except Exception as e:
            self.logger.error(f"[Enhancement] Failed: {e}")
            return existing_content

    def _merge_content(self, original: str, enhanced: str) -> str:
        """Intelligently merge original and enhanced content."""
        original_lines = original.split('\n')
        enhanced_lines = enhanced.split('\n')
        
        merged_lines = original_lines.copy()
        
        # Add non-duplicate enhanced lines
        for enhanced_line in enhanced_lines:
            if enhanced_line.strip() and not any(
                enhanced_line.strip().lower() in original_line.lower() 
                for original_line in original_lines
            ):
                merged_lines.append(enhanced_line)
        
        return '\n'.join(merged_lines)

    @universal_error_handler("Context Retrieval", {})
    async def _retrieve_stage_context(self, stage_name: str) -> Dict[str, Any]:
        """Retrieve context and outputs from previous stages."""
        context_file = f".chungoid/pipeline_context/{stage_name}_output.json"
        
        stage_output = await self._call_mcp_tool("filesystem_read_file", {
            "file_path": context_file,
            "project_path": "."  # Use current project directory
        })
        
        if stage_output.get("success"):
            content = stage_output.get("content", "{}")
            return json.loads(content) if isinstance(content, str) else content
        else:
            return {}

    @universal_error_handler("Context Save", False)
    async def _save_stage_context(self, stage_name: str, outputs: Dict[str, Any]) -> bool:
        """Save this stage's outputs for subsequent stages."""
        await self._call_mcp_tool("filesystem_create_directory", {
            "directory_path": ".chungoid/pipeline_context",
            "project_path": "."  # Use current project directory
        })
        
        context_file = f".chungoid/pipeline_context/{stage_name}_output.json"
        
        save_result = await self._call_mcp_tool("filesystem_write_file", {
            "file_path": context_file,
            "content": json.dumps(outputs, indent=2, default=str),
            "project_path": "."  # Use current project directory
        })
        
        return save_result.get("success", False)

    def _stage_owns_file_type_universally(self, file_path: str, agent_category: str) -> bool:
        """Determine file ownership using universal patterns."""
        ownership_patterns = {
            "environment": ["*env*", "*config*", "docker*", "setup*"],
            "dependencies": ["*requirements*", "*package*", "*.lock"],
            "documentation": ["*readme*", "*docs*", "*.md"],
            "source_code": ["*.py", "*.js", "*.java", "src/*", "lib/*"],
            "testing": ["*test*", "*spec*", "test/*", "tests/*"]
        }
        
        if agent_category not in ownership_patterns:
            return False
        
        patterns = ownership_patterns[agent_category]
        file_name = os.path.basename(file_path).lower()
        file_path_lower = file_path.lower()
        
        return any(
            UniversalPatternMatcher.match_pattern(file_path_lower, pattern) or 
            UniversalPatternMatcher.match_pattern(file_name, pattern)
            for pattern in patterns
        )

    async def _research_technology_best_practices(
        self, 
        technologies: List[str], 
        context: str = "general"
    ) -> Dict[str, Any]:
        """Research current best practices for discovered technologies."""
        research_results = {}
        
        for tech in technologies:
            try:
                research_query = f"{tech} best practices {context} 2025"
                
                research_result = await self._call_mcp_tool("web_search", {
                    "query": research_query
                })
                
                if research_result.get("success"):
                    research_results[tech] = research_result.get("result", {})
                else:
                    research_results[tech] = {"error": "Research failed"}
                    
            except Exception as e:
                research_results[tech] = {"error": str(e)}
        
        return research_results

    @universal_error_handler("Domain Analysis", {"domain": "general", "project_type": "application"})
    async def _analyze_project_domain(
        self, 
        user_goal: str, 
        project_path: str,
        tech_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Universal project domain analysis for any project type."""
        tech_context = tech_context or {}
        
        # Extract domain indicators
        domain_keywords = self._extract_domain_keywords(user_goal)
        
        # Classify domain and type
        return {
            "domain": domain_keywords[0] if domain_keywords else "general",
            "project_type": self._classify_project_type(domain_keywords, tech_context),
            "complexity_level": "medium",
            "target_audience": self._infer_target_audience(user_goal)
        }

    def _extract_domain_keywords(self, user_goal: str) -> List[str]:
        """Extract domain-specific keywords from user goal."""
        domain_patterns = {
            "web": ["website", "web", "frontend", "backend", "api"],
            "mobile": ["mobile", "app", "android", "ios"],
            "data": ["data", "analytics", "ml", "ai"],
            "game": ["game", "gaming", "unity"],
            "enterprise": ["enterprise", "business", "corporate"]
        }
        
        keywords = []
        user_goal_lower = user_goal.lower()
        
        for domain, patterns in domain_patterns.items():
            if any(pattern in user_goal_lower for pattern in patterns):
                keywords.append(domain)
        
        return keywords

    def _classify_project_type(self, domain_keywords: List[str], tech_context: Dict[str, Any]) -> str:
        """Classify the project type based on domain and tech context."""
        if "web" in domain_keywords:
            return "web_application"
        elif "mobile" in domain_keywords:
            return "mobile_application"
        elif "data" in domain_keywords:
            return "data_pipeline"
        elif "api" in str(tech_context).lower():
            return "api_service"
        else:
            return "application"

    def _infer_target_audience(self, user_goal: str) -> str:
        """Infer target audience from user goal."""
        user_goal_lower = user_goal.lower()
        
        if any(word in user_goal_lower for word in ["enterprise", "business"]):
            return "enterprise"
        elif any(word in user_goal_lower for word in ["consumer", "user", "public"]):
            return "consumer"
        elif any(word in user_goal_lower for word in ["developer", "api"]):
            return "developer"
        else:
            return "general"

    # ========================================
    # JSON VALIDATION INFRASTRUCTURE
    # ========================================

    async def _extract_and_validate_json(
        self, 
        response: str, 
        schema: Optional[Type[BaseModel]] = None
    ) -> Union[BaseModel, Dict[str, Any], str]:
        """Complete JSON extraction and validation pipeline."""
        try:
            json_str = self._extract_json_from_response(response)
            
            if schema:
                return self._validate_json_against_schema(json_str, schema)
            else:
                return json.loads(json_str)
                
        except Exception as e:
            self.logger.error(f"[JSON] Extraction/validation failed: {e}")
            
            if self.json_validation_config.fallback_to_text:
                return response
            else:
                raise

    def _extract_json_from_response(self, response: str) -> str:
        """Universal JSON extraction with multiple strategies."""
        if not response or not response.strip():
            raise ValueError("Empty response provided")
        
        response = response.strip()
        
        # Try markdown code blocks first
        json_pattern = r'```json\s*\n(.*?)\n```'
        match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try bracket matching
        start_idx = response.find('{')
        if start_idx != -1:
            brace_count = 0
            for i, char in enumerate(response[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        potential_json = response[start_idx:i+1]
                        if self._is_valid_json_syntax(potential_json):
                            return potential_json
        
        # If all else fails, try to repair
        if self.json_validation_config.enable_json_repair:
            repaired = self._repair_json(response)
            if repaired:
                return repaired
        
        raise ValueError(f"Could not extract valid JSON from response")

    def _is_valid_json_syntax(self, json_str: str) -> bool:
        """Check if string is valid JSON syntax."""
        try:
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def _repair_json(self, json_str: str) -> Optional[str]:
        """Attempt to repair malformed JSON using simple strategies."""
        # Try common fixes
        fixes = [
            (r',(\s*[}\]])', r'\1'),  # Remove trailing commas
            (r'(\w+):', r'"\1":'),   # Add quotes around keys
            (r"'([^']*)'", r'"\1"'), # Convert single to double quotes
        ]
        
        repaired = json_str
        for pattern, replacement in fixes:
            repaired = re.sub(pattern, replacement, repaired)
        
        if self._is_valid_json_syntax(repaired):
            return repaired
        
        return None

    def _validate_json_against_schema(self, json_str: str, schema: Type[BaseModel]) -> BaseModel:
        """Validate JSON string against Pydantic schema."""
        try:
            return schema.model_validate_json(json_str)
        except ValidationError as e:
            if not self.json_validation_config.strict_validation:
                # Try with cleaned data
                json_data = json.loads(json_str)
                cleaned_data = self._clean_data_for_schema(json_data, schema)
                return schema.model_validate(cleaned_data)
            raise e

    def _clean_data_for_schema(self, data: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
        """Clean data to match schema requirements."""
        if not isinstance(data, dict):
            return data
        
        schema_fields = schema.model_fields
        cleaned_data = {}
        
        # Process existing fields
        for key, value in data.items():
            if key in schema_fields:
                cleaned_data[key] = value
        
        # Add defaults for missing required fields
        for field_name, field_info in schema_fields.items():
            if field_name not in cleaned_data and field_info.is_required():
                # Provide sensible defaults
                annotation = field_info.annotation
                if annotation == str:
                    cleaned_data[field_name] = f"[Missing {field_name}]"
                elif annotation == int:
                    cleaned_data[field_name] = 0
                elif annotation == float:
                    cleaned_data[field_name] = 0.0
                elif annotation == bool:
                    cleaned_data[field_name] = False
                elif annotation == list:
                    cleaned_data[field_name] = []
                elif annotation == dict:
                    cleaned_data[field_name] = {}
        
        return cleaned_data

    # ========================================
    # LEGACY ALIASES FOR BACKWARD COMPATIBILITY
    # ========================================

    async def _discover_existing_environment_files(self, project_path: str) -> Dict[str, List[str]]:
        """Legacy alias - use _universal_discovery instead."""
        result = await self._universal_discovery(project_path, ["environment"])
        return result.get("environment", {})

    async def _discover_all_dependency_systems(self, project_path: str) -> Dict[str, List[str]]:
        """Legacy alias - use _universal_discovery instead."""
        result = await self._universal_discovery(project_path, ["dependencies"])
        return result.get("dependencies", {})

    async def _get_all_available_mcp_tools(self) -> Dict[str, Any]:
        """Legacy alias - use consolidated tool discovery."""
        try:
            from ..mcp_tools import __all__ as tool_names
            return {
                "discovery_successful": True,
                "tools": {name: {"name": name, "available": True} for name in tool_names},
                "total_tools": len(tool_names)
            }
        except Exception:
            return {"discovery_successful": False, "tools": {}, "total_tools": 0}
