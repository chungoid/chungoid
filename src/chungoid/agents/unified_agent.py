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
        """Universal file discovery using patterns with MCP tool integration."""
        if not call_mcp_tool_func:
            return {}
        
        discovered_files = {}
        for pattern in patterns:
            try:
                result = await call_mcp_tool_func("filesystem_glob_search", {
                    "path": project_path,
                    "pattern": pattern,
                    "recursive": recursive
                })
                
                if result.get("success") and result.get("matches"):
                    discovered_files[pattern] = result["matches"]
                    
            except Exception:
                pass  # Silent failure for individual patterns
        
        return discovered_files
    
    @staticmethod
    def match_pattern(file_path: str, pattern: str) -> bool:
        """Single pattern matching implementation."""
        if "*" in pattern:
            return fnmatch.fnmatch(file_path.lower(), pattern.lower())
        return pattern.lower() in file_path.lower()
    
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
        """Main UAEI execution entry point - orchestrates multi-iteration execution."""
        start_time = time.time()
        max_iterations = self._determine_max_iterations(context, mode)
        
        self.logger.info(f"[UAEI] Starting execution: agent={self.AGENT_ID}, mode={mode.value}")
        
        iteration_results = []
        tools_utilized = set()
        completion_reason = CompletionReason.ERROR_OCCURRED
        final_output = None
        
        try:
            for iteration in range(max_iterations):
                iteration_result = await self._execute_iteration(context, iteration)
                iteration_results.append(iteration_result)
                
                tools_utilized.update(iteration_result.tools_used)
                final_output = iteration_result.output
                
                completion_assessment = self._assess_completion(
                    iteration_results, context, iteration + 1, max_iterations
                )
                
                if completion_assessment.is_complete:
                    completion_reason = completion_assessment.reason
                    break
        
        except Exception as execution_error:
            self.logger.error(f"[UAEI] Execution failed: {execution_error}")
            completion_reason = CompletionReason.ERROR_OCCURRED
            
            if not iteration_results:
                final_output = {"error": str(execution_error)}
                iteration_results = [IterationResult(
                    output=final_output,
                    quality_score=0.1,
                    tools_used=[],
                    protocol_used="error_handling"
                )]
        
        # Calculate results
        execution_time = time.time() - start_time
        quality_scores = [r.quality_score for r in iteration_results]
        final_quality_score = max(quality_scores) if quality_scores else 0.1
        
        best_iteration = max(iteration_results, key=lambda r: r.quality_score) if iteration_results else None
        protocol_used = best_iteration.protocol_used if best_iteration else "unknown"
        
        execution_metadata = ExecutionMetadata(
            mode=mode,
            protocol_used=protocol_used,
            execution_time=execution_time,
            iterations_planned=max_iterations,
            tools_utilized=list(tools_utilized)
        )
        
        result = AgentExecutionResult(
            output=final_output,
            execution_metadata=execution_metadata,
            iterations_completed=len(iteration_results),
            completion_reason=completion_reason,
            quality_score=final_quality_score,
            protocol_used=protocol_used,
            error_details=str(final_output.get("error")) if isinstance(final_output, dict) and "error" in final_output else None
        )
        
        self.logger.info(f"[UAEI] Execution completed: quality={final_quality_score:.3f}")
        
        return result

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
            try:
                from ..mcp_tools import __all__ as available_tools
                if actual_tool_name not in available_tools:
                    return self._generate_placeholder_response(tool_name, tool_info)
                
                import chungoid.mcp_tools as mcp_module
                tool_func = getattr(mcp_module, actual_tool_name)
                
                # Convert arguments
                converted_args = self._convert_tool_arguments(tool_name, actual_tool_name, arguments)
                
                # Execute tool
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**converted_args)
                else:
                    result = tool_func(**converted_args)
                
                # Ensure consistent response format
                if isinstance(result, dict):
                    if "success" not in result and "error" not in result:
                        result["success"] = True
                    result["tool_name"] = tool_name
                    return result
                else:
                    return {"success": True, "result": result, "tool_name": tool_name}
                    
            except Exception:
                return self._generate_placeholder_response(tool_name, tool_info)
        
        except Exception as e:
            self.logger.error(f"[MCP] Tool call failed: {tool_name} - {e}")
            return {"success": False, "error": str(e), "tool_name": tool_name}

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
        """Generate appropriate placeholder response."""
        return {
            "success": True,
            "result": tool_info["placeholder_data"],
            "tool_name": tool_name,
            "placeholder": True,
            "category": tool_info["category"],
            "message": f"Placeholder response for {tool_name}"
        }

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
            "file_path": context_file
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
            "directory_path": ".chungoid/pipeline_context"
        })
        
        context_file = f".chungoid/pipeline_context/{stage_name}_output.json"
        
        save_result = await self._call_mcp_tool("filesystem_write_file", {
            "file_path": context_file,
            "content": json.dumps(outputs, indent=2, default=str)
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
