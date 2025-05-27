"""UnifiedAgent - UAEI Base Class (Phase 1)

Single interface for ALL agent execution - eliminates dual interface complexity.
According to enhanced_cycle.md Phase 1 implementation.
Enhanced with refinement capabilities for intelligent iteration cycles.
"""

from __future__ import annotations

import logging
import time
import os
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
    enable_refinement: bool = Field(default=False, description="Enable intelligent refinement using MCP tools and ChromaDB")
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
                from ..mcp_tools import get_mcp_tools_registry
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
        """Use MCP tools to analyze current project state"""
        if not self.enable_refinement or not self.mcp_tools:
            return {}
        
        try:
            current_state = {}
            project_path = context.shared_context.get("project_root_path", ".")
            
            # Use MCP tools based on agent capabilities
            if "code_generation" in self.CAPABILITIES:
                # Analyze existing code
                current_state["code_analysis"] = await self._analyze_code_with_mcp_tools(project_path)
            
            if "architecture_design" in self.CAPABILITIES:
                # Analyze project structure
                current_state["project_structure"] = await self._analyze_project_structure_with_mcp_tools(project_path)
            
            if "documentation" in self.CAPABILITIES:
                # Analyze existing documentation
                current_state["documentation_analysis"] = await self._analyze_documentation_with_mcp_tools(project_path)
            
            # Common analysis for all agents
            current_state["file_structure"] = await self._analyze_file_structure_with_mcp_tools(project_path)
            
            self.logger.info(f"[Refinement] Analyzed current state with {len(current_state)} analysis types")
            return current_state
            
        except Exception as e:
            self.logger.warning(f"[Refinement] Failed to analyze current state: {e}")
            return {}
    
    async def _analyze_code_with_mcp_tools(self, project_path: str) -> Dict[str, Any]:
        """Analyze code using MCP tools"""
        try:
            # Use MCP tools to analyze code
            if hasattr(self.mcp_tools, 'analyze_code_ast'):
                return await self.mcp_tools.analyze_code_ast(project_path)
            return {"status": "mcp_tool_not_available"}
        except Exception as e:
            self.logger.warning(f"[Refinement] Code analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_project_structure_with_mcp_tools(self, project_path: str) -> Dict[str, Any]:
        """Analyze project structure using MCP tools"""
        try:
            # Use MCP tools to analyze project structure
            if hasattr(self.mcp_tools, 'analyze_project_structure'):
                return await self.mcp_tools.analyze_project_structure(project_path)
            return {"status": "mcp_tool_not_available"}
        except Exception as e:
            self.logger.warning(f"[Refinement] Project structure analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_documentation_with_mcp_tools(self, project_path: str) -> Dict[str, Any]:
        """Analyze documentation using MCP tools"""
        try:
            # Use MCP tools to analyze documentation
            if hasattr(self.mcp_tools, 'analyze_documentation'):
                return await self.mcp_tools.analyze_documentation(project_path)
            return {"status": "mcp_tool_not_available"}
        except Exception as e:
            self.logger.warning(f"[Refinement] Documentation analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_file_structure_with_mcp_tools(self, project_path: str) -> Dict[str, Any]:
        """Analyze file structure using MCP tools"""
        try:
            # Use MCP tools to analyze file structure
            if hasattr(self.mcp_tools, 'list_project_files'):
                return await self.mcp_tools.list_project_files(project_path)
            return {"status": "mcp_tool_not_available"}
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