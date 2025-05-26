"""UnifiedAgent - UAEI Base Class (Phase 1)

Single interface for ALL agent execution - eliminates dual interface complexity.
According to enhanced_cycle.md Phase 1 implementation.
"""

from __future__ import annotations

import logging
import time
from abc import ABC
from typing import Any, ClassVar, List, Optional

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
    """
    
    # Required class metadata (enforced by validation)
    AGENT_ID: ClassVar[str]
    AGENT_VERSION: ClassVar[str] 
    PRIMARY_PROTOCOLS: ClassVar[List[str]]
    CAPABILITIES: ClassVar[List[str]]
    
    # Standard initialization
    llm_provider: LLMProvider = Field(..., description="LLM provider for AI capabilities")
    prompt_manager: PromptManager = Field(..., description="Prompt manager for templates")
    
    # Internal
    logger: Optional[logging.Logger] = Field(default=None)
    
    # Model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

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
        
        self.logger.info(f"[UAEI] Starting {execution_mode.value} execution with max_iterations={max_iterations}")
        
        # Multi-iteration execution loop (Phase 3 implementation)
        best_result = None
        all_tools_used = []
        
        for iteration in range(max_iterations):
            self.logger.info(f"[UAEI] Starting iteration {iteration + 1}/{max_iterations}")
            
            # Execute single iteration using agent's core logic
            iteration_result = await self._execute_iteration(current_context, iteration)
            all_tools_used.extend(iteration_result.tools_used)
            
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



    # Phase 3 implementation methods
    async def _assess_completion(
        self, 
        iteration_result: IterationResult, 
        completion_criteria: Any,
        context: ExecutionContext
    ) -> CompletionAssessment:
        """Assess if execution is complete - Phase 3 implementation"""
        
        quality_score = iteration_result.quality_score
        min_quality = completion_criteria.min_quality_score if completion_criteria else 0.85
        
        # Check quality threshold
        if quality_score >= min_quality:
            return CompletionAssessment(
                is_complete=True,
                reason=CompletionReason.QUALITY_THRESHOLD_MET,
                quality_score=quality_score
            )
        
        # Check if required outputs are present
        if completion_criteria and completion_criteria.required_outputs:
            output_dict = iteration_result.output if isinstance(iteration_result.output, dict) else {}
            missing_outputs = [req for req in completion_criteria.required_outputs if req not in output_dict]
            
            if not missing_outputs:
                return CompletionAssessment(
                    is_complete=True,
                    reason=CompletionReason.COMPLETION_CRITERIA_MET,
                    quality_score=quality_score
                )
            else:
                return CompletionAssessment(
                    is_complete=False,
                    reason=CompletionReason.QUALITY_THRESHOLD_MET,  # Not met yet
                    quality_score=quality_score,
                    gaps_identified=missing_outputs,
                    recommendations=[f"Complete missing output: {output}" for output in missing_outputs]
                )
        
        # Default: not complete yet
        return CompletionAssessment(
            is_complete=False,
            reason=CompletionReason.QUALITY_THRESHOLD_MET,  # Not met yet  
            quality_score=quality_score,
            gaps_identified=["Quality score below threshold"],
            recommendations=["Improve output quality", "Use more comprehensive tools"]
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
        if hasattr(enhanced_inputs, 'dict') and callable(enhanced_inputs.dict):
            inputs_dict = enhanced_inputs.dict()
            inputs_dict["previous_iteration_context"] = {
                "quality_score": iteration_result.quality_score,
                "gaps_to_address": completion_assessment.gaps_identified,
                "recommendations": completion_assessment.recommendations,
                "tools_already_used": iteration_result.tools_used
            }
            # Note: We keep enhanced_inputs as is since we can't easily recreate the original type
        
        # Update tool utilization mode if needed
        enhanced_config = context.execution_config
        if completion_assessment.gaps_identified and "Use more comprehensive tools" in completion_assessment.recommendations:
            enhanced_config.tool_utilization_mode = ToolMode.COMPREHENSIVE
        
        # Create new ExecutionContext with enhanced data
        return ExecutionContext(
            inputs=enhanced_inputs,
            shared_context=enhanced_shared_context,
            stage_info=context.stage_info,
            execution_config=enhanced_config
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