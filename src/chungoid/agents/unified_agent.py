"""UnifiedAgent - UAEI Base Class (Phase 1)

Single interface for ALL agent execution - eliminates dual interface complexity.
According to enhanced_cycle.md Phase 1 implementation.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
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
        
        # Phase 1: Single iteration execution (will be expanded in Phase 3)
        try:
            # Execute single iteration using appropriate protocol
            iteration_result = await self._execute_iteration(current_context, 0)
            
            execution_time = time.time() - start_time
            
            return AgentExecutionResult(
                output=iteration_result.output,
                execution_metadata=ExecutionMetadata(
                    mode=execution_mode,
                    protocol_used=iteration_result.protocol_used,
                    execution_time=execution_time,
                    iterations_planned=max_iterations,
                    tools_utilized=iteration_result.tools_used
                ),
                iterations_completed=1,
                completion_reason=CompletionReason.QUALITY_THRESHOLD_MET,  # Phase 1: assume success
                quality_score=iteration_result.quality_score,
                protocol_used=iteration_result.protocol_used
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"[UAEI] Execution failed: {e}")
            
            return AgentExecutionResult(
                output=None,
                execution_metadata=ExecutionMetadata(
                    mode=execution_mode,
                    protocol_used="unknown",
                    execution_time=execution_time,
                    iterations_planned=max_iterations,
                    tools_utilized=[]
                ),
                iterations_completed=0,
                completion_reason=CompletionReason.ERROR_OCCURRED,
                quality_score=0.0,
                protocol_used="unknown",
                error_details=str(e)
            )

    def _determine_optimal_mode(self, context: ExecutionContext) -> ExecutionMode:
        """Intelligent execution mode selection based on agent capabilities and task complexity"""
        # Phase 1: Simple logic - will be enhanced in Phase 3
        
        # Simple agents get single-pass
        if "simple_operations" in self.CAPABILITIES:
            return ExecutionMode.SINGLE_PASS
        
        # Complex analysis agents could benefit from multi-iteration
        elif "complex_analysis" in self.CAPABILITIES:
            # Phase 1: Still use single-pass, will enable multi-iteration in Phase 3
            return ExecutionMode.SINGLE_PASS
        
        else:
            return ExecutionMode.SINGLE_PASS  # Default for Phase 1

    async def _execute_iteration(
        self, 
        context: ExecutionContext, 
        iteration: int
    ) -> IterationResult:
        """Execute a single iteration of the agent's core functionality"""
        
        self.logger.info(f"[UAEI] Executing iteration {iteration}")
        
        # Phase 1: Delegate to agent-specific implementation
        # In Phase 2, this will contain the direct agent logic
        # In Phase 3, this will support true multi-iteration
        
        try:
            output = await self._execute_agent_logic(context)
            
            return IterationResult(
                output=output,
                quality_score=0.85,  # Phase 1: default quality score
                tools_used=[],       # Phase 1: basic tool tracking
                protocol_used=self._get_preferred_protocol(context),
                iteration_metadata={"iteration": iteration}
            )
            
        except Exception as e:
            self.logger.error(f"[UAEI] Iteration {iteration} failed: {e}")
            raise

    def _get_preferred_protocol(self, context: ExecutionContext) -> str:
        """Get the preferred protocol for this agent"""
        if context.execution_config.protocol_preference:
            return context.execution_config.protocol_preference
        elif self.PRIMARY_PROTOCOLS:
            return self.PRIMARY_PROTOCOLS[0]
        else:
            return "basic_protocol"

    @abstractmethod
    async def _execute_agent_logic(self, context: ExecutionContext) -> Any:
        """
        Agent-specific implementation - each agent must implement this.
        
        Phase 1: This delegates to existing invoke_async/execute_with_protocol methods
        Phase 2: This contains the direct agent logic (legacy methods removed)
        Phase 3: This supports multi-iteration enhancement
        """
        pass

    # Future Phase 3 methods (stubs for now)
    async def _assess_completion(
        self, 
        iteration_result: IterationResult, 
        completion_criteria: Any,
        context: ExecutionContext
    ) -> CompletionAssessment:
        """Assess if execution is complete - Phase 3 implementation"""
        # Phase 1: Always complete after one iteration
        return CompletionAssessment(
            is_complete=True,
            reason=CompletionReason.QUALITY_THRESHOLD_MET,
            quality_score=iteration_result.quality_score
        )

    async def _enhance_context_for_next_iteration(
        self,
        context: ExecutionContext, 
        iteration_result: IterationResult, 
        completion_assessment: CompletionAssessment
    ) -> ExecutionContext:
        """Enhance context for next iteration - Phase 3 implementation"""
        # Phase 1: Return unchanged context
        return context 