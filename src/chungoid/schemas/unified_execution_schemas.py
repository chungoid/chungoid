"""Unified Agent Execution Interface (UAEI) Schemas

Data models for the unified execution architecture according to enhanced_cycle.md
Phase 1 implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class ExecutionMode(Enum):
    """Execution strategy selection"""
    SINGLE_PASS = "single_pass"          # Legacy behavior (max_iterations=1)
    MULTI_ITERATION = "multi_iteration"  # Enhanced cycle behavior  
    OPTIMAL = "optimal"                  # Agent decides based on complexity


class CompletionReason(Enum):
    """Why execution stopped"""
    QUALITY_THRESHOLD_MET = "quality_threshold_met"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    COMPLETION_CRITERIA_MET = "completion_criteria_met"
    ERROR_OCCURRED = "error_occurred"
    USER_REQUESTED_STOP = "user_requested_stop"
    
    # Compatibility aliases for common usage patterns
    SUCCESS = "quality_threshold_met"  # Alias for successful completion
    ERROR = "error_occurred"  # Alias for error
    QUALITY_THRESHOLD = "quality_threshold_met"  # Alias for threshold met


class ToolMode(Enum):
    """Tool utilization strategy"""
    MINIMAL = "minimal"           # Use minimal tools
    COMPREHENSIVE = "comprehensive"  # Use all available tools optimally
    ADAPTIVE = "adaptive"         # Adapt tool usage based on task


@dataclass
class StageInfo:
    """Stage metadata for execution context"""
    stage_id: str
    attempt_number: int = 1
    parent_stage_id: Optional[str] = None
    stage_type: Optional[str] = None


@dataclass
class CompletionCriteria:
    """Defines when to stop iterating"""
    min_quality_score: float = 0.85
    required_outputs: List[str] = field(default_factory=list)
    tool_utilization_threshold: float = 0.8
    comprehensive_validation: bool = True
    custom_criteria: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionConfig:
    """Unified execution configuration"""
    max_iterations: int = 1               # 1 = single-pass, >1 = multi-iteration
    completion_criteria: Optional[CompletionCriteria] = None
    protocol_preference: Optional[str] = None  # Protocol to use
    tool_utilization_mode: ToolMode = ToolMode.COMPREHENSIVE
    quality_threshold: float = 0.85      # Minimum quality score
    execution_mode: ExecutionMode = ExecutionMode.OPTIMAL
    
    def __post_init__(self):
        if self.completion_criteria is None:
            self.completion_criteria = CompletionCriteria(
                min_quality_score=self.quality_threshold
            )


@dataclass
class ExecutionContext:
    """Universal context for all execution types"""
    inputs: Any                           # Task inputs (any format)
    shared_context: Dict[str, Any]        # Full shared context
    stage_info: StageInfo                 # Stage metadata
    execution_config: ExecutionConfig    # Execution parameters


@dataclass
class ExecutionMetadata:
    """Metadata about how execution was performed"""
    mode: ExecutionMode
    protocol_used: str
    execution_time: float
    iterations_planned: int
    tools_utilized: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    @property
    def duration(self) -> float:
        return self.execution_time


@dataclass
class AgentExecutionResult:
    """Standardized result format for unified execution"""
    output: Any                          # Agent's primary output
    execution_metadata: ExecutionMetadata  # How it was executed
    iterations_completed: int            # Actual iterations used
    completion_reason: CompletionReason  # Why execution stopped
    quality_score: float                 # Assessed output quality
    protocol_used: str                   # Protocol that was executed
    error_details: Optional[str] = None  # Error information if any
    
    @property
    def success(self) -> bool:
        """Legacy compatibility property"""
        return self.completion_reason not in [
            CompletionReason.ERROR_OCCURRED,
            CompletionReason.USER_REQUESTED_STOP
        ]


@dataclass
class IterationResult:
    """Result from a single iteration within multi-iteration execution"""
    output: Any
    quality_score: float
    tools_used: List[str]
    protocol_used: str
    iteration_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionAssessment:
    """Assessment of whether execution is complete"""
    is_complete: bool
    reason: CompletionReason
    quality_score: float
    gaps_identified: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# Legacy compatibility types for Phase 1
class AgentOutput(BaseModel):
    """Legacy compatibility for existing agent outputs"""
    success: bool = True
    error_message: Optional[str] = None
    execution_time: float = 0.0
    output: Any = None
    
    @classmethod
    def from_execution_result(cls, result: AgentExecutionResult) -> "AgentOutput":
        """Convert from new format to legacy format"""
        return cls(
            success=result.success,
            error_message=result.error_details,
            execution_time=result.execution_metadata.execution_time,
            output=result.output
        ) 