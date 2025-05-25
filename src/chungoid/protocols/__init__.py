"""
Chungoid Protocol System

Comprehensive protocol ecosystem for autonomous agent execution.
"""

# Base protocol infrastructure
from .base.protocol_interface import ProtocolInterface, ProtocolPhase, ProtocolTemplate, PhaseStatus
from .base.validation import ProtocolValidator, ValidationResult
from .base.execution_engine import ProtocolExecutionEngine

# Universal protocols (5 protocols)
from .universal.agent_communication import AgentCommunicationProtocol
from .universal.context_sharing import ContextSharingProtocol
from .universal.tool_validation import ToolValidationProtocol
from .universal.error_recovery import ErrorRecoveryProtocol
from .universal.goal_tracking import GoalTrackingProtocol

# Planning protocols (4 protocols)
from .planning.architecture_planning import ArchitecturePlanningProtocol
from .planning.deep_planning_verification import DeepPlanningVerificationProtocol
from .planning.enhanced_deep_planning import EnhancedDeepPlanningProtocol
from .planning.planning_agent_protocol import PlanningAgentProtocol

# Collaboration protocols (2 protocols - Week 4)
from .collaboration.autonomous_team_formation import AutonomousTeamFormationProtocol
from .collaboration.shared_execution_context import SharedExecutionContextProtocol

# Observability protocols (2 protocols - Week 5)
from .observability import (
    AutonomousArchitectureVisualizerProtocol,
    ArchitectureDriftDetectorProtocol,
    ArchitectureSnapshot,
    C4ModelDiagram,
    DiagramLevel,
    DiagramFormat,
    VisualizationConfig,
    ArchitectureBaseline,
    DriftMetric,
    DriftAnalysis,
    DriftRecommendation,
    DriftAlert
)

# Week 6: Production Readiness & Comprehensive Evaluation
from .evaluation import (
    AutonomousExecutionEvaluatorProtocol,
    EvaluationMetric,
    EvaluationResult,
    EvaluationConfig,
    MetricType,
    EvaluationSeverity,
    PerformanceAnalysis,
    OptimizationRecommendation
)

__all__ = [
    # Base infrastructure
    'ProtocolInterface', 'ProtocolPhase', 'ProtocolTemplate', 'PhaseStatus',
    'ProtocolValidator', 'ValidationResult', 'ProtocolExecutionEngine',
    
    # Universal protocols (5)
    'AgentCommunicationProtocol', 'ContextSharingProtocol', 'ToolValidationProtocol',
    'ErrorRecoveryProtocol', 'GoalTrackingProtocol',
    
    # Planning protocols (4)
    'ArchitecturePlanningProtocol', 'DeepPlanningVerificationProtocol',
    'EnhancedDeepPlanningProtocol', 'PlanningAgentProtocol',
    
    # Collaboration protocols (2 - Week 4)
    'AutonomousTeamFormationProtocol', 'SharedExecutionContextProtocol',
    
    # Observability protocols (2 - Week 5)
    'AutonomousArchitectureVisualizerProtocol', 'ArchitectureDriftDetectorProtocol',
    'ArchitectureSnapshot', 'C4ModelDiagram', 'DiagramLevel', 'DiagramFormat',
    'VisualizationConfig', 'ArchitectureBaseline', 'DriftMetric', 'DriftAnalysis',
    'DriftRecommendation', 'DriftAlert',
    
    # Week 6: Production Readiness & Comprehensive Evaluation
    'AutonomousExecutionEvaluatorProtocol', 'EvaluationMetric', 'EvaluationResult',
    'EvaluationConfig', 'MetricType', 'EvaluationSeverity', 'PerformanceAnalysis',
    'OptimizationRecommendation'
]

# Protocol registry for easy access
AVAILABLE_PROTOCOLS = {
    # Universal protocols
    "agent_communication": AgentCommunicationProtocol,
    "context_sharing": ContextSharingProtocol,
    "tool_validation": ToolValidationProtocol,
    "error_recovery": ErrorRecoveryProtocol,
    "goal_tracking": GoalTrackingProtocol,
    
    # Planning protocols
    "architecture_planning": ArchitecturePlanningProtocol,
    "deep_planning_verification": DeepPlanningVerificationProtocol,
    "enhanced_deep_planning": EnhancedDeepPlanningProtocol,
    "planning_agent": PlanningAgentProtocol,
    
    # Collaboration protocols
    "autonomous_team_formation": AutonomousTeamFormationProtocol,
    "shared_execution_context": SharedExecutionContextProtocol,
    
    # Observability protocols
    "autonomous_architecture_visualizer": AutonomousArchitectureVisualizerProtocol,
    "architecture_drift_detector": ArchitectureDriftDetectorProtocol,
    
    # Evaluation protocols
    "autonomous_execution_evaluator": AutonomousExecutionEvaluatorProtocol,
}

def get_protocol(protocol_name: str) -> ProtocolInterface:
    """Get a protocol instance by name."""
    if protocol_name in AVAILABLE_PROTOCOLS:
        return AVAILABLE_PROTOCOLS[protocol_name]()
    else:
        raise ValueError(f"Protocol '{protocol_name}' not found. Available protocols: {list(AVAILABLE_PROTOCOLS.keys())}")

def list_available_protocols() -> list:
    """List all available protocol names."""
    return list(AVAILABLE_PROTOCOLS.keys())

# Total: 14 protocols (5 universal + 4 planning + 2 collaboration + 2 observability + 1 evaluation) 