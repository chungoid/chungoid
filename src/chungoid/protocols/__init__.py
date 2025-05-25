"""
Chungoid Protocol System

This module provides a framework for agents to follow rigorous, proven methodologies
for investigation, implementation, and quality assurance.

Key Benefits:
- Transforms agents from "single-shot executors" to "protocol-driven engineers"
- Ensures consistent, thorough approaches across all autonomous development
- Enables iterative improvement through systematic validation
- Provides quality gates and validation checkpoints

Change Reference: 3.01 (ENHANCE) - Added universal protocols and expanded registry
"""

from .base.protocol_interface import ProtocolInterface, ProtocolPhase
from .base.validation import ProtocolValidator, ValidationResult
from .base.execution_engine import ProtocolExecutionEngine

# Existing protocols
from .investigation.deep_investigation import DeepInvestigationProtocol
from .implementation.deep_implementation import DeepImplementationProtocol
from .quality.quality_gates import QualityGateValidator

# Universal protocols (NEW)
from .universal.agent_communication import AgentCommunicationProtocol
from .universal.context_sharing import ContextSharingProtocol
from .universal.tool_validation import ToolValidationProtocol
from .universal.error_recovery import ErrorRecoveryProtocol
from .universal.goal_tracking import GoalTrackingProtocol

__all__ = [
    # Base protocol infrastructure
    "ProtocolInterface",
    "ProtocolPhase", 
    "ProtocolValidator",
    "ValidationResult",
    "ProtocolExecutionEngine",
    
    # Existing protocols
    "DeepInvestigationProtocol",
    "DeepImplementationProtocol",
    "QualityGateValidator",
    
    # Universal protocols
    "AgentCommunicationProtocol",
    "ContextSharingProtocol",
    "ToolValidationProtocol",
    "ErrorRecoveryProtocol",
    "GoalTrackingProtocol",
    
    # Registry functions
    "get_protocol",
    "list_available_protocols"
]

# Enhanced Protocol Registry for dynamic loading
PROTOCOL_REGISTRY = {
    # Existing protocols
    "deep_investigation": DeepInvestigationProtocol,
    "deep_implementation": DeepImplementationProtocol,
    
    # Universal protocols
    "agent_communication": AgentCommunicationProtocol,
    "context_sharing": ContextSharingProtocol,
    "tool_validation": ToolValidationProtocol,
    "error_recovery": ErrorRecoveryProtocol,
    "goal_tracking": GoalTrackingProtocol,
}

def get_protocol(name: str) -> ProtocolInterface:
    """Get a protocol instance by name."""
    if name not in PROTOCOL_REGISTRY:
        raise ValueError(f"Unknown protocol: {name}")
    
    return PROTOCOL_REGISTRY[name]()

def list_available_protocols() -> list[str]:
    """List all available protocol names."""
    return list(PROTOCOL_REGISTRY.keys()) 