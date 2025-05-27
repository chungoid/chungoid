"""
Universal Protocol Infrastructure

Universal protocols that provide foundational capabilities across all agents
and workflows in the chungoid system.

Change Reference: 3.14 (NEW)
"""

from .agent_communication import AgentCommunicationProtocol
from .context_sharing import ContextSharingProtocol
from .tool_validation import ToolValidationProtocol
from .error_recovery import ErrorRecoveryProtocol
from .goal_tracking import GoalTrackingProtocol
from .tool_use import ToolUseProtocol
from .reflection import ReflectionProtocol

__all__ = [
    'AgentCommunicationProtocol',
    'ContextSharingProtocol', 
    'ToolValidationProtocol',
    'ErrorRecoveryProtocol',
    'GoalTrackingProtocol',
    'ReflectionProtocol',
    'ToolUseProtocol'
] 