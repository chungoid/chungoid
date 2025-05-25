"""
Collaboration Protocol Suite

Contains protocols for multi-agent collaboration, team formation,
shared execution context, and collaborative result integration.

Week 4 Implementation: Multi-Agent Collaboration with Shared Tools
"""

from .autonomous_team_formation import AutonomousTeamFormationProtocol
from .shared_execution_context import SharedExecutionContextProtocol

__all__ = [
    'AutonomousTeamFormationProtocol',
    'SharedExecutionContextProtocol'
] 