"""
Autonomous Engine Agents

This module contains all the specialized agents for the Autonomous Project Engine.
These agents work together to handle different aspects of software development
through the Autonomous Refinement Cycle Architecture (ARCA).

All agents now use registry-first architecture with auto-registration decorators.
No fallback maps needed - agents are auto-registered when imported.
"""

from __future__ import annotations

from typing import List, Type, TYPE_CHECKING
from chungoid.utils.agent_registry import AgentCard

# Import agent classes - these will auto-register via @register_agent decorators
from .automated_refinement_coordinator_agent import AutomatedRefinementCoordinatorAgent_v1
from .architect_agent import EnhancedArchitectAgent_v1
from .requirements_tracer_agent import RequirementsTracerAgent_v1
from .project_documentation_agent import ProjectDocumentationAgent_v1
from .code_debugging_agent import CodeDebuggingAgent_v1
from .smart_code_generator_agent import SmartCodeGeneratorAgent_v1
from .project_setup_agent import ProjectSetupAgent_v1

# NEW: Consolidated Requirements & Risk Agent (replaces ProductAnalystAgent + ProactiveRiskAssessorAgent)
from .requirements_risk_agent import RequirementsRiskAgent

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from chungoid.agents.unified_agent import UnifiedAgent

AUTONOMOUS_ENGINE_AGENTS_WITH_CARDS: List[Type['UnifiedAgent']] = [
    # NEW: Consolidated agent replacing ProductAnalystAgent_v1 + ProactiveRiskAssessorAgent_v1
    RequirementsRiskAgent,
    
    AutomatedRefinementCoordinatorAgent_v1,
    EnhancedArchitectAgent_v1, 
    RequirementsTracerAgent_v1,
    ProjectDocumentationAgent_v1,
    CodeDebuggingAgent_v1,
    SmartCodeGeneratorAgent_v1,
    
    # NOTE: ProjectSetupAgent_v1 consolidated environment_bootstrap + dependency_management + documentation
    ProjectSetupAgent_v1,
]

def get_autonomous_engine_agent_cards() -> List[AgentCard]:
    """Collects and returns AgentCards for all primary Autonomous Project Engine agents."""
    cards: List[AgentCard] = []
    for agent_class in AUTONOMOUS_ENGINE_AGENTS_WITH_CARDS:
        if hasattr(agent_class, 'get_agent_card_static') and callable(agent_class.get_agent_card_static):
            try:
                cards.append(agent_class.get_agent_card_static())
            except Exception as e:
                # Log this error appropriately in a real system
                print(f"Error getting agent card for {agent_class.AGENT_ID if hasattr(agent_class, 'AGENT_ID') else agent_class.__name__}: {e}")
        else:
            print(f"Warning: Agent class {agent_class.__name__} does not have a get_agent_card_static method.")
    return cards

# REMOVED: get_autonomous_engine_agent_fallback_map() - replaced with registry-first architecture
# All agents are now auto-registered via @register_agent decorators when imported
# No fallback maps needed - registry is the single source of truth

# Export all agents
__all__ = [
    "get_autonomous_engine_agent_cards",
    # REMOVED: "get_autonomous_engine_agent_fallback_map" - no longer needed
    "RequirementsRiskAgent",
    "AutomatedRefinementCoordinatorAgent_v1",
    "EnhancedArchitectAgent_v1",
    "RequirementsTracerAgent_v1",
    "ProjectDocumentationAgent_v1",
    "CodeDebuggingAgent_v1",
    "SmartCodeGeneratorAgent_v1",
    "ProjectSetupAgent_v1",
]

# Legacy agent map for backward compatibility
LEGACY_AGENT_MAP = {
    # ... existing code ...
    # "BlueprintReviewerAgent_v1",  # REMOVED - Use EnhancedArchitectAgent_v1 instead
    # ... existing code ...
} 