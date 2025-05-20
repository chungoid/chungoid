"""Initializes and provides access to Autonomous Project Engine agent cards."""

from __future__ import annotations

from typing import List, Dict, Any, Type
from chungoid.utils.agent_registry import AgentCard
from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.schemas.common import AgentCallable

# Import agent classes
from .product_analyst_agent import ProductAnalystAgent_v1
from .proactive_risk_assessor_agent import ProactiveRiskAssessorAgent_v1
from .automated_refinement_coordinator_agent import AutomatedRefinementCoordinatorAgent_v1
from .architect_agent import ArchitectAgent_v1
from .requirements_tracer_agent import RequirementsTracerAgent_v1
from .blueprint_reviewer_agent import BlueprintReviewerAgent_v1
from .project_chroma_manager_agent import ProjectChromaManagerAgent_v1
from .project_documentation_agent import ProjectDocumentationAgent_v1
from .code_debugging_agent import CodeDebuggingAgent_v1
# The "Smart" Code Generator and Integrator are enhancements of core agents.
# Their registration will be handled by ensuring their core counterparts are updated
# or by registering their specific "Smart" cards if they are distinct entities.
# For now, we assume the Core agents *become* Smart, so their existing registration (if any) is key.
# If SmartCodeGeneratorAgent_v1 is a new distinct class, it should be imported and added here.
# from .smart_code_generator_agent import SmartCodeGeneratorAgent_v1 # Example if it were separate

# ProjectChromaManagerAgent is more of a utility/service, not typically called directly in a flow stage by ID in the same way.
# It will be instantiated and used by other agents or the orchestrator contextually.

AUTONOMOUS_ENGINE_AGENTS_WITH_CARDS: List[Type[BaseAgent]] = [
    ProductAnalystAgent_v1,
    ProactiveRiskAssessorAgent_v1,
    AutomatedRefinementCoordinatorAgent_v1,
    ArchitectAgent_v1,
    RequirementsTracerAgent_v1,
    BlueprintReviewerAgent_v1,
    ProjectChromaManagerAgent_v1,
    ProjectDocumentationAgent_v1,
    CodeDebuggingAgent_v1,
    # SmartCodeGeneratorAgent_v1, # if separate
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


def get_autonomous_engine_agent_fallback_map() -> Dict[str, AgentCallable]:
    """
    Returns a map of agent IDs to their classes for RegistryAgentProvider's fallback mechanism.
    This allows the provider to instantiate them directly if they aren't full MCP tools.
    """
    fallback_map: Dict[str, AgentCallable] = {}
    for agent_class in AUTONOMOUS_ENGINE_AGENTS_WITH_CARDS:
        if hasattr(agent_class, 'AGENT_ID') and isinstance(agent_class.AGENT_ID, str):
            fallback_map[agent_class.AGENT_ID] = agent_class # Map ID to the class itself
        else:
            print(f"Warning: Agent class {agent_class.__name__} does not have a valid AGENT_ID.")
    return fallback_map

# Example usage (for testing this file):
# if __name__ == "__main__":
#     all_cards = get_autonomous_engine_agent_cards()
#     print(f"Collected {len(all_cards)} Autonomous Engine agent cards:")
#     for card in all_cards:
#         print(f"  - {card.agent_id} (Version: {card.version})")
# 
#     fallback_map = get_autonomous_engine_agent_fallback_map()
#     print(f"\nCollected {len(fallback_map)} Autonomous Engine agents for fallback map:")
#     for agent_id, agent_cls in fallback_map.items():
#         print(f"  - {agent_id} -> {agent_cls.__name__}") 

__all__ = [
    "get_autonomous_engine_agent_cards",
    "get_autonomous_engine_agent_fallback_map",
    "ProductAnalystAgent_v1",
    "ProactiveRiskAssessorAgent_v1",
    "AutomatedRefinementCoordinatorAgent_v1",
    "ArchitectAgent_v1",
    "RequirementsTracerAgent_v1",
    "BlueprintReviewerAgent_v1",
    "ProjectChromaManagerAgent_v1",
    "ProjectDocumentationAgent_v1",
    "CodeDebuggingAgent_v1",
    # SmartCodeGeneratorAgent_v1, # if separate
] 