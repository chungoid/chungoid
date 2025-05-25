"""
Agent Auto-Registration System for Registry-First Architecture

This module provides the registry-first architecture components that replace
the fragmented fallback map system with a unified agent registration approach.
"""

import logging
from typing import Dict, List

# REMOVED: Circular import - moved to function level
# from chungoid.registry import get_global_agent_registry, reset_global_registry

logger = logging.getLogger(__name__)

def initialize_all_agents() -> Dict[str, bool]:
    """Initialize all agents by importing modules (triggers @register_agent decorators).
    
    Returns:
        Dict mapping agent_id to validation success status
        
    Raises:
        RuntimeError: If any agents fail validation
    """
    from chungoid.registry import get_global_agent_registry  # Import here to avoid circular import
    
    logger.info("Starting agent auto-registration via module imports...")
    
    # Import all agent modules to trigger registration
    # This will cause all @register_agent decorators to execute
    
    # System agents
    try:
        from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent
        from chungoid.runtime.agents.system_requirements_gathering_agent import SystemRequirementsGatheringAgent_v1
        from chungoid.runtime.agents.core_code_generator_agent import CoreCodeGeneratorAgent_v1
        from chungoid.runtime.agents.smart_code_integration_agent import SmartCodeIntegrationAgent_v1
        from chungoid.runtime.agents.system_file_system_agent import SystemFileSystemAgent_v1
        from chungoid.runtime.agents.system_master_planner_reviewer_agent import MasterPlannerReviewerAgent
        from chungoid.runtime.agents.system_agents.noop_agent import NoOpAgent_v1
        from chungoid.runtime.agents.system_intervention_agent import SystemInterventionAgent_v1
        logger.info("System agents imported successfully")
    except Exception as e:
        logger.error(f"Failed to import system agents: {e}")
        raise RuntimeError(f"System agent import failed: {e}")
    
    # Autonomous engine agents
    try:
        from chungoid.agents.autonomous_engine.environment_bootstrap_agent import EnvironmentBootstrapAgent
        from chungoid.agents.autonomous_engine.dependency_management_agent import DependencyManagementAgent_v1
        from chungoid.agents.autonomous_engine.proactive_risk_assessor_agent import ProactiveRiskAssessorAgent_v1
        from chungoid.agents.autonomous_engine.product_analyst_agent import ProductAnalystAgent_v1
        from chungoid.agents.autonomous_engine.architect_agent import ArchitectAgent_v1
        from chungoid.agents.autonomous_engine.blueprint_reviewer_agent import BlueprintReviewerAgent_v1
        from chungoid.agents.autonomous_engine.requirements_tracer_agent import RequirementsTracerAgent_v1
        from chungoid.agents.autonomous_engine.project_documentation_agent import ProjectDocumentationAgent_v1
        from chungoid.agents.autonomous_engine.code_debugging_agent import CodeDebuggingAgent_v1
        from chungoid.agents.autonomous_engine.automated_refinement_coordinator_agent import AutomatedRefinementCoordinatorAgent_v1
        logger.info("Autonomous engine agents imported successfully")
    except Exception as e:
        logger.error(f"Failed to import autonomous engine agents: {e}")
        raise RuntimeError(f"Autonomous engine agent import failed: {e}")
    
    # Get the global registry and validate all registered agents
    registry = get_global_agent_registry()
    
    # Validate all registered agents
    validation_results = registry.validate_agents()
    failed_agents = [agent_id for agent_id, success in validation_results.items() if not success]
    
    if failed_agents:
        logger.error(f"Failed to validate agents: {failed_agents}")
        raise RuntimeError(f"Agent validation failed for: {failed_agents}")
    
    total_agents = len(registry.list_agents())
    logger.info(f"Successfully registered and validated {total_agents} agents")
    
    # Log all registered agents for verification
    registered_agents = registry.list_agents()
    logger.info("Registered agents:")
    for agent_id, agent_class in registered_agents.items():
        metadata = registry.get_agent_metadata(agent_id)
        category = metadata.category if metadata else "unknown"
        logger.info(f"   • {agent_id} ({agent_class.__name__}) - {category}")
    
    return validation_results

def get_registry_agent_provider(llm_provider=None, prompt_manager=None):
    """Create a registry-first agent provider with NO fallback maps.
    
    This replaces the old fallback map approach with pure registry lookups.
    Explicitly initializes agents to ensure registry is populated.
    """
    from chungoid.registry import get_global_agent_registry  # Import here to avoid circular import
    from chungoid.utils.agent_resolver import RegistryAgentProvider
    
    # Explicitly initialize all agents to populate the registry
    logger.info("Initializing agents for registry-first provider...")
    try:
        initialize_all_agents()
        logger.info("Agent initialization completed successfully")
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize agents for registry provider: {e}")
    
    registry = get_global_agent_registry()
    
    # NO FALLBACK - Registry is the single source of truth
    return RegistryAgentProvider(
        registry=registry,
        fallback=None,  # NO FALLBACK MAPS
        llm_provider=llm_provider,
        prompt_manager=prompt_manager
    )

def discover_agents_by_capability(capability: str) -> List[str]:
    """Find agents that can handle specific capabilities."""
    from chungoid.registry import get_global_agent_registry  # Import here to avoid circular import
    registry = get_global_agent_registry()
    return registry.discover_agents(capability=capability)

def discover_agents_by_category(category: str) -> List[str]:
    """Find agents in specific category (system, autonomous_engine, etc.)."""
    from chungoid.registry import get_global_agent_registry  # Import here to avoid circular import
    registry = get_global_agent_registry()
    return registry.discover_agents(category=category)

def monitor_agent_health() -> Dict[str, str]:
    """Monitor health of all registered agents."""
    from chungoid.registry import get_global_agent_registry  # Import here to avoid circular import
    registry = get_global_agent_registry()
    health_status = {}
    
    for agent_id, agent_class in registry.list_agents().items():
        try:
            # Basic health check - verify class is importable and has required methods
            if hasattr(agent_class, 'AGENT_ID') and hasattr(agent_class, 'execute'):
                health_status[agent_id] = "HEALTHY"
            else:
                health_status[agent_id] = "UNHEALTHY: Missing required attributes"
        except Exception as e:
            health_status[agent_id] = f"UNHEALTHY: {e}"
    
    return health_status

# REMOVED: Auto-initialization to prevent circular imports
# The registry-first architecture should be initialized explicitly by the CLI
# when needed, not automatically during module import
# 
# Auto-initialize agents when this module is imported
# This ensures the registry is populated as soon as the agents module is used
# try:
#     initialize_all_agents()
#     logger.info("Agent auto-registration completed successfully")
# except Exception as e:
#     logger.error(f"Agent auto-registration failed: {e}")
#     # Don't raise here to avoid breaking imports, but log the error
#     # The calling code should handle validation failures appropriately 