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
    
    # Legacy system agents temporarily disabled during Phase-3 migration
    # These will be migrated to UnifiedAgent pattern and re-enabled
    try:
        # from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent
        # from chungoid.runtime.agents.system_requirements_gathering_agent import SystemRequirementsGatheringAgent_v1
        # from chungoid.runtime.agents.core_code_generator_agent import CoreCodeGeneratorAgent_v1
        # from chungoid.runtime.agents.smart_code_integration_agent import SmartCodeIntegrationAgent_v1
        # from chungoid.runtime.agents.system_file_system_agent import SystemFileSystemAgent_v1
        # from chungoid.runtime.agents.system_master_planner_reviewer_agent import MasterPlannerReviewerAgent
        # from chungoid.runtime.agents.system_agents.noop_agent import NoOpAgent_v1
        # from chungoid.runtime.agents.system_intervention_agent import SystemInterventionAgent_v1
        logger.info("Legacy system agents temporarily disabled during Phase-3 migration")
    except Exception as e:
        logger.error(f"System agent import issue during Phase-3 migration: {e}")
        # Don't raise during migration phase
    
    # Autonomous engine agents
    try:
        # REMOVED: These agents were consolidated into other agents:
        # - EnvironmentBootstrapAgent → ProjectSetupAgent_v1 (environment capability)
        # - DependencyManagementAgent_v1 → ProjectSetupAgent_v1 (dependencies capability)
        # - ProductAnalystAgent_v1 + ProactiveRiskAssessorAgent_v1 → RequirementsRiskAgent
        
        from chungoid.agents.autonomous_engine.requirements_risk_agent import RequirementsRiskAgent_v1
        from chungoid.agents.autonomous_engine.architect_agent import ArchitectAgent_v1
        # NOTE: BlueprintToFlowAgent_v1 was consolidated into EnhancedArchitectAgent_v1
        from chungoid.agents.autonomous_engine.requirements_tracer_agent import RequirementsTracerAgent_v1
        from chungoid.agents.autonomous_engine.smart_code_generator_agent import SmartCodeGeneratorAgent_v1
        from chungoid.agents.autonomous_engine.code_debugging_agent import CodeDebuggingAgent_v1
        from chungoid.agents.autonomous_engine.automated_refinement_coordinator_agent import AutomatedRefinementCoordinatorAgent_v1
        from chungoid.agents.autonomous_engine.project_documentation_agent import ProjectDocumentationAgent_v1
        from chungoid.agents.autonomous_engine.project_setup_agent import ProjectSetupAgent_v1
        
        # NOTE: SystemRequirementsGatheringAgent_v1 and ARCAOptimizationEvaluatorAgent_v1 are 
        # orphaned prompts (prompt files exist but no agent class implementations)
        # The functionality exists in other agents:
        # - RequirementsRiskAgent handles requirements gathering and risk assessment
        # - AutomatedRefinementCoordinatorAgent_v1 handles optimization evaluation
        # - Protocol classes provide additional optimization capabilities
        
        logger.info("Autonomous engine agents imported successfully")
    except Exception as e:
        logger.error(f"Failed to import autonomous engine agents: {e}")
        raise RuntimeError(f"Autonomous engine agent import failed: {e}")
    
    # Interactive and utility agents
    try:
        from chungoid.agents.interactive_requirements_agent import InteractiveRequirementsAgent
        logger.info("Interactive and utility agents imported successfully")
    except Exception as e:
        logger.error(f"Failed to import interactive and utility agents: {e}")
        raise RuntimeError(f"Interactive and utility agent import failed: {e}")
    
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
    """Create a unified agent resolver for Phase 3 UAEI architecture.
    
    PHASE 3 MIGRATION: This now returns UnifiedAgentResolver instead of legacy RegistryAgentProvider.
    Eliminates all technical debt from complex resolver patterns.
    """
    from chungoid.registry import get_global_agent_registry  # Import here to avoid circular import
    from chungoid.runtime.unified_agent_resolver import UnifiedAgentResolver
    
    # Explicitly initialize all agents to populate the registry
    logger.info("Initializing agents for Phase 3 UAEI resolver...")
    try:
        initialize_all_agents()
        logger.info("Agent initialization completed successfully")
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize agents for UAEI resolver: {e}")
    
    registry = get_global_agent_registry()
    
    # Phase 3: Use UnifiedAgentResolver - simple, single-path resolution
    return UnifiedAgentResolver(
        agent_registry=registry,
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