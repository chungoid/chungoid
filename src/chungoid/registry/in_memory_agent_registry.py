"""
In-Memory Agent Registry for Registry-First Architecture

This module provides a fast, in-memory agent registry that serves as the single source of truth
for all agent registration and discovery. It replaces the fragmented fallback map system with
a unified registry-first approach.
"""

import logging
from typing import Dict, List, Optional, Type, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility

logger = logging.getLogger(__name__)


@dataclass
class AgentMetadata:
    """Metadata for registered agents."""
    category: str = "system"
    capabilities: List[str] = None
    module: str = ""
    class_name: str = ""
    priority: int = 0
    visibility: AgentVisibility = AgentVisibility.PUBLIC
    registered_at: datetime = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.registered_at is None:
            self.registered_at = datetime.now(timezone.utc)
    
    @classmethod
    def from_agent_class(cls, agent_class: Type[BaseAgent]) -> 'AgentMetadata':
        """Create metadata from agent class inspection."""
        return cls(
            category="system",  # Default, can be overridden by decorator
            capabilities=[],
            module=agent_class.__module__,
            class_name=agent_class.__name__,
            priority=0,
            visibility=AgentVisibility.PUBLIC
        )


class InMemoryAgentRegistry:
    """
    Fast, in-memory agent registry that serves as the single source of truth.
    
    This registry replaces the fragmented fallback map system with a unified
    approach where all agents register themselves and all lookups go through
    the registry.
    """
    
    def __init__(self):
        self._agents: Dict[str, Type[BaseAgent]] = {}
        self._agent_metadata: Dict[str, AgentMetadata] = {}
        self._initialized = False
        logger.info("InMemoryAgentRegistry initialized")
    
    def register_agent(self, agent_class: Type[BaseAgent], metadata: Optional[AgentMetadata] = None):
        """Register an agent in the registry."""
        if not hasattr(agent_class, 'AGENT_ID'):
            raise ValueError(f"Agent class {agent_class.__name__} must have AGENT_ID attribute")
        
        agent_id = agent_class.AGENT_ID
        
        if agent_id in self._agents:
            logger.warning(f"Agent {agent_id} already registered, overwriting")
        
        self._agents[agent_id] = agent_class
        self._agent_metadata[agent_id] = metadata or AgentMetadata.from_agent_class(agent_class)
        
        logger.info(f"Registered agent: {agent_id} ({agent_class.__name__})")
    
    def get_agent(self, agent_id: str) -> Optional[Type[BaseAgent]]:
        """Get agent class by ID."""
        agent_class = self._agents.get(agent_id)
        if agent_class:
            logger.debug(f"Found agent: {agent_id}")
        else:
            logger.warning(f"Agent not found: {agent_id}")
        return agent_class
    
    def list_agents(self) -> Dict[str, Type[BaseAgent]]:
        """List all registered agents."""
        logger.debug(f"Listing {len(self._agents)} registered agents")
        return self._agents.copy()
    
    def get_agent_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
        """Get metadata for an agent."""
        return self._agent_metadata.get(agent_id)
    
    def discover_agents(self, capability: str = None, category: str = None) -> List[Type[BaseAgent]]:
        """Discover agents by capability or category."""
        matching_agents = []
        
        for agent_id, agent_class in self._agents.items():
            metadata = self._agent_metadata.get(agent_id)
            if not metadata:
                continue
            
            # Filter by category
            if category and metadata.category != category:
                continue
            
            # Filter by capability
            if capability and capability not in metadata.capabilities:
                continue
            
            matching_agents.append(agent_class)
        
        logger.debug(f"Discovered {len(matching_agents)} agents for capability='{capability}', category='{category}'")
        return matching_agents
    
    def list_agents_by_capability(self, capability: str) -> List[str]:
        """List all agent IDs that have a specific capability."""
        matching_agent_ids = []
        
        for agent_id, agent_class in self._agents.items():
            # Check metadata capabilities
            metadata = self._agent_metadata.get(agent_id)
            if metadata and capability in metadata.capabilities:
                matching_agent_ids.append(agent_id)
                continue
            
            # Check agent class protocols for autonomous agents
            if hasattr(agent_class, 'PRIMARY_PROTOCOLS'):
                primary_protocols = getattr(agent_class, 'PRIMARY_PROTOCOLS', [])
                if capability in primary_protocols:
                    matching_agent_ids.append(agent_id)
                    continue
            
            if hasattr(agent_class, 'SECONDARY_PROTOCOLS'):
                secondary_protocols = getattr(agent_class, 'SECONDARY_PROTOCOLS', [])
                if capability in secondary_protocols:
                    matching_agent_ids.append(agent_id)
                    continue
        
        logger.debug(f"Found {len(matching_agent_ids)} agents with capability '{capability}': {matching_agent_ids}")
        return matching_agent_ids
    
    def list_autonomous_agents(self) -> List[str]:
        """List all autonomous-capable agent IDs."""
        autonomous_agent_ids = []
        
        for agent_id, agent_class in self._agents.items():
            # Check if it's a ProtocolAwareAgent with protocols
            try:
                from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
                if (issubclass(agent_class, ProtocolAwareAgent) and
                    hasattr(agent_class, 'PRIMARY_PROTOCOLS') and
                    getattr(agent_class, 'PRIMARY_PROTOCOLS', [])):
                    autonomous_agent_ids.append(agent_id)
            except ImportError:
                # ProtocolAwareAgent not available, skip autonomous detection
                continue
        
        logger.debug(f"Found {len(autonomous_agent_ids)} autonomous agents: {autonomous_agent_ids}")
        return autonomous_agent_ids
    
    def find_best_agent_for_task(self, task_type: str, required_capabilities: List[str], 
                                prefer_autonomous: bool = True) -> Optional[str]:
        """Find the best agent ID for a specific task type and capabilities."""
        
        # Task type to preferred agent mapping
        task_type_preferences = {
            "requirements_analysis": ["ProductAnalystAgent_v1"],
            "architecture_design": ["ArchitectAgent_v1"],
            "environment_setup": ["EnvironmentBootstrapAgent"],
            "dependency_management": ["DependencyManagementAgent_v1"],
            "code_generation": ["SmartCodeGeneratorAgent_v1", "CoreCodeGeneratorAgent_v1"],
            "code_debugging": ["CodeDebuggingAgent_v1"],
            "quality_validation": ["BlueprintReviewerAgent_v1"],
            "risk_assessment": ["ProactiveRiskAssessorAgent_v1"],
            "documentation": ["ProjectDocumentationAgent_v1"],
            "file_operations": ["SystemFileSystemAgent_v1"],
            "project_coordination": ["AutomatedRefinementCoordinatorAgent_v1"],
            "requirements_traceability": ["RequirementsTracerAgent_v1"]
        }
        
        # 1. Try preferred agents for task type first
        preferred_agents = task_type_preferences.get(task_type, [])
        for preferred_agent_id in preferred_agents:
            if preferred_agent_id in self._agents:
                agent_class = self._agents[preferred_agent_id]
                if prefer_autonomous:
                    # Check if it's autonomous
                    if preferred_agent_id in self.list_autonomous_agents():
                        if self._agent_matches_capabilities(preferred_agent_id, required_capabilities):
                            logger.debug(f"Found preferred autonomous agent '{preferred_agent_id}' for task_type '{task_type}'")
                            return preferred_agent_id
                else:
                    # Any agent is fine
                    if self._agent_matches_capabilities(preferred_agent_id, required_capabilities):
                        logger.debug(f"Found preferred agent '{preferred_agent_id}' for task_type '{task_type}'")
                        return preferred_agent_id
        
        # 2. Try autonomous agents if preferred
        if prefer_autonomous:
            for agent_id in self.list_autonomous_agents():
                if self._agent_matches_capabilities(agent_id, required_capabilities):
                    logger.debug(f"Found autonomous agent '{agent_id}' matching capabilities {required_capabilities}")
                    return agent_id
        
        # 3. Try any agent with matching capabilities
        for capability in required_capabilities:
            matching_agents = self.list_agents_by_capability(capability)
            for agent_id in matching_agents:
                if self._agent_matches_capabilities(agent_id, required_capabilities):
                    logger.debug(f"Found agent '{agent_id}' with matching capabilities")
                    return agent_id
        
        # 4. Last resort - any agent that exists
        if preferred_agents:
            for preferred_agent_id in preferred_agents:
                if preferred_agent_id in self._agents:
                    logger.warning(f"Using fallback agent '{preferred_agent_id}' for task_type '{task_type}' without capability match")
                    return preferred_agent_id
        
        logger.warning(f"No suitable agent found for task_type='{task_type}' with capabilities={required_capabilities}")
        return None
    
    def _agent_matches_capabilities(self, agent_id: str, required_capabilities: List[str]) -> bool:
        """Check if an agent matches the required capabilities."""
        if not required_capabilities:
            return True
        
        agent_class = self._agents.get(agent_id)
        if not agent_class:
            return False
        
        # Get all agent capabilities
        agent_capabilities = set()
        
        # From metadata
        metadata = self._agent_metadata.get(agent_id)
        if metadata and metadata.capabilities:
            agent_capabilities.update(metadata.capabilities)
        
        # From protocols (for autonomous agents)
        if hasattr(agent_class, 'PRIMARY_PROTOCOLS'):
            agent_capabilities.update(getattr(agent_class, 'PRIMARY_PROTOCOLS', []))
        
        if hasattr(agent_class, 'SECONDARY_PROTOCOLS'):
            agent_capabilities.update(getattr(agent_class, 'SECONDARY_PROTOCOLS', []))
        
        # Check for matches (require at least 70% match)
        required_set = set(required_capabilities)
        matched = agent_capabilities & required_set
        match_ratio = len(matched) / len(required_set) if required_set else 1.0
        
        return match_ratio >= 0.7
    
    def get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Get all capabilities for a specific agent."""
        agent_class = self._agents.get(agent_id)
        if not agent_class:
            return []
        
        capabilities = set()
        
        # From metadata
        metadata = self._agent_metadata.get(agent_id)
        if metadata and metadata.capabilities:
            capabilities.update(metadata.capabilities)
        
        # From protocols (for autonomous agents)
        if hasattr(agent_class, 'PRIMARY_PROTOCOLS'):
            capabilities.update(getattr(agent_class, 'PRIMARY_PROTOCOLS', []))
        
        if hasattr(agent_class, 'SECONDARY_PROTOCOLS'):
            capabilities.update(getattr(agent_class, 'SECONDARY_PROTOCOLS', []))
        
        return list(capabilities)
    
    def update_agent_capabilities(self, agent_id: str, capabilities: List[str]):
        """Update capabilities for an agent."""
        if agent_id not in self._agent_metadata:
            logger.warning(f"Agent '{agent_id}' not found for capability update")
            return
        
        self._agent_metadata[agent_id].capabilities = capabilities
        logger.info(f"Updated capabilities for agent '{agent_id}': {capabilities}")
    
    def validate_agents(self) -> Dict[str, bool]:
        """Validate all registered agents can be instantiated."""
        results = {}
        
        for agent_id, agent_class in self._agents.items():
            try:
                # Test basic class structure
                if not hasattr(agent_class, 'AGENT_ID'):
                    results[agent_id] = False
                    logger.error(f"Agent {agent_id} missing AGENT_ID attribute")
                    continue
                
                if not hasattr(agent_class, 'invoke_async'):
                    results[agent_id] = False
                    logger.error(f"Agent {agent_id} missing invoke_async method")
                    continue
                
                # Test that AGENT_ID matches registration
                if agent_class.AGENT_ID != agent_id:
                    results[agent_id] = False
                    logger.error(f"Agent {agent_id} AGENT_ID mismatch: {agent_class.AGENT_ID}")
                    continue
                
                results[agent_id] = True
                logger.debug(f"Agent {agent_id} validation passed")
                
            except Exception as e:
                logger.error(f"Agent {agent_id} validation failed: {e}")
                results[agent_id] = False
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Agent validation complete: {passed}/{total} passed")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        categories = {}
        for metadata in self._agent_metadata.values():
            categories[metadata.category] = categories.get(metadata.category, 0) + 1
        
        return {
            "total_agents": len(self._agents),
            "categories": categories,
            "initialized": self._initialized
        }
    
    def mark_initialized(self):
        """Mark registry as fully initialized."""
        self._initialized = True
        stats = self.get_stats()
        logger.info(f"Registry marked as initialized: {stats}")


# Global registry instance
_global_registry = InMemoryAgentRegistry()


def get_global_agent_registry() -> InMemoryAgentRegistry:
    """Get the global agent registry instance."""
    return _global_registry


def reset_global_registry():
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = InMemoryAgentRegistry()
    logger.info("Global registry reset") 