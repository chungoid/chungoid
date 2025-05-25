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