"""
In-Memory Agent Registry for Registry-First Architecture

This module provides a fast, in-memory agent registry that serves as the single source of truth
for all agent registration and discovery. It replaces the fragmented fallback map system with
a unified registry-first approach.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass, field
from datetime import datetime

from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility

logger = logging.getLogger(__name__)


@dataclass
class AgentMetadata:
    """Metadata for registered agents"""
    agent_id: str
    name: str
    description: str
    version: str
    category: AgentCategory
    visibility: AgentVisibility
    capabilities: List[str] = field(default_factory=list)
    primary_protocols: List[str] = field(default_factory=list)
    secondary_protocols: List[str] = field(default_factory=list)
    priority: int = 0
    registered_at: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_agent_class(cls, agent_class: Type[ProtocolAwareAgent]) -> 'AgentMetadata':
        """Create metadata from agent class"""
        return cls(
            agent_id=getattr(agent_class, 'AGENT_ID', agent_class.__name__),
            name=getattr(agent_class, 'AGENT_NAME', agent_class.__name__),
            description=getattr(agent_class, 'AGENT_DESCRIPTION', ''),
            version=getattr(agent_class, 'AGENT_VERSION', '1.0.0'),
            category=getattr(agent_class, 'CATEGORY', AgentCategory.SYSTEM_ORCHESTRATION),
            visibility=getattr(agent_class, 'VISIBILITY', AgentVisibility.PUBLIC),
            capabilities=list(getattr(agent_class, 'CAPABILITIES', [])),
            primary_protocols=list(getattr(agent_class, 'PRIMARY_PROTOCOLS', [])),
            secondary_protocols=list(getattr(agent_class, 'SECONDARY_PROTOCOLS', []))
        )


class InMemoryAgentRegistry:
    """In-memory registry for protocol-aware agents"""
    
    def __init__(self):
        self._agents: Dict[str, Type[ProtocolAwareAgent]] = {}
        self._metadata: Dict[str, AgentMetadata] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_agent(self, agent_class: Type[ProtocolAwareAgent], metadata: Optional[AgentMetadata] = None):
        """Register a protocol-aware agent"""
        if not hasattr(agent_class, 'AGENT_ID'):
            raise ValueError(f"Agent class {agent_class.__name__} must have AGENT_ID attribute")
        
        agent_id = agent_class.AGENT_ID
        
        if metadata is None:
            metadata = AgentMetadata.from_agent_class(agent_class)
        
        self._agents[agent_id] = agent_class
        self._metadata[agent_id] = metadata
        
        self.logger.info(f"Registered agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[Type[ProtocolAwareAgent]]:
        """Get agent class by ID"""
        return self._agents.get(agent_id)
    
    def get_agent_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
        """Get agent metadata by ID"""
        return self._metadata.get(agent_id)
    
    def list_agents(self) -> Dict[str, Type[ProtocolAwareAgent]]:
        """List all registered agents"""
        return self._agents.copy()
    
    def list_metadata(self) -> Dict[str, AgentMetadata]:
        """List all agent metadata"""
        return self._metadata.copy()
    
    def discover_agents(self, capability: str = None, category: str = None) -> List[Type[ProtocolAwareAgent]]:
        """Discover agents by capability or category"""
        results = []
        
        for agent_id, metadata in self._metadata.items():
            if capability and capability not in metadata.capabilities:
                continue
            if category and metadata.category.value != category:
                continue
            
            agent_class = self._agents.get(agent_id)
            if agent_class:
                results.append(agent_class)
        
        return results
    
    def get_agents_by_protocol(self, protocol: str) -> List[Type[ProtocolAwareAgent]]:
        """Get agents that support a specific protocol"""
        results = []
        
        for agent_id, metadata in self._metadata.items():
            if protocol in metadata.primary_protocols or protocol in metadata.secondary_protocols:
                agent_class = self._agents.get(agent_id)
                if agent_class:
                    results.append(agent_class)
        
        return results
    
    def validate_agents(self) -> Dict[str, bool]:
        """Validate all registered agents"""
        validation_results = {}
        
        for agent_id, agent_class in self._agents.items():
            try:
                # Basic validation - check required attributes
                required_attrs = ['AGENT_ID', 'AGENT_VERSION', 'CAPABILITIES']
                for attr in required_attrs:
                    if not hasattr(agent_class, attr):
                        self.logger.error(f"Agent {agent_id} missing required attribute: {attr}")
                        validation_results[agent_id] = False
                        break
                else:
                    # All required attributes present
                    validation_results[agent_id] = True
                    self.logger.debug(f"Agent {agent_id} validation passed")
                    
            except Exception as e:
                self.logger.error(f"Agent {agent_id} validation failed: {e}")
                validation_results[agent_id] = False
        
        return validation_results


# Global registry instances
_global_registry = InMemoryAgentRegistry()
_system_registry = InMemoryAgentRegistry()
_autonomous_engine_registry = InMemoryAgentRegistry()


def get_global_agent_registry() -> InMemoryAgentRegistry:
    """Get the global agent registry"""
    return _global_registry


def get_system_registry() -> InMemoryAgentRegistry:
    """Get the system agent registry"""
    return _system_registry


def get_autonomous_engine_registry() -> InMemoryAgentRegistry:
    """Get the autonomous engine agent registry"""
    return _autonomous_engine_registry


def reset_global_registry():
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = InMemoryAgentRegistry()
    logger.info("Global registry reset") 