"""
Auto-Registration Decorators for Registry-First Architecture

This module provides decorators that automatically register agents in the global registry
when they are imported, eliminating the need for manual registration and fallback maps.

Example usage:
    @register_system_agent(capabilities=["file_operations", "system_management"])
    class EnvironmentBootstrapAgent(ProtocolAwareAgent):
        AGENT_ID = "EnvironmentBootstrapAgent"
        # ... rest of agent implementation
"""

from __future__ import annotations

import logging
import inspect
from typing import Type, List, Optional, Dict, Any, Callable, Union
from functools import wraps

from chungoid.agents.unified_agent import UnifiedAgent
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from .in_memory_agent_registry import get_global_agent_registry, AgentMetadata

logger = logging.getLogger(__name__)


def register_agent(
    category: str = "system", 
    capabilities: Optional[List[str]] = None,
    priority: int = 0,
    visibility: AgentVisibility = AgentVisibility.PUBLIC
):
    """
    Decorator to automatically register agents in the global registry.
    
    This decorator should be applied to all agent classes to ensure they are
    automatically registered when imported, eliminating the need for manual
    registration and fallback maps.
    
    Args:
        category: Agent category (e.g., "system", "autonomous_engine")
        capabilities: List of capabilities this agent provides
        priority: Priority for agent selection (higher wins)
        visibility: Agent visibility level
    
    Example:
        @register_agent(category="autonomous_engine", capabilities=["environment_setup"])
        class EnvironmentBootstrapAgent(ProtocolAwareAgent):
            AGENT_ID: ClassVar[str] = "EnvironmentBootstrapAgent"
            # ... rest of implementation
    """
    def decorator(agent_class: Type[UnifiedAgent]):
        # Validate agent class
        if not hasattr(agent_class, 'AGENT_ID'):
            raise ValueError(f"Agent class {agent_class.__name__} must have AGENT_ID attribute")
        
        # Phase 3 UAEI: Check for unified execute method, NOT legacy invoke_async
        if not hasattr(agent_class, 'execute'):
            raise ValueError(f"Agent class {agent_class.__name__} must implement UAEI execute() method")
        
        # Ensure no legacy methods remain
        legacy_methods = ['invoke_async', 'execute_with_protocol', 'execute_with_protocols']
        for method in legacy_methods:
            if hasattr(agent_class, method):
                logger.warning(f"Agent {agent_class.__name__} contains legacy method {method} - should be removed for UAEI compliance")
        
        # Create metadata
        metadata = AgentMetadata(
            agent_id=agent_class.AGENT_ID,
            name=getattr(agent_class, 'AGENT_NAME', agent_class.__name__),
            description=getattr(agent_class, 'AGENT_DESCRIPTION', ''),
            version=agent_class.AGENT_VERSION,
            category=AgentCategory.SYSTEM_ORCHESTRATION if category == "system" else AgentCategory.AUTONOMOUS_COORDINATION,
            visibility=visibility,
            capabilities=capabilities or [],
            primary_protocols=list(getattr(agent_class, 'PRIMARY_PROTOCOLS', [])),
            secondary_protocols=list(getattr(agent_class, 'SECONDARY_PROTOCOLS', [])),
            priority=priority
        )
        
        # Register in global registry
        registry = get_global_agent_registry()
        registry.register_agent(agent_class, metadata)
        
        logger.debug(f"Auto-registered agent: {agent_class.AGENT_ID} ({agent_class.__name__})")
        
        return agent_class
    
    return decorator


def register_system_agent(capabilities: List[str]):
    """Register a system agent with specified capabilities"""
    def decorator(agent_class: Type[UnifiedAgent]):
        # Validate agent interface
        validate_agent_interface(agent_class)
        
        # Set capabilities if not already set
        if not hasattr(agent_class, 'CAPABILITIES'):
            agent_class.CAPABILITIES = capabilities
        
        # Register with global registry (not separate system registry)
        registry = get_global_agent_registry()
        
        # Create metadata for the agent
        metadata = AgentMetadata(
            agent_id=agent_class.AGENT_ID,
            name=getattr(agent_class, 'AGENT_NAME', agent_class.__name__),
            description=getattr(agent_class, 'DESCRIPTION', ''),
            version=agent_class.AGENT_VERSION,
            category=AgentCategory.SYSTEM_ORCHESTRATION,
            visibility=AgentVisibility.PUBLIC,
            capabilities=capabilities,
            primary_protocols=list(getattr(agent_class, 'PRIMARY_PROTOCOLS', [])),
            secondary_protocols=list(getattr(agent_class, 'SECONDARY_PROTOCOLS', []))
        )
        
        registry.register_agent(agent_class, metadata)
        
        logger.info(f"Registered system agent: {agent_class.__name__}")
        return agent_class
    
    return decorator


def register_autonomous_engine_agent(capabilities: List[str]):
    """Register an autonomous engine agent with specified capabilities"""
    def decorator(agent_class: Type[UnifiedAgent]):
        # Validate agent interface
        validate_agent_interface(agent_class)
        
        # Set capabilities if not already set
        if not hasattr(agent_class, 'CAPABILITIES'):
            agent_class.CAPABILITIES = capabilities
        
        # Register with global registry (not separate autonomous engine registry)
        registry = get_global_agent_registry()
        
        # Create metadata for the agent
        metadata = AgentMetadata(
            agent_id=agent_class.AGENT_ID,
            name=getattr(agent_class, 'AGENT_NAME', agent_class.__name__),
            description=getattr(agent_class, 'DESCRIPTION', ''),
            version=agent_class.AGENT_VERSION,
            category=AgentCategory.AUTONOMOUS_COORDINATION,
            visibility=AgentVisibility.PUBLIC,
            capabilities=capabilities,
            primary_protocols=list(getattr(agent_class, 'PRIMARY_PROTOCOLS', [])),
            secondary_protocols=list(getattr(agent_class, 'SECONDARY_PROTOCOLS', []))
        )
        
        registry.register_agent(agent_class, metadata)
        
        logger.info(f"Registered autonomous engine agent: {agent_class.__name__}")
        return agent_class
    
    return decorator


def register_test_agent(capabilities: Optional[List[str]] = None):
    """
    Convenience decorator for test/mock agents.
    
    Equivalent to @register_agent(category="testing", visibility=INTERNAL, ...)
    """
    return register_agent(
        category="testing",
        capabilities=capabilities,
        priority=0,
        visibility=AgentVisibility.INTERNAL
    )


def validate_agent_interface(agent_class: Type[UnifiedAgent]):
    """Validate that agent class conforms to Phase 3 UAEI interface"""
    required_class_vars = [
        'AGENT_ID', 'AGENT_VERSION', 'PRIMARY_PROTOCOLS', 
        'CAPABILITIES'
    ]
    
    for var_name in required_class_vars:
        if not hasattr(agent_class, var_name):
            raise ValueError(f"Agent {agent_class.__name__} missing required class variable: {var_name}")
    
    # Validate protocols are not empty
    if not agent_class.PRIMARY_PROTOCOLS:
        raise ValueError(f"Agent {agent_class.__name__} must have at least one PRIMARY_PROTOCOL")
    
    # UAEI: Only support unified interface (no legacy)
    if not hasattr(agent_class, 'execute'):
        raise ValueError(f"Agent {agent_class.__name__} must implement UAEI execute() method")
    
    # Ensure no legacy methods remain
    legacy_methods = ['invoke_async', 'execute_with_protocol', 'execute_with_protocols']
    for method in legacy_methods:
        if hasattr(agent_class, method):
            raise ValueError(f"Agent {agent_class.__name__} contains legacy method {method} - must be removed for UAEI compliance")
    
    logger.debug(f"Agent {agent_class.__name__} passed UAEI interface validation")
    
    logger.debug(f"Agent {agent_class.__name__} passed interface validation")


# Combined decorator for full validation and registration
def register_validated_agent(
    category: str = "system",
    capabilities: Optional[List[str]] = None,
    priority: int = 0,
    visibility: AgentVisibility = AgentVisibility.PUBLIC
):
    """
    Combined decorator that validates agent interface and registers it.
    
    This is the recommended decorator for new agents as it ensures both
    interface compliance and automatic registration.
    """
    def decorator(agent_class: Type[UnifiedAgent]):
        # First validate interface
        validate_agent_interface(agent_class)
        
        # Then register
        registered_class = register_agent(
            category=category,
            capabilities=capabilities,
            priority=priority,
            visibility=visibility
        )(agent_class)
        
        return registered_class
    
    return decorator 