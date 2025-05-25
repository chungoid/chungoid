"""
Auto-Registration Decorators for Registry-First Architecture

This module provides decorators that automatically register agents in the global registry
when they are imported, eliminating the need for manual registration and fallback maps.
"""

import logging
from typing import List, Type, Optional
from functools import wraps

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.agent_registry_meta import AgentVisibility
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
        class EnvironmentBootstrapAgent(BaseAgent):
            AGENT_ID: ClassVar[str] = "EnvironmentBootstrapAgent"
            # ... rest of implementation
    """
    def decorator(agent_class: Type[BaseAgent]):
        # Validate agent class
        if not hasattr(agent_class, 'AGENT_ID'):
            raise ValueError(f"Agent class {agent_class.__name__} must have AGENT_ID attribute")
        
        if not hasattr(agent_class, 'invoke_async'):
            logger.warning(f"Agent class {agent_class.__name__} missing invoke_async method")
        
        # Create metadata
        metadata = AgentMetadata(
            category=category,
            capabilities=capabilities or [],
            module=agent_class.__module__,
            class_name=agent_class.__name__,
            priority=priority,
            visibility=visibility
        )
        
        # Register in global registry
        registry = get_global_agent_registry()
        registry.register_agent(agent_class, metadata)
        
        logger.debug(f"Auto-registered agent: {agent_class.AGENT_ID} ({agent_class.__name__})")
        
        return agent_class
    
    return decorator


def register_system_agent(capabilities: Optional[List[str]] = None, priority: int = 0):
    """
    Convenience decorator for system agents.
    
    Equivalent to @register_agent(category="system", ...)
    """
    return register_agent(
        category="system",
        capabilities=capabilities,
        priority=priority,
        visibility=AgentVisibility.PUBLIC
    )


def register_autonomous_engine_agent(capabilities: Optional[List[str]] = None, priority: int = 0):
    """
    Convenience decorator for autonomous engine agents.
    
    Equivalent to @register_agent(category="autonomous_engine", ...)
    """
    return register_agent(
        category="autonomous_engine", 
        capabilities=capabilities,
        priority=priority,
        visibility=AgentVisibility.PUBLIC
    )


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


# Validation decorator for ensuring agent compliance
def validate_agent_interface(agent_class: Type[BaseAgent]):
    """
    Decorator to validate agent interface compliance.
    
    This can be used in combination with @register_agent to ensure
    agents meet the required interface standards.
    """
    # Check required attributes
    required_attrs = ['AGENT_ID', 'invoke_async']
    missing_attrs = [attr for attr in required_attrs if not hasattr(agent_class, attr)]
    
    if missing_attrs:
        raise ValueError(f"Agent {agent_class.__name__} missing required attributes: {missing_attrs}")
    
    # Check AGENT_ID is string
    if not isinstance(agent_class.AGENT_ID, str):
        raise ValueError(f"Agent {agent_class.__name__} AGENT_ID must be a string")
    
    # Check AGENT_ID is not empty
    if not agent_class.AGENT_ID.strip():
        raise ValueError(f"Agent {agent_class.__name__} AGENT_ID cannot be empty")
    
    logger.debug(f"Agent interface validation passed: {agent_class.__name__}")
    return agent_class


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
    def decorator(agent_class: Type[BaseAgent]):
        # First validate interface
        validated_class = validate_agent_interface(agent_class)
        
        # Then register
        registered_class = register_agent(
            category=category,
            capabilities=capabilities,
            priority=priority,
            visibility=visibility
        )(validated_class)
        
        return registered_class
    
    return decorator 