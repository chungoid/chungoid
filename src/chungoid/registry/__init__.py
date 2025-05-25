"""
Registry-First Architecture Module

This module provides the registry-first architecture components that replace
the fragmented fallback map system with a unified agent registration approach.
"""

from .in_memory_agent_registry import (
    InMemoryAgentRegistry,
    AgentMetadata,
    get_global_agent_registry,
    reset_global_registry
)

from .decorators import (
    register_agent,
    register_system_agent,
    register_autonomous_engine_agent,
    register_test_agent,
    register_validated_agent,
    validate_agent_interface
)

__all__ = [
    # Registry classes
    'InMemoryAgentRegistry',
    'AgentMetadata',
    
    # Registry functions
    'get_global_agent_registry',
    'reset_global_registry',
    
    # Decorators
    'register_agent',
    'register_system_agent', 
    'register_autonomous_engine_agent',
    'register_test_agent',
    'register_validated_agent',
    'validate_agent_interface'
] 