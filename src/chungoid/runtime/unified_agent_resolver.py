"""
Unified Agent Resolver for UAEI Architecture

Simple, single-path agent resolution that eliminates ALL technical debt
from complex resolver patterns. Part of Phase 3 enhanced cycle implementation.

Replaces: EnhancedAgentResolver, RegistryAgentProvider, and all fallback logic.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from ..agents.unified_agent import UnifiedAgent
from ..utils.llm_provider import LLMProvider
from ..utils.prompt_manager import PromptManager
from ..utils.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


class UnifiedAgentResolver:
    """
    Simple, unified agent resolver that eliminates technical debt.
    
    Phase 3 UAEI Implementation:
    - Single resolution path (no fallbacks)
    - No complex capability matching 
    - No branching logic
    - Direct agent instantiation
    - Zero technical debt
    """
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        llm_provider: LLMProvider,
        prompt_manager: PromptManager
    ):
        self.agent_registry = agent_registry
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("UnifiedAgentResolver initialized - Phase 3 UAEI architecture")
    
    async def resolve_agent(self, agent_id: str) -> UnifiedAgent:
        """
        Single-path agent resolution - no fallbacks, no complexity.
        
        Args:
            agent_id: The ID of the agent to resolve
            
        Returns:
            UnifiedAgent instance ready for execute() calls
            
        Raises:
            AgentResolutionError: If agent cannot be resolved
        """
        try:
            # 1. Get agent class from registry (single path)
            agent_class = self.agent_registry.get_agent(agent_id)
            if not agent_class:
                raise AgentResolutionError(f"Agent {agent_id} not found in registry")
            
            # 2. Verify it's a UnifiedAgent (no complex checks)
            if not issubclass(agent_class, UnifiedAgent):
                raise AgentResolutionError(
                    f"Agent {agent_id} is not a UnifiedAgent. "
                    f"All agents must inherit from UnifiedAgent in Phase 3."
                )
            
            # 3. Instantiate agent (simple, direct)
            agent_instance = agent_class(
                llm_provider=self.llm_provider,
                prompt_manager=self.prompt_manager
            )
            
            self.logger.info(f"Successfully resolved UnifiedAgent: {agent_id}")
            return agent_instance
            
        except Exception as e:
            error_msg = f"Failed to resolve agent {agent_id}: {str(e)}"
            self.logger.error(error_msg)
            raise AgentResolutionError(error_msg) from e


class AgentResolutionError(Exception):
    """Raised when unified agent resolution fails."""
    pass 