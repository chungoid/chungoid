"""
Enhanced Agent Resolver for Task-Type Based Autonomous Orchestration

This module implements capability-based agent resolution that maps task types
to appropriate autonomous agents while preserving all agent specializations
and ensuring no agents are left behind.

Following the "rip the bandaid off" approach with pure autonomous architecture.
"""

from __future__ import annotations

import logging
import inspect
import asyncio
from typing import Dict, List, Optional, Any, Callable, Type, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from chungoid.schemas.master_flow import EnhancedMasterStageSpec
from chungoid.registry.in_memory_agent_registry import InMemoryAgentRegistry, AgentMetadata
from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.utils.agent_registry import AgentRegistry
from chungoid.schemas.agent_outputs import AgentOutput

logger = logging.getLogger(__name__)


class ResolutionMethod(Enum):
    """Methods used to resolve agents."""
    CAPABILITY_MATCH = "capability_match"
    TASK_TYPE_MAPPING = "task_type_mapping"
    AUTONOMOUS_PREFERENCE = "autonomous_preference"
    CONCRETE_FALLBACK = "concrete_fallback"
    DIRECT_AGENT_ID = "direct_agent_id"
    FALLBACK_AGENT_ID = "fallback_agent_id"


class ExecutionMode(Enum):
    """Execution modes for resolved agents."""
    AUTONOMOUS = "autonomous"
    CONCRETE = "concrete"
    FALLBACK = "fallback"
    DIRECT = "direct"


@dataclass
class AgentResolutionResult:
    """Result of agent resolution attempt"""
    def __init__(self, 
                 agent_id: str,
                 success: bool,
                 agent_instance: Optional[ProtocolAwareAgent] = None,
                 error: Optional[str] = None,
                 capabilities: Optional[List[str]] = None,
                 execution_mode: Optional[ExecutionMode] = None,
                 resolution_method: Optional[ResolutionMethod] = None,
                 confidence_score: float = 0.0,
                 fallback_reason: Optional[str] = None,
                 capabilities_matched: Optional[List[str]] = None):
        self.agent_id = agent_id
        self.success = success
        self.agent_instance = agent_instance
        self.error = error
        self.capabilities = capabilities or []
        self.execution_mode = execution_mode or ExecutionMode.DIRECT
        self.resolution_method = resolution_method or ResolutionMethod.DIRECT_AGENT_ID
        self.confidence_score = confidence_score
        self.fallback_reason = fallback_reason
        self.capabilities_matched = capabilities_matched or []


class AgentResolutionError(Exception):
    """Raised when agent resolution fails."""
    
    def __init__(self, message: str, task_type: Optional[str] = None, 
                 required_capabilities: Optional[List[str]] = None):
        super().__init__(message)
        self.task_type = task_type
        self.required_capabilities = required_capabilities


class EnhancedAgentResolver:
    """Enhanced agent resolver for protocol-aware agents"""
    
    def __init__(self, 
                 agent_registry: AgentRegistry,
                 llm_provider: LLMProvider,
                 prompt_manager: PromptManager):
        self.agent_registry = agent_registry
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize task type mappings and autonomous capabilities cache
        self._task_type_mappings = self._initialize_task_type_mappings()
        self._autonomous_capabilities_cache = {}
        self._refresh_autonomous_capabilities_cache()
        self.logger.info("Initialized EnhancedAgentResolver with task type mappings and autonomous capabilities cache")
        
    async def resolve_agent(self, agent_id: str, context: Optional[Dict[str, Any]] = None) -> AgentResolutionResult:
        """Resolve and instantiate a protocol-aware agent"""
        try:
            # Get agent class from registry
            agent_class = self.agent_registry.get_agent(agent_id)
            if not agent_class:
                return AgentResolutionResult(
                    agent_id=agent_id,
                    success=False,
                    error=f"Agent {agent_id} not found in registry"
                )
            
            # Verify it's a protocol-aware agent
            if not self._is_protocol_aware_agent(agent_class):
                return AgentResolutionResult(
                    agent_id=agent_id,
                    success=False,
                    error=f"Agent {agent_id} is not a ProtocolAwareAgent"
                )
            
            # Extract capabilities
            capabilities = self._extract_agent_capabilities(agent_class)
            
            # Instantiate agent
            agent_instance = await self._instantiate_agent(agent_class, context)
            if not agent_instance:
                return AgentResolutionResult(
                    agent_id=agent_id,
                    success=False,
                    error=f"Failed to instantiate agent {agent_id}",
                    capabilities=capabilities
                )
            
            return AgentResolutionResult(
                agent_id=agent_id,
                success=True,
                agent_instance=agent_instance,
                capabilities=capabilities
            )
            
        except Exception as e:
            self.logger.error(f"Error resolving agent {agent_id}: {e}")
            return AgentResolutionResult(
                agent_id=agent_id,
                success=False,
                error=str(e)
            )
    
    def _is_protocol_aware_agent(self, agent_class: Type[ProtocolAwareAgent]) -> bool:
        """Check if agent class is protocol-aware"""
        try:
            return (
                inspect.isclass(agent_class) and
                issubclass(agent_class, ProtocolAwareAgent) and
                hasattr(agent_class, 'PRIMARY_PROTOCOLS') and
                hasattr(agent_class, 'SECONDARY_PROTOCOLS') and
                hasattr(agent_class, 'CAPABILITIES') and
                len(agent_class.PRIMARY_PROTOCOLS) > 0
            )
        except Exception:
            return False
    
    def _extract_agent_capabilities(self, agent_class: Type[ProtocolAwareAgent]) -> List[str]:
        """Extract capabilities from protocol-aware agent"""
        try:
            if hasattr(agent_class, 'CAPABILITIES'):
                return list(agent_class.CAPABILITIES)
            elif hasattr(agent_class, 'PRIMARY_PROTOCOLS'):
                return list(agent_class.PRIMARY_PROTOCOLS)
            else:
                return []
        except Exception:
            return []
    
    async def _instantiate_agent(self, agent_class: Type[ProtocolAwareAgent], context: Optional[Dict[str, Any]] = None) -> Optional[ProtocolAwareAgent]:
        """Instantiate a protocol-aware agent"""
        try:
            # Standard initialization for ProtocolAwareAgent
            agent_instance = agent_class(
                llm_provider=self.llm_provider,
                prompt_manager=self.prompt_manager,
                system_context=context
            )
            
            self.logger.info(f"Successfully instantiated {agent_class.__name__}")
            return agent_instance
            
        except Exception as e:
            self.logger.error(f"Failed to instantiate {agent_class.__name__}: {e}")
            return None

    def _initialize_task_type_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize task-type to capability mappings based on autonomous agents."""
        return {
            "requirements_analysis": {
                "primary_capabilities": ["requirements_analysis", "stakeholder_analysis"],
                "secondary_capabilities": ["documentation", "analysis"],
                "preferred_agents": ["ProductAnalystAgent_v1"],
                "fallback_agents": ["SystemRequirementsGatheringAgent_v1"]
            },
            "architecture_design": {
                "primary_capabilities": ["architecture_design", "system_planning", "blueprint_generation"],
                "secondary_capabilities": ["design", "planning"],
                "preferred_agents": ["ArchitectAgent_v1"],
                "fallback_agents": []
            },
            "environment_setup": {
                "primary_capabilities": ["environment_setup", "dependency_management", "project_bootstrapping"],
                "secondary_capabilities": ["setup", "configuration"],
                "preferred_agents": ["EnvironmentBootstrapAgent"],
                "fallback_agents": []
            },
            "dependency_management": {
                "primary_capabilities": ["dependency_analysis", "package_management", "conflict_resolution"],
                "secondary_capabilities": ["dependencies", "packages"],
                "preferred_agents": ["DependencyManagementAgent_v1"],
                "fallback_agents": []
            },
            "code_generation": {
                "primary_capabilities": ["code_generation", "implementation_planning"],
                "secondary_capabilities": ["coding", "implementation"],
                "preferred_agents": ["SmartCodeGeneratorAgent_v1", "CoreCodeGeneratorAgent_v1"],
                "fallback_agents": []
            },
            "code_debugging": {
                "primary_capabilities": ["code_debugging", "error_analysis", "automated_fixes"],
                "secondary_capabilities": ["debugging", "error_handling"],
                "preferred_agents": ["CodeDebuggingAgent_v1"],
                "fallback_agents": []
            },
            "testing": {
                "primary_capabilities": ["test_generation", "test_execution", "validation"],
                "secondary_capabilities": ["testing", "quality_assurance"],
                "preferred_agents": [],
                "fallback_agents": []
            },
            "quality_validation": {
                "primary_capabilities": ["review_protocol", "quality_validation", "architectural_review"],
                "secondary_capabilities": ["review", "validation", "quality"],
                "preferred_agents": ["BlueprintReviewerAgent_v1"],
                "fallback_agents": []
            },
            "risk_assessment": {
                "primary_capabilities": ["risk_assessment", "deep_investigation", "impact_analysis"],
                "secondary_capabilities": ["risk", "analysis"],
                "preferred_agents": ["ProactiveRiskAssessorAgent_v1"],
                "fallback_agents": []
            },
            "documentation": {
                "primary_capabilities": ["documentation_generation", "project_analysis", "comprehensive_reporting"],
                "secondary_capabilities": ["documentation", "reporting"],
                "preferred_agents": ["ProjectDocumentationAgent_v1"],
                "fallback_agents": []
            },
            "file_operations": {
                "primary_capabilities": ["file_operations", "directory_management"],
                "secondary_capabilities": ["filesystem", "files"],
                "preferred_agents": [],
                "fallback_agents": ["SystemFileSystemAgent_v1"]
            },
            "project_coordination": {
                "primary_capabilities": ["autonomous_coordination", "quality_gates", "refinement_orchestration"],
                "secondary_capabilities": ["coordination", "orchestration"],
                "preferred_agents": ["AutomatedRefinementCoordinatorAgent_v1"],
                "fallback_agents": []
            },
            "requirements_traceability": {
                "primary_capabilities": ["requirements_traceability", "artifact_analysis", "quality_validation"],
                "secondary_capabilities": ["traceability", "tracking"],
                "preferred_agents": ["RequirementsTracerAgent_v1"],
                "fallback_agents": []
            }
        }
    
    def _refresh_autonomous_capabilities_cache(self):
        """Refresh the cache of autonomous agent capabilities."""
        self._autonomous_capabilities_cache.clear()
        
        for agent_id, agent_class in self.agent_registry.list_agents().items():
            if self._is_autonomous_capable_class(agent_class):
                capabilities = self._extract_agent_capabilities(agent_class)
                self._autonomous_capabilities_cache[agent_id] = {
                    "capabilities": capabilities,
                    "agent_class": agent_class
                }
        
        self.logger.info(f"Refreshed autonomous capabilities cache: {len(self._autonomous_capabilities_cache)} autonomous agents")
    
    def _is_autonomous_capable_class(self, agent_class: Type[ProtocolAwareAgent]) -> bool:
        """Check if an agent class is autonomous-capable."""
        # Check if it's a ProtocolAwareAgent with protocols
        if not issubclass(agent_class, ProtocolAwareAgent):
            return False
        
        # Check for protocol definitions
        has_protocols = (
            hasattr(agent_class, 'PRIMARY_PROTOCOLS') and 
            getattr(agent_class, 'PRIMARY_PROTOCOLS', [])
        )
        
        return has_protocols
    
    def _matches_capabilities(self, agent_capabilities: List[str], required_capabilities: List[str]) -> bool:
        """Check if agent capabilities match required capabilities."""
        if not required_capabilities:
            return True
        
        # Check for exact matches first
        matched = set(agent_capabilities) & set(required_capabilities)
        if len(matched) >= len(required_capabilities) * 0.7:  # 70% match threshold
            return True
        
        # Check for partial matches with synonyms
        for required in required_capabilities:
            if any(self._are_capability_synonyms(required, agent_cap) for agent_cap in agent_capabilities):
                return True
        
        return False
    
    def _are_capability_synonyms(self, cap1: str, cap2: str) -> bool:
        """Check if two capabilities are synonyms."""
        synonyms = {
            "requirements_analysis": ["analysis", "requirements", "stakeholder_analysis"],
            "architecture_design": ["design", "architecture", "system_planning"],
            "code_generation": ["coding", "implementation", "generation"],
            "quality_validation": ["validation", "review", "quality"],
            "file_operations": ["filesystem", "files", "directory_management"]
        }
        
        for base_cap, synonym_list in synonyms.items():
            if (cap1 == base_cap and cap2 in synonym_list) or (cap2 == base_cap and cap1 in synonym_list):
                return True
        
        return False
    
    def get_task_type_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Get the current task-type mappings."""
        return self._task_type_mappings.copy()
    
    def get_autonomous_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get the current autonomous agents cache."""
        return self._autonomous_capabilities_cache.copy()
    
    def add_task_type_mapping(self, task_type: str, mapping: Dict[str, Any]):
        """Add or update a task-type mapping."""
        self._task_type_mappings[task_type] = mapping
        self.logger.info(f"Added task-type mapping for '{task_type}'")
    
    def refresh_cache(self):
        """Refresh all internal caches."""
        self._refresh_autonomous_capabilities_cache()
        self.logger.info("Refreshed EnhancedAgentResolver caches")
    
    async def resolve_agent_for_stage(self, stage_spec) -> AgentResolutionResult:
        """Resolve agent for a specific stage using task-type orchestration"""
        try:
            # Initialize task type mappings and autonomous capabilities cache
            if not hasattr(self, '_task_type_mappings'):
                self._task_type_mappings = self._initialize_task_type_mappings()
            if not hasattr(self, '_autonomous_capabilities_cache'):
                self._autonomous_capabilities_cache = {}
                self._refresh_autonomous_capabilities_cache()
            
            task_type = getattr(stage_spec, 'task_type', None)
            required_capabilities = getattr(stage_spec, 'required_capabilities', [])
            preferred_execution = getattr(stage_spec, 'preferred_execution', 'any')
            fallback_agent_id = getattr(stage_spec, 'fallback_agent_id', None)
            
            self.logger.info(f"Resolving agent for stage: task_type={task_type}, capabilities={required_capabilities}, execution={preferred_execution}")
            
            # Method 1: Try task-type mapping first
            if task_type and task_type in self._task_type_mappings:
                mapping = self._task_type_mappings[task_type]
                preferred_agents = mapping.get('preferred_agents', [])
                
                for agent_id in preferred_agents:
                    agent_class = self.agent_registry.get_agent(agent_id)
                    if agent_class and self._is_protocol_aware_agent(agent_class):
                        agent_capabilities = self._extract_agent_capabilities(agent_class)
                        if self._matches_capabilities(agent_capabilities, required_capabilities):
                            agent_instance = await self._instantiate_agent(agent_class)
                            if agent_instance:
                                return AgentResolutionResult(
                                    agent_id=agent_id,
                                    success=True,
                                    agent_instance=agent_instance,
                                    capabilities=agent_capabilities,
                                    execution_mode=ExecutionMode.AUTONOMOUS if self._is_autonomous_capable_class(agent_class) else ExecutionMode.CONCRETE,
                                    resolution_method=ResolutionMethod.TASK_TYPE_MAPPING,
                                    confidence_score=0.9,
                                    capabilities_matched=required_capabilities
                                )
            
            # Method 2: Try capability-based matching
            if required_capabilities:
                for agent_id, agent_info in self._autonomous_capabilities_cache.items():
                    agent_capabilities = agent_info['capabilities']
                    if self._matches_capabilities(agent_capabilities, required_capabilities):
                        agent_class = agent_info['agent_class']
                        agent_instance = await self._instantiate_agent(agent_class)
                        if agent_instance:
                            return AgentResolutionResult(
                                agent_id=agent_id,
                                success=True,
                                agent_instance=agent_instance,
                                capabilities=agent_capabilities,
                                execution_mode=ExecutionMode.AUTONOMOUS,
                                resolution_method=ResolutionMethod.CAPABILITY_MATCH,
                                confidence_score=0.8,
                                capabilities_matched=required_capabilities
                            )
            
            # Method 3: Try fallback agent if specified
            if fallback_agent_id:
                agent_class = self.agent_registry.get_agent(fallback_agent_id)
                if agent_class:
                    agent_instance = await self._instantiate_agent(agent_class)
                    if agent_instance:
                        agent_capabilities = self._extract_agent_capabilities(agent_class)
                        return AgentResolutionResult(
                            agent_id=fallback_agent_id,
                            success=True,
                            agent_instance=agent_instance,
                            capabilities=agent_capabilities,
                            execution_mode=ExecutionMode.FALLBACK,
                            resolution_method=ResolutionMethod.FALLBACK_AGENT_ID,
                            confidence_score=0.6,
                            capabilities_matched=[]
                        )
            
            # Method 4: Try task-type fallback agents
            if task_type and task_type in self._task_type_mappings:
                mapping = self._task_type_mappings[task_type]
                fallback_agents = mapping.get('fallback_agents', [])
                
                for agent_id in fallback_agents:
                    agent_class = self.agent_registry.get_agent(agent_id)
                    if agent_class:
                        agent_instance = await self._instantiate_agent(agent_class)
                        if agent_instance:
                            agent_capabilities = self._extract_agent_capabilities(agent_class)
                            return AgentResolutionResult(
                                agent_id=agent_id,
                                success=True,
                                agent_instance=agent_instance,
                                capabilities=agent_capabilities,
                                execution_mode=ExecutionMode.CONCRETE,
                                resolution_method=ResolutionMethod.CONCRETE_FALLBACK,
                                confidence_score=0.5,
                                capabilities_matched=[]
                            )
            
            # No suitable agent found
            return AgentResolutionResult(
                agent_id="",
                success=False,
                error=f"No suitable agent found for task_type='{task_type}' with capabilities={required_capabilities}",
                fallback_reason=f"Task type '{task_type}' not mapped and no matching capabilities found"
            )
            
        except Exception as e:
            self.logger.error(f"Error resolving agent for stage: {e}")
            return AgentResolutionResult(
                agent_id="",
                success=False,
                error=str(e),
                fallback_reason=f"Exception during resolution: {str(e)}"
            ) 