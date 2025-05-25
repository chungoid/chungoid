"""
Enhanced Agent Resolver for Task-Type Based Autonomous Orchestration

This module implements capability-based agent resolution that maps task types
to appropriate autonomous agents while preserving all agent specializations
and ensuring no agents are left behind.

Following the "rip the bandaid off" approach with pure autonomous architecture.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Type, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from chungoid.schemas.master_flow import EnhancedMasterStageSpec
from chungoid.registry.in_memory_agent_registry import InMemoryAgentRegistry, AgentMetadata
from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
from chungoid.utils.agent_resolver import AgentProvider

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
    """Result of agent resolution with metadata."""
    agent_instance: Optional[BaseAgent]
    agent_id: str
    execution_mode: ExecutionMode
    resolution_method: ResolutionMethod
    capabilities_matched: List[str]
    task_type: Optional[str] = None
    confidence_score: float = 1.0
    fallback_reason: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Whether resolution was successful."""
        return self.agent_instance is not None


class AgentResolutionError(Exception):
    """Raised when agent resolution fails."""
    
    def __init__(self, message: str, task_type: Optional[str] = None, 
                 required_capabilities: Optional[List[str]] = None):
        super().__init__(message)
        self.task_type = task_type
        self.required_capabilities = required_capabilities


class EnhancedAgentResolver:
    """
    Enhanced agent resolver that implements capability-based resolution
    for task-type orchestration while ensuring no agents are left behind.
    """
    
    def __init__(self, agent_registry: InMemoryAgentRegistry, agent_provider: AgentProvider):
        self.agent_registry = agent_registry
        self.agent_provider = agent_provider
        self.logger = logging.getLogger(__name__)
        
        # Initialize task-type to capability mappings
        self._task_type_mappings = self._initialize_task_type_mappings()
        
        # Initialize autonomous agent capabilities cache
        self._autonomous_capabilities_cache = {}
        self._refresh_autonomous_capabilities_cache()
        
        self.logger.info("EnhancedAgentResolver initialized with task-type orchestration")
    
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
    
    def _is_autonomous_capable_class(self, agent_class: Type[BaseAgent]) -> bool:
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
    
    def _extract_agent_capabilities(self, agent_class: Type[BaseAgent]) -> List[str]:
        """Extract capabilities from an agent class."""
        capabilities = []
        
        # Extract from PRIMARY_PROTOCOLS
        if hasattr(agent_class, 'PRIMARY_PROTOCOLS'):
            capabilities.extend(getattr(agent_class, 'PRIMARY_PROTOCOLS', []))
        
        # Extract from SECONDARY_PROTOCOLS
        if hasattr(agent_class, 'SECONDARY_PROTOCOLS'):
            capabilities.extend(getattr(agent_class, 'SECONDARY_PROTOCOLS', []))
        
        # Extract from metadata if available
        agent_id = getattr(agent_class, 'AGENT_ID', None)
        if agent_id:
            metadata = self.agent_registry.get_agent_metadata(agent_id)
            if metadata and metadata.capabilities:
                capabilities.extend(metadata.capabilities)
        
        return list(set(capabilities))  # Remove duplicates
    
    async def resolve_agent_for_stage(self, stage_spec: EnhancedMasterStageSpec) -> AgentResolutionResult:
        """
        Resolve the best agent for a stage based on task type and capabilities.
        Ensures no agents are left behind with comprehensive fallback mechanisms.
        """
        self.logger.info(f"Resolving agent for stage '{stage_spec.id}' with task_type='{stage_spec.task_type}'")
        
        # Strategy 1: Try autonomous agents first (if preferred)
        if stage_spec.preferred_execution in ["autonomous", "any"]:
            autonomous_result = await self._resolve_autonomous_agent(stage_spec)
            if autonomous_result.success:
                self.logger.info(f"Resolved autonomous agent '{autonomous_result.agent_id}' for task_type '{stage_spec.task_type}'")
                return autonomous_result
        
        # Strategy 2: Try concrete agents with invoke_async
        if stage_spec.preferred_execution in ["concrete", "any"]:
            concrete_result = await self._resolve_concrete_agent(stage_spec)
            if concrete_result.success:
                self.logger.info(f"Resolved concrete agent '{concrete_result.agent_id}' for task_type '{stage_spec.task_type}'")
                return concrete_result
        
        # Strategy 3: Try direct agent_id if provided
        if stage_spec.agent_id:
            direct_result = await self._resolve_direct_agent(stage_spec.agent_id, stage_spec)
            if direct_result.success:
                self.logger.info(f"Resolved direct agent '{direct_result.agent_id}' for stage '{stage_spec.id}'")
                return direct_result
        
        # Strategy 4: Try fallback_agent_id if provided
        if stage_spec.fallback_agent_id:
            fallback_result = await self._resolve_direct_agent(stage_spec.fallback_agent_id, stage_spec)
            if fallback_result.success:
                fallback_result.execution_mode = ExecutionMode.FALLBACK
                fallback_result.resolution_method = ResolutionMethod.FALLBACK_AGENT_ID
                fallback_result.fallback_reason = "Used fallback_agent_id after primary resolution failed"
                self.logger.info(f"Resolved fallback agent '{fallback_result.agent_id}' for stage '{stage_spec.id}'")
                return fallback_result
        
        # Strategy 5: Last resort - try any agent that matches capabilities
        last_resort_result = await self._resolve_any_matching_agent(stage_spec)
        if last_resort_result.success:
            self.logger.warning(f"Used last resort resolution for stage '{stage_spec.id}': {last_resort_result.agent_id}")
            return last_resort_result
        
        # Complete failure - no suitable agent found
        error_msg = (f"No suitable agent found for task_type='{stage_spec.task_type}' "
                    f"with capabilities={stage_spec.required_capabilities}")
        self.logger.error(error_msg)
        raise AgentResolutionError(
            error_msg,
            task_type=stage_spec.task_type,
            required_capabilities=stage_spec.required_capabilities
        )
    
    async def _resolve_autonomous_agent(self, stage_spec: EnhancedMasterStageSpec) -> AgentResolutionResult:
        """Resolve autonomous-capable agent for the stage."""
        task_mapping = self._task_type_mappings.get(stage_spec.task_type, {})
        
        # Try preferred agents first
        for preferred_agent_id in task_mapping.get("preferred_agents", []):
            if preferred_agent_id in self._autonomous_capabilities_cache:
                agent_info = self._autonomous_capabilities_cache[preferred_agent_id]
                if self._matches_capabilities(agent_info["capabilities"], stage_spec.required_capabilities):
                    agent_instance = await self._instantiate_agent(preferred_agent_id)
                    if agent_instance:
                        return AgentResolutionResult(
                            agent_instance=agent_instance,
                            agent_id=preferred_agent_id,
                            execution_mode=ExecutionMode.AUTONOMOUS,
                            resolution_method=ResolutionMethod.AUTONOMOUS_PREFERENCE,
                            capabilities_matched=stage_spec.required_capabilities,
                            task_type=stage_spec.task_type,
                            confidence_score=0.9
                        )
        
        # Try any autonomous agent that matches capabilities
        for agent_id, agent_info in self._autonomous_capabilities_cache.items():
            if self._matches_capabilities(agent_info["capabilities"], stage_spec.required_capabilities):
                agent_instance = await self._instantiate_agent(agent_id)
                if agent_instance:
                    return AgentResolutionResult(
                        agent_instance=agent_instance,
                        agent_id=agent_id,
                        execution_mode=ExecutionMode.AUTONOMOUS,
                        resolution_method=ResolutionMethod.CAPABILITY_MATCH,
                        capabilities_matched=stage_spec.required_capabilities,
                        task_type=stage_spec.task_type,
                        confidence_score=0.7
                    )
        
        # No autonomous agent found
        return AgentResolutionResult(
            agent_instance=None,
            agent_id="",
            execution_mode=ExecutionMode.AUTONOMOUS,
            resolution_method=ResolutionMethod.CAPABILITY_MATCH,
            capabilities_matched=[],
            task_type=stage_spec.task_type,
            confidence_score=0.0,
            fallback_reason="No autonomous agent matches required capabilities"
        )
    
    async def _resolve_concrete_agent(self, stage_spec: EnhancedMasterStageSpec) -> AgentResolutionResult:
        """Resolve concrete agent with invoke_async method."""
        task_mapping = self._task_type_mappings.get(stage_spec.task_type, {})
        
        # Try fallback agents from task mapping
        for fallback_agent_id in task_mapping.get("fallback_agents", []):
            agent_instance = await self._instantiate_agent(fallback_agent_id)
            if agent_instance and hasattr(agent_instance, 'invoke_async'):
                return AgentResolutionResult(
                    agent_instance=agent_instance,
                    agent_id=fallback_agent_id,
                    execution_mode=ExecutionMode.CONCRETE,
                    resolution_method=ResolutionMethod.CONCRETE_FALLBACK,
                    capabilities_matched=[],
                    task_type=stage_spec.task_type,
                    confidence_score=0.6
                )
        
        # Try any registered agent with invoke_async
        for agent_id, agent_class in self.agent_registry.list_agents().items():
            if not self._is_autonomous_capable_class(agent_class):
                agent_instance = await self._instantiate_agent(agent_id)
                if agent_instance and hasattr(agent_instance, 'invoke_async'):
                    return AgentResolutionResult(
                        agent_instance=agent_instance,
                        agent_id=agent_id,
                        execution_mode=ExecutionMode.CONCRETE,
                        resolution_method=ResolutionMethod.CONCRETE_FALLBACK,
                        capabilities_matched=[],
                        task_type=stage_spec.task_type,
                        confidence_score=0.4
                    )
        
        # No concrete agent found
        return AgentResolutionResult(
            agent_instance=None,
            agent_id="",
            execution_mode=ExecutionMode.CONCRETE,
            resolution_method=ResolutionMethod.CONCRETE_FALLBACK,
            capabilities_matched=[],
            task_type=stage_spec.task_type,
            confidence_score=0.0,
            fallback_reason="No concrete agent with invoke_async found"
        )
    
    async def _resolve_direct_agent(self, agent_id: str, stage_spec: EnhancedMasterStageSpec) -> AgentResolutionResult:
        """Resolve agent by direct agent_id."""
        agent_instance = await self._instantiate_agent(agent_id)
        if agent_instance:
            execution_mode = ExecutionMode.AUTONOMOUS if self._is_autonomous_capable_class(type(agent_instance)) else ExecutionMode.DIRECT
            return AgentResolutionResult(
                agent_instance=agent_instance,
                agent_id=agent_id,
                execution_mode=execution_mode,
                resolution_method=ResolutionMethod.DIRECT_AGENT_ID,
                capabilities_matched=[],
                task_type=stage_spec.task_type,
                confidence_score=1.0
            )
        
        return AgentResolutionResult(
            agent_instance=None,
            agent_id=agent_id,
            execution_mode=ExecutionMode.DIRECT,
            resolution_method=ResolutionMethod.DIRECT_AGENT_ID,
            capabilities_matched=[],
            task_type=stage_spec.task_type,
            confidence_score=0.0,
            fallback_reason=f"Agent '{agent_id}' could not be instantiated"
        )
    
    async def _resolve_any_matching_agent(self, stage_spec: EnhancedMasterStageSpec) -> AgentResolutionResult:
        """Last resort: try any agent that might work."""
        # Try any agent in the registry
        for agent_id, agent_class in self.agent_registry.list_agents().items():
            agent_instance = await self._instantiate_agent(agent_id)
            if agent_instance and hasattr(agent_instance, 'invoke_async'):
                execution_mode = ExecutionMode.AUTONOMOUS if self._is_autonomous_capable_class(agent_class) else ExecutionMode.CONCRETE
                return AgentResolutionResult(
                    agent_instance=agent_instance,
                    agent_id=agent_id,
                    execution_mode=execution_mode,
                    resolution_method=ResolutionMethod.CONCRETE_FALLBACK,
                    capabilities_matched=[],
                    task_type=stage_spec.task_type,
                    confidence_score=0.2,
                    fallback_reason="Last resort: using any available agent"
                )
        
        # Absolute failure
        return AgentResolutionResult(
            agent_instance=None,
            agent_id="",
            execution_mode=ExecutionMode.CONCRETE,
            resolution_method=ResolutionMethod.CONCRETE_FALLBACK,
            capabilities_matched=[],
            task_type=stage_spec.task_type,
            confidence_score=0.0,
            fallback_reason="No agents available in registry"
        )
    
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
    
    async def _instantiate_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Instantiate an agent by ID using the agent provider."""
        try:
            # Use the agent provider to get the agent instance
            return self.agent_provider.get(identifier=agent_id)
        except Exception as e:
            self.logger.debug(f"Could not get raw agent instance for agent_id: {agent_id}. Not found in fallback or registry (registry lookup not fully implemented here).")
            return None
    
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