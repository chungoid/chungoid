"""
Protocol-Aware Agent Base Class

Enhances agents with the ability to follow systematic protocols,
transforming them from single-shot executors to protocol-driven engineers.
"""

import logging
import time
from typing import Any, Dict, List, Optional, TypeVar, Generic
from abc import abstractmethod

from pydantic import Field, PrivateAttr

from ..protocols import get_protocol, ProtocolInterface
from ..protocols.base.protocol_interface import PhaseStatus, ProtocolPhase
from chungoid.runtime.agents.agent_base import BaseAgent, InputSchema, OutputSchema

logger = logging.getLogger(__name__)

class ProtocolAwareAgent(BaseAgent[InputSchema, OutputSchema], Generic[InputSchema, OutputSchema]):
    """
    Base agent class that can follow systematic protocols.
    
    Transforms agents from "single-shot executors" to "protocol-driven engineers"
    by enabling them to follow rigorous, proven methodologies.
    """
    
    # Protocol-related private attributes
    _current_protocol: Optional[ProtocolInterface] = PrivateAttr(default=None)
    _protocol_context: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    @property
    def current_protocol(self) -> Optional[ProtocolInterface]:
        return self._current_protocol
    
    @current_protocol.setter
    def current_protocol(self, value: Optional[ProtocolInterface]):
        self._current_protocol = value
    
    @property
    def protocol_context(self) -> Dict[str, Any]:
        return self._protocol_context
    
    @protocol_context.setter
    def protocol_context(self, value: Dict[str, Any]):
        self._protocol_context = value
    
    def execute_with_protocol(self, task: Dict[str, Any], protocol_name: str) -> Dict[str, Any]:
        """
        Execute a task following a specific protocol.
        
        Args:
            task: The task to execute
            protocol_name: Name of the protocol to follow
            
        Returns:
            Protocol execution results
        """
        logger.info(f"Starting protocol execution: {protocol_name}")
        
        # Load and setup protocol
        self.current_protocol = get_protocol(protocol_name)
        self.current_protocol.setup(context=task)
        
        results = {
            "protocol_name": protocol_name,
            "task": task,
            "phases": [],
            "overall_success": False,
            "execution_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # Execute each protocol phase
            for phase in self.current_protocol.phases:
                phase_result = self._execute_protocol_phase(phase)
                results["phases"].append(phase_result)
                
                if not phase_result["success"]:
                    logger.error(f"Protocol phase failed: {phase.name}")
                    break
            
            # Check overall protocol completion
            results["overall_success"] = all(
                phase["success"] for phase in results["phases"]
            )
            
            results["execution_time"] = time.time() - start_time
            results["progress_summary"] = self.current_protocol.get_progress_summary()
            
            logger.info(f"Protocol execution completed: {results['overall_success']}")
            return results
            
        except Exception as e:
            logger.error(f"Protocol execution failed: {str(e)}")
            results["error"] = str(e)
            results["execution_time"] = time.time() - start_time
            return results
    
    def _execute_protocol_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute a single protocol phase."""
        logger.info(f"Executing protocol phase: {phase.name}")
        
        phase_result = {
            "phase_name": phase.name,
            "success": False,
            "outputs": {},
            "validation_results": {},
            "execution_time": 0.0,
            "retry_count": 0
        }
        
        start_time = time.time()
        max_retries = 3
        
        while phase.retry_count < max_retries:
            try:
                # Check if phase is ready (dependencies met)
                if not self.current_protocol.is_phase_ready(phase):
                    logger.warning(f"Phase {phase.name} dependencies not met")
                    break
                
                phase.status = PhaseStatus.IN_PROGRESS
                
                # Execute phase-specific logic
                phase_outputs = self._execute_phase_logic(phase)
                phase.outputs.update(phase_outputs)
                
                # Validate phase completion
                validation_results = self._validate_phase_completion(phase)
                phase.validation_results.update(validation_results)
                
                # Check if all validation criteria passed
                if all(validation_results.values()):
                    phase.status = PhaseStatus.COMPLETED
                    phase_result["success"] = True
                    logger.info(f"Phase {phase.name} completed successfully")
                    break
                else:
                    phase.status = PhaseStatus.REQUIRES_RETRY
                    phase.retry_count += 1
                    logger.warning(f"Phase {phase.name} validation failed, retrying ({phase.retry_count}/{max_retries})")
                    
            except Exception as e:
                phase.status = PhaseStatus.FAILED
                phase_result["error"] = str(e)
                logger.error(f"Phase {phase.name} failed: {str(e)}")
                break
        
        phase.execution_time = time.time() - start_time
        
        # Update phase result
        phase_result.update({
            "outputs": dict(phase.outputs),
            "validation_results": dict(phase.validation_results),
            "execution_time": phase.execution_time,
            "retry_count": phase.retry_count
        })
        
        return phase_result
    
    @abstractmethod
    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """
        Execute the agent-specific logic for a protocol phase.
        
        Subclasses must implement this to define how they handle each phase.
        """
        pass
    
    def _validate_phase_completion(self, phase: ProtocolPhase) -> Dict[str, bool]:
        """
        Validate that a phase has been completed according to its criteria.
        
        Default implementation checks for required outputs.
        Subclasses can override for more sophisticated validation.
        """
        validation_results = {}
        
        # Check required outputs are present
        for required_output in phase.required_outputs:
            validation_results[f"has_{required_output}"] = required_output in phase.outputs
        
        # Check validation criteria (basic implementation)
        for i, criteria in enumerate(phase.validation_criteria):
            # This is a simplified check - real implementation would be more sophisticated
            validation_results[f"criteria_{i}"] = True  # Placeholder
        
        return validation_results
    
    def get_protocol_status(self) -> Optional[Dict[str, Any]]:
        """Get current protocol execution status."""
        if not self.current_protocol:
            return None
        
        return {
            "protocol_name": self.current_protocol.name,
            "current_phase": self.current_protocol.get_current_phase().name if self.current_protocol.get_current_phase() else None,
            "progress_summary": self.current_protocol.get_progress_summary()
        }
    
    def use_protocol_template(self, template_name: str, **variables) -> str:
        """Use a protocol template with variable substitution."""
        if not self.current_protocol:
            raise ValueError("No active protocol")
        
        return self.current_protocol.get_template(template_name, **variables)

class ArchitectureDiscoveryAgent(ProtocolAwareAgent):
    """
    Example agent that follows the Deep Implementation Protocol
    for architecture discovery and implementation planning.
    """
    
    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute architecture discovery specific logic for each phase."""
        
        if phase.name == "architecture_discovery":
            return self._discover_existing_architecture(phase)
        elif phase.name == "integration_analysis":
            return self._analyze_integration_points(phase)
        elif phase.name == "compatibility_design":
            return self._design_compatibility_layer(phase)
        elif phase.name == "implementation_planning":
            return self._create_implementation_plan(phase)
        else:
            logger.warning(f"Unknown phase: {phase.name}")
            return {}
    
    def _discover_existing_architecture(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 1: Discover existing architecture patterns."""
        logger.info("Discovering existing architecture...")
        
        outputs = {}
        
        # Use required tools to understand codebase
        if "codebase_search" in phase.tools_required:
            # Example: Search for existing agent patterns
            outputs["agent_patterns"] = self._search_agent_patterns()
        
        if "list_dir" in phase.tools_required:
            # Example: Analyze project structure
            outputs["project_structure"] = self._analyze_project_structure()
        
        # Generate architecture discovery document using protocol template
        template_vars = {
            "feature_name": self.current_protocol.context.get("feature_name", "unknown"),
            "agent_base_classes": outputs.get("agent_patterns", {}),
            "list_main_source_directories": outputs.get("project_structure", {})
        }
        
        outputs["architecture_discovery_document"] = self.use_protocol_template(
            "architecture_discovery", 
            **template_vars
        )
        
        return outputs
    
    def _analyze_integration_points(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 2: Analyze integration points."""
        logger.info("Analyzing integration points...")
        
        # Implementation would use tools to map dependencies
        # This is a simplified example
        return {
            "integration_analysis_document": "# Integration analysis completed",
            "dependency_map": {"direct": [], "inverse": []},
            "interface_specs": {}
        }
    
    def _design_compatibility_layer(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 3: Design compatibility layer."""
        logger.info("Designing compatibility layer...")
        
        # Implementation would create detailed design
        return {
            "compatibility_design_document": "# Compatibility design completed",
            "interface_definitions": {},
            "migration_strategy": {}
        }
    
    def _create_implementation_plan(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 4: Create detailed implementation plan."""
        logger.info("Creating implementation plan...")
        
        # Implementation would create detailed plan with risk assessment
        return {
            "implementation_plan_document": "# Implementation plan completed",
            "phase_breakdown": {},
            "risk_assessment": {},
            "timeline": {}
        }
    
    def _search_agent_patterns(self) -> Dict[str, Any]:
        """Use codebase_search to find existing agent patterns."""
        # This would use the actual codebase_search tool
        return {"pattern": "BaseAgent inheritance pattern found"}
    
    def _analyze_project_structure(self) -> Dict[str, Any]:
        """Use list_dir to analyze project structure."""
        # This would use the actual list_dir tool
        return {"structure": "src/chungoid/ structure analyzed"} 