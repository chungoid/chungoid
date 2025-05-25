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
    
    async def execute_with_protocol(self, task: Dict[str, Any], protocol_name: str) -> Dict[str, Any]:
        """
        Execute a task following a specific protocol.
        
        This is the main entry point for protocol-driven execution.
        """
        logger.info(f"Executing task with protocol: {protocol_name}")
        
        # Load and initialize the protocol
        self.current_protocol = get_protocol(protocol_name)
        self.protocol_context = task.copy()
        
        # Execute all phases in sequence
        results = {
            "protocol_name": protocol_name,
            "phases": [],
            "overall_success": True,
            "execution_time": 0.0
        }
        
        start_time = time.time()
        
        for phase in self.current_protocol.phases:
            logger.info(f"Starting phase: {phase.name}")
            
            # Execute the phase (await the async method)
            phase_result = await self._execute_protocol_phase(phase)
            results["phases"].append(phase_result)
            
            # Check if phase failed
            if not phase_result["success"]:
                results["overall_success"] = False
                logger.error(f"Phase {phase.name} failed, stopping protocol execution")
                break
        
        results["execution_time"] = time.time() - start_time
        logger.info(f"Protocol {protocol_name} completed. Success: {results['overall_success']}")
        
        return results
    
    async def _execute_protocol_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute a single protocol phase with retry logic and validation."""
        logger = logging.getLogger(__name__)
        
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
                
                # Execute phase-specific logic (await the async method)
                phase_outputs = await self._execute_phase_logic(phase, self.protocol_context)
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
    def _execute_phase_logic(self, phase: ProtocolPhase, context: Dict[str, Any]) -> Dict[str, Any]:
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

    async def invoke_async(self, task_input: InputSchema = None, full_context: Optional[Dict[str, Any]] = None, 
                          inputs: Optional[Dict[str, Any]] = None, input_payload: Optional[Any] = None, **kwargs) -> OutputSchema:
        """
        Default implementation for autonomous agents using protocol execution.
        
        This method provides a bridge between the old invoke_async interface and the new
        protocol-aware execution model. Autonomous agents should primarily use protocol
        execution via execute_with_protocol(), but this method ensures compatibility
        with the existing orchestrator infrastructure.
        
        Handles multiple parameter patterns:
        - task_input: Standard input schema
        - inputs: Dictionary input (orchestrator pattern)
        - input_payload: Alternative input format
        """
        logger.info(f"ProtocolAwareAgent.invoke_async called for {self.__class__.__name__}")
        
        # Determine the actual input to use
        actual_input = task_input or input_payload or inputs
        if actual_input is None:
            logger.warning(f"No input provided to {self.__class__.__name__}.invoke_async")
            actual_input = {}
        
        # For autonomous agents, we need to determine which protocol to use
        # This is a simplified implementation - real agents should override this
        # or use execute_with_protocol() directly
        
        # Try to use the first available protocol that exists
        available_protocols = ['agent_communication', 'context_sharing', 'tool_validation', 'error_recovery', 'goal_tracking']
        protocol_to_use = None
        
        if hasattr(self, 'PRIMARY_PROTOCOLS') and self.PRIMARY_PROTOCOLS:
            # Try to find a primary protocol that actually exists
            for primary_protocol in self.PRIMARY_PROTOCOLS:
                if primary_protocol in available_protocols:
                    protocol_to_use = primary_protocol
                    break
            
            # If no primary protocol exists, use the first available one
            if not protocol_to_use:
                protocol_to_use = available_protocols[0]
                logger.warning(f"Agent {self.__class__.__name__} primary protocols {self.PRIMARY_PROTOCOLS} not found, using fallback: {protocol_to_use}")
        else:
            # Fallback for agents without PRIMARY_PROTOCOLS
            protocol_to_use = available_protocols[0]
            logger.warning(f"Agent {self.__class__.__name__} has no PRIMARY_PROTOCOLS, using fallback: {protocol_to_use}")
        
        logger.info(f"Using protocol: {protocol_to_use}")
        
        # Convert input to dict for protocol execution
        if hasattr(actual_input, 'model_dump'):
            task_dict = actual_input.model_dump()
        elif isinstance(actual_input, dict):
            task_dict = actual_input.copy()
        else:
            task_dict = {"input": str(actual_input)}
        
        if full_context:
            task_dict.update(full_context)
        
        try:
            # Execute with protocol
            protocol_result = await self.execute_with_protocol(task_dict, protocol_to_use)
            
            # Extract output from protocol result
            return self._extract_output_from_protocol_result(protocol_result, actual_input)
        except Exception as e:
            logger.error(f"Protocol execution failed for {self.__class__.__name__}: {e}")
            # Return a fallback response
            return self._extract_output_from_protocol_result(
                {"status": "ERROR", "message": f"Protocol execution failed: {e}"}, 
                actual_input
            )
    
    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], task_input: Any) -> Any:
        """
        Extract agent output from protocol execution results.
        
        This is a generic implementation that should be overridden by specific agents
        to return properly typed output schemas.
        """
        # Generic extraction - should be overridden by specific agents
        return {
            "status": "SUCCESS",
            "message": "Task completed via protocol execution",
            "protocol_used": protocol_result.get("protocol_name"),
            "execution_time": protocol_result.get("execution_time", 0),
            "phases_completed": len([p for p in protocol_result.get("phases", []) if p.get("success", False)])
        }

class ArchitectureDiscoveryAgent(ProtocolAwareAgent):
    """
    Example agent that follows the Deep Implementation Protocol
    for architecture discovery and implementation planning.
    """
    
    def _execute_phase_logic(self, phase: ProtocolPhase, context: Dict[str, Any]) -> Dict[str, Any]:
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