"""
Protocol-Aware Agent Base Class

Streamlined autonomous agent base class for protocol-driven execution.
Eliminates all backwards compatibility and legacy patterns.
"""

import logging
import time
from typing import Any, Dict, List, Optional, ClassVar
from abc import abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from ..protocols import get_protocol, ProtocolInterface
from ..protocols.base.protocol_interface import PhaseStatus, ProtocolPhase
from ..schemas.agent_outputs import AgentOutput
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class ProtocolAwareAgent(BaseModel):
    """
    Streamlined autonomous agent base class for protocol-driven execution.
    
    This is the ONLY base class for all autonomous agents. It eliminates all
    backwards compatibility and legacy patterns in favor of pure protocol execution.
    
    Key principles:
    - Protocol-first execution (no invoke_async)
    - Type-safe outputs (AgentOutput subclasses)
    - Autonomous operation (no manual orchestration)
    - Consistent initialization patterns
    """
    
    # Required class variables (enforced by validation)
    AGENT_ID: ClassVar[str]
    AGENT_VERSION: ClassVar[str] 
    PRIMARY_PROTOCOLS: ClassVar[List[str]]
    SECONDARY_PROTOCOLS: ClassVar[List[str]]
    CAPABILITIES: ClassVar[List[str]]
    
    # Standard initialization fields
    llm_provider: LLMProvider = Field(..., description="LLM provider for AI capabilities")
    prompt_manager: PromptManager = Field(..., description="Prompt manager for template rendering")
    system_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="System-level context")
    agent_id: Optional[str] = Field(None, description="Runtime agent identifier")
    
    # Protocol-related private attributes
    _current_protocol: Optional[ProtocolInterface] = PrivateAttr(default=None)
    _protocol_context: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _logger: Optional[logging.Logger] = PrivateAttr(default=None)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data: Any):
        """Initialize the protocol-aware agent with required dependencies."""
        super().__init__(**data)
        
        # Validate required class variables
        self._validate_class_variables()
        
        # Set runtime agent_id if not provided
        if not self.agent_id:
            self.agent_id = self.AGENT_ID
        
        # Initialize logger
        self._logger = self._setup_logger()
        
        self._logger.info(f"Initialized autonomous agent: {self.agent_id} (v{self.AGENT_VERSION})")
    
    def _validate_class_variables(self):
        """Validate that all required class variables are defined."""
        required_vars = ['AGENT_ID', 'AGENT_VERSION', 'PRIMARY_PROTOCOLS', 'SECONDARY_PROTOCOLS', 'CAPABILITIES']
        
        for var_name in required_vars:
            if not hasattr(self.__class__, var_name):
                raise ValueError(f"Agent class {self.__class__.__name__} must define {var_name}")
        
        # Validate PRIMARY_PROTOCOLS is not empty
        if not self.PRIMARY_PROTOCOLS:
            raise ValueError(f"Agent {self.AGENT_ID} must have at least one PRIMARY_PROTOCOL")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this agent instance."""
        if self.system_context and 'logger' in self.system_context:
            return self.system_context['logger']
        
        # Create agent-specific logger
        logger_name = f"agent.{self.AGENT_ID}"
        agent_logger = logging.getLogger(logger_name)
        
        if not agent_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            agent_logger.addHandler(handler)
            agent_logger.setLevel(logging.INFO)
            agent_logger.propagate = False
        
        return agent_logger
    
    @property
    def logger(self) -> logging.Logger:
        """Get the agent's logger."""
        return self._logger
    
    @property
    def current_protocol(self) -> Optional[ProtocolInterface]:
        """Get the currently active protocol."""
        return self._current_protocol
    
    @current_protocol.setter
    def current_protocol(self, value: Optional[ProtocolInterface]):
        """Set the currently active protocol."""
        self._current_protocol = value
    
    @property
    def protocol_context(self) -> Dict[str, Any]:
        """Get the current protocol execution context."""
        return self._protocol_context
    
    @protocol_context.setter
    def protocol_context(self, value: Dict[str, Any]):
        """Set the protocol execution context."""
        self._protocol_context = value
    
    async def execute_with_protocol(self, protocol: str, context: Dict[str, Any]) -> AgentOutput:
        """
        Execute a task following a specific protocol.
        
        This is the ONLY entry point for autonomous agent execution.
        No invoke_async, no backwards compatibility.
        """
        self.logger.info(f"Executing with protocol: {protocol}")
        
        start_time = time.time()
        
        try:
            # Load and initialize the protocol
            self.current_protocol = get_protocol(protocol)
            self.protocol_context = context.copy()
            
            # Execute all phases in sequence
            phases_completed = []
            overall_success = True
            
            for phase in self.current_protocol.phases:
                self.logger.info(f"Starting phase: {phase.name}")
                
                # Execute the phase
                phase_result = await self._execute_protocol_phase(phase)
                phases_completed.append(phase.name)
                
                # Check if phase failed
                if not phase_result["success"]:
                    overall_success = False
                    self.logger.error(f"Phase {phase.name} failed, stopping protocol execution")
                    break
            
            execution_time = time.time() - start_time
            
            # Create standardized output
            return AgentOutput(
                success=overall_success,
                data=self.protocol_context,
                agent_id=self.agent_id,
                protocol_used=protocol,
                phases_completed=phases_completed,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Protocol execution failed: {str(e)}")
            
            return AgentOutput(
                success=False,
                data={},
                error=str(e),
                agent_id=self.agent_id,
                protocol_used=protocol,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    async def _execute_protocol_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute a single protocol phase with retry logic and validation."""
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
                    self.logger.warning(f"Phase {phase.name} dependencies not met")
                    break
                
                phase.status = PhaseStatus.IN_PROGRESS
                
                # Execute phase-specific logic
                phase_outputs = await self._execute_phase_logic(phase, self.protocol_context)
                phase.outputs.update(phase_outputs)
                
                # Validate phase completion
                validation_results = self._validate_phase_completion(phase)
                phase.validation_results.update(validation_results)
                
                # Check if all validation criteria passed
                if all(validation_results.values()):
                    phase.status = PhaseStatus.COMPLETED
                    phase_result["success"] = True
                    self.logger.info(f"Phase {phase.name} completed successfully")
                    break
                else:
                    phase.status = PhaseStatus.REQUIRES_RETRY
                    phase.retry_count += 1
                    self.logger.warning(f"Phase {phase.name} validation failed, retrying ({phase.retry_count}/{max_retries})")
                    
            except Exception as e:
                phase.status = PhaseStatus.FAILED
                phase_result["error"] = str(e)
                self.logger.error(f"Phase {phase.name} failed: {str(e)}")
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
    async def _execute_phase_logic(self, phase: ProtocolPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent-specific logic for a protocol phase.
        
        Subclasses must implement this to define how they handle each phase.
        This is the core method that defines agent behavior.
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
    
    def get_name(self) -> str:
        """Get the agent's name."""
        return getattr(self.__class__, 'AGENT_NAME', self.__class__.__name__)
    
    def get_id(self) -> str:
        """Get the agent's unique identifier."""
        return self.AGENT_ID
    
    @classmethod
    def get_agent_card_static(cls):
        """Subclasses should override this to return their AgentCard."""
        raise NotImplementedError("Subclasses must implement get_agent_card_static to provide an AgentCard.")


# NO invoke_async method - completely removed
# NO BaseAgent inheritance - standalone implementation
# NO backwards compatibility - pure protocol execution only 