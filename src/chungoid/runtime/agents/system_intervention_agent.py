"""
System Intervention Agent

This agent handles human intervention requests and user clarification needs.
It provides a simple interface for pausing execution and requesting user input.
"""

import logging
from typing import Dict, Any, Optional, ClassVar
from pydantic import BaseModel, Field

from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
from chungoid.registry.decorators import register_system_agent
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility

logger = logging.getLogger(__name__)


class SystemInterventionInput(BaseModel):
    """Input schema for SystemInterventionAgent_v1."""
    
    prompt_message_for_user: str = Field(..., description="Message to display to the user")
    context_data: Optional[Dict[str, Any]] = Field(None, description="Additional context data")
    intervention_type: str = Field(default="clarification", description="Type of intervention needed")


class SystemInterventionOutput(BaseModel):
    """Output schema for SystemInterventionAgent_v1."""
    
    message: str = Field(..., description="Response message")
    user_response: Optional[str] = Field(None, description="User's response if available")
    intervention_status: str = Field(..., description="Status of the intervention")
    next_action: str = Field(default="continue", description="Recommended next action")


@register_system_agent(capabilities=["human_interaction", "system_intervention", "user_clarification"])
class SystemInterventionAgent_v1(ProtocolAwareAgent[SystemInterventionInput, SystemInterventionOutput]):
    """
    System agent for handling human intervention and user clarification requests.
    
    This agent provides a standardized interface for pausing execution and
    requesting user input when automated processing cannot proceed.
    """
    
    AGENT_ID: ClassVar[str] = "SystemInterventionAgent_v1"
    AGENT_NAME: ClassVar[str] = "System Intervention Agent"
    VERSION: ClassVar[str] = "1.0.0"
    DESCRIPTION: ClassVar[str] = "Handles human intervention requests and user clarification needs"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.SYSTEM_ORCHESTRATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL
    
    # Protocol definitions
    PRIMARY_PROTOCOLS: ClassVar[list[str]] = ['human_interaction']
    SECONDARY_PROTOCOLS: ClassVar[list[str]] = ['system_intervention', 'user_clarification']
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing']
    
    def __init__(self, 
                 llm_provider=None, 
                 prompt_manager=None, 
                 system_context: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            system_context=system_context,
            **kwargs
        )
        self._logger = logging.getLogger(f"{__name__}.{self.AGENT_ID}")
        self._logger.info(f"{self.AGENT_ID} (v{self.VERSION}) initialized.")
    
    async def invoke_async(
        self,
        inputs: SystemInterventionInput,
        full_context: Optional[Dict[str, Any]] = None,
    ) -> SystemInterventionOutput:
        """
        Handle system intervention request.
        
        In a real implementation, this would pause execution and wait for user input.
        For now, it provides a mock response to allow the system to continue.
        """
        self._logger.info(f"SystemInterventionAgent_v1 invoked with message: {inputs.prompt_message_for_user}")
        
        # In a real implementation, this would:
        # 1. Display the prompt_message_for_user to the user
        # 2. Wait for user input
        # 3. Return the user's response
        
        # For now, provide a mock response to allow testing
        mock_response = f"Mock user response to: {inputs.prompt_message_for_user}"
        
        return SystemInterventionOutput(
            message=f"Intervention request processed: {inputs.prompt_message_for_user}",
            user_response=mock_response,
            intervention_status="completed_mock",
            next_action="continue"
        )
    
    def _execute_phase_logic(self, phase) -> Dict[str, Any]:
        """Execute agent-specific logic for protocol phases."""
        if phase.name == "human_interaction":
            return {
                "interaction_type": "user_clarification",
                "status": "awaiting_input",
                "method": "intervention_protocol"
            }
        else:
            return {"phase_completed": True, "method": "generic"}
    
    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], task_input) -> SystemInterventionOutput:
        """Extract agent output from protocol execution results."""
        return SystemInterventionOutput(
            message=f"Protocol execution completed for intervention request",
            intervention_status="completed_protocol",
            next_action="continue"
        )
    
    @classmethod
    def get_input_schema(cls):
        return SystemInterventionInput
    
    @classmethod
    def get_output_schema(cls):
        return SystemInterventionOutput 