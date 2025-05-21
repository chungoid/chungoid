from __future__ import annotations

import logging
from typing import Any, Dict, Union, Optional

# Let's assume the schemas are small enough to be co-located for this mock agent
from pydantic import BaseModel, Field # Added Pydantic import

class MockSystemInterventionAgentInput(BaseModel):
    prompt_message_for_user: str = Field(..., description="The message/question to present to the human user.")
    target_context_path: Optional[str] = Field(None, description="Suggested path in context for the output. Mock may ignore this.")
    # Other fields from the planner's 'inputs' for this stage can be added if needed

class MockSystemInterventionAgentOutput(BaseModel):
    user_response: Any = Field(..., description="The response received from the human user.")
    # Note: 'Any' is used here because the user response could be complex.


from chungoid.schemas.errors import AgentErrorDetails # For type hinting
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard

logger = logging.getLogger(__name__)

class MockSystemInterventionAgent:
    AGENT_ID = "SystemInterventionAgent_v1"
    AGENT_NAME = "Mock System Intervention Agent"
    VERSION = "1.0.0"
    DESCRIPTION = "Mocks system-level intervention by providing a predefined specification. For testing autonomous flows when planner-level clarification would be needed."
    CATEGORY = AgentCategory.TESTING_MOCK 
    VISIBILITY = AgentVisibility.INTERNAL

    async def invoke_async(
        self,
        inputs: Union[MockSystemInterventionAgentInput, Dict[str, Any]], # Allow dict for flexibility
        full_context: Optional[Dict[str, Any]] = None, 
    ) -> Union[MockSystemInterventionAgentOutput, AgentErrorDetails]:
        
        if isinstance(inputs, dict):
            try:
                parsed_inputs = MockSystemInterventionAgentInput(**inputs)
            except Exception as e:
                logger.error(f"Error parsing inputs for {self.AGENT_ID}: {e}")
                return AgentErrorDetails(error_code="INPUT_PARSING_ERROR", message=f"Failed to parse inputs: {e}")
        else:
            parsed_inputs = inputs

        logger.info(f"{self.AGENT_ID} invoked. Mocking system intervention for prompt: '{parsed_inputs.prompt_message_for_user}'")
        logger.info(f"Target context path suggested in input: {parsed_inputs.target_context_path}")

        mock_spec = {
            "command_name": "show-config",
            "purpose": "Displays the current Chungoid project's effective configuration settings.",
            "options": [
                {"name": "--format", "type": "Choice['json', 'yaml', 'text']", "default": "text", "help": "Output format."},
                {"name": "--all", "type": "bool", "default": False, "help": "Show all settings, including defaults."}
            ],
            "output_description": "Outputs configuration settings. Text format is human-readable, JSON/YAML for machine parsing."
        }

        logger.info(f"Providing mock specification: {mock_spec}")

        return MockSystemInterventionAgentOutput(user_response=mock_spec)

def get_agent_card_static() -> AgentCard:
    """Returns the static AgentCard for the MockSystemInterventionAgent."""
    return AgentCard(
        agent_id=MockSystemInterventionAgent.AGENT_ID,
        name=MockSystemInterventionAgent.AGENT_NAME,
        version=MockSystemInterventionAgent.VERSION,
        description=MockSystemInterventionAgent.DESCRIPTION,
        category=MockSystemInterventionAgent.CATEGORY,
        visibility=MockSystemInterventionAgent.VISIBILITY,
        input_schema=MockSystemInterventionAgentInput.model_json_schema(),
        output_schema=MockSystemInterventionAgentOutput.model_json_schema(),
    ) 