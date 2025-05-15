from __future__ import annotations

import logging
from typing import Any, Dict, Union, Optional

from chungoid.schemas.agent_mock_human_input import MockHumanInputAgentInput, MockHumanInputAgentOutput
from chungoid.schemas.errors import AgentErrorDetails # For type hinting, though mock won't error
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard

logger = logging.getLogger(__name__)

class MockHumanInputAgent:
    AGENT_ID = "MockHumanInputAgent_v1"
    AGENT_NAME = "Mock Human Input Agent"
    VERSION = "0.1.0"
    DESCRIPTION = "Mocks human input by providing a predefined specification. For testing autonomous flows."
    CATEGORY = AgentCategory.TESTING_MOCK # Or a more specific category if available
    VISIBILITY = AgentVisibility.INTERNAL # Not for general public use

    async def invoke_async(
        self,
        inputs: MockHumanInputAgentInput, # Expecting Pydantic model from orchestrator if input validation is set up
                                          # Or raw dict if not. For mocks, let's be flexible.
        full_context: Optional[Dict[str, Any]] = None, 
    ) -> Union[MockHumanInputAgentOutput, AgentErrorDetails]: # Return Pydantic model for consistency
        
        # If inputs come as a dict, load into Pydantic model first
        if isinstance(inputs, dict):
            try:
                parsed_inputs = MockHumanInputAgentInput(**inputs)
            except Exception as e:
                logger.error(f"Error parsing inputs for {self.AGENT_ID}: {e}")
                # This mock shouldn't ideally fail here, but good practice for real agents
                return AgentErrorDetails(error_code="INPUT_PARSING_ERROR", message=f"Failed to parse inputs: {e}")
        else:
            parsed_inputs = inputs

        logger.info(f"{self.AGENT_ID} invoked. Mocking human input for prompt: '{parsed_inputs.prompt_message_for_user}'")
        logger.info(f"Target context path suggested in input: {parsed_inputs.target_context_path}")

        # Predefined mock specification for 'chungoid utils show-config'
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

        return MockHumanInputAgentOutput(specification_output=mock_spec)

def get_agent_card_static() -> AgentCard:
    """Returns the static AgentCard for the MockHumanInputAgent."""
    return AgentCard(
        agent_id=MockHumanInputAgent.AGENT_ID,
        name=MockHumanInputAgent.AGENT_NAME,
        version=MockHumanInputAgent.VERSION,
        description=MockHumanInputAgent.DESCRIPTION,
        category=MockHumanInputAgent.CATEGORY,
        visibility=MockHumanInputAgent.VISIBILITY,
        input_schema=MockHumanInputAgentInput.model_json_schema(),
        output_schema=MockHumanInputAgentOutput.model_json_schema(),
    ) 