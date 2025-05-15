from __future__ import annotations

import logging
from typing import Any, Dict, Union, Optional

from chungoid.schemas.agent_mock_code_generator import MockCodeGeneratorAgentInput, MockCodeGeneratorAgentOutput
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard

logger = logging.getLogger(__name__)

class MockCodeGeneratorAgent:
    AGENT_ID = "MockCodeGeneratorAgent_v1"
    AGENT_NAME = "Mock Code Generator Agent"
    VERSION = "0.1.0"
    DESCRIPTION = "Mocks code generation. Does not actually write files. For testing autonomous flows."
    CATEGORY = AgentCategory.TESTING_MOCK 
    VISIBILITY = AgentVisibility.INTERNAL

    async def invoke_async(
        self,
        inputs: MockCodeGeneratorAgentInput, # Expecting Pydantic model or dict
        full_context: Optional[Dict[str, Any]] = None,
    ) -> Union[MockCodeGeneratorAgentOutput, AgentErrorDetails]:
        
        if isinstance(inputs, dict):
            try:
                parsed_inputs = MockCodeGeneratorAgentInput(**inputs)
            except Exception as e:
                logger.error(f"Error parsing inputs for {self.AGENT_ID}: {e}")
                return AgentErrorDetails(error_code="INPUT_PARSING_ERROR", message=f"Failed to parse inputs: {e}")
        else:
            parsed_inputs = inputs

        logger.info(f"{self.AGENT_ID} invoked. Mocking code generation for target file: '{parsed_inputs.target_file_path}'")
        logger.info(f"Specification prompt (mock ignored): {parsed_inputs.code_specification_prompt}")
        if parsed_inputs.specification:
            logger.info(f"Received direct specification: {parsed_inputs.specification}")
        
        # In a real agent, actual_code_generation_logic_here()
        # For the mock, we just return success and the path.
        
        return MockCodeGeneratorAgentOutput(
            code_changes_applied=True, 
            generated_artifact_path=parsed_inputs.target_file_path
        )

def get_agent_card_static() -> AgentCard:
    """Returns the static AgentCard for the MockCodeGeneratorAgent."""
    return AgentCard(
        agent_id=MockCodeGeneratorAgent.AGENT_ID,
        name=MockCodeGeneratorAgent.AGENT_NAME,
        version=MockCodeGeneratorAgent.VERSION,
        description=MockCodeGeneratorAgent.DESCRIPTION,
        category=MockCodeGeneratorAgent.CATEGORY,
        visibility=MockCodeGeneratorAgent.VISIBILITY,
        input_schema=MockCodeGeneratorAgentInput.model_json_schema(),
        output_schema=MockCodeGeneratorAgentOutput.model_json_schema(),
    ) 