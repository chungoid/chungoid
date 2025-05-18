from __future__ import annotations

import logging
from typing import Any, Dict, Union, Optional
from pathlib import Path

from chungoid.schemas.agent_mock_test_generator import MockTestGeneratorAgentInput, MockTestGeneratorAgentOutput
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard

logger = logging.getLogger(__name__)

class MockTestGeneratorAgent:
    AGENT_ID = "MockTestGeneratorAgent_v1"
    AGENT_NAME = "Mock Test Generator Agent"
    VERSION = "0.1.0"
    DESCRIPTION = "Mocks test generation. Does not actually write test files. For testing autonomous flows."
    CATEGORY = AgentCategory.TESTING_MOCK
    VISIBILITY = AgentVisibility.INTERNAL

    async def invoke_async(
        self,
        inputs: MockTestGeneratorAgentInput, # Expecting Pydantic model or dict
        full_context: Optional[Dict[str, Any]] = None,
    ) -> Union[MockTestGeneratorAgentOutput, AgentErrorDetails]:
        
        if isinstance(inputs, dict):
            try:
                parsed_inputs = MockTestGeneratorAgentInput(**inputs)
            except Exception as e:
                logger.error(f"Error parsing inputs for {self.AGENT_ID}: {e}")
                return AgentErrorDetails(error_code="INPUT_PARSING_ERROR", message=f"Failed to parse inputs: {e}")
        else:
            parsed_inputs = inputs

        logger.info(f"{self.AGENT_ID} invoked. Mocking test generation for file: '{parsed_inputs.target_file_path_to_test}'")
        logger.info(f"Specification prompt (mock ignored): {parsed_inputs.code_specification_prompt}")
        if parsed_inputs.specification:
            logger.info(f"Received direct specification for tests: {parsed_inputs.specification}")

        # Generate a dummy test file path
        source_file = Path(parsed_inputs.target_file_path_to_test)
        # Example: chungoid-core/src/chungoid/cli.py -> chungoid-core/tests/unit/test_cli.py
        # This is a simplified mock path generation.
        if "src/chungoid/" in parsed_inputs.target_file_path_to_test:
            test_file_name = f"test_{source_file.stem}.py"
            # Try to place it in a plausible tests/unit structure if possible
            parts = list(source_file.parts)
            try:
                src_index = parts.index("src")
                # Replace 'src' with 'tests/unit' and keep the module structure
                test_path_parts = parts[:src_index] + ["tests", "unit"] + parts[src_index+2:]
                mock_test_file_path = Path(*test_path_parts).parent / test_file_name
            except ValueError:
                mock_test_file_path = source_file.parent / f"test_{source_file.name}" # Fallback
        else:
            mock_test_file_path = source_file.parent / f"test_{source_file.name}"

        logger.info(f"Mock generated test file path: {str(mock_test_file_path)}")
        
        return MockTestGeneratorAgentOutput(
            test_file_generated_path=str(mock_test_file_path),
            tests_generated_count=5 # Mock number of tests
        )

def get_agent_card_static() -> AgentCard:
    """Returns the static AgentCard for the MockTestGeneratorAgent."""
    return AgentCard(
        agent_id=MockTestGeneratorAgent.AGENT_ID,
        name=MockTestGeneratorAgent.AGENT_NAME,
        version=MockTestGeneratorAgent.VERSION,
        description=MockTestGeneratorAgent.DESCRIPTION,
        category=MockTestGeneratorAgent.CATEGORY,
        visibility=MockTestGeneratorAgent.VISIBILITY,
        input_schema=MockTestGeneratorAgentInput.model_json_schema(),
        output_schema=MockTestGeneratorAgentOutput.model_json_schema(),
    ) 