import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from chungoid.schemas.agent_test_generation import TestGenerationInput, TestGenerationOutput
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard

logger = logging.getLogger(__name__)

class MockTestGenerationAgentV1:
    """Mock Test Generation Agent (Version 1).

    Simulates the generation of tests to allow flows to proceed.
    """

    AGENT_ID = "core.mock_test_generation_agent_v1"
    AGENT_NAME = "Mock Test Generation Agent V1"
    AGENT_DESCRIPTION = "A mock agent that simulates test generation for flow progression."
    CATEGORY = AgentCategory.TEST_GENERATION
    VISIBILITY = AgentVisibility.INTERNAL  # Or PUBLIC if intended for wider mock use
    VERSION = "0.1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        logger.info(f"{self.AGENT_NAME} initialized with config: {self.config}")

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> TestGenerationOutput:
        logger.info(f"{self.AGENT_NAME} invoked with inputs: {inputs}")
        
        try:
            parsed_inputs = TestGenerationInput(**inputs)
        except Exception as e:
            logger.error(f"Failed to parse inputs for {self.AGENT_ID}: {e}")
            return TestGenerationOutput(
                status="FAILURE", 
                message=f"Input parsing failed: {e}",
                generated_tests=None,
                tests_generated_count=0
            )
        
        logger.debug(f"{self.AGENT_NAME} parsed_inputs: {parsed_inputs}")
        
        output_filepath_str: Optional[str] = None
        generated_test_content = f"# Mock test for code:\n# {parsed_inputs.command_code[:100]}...\n\ndef test_mock_command_execution():\n    assert True, \"Mock test passed!\"\n"

        if parsed_inputs.output_test_file_path:
            try:
                target_file = Path(parsed_inputs.output_test_file_path)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(generated_test_content)
                output_filepath_str = str(target_file.resolve())
                logger.info(f"Mock test file written to: {output_filepath_str}")
                message = f"Mock tests successfully written to {output_filepath_str}."
            except Exception as e:
                logger.error(f"Failed to write mock test file to {parsed_inputs.output_test_file_path}: {e}")
                message = f"Mock tests generated in memory, but failed to write to file: {e}"
        else:
            message = "Mock tests generated in memory (no output file path provided)."

        return TestGenerationOutput(
            status="SUCCESS",
            message=message,
            generated_tests=generated_test_content,
            generated_test_filepath=output_filepath_str,
            tests_generated_count=1
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Returns the static AgentCard for this agent."""
        return AgentCard(
            agent_id=MockTestGenerationAgentV1.AGENT_ID,
            name=MockTestGenerationAgentV1.AGENT_NAME,
            description=MockTestGenerationAgentV1.AGENT_DESCRIPTION,
            categories=[MockTestGenerationAgentV1.CATEGORY.value],
            visibility=MockTestGenerationAgentV1.VISIBILITY.value,
            capability_profile={
                "language": "python",
                "framework": "pytest",
                "mock_agent": True  # To match preferences and indicate it's a mock
            },
            input_schema=TestGenerationInput.model_json_schema(),
            output_schema=TestGenerationOutput.model_json_schema(),
            version=MockTestGenerationAgentV1.VERSION
        )

# Alias for consistency if used elsewhere, though registry typically calls the static method
get_agent_card_static = MockTestGenerationAgentV1.get_agent_card_static 