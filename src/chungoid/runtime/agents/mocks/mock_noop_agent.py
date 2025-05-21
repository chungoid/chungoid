from __future__ import annotations

import logging
from typing import Any, Dict, Optional, ClassVar

from pydantic import BaseModel, Field

from chungoid.runtime.agents.agent_base import BaseAgent, InputSchema, OutputSchema
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility

logger = logging.getLogger(__name__)

class NoOpInput(BaseModel):
    message: Optional[str] = Field(None, description="An optional message that can be passed to the NoOp agent.")
    data: Optional[Dict[str, Any]] = Field(None, description="Optional arbitrary data.")

class NoOpOutput(BaseModel):
    status: str = Field("SUCCESS", description="Status of the No-Op operation.")
    message: str = Field("NoOpAgent executed successfully.", description="A confirmation message.")
    received_message: Optional[str] = Field(None, description="The message received by the agent, if any.")

class MockNoOpAgent(BaseAgent[NoOpInput, NoOpOutput]):
    AGENT_ID: ClassVar[str] = "NoOpAgent_v1"
    AGENT_NAME: ClassVar[str] = "Mock No-Op Agent"
    VERSION: ClassVar[str] = "1.0.0"
    DESCRIPTION: ClassVar[str] = "A mock agent that performs no operation and always returns success. Used for testing and as a placeholder."
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.TESTING_MOCK
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL # Or PUBLIC if direct invocation is desired

    def __init__(self, system_context: Optional[Dict[str, Any]] = None):
        super().__init__(system_context=system_context)
        logger.info(f"{self.AGENT_NAME} ({self.AGENT_ID}) initialized.")

    async def invoke_async(
        self, inputs: NoOpInput, full_context: Optional[Dict[str, Any]] = None
    ) -> NoOpOutput:
        logger.info(f"{self.AGENT_NAME} ({self.AGENT_ID}) invoked.")
        
        # Parse the input dictionary into the NoOpInput model
        parsed_inputs: NoOpInput
        if isinstance(inputs, dict):
            parsed_inputs = NoOpInput(**inputs)
        elif isinstance(inputs, NoOpInput): # Already an instance
            parsed_inputs = inputs
        else:
            logger.warning(f"Unexpected input type for MockNoOpAgent: {type(inputs)}. Attempting to cast.")
            try:
                parsed_inputs = NoOpInput(**dict(inputs)) # Try to convert to dict then parse
            except Exception as e:
                logger.error(f"Failed to parse inputs for MockNoOpAgent: {e}. Falling back to default NoOpInput.")
                parsed_inputs = NoOpInput()

        if parsed_inputs.message:
            logger.info(f"Received message: {parsed_inputs.message}")
        if parsed_inputs.data:
            logger.info(f"Received data: {parsed_inputs.data}")
        
        return NoOpOutput(received_message=parsed_inputs.message)

    @classmethod
    def get_agent_card_static(cls) -> AgentCard:
        return AgentCard(
            agent_id=cls.AGENT_ID,
            name=cls.AGENT_NAME,
            version=cls.VERSION,
            description=cls.DESCRIPTION,
            category=cls.CATEGORY.value,
            visibility=cls.VISIBILITY.value,
            input_schema=NoOpInput.model_json_schema(),
            output_schema=NoOpOutput.model_json_schema(),
        )

ALL_MOCK_AGENTS_IN_THIS_FILE = [MockNoOpAgent] # If other mocks were in this file

def get_fallback_map_for_this_file() -> Dict[str, Any]:
    return {
        MockNoOpAgent.AGENT_ID: MockNoOpAgent,
    } 