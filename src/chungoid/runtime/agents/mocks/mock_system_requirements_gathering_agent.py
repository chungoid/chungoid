from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from chungoid.schemas.common import AgentID
from chungoid.schemas.agent_mock_system_requirements_gathering import (
    MockSystemRequirementsGatheringAgentInput,
    MockSystemRequirementsGatheringAgentOutput,
)
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard
from chungoid.runtime.agents.agent_base import BaseAgent, InputSchema, OutputSchema

logger = logging.getLogger(__name__)

class MockSystemRequirementsGatheringAgent:
    """
    Mock agent for gathering system requirements based on a goal description.
    For the 'show-config' MVP, this will produce a predefined specification.
    """

    AGENT_ID: AgentID = "system_requirements_gathering_agent" # Matching the plan
    AGENT_NAME: str = "Mock System Requirements Gathering Agent"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Mocks system requirements gathering. For testing autonomous flows."
    CATEGORY: AgentCategory = AgentCategory.TESTING_MOCK
    VISIBILITY: AgentVisibility = AgentVisibility.INTERNAL

    async def invoke_async(
        self, 
        inputs: Dict[str, Any], 
        full_context: Optional[Dict[str, Any]] = None
    ) -> MockSystemRequirementsGatheringAgentOutput:
        try:
            parsed_inputs = MockSystemRequirementsGatheringAgentInput(**inputs)
        except Exception as e:
            logger.error(f"Failed to parse inputs for MockSystemRequirementsGatheringAgent: {e}")
            # For a mock agent, re-raising or returning a simple error output might be enough.
            # Consider raising a more specific AgentErrorDetails if this were a production agent.
            raise AgentErrorDetails(error_type="InputValidationError", message=f"Input parsing failed for MockSystemRequirementsGatheringAgent: {e}", stage_id=self.AGENT_ID, agent_id=self.AGENT_ID) from e

        logger.info(
            f"MockSystemRequirementsGatheringAgent invoked with goal: {parsed_inputs.goal_description}"
        )
        
        command_name_suggestion = "show-config"
        if "show-config" not in parsed_inputs.goal_description:
            command_name_suggestion = "unknown_command"
        
        if "display" in parsed_inputs.goal_description.lower() and "config" in parsed_inputs.goal_description.lower():
            command_name_suggestion = "show-config"

        spec_doc = {
            "command_name_suggestion": command_name_suggestion,
            "purpose": f"To address the goal: {parsed_inputs.goal_description}",
            "full_goal_statement": parsed_inputs.goal_description,
            "target_cli_group": "utils",
            "key_information_to_display": [
                "project_root",
                ".chungoid directory path",
                "state_manager_path",
                "master_flows_dir",
                "loaded ProjectConfig settings (summary or key parts)",
            ],
            "output_format_notes": "Clear, human-readable text. Key-value pairs or sections."
        }

        logger.info(f"MockSystemRequirementsGatheringAgent produced spec: {spec_doc}")
        return MockSystemRequirementsGatheringAgentOutput(
            command_specification_document=spec_doc
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Returns the static AgentCard for this agent."""
        return AgentCard(
            agent_id=MockSystemRequirementsGatheringAgent.AGENT_ID,
            name=MockSystemRequirementsGatheringAgent.AGENT_NAME,
            version=MockSystemRequirementsGatheringAgent.VERSION,
            description=MockSystemRequirementsGatheringAgent.DESCRIPTION,
            categories=[MockSystemRequirementsGatheringAgent.CATEGORY.value],
            visibility=MockSystemRequirementsGatheringAgent.VISIBILITY,
            input_schema=MockSystemRequirementsGatheringAgentInput.model_json_schema(),
            output_schema=MockSystemRequirementsGatheringAgentOutput.model_json_schema(),
        )

# Alias for consistency with other agents if preferred for imports in cli.py
get_agent_card_static = MockSystemRequirementsGatheringAgent.get_agent_card_static 