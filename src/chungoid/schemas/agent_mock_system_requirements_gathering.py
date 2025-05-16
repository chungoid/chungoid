from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel, Field

class MockSystemRequirementsGatheringAgentInput(BaseModel):
    goal_description: str = Field(..., description="The high-level goal description provided by the user or plan.")

class MockSystemRequirementsGatheringAgentOutput(BaseModel):
    command_specification_document: Dict[str, Any] = Field(
        ..., 
        description="A structured document detailing the specification for the command, derived from the goal."
    )
    # For this mock, the document will be simple:
    # {
    #   "command_name_suggestion": "derived_from_goal (e.g., show-config)",
    #   "purpose": "derived_from_goal",
    #   "full_goal_statement": "original_goal_description"
    # } 