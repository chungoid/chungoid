from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

# Assuming AgentCard might be defined elsewhere, e.g., utils.agent_registry
# For now, we can use a placeholder or assume it's a dict if needed for available_agents.
# from chungoid.utils.agent_registry import AgentCard 
from .user_goal_schemas import UserGoalRequest


class MasterPlannerInput(BaseModel):
    user_goal: Optional[str] = Field(None, description="The high-level user goal string. Optional if blueprint_doc_id is provided.")
    original_request: Optional[UserGoalRequest] = Field(
        None, 
        description="The original UserGoalRequest object, if available."
    )
    max_stages: int = Field(
        default=10, 
        description="Maximum number of stages the planner should generate."
    )
    # available_agents: Optional[List[AgentCard]] = Field( # Placeholder for now
    #     None,
    #     description="List of available agent cards for more dynamic planning in future versions."
    # )
    current_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional current execution context if planning is invoked mid-flow or for replanning."
    )
    project_id: Optional[str] = Field(None, description="Identifier for the current project, used when generating plan from blueprint.")
    blueprint_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the ProjectBlueprint.md to be transformed into a plan.")
    blueprint_reviewer_feedback_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the BlueprintReviewerAgent's feedback report.")

    @model_validator(mode='before')
    def check_goal_or_blueprint_provided(cls, values):
        user_goal, blueprint_doc_id = values.get('user_goal'), values.get('blueprint_doc_id')
        if not user_goal and not blueprint_doc_id:
            raise ValueError('Either user_goal or blueprint_doc_id must be provided.')
        if user_goal and blueprint_doc_id:
            # Allowing both for now, agent logic will decide priority or how to combine.
            # Could also raise ValueError('Provide either user_goal or blueprint_doc_id, not both.')
            pass # Or log a warning
        return values


class MasterPlannerOutput(BaseModel):
    master_plan_json: str = Field(
        ..., 
        description="A JSON string representing the generated MasterExecutionPlan."
    )
    confidence_score: Optional[float] = Field(
        None, 
        description="The planner's confidence in the generated plan (0.0 to 1.0)."
    )
    planner_notes: Optional[str] = Field(
        None, 
        description="Any notes or explanations from the planner about the generated plan."
    )
    error_message: Optional[str] = Field(
        None,
        description="If planning failed, an error message explaining why."
    )
    # NEW fields for PCMA storage
    generated_plan_artifact_id: Optional[str] = Field(None, description="The document ID of the stored MasterExecutionPlan artifact in ChromaDB.")
    stored_in_collection: Optional[str] = Field(None, description="The ChromaDB collection where the MasterExecutionPlan artifact was stored.")

    # Example of how to ensure master_plan_json is valid (can be added later if complex)
    # @validator('master_plan_json')
    # def check_master_plan_json(cls, v):
    #     try:
    #         # Attempt to parse it into MasterExecutionPlan to validate
    #         from .master_flow import MasterExecutionPlan # Circular import risk, handle carefully
    #         MasterExecutionPlan.model_validate_json(v)
    #     except Exception as e:
    #         raise ValueError(f'master_plan_json is not a valid MasterExecutionPlan: {e}')
    #     return v 