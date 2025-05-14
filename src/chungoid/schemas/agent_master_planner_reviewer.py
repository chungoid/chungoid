from typing import Any, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.master_flow import MasterExecutionPlan # For type hint, can be str if circular

class ReviewerActionType(str, Enum):
    RETRY_STAGE_WITH_MODIFIED_INPUTS = "RETRY_STAGE_WITH_MODIFIED_INPUTS"
    MODIFY_PLAN_REPLACE_AGENT = "MODIFY_PLAN_REPLACE_AGENT"
    MODIFY_PLAN_ADD_STAGE_BEFORE = "MODIFY_PLAN_ADD_STAGE_BEFORE"
    MODIFY_PLAN_ADD_STAGE_AFTER = "MODIFY_PLAN_ADD_STAGE_AFTER"
    MODIFY_PLAN_REMOVE_STAGE = "MODIFY_PLAN_REMOVE_STAGE"
    ESCALATE_TO_USER = "ESCALATE_TO_USER" # Default for no-op
    ABORT_FLOW = "ABORT_FLOW"

class MasterPlannerReviewerInput(BaseModel):
    original_goal: str = Field(..., description="The initial high-level goal given to the MasterPlannerAgent.")
    failed_master_plan_json: str = Field(..., description="The JSON representation of the MasterExecutionPlan that failed.")
    failed_stage_id: str = Field(..., description="The ID of the stage within the plan that failed or triggered review.")
    error_details: AgentErrorDetails = Field(..., description="Details of the error that occurred.")
    full_context_snapshot: Dict[str, Any] = Field(..., description="The full execution context at the time of failure/pause.")
    # Potentially add:
    # current_run_id: str 
    # available_agents_summary: List[Dict] - To help reviewer pick alternatives

class MasterPlannerReviewerOutput(BaseModel):
    suggestion_id: str = Field(..., description="A unique ID for this review suggestion.")
    suggested_action: ReviewerActionType = Field(..., description="The type of action suggested by the reviewer.")
    action_details: Optional[Dict[str, Any]] = Field(None, description="Specific details for the suggested action. Structure depends on the action_type.")
    confidence_score: Optional[float] = Field(None, description="Reviewer's confidence in this suggestion (0.0 to 1.0).")
    justification: str = Field(..., description="Explanation for the suggested action.")

    # Example action_details structures:
    # For RETRY_STAGE_WITH_MODIFIED_INPUTS: {"target_stage_id": "stage_x", "new_inputs": {"param1": "value_retry"}}
    # For MODIFY_PLAN_REPLACE_AGENT: {"target_stage_id": "stage_x", "new_agent_id": "agent_y", "new_inputs": {"param_y": "val_y"}}
    # For ESCALATE_TO_USER: {"message_to_user": "The plan failed due to X. What do you want to do?"}
    # For ABORT_FLOW: {} 