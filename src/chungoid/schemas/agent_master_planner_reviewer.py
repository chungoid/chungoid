from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from chungoid.schemas.common_enums import FlowPauseStatus
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.master_flow import MasterExecutionPlan


class ReviewerActionType(Enum):
    """Defines actions the MasterPlannerReviewerAgent can suggest."""
    RETRY_STAGE_AS_IS = "RETRY_STAGE_AS_IS"
    RETRY_STAGE_WITH_MODIFIED_INPUT = "RETRY_STAGE_WITH_MODIFIED_INPUT"
    MODIFY_MASTER_PLAN = "MODIFY_MASTER_PLAN" # Suggest changes to the overall MasterExecutionPlan
    ESCALATE_TO_USER = "ESCALATE_TO_USER"
    PROCEED_AS_IS = "PROCEED_AS_IS" # If the 'failure' is deemed acceptable or a warning
    NO_ACTION_SUGGESTED = "NO_ACTION_SUGGESTED" # If the reviewer cannot determine a useful action


class MasterPlannerReviewerInput(BaseModel):
    """Input for the MasterPlannerReviewerAgent."""
    current_master_plan: MasterExecutionPlan = Field(..., description="The current MasterExecutionPlan that is being executed.")
    paused_run_details: Dict[str, Any] = Field(..., description="The PausedRunDetails dictionary from StateManager when the flow was paused.")
    # PausedRunDetails contains: run_id, flow_id (which is master_plan.id), paused_at_stage_id, timestamp,
    # status (FlowPauseStatus), context_snapshot, error_details, clarification_request.

    # For convenience, direct access to some crucial fields from PausedRunDetails:
    pause_status: FlowPauseStatus = Field(..., description="The reason the flow was paused.")
    paused_stage_id: str = Field(..., description="The ID of the master stage where the pause occurred.")
    triggering_error_details: Optional[AgentErrorDetails] = Field(None, description="Error details from the agent that triggered the pause, if applicable.")
    full_context_at_pause: Dict[str, Any] = Field(..., description="The full execution context snapshot at the time of pause.")


class MasterPlannerReviewerOutput(BaseModel):
    """Output from the MasterPlannerReviewerAgent."""
    suggestion_type: ReviewerActionType = Field(..., description="The type of action suggested by the reviewer.")
    suggestion_details: Optional[Dict[str, Any]] = Field(None, description="Details for the suggestion. E.g., for RETRY_STAGE_WITH_MODIFIED_INPUT, this could contain the new inputs. For MODIFY_MASTER_PLAN, this could be a patch or new plan structure.")
    confidence_score: Optional[float] = Field(None, description="Confidence in the suggestion (0.0 to 1.0).")
    reasoning: Optional[str] = Field(None, description="Explanation for the suggestion.")


# Example action_details structures:
# For RETRY_STAGE_WITH_MODIFIED_INPUTS: {"target_stage_id": "stage_x", "new_inputs": {"param1": "value_retry"}}
# For MODIFY_PLAN_REPLACE_AGENT: {"target_stage_id": "stage_x", "new_agent_id": "agent_y", "new_inputs": {"param_y": "val_y"}}
# For ESCALATE_TO_USER: {"message_to_user": "The plan failed due to X. What do you want to do?"}
# For ABORT_FLOW: {} 