from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal, Annotated

import pydantic
from pydantic import BaseModel, Field, validator, ConfigDict

from chungoid.schemas.common_enums import FlowPauseStatus, ReviewerActionType
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec


class ReviewerModifyPlanAction(Enum):
    """Specific actions for MODIFY_MASTER_PLAN suggestion type."""
    REMOVE_STAGE = "remove_stage"
    MODIFY_STAGE_SPEC = "modify_stage_spec"
    # ADD_STAGE could be another if needed for direct add via MODIFY_MASTER_PLAN
    # Currently, ADD_CLARIFICATION_STAGE is a separate ReviewerActionType


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

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --- Details models for suggestion_details ---

class RetryStageWithChangesDetails(BaseModel):
    target_stage_id: str = Field(..., description="The ID of the stage to retry with changes.")
    # Using Dict for partial updates to avoid needing all MasterStageSpec fields.
    # The orchestrator will apply these as patches to the existing stage spec.
    changes_to_stage_spec: Dict[str, Any] = Field(
        ..., 
        description="A dictionary representing partial updates to the MasterStageSpec of the target stage. E.g., {'inputs': {'new_param': 'val'}, 'agent_id': 'new_agent_v2'}"
    )

class NewStageOutputMappingSpec(BaseModel):
    """Defines how an output from the newly added stage should be mapped to an input of a subsequent verification stage."""
    source_output_field: str = Field(..., description="The name of the output field from the new stage's result (e.g., 'clarification_response').")
    target_input_field_in_verification_stage: str = Field(..., description="The name of the input field in the designated verification stage where the new stage's output should be mapped (e.g., 'clarification_data').")


class AddClarificationStageDetails(BaseModel):
    """Details for adding a new clarification stage."""
    new_stage_spec: MasterStageSpec = Field(..., description="Full specification for the new stage to be added.")
    original_failed_stage_id: str = Field( # Renaming from insert_before_stage_id for clarity based on typical trigger
        ..., 
        description="The ID of the stage that originally failed or triggered the need for clarification, which the new stage will typically precede or be related to."
    )
    # Make sure NewStageOutputMappingSpec is defined before this line
    new_stage_output_to_map_to_verification_stage_input: Optional[NewStageOutputMappingSpec] = Field(
        None,
        description="Optional: Specifies how an output from the new stage should be mapped to an input of a designated verification stage (e.g., 'stage_C_verify')."
    )

class ModifyMasterPlanRemoveStageDetails(BaseModel):
    action: Literal["remove_stage"] = ReviewerModifyPlanAction.REMOVE_STAGE.value
    target_stage_id: str = Field(..., description="The ID of the master stage to remove from the plan.")

class ModifyMasterPlanModifyStageDetails(BaseModel):
    action: Literal["modify_stage_spec"] = ReviewerModifyPlanAction.MODIFY_STAGE_SPEC.value
    target_stage_id: str = Field(..., description="The ID of the master stage to modify.")
    updated_stage_spec: MasterStageSpec = Field(..., description="The new, complete specification for the target stage.")

# Discriminated union for MODIFY_MASTER_PLAN details
ModifyMasterPlanDetails = Annotated[
    Union[ModifyMasterPlanRemoveStageDetails, ModifyMasterPlanModifyStageDetails],
    Field(discriminator="action")
]


class MasterPlannerReviewerOutput(BaseModel):
    """Output from the MasterPlannerReviewerAgent."""
    suggestion_type: ReviewerActionType = Field(..., description="The type of action suggested by the reviewer.")
    suggestion_details: Optional[Union[
        RetryStageWithChangesDetails,
        AddClarificationStageDetails,
        ModifyMasterPlanDetails,
        Dict[str, Any] # For simpler actions like ESCALATE, RETRY_AS_IS, PROCEED_AS_IS if they have ad-hoc details
    ]] = Field(None, description="Specific details for the suggested action. Structure depends on suggestion_type.")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Reviewer's confidence in the suggestion (0.0 to 1.0).")
    reasoning: Optional[str] = Field(None, description="Explanation from the reviewer for their suggestion.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @pydantic.validator("suggestion_details", pre=True, always=True)
    def check_suggestion_details_type(cls, v, values):
        suggestion_type = values.get("suggestion_type")
        if not suggestion_type:
            # This case should ideally not happen if suggestion_type is validated first or always present
            return v

        if suggestion_type == ReviewerActionType.RETRY_STAGE_WITH_CHANGES:
            if not isinstance(v, RetryStageWithChangesDetails):
                # Attempt to parse if it's a dict, otherwise raise
                if isinstance(v, dict):
                    return RetryStageWithChangesDetails(**v)
                raise ValueError("suggestion_details must be RetryStageWithChangesDetails for RETRY_STAGE_WITH_CHANGES")
        elif suggestion_type == ReviewerActionType.ADD_CLARIFICATION_STAGE:
            if not isinstance(v, AddClarificationStageDetails):
                if isinstance(v, dict):
                    return AddClarificationStageDetails(**v)
                raise ValueError("suggestion_details must be AddClarificationStageDetails for ADD_CLARIFICATION_STAGE")
        elif suggestion_type == ReviewerActionType.MODIFY_MASTER_PLAN:
            if not isinstance(v, (ModifyMasterPlanRemoveStageDetails, ModifyMasterPlanModifyStageDetails)):
                if isinstance(v, dict) and "action" in v:
                    action_str = v.get("action")
                    try:
                        action_enum = ReviewerModifyPlanAction(action_str)
                        if action_enum == ReviewerModifyPlanAction.REMOVE_STAGE:
                            return ModifyMasterPlanRemoveStageDetails(**v)
                        elif action_enum == ReviewerModifyPlanAction.MODIFY_STAGE_SPEC:
                            return ModifyMasterPlanModifyStageDetails(**v)
                        else:
                            raise ValueError(f"Unknown action enum '{action_enum}' for MODIFY_MASTER_PLAN details")
                    except ValueError:
                        raise ValueError(f"Invalid action string '{action_str}' for MODIFY_MASTER_PLAN details")
                raise ValueError("suggestion_details for MODIFY_MASTER_PLAN must be ModifyMasterPlanRemoveStageDetails or ModifyMasterPlanModifyStageDetails, or a dict with a valid 'action' field.")
        # Add checks for other types if their details become strictly typed
        return v


# Updated example action_details structures (now mostly captured by the Pydantic models):
# For RETRY_STAGE_WITH_CHANGES: RetryStageWithChangesDetails(target_stage_id="stage_x", changes_to_stage_spec={"inputs": {"param1": "value_retry"}, "agent_id": "agent_y_v2"})
# For ADD_CLARIFICATION_STAGE: AddClarificationStageDetails(new_stage_spec=MasterStageSpec(...), insert_before_stage_id="stage_x")
# For MODIFY_MASTER_PLAN (remove_stage): ModifyMasterPlanRemoveStageDetails(action="remove_stage", target_stage_id="stage_to_remove")
# For ESCALATE_TO_USER: {"message_to_user": "The plan failed due to X. What do you want to do?"} # Remains Dict[str, Any]
# For other simple actions like RETRY_STAGE_AS_IS, PROCEED_AS_IS, NO_ACTION_SUGGESTED: None or empty Dict 