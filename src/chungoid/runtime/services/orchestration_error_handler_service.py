"""
Service for handling errors during MasterExecutionPlan orchestration.
"""
import logging
import traceback
from typing import Dict, Any, Optional, cast
import asyncio
import json

from pydantic import BaseModel, Field, ConfigDict, ValidationError

# Project-specific imports (adjust paths as necessary)
from chungoid.schemas.common_enums import StageStatus, FlowPauseStatus, OnFailureAction, ReviewerActionType
from chungoid.schemas.errors import AgentErrorDetails, OrchestratorError
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec, MasterStageFailurePolicy
from chungoid.schemas.orchestration import SharedContext
from chungoid.schemas.agent_master_planner_reviewer import (
    MasterPlannerReviewerInput,
    MasterPlannerReviewerOutput,
    RetryStageWithChangesDetails,
)
from chungoid.schemas.metrics import MetricEvent, MetricEventType # Assuming MetricEventType is an enum
from chungoid.schemas.flows import PausedRunDetails # For constructing input to reviewer

from chungoid.utils.agent_resolver import AgentProvider # For reviewer agent
from chungoid.utils.state_manager import StateManager
from chungoid.utils.metrics_store import MetricsStore

# Constants for next_stage signals (mirroring orchestrator.py)
NEXT_STAGE_END_SUCCESS = "__END_SUCCESS__"
NEXT_STAGE_END_FAILURE = "__END_FAILURE__"


class OrchestrationErrorResult(BaseModel):
    """
    Represents the outcome of the error handling process for a stage.
    """
    next_stage_to_execute: Optional[str] = Field(
        None,
        description="The ID of the next stage to execute. Can be current stage for retry, "
                    "a new stage, a terminal signal like NEXT_STAGE_END_FAILURE, or None if pausing."
    )
    updated_agent_error_details: AgentErrorDetails = Field(
        ...,
        description="The error details, potentially updated by reviewer or enriched by the handler."
    )
    flow_pause_status: FlowPauseStatus = Field(
        ...,
        description="Indicates if the flow should pause and why."
    )
    reviewer_output: Optional[MasterPlannerReviewerOutput] = Field(
        None,
        description="Output from the reviewer agent, if invoked."
    )
    handled_attempt_number: int = Field(
        ...,
        description="The attempt number of the stage error that was just processed by the handler."
    )
    modified_stage_inputs: Optional[Dict[str, Any]] = Field(
        None,
        description="New inputs for the stage if reviewer suggested RETRY_STAGE_WITH_CHANGES."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OrchestrationErrorHandlerService:
    """
    Handles errors encountered during the execution of stages in a MasterExecutionPlan.
    This includes managing retries, invoking reviewer agents, and interacting with
    StateManager and MetricsStore.
    """

    def __init__(
        self,
        logger: logging.Logger,
        agent_provider: AgentProvider,
        state_manager: StateManager,
        metrics_store: MetricsStore,
        master_planner_reviewer_agent_id: str,
        default_on_failure_action: OnFailureAction,
        default_agent_retries: int,
    ):
        self.logger = logger
        self.agent_provider = agent_provider
        self.state_manager = state_manager
        self.metrics_store = metrics_store
        self.master_planner_reviewer_agent_id = master_planner_reviewer_agent_id
        self.default_on_failure_action = default_on_failure_action
        self.default_agent_retries = default_agent_retries
        self.logger.info(
            f"OrchestrationErrorHandlerService initialized. Reviewer: {self.master_planner_reviewer_agent_id}, "
            f"Default Failure: {self.default_on_failure_action.value}, Default Retries: {self.default_agent_retries}"
        )

    def _emit_metric(
        self,
        event_type: MetricEventType,
        flow_id: str,
        run_id: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Helper to create and add a metric event."""
        metric_data_payload = data or {}
        # Combine kwargs directly into the metric_data_payload, ensuring they don't overwrite core fields if passed.
        # Core fields like flow_id, run_id are passed as direct MetricEvent args later.
        
        # Filter out None values from kwargs to keep events clean if they are added to data
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        metric_data_payload.update(filtered_kwargs)


        # Prepare arguments for MetricEvent, ensuring required ones are present
        # and optional ones from kwargs are correctly placed.
        metric_args = {
            "flow_id": flow_id,
            "run_id": run_id,
        }
        # Add stage_id and agent_id from kwargs if present (these are common)
        if "stage_id" in kwargs:
            metric_args["stage_id"] = kwargs["stage_id"]
        if "agent_id" in kwargs:
            metric_args["agent_id"] = kwargs["agent_id"]
        
        try:
            event = MetricEvent(
                event_type=event_type,
                data=metric_data_payload,
                **metric_args # Pass core args directly
            )
            self.metrics_store.add_event(event)
        except Exception as e:
            self.logger.error(
                f"Failed to emit metric event {event_type} for run {run_id}, flow {flow_id}: {e}",
                exc_info=True,
            )

    def standardize_error(
        self,
        error: Exception,
        agent_id: Optional[str],
        stage_id: str,
        resolved_inputs_at_failure: Optional[Dict[str, Any]],
        default_can_retry: bool = False
    ) -> AgentErrorDetails:
        """
        Standardizes various exceptions into AgentErrorDetails.
        If the error is an OrchestrationError caused by a Pydantic ValidationError,
        it extracts details from the ValidationError.
        """
        tb_str = traceback.format_exc()

        # Handle OrchestrationError caused by Pydantic ValidationError specifically
        if isinstance(error, OrchestratorError) and isinstance(error.__cause__, ValidationError):
            validation_error = cast(ValidationError, error.__cause__)
            error_type_str = "ValidationError" # Standardize to this string
            # Construct a detailed message
            detailed_message = f"Pydantic ValidationError for agent {agent_id or 'Unknown'} in stage {stage_id}: {str(validation_error)}. Details: {json.dumps(validation_error.errors(), indent=2)}"
            
            return AgentErrorDetails(
                agent_id=agent_id or "OrchestratorCausedValidationError", # Could be orchestrator trying to prep inputs
                stage_id=stage_id,
                error_type=error_type_str,
                message=detailed_message,
                traceback=tb_str, # Include full traceback for context
                can_retry=getattr(error, 'can_retry', False), # OrchestrationError might have can_retry
                resolved_inputs_at_failure=resolved_inputs_at_failure or {},
                is_pydantic_validation_error=True, # Explicitly flag it
                raw_pydantic_validation_errors=validation_error.errors() # Add raw errors
            )

        if isinstance(error, AgentErrorDetails):
            # Augment existing AgentErrorDetails if necessary
            if not error.agent_id and agent_id:
                error.agent_id = agent_id
            if not error.stage_id and stage_id:
                error.stage_id = stage_id
            if error.resolved_inputs_at_failure is None and resolved_inputs_at_failure is not None:
                 error.resolved_inputs_at_failure = resolved_inputs_at_failure
            if not error.traceback: # Add traceback if missing
                error.traceback = tb_str
            return error
        
        if isinstance(error, OrchestratorError) and isinstance(error.__cause__, AgentErrorDetails):
            cause_error_details = cast(AgentErrorDetails, error.__cause__)
            if not cause_error_details.agent_id and agent_id:
                cause_error_details.agent_id = agent_id
            if not cause_error_details.stage_id and stage_id:
                cause_error_details.stage_id = stage_id
            if cause_error_details.resolved_inputs_at_failure is None and resolved_inputs_at_failure is not None:
                cause_error_details.resolved_inputs_at_failure = resolved_inputs_at_failure
            if not cause_error_details.traceback: # Add traceback if missing
                 cause_error_details.traceback = tb_str
            return cause_error_details

        error_type_str = error.__class__.__name__
        final_agent_id = agent_id
        can_retry_flag = default_can_retry

        if isinstance(error, OrchestratorError):
            final_agent_id = agent_id or "Orchestrator"
            # Orchestrator errors are generally not retriable unless specified
            can_retry_flag = getattr(error, 'can_retry', False) 
        else: # Generic Python exception
            final_agent_id = agent_id or "UnknownAgent"
            # Generic exceptions are generally not retriable
            can_retry_flag = getattr(error, 'can_retry', False) 

        return AgentErrorDetails(
            agent_id=final_agent_id,
            stage_id=stage_id,
            error_type=error_type_str,
            message=str(error),
            traceback=tb_str,
            can_retry=can_retry_flag,
            resolved_inputs_at_failure=resolved_inputs_at_failure or {}
        )

    async def _invoke_reviewer(
        self,
        run_id: str,
        flow_id: str,
        current_stage_name: str,
        agent_error_details_obj: AgentErrorDetails,
        current_shared_context: SharedContext,
        current_plan: MasterExecutionPlan
    ) -> Optional[MasterPlannerReviewerOutput]:
        """
        Invokes the master planner reviewer agent to get a suggestion on how to handle an error.
        """
        if not self.master_planner_reviewer_agent_id:
            self.logger.warning(
                f"Run {run_id}, Flow {flow_id}: Master planner reviewer agent ID is not configured. "
                f"Cannot invoke reviewer for stage '{current_stage_name}'."
            )
            return None

        try:
            reviewer_agent_callable = self.agent_provider.get(
                identifier=self.master_planner_reviewer_agent_id 
            )
            if not asyncio.iscoroutinefunction(reviewer_agent_callable):
                self.logger.warning(f"Run {run_id}, Flow {flow_id}: Reviewer agent '{self.master_planner_reviewer_agent_id}' is not an async function. This might cause issues if it's long-running.")

        except Exception as e_get_agent:
            self.logger.error(
                f"Run {run_id}, Flow {flow_id}: Failed to get reviewer agent '{self.master_planner_reviewer_agent_id}': {e_get_agent}",
                exc_info=True
            )
            return None

        # Construct inputs for MasterPlannerReviewerInput
        # Assuming FlowPauseStatus.PAUSED_FOR_AGENT_FAILURE_IN_MASTER is appropriate here
        pause_status_for_reviewer = FlowPauseStatus.PAUSED_FOR_AGENT_FAILURE_IN_MASTER 
        # error_details_dict_for_pause = agent_error_details_obj.to_dict() if agent_error_details_obj else None # OLD: Incorrectly converted to dict

        # PausedRunDetails needs a timestamp, which should be current time - NO, PausedRunDetails does not have timestamp field
        # from datetime import datetime, timezone # Local import for now
        # timestamp_now_iso = datetime.now(timezone.utc).isoformat()

        paused_run_details_dict = {
            "run_id": run_id,
            "flow_id": flow_id,
            "paused_stage_id": current_stage_name,
            # "timestamp": timestamp_now_iso, # REMOVED: PausedRunDetails does not have this field
            "status": pause_status_for_reviewer.value, # Ensure .value is used for enum
            "current_master_plan_snapshot": current_plan.model_copy(deep=True) if current_plan else None, # ADDED: ensure plan is passed
            "context_snapshot_ref": None,  # Snapshotting is orchestrator's concern, not error handler directly
            "error_details": agent_error_details_obj, # MODIFIED: Pass the instance, not a dict
            "clarification_details": None, # MODIFIED: Renamed from clarification_request and set to None for error review
            "last_stage_attempt_number": None # ADDED: Placeholder, this might need to be the actual attempt number
        }
        
        # Populate last_stage_id and last_stage_attempt_number if available from agent_error_details_obj
        # (though agent_error_details_obj.stage_id is current_stage_name, so last_stage_id can be that)
        # For attempt number, it was passed into handle_stage_execution_error and then to _invoke_reviewer
        # Let's assume it's not directly available on agent_error_details_obj unless it was specifically set there.
        # The orchestrator provides attempt_number to handle_stage_execution_error.
        # This dict is for PausedRunDetails, which is then part of MasterPlannerReviewerInput.
        # MasterPlannerReviewerInput itself does not directly take attempt_number, but uses paused_run_details.
        # It might be better if _invoke_reviewer took `attempt_number` if it needs to be in `paused_run_details_dict` accurately.
        # For now, keeping it None in this specific dict as its primary consumer (MasterPlannerReviewerInput) doesn't seem to use it from here.

        # Create PausedRunDetails from dict for validation and type safety if needed by reviewer input schema
        try:
            paused_run_details_obj = PausedRunDetails(**paused_run_details_dict)
        except Exception as prd_val_err: # Catch Pydantic ValidationError or others
            self.logger.error(f"Run {run_id}, Flow {flow_id}: Validation error creating PausedRunDetails for reviewer input: {prd_val_err}", exc_info=True)
            return None

        full_context_at_pause_dict = current_shared_context.model_dump(warnings=False)

        reviewer_input_data = {
            "current_master_plan": current_plan,
            "paused_run_details": paused_run_details_obj.model_dump(warnings=False) if paused_run_details_obj else None, # MODIFIED: Dump the object to dict
            "pause_status": pause_status_for_reviewer, 
            "paused_stage_id": current_stage_name,
            "triggering_error_details": agent_error_details_obj,
            "full_context_at_pause": full_context_at_pause_dict
        }
        
        try:
            reviewer_input = MasterPlannerReviewerInput(**reviewer_input_data)
        except Exception as r_input_val_err: # Catch Pydantic ValidationError or others
            self.logger.error(f"Run {run_id}, Flow {flow_id}: Validation error creating MasterPlannerReviewerInput: {r_input_val_err}", exc_info=True)
            return None
        
        self.logger.info(
            f"Run {run_id}, Flow {flow_id}: Invoking reviewer agent '{self.master_planner_reviewer_agent_id}' for stage '{current_stage_name}'."
        )
        self._emit_metric(
            MetricEventType.REVIEWER_AGENT_INVOCATION_START, 
            flow_id, 
            run_id, 
            stage_id=current_stage_name, 
            agent_id=self.master_planner_reviewer_agent_id
        )

        suggestion_output: Optional[MasterPlannerReviewerOutput] = None
        try:
            # agent_provider.get() for MasterPlannerReviewerAgent is expected to return its invoke_async method directly.
            # This method is already a coroutine function.
            # The signature is: async def invoke_async(self, input_payload: MasterPlannerReviewerInput)
            suggestion_output = await reviewer_agent_callable(input_payload=reviewer_input) # MODIFIED: Direct call with correct arg name

            if not isinstance(suggestion_output, MasterPlannerReviewerOutput):
                self.logger.error(
                    f"Run {run_id}, Flow {flow_id}: Reviewer agent '{self.master_planner_reviewer_agent_id}' returned unexpected type: "
                    f"{type(suggestion_output)}. Expected MasterPlannerReviewerOutput."
                )
                self._emit_metric(
                    MetricEventType.REVIEWER_AGENT_INVOCATION_END, flow_id, run_id, 
                    stage_id=current_stage_name, agent_id=self.master_planner_reviewer_agent_id, 
                    data={"status": "ERROR_UNEXPECTED_OUTPUT_TYPE"}
                )
                return None
            
            self.logger.info(
                f"Run {run_id}, Flow {flow_id}: Reviewer agent '{self.master_planner_reviewer_agent_id}' provided suggestion: "
                f"{suggestion_output.suggestion_type.value if suggestion_output else 'None'}"
            )
            self._emit_metric(
                MetricEventType.REVIEWER_AGENT_INVOCATION_END, flow_id, run_id, 
                stage_id=current_stage_name, agent_id=self.master_planner_reviewer_agent_id, 
                data={"status": "SUCCESS", "suggestion_type": suggestion_output.suggestion_type.value if suggestion_output else "None"}
            )
            return suggestion_output

        except Exception as e_invoke_reviewer:
            self.logger.error(
                f"Run {run_id}, Flow {flow_id}: Error invoking reviewer agent '{self.master_planner_reviewer_agent_id}' "
                f"for stage '{current_stage_name}': {e_invoke_reviewer}", 
                exc_info=True
            )
            self._emit_metric(
                MetricEventType.REVIEWER_AGENT_INVOCATION_END, flow_id, run_id, 
                stage_id=current_stage_name, agent_id=self.master_planner_reviewer_agent_id, 
                data={"status": "ERROR_INVOCATION_FAILED", "error": str(e_invoke_reviewer)}
            )
            return None

    async def handle_stage_execution_error(
        self,
        current_stage_name: str,
        flow_id: str,
        run_id: str,
        current_plan: MasterExecutionPlan,
        error: Exception, # Original exception from agent/orchestrator
        agent_id_that_erred: Optional[str], # ID of the agent that was supposed to run
        attempt_number: int, # Current attempt for this stage
        shared_context_at_error: SharedContext,
        resolved_inputs_at_failure: Optional[Dict[str, Any]]
    ) -> OrchestrationErrorResult:
        """
        Handles a stage execution error, including retries, reviewer invocation,
        and determining the next course of action for the orchestrator.
        """
        # 1. Standardize Error
        agent_error_obj = self.standardize_error(
            error,
            agent_id_that_erred,
            current_stage_name,
            resolved_inputs_at_failure
        )

        # 2. Log Initial Error & Metric
        self.logger.warning(
            f"Run {run_id}, Flow {flow_id}: Handling error for stage '{current_stage_name}' "
            f"(Agent: {agent_error_obj.agent_id}, Attempt: {attempt_number}). "
            f"Error: {agent_error_obj.error_type} - {agent_error_obj.message}"
        )
        self._emit_metric(
            MetricEventType.MASTER_STAGE_ERROR_ENCOUNTERED,
            flow_id,
            run_id,
            stage_id=current_stage_name,
            agent_id=agent_error_obj.agent_id,
            data=agent_error_obj.to_dict()
        )

        # 3. Get Stage Spec & Max Retries
        stage_spec = current_plan.stages.get(current_stage_name)
        if not stage_spec:
            self.logger.error(
                f"Run {run_id}, Flow {flow_id}: Stage spec for '{current_stage_name}' not found "
                f"in _handle_stage_error. Cannot determine retry/failure policy."
            )
            # Update error object to reflect this critical failure
            agent_error_obj.error_type = "StageSpecNotFound"
            agent_error_obj.message = (
                f"Critical: Stage specification for '{current_stage_name}' was not found in the execution plan."
            )
            agent_error_obj.agent_id = "Orchestrator" # Error is from orchestrator context now

            # Record this critical failure with state_manager if possible
            # (record_stage_run might not be appropriate if stage_spec itself is missing)
            # For now, we assume StateManager can handle a general run update if stage-specific recording fails.
            # This part might need refinement in StateManager interaction for such edge cases.
            return OrchestrationErrorResult(
                next_stage_to_execute=NEXT_STAGE_END_FAILURE,
                updated_agent_error_details=agent_error_obj,
                flow_pause_status=FlowPauseStatus.PAUSED_ON_ERROR, # A critical error should pause
                reviewer_output=None,
                handled_attempt_number=attempt_number,
                modified_stage_inputs=None
            )
        
        max_retries_for_stage = stage_spec.max_retries if stage_spec.max_retries is not None \
                                else self.default_agent_retries

        # 4. Record Failed Attempt with StateManager
        try:
            # Convert AgentErrorDetails to dict for StateManager
            error_details_dict = agent_error_obj.to_dict() if agent_error_obj else None
            
            self.state_manager.record_stage_end(
                run_id=run_id,
                flow_id=flow_id,
                stage_id=current_stage_name,
                status=StageStatus.COMPLETED_FAILURE.value,
                outputs=None,
                error_details=error_details_dict
            )
        except Exception as e_state_rec:
            self.logger.error(f"Run {run_id}, Flow {flow_id}: Failed to record stage end with StateManager for stage '{current_stage_name}': {e_state_rec}", exc_info=True)
            # Continue with error handling, but this is a sign of deeper issues.

        # 5. Retry Logic
        # Note: Orchestrator loop increments attempt number for the *next* try.
        # Here, `attempt_number` is the number of the attempt that just failed.
        if agent_error_obj.can_retry and attempt_number <= max_retries_for_stage:
            self.logger.info(
                f"Run {run_id}, Flow {flow_id}: Stage '{current_stage_name}' will be retried. "
                f"Next attempt will be {attempt_number + 1} (max configured: {max_retries_for_stage +1} including initial)."
            )
            self._emit_metric(
                MetricEventType.MASTER_STAGE_RETRY_ATTEMPT,
                flow_id,
                run_id,
                stage_id=current_stage_name,
                agent_id=agent_error_obj.agent_id,
                data={"attempt_number_for_next_try": attempt_number + 1, "max_attempts_total": max_retries_for_stage + 1}
            )
            return OrchestrationErrorResult(
                next_stage_to_execute=current_stage_name, # Retry current stage
                updated_agent_error_details=agent_error_obj,
                flow_pause_status=FlowPauseStatus.NOT_PAUSED,
                reviewer_output=None,
                handled_attempt_number=attempt_number,
                modified_stage_inputs=None
            )

        # 6. Non-Retriable / Retries Exhausted
        self.logger.warning(
            f"Run {run_id}, Flow {flow_id}: Stage '{current_stage_name}' retries exhausted "
            f"(handled {attempt_number} attempts, max_retries: {max_retries_for_stage}) "
            f"or error not retriable (can_retry={agent_error_obj.can_retry}). Proceeding to on_failure policy."
        )
        
        on_failure_policy = stage_spec.on_failure or MasterStageFailurePolicy(action=self.default_on_failure_action)
        reviewer_output: Optional[MasterPlannerReviewerOutput] = None
        modified_inputs_from_reviewer: Optional[Dict[str, Any]] = None

        # 7. Handle OnFailureAction.FAIL_MASTER_FLOW
        if on_failure_policy.action == OnFailureAction.FAIL_MASTER_FLOW:
            self.logger.info(
                f"Run {run_id}, Flow {flow_id}: Stage '{current_stage_name}' on_failure is FAIL_MASTER_FLOW."
            )
            return OrchestrationErrorResult(
                next_stage_to_execute=NEXT_STAGE_END_FAILURE,
                updated_agent_error_details=agent_error_obj,
                flow_pause_status=FlowPauseStatus.NOT_PAUSED, # Not paused, but failing
                reviewer_output=None,
                handled_attempt_number=attempt_number,
                modified_stage_inputs=None
            )

        # 8. Handle OnFailureAction.INVOKE_REVIEWER
        if on_failure_policy.action == OnFailureAction.INVOKE_REVIEWER:
            self.logger.info(
                f"Run {run_id}, Flow {flow_id}: Stage '{current_stage_name}' on_failure is INVOKE_REVIEWER. "
                f"Invoking reviewer agent '{self.master_planner_reviewer_agent_id}'."
            )
            reviewer_output = await self._invoke_reviewer(
                run_id=run_id,
                flow_id=flow_id,
                current_stage_name=current_stage_name,
                agent_error_details_obj=agent_error_obj,
                current_shared_context=shared_context_at_error,
                current_plan=current_plan
            )

            if not reviewer_output:
                self.logger.warning(
                    f"Run {run_id}, Flow {flow_id}: Reviewer agent for stage '{current_stage_name}' "
                    f"failed or returned no suggestion. Defaulting to PAUSE for user intervention."
                )
                return OrchestrationErrorResult(
                    next_stage_to_execute=None, # Pause
                    updated_agent_error_details=agent_error_obj,
                    flow_pause_status=FlowPauseStatus.USER_INTERVENTION_REQUIRED,
                    reviewer_output=None, # No valid output from reviewer
                    handled_attempt_number=attempt_number,
                    modified_stage_inputs=None
                )

            suggestion_type = reviewer_output.suggestion_type
            suggestion_details = reviewer_output.suggestion_details
            self.logger.info(f"Run {run_id}, Flow {flow_id}: Reviewer for stage '{current_stage_name}' suggested: {suggestion_type.value}")

            if suggestion_type == ReviewerActionType.RETRY_STAGE_AS_IS:
                return OrchestrationErrorResult(
                    next_stage_to_execute=current_stage_name, # Retry current stage
                    updated_agent_error_details=agent_error_obj, # Error details remain the same
                    flow_pause_status=FlowPauseStatus.NOT_PAUSED,
                    reviewer_output=reviewer_output,
                    handled_attempt_number=attempt_number,
                    modified_stage_inputs=None
                )
            
            elif suggestion_type == ReviewerActionType.RETRY_STAGE_WITH_CHANGES:
                if isinstance(suggestion_details, RetryStageWithChangesDetails) and \
                   suggestion_details.target_stage_id == current_stage_name and \
                   isinstance(suggestion_details.changes_to_stage_spec, dict):
                    
                    self.logger.info(
                        f"Run {run_id}, Flow {flow_id}: Applying reviewer changes to inputs for stage "
                        f"'{current_stage_name}': {modified_inputs_from_reviewer}"
                    )
                    
                    # Extract the actual agent inputs from changes_to_stage_spec
                    # changes_to_stage_spec may contain {"inputs": {...}} but we need just the {...} part
                    if "inputs" in suggestion_details.changes_to_stage_spec:
                        modified_inputs_from_reviewer = suggestion_details.changes_to_stage_spec["inputs"]
                        self.logger.info(
                            f"Run {run_id}, Flow {flow_id}: Extracted agent inputs from changes_to_stage_spec: "
                            f"{modified_inputs_from_reviewer}"
                        )
                    else:
                        # Fallback: assume the entire changes_to_stage_spec is the inputs
                        modified_inputs_from_reviewer = suggestion_details.changes_to_stage_spec
                        self.logger.warning(
                            f"Run {run_id}, Flow {flow_id}: No 'inputs' key found in changes_to_stage_spec, "
                            f"using entire dict as agent inputs: {modified_inputs_from_reviewer}"
                        )
                    
                    # The orchestrator loop will apply these inputs to stage_spec if this result is returned.
                    return OrchestrationErrorResult(
                        next_stage_to_execute=current_stage_name, # Retry current stage
                        updated_agent_error_details=agent_error_obj, # Error might be considered "resolved" by new inputs
                        flow_pause_status=FlowPauseStatus.NOT_PAUSED,
                        reviewer_output=reviewer_output,
                        handled_attempt_number=attempt_number,
                        modified_stage_inputs=modified_inputs_from_reviewer
                    )
                else:
                    self.logger.warning(
                        f"Run {run_id}, Flow {flow_id}: Reviewer suggestion RETRY_STAGE_WITH_CHANGES for "
                        f"'{current_stage_name}' had invalid details. Expected RetryStageWithChangesDetails "
                        f"with matching target_stage_id and dict for changes_to_stage_spec. "
                        f"Details: {suggestion_details}. Defaulting to PAUSE."
                    )
                    # Fall through to default pause logic at the end

            elif suggestion_type == ReviewerActionType.PROCEED_AS_IS:
                # The orchestrator's loop will handle the specifics of PROCEED_AS_IS,
                # including advancing to the next stage and marking the current one with warnings.
                # This result just signals that the error handler is not pausing or retrying.
                return OrchestrationErrorResult(
                    next_stage_to_execute=current_stage_name, # Signal to orchestrator to determine actual next stage
                    updated_agent_error_details=agent_error_obj,
                    flow_pause_status=FlowPauseStatus.NOT_PAUSED,
                    reviewer_output=reviewer_output,
                    handled_attempt_number=attempt_number,
                    modified_stage_inputs=None
                )

            # For ESCALATE_TO_USER, MODIFY_MASTER_PLAN, NO_ACTION_SUGGESTED, or unhandled reviewer suggestions:
            self.logger.info(
                f"Run {run_id}, Flow {flow_id}: Reviewer suggested '{suggestion_type.value}' for "
                f"stage '{current_stage_name}'. This action typically requires user intervention. Pausing flow."
            )
            pause_status_map = {
                ReviewerActionType.ESCALATE_TO_USER: FlowPauseStatus.USER_INTERVENTION_REQUIRED,
                ReviewerActionType.MODIFY_MASTER_PLAN: FlowPauseStatus.USER_INTERVENTION_REQUIRED, # User needs to approve plan changes
                ReviewerActionType.NO_ACTION_SUGGESTED: FlowPauseStatus.USER_INTERVENTION_REQUIRED,
            }
            final_pause_status = pause_status_map.get(suggestion_type, FlowPauseStatus.USER_INTERVENTION_REQUIRED)
            return OrchestrationErrorResult(
                next_stage_to_execute=None, # Pause
                updated_agent_error_details=agent_error_obj,
                flow_pause_status=final_pause_status,
                reviewer_output=reviewer_output,
                handled_attempt_number=attempt_number,
                modified_stage_inputs=None
            )

        # 9. Default Pause (if on_failure_policy was, e.g., PAUSE_FOR_USER_REVIEW, or unhandled INVOKE_REVIEWER outcome)
        self.logger.info(
            f"Run {run_id}, Flow {flow_id}: Stage '{current_stage_name}' "
            f"on_failure policy is '{on_failure_policy.action.value if on_failure_policy else 'None'}' "
            f"or reviewer path led here. Defaulting to PAUSE for user intervention."
        )
        return OrchestrationErrorResult(
            next_stage_to_execute=None, # Pause
            updated_agent_error_details=agent_error_obj,
            flow_pause_status=FlowPauseStatus.USER_INTERVENTION_REQUIRED,
            reviewer_output=reviewer_output, # Could be None if reviewer wasn't invoked
            handled_attempt_number=attempt_number,
            modified_stage_inputs=None
        )

    # ... _invoke_reviewer implementation will go here ... 