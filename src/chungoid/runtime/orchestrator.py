"""Minimal execution runtime skeleton (Phase-6 scaffold).

This module will gradually evolve into the production‐grade orchestrator that
executes Flow YAML graphs.  For now we expose a *SyncOrchestrator* that can run
very simple flows in-process so we can start adding tests and build up the API
surface incrementally.
"""

from __future__ import annotations

import os
import asyncio
import datetime
import yaml
import copy
import json
import uuid
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from typing import Any, Dict, List, Optional, Union, Callable, cast, ClassVar, Tuple
import re

from chungoid.utils.agent_resolver import AgentProvider, RegistryAgentProvider, NoAgentFoundForCategoryError, AmbiguousAgentCategoryError
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.common_enums import StageStatus, FlowPauseStatus, OnFailureAction
from chungoid.schemas.errors import AgentErrorDetails, OrchestratorError
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec, ClarificationCheckpointSpec, MasterStageFailurePolicy, UserGoalRequest
from chungoid.schemas.orchestration import SharedContext, ResumeContext # ADDED ResumeContext
from chungoid.schemas.agent_master_planner_reviewer import MasterPlannerReviewerInput, MasterPlannerReviewerOutput, ReviewerActionType, RetryStageWithChangesDetails
from chungoid.schemas.metrics import MetricEvent, MetricEventType
from chungoid.utils.metrics_store import MetricsStore
from chungoid.runtime.agents.system_file_system_agent import SystemFileSystemAgent_v1, WriteArtifactToFileInput
from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import EXECUTION_PLANS_COLLECTION, ProjectChromaManagerAgent_v1
from chungoid.schemas.project_state import ProjectStateV2, RunRecord, StageRecord
from chungoid.runtime.agents.core_code_generator_agent import CoreCodeGeneratorAgent_v1 # For checking agent_id
from chungoid.runtime.agents.system_requirements_gathering_agent import (
    SystemRequirementsGatheringAgent_v1,
    SystemRequirementsGatheringInput
) # ADDED
from chungoid.runtime.agents.agent_base import BaseAgent # ADDED for type annotations
from chungoid.schemas.agent_code_generator import SmartCodeGeneratorAgentInput # MODIFIED
from chungoid.schemas.agent_master_planner import MasterPlannerInput, MasterPlannerOutput # ADDED

# ADDED: Import for the correct collection name
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import GENERATED_CODE_ARTIFACTS_COLLECTION
from chungoid.runtime.services.context_resolution_service import ContextResolutionService # ADDED IMPORT
from chungoid.runtime.services.condition_evaluation_service import ConditionEvaluationService, ConditionEvaluationError # ADDED IMPORT
from chungoid.runtime.services.success_criteria_service import SuccessCriteriaService # ADDED IMPORT
from chungoid.runtime.services.orchestration_error_handler_service import OrchestrationErrorHandlerService, OrchestrationErrorResult, NEXT_STAGE_END_FAILURE, NEXT_STAGE_END_SUCCESS # MODIFIED: Added constants
from chungoid.runtime.services.input_validation_service import InputValidationService # ADDED IMPORT
from chungoid.schemas.flows import PausedRunDetails # ADDED

# Constants for next_stage signals (these are now also in error handler, ensure consistency or import from one source)
# NEXT_STAGE_END_SUCCESS = "__END_SUCCESS__" # REMOVED - imported from error handler
# NEXT_STAGE_END_FAILURE = "__END_FAILURE__" # REMOVED - imported from error handler

__all__ = [
    "StageSpec",
    "ExecutionPlan",
    "SyncOrchestrator",
    "AsyncOrchestrator",
    "OrchestrationError", 
    "NEXT_STAGE_END_SUCCESS", # EXPORTED
    "NEXT_STAGE_END_FAILURE"  # EXPORTED
]

DEFAULT_MAX_HOPS = 20  # Default max hops to prevent infinite loops
DEFAULT_AGENT_RETRIES = 0 # Default retries for an agent stage

class OrchestrationError(Exception):
    """Base exception for orchestration errors."""
    def __init__(self, message: str, stage_name: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        super().__init__(message)
        self.stage_name = stage_name
        self.agent_id = agent_id
        self.run_id = run_id

# ---------------------------------------------------------------------------
# DSL → Python models
# ---------------------------------------------------------------------------


class StageSpec(BaseModel):
    """Specification of a single stage inside a flow."""

    agent_id: str = Field(..., description="ID of the agent to invoke for this stage")
    inputs: Optional[dict] = Field(None, description="Input parameters for the agent")
    condition: Optional[str] = Field(None, description="Condition for branching")
    next_stage_true: Optional[str] = Field(
        None, description="Next stage if condition is true"
    )
    next_stage_false: Optional[str] = Field(
        None, description="Next stage if condition is false"
    )
    next_stage: Optional[str] = Field(
        None, description="Next stage or conditional object"
    )
    number: Optional[float] = Field(
        None, description="Unique stage number for status tracking"
    )
    on_error: Optional[Any] = Field(
        None, description="Error handler stage or conditional object"
    )
    parallel_group: Optional[str] = Field(
        None, description="Group name for parallel execution"
    )
    plugins: Optional[List[str]] = Field(
        None, description="List of plugin names to apply at this stage"
    )
    output_context_path: Optional[str] = None
    extra: Optional[dict] = Field(
        None, description="Arbitrary extra data for extensibility"
    )

    model_config = ConfigDict(extra="forbid")


class ExecutionPlan(BaseModel):
    """Validated, structured representation of the Flow YAML."""

    id: str
    created: datetime = Field(default_factory=datetime.utcnow) # MODIFIED: _dt.datetime to datetime
    start_stage: str
    stages: Dict[str, StageSpec]

    @classmethod
    def from_yaml(cls, yaml_text: str, flow_id: str | None = None) -> "ExecutionPlan":
        """Parse the *yaml_text* of a FlowCard and convert it to a plan."""

        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError("Flow YAML must map keys → values")

        if "stages" not in data or "start_stage" not in data:
            raise ValueError("Flow YAML missing required 'stages' or 'start_stage' key")

        raw_stages_from_yaml = data["stages"]
        transformed_stages_for_pydantic = {}
        for stage_name, stage_data_dict_from_yaml in raw_stages_from_yaml.items():
            current_stage_attrs = stage_data_dict_from_yaml.copy()
            if "next" in current_stage_attrs and isinstance(
                current_stage_attrs["next"], dict
            ):
                next_obj_from_yaml = current_stage_attrs.pop("next")
                # Map fields from YAML 'next' object to StageSpec fields
                if "condition" in next_obj_from_yaml:
                    current_stage_attrs["condition"] = next_obj_from_yaml["condition"]
                if "true" in next_obj_from_yaml:  # YAML uses 'true'
                    current_stage_attrs["next_stage_true"] = next_obj_from_yaml["true"]
                if "false" in next_obj_from_yaml:  # YAML uses 'false'
                    current_stage_attrs["next_stage_false"] = next_obj_from_yaml[
                        "false"
                    ]
                # Ensure StageSpec.next (the simple string one) is None for conditional branches
                current_stage_attrs["next_stage"] = None
            elif "next" in current_stage_attrs and isinstance(
                current_stage_attrs["next"], str
            ):
                # It's a simple string next, ensure it's assigned to StageSpec.next_stage
                # If the key in YAML is already 'next_stage', this won't hurt.
                # If it's 'next', we move it to 'next_stage'.
                if (
                    current_stage_attrs.get("next_stage") is None
                ):  # Avoid overwriting if next_stage already set
                    current_stage_attrs["next_stage"] = current_stage_attrs.pop("next")
                elif (
                    "next" in current_stage_attrs
                ):  # if next_stage was already there, just remove 'next' if it exists
                    current_stage_attrs.pop("next")

            transformed_stages_for_pydantic[stage_name] = current_stage_attrs

        return cls(
            id=flow_id or "<unknown>",
            start_stage=data["start_stage"],
            stages=transformed_stages_for_pydantic,  # Pass dict of dicts, Pydantic will create StageSpec objects
        )


# ---------------------------------------------------------------------------
# Orchestrator (sync only for now)
# ---------------------------------------------------------------------------


class SyncOrchestrator:
    """Very small synchronous orchestrator.

    It doesn't do agent routing yet – instead, it simply walks through the
    stages in breadth-first order starting from *start_stage* and returns a log
    of visited stage names.  This is enough for early unit tests and will be
    replaced by real agent invocation later.
    """

    def __init__(self, project_config: Dict[str, Any]):
        self.project_config = project_config
        self.logger = logging.getLogger(__name__)

    def _parse_condition(self, condition_str: str, context: Dict[str, Any]) -> bool:
        if not condition_str:
            return True  # No condition means proceed

        self.logger.debug(f"Parsing condition: {condition_str}")
        try:
            parts = []
            comparator = None
            # Order matters for multi-character operators
            if ">=" in condition_str:
                parts = condition_str.split(">=", 1)
                comparator = ">="
            elif "<=" in condition_str:
                parts = condition_str.split("<=", 1)
                comparator = "<="
            elif (
                "==" in condition_str
            ):  # Should be before single char '=' if that were supported
                parts = condition_str.split("==", 1)
                comparator = "=="
            elif "!=" in condition_str:
                parts = condition_str.split("!=", 1)
                comparator = "!="
            elif ">" in condition_str:
                parts = condition_str.split(">", 1)
                comparator = ">"
            elif "<" in condition_str:
                parts = condition_str.split("<", 1)
                comparator = "<"
            else:
                self.logger.error(
                    f"Unsupported condition format or unknown operator: {condition_str}"
                )
                return False

            if len(parts) != 2:
                self.logger.error(f"Invalid condition structure: {condition_str}")
                return False

            var_path_str = parts[0].strip()
            expected_value_str = parts[1].strip()

            current_val = context
            for key in var_path_str.split("."):
                if isinstance(current_val, dict) and key in current_val:
                    current_val = current_val[key]
                elif isinstance(current_val, list) and key.isdigit():
                    try:
                        current_val = current_val[int(key)]
                    except IndexError:
                        self.logger.warning(
                            f"Index out of bounds for '{key}' in path '{var_path_str}'."
                        )
                        return False
                else:
                    self.logger.warning(
                        f"Condition variable path '{var_path_str}' (key: '{key}') not fully found in context."
                    )
                    return False

            numeric_comparators = [">", "<", ">=", "<="]
            is_numeric_comparison = comparator in numeric_comparators

            if is_numeric_comparison:
                try:
                    val1 = float(current_val)  # Attempt to convert current_val to float
                    val2 = float(
                        expected_value_str.strip("'\\\"")
                    )  # Attempt to convert expected_value_str to float

                    self.logger.debug(
                        f"Numeric condition check: {val1} {comparator} {val2}"
                    )
                    if comparator == ">":
                        return val1 > val2
                    if comparator == "<":
                        return val1 < val2
                    if comparator == ">=":
                        return val1 >= val2
                    if comparator == "<=":
                        return val1 <= val2
                except ValueError:
                    self.logger.warning(
                        f"Type mismatch for numeric comparison: '{current_val}' vs '{expected_value_str}'. Condition evaluates to False."
                    )
                    return False  # If conversion to float fails for numeric comparison
            else:  # Handling '==' and '!='
                try:
                    coerced_expected_value = expected_value_str.strip(
                        "'\""
                    )  # Default to string
                    if isinstance(current_val, bool):
                        coerced_expected_value = expected_value_str.lower() in [
                            "true",
                            "1",
                            "yes",
                        ]
                    elif isinstance(current_val, int):
                        coerced_expected_value = int(expected_value_str.strip("'\""))
                    elif isinstance(current_val, float):
                        coerced_expected_value = float(expected_value_str.strip("'\""))
                    # If current_val is a string, coerced_expected_value remains a string as per default

                    self.logger.debug(
                        f"Equality condition check: '{current_val}' ({type(current_val)}) {comparator} '{coerced_expected_value}' ({type(coerced_expected_value)})"
                    )
                    if comparator == "==":
                        return current_val == coerced_expected_value
                    elif comparator == "!=":
                        return current_val != coerced_expected_value
                except ValueError as e:
                    self.logger.error(
                        f"Type conversion error for expected value in '=='/'=' condition '{condition_str}': {e}. Treating as unequal."
                    )
                    return (
                        True if comparator == "!=" else False
                    )  # Default to not equal on conversion error for ==/!=

            return False  # Fallback, should ideally be covered by above

        except Exception as e:
            self.logger.exception(f"Error evaluating condition '{condition_str}': {e}")
            return False  # Default to false on any error

    def run(self, plan: ExecutionPlan, context: Dict[str, Any]) -> List[str]:
        # This is a placeholder for the synchronous orchestrator logic
        self.logger.info("SyncOrchestrator.run called (placeholder)")
        # Simple sequential execution for now, ignoring conditions
        visited_stages: List[str] = []  # List to track visited stages
        current_stage_name = plan.start_stage
        max_hops = len(plan.stages) + 5  # Safety break
        hops = 0

        while current_stage_name and hops < max_hops:
            hops += 1
            if hops >= max_hops:
                self.logger.warning("Max hops reached, breaking execution.")
                break

            stage = plan.stages.get(current_stage_name)
            if not stage:
                self.logger.error(
                    f"Stage '{current_stage_name}' not found in plan. Aborting."
                )
                break

            self.logger.info(
                f"Executing stage: {current_stage_name} (Agent: {stage.agent_id})"
            )
            visited_stages.append(current_stage_name)  # Add visited stage to list
            # Placeholder for actual agent execution and context update
            # context['outputs'][current_stage_name] = {"message": f"Output from {current_stage_name}"}

            if stage.condition:
                if self._parse_condition(stage.condition, context):
                    current_stage_name = stage.next_stage_true
                else:
                    current_stage_name = stage.next_stage_false
            else:
                current_stage_name = stage.next_stage

        return visited_stages  # Return list of visited stages


# ---------------------------------------------------------------------------
# Async variant (placeholder – internally reuses SyncOrchestrator for now)
# ---------------------------------------------------------------------------


class BaseOrchestrator:
    """Base class for orchestrators."""

    def __init__(
        self,
        pipeline_def: ExecutionPlan,
        config: Dict[str, Any],
    ):
        self.pipeline_def = pipeline_def
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_next_stage(
        self, current_stage_name: str, context: Dict[str, Any]
    ) -> str | None:
        stage_def = self.pipeline_def.stages[current_stage_name]
        next_stage = stage_def.next_stage

        if isinstance(next_stage, dict) and next_stage.get("condition"):
            cond = next_stage["condition"]
            true_stage = next_stage["true"]
            false_stage = next_stage["false"]
            self.logger.debug(
                f"Evaluating condition '{cond}' for stage '{current_stage_name}'"
            )
            condition_met = SyncOrchestrator._eval_condition_expr(cond, context)
            next_stage = true_stage if condition_met else false_stage
            self.logger.debug(
                f"Condition result: {condition_met}, next stage: '{next_stage}'"
            )
        elif isinstance(next_stage, str):
            self.logger.debug(f"Direct next stage: '{next_stage}'")
        elif next_stage is None:
            self.logger.debug(f"No next stage defined after '{current_stage_name}'.")
        else:
            self.logger.warning(
                f"Invalid 'next' field type ({type(next_stage)}) for stage '{current_stage_name}'. Ending flow."
            )
            next_stage = None

        return next_stage


_SENTINEL = object()  # Add this sentinel object


class AsyncOrchestrator(BaseOrchestrator):
    """Asynchronous orchestrator for executing MasterExecutionPlans.

    Handles agent invocation, context management, conditional branching,
    error handling (including invoking a reviewer agent), success criteria checking,
    user clarification checkpoints, and metrics emission.
    """

    MAX_HOPS = DEFAULT_MAX_HOPS
    DEFAULT_AGENT_RETRIES = DEFAULT_AGENT_RETRIES # Class level default

    _current_run_id: Optional[str] = None
    _last_successful_stage_output: Optional[Any] = None
    _current_flow_config: Optional[Dict[str, Any]] = None
    shared_context: Optional[SharedContext] # MODIFIED: Allow None initially
    initial_goal_str: Optional[str] = None # ADDED: To store the initial goal string for the run
    context_resolver: ContextResolutionService # ADDED
    condition_evaluator: ConditionEvaluationService # ADDED
    success_criteria_evaluator: SuccessCriteriaService # ADDED
    input_validator: InputValidationService # ADDED

    def __init__(
        self,
        config: Dict[str, Any],
        agent_provider: AgentProvider,
        state_manager: StateManager,
        metrics_store: MetricsStore,
        master_planner_reviewer_agent_id: str, # Moved from config
        default_on_failure_action: OnFailureAction # Moved from config
    ):
        self.config = config
        self.agent_provider = agent_provider
        self.state_manager = state_manager
        self.metrics_store = metrics_store
        self.logger = logging.getLogger(__name__)
        self.master_planner_reviewer_agent_id = master_planner_reviewer_agent_id
        self.default_on_failure_action = default_on_failure_action
        self.shared_context = None # MODIFIED: Initialize to None

        # Initialize services
        self.context_resolver = ContextResolutionService(shared_context=self.shared_context) # Pass None initially
        self.condition_evaluator = ConditionEvaluationService(logger=self.logger, context_resolver=self.context_resolver)
        self.success_criteria_evaluator = SuccessCriteriaService(logger=self.logger, context_resolver=self.context_resolver) # Initialize here
        self.input_validator = InputValidationService(logger=self.logger) # ADDED
        
        # Initialize OrchestrationErrorHandlerService
        self.error_handler_service = OrchestrationErrorHandlerService(
            logger=self.logger,
            agent_provider=self.agent_provider,
            state_manager=self.state_manager,
            metrics_store=self.metrics_store,
            master_planner_reviewer_agent_id=self.master_planner_reviewer_agent_id,
            default_on_failure_action=self.default_on_failure_action,
            default_agent_retries=self.DEFAULT_AGENT_RETRIES 
        )

        self.current_plan: Optional[MasterExecutionPlan] = None

    def _emit_metric(
        self,
        event_type: MetricEventType,
        flow_id: str,
        run_id: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Helper to create and add a metric event."""
        # Ensure common fields are not duplicated if passed in kwargs
        metric_data = {
            "flow_id": flow_id,
            "run_id": run_id,
        }
        if data:  # Merge event-specific data payload
            metric_data_payload = data
        else:
            metric_data_payload = {}

        # Filter out None values from metric_data to keep events clean
        filtered_metric_data_args = {
            k: v for k, v in metric_data.items() if v is not None
        }

        try:
            event = MetricEvent(
                event_type=event_type,
                data=metric_data_payload,
                **filtered_metric_data_args,
            )
            self.metrics_store.add_event(event)
        except Exception as e:
            self.logger.error(
                f"Run {run_id}: Failed to emit metric event {event_type} for flow {flow_id}: {e}", exc_info=True
            )

    def _get_next_stage(self, current_stage_name: str) -> Optional[str]:
        """Returns the next stage to execute after current_stage_name, or None if terminating."""
        if current_stage_name not in self.current_plan.stages:
            self.logger.error(f"Stage '{current_stage_name}' not found in plan. No next stage can be determined.")
            return None

        stage_spec = self.current_plan.stages[current_stage_name]

        # Using ConditionEvaluationService for condition parsing and evaluation
        condition_str = stage_spec.condition
        if condition_str:
            try:
                condition_satisfied = self.condition_evaluator.parse_and_evaluate_condition(condition_str, self.shared_context)
                if condition_satisfied:
                    return stage_spec.next_stage_true
                else:
                    return stage_spec.next_stage_false
            except Exception as e_cond_eval:
                self.logger.error(f"Error evaluating condition '{condition_str}' for stage '{current_stage_name}': {e_cond_eval}")
                return stage_spec.next_stage_false

        return stage_spec.next_stage

    def _unwrap_inputs_if_needed(self, inputs_dict: dict) -> dict:
        """
        Helper method to unwrap nested 'inputs' structures that can occur during
        retry scenarios with reviewer agent modifications.
        
        This fixes the double-wrapping bug where agents receive:
        {'inputs': {'user_goal': 'value'}} instead of {'user_goal': 'value'}
        
        Args:
            inputs_dict: The input dictionary that might have nested 'inputs' wrapper(s)
            
        Returns:
            The unwrapped input dictionary
        """
        if not isinstance(inputs_dict, dict):
            return inputs_dict
            
        result = inputs_dict.copy()
        
        # Handle nested 'inputs' structures (commonly from reviewer modifications)
        # If we have an 'inputs' key with a dict value, unwrap it
        if (isinstance(result, dict) and 
            "inputs" in result and 
            isinstance(result["inputs"], dict)):
            
            self.logger.debug(f"Unwrapping nested 'inputs' layer: {result}")
            result = result["inputs"]
            
        return result

    async def _invoke_agent_for_stage(
        self,
        stage_name: str,
        stage_spec: MasterStageSpec,
        run_id: str,
        flow_id: str,
        attempt_number: int
    ) -> Any: # Can return any agent output type or raise AgentErrorDetails / Exception
        """Invokes the agent for the given stage and returns its output."""
        self.logger.info(f"Run {run_id}: Invoking agent for stage '{stage_name}' (Agent: {stage_spec.agent_id}, Attempt: {attempt_number}).")
        self.shared_context.current_stage_id = stage_name
        self.shared_context.current_stage_status = StageStatus.RUNNING
        self.shared_context.current_attempt_number_for_stage = attempt_number
        self._emit_metric(MetricEventType.MASTER_STAGE_START, flow_id, run_id, stage_id=stage_name, agent_id=stage_spec.agent_id, data={"attempt_number": attempt_number})

        agent_callable: Optional[Callable[..., Any]] = None
        agent_instance_for_type_check: Optional[BaseAgent] = None # To hold the resolved agent instance
        resolved_inputs = {}
        try:
            # Resolve inputs first using ContextResolutionService
            resolved_inputs = self.context_resolver.resolve_inputs_for_stage(
                inputs_spec=stage_spec.inputs or {},
                shared_context_override=self.shared_context
            )
            self.logger.debug(f"Run {run_id}: Resolved inputs for stage '{stage_name}': {resolved_inputs}")

            self.shared_context.update_resolved_inputs_for_current_stage(resolved_inputs)

            # Get agent instance for type checking and its callable
            # Ensure shared_context is passed to provider if it needs it for instantiation
            if isinstance(self.agent_provider, RegistryAgentProvider):
                # If it's a RegistryAgentProvider, it should have set_orchestrator_shared_context
                # or accept shared_context in its get methods.
                # The get_raw_agent_instance in RegistryAgentProvider was updated to accept shared_context.data
                self.agent_provider.set_orchestrator_shared_context(self.shared_context.data) # Pass the .data part
                agent_instance_for_type_check = self.agent_provider.get_raw_agent_instance(
                    stage_spec.agent_id, 
                    shared_context=self.shared_context.data # Pass the .data part
                )
            else: # Fallback for other providers, may not support raw instance easily
                 agent_instance_for_type_check = None # Or try a more generic way if available

            # ADDED: Debug logging for agent instance detection
            self.logger.info(f"Run {run_id}: [DEBUG] Agent instance detection for '{stage_spec.agent_id}':")
            self.logger.info(f"Run {run_id}: [DEBUG] - agent_instance_for_type_check: {agent_instance_for_type_check}")
            self.logger.info(f"Run {run_id}: [DEBUG] - agent_instance_for_type_check type: {type(agent_instance_for_type_check)}")
            self.logger.info(f"Run {run_id}: [DEBUG] - isinstance(SystemRequirementsGatheringAgent_v1): {isinstance(agent_instance_for_type_check, SystemRequirementsGatheringAgent_v1) if agent_instance_for_type_check else 'agent_instance is None'}")

            if agent_instance_for_type_check and hasattr(agent_instance_for_type_check, 'invoke_async'):
                agent_callable = agent_instance_for_type_check.invoke_async
            else: # Fallback if raw instance not available or no invoke_async
                agent_callable = self.agent_provider.get(identifier=stage_spec.agent_id, shared_context=self.shared_context.data)


            # Ensure agent_callable is actually callable
            if not callable(agent_callable):
                raise OrchestrationError(
                    f"Agent ID '{stage_spec.agent_id}' resolved to a non-callable item: {type(agent_callable)}.",
                    stage_name=stage_name, agent_id=stage_spec.agent_id, run_id=run_id
                )

            # ENHANCED: Use InputValidationService for ALL agents - generic input validation and injection
            # First, unwrap any nested input structures
            self.logger.info(f"Run {run_id}: Before unwrapping, resolved_inputs = {resolved_inputs}")
            unwrapped_inputs = self._unwrap_inputs_if_needed(resolved_inputs)
            self.logger.info(f"Run {run_id}: After unwrapping, unwrapped_inputs = {unwrapped_inputs}")
            
            # Prepare injection context for default value injection
            injection_context = {
                "initial_goal_str": self.initial_goal_str,
                "project_id": self.shared_context.data.get('project_id') if self.shared_context and self.shared_context.data else None
            }
            self.logger.debug(f"Run {run_id}: Injection context = {injection_context}")
            
            # Use InputValidationService to validate and inject defaults
            validation_result = self.input_validator.validate_and_inject_inputs(
                agent_id=stage_spec.agent_id,
                agent_instance=agent_instance_for_type_check,
                resolved_inputs=unwrapped_inputs,
                injection_context=injection_context,
                run_id=run_id
            )
            self.logger.debug(f"Run {run_id}: Validation result: is_valid={validation_result.is_valid}, final_inputs={validation_result.final_inputs}")
            
            # Log validation results
            if validation_result.injected_fields:
                self.logger.info(f"Run {run_id}: Injected default values for agent '{stage_spec.agent_id}': {validation_result.injected_fields}")
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    self.logger.warning(f"Run {run_id}: {warning}")
            
            if not validation_result.is_valid:
                self.logger.error(f"Run {run_id}: Input validation failed for agent '{stage_spec.agent_id}': {validation_result.validation_errors}")
                # Continue anyway - let the agent handle the validation error, or it might be handled by error handler
            
            # Use the validated inputs for agent invocation
            final_inputs_for_agent = validation_result.final_inputs
            self.logger.info(f"Run {run_id}: final_inputs_for_agent (after validation) = {final_inputs_for_agent}")
            
            # Agent-specific invocation logic using validated inputs
            if isinstance(agent_instance_for_type_check, MasterPlannerAgent):
                if not isinstance(final_inputs_for_agent, MasterPlannerInput):
                     self.logger.info(f"Run {run_id}: Constructing MasterPlannerInput for stage '{stage_name}'")
                     current_plan_id = self.current_plan.id if self.current_plan else f"unknown_plan_for_run_{run_id}"
                     user_goal_for_planner = final_inputs_for_agent.get("user_goal", self.initial_goal_str or f"Execute plan {current_plan_id}")
                     
                     # Get project_id from shared_context to fix the warning
                     project_id_for_planner = self.shared_context.data.get('project_id') if self.shared_context and self.shared_context.data else None
                     
                     master_plan_input = MasterPlannerInput(
                         master_plan_id=current_plan_id,
                         flow_id=flow_id,
                         run_id=run_id,
                         user_goal=user_goal_for_planner,
                         project_id=project_id_for_planner,
                     )
                     raw_output = await agent_callable(inputs=master_plan_input, full_context=self.shared_context)
                else:
                    raw_output = await agent_callable(inputs=final_inputs_for_agent, full_context=self.shared_context)
            
            elif isinstance(agent_instance_for_type_check, CoreCodeGeneratorAgent_v1):
                smart_input_for_core_gen: SmartCodeGeneratorAgentInput
                if not isinstance(final_inputs_for_agent, SmartCodeGeneratorAgentInput):
                    self.logger.warning(f"Run {run_id}: Stage '{stage_name}' uses agent '{stage_spec.agent_id}' (resolved to CoreCodeGeneratorAgent_v1) but final inputs are not SmartCodeGeneratorAgentInput. Attempting conversion.")
                    try:
                        smart_input_for_core_gen = SmartCodeGeneratorAgentInput(**final_inputs_for_agent)
                    except Exception as e_conv_smart:
                        self.logger.error(f"Run {run_id}: Failed to convert final inputs to SmartCodeGeneratorAgentInput for '{stage_name}': {e_conv_smart}. Raising error.")
                        raise OrchestrationError(f"Input conversion failed for {stage_spec.agent_id}: {e_conv_smart}", stage_name=stage_name, agent_id=stage_spec.agent_id, run_id=run_id) from e_conv_smart
                else:
                    smart_input_for_core_gen = final_inputs_for_agent
                raw_output = await agent_callable(task_input=smart_input_for_core_gen, full_context=self.shared_context)
            
            elif isinstance(agent_instance_for_type_check, SystemRequirementsGatheringAgent_v1):
                self.logger.info(f"Run {run_id}: About to invoke SystemRequirementsGatheringAgent_v1")
                self.logger.info(f"Run {run_id}: final_inputs_for_agent = {final_inputs_for_agent}")
                self.logger.info(f"Run {run_id}: type(final_inputs_for_agent) = {type(final_inputs_for_agent)}")
                raw_output = await agent_callable(inputs=final_inputs_for_agent, full_context=self.shared_context)
            
            elif isinstance(agent_instance_for_type_check, SystemFileSystemAgent_v1):
                # Determine the effective project root for file system operations
                # This is crucial for SystemFileSystemAgent_v1
                
                # ADDED: Diagnostic log
                self.logger.info(f"Orchestrator[{stage_name}]: Inspecting shared_context.data before determining effective_project_root_for_fs_agent: {self.shared_context.data}")

                effective_project_root_for_fs_agent = self.shared_context.data.get('project_root_path')
                
                # REFINED: Fallback logic
                if not effective_project_root_for_fs_agent:
                    self.logger.warning(f"Orchestrator[{stage_name}]: 'project_root_path' was not found or was empty in shared_context.data. "
                                        f"Attempting to fall back to 'mcp_root_workspace_path'.")
                    effective_project_root_for_fs_agent = self.shared_context.data.get('mcp_root_workspace_path')
                    if effective_project_root_for_fs_agent:
                        self.logger.warning(f"Orchestrator[{stage_name}]: Using 'mcp_root_workspace_path' ({effective_project_root_for_fs_agent}) as fallback for FileSystemAgent operations. "
                                            f"This might be incorrect if a specific project sub-directory was intended.")
                    else:
                        self.logger.error(f"Orchestrator[{stage_name}]: CRITICAL - Neither 'project_root_path' nor 'mcp_root_workspace_path' "
                                          f"could be determined from shared_context.data for FileSystemAgent. File operations will likely fail or use CWD.")
                
                # Ensure it's a Path object if found, otherwise it might remain None
                if isinstance(effective_project_root_for_fs_agent, str):
                    effective_project_root_for_fs_agent = Path(effective_project_root_for_fs_agent)
                elif not isinstance(effective_project_root_for_fs_agent, Path) and effective_project_root_for_fs_agent is not None:
                    self.logger.warning(f"Orchestrator[{stage_name}]: effective_project_root_for_fs_agent was not a str or Path, but {type(effective_project_root_for_fs_agent)}. Setting to None.")
                    effective_project_root_for_fs_agent = None

                self.logger.info(f"Orchestrator[{stage_name}]: Invoking {stage_spec.agent_id} (resolved to actual ID: {getattr(agent_instance_for_type_check, 'AGENT_ID', 'N/A')}) "
                                 f"with calculated explicit project_root for its invoke_async: '{effective_project_root_for_fs_agent}'")
                
                # SystemFileSystemAgent.invoke_async has a 'project_root' parameter
                # FIXED: Use final_inputs_for_agent instead of resolved_inputs
                raw_output = await agent_callable( # Use agent_callable
                    inputs=final_inputs_for_agent,
                    project_root=effective_project_root_for_fs_agent, # Pass it explicitly
                    shared_context=self.shared_context # Pass full shared context
                )
            
            else: # Generic agent invocation pattern
                try:
                    self.logger.debug(f"Run {run_id}: Invoking generic agent '{stage_spec.agent_id}' with final inputs")
                    raw_output = await agent_callable(inputs=final_inputs_for_agent, full_context=self.shared_context)
                except TypeError as te_invoke:
                    if "got an unexpected keyword argument 'inputs'" in str(te_invoke) or \
                       "missing 1 required positional argument: 'task_input'" in str(te_invoke) or \
                       (hasattr(agent_callable, '__self__') and hasattr(agent_callable.__self__.__class__, 'INPUT_SCHEMA') and 'task_input' in agent_callable.__code__.co_varnames):
                        
                        self.logger.warning(f"Run {run_id}: Agent '{stage_spec.agent_id}' call with 'inputs' failed or agent expects 'task_input'. Retrying with 'task_input'. Error (if any): {te_invoke}")
                        try:
                            # For retry, determine input model if possible for conversion
                            input_model_cls_for_retry = None
                            if agent_instance_for_type_check and hasattr(agent_instance_for_type_check, 'INPUT_SCHEMA'):
                                input_model_cls_for_retry = agent_instance_for_type_check.INPUT_SCHEMA
                            elif hasattr(agent_callable, '__self__') and hasattr(agent_callable.__self__.__class__, 'INPUT_SCHEMA'): # Fallback if instance not resolved early
                                input_model_cls_for_retry = agent_callable.__self__.__class__.INPUT_SCHEMA
                            
                            processed_task_input_for_retry = final_inputs_for_agent
                            if input_model_cls_for_retry and isinstance(final_inputs_for_agent, dict) and not isinstance(final_inputs_for_agent, input_model_cls_for_retry):
                                self.logger.info(f"Run {run_id}: Attempting to parse final_inputs_for_agent into {input_model_cls_for_retry.__name__} for agent '{stage_spec.agent_id}' (retry with task_input).")
                                try:
                                    actual_payload_dict = final_inputs_for_agent
                                    if len(final_inputs_for_agent) == 1:
                                        first_key = next(iter(final_inputs_for_agent))
                                        temp_payload = final_inputs_for_agent[first_key]
                                        while isinstance(temp_payload, dict) and len(temp_payload) == 1 and next(iter(temp_payload)) == first_key:
                                            temp_payload = temp_payload[first_key]
                                        if isinstance(temp_payload, dict):
                                            actual_payload_dict = temp_payload
                                        elif first_key.lower() == input_model_cls_for_retry.__name__.lower() and isinstance(final_inputs_for_agent[first_key], dict):
                                            actual_payload_dict = final_inputs_for_agent[first_key]

                                    processed_task_input_for_retry = input_model_cls_for_retry.model_validate(actual_payload_dict)
                                    self.logger.info(f"Run {run_id}: Successfully parsed inputs into {input_model_cls_for_retry.__name__}.")
                                except ValidationError as e_parse_validation:
                                    self.logger.error(f"Run {run_id}: Pydantic ValidationError parsing resolved_inputs into {input_model_cls_for_retry.__name__} for agent '{stage_spec.agent_id}': {e_parse_validation}", exc_info=True)
                                    raise OrchestrationError(
                                        message=f"Input validation failed for agent {stage_spec.agent_id}: {e_parse_validation}",
                                        stage_name=stage_name,
                                        agent_id=stage_spec.agent_id,
                                        run_id=run_id
                                    ) from e_parse_validation
                                except (TypeError, AttributeError) as e_parse_other: # Catch other parsing/validation related errors
                                    self.logger.error(f"Run {run_id}: Failed to parse final_inputs_for_agent into {input_model_cls_for_retry.__name__} for agent '{stage_spec.agent_id}' (non-ValidationError): {e_parse_other}. Passing dict as is.", exc_info=True)
                                    processed_task_input_for_retry = final_inputs_for_agent
                            elif not isinstance(final_inputs_for_agent, BaseModel):
                                self.logger.warning(f"Run {run_id}: Agent '{stage_spec.agent_id}' expects '{input_model_cls_for_retry.__name__ if input_model_cls_for_retry else 'Pydantic model'}' but received dict that could not be parsed, or non-dict. Passing as is.")

                            raw_output = await agent_callable(task_input=processed_task_input_for_retry, full_context=self.shared_context)
                        except Exception as e_task_input_retry:
                            self.logger.error(f"Run {run_id}: Agent '{stage_spec.agent_id}' call with 'task_input' (type: {type(processed_task_input_for_retry).__name__}) also failed: {e_task_input_retry}", exc_info=True)
                            raise OrchestrationError(f"Agent invocation failed for {stage_spec.agent_id} with both 'inputs' and 'task_input' parameters. Last error: {e_task_input_retry}", stage_name=stage_name, agent_id=stage_spec.agent_id, run_id=run_id) from e_task_input_retry
                    else:
                        raise # Re-raise original TypeError if not related to inputs/task_input mismatch

            self.logger.info(f"Run {run_id}: Agent for stage '{stage_name}' completed.")
            return raw_output

        except AgentErrorDetails as agent_err_details: # Agent itself raised a pre-formatted error
            self.logger.warning(f"Run {run_id}: Agent for stage '{stage_name}' raised AgentErrorDetails: {agent_err_details.message}")
            # Ensure resolved_inputs are attached if missing
            if agent_err_details.resolved_inputs_at_failure is None:
                agent_err_details.resolved_inputs_at_failure = final_inputs_for_agent if 'final_inputs_for_agent' in locals() else resolved_inputs or {}
            raise agent_err_details # Re-raise to be caught by the main loop's error handler
        
        except Exception as e_invoke:
            self.logger.error(f"Run {run_id}: Error invoking agent for stage '{stage_name}': {e_invoke}", exc_info=True)
            # Wrap in AgentErrorDetails before raising to ensure it's handled consistently
            # The main loop's handler will further process this.
            error_type = e_invoke.__class__.__name__
            # Determine can_retry based on exception type or attributes if available
            can_retry_flag = getattr(e_invoke, 'can_retry', False) 

            # Create a new AgentErrorDetails, ensuring resolved_inputs are included
            agent_error = AgentErrorDetails(
                agent_id=stage_spec.agent_id if stage_spec else "UnknownAgentFromInvoke",
                stage_id=stage_name,
                error_type=error_type,
                message=str(e_invoke),
                traceback=traceback.format_exc(),
                can_retry=can_retry_flag,
                resolved_inputs_at_failure=final_inputs_for_agent if 'final_inputs_for_agent' in locals() else resolved_inputs or {}
            )
            raise agent_error # Raise the standardized error

    async def _execute_flow_loop(
        self,
        flow_id: str,
        run_id: str,
        start_stage_name: str,
        max_hops: int,
        resume_context: Optional[ResumeContext] = None
    ) -> Tuple[StageStatus, Optional[AgentErrorDetails]]:
        """The main loop for executing stages in a flow."""
        if not self.current_plan:
            self.logger.error(f"Run {run_id}: MasterExecutionPlan not loaded. Cannot execute flow.")
            return StageStatus.COMPLETED_FAILURE, AgentErrorDetails(message="Master plan not loaded.", stage_id=start_stage_name, agent_id="Orchestrator", error_type="OrchestrationSetupError")

        current_stage_name: Optional[str] = start_stage_name
        current_attempt_number_for_stage: int = 1
        hops = 0
        
        agent_error_obj: Optional[AgentErrorDetails] = None
        if resume_context and resume_context.error_details:
            if isinstance(resume_context.error_details, AgentErrorDetails):
                agent_error_obj = resume_context.error_details
            elif isinstance(resume_context.error_details, dict):
                try:
                    agent_error_obj = AgentErrorDetails(**resume_context.error_details)
                except Exception as e_ae_conv:
                    self.logger.warning(f"Run {run_id}: Could not convert error_details from resume_context to AgentErrorDetails: {e_ae_conv}")
                    agent_error_obj = AgentErrorDetails(stage_id=current_stage_name or "UnknownOnResume", agent_id="Orchestrator", message="Error details from resume were not loadable.", error_type="DeserializationError")
            else:
                self.logger.warning(f"Run {run_id}: Unknown type for error_details in resume_context: {type(resume_context.error_details)}")
        
        stage_output: Any = None 
        
        # MODIFIED: Add "FINAL_STEP" to the list of terminal signals for the loop condition
        while current_stage_name and current_stage_name not in [NEXT_STAGE_END_SUCCESS, NEXT_STAGE_END_FAILURE, "FINAL_STEP"] and hops < max_hops:
            hops += 1
            self.logger.info(f"Run {run_id}: Processing stage '{current_stage_name}' (Attempt: {current_attempt_number_for_stage}, Hop: {hops}).")

            if current_stage_name not in self.current_plan.stages:
                self.logger.error(f"Run {run_id}: Stage '{current_stage_name}' not found in execution plan. Terminating flow.")
                agent_error_obj = AgentErrorDetails(message=f"Stage '{current_stage_name}' not found in plan.", stage_id=current_stage_name, agent_id="Orchestrator", error_type="PlanValidationError")
                # await self.state_manager.update_status(run_id, StageStatus.COMPLETED_FAILURE, error_details=agent_error_obj) # OLD CALL
                self.state_manager.record_flow_end( # MODIFIED: Removed await
                    run_id=run_id, 
                    flow_id=self._current_flow_id, 
                    final_status=StageStatus.COMPLETED_FAILURE.value, 
                    error_message=agent_error_obj.message if agent_error_obj else f"Stage '{current_stage_name}' not found in plan.",
                    final_outputs=self.shared_context.data.get("outputs")
                )
                return StageStatus.COMPLETED_FAILURE, agent_error_obj

            stage_spec = self.current_plan.stages[current_stage_name]
            resolved_inputs_for_error_handler: Optional[Dict[str, Any]] = None # Initialize for error case

            try:
                self.state_manager.record_stage_start(
                    run_id=run_id,
                    flow_id=flow_id, 
                    stage_id=current_stage_name,
                    agent_id=stage_spec.agent_id
                )

                # Check for clarification checkpoint before agent invocation
                if stage_spec.clarification_checkpoint and stage_spec.clarification_checkpoint.condition:
                    self.logger.info(f"Run {run_id}: Stage '{current_stage_name}' has a clarification checkpoint. Evaluating condition: {stage_spec.clarification_checkpoint.condition}")
                    try:
                        # Resolve context values needed for the condition string if it uses them
                        # This uses a simplified resolver for the condition string itself, not full stage inputs
                        # Condition evaluation service handles resolving path values within the condition string.
                        condition_met = self.condition_evaluator.evaluate_condition(stage_spec.clarification_checkpoint.condition, self.shared_context, None)
                    except ConditionEvaluationError as e_cond_clarify:
                        self.logger.warning(f"Run {run_id}: Error evaluating clarification condition for stage '{current_stage_name}': {e_cond_clarify}. Condition treated as false (no pause)." )
                        condition_met = False
                    except Exception as e_cond_clarify_unexp:
                        self.logger.error(f"Run {run_id}: Unexpected error evaluating clarification condition for stage '{current_stage_name}': {e_cond_clarify_unexp}. Condition treated as false.", exc_info=True)
                        condition_met = False
                        
                    if condition_met:
                        self.logger.info(f"Run {run_id}: Clarification condition for stage '{current_stage_name}' met. Pausing for user input.")
                        pause_status = FlowPauseStatus.CLARIFICATION_REQUIRED
                        self.shared_context.current_stage_status = StageStatus.PAUSED # Mark as paused in context
                        
                        # clarification_details is required for PausedRunDetails if pause_status is CLARIFICATION_REQUIRED
                        # stage_spec.clarification_checkpoint is ClarificationCheckpointDetails
                        if not stage_spec.clarification_checkpoint:
                            self.logger.error(f"Run {run_id}: Clarification required for stage '{current_stage_name}' but no clarification_checkpoint defined in stage_spec!")
                            # Fallback to a generic error pause if checkpoint details are missing
                            # This is an unexpected state.
                            generic_error_for_missing_clarif = AgentErrorDetails(
                                error_type="OrchestrationError", 
                                message=f"Clarification checkpoint missing for stage {current_stage_name}",
                                stage_id=current_stage_name,
                                agent_id=stage_spec.agent_id
                            )
                            paused_details_clarify_err = PausedRunDetails(
                                run_id=run_id,
                                flow_id=flow_id,
                                current_master_plan_snapshot=self.current_plan.model_copy(deep=True) if self.current_plan else None,
                                paused_stage_id=current_stage_name,
                                pause_status=FlowPauseStatus.USER_INTERVENTION_REQUIRED, # Fallback status
                                error_details=generic_error_for_missing_clarif,
                                pause_reason_summary="Internal error: Clarification checkpoint details missing.",
                                last_stage_id=current_stage_name,
                                last_stage_attempt_number=current_attempt_number_for_stage
                            )
                            self.state_manager.save_paused_flow_state(paused_details_clarify_err)
                            self._emit_metric(MetricEventType.FLOW_PAUSED, flow_id, run_id, data={
                                "paused_at_stage_id": current_stage_name,
                                "pause_status": FlowPauseStatus.USER_INTERVENTION_REQUIRED.value,
                                "error_details": generic_error_for_missing_clarif.to_dict(),
                                "reason": "Clarification checkpoint details missing"
                            })
                            return FlowPauseStatus.USER_INTERVENTION_REQUIRED, generic_error_for_missing_clarif


                        paused_details_clarify = PausedRunDetails(
                            run_id=run_id,
                            flow_id=flow_id,
                            current_master_plan_snapshot=self.current_plan.model_copy(deep=True) if self.current_plan else None,
                            paused_stage_id=current_stage_name,
                            pause_status=pause_status, # CLARIFICATION_REQUIRED
                            clarification_details=stage_spec.clarification_checkpoint, # Pass the details
                            # error_details would be None here unless there was a preceding error
                            pause_reason_summary=f"Clarification required for stage: {current_stage_name}. {stage_spec.clarification_checkpoint.condition_description}",
                            last_stage_id=current_stage_name,
                            last_stage_attempt_number=current_attempt_number_for_stage
                        )
                        self.state_manager.save_paused_flow_state(paused_details_clarify)
                        # Emit metric for flow paused
                        self._emit_metric(MetricEventType.FLOW_PAUSED, flow_id, run_id, data={
                            "paused_at_stage_id": current_stage_name,
                            "pause_status": pause_status.value,
                            "clarification_condition": stage_spec.clarification_checkpoint.condition_description
                        })
                        return pause_status, None # No agent_error_obj when pausing for clarification

                # Invoke the agent for the current stage
                stage_output_raw = await self._invoke_agent_for_stage(
                    stage_name=current_stage_name,
                    stage_spec=stage_spec,
                    run_id=run_id,
                    flow_id=flow_id,
                    attempt_number=current_attempt_number_for_stage
                )
                
                # Extract final output and any agent-emitted metrics
                stage_output, agent_emitted_metrics = self._extract_output_and_metrics(stage_output_raw)
                if agent_emitted_metrics:
                    self.logger.info(f"Run {run_id}: Agent for stage '{current_stage_name}' emitted metrics: {agent_emitted_metrics}")
                    # TODO: Process/store agent_emitted_metrics if a formal system for this exists
                
                self.shared_context.update_current_stage_output(stage_output)
                self.shared_context.add_stage_output_to_history(current_stage_name, stage_output)

                # E. Check Success Criteria
                # =========================
                overall_success_criteria_passed = True
                failed_criteria_list: List[str] = []
                if stage_spec.success_criteria:
                    self.logger.info(f"Run {run_id}, Flow {flow_id}: Checking success criteria for stage '{current_stage_name}'.")
                    # MODIFIED: Call service directly
                    shared_data_for_criteria = self.shared_context.data if self.shared_context else {}
                    overall_success_criteria_passed, failed_criteria_list = await self.success_criteria_evaluator.check_criteria_for_stage(
                        stage_name=current_stage_name,
                        stage_spec=stage_spec,
                        stage_outputs=stage_output,
                        shared_context_for_stage=shared_data_for_criteria
                    )

                    if not overall_success_criteria_passed:
                        self.logger.warning(
                            f"Run {run_id}, Flow {flow_id}: Stage '{current_stage_name}' failed one or more success criteria: {failed_criteria_list}")

                # F. Handle Clarification Checkpoint (if success criteria passed)
                # ==============================================================
                if overall_success_criteria_passed and stage_spec.clarification_checkpoint:
                    self.logger.info(f"Run {run_id}, Flow {flow_id}: Stage '{current_stage_name}' has a clarification checkpoint.")
                    # Use ConditionEvaluationService to check if context path needs to be created by user.
                    # For now, assume if a checkpoint exists, we always pause.
                    # A more advanced check might involve a condition on the checkpoint itself.
                    
                    # Create PausedRunDetails for clarification
                    paused_details_clarification = PausedRunDetails(
                        run_id=run_id,
                        flow_id=flow_id,
                        paused_stage_id=current_stage_name,
                        status=FlowPauseStatus.CLARIFICATION_REQUIRED,
                        reason=f"Clarification requested after stage '{current_stage_name}'.",
                        clarification_checkpoint_details=stage_spec.clarification_checkpoint, # Store the spec
                        context_snapshot_ref=None, # TODO: Implement context snapshotting
                        current_master_plan_snapshot=self.current_plan.model_copy(deep=True) if self.current_plan else None
                    )
                    await self.state_manager.record_paused_run(paused_details_clarification)
                    self._emit_metric(MetricEventType.FLOW_PAUSED, flow_id, run_id, {"stage": current_stage_name, "reason": "clarification_checkpoint"})
                    return StageStatus.PAUSED, None # Indicate pause for clarification

                # G. Determine Next Stage or End Flow
                # ==================================
                if not overall_success_criteria_passed:
                    # If success criteria failed, this is treated as a stage error
                    # Construct an error detail for success criteria failure
                    # This error will then be handled by the OrchestrationErrorHandlerService
                    success_criteria_error_msg = f"Stage '{current_stage_name}' failed one or more success criteria: {failed_criteria_list}"
                    self.logger.error(f"Run {run_id}, Flow {flow_id}: {success_criteria_error_msg}")
                    agent_error_for_criteria = AgentErrorDetails(
                        agent_id=stage_spec.agent_id or "OrchestratorSuccessCheck",
                        stage_id=current_stage_name,
                        error_type="SuccessCriteriaFailure",
                        message=success_criteria_error_msg,
                        details={"failed_criteria": failed_criteria_list},
                        can_retry=False, # Success criteria failures are typically not retried directly without intervention
                        resolved_inputs_at_failure=self.shared_context.get_resolved_inputs_for_current_stage() # inputs for the stage that failed criteria
                    )
                    
                    # Delegate to error handler
                    error_handling_result = await self.error_handler_service.handle_stage_execution_error(
                        current_stage_name=current_stage_name,
                        flow_id=flow_id,
                        run_id=run_id,
                        current_plan=self.current_plan,
                        error=agent_error_for_criteria, # The constructed error
                        agent_id_that_erred=stage_spec.agent_id or "OrchestratorSuccessCheck",
                        attempt_number=current_attempt_number_for_stage, # Current attempt for the stage
                        shared_context_at_error=self.shared_context,
                        resolved_inputs_at_failure=self.shared_context.get_resolved_inputs_for_current_stage()
                    )

                    if error_handling_result.flow_pause_status != FlowPauseStatus.NOT_PAUSED:
                        # The error handler decided to pause or requires user intervention
                        self.logger.info(f"Run {run_id}, Flow {flow_id}: Flow pausing due to success criteria failure handling for stage '{current_stage_name}'. Status: {error_handling_result.flow_pause_status}")
                        return StageStatus.PAUSED, error_handling_result.updated_agent_error_details

                    if error_handling_result.next_stage_to_execute == NEXT_STAGE_END_FAILURE:
                        self.logger.info(f"Run {run_id}, Flow {flow_id}: Flow ending in failure due to success criteria failure handling for stage '{current_stage_name}'.")
                        return StageStatus.COMPLETED_FAILURE, error_handling_result.updated_agent_error_details
                    
                    if error_handling_result.next_stage_to_execute == current_stage_name: # Retry current stage
                        self.logger.info(f"Run {run_id}, Flow {flow_id}: Retrying stage '{current_stage_name}' due to success criteria failure handling.")
                        # Loop will continue with current_stage_name, attempt_number will increment
                        # Any input modifications from reviewer would be in error_handling_result.modified_stage_inputs
                        # The orchestrator needs to apply these if present.
                        # For now, direct retry logic is simpler; advanced input modification on retry needs more work here.
                        next_stage_candidate = current_stage_name # stay on current stage for retry
                        # attempt_number is incremented at the start of the next loop iteration if retrying the same stage
                    
                    elif error_handling_result.next_stage_to_execute:
                        next_stage_candidate = error_handling_result.next_stage_to_execute
                    else: # Should not happen if not paused and not ending
                        self.logger.error(f"Run {run_id}, Flow {flow_id}: Error handler returned unexpected state for success criteria failure of stage '{current_stage_name}'. Forcing failure.")
                        return StageStatus.COMPLETED_FAILURE, agent_error_for_criteria

                elif stage_spec.condition: # Stage has a condition for branching
                    self.logger.info(f"Run {run_id}, Flow {flow_id}: Evaluating condition for stage '{current_stage_name}': {stage_spec.condition}")
                    try:
                        # MODIFIED: Call service directly
                        condition_eval_result = self.condition_evaluator.evaluate_condition(
                            condition_str=stage_spec.condition,
                            shared_context_data=self.shared_context.data if self.shared_context else {},
                            stage_name=current_stage_name # For logging context
                        )
                        # Based on condition_eval_result, determine next_stage_candidate
                        # This part was missing assignment to next_stage_candidate
                        if condition_eval_result:
                            next_stage_candidate = stage_spec.next_stage_id_true # Assuming MasterStageSpec has these distinct fields
                        else:
                            next_stage_candidate = stage_spec.next_stage_id_false # Assuming MasterStageSpec has these distinct fields
                        
                        if next_stage_candidate is None:
                            # If true/false path not explicitly defined, it might fall to a default next_stage_id or end.
                            # The _get_next_stage logic handles this more comprehensively.
                            self.logger.info(f"Run {run_id}: Condition for '{current_stage_name}' evaluated to {condition_eval_result}, but specific true/false path not set. Checking default next stage.")
                            next_stage_candidate = self._get_next_stage(current_stage_name)

                    except ConditionEvaluationError as e:
                        self.logger.error(f"Run {run_id}, Flow {flow_id}: Error evaluating condition for stage '{current_stage_name}': {e}. Treating as False. Defaulting next stage.")
                        condition_eval_result = False
                        next_stage_candidate = self._get_next_stage(current_stage_name) # Fallback to default next logic
                else: # No success criteria failure, no explicit condition, so get default next stage
                    next_stage_candidate = self._get_next_stage(current_stage_name)

                if next_stage_candidate:
                    self.logger.info(f"Run {run_id}, Flow {flow_id}: Transitioning to next stage '{next_stage_candidate}'.")
                    self.logger.info(f"Run {run_id}, Flow {flow_id}: Transitioning to next stage '{next_stage_candidate}'.")
                    current_stage_name = next_stage_candidate
                    current_attempt_number_for_stage = 1 # Reset for the new stage
                    agent_error_obj = None # Clear any previous error from a prior stage that might have been PROCEED_AS_IS
                else:
                    self.logger.info(f"Run {run_id}, Flow {flow_id}: Stage '{current_stage_name}' has no explicit next stage and no conditions met. Assuming end of flow.")
                    # This typically means success if it's the last stage in a sequence without explicit end markers being hit.
                    # However, if a stage is *supposed* to have a next_stage_id or conditional_transitions but doesn't, it's a plan defect.
                    # For now, assume it signals successful completion of this path. The loop will handle this.
                    current_stage_name = NEXT_STAGE_END_SUCCESS # Default to success if no other transition specified
                    current_stage_status = StageStatus.COMPLETED_SUCCESS if not self.shared_context.flow_has_warnings else StageStatus.COMPLETED_WITH_WARNINGS
                    agent_error_obj = None # Clear any previous error from a prior stage that might have been PROCEED_AS_IS

            except Exception as e: # Handles errors from _invoke_agent_for_stage or criteria_error
                self.logger.error(f"Run {run_id}: Exception during stage '{current_stage_name}' execution (Attempt {current_attempt_number_for_stage}): {e}", exc_info=True)
                if not resolved_inputs_for_error_handler: # If not set by criteria failure block
                    resolved_inputs_for_error_handler = self.shared_context.get_resolved_inputs_for_current_stage() or {}
                
                # MODIFIED: Selective deepcopy for shared_context_at_error
                shared_context_at_error: Optional[SharedContext] = None
                if self.shared_context:
                    try:
                        # Start with a shallow copy of the SharedContext object itself
                        shared_context_at_error = self.shared_context.model_copy(deep=False)
                        
                        # Now, selectively deepcopy self.shared_context.data, excluding known manager objects
                        # These manager keys are based on how build_shared_context_data in cli.py populates it.
                        manager_keys = ['llm_manager', 'prompt_manager', 'agent_provider', 'state_manager', 'metrics_store']
                        
                        copied_data = {}
                        for key, value in self.shared_context.data.items():
                            if key in manager_keys:
                                copied_data[key] = value  # Reference copy for managers
                            else:
                                copied_data[key] = copy.deepcopy(value) # Deepcopy for other data
                        
                        shared_context_at_error.data = copied_data
                        
                    except TypeError as e_pickle_retry:
                        self.logger.warning(f"Run {run_id}: Selective deepcopy of shared_context.data also failed ({e_pickle_retry}). Falling back to shallow copy of SharedContext entirely for error reporting. Some context details might be mutable references.", exc_info=True)
                        # Fallback: shallow copy the whole SharedContext object if selective deepcopy of .data also fails
                        shared_context_at_error = self.shared_context.model_copy(deep=False)
                    except Exception as e_copy_unexpected:
                         self.logger.error(f"Run {run_id}: Unexpected error during custom shared_context copy: {e_copy_unexpected}. Using shallow copy.", exc_info=True)
                         shared_context_at_error = self.shared_context.model_copy(deep=False)
                
                error_result: OrchestrationErrorResult = await self.error_handler_service.handle_stage_execution_error(
                    current_stage_name=current_stage_name,
                    flow_id=flow_id,
                    run_id=run_id,
                    current_plan=self.current_plan,
                    error=e, # Pass the original exception `e`
                    agent_id_that_erred=stage_spec.agent_id,
                    attempt_number=current_attempt_number_for_stage,
                    shared_context_at_error=shared_context_at_error, # Use the selectively copied context
                    resolved_inputs_at_failure=resolved_inputs_for_error_handler
                )

                self.shared_context.current_stage_status = None # Clear current stage status as we are handling its end/error
                agent_error_obj = error_result.updated_agent_error_details # Update loop's main error object

                if error_result.flow_pause_status != FlowPauseStatus.NOT_PAUSED:
                    self.logger.info(f"Run {run_id}, Flow {flow_id}: Flow paused by error handler. Status: {error_result.flow_pause_status}. Reviewer: {error_result.reviewer_output}")
                    
                    # Construct PausedRunDetails for state_manager
                    # Ensure all required fields are correctly sourced.
                    paused_details = PausedRunDetails(
                        run_id=run_id,
                        flow_id=flow_id,
                        current_master_plan_snapshot=self.current_plan.model_copy(deep=True) if self.current_plan else None,
                        paused_stage_id=current_stage_name, # Explicitly use current_stage_name from orchestrator
                        pause_status=error_result.flow_pause_status,
                        error_details=agent_error_obj, # Explicitly use agent_error_obj from orchestrator's scope
                        pause_reason_summary=error_result.reviewer_output.reasoning if error_result.reviewer_output and error_result.reviewer_output.reasoning else (agent_error_obj.message if agent_error_obj else "Flow paused by error handler."), # MODIFIED
                        reviewer_suggestion=error_result.reviewer_output,
                        last_stage_id=current_stage_name, # Redundant with paused_stage_id but often kept for clarity
                        last_stage_attempt_number=current_attempt_number_for_stage # Use orchestrator's current attempt number
                        # clarification_details and context_snapshot_ref are optional and handled if present
                    )
                    self.state_manager.save_paused_flow_state(paused_details)
                    # Emit metric for flow paused
                    self._emit_metric(MetricEventType.FLOW_PAUSED, flow_id, run_id, data={
                        "paused_at_stage_id": current_stage_name,
                        "pause_status": error_result.flow_pause_status.value,
                        "error_details": agent_error_obj.to_dict() if agent_error_obj else None,
                        "reviewer_suggestion": error_result.reviewer_output.model_dump(warnings=False) if error_result.reviewer_output else None
                    })
                    # return error_result.flow_pause_status, agent_error_obj # OLD RETURN
                    # New: save state and then return
                    self.state_manager.save_paused_flow_state(paused_details) # MODIFIED: Removed await
                    return error_result.flow_pause_status, agent_error_obj

                # If not paused, proceed based on next_stage_to_execute
                if error_result.next_stage_to_execute == current_stage_name and not (error_result.reviewer_output and error_result.reviewer_output.suggestion_type == ReviewerActionType.PROCEED_AS_IS):
                    self.logger.info(f"Run {run_id}, Flow {flow_id}: Retrying stage '{current_stage_name}' based on error handler result (next attempt: {current_attempt_number_for_stage}).")
                    
                    if error_result.modified_stage_inputs is not None:
                        actual_inputs_to_apply = None
                        # Check if modified_stage_inputs has the nested structure {"inputs": {...}}
                        if isinstance(error_result.modified_stage_inputs, dict) and \
                           list(error_result.modified_stage_inputs.keys()) == ["inputs"] and \
                           isinstance(error_result.modified_stage_inputs.get("inputs"), dict):
                            
                            actual_inputs_to_apply = error_result.modified_stage_inputs["inputs"]
                            self.logger.info(f"Run {run_id}: Unwrapped reviewer modified inputs for stage '{current_stage_name}'. Applying: {actual_inputs_to_apply}")
                        else:
                            # Assume modified_stage_inputs is already the flat dictionary of inputs
                            actual_inputs_to_apply = error_result.modified_stage_inputs
                            self.logger.info(f"Run {run_id}: Applying reviewer modified inputs (assumed flat) for stage '{current_stage_name}'. Applying: {actual_inputs_to_apply}")

                        if actual_inputs_to_apply is not None:
                            # Get the MasterStageSpec object for the current stage
                            # Ensure we are modifying the actual plan's stage spec
                            if self.current_plan and current_stage_name in self.current_plan.stages:
                                stage_spec_to_modify = self.current_plan.stages[current_stage_name]
                                stage_spec_to_modify.inputs = actual_inputs_to_apply.copy() # Use .copy() for safety
                                self.logger.info(f"Run {run_id}: Applied modified inputs to stage spec for '{current_stage_name}': {stage_spec_to_modify.inputs}")
                            else:
                                self.logger.warning(f"Run {run_id}: Could not update self.current_plan with modified inputs for stage '{current_stage_name}'. Plan or stage not found.")
                    
                    agent_error_obj = None 
                    self.shared_context.current_stage_id = current_stage_name
                    self.shared_context.current_stage_status = StageStatus.PENDING 
                    self.shared_context.current_attempt_number_for_stage = current_attempt_number_for_stage
                    continue 

                elif error_result.next_stage_to_execute == NEXT_STAGE_END_FAILURE:
                    self.logger.error(f"Run {run_id}, Flow {flow_id}: Error handler determined flow should fail. Error: {agent_error_obj.message if agent_error_obj else 'N/A'}")
                    await self.state_manager.update_status(run_id, StageStatus.COMPLETED_FAILURE, error_details=agent_error_obj)
                    return StageStatus.COMPLETED_FAILURE, agent_error_obj
                
                elif error_result.reviewer_output and error_result.reviewer_output.suggestion_type == ReviewerActionType.PROCEED_AS_IS and error_result.next_stage_to_execute == current_stage_name:
                    self.logger.info(f"Run {run_id}, Flow {flow_id}: Stage '{current_stage_name}' proceeding as per reviewer. Marking COMPLETED_WITH_WARNINGS.")
                    
                    output_for_context_proceed_as_is = error_result.updated_agent_error_details.output_payload_if_proceeding
                    if output_for_context_proceed_as_is is None:
                        output_for_context_proceed_as_is = {"message": "Proceeded as is after error", "error_details": error_result.updated_agent_error_details.to_dict()}
                    
                    output_key_for_proceed = stage_spec.output_context_path or current_stage_name
                    if self.shared_context.outputs is None: self.shared_context.outputs = {}
                    self.shared_context.outputs[output_key_for_proceed] = output_for_context_proceed_as_is
                    self.shared_context.add_stage_output_to_history(current_stage_name, output_for_context_proceed_as_is)

                    self.shared_context.flow_has_warnings = True
                    
                    self.state_manager.record_stage_end(
                        run_id=run_id, flow_id=flow_id, stage_id=current_stage_name, # MODIFIED: stage_name to stage_id
                        status=StageStatus.COMPLETED_WITH_WARNINGS, 
                        outputs=output_for_context_proceed_as_is, # MODIFIED: output to outputs
                        error_details=error_result.updated_agent_error_details 
                    )
                    self._emit_metric(MetricEventType.MASTER_STAGE_END, flow_id, run_id, stage_id=current_stage_name, 
                                      agent_id=stage_spec.agent_id, 
                                      data={"status": StageStatus.COMPLETED_WITH_WARNINGS.value, 
                                            "original_error": error_result.updated_agent_error_details.to_dict(),
                                            "output_summary": f"Proceeded as is. Output type: {type(output_for_context_proceed_as_is).__name__}"})
                    
                    current_stage_name = self._get_next_stage(current_stage_name) 
                    current_attempt_number_for_stage = 1 
                    agent_error_obj = None 
                    self.shared_context.current_stage_id = current_stage_name
                    self.shared_context.current_stage_status = StageStatus.PENDING if current_stage_name not in [NEXT_STAGE_END_SUCCESS, NEXT_STAGE_END_FAILURE] else None
                    self.shared_context.current_attempt_number_for_stage = 1
                    continue 
                
                elif error_result.next_stage_to_execute is None and error_result.flow_pause_status == FlowPauseStatus.NOT_PAUSED:
                    self.logger.error(f"Run {run_id}, Flow {flow_id}: Error handler returned no next stage and no pause signal. This is a critical error in handler logic. Failing flow.")
                    final_error_obj_logic = agent_error_obj or AgentErrorDetails(stage_id=current_stage_name, agent_id="Orchestrator", error_type="ErrorHandlerLogicError", message="Error handler failed to provide clear next step.")
                    # REMOVED: await self.state_manager.update_run_status(run_id, StageStatus.COMPLETED_FAILURE, error_details=final_error_obj_logic)
                    return StageStatus.COMPLETED_FAILURE, final_error_obj_logic
                
                else: 
                    self.logger.info(f"Run {run_id}, Flow {flow_id}: Error handler resulted in transitioning to stage '{error_result.next_stage_to_execute}'.")
                    current_stage_name = error_result.next_stage_to_execute
                    current_attempt_number_for_stage = 1 
                    agent_error_obj = None 
                    self.shared_context.current_stage_id = current_stage_name
                    self.shared_context.current_stage_status = StageStatus.PENDING if current_stage_name not in [NEXT_STAGE_END_SUCCESS, NEXT_STAGE_END_FAILURE] else None
                    self.shared_context.current_attempt_number_for_stage = 1
                    continue 
            
        # Loop completion checks
        if hops >= max_hops:
            self.logger.error(f"Run {run_id}: Maximum hop limit ({max_hops}) reached. Terminating flow as failure to prevent infinite loop.")
            final_error_max_hops = agent_error_obj or AgentErrorDetails(stage_id=str(current_stage_name), agent_id="Orchestrator", error_type="MaxHopsReached", message=f"Max hop limit of {max_hops} reached.")
            # REMOVED: await self.state_manager.update_run_status(run_id, StageStatus.COMPLETED_FAILURE, error_details=final_error_max_hops)
            return StageStatus.COMPLETED_FAILURE, final_error_max_hops
        
        # MODIFIED: Treat "FINAL_STEP" like NEXT_STAGE_END_SUCCESS
        if current_stage_name == NEXT_STAGE_END_SUCCESS or current_stage_name == "FINAL_STEP":
            self.logger.info(f"Run {run_id}: Flow completed successfully (terminated with '{current_stage_name}').")
            final_status_enum = StageStatus.COMPLETED_SUCCESS if not self.shared_context.flow_has_warnings else StageStatus.COMPLETED_WITH_WARNINGS
            self.state_manager.record_flow_end( 
                run_id=run_id,
                flow_id=self._current_flow_id,
                final_status=final_status_enum.value,
                final_outputs=self.shared_context.data.get("outputs")
            )
            return final_status_enum, agent_error_obj 
        
        elif current_stage_name == NEXT_STAGE_END_FAILURE: 
            self.logger.error(f"Run {run_id}: Flow ended in explicit failure state.")
            final_error = agent_error_obj or AgentErrorDetails(stage_id="UnknownStageLeadingToFailure", agent_id="Orchestrator", error_type="FlowFailure", message="Flow ended with explicit failure signal.")
            self.state_manager.record_flow_end( 
                run_id=run_id,
                flow_id=self._current_flow_id,
                final_status=StageStatus.COMPLETED_FAILURE.value,
                error_message=final_error.message if final_error else "Flow ended with explicit failure signal.",
                final_outputs=self.shared_context.data.get("outputs") 
            )
            return StageStatus.COMPLETED_FAILURE, final_error 
        
        self.logger.error(f"Run {run_id}: Flow loop terminated unexpectedly. Current stage: {current_stage_name}. Assuming failure.")
        final_error_fallback = agent_error_obj or AgentErrorDetails(stage_id=str(current_stage_name), agent_id="Orchestrator", error_type="UnexpectedFlowTermination", message="Flow loop ended without explicit success/failure signal.") 
        self.state_manager.record_flow_end( 
            run_id=run_id,
            flow_id=self._current_flow_id,
            final_status=StageStatus.COMPLETED_FAILURE.value,
            error_message=final_error_fallback.message if final_error_fallback else "Flow loop ended without explicit success/failure signal.",
            final_outputs=self.shared_context.data.get("outputs") 
        )
        return StageStatus.COMPLETED_FAILURE, final_error_fallback

    def _extract_output_and_metrics(self, agent_output_raw: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Extracts the primary output and any embedded metrics from agent\'s raw output."""
        if isinstance(agent_output_raw, tuple) and len(agent_output_raw) == 2 and isinstance(agent_output_raw[1], dict):
            # Assuming convention: (primary_output, metrics_dict)
            return agent_output_raw[0], agent_output_raw[1]
        return agent_output_raw, None # No separate metrics dict

    async def run(
        self,
        goal_str: Optional[str] = None,
        flow_yaml_path: Optional[str] = None,
        master_plan_id: Optional[str] = None,
        master_plan_obj: Optional[MasterExecutionPlan] = None, # For direct plan passing
        initial_context: Optional[Dict[str, Any]] = None, # Added initial_context
        run_id_override: Optional[str] = None, # Changed from run_id
        flow_id_override: Optional[str] = None, # To allow overriding flow_id if not from plan
        resume_context: Optional[ResumeContext] = None,
        max_hops: Optional[int] = None
    ) -> Tuple[StageStatus, Optional[SharedContext], Optional[AgentErrorDetails]]:
        """Executes a MasterExecutionPlan determined by goal, YAML path, ID, or direct object."""

        run_id = run_id_override or str(uuid.uuid4())
        self._current_run_id = run_id
        
        loaded_master_plan: Optional[MasterExecutionPlan] = None

        if master_plan_obj:
            self.logger.info(f"Run {run_id}: Using directly provided MasterExecutionPlan object.")
            loaded_master_plan = master_plan_obj
        elif goal_str:
            log_message_goal_plan = f"Run {run_id}: Generating MasterExecutionPlan from goal: {goal_str[:100]}..."
            self.logger.info(log_message_goal_plan)
            self.initial_goal_str = goal_str # Store initial goal for the run

            # Prepare MasterPlannerInput
            # The flow_id for a new plan from goal could be a new UUID or derived.
            # The plan_id will also be new.
            plan_id_from_goal = f"plan_goal_{str(uuid.uuid4())[:8]}"
            # Flow ID might be the same as plan ID for goal-generated plans, or a new UUID
            flow_id_for_goal_plan = flow_id_override or f"flow_goal_{str(uuid.uuid4())[:8]}"

            planner_input = MasterPlannerInput(
                user_goal=goal_str,
                master_plan_id=plan_id_from_goal,
                flow_id=flow_id_for_goal_plan,
                run_id=run_id
            )
            try:
                planner_agent_id = MasterPlannerAgent.AGENT_ID # Use agent's defined ID
                planner_callable = self.agent_provider.get(identifier=planner_agent_id)
                
                planner_input_summary = str(planner_input.model_dump())
                self.logger.info(f"Run {run_id}: Invoking MasterPlannerAgent ('{planner_agent_id}') with input: {planner_input_summary}")
                planner_output_raw = await planner_callable(inputs=planner_input, full_context=self.shared_context) # Ensure agent signature matches
                
                planner_output: Optional[MasterPlannerOutput] = None
                if isinstance(planner_output_raw, MasterPlannerOutput):
                    planner_output = planner_output_raw
                elif isinstance(planner_output_raw, dict): # Attempt to parse if it's a dict
                    try:
                        planner_output = MasterPlannerOutput(**planner_output_raw)
                    except Exception as e_parse_planner_out:
                        self.logger.error(f"Run {run_id}: Failed to parse MasterPlannerAgent output dict: {e_parse_planner_out}")
                        raise OrchestrationError(f"MasterPlannerAgent output parsing failed: {e_parse_planner_out}", run_id=run_id, agent_id=planner_agent_id) from e_parse_planner_out
                else:
                    self.logger.error(f"Run {run_id}: MasterPlannerAgent returned unexpected output type: {type(planner_output_raw)}. Expected MasterPlannerOutput or dict.")
                    raise OrchestrationError(f"MasterPlannerAgent returned unexpected output type: {type(planner_output_raw)}", run_id=run_id, agent_id=planner_agent_id)

                if not planner_output or not planner_output.master_plan_json:
                    self.logger.error(f"Run {run_id}: MasterPlannerAgent did not return a JSON plan.")
                    raise OrchestrationError("MasterPlannerAgent failed to generate a plan JSON.", run_id=run_id, agent_id=planner_agent_id)

                self.logger.info(f"Run {run_id}: MasterPlannerAgent generated JSON plan:\n{planner_output.master_plan_json[:500]}...")
                
                # UserGoalRequest is constructed from the goal_str.
                # If initial_context contained target_platform or key_constraints, they could be extracted here.
                user_goal_req_for_plan = UserGoalRequest(
                    goal_description=goal_str
                )
                
                # Parse the JSON plan string into a MasterExecutionPlan object.
                # The from_yaml method handles parsing of JSON as it's a subset of YAML.
                # The plan's 'id' should be present in the master_plan_json content.
                loaded_master_plan = MasterExecutionPlan.from_yaml(
                    yaml_text=planner_output.master_plan_json
                )
                
                # Assign the user_goal_request to the loaded plan.
                loaded_master_plan.original_request = user_goal_req_for_plan
                
                # The plan_id_from_goal (used in MasterPlannerInput) can be used to reconcile
                # or override the ID from the LLM if necessary, though usually, the ID from the
                # generated plan JSON is preferred. The orchestrator has later logic to
                # potentially override self.current_plan.id if master_plan_id (from CLI) is given.
                # For now, we assume the ID from planner_output.master_plan_json is the primary one.
                
                self.logger.info(f"Run {run_id}: Successfully parsed generated plan. Plan ID: {loaded_master_plan.id}, Start Stage: {loaded_master_plan.start_stage}") # Corrected to start_stage

            except NoAgentFoundForCategoryError as e_planner_not_found:
                self.logger.error(f"Run {run_id}: MasterPlannerAgent (\'{planner_agent_id}\') not found: {e_planner_not_found}", exc_info=True)
                # Return a specific error status or raise an OrchestrationError
                return StageStatus.COMPLETED_FAILURE, self.shared_context, AgentErrorDetails(message=f"MasterPlannerAgent not found: {e_planner_not_found}", agent_id=planner_agent_id, error_type="AgentNotFound")
            except Exception as e_planner:
                self.logger.error(f"Run {run_id}: Error during plan generation by MasterPlannerAgent: {e_planner}", exc_info=True)
                # Return a specific error status or raise an OrchestrationError
                return StageStatus.COMPLETED_FAILURE, self.shared_context, AgentErrorDetails(message=f"Plan generation failed: {e_planner}", agent_id="system.master_planner_agent", error_type="PlanGenerationError")
        
        elif flow_yaml_path:
            self.logger.info(f"Run {run_id}: Loading MasterExecutionPlan from YAML file: {flow_yaml_path}")
            try:
                yaml_content = Path(flow_yaml_path).read_text()
                # from_yaml expects plan_id and user_goal_request.
                # plan_id could be derived from filename or a default.
                # user_goal_request would be constructed with the initial_context.
                # For now, let\'s assume the YAML itself contains an ID, or we generate one.
                # And UserGoalRequest is built from initial_context.
                # A more robust from_yaml might be needed if it\'s not extracting ID from YAML.
                
                temp_parsed_yaml_for_id = yaml.safe_load(yaml_content)
                plan_id_from_yaml = temp_parsed_yaml_for_id.get("id", f"plan_yaml_{Path(flow_yaml_path).stem}")
                
                user_goal_req_for_yaml_plan = UserGoalRequest(goal=f"Execution from YAML: {Path(flow_yaml_path).name}", initial_context=initial_context or {})

                loaded_master_plan = MasterExecutionPlan.from_yaml(
                    yaml_text=yaml_content, 
                    plan_id=plan_id_from_yaml,
                    user_goal_request=user_goal_req_for_yaml_plan
                )
                self.logger.info(f"Run {run_id}: Successfully loaded plan from YAML. Plan ID: {loaded_master_plan.id}, Start Stage: {loaded_master_plan.start_stage_id}")
                self.initial_goal_str = loaded_master_plan.original_request.goal_description if loaded_master_plan.original_request else f"YAML Execution: {Path(flow_yaml_path).name}"

            except FileNotFoundError:
                self.logger.error(f"Run {run_id}: Flow YAML file not found: {flow_yaml_path}", exc_info=True)
                return StageStatus.COMPLETED_FAILURE, self.shared_context, AgentErrorDetails(message=f"Flow YAML file not found: {flow_yaml_path}", error_type="FileNotFound")
            except Exception as e_yaml:
                self.logger.error(f"Run {run_id}: Error loading plan from YAML \'{flow_yaml_path}\': {e_yaml}", exc_info=True)
                return StageStatus.COMPLETED_FAILURE, self.shared_context, AgentErrorDetails(message=f"Error loading plan from YAML: {e_yaml}", error_type="PlanLoadError")

        elif master_plan_id:
            self.logger.info(f"Run {run_id}: Attempting to load MasterExecutionPlan by ID: {master_plan_id}")
            # TODO: Implement loading plan by ID (e.g., from ChromaDB via ProjectChromaManagerAgent or file system)
            # This would involve:
            # 1. Getting ProjectChromaManagerAgent (or a similar service).
            # 2. Calling a method like `load_plan(plan_id)` which returns the plan YAML or object.
            # 3. If YAML, parse using MasterExecutionPlan.from_yaml().
            self.logger.warning(f"Run {run_id}: Loading plan by ID (\'{master_plan_id}\') is not yet fully implemented. Proceeding as if plan not found.")
            return StageStatus.COMPLETED_FAILURE, self.shared_context, AgentErrorDetails(message=f"Loading plan by ID \'{master_plan_id}\' not implemented.", error_type="NotImplementedError")
            # Example placeholder for future:
            # try:
            #     # loaded_master_plan = await self._load_plan_by_id(master_plan_id)
            #     pass # Placeholder
            # except Exception as e_load_id:
            #     self.logger.error(f"Run {run_id}: Failed to load plan by ID \'{master_plan_id}\': {e_load_id}", exc_info=True)
            #     return StageStatus.COMPLETED_FAILURE, self.shared_context, AgentErrorDetails(message=f"Failed to load plan by ID \'{master_plan_id}\': {e_load_id}", error_type="PlanLoadError")


        if not loaded_master_plan:
            self.logger.error(f"Run {run_id}: No MasterExecutionPlan could be determined. Orchestrator cannot run.")
            return StageStatus.COMPLETED_FAILURE, self.shared_context, AgentErrorDetails(message="No execution plan provided or generated.", error_type="OrchestrationSetupError")
        
        self.current_plan = loaded_master_plan
        
        # Determine final flow_id for this run
        # Priority: override > plan.flow_id (if plan was loaded and has one) > default
        final_flow_id = flow_id_override or getattr(self.current_plan, 'flow_id', None) or getattr(self.current_plan, 'id', None) or f"flow_{run_id}"
        self._current_flow_id = final_flow_id
        
        # Update plan\'s internal ID and flow_id if they were dynamic or need overriding
        # This ensures consistency if a plan loaded from YAML/ID had a different ID than expected.
        if self.current_plan.id != master_plan_id and master_plan_id: # If an ID was given but plan\'s is different
            self.logger.warning(f"Run {run_id}: Overriding loaded plan ID \'{self.current_plan.id}\' with provided master_plan_id \'{master_plan_id}\'.")
            self.current_plan.id = master_plan_id
        
        # Ensure the plan has a user_goal_request, create a default if not.
        if not self.current_plan.original_request:
            self.current_plan.original_request = UserGoalRequest(
                goal_description=self.initial_goal_str or f"Execution of plan {self.current_plan.id}",
            )

        # Initialize SharedContext for this run
        self.shared_context = SharedContext(run_id=run_id, flow_id=final_flow_id) # Use final_flow_id
        
        # Populate SharedContext with initial_context from CLI (e.g., mcp_root_workspace_path)
        if initial_context:
            self.shared_context.update_data(initial_context)
        
        # Ensure project_id and project_root_path from orchestrator's config are in shared_context.data
        # if not already provided by initial_context.
        if 'project_id' not in self.shared_context.data and self.config.get('project_id'):
            self.shared_context.data['project_id'] = self.config['project_id']
            self.logger.info(f"Run {run_id}: Set shared_context.data['project_id'] from orchestrator config: {self.config['project_id']}")
        
        if 'project_root_path' not in self.shared_context.data and self.config.get('project_root_path'):
            self.shared_context.data['project_root_path'] = self.config['project_root_path']
            self.logger.info(f"Run {run_id}: Set shared_context.data['project_root_path'] from orchestrator config: {self.config['project_root_path']}")
        
        self.logger.info(f"Run {run_id}: Initial self.shared_context.data['project_id'] = {self.shared_context.data.get('project_id')}")
        
        # Re-initialize services that depend on shared_context
        self.context_resolver.shared_context = self.shared_context
        # ConditionEvaluationService and SuccessCriteriaService receive shared_context per call.
        
        self.logger.info(f"Run {run_id}, Flow {final_flow_id}: Starting execution of master plan \'{self.current_plan.id}\'. Goal: {self.initial_goal_str or 'N/A'}")


        start_stage_name: Optional[str] = None
        initial_attempt_number = 1

        if resume_context:
            self.logger.info(f"Run {run_id}: Resuming flow '{final_flow_id}' from stage '{resume_context.paused_at_stage_id}'.")
            self._emit_metric(MetricEventType.FLOW_RESUME, final_flow_id, run_id, data=resume_context.model_dump(warnings=False))
            # Load context snapshot if available
            if resume_context.context_snapshot_ref:
                try:
                    loaded_context = await self.state_manager.load_context_snapshot(resume_context.context_snapshot_ref)
                    if loaded_context:
                        self.shared_context = loaded_context
                        self.logger.info(f"Run {run_id}: Successfully loaded context snapshot '{resume_context.context_snapshot_ref}'.")
                    else:
                        self.logger.warning(f"Run {run_id}: Context snapshot '{resume_context.context_snapshot_ref}' not found or empty. Resuming with default context.")
                except Exception as e_snap_load:
                    self.logger.error(f"Run {run_id}: Error loading context snapshot '{resume_context.context_snapshot_ref}': {e_snap_load}. Resuming with default context.", exc_info=True)
            
            start_stage_name = resume_context.paused_at_stage_id
            # If resuming, and there was a last_stage_attempt_number, it implies the next attempt for that stage
            # The error handler deals with attempt numbers that *failed*. If we are resuming *after* a pause (not retry suggestion)
            # then the attempt number should be for the *next* execution of that stage.
            initial_attempt_number = (resume_context.last_stage_attempt_number or 0) + 1 
            if resume_context.modified_inputs_for_resume:
                self.logger.info(f"Run {run_id}: Applying modified inputs from resume_context for stage '{start_stage_name}'.")
                if start_stage_name and start_stage_name in self.current_plan.stages:
                    self.current_plan.stages[start_stage_name].inputs = resume_context.modified_inputs_for_resume
                else:
                    self.logger.error(f"Run {run_id}: Cannot apply modified inputs from resume_context. Stage '{start_stage_name}' not found in plan.")
        else:
            # Start of a new flow
            start_stage_name = self.current_plan.start_stage
            self.logger.info(f"Run {run_id}: Starting new flow '{final_flow_id}' from stage '{start_stage_name}'.")
            self._emit_metric(MetricEventType.FLOW_START, final_flow_id, run_id, data={"start_stage_id": start_stage_name})
            self.state_manager.record_flow_start(run_id, final_flow_id, initial_context=self.shared_context.data if self.shared_context else None)

        if not start_stage_name:
            self.logger.error(f"Run {run_id}: No start stage identified for flow '{final_flow_id}'. Cannot execute.")
            await self.state_manager.update_status(run_id, StageStatus.COMPLETED_FAILURE, error_details=AgentErrorDetails(message="No start stage defined.", agent_id="Orchestrator", error_type="PlanValidationError"))
            return StageStatus.COMPLETED_FAILURE, self.shared_context, AgentErrorDetails(message="No start stage defined.")

        # Pass the initial_attempt_number to _execute_flow_loop via resume_context if applicable.
        # _execute_flow_loop's internal `current_attempt_number_for_stage` will be initialized with it.
        if resume_context: # Update resume_context if we synthesized an attempt number
            resume_context.last_stage_attempt_number = initial_attempt_number -1 # Store the one that led to pause/next try

        effective_max_hops = max_hops if max_hops is not None else DEFAULT_MAX_HOPS

        final_status, agent_error_details = await self._execute_flow_loop(
            flow_id=final_flow_id,
            run_id=run_id,
            start_stage_name=start_stage_name,
            max_hops=effective_max_hops,
            resume_context=resume_context 
        )

        self.logger.info(f"Run {run_id}: Flow '{final_flow_id}' finished with status: {final_status}.")
        event_type_end = MetricEventType.FLOW_END # MODIFIED HERE - success/failure is in data payload
        self._emit_metric(event_type_end, final_flow_id, run_id, data={
            "final_status": final_status.value,
            "error_details": agent_error_details.to_dict() if agent_error_details else None,
            "flow_has_warnings": self.shared_context.flow_has_warnings
        })
        
        # The run record status should have been updated by _execute_flow_loop (via state_manager calls within it)
        # or by the final state_manager.update_run_record_status calls in _execute_flow_loop.
        # No need to call update_run_status here again unless it's a fallback.

        return final_status, self.shared_context, agent_error_details