"""Minimal execution runtime skeleton (Phase-6 scaffold).

This module will gradually evolve into the production‐grade orchestrator that
executes Flow YAML graphs.  For now we expose a *SyncOrchestrator* that can run
very simple flows in-process so we can start adding tests and build up the API
surface incrementally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Callable, Awaitable, Union

import datetime as _dt
from datetime import datetime, timezone
import yaml
from pydantic import BaseModel, Field, ConfigDict
from chungoid.utils.agent_resolver import AgentProvider, RegistryAgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.common_enums import StageStatus, FlowPauseStatus
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.flows import PausedRunDetails
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec
# Import for reviewer agent and its schemas
from chungoid.runtime.agents.system_master_planner_reviewer_agent import MasterPlannerReviewerAgent # Assuming AGENT_ID is on class
from chungoid.schemas.agent_master_planner_reviewer import MasterPlannerReviewerInput, MasterPlannerReviewerOutput, ReviewerActionType

import logging
import traceback
import copy
import inspect
from unittest.mock import AsyncMock
import asyncio
import uuid
import collections
# import dpath # For safe nested dictionary access by path string - REMOVED AS NOT USED

__all__ = [
    "StageSpec",
    "ExecutionPlan",
    "SyncOrchestrator",
    "AsyncOrchestrator",
]


# ---------------------------------------------------------------------------
# DSL → Python models
# ---------------------------------------------------------------------------


class StageSpec(BaseModel):
    """Specification of a single stage inside a flow."""

    agent_id: str = Field(..., description="ID of the agent to invoke for this stage")
    inputs: Optional[dict] = Field(None, description="Input parameters for the agent")
    condition: Optional[str] = Field(None, description="Condition for branching")
    next_stage_true: Optional[str] = Field(None, description="Next stage if condition is true")
    next_stage_false: Optional[str] = Field(None, description="Next stage if condition is false")
    next_stage: Optional[str] = Field(None, description="Next stage or conditional object")
    number: Optional[float] = Field(None, description="Unique stage number for status tracking")
    on_error: Optional[Any] = Field(None, description="Error handler stage or conditional object")
    parallel_group: Optional[str] = Field(None, description="Group name for parallel execution")
    plugins: Optional[List[str]] = Field(None, description="List of plugin names to apply at this stage")
    extra: Optional[dict] = Field(None, description="Arbitrary extra data for extensibility")

    model_config = ConfigDict(extra="forbid")


class ExecutionPlan(BaseModel):
    """Validated, structured representation of the Flow YAML."""

    id: str
    created: _dt.datetime = Field(default_factory=_dt.datetime.utcnow)
    start_stage: str
    stages: Dict[str, StageSpec]

    @classmethod
    def from_yaml(cls, yaml_text: str, flow_id: str | None = None) -> "ExecutionPlan":
        """Parse the *yaml_text* of a FlowCard and convert it to a plan."""

        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError("Flow YAML must map keys → values")

        try:
            _validate_dsl(data)
        except Exception as exc:
            raise ValueError(f"Flow DSL validation error: {exc}") from exc

        if "stages" not in data or "start_stage" not in data:
            raise ValueError("Flow YAML missing required 'stages' or 'start_stage' key")

        raw_stages_from_yaml = data["stages"]
        transformed_stages_for_pydantic = {}
        for stage_name, stage_data_dict_from_yaml in raw_stages_from_yaml.items():
            current_stage_attrs = stage_data_dict_from_yaml.copy()
            if 'next' in current_stage_attrs and isinstance(current_stage_attrs['next'], dict):
                next_obj_from_yaml = current_stage_attrs.pop('next')
                # Map fields from YAML 'next' object to StageSpec fields
                if 'condition' in next_obj_from_yaml:
                    current_stage_attrs['condition'] = next_obj_from_yaml['condition']
                if 'true' in next_obj_from_yaml: # YAML uses 'true'
                    current_stage_attrs['next_stage_true'] = next_obj_from_yaml['true']
                if 'false' in next_obj_from_yaml: # YAML uses 'false'
                    current_stage_attrs['next_stage_false'] = next_obj_from_yaml['false']
                # Ensure StageSpec.next (the simple string one) is None for conditional branches
                current_stage_attrs['next_stage'] = None
            elif 'next' in current_stage_attrs and isinstance(current_stage_attrs['next'], str):
                # It's a simple string next, ensure it's assigned to StageSpec.next_stage
                # If the key in YAML is already 'next_stage', this won't hurt.
                # If it's 'next', we move it to 'next_stage'.
                if current_stage_attrs.get('next_stage') is None: # Avoid overwriting if next_stage already set
                    current_stage_attrs['next_stage'] = current_stage_attrs.pop('next')
                elif 'next' in current_stage_attrs: # if next_stage was already there, just remove 'next' if it exists
                    current_stage_attrs.pop('next')
            
            transformed_stages_for_pydantic[stage_name] = current_stage_attrs

        return cls(
            id=flow_id or "<unknown>",
            start_stage=data["start_stage"],
            stages=transformed_stages_for_pydantic, # Pass dict of dicts, Pydantic will create StageSpec objects
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
            return True # No condition means proceed

        self.logger.debug(f"Parsing condition: {condition_str}")
        try:
            parts = []
            comparator = None
            # Order matters for multi-character operators
            if '>=' in condition_str:
                parts = condition_str.split('>=', 1)
                comparator = '>='
            elif '<=' in condition_str:
                parts = condition_str.split('<=', 1)
                comparator = '<='
            elif '==' in condition_str: # Should be before single char '=' if that were supported
                parts = condition_str.split('==', 1)
                comparator = '=='
            elif '!=' in condition_str:
                parts = condition_str.split('!=', 1)
                comparator = '!='
            elif '>' in condition_str:
                parts = condition_str.split('>', 1)
                comparator = '>'
            elif '<' in condition_str:
                parts = condition_str.split('<', 1)
                comparator = '<'
            else:
                self.logger.error(f"Unsupported condition format or unknown operator: {condition_str}")
                return False

            if len(parts) != 2:
                self.logger.error(f"Invalid condition structure: {condition_str}")
                return False

            var_path_str = parts[0].strip()
            expected_value_str = parts[1].strip()

            current_val = context
            for key in var_path_str.split('.'):
                if isinstance(current_val, dict) and key in current_val:
                    current_val = current_val[key]
                elif isinstance(current_val, list) and key.isdigit():
                    try:
                        current_val = current_val[int(key)]
                    except IndexError:
                        self.logger.warning(f"Index out of bounds for '{key}' in path '{var_path_str}'.")
                        return False
                else:
                    self.logger.warning(f"Condition variable path '{var_path_str}' (key: '{key}') not fully found in context.")
                    return False
            
            numeric_comparators = ['>', '<', '>=', '<=']
            is_numeric_comparison = comparator in numeric_comparators

            if is_numeric_comparison:
                try:
                    val1 = float(current_val) # Attempt to convert current_val to float
                    val2 = float(expected_value_str.strip("'\"")) # Attempt to convert expected_value_str to float
                    
                    self.logger.debug(f"Numeric condition check: {val1} {comparator} {val2}")
                    if comparator == '>': return val1 > val2
                    if comparator == '<': return val1 < val2
                    if comparator == '>=': return val1 >= val2
                    if comparator == '<=': return val1 <= val2
                except ValueError:
                    self.logger.warning(f"Type mismatch for numeric comparison: '{current_val}' vs '{expected_value_str}'. Condition evaluates to False.")
                    return False # If conversion to float fails for numeric comparison
            else: # Handling '==' and '!='
                try:
                    coerced_expected_value = expected_value_str.strip("'\"") # Default to string
                    if isinstance(current_val, bool):
                        coerced_expected_value = expected_value_str.lower() in ['true', '1', 'yes']
                    elif isinstance(current_val, int):
                        coerced_expected_value = int(expected_value_str.strip("'\""))
                    elif isinstance(current_val, float):
                        coerced_expected_value = float(expected_value_str.strip("'\""))
                    # If current_val is a string, coerced_expected_value remains a string as per default

                    self.logger.debug(f"Equality condition check: '{current_val}' ({type(current_val)}) {comparator} '{coerced_expected_value}' ({type(coerced_expected_value)})")
                    if comparator == '==':
                        return current_val == coerced_expected_value
                    elif comparator == '!=':
                        return current_val != coerced_expected_value
                except ValueError as e:
                    self.logger.error(f"Type conversion error for expected value in '=='/'=' condition '{condition_str}': {e}. Treating as unequal.")
                    return True if comparator == '!=' else False # Default to not equal on conversion error for ==/!=
            
            return False # Fallback, should ideally be covered by above

        except Exception as e:
            self.logger.exception(f"Error evaluating condition '{condition_str}': {e}")
            return False # Default to false on any error

    def run(self, plan: ExecutionPlan, context: Dict[str, Any]) -> List[str]:
        # This is a placeholder for the synchronous orchestrator logic
        self.logger.info("SyncOrchestrator.run called (placeholder)")
        # Simple sequential execution for now, ignoring conditions
        visited_stages: List[str] = [] # List to track visited stages
        current_stage_name = plan.start_stage
        max_hops = len(plan.stages) + 5 # Safety break
        hops = 0

        while current_stage_name and hops < max_hops:
            hops += 1
            if hops >= max_hops:
                self.logger.warning("Max hops reached, breaking execution.")
                break

            stage = plan.stages.get(current_stage_name)
            if not stage:
                self.logger.error(f"Stage '{current_stage_name}' not found in plan. Aborting.")
                break

            self.logger.info(f"Executing stage: {current_stage_name} (Agent: {stage.agent_id})")
            visited_stages.append(current_stage_name) # Add visited stage to list
            # Placeholder for actual agent execution and context update
            # context['outputs'][current_stage_name] = {"message": f"Output from {current_stage_name}"}
            
            if stage.condition:
                if self._parse_condition(stage.condition, context):
                    current_stage_name = stage.next_stage_true
                else:
                    current_stage_name = stage.next_stage_false
            else:
                current_stage_name = stage.next_stage
        
        return visited_stages # Return list of visited stages


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

    def _get_next_stage(self, current_stage_name: str, context: Dict[str, Any]) -> str | None:
        stage_def = self.pipeline_def.stages[current_stage_name]
        next_stage = stage_def.next_stage

        if isinstance(next_stage, dict) and next_stage.get("condition"):
            cond = next_stage["condition"]
            true_stage = next_stage["true"]
            false_stage = next_stage["false"]
            self.logger.debug(f"Evaluating condition '{cond}' for stage '{current_stage_name}'")
            condition_met = SyncOrchestrator._eval_condition_expr(cond, context)
            next_stage = true_stage if condition_met else false_stage
            self.logger.debug(f"Condition result: {condition_met}, next stage: '{next_stage}'")
        elif isinstance(next_stage, str):
            self.logger.debug(f"Direct next stage: '{next_stage}'")
        elif next_stage is None:
            self.logger.debug(f"No next stage defined after '{current_stage_name}'.")
        else:
            self.logger.warning(f"Invalid 'next' field type ({type(next_stage)}) for stage '{current_stage_name}'. Ending flow.")
            next_stage = None

        return next_stage


class AsyncOrchestrator(BaseOrchestrator):
    """Asynchronous orchestrator capable of running complex flows with conditions and context passing."""

    def __init__(
        self,
        pipeline_def: MasterExecutionPlan,
        config: Dict[str, Any],
        agent_provider: AgentProvider,
        state_manager: StateManager
    ):
        super().__init__(pipeline_def, config)
        self.agent_provider = agent_provider
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        self.context_history: Dict[str, Any] = {}
        self.visited_stages_in_current_run: List[str] = []
        self.run_status_updates: List[Dict[str, Any]] = []
        self.logger.debug(f"Orchestrator.run: Initialized self.run_status_updates: {type(self.run_status_updates)}")

    def _evaluate_criterion(self, criterion: str, stage_outputs: Dict[str, Any]) -> bool:
        """Evaluates a single success criterion string against stage_outputs."""
        self.logger.debug(f"Evaluating criterion: '{criterion}' against outputs: {stage_outputs}")
        
        # EXISTS Check (e.g., "outputs.some_key EXISTS")
        if criterion.upper().endswith(" EXISTS"):
            path_to_check = criterion[:criterion.upper().rfind(" EXISTS")].strip()
            current_val = stage_outputs
            try:
                for key_part in path_to_check.split('.'):
                    if isinstance(current_val, dict):
                        current_val = current_val[key_part]
                    elif isinstance(current_val, list) and key_part.isdigit():
                        current_val = current_val[int(key_part)]
                    else:
                        self.logger.debug(f"Criterion '{criterion}' FAILED: Path '{path_to_check}' not fully found (part: {key_part}).")
                        return False # Path part not found
                self.logger.debug(f"Criterion '{criterion}' PASSED (EXISTS check). Value at path: {current_val}")
                return True # Path exists
            except (KeyError, IndexError, TypeError):
                self.logger.debug(f"Criterion '{criterion}' FAILED: Path '{path_to_check}' not found (exception).")
                return False # Path does not exist

        # Simple Comparison (e.g., "outputs.metric == true", "outputs.count > 0")
        # Re-use parts of _parse_condition logic but scoped to stage_outputs
        try:
            parts = []
            comparator = None
            supported_comparators = {
                '>=': lambda a, b: a >= b,
                '<=': lambda a, b: a <= b,
                '==': lambda a, b: a == b,
                '!=': lambda a, b: a != b,
                '>': lambda a, b: a > b,
                '<': lambda a, b: a < b,
            }
            for op_str in supported_comparators.keys(): # Check longer ops first
                if op_str in criterion:
                    parts = criterion.split(op_str, 1)
                    comparator = op_str
                    break
            
            if not comparator or len(parts) != 2:
                self.logger.warning(f"Unsupported criterion format or unknown operator: {criterion}. Evaluates to FALSE.")
                return False

            path_str = parts[0].strip()
            expected_value_literal_str = parts[1].strip()

            actual_val = stage_outputs
            for key in path_str.split('.'):
                if isinstance(actual_val, dict) and key in actual_val:
                    actual_val = actual_val[key]
                elif isinstance(actual_val, list) and key.isdigit():
                    actual_val = actual_val[int(key)]
                else:
                    self.logger.debug(f"Criterion '{criterion}' FAILED: Path '{path_str}' not fully found in outputs.")
                    return False
            
            # Attempt to coerce expected_value_literal_str to type of actual_val
            coerced_expected_val: Any
            if isinstance(actual_val, bool):
                coerced_expected_val = expected_value_literal_str.lower() in ['true', '1', 'yes']
            elif isinstance(actual_val, int):
                coerced_expected_val = int(expected_value_literal_str.strip("'\""))
            elif isinstance(actual_val, float):
                coerced_expected_val = float(expected_value_literal_str.strip("'\""))
            else: # Default to string comparison
                coerced_expected_val = expected_value_literal_str.strip("'\"")

            result = supported_comparators[comparator](actual_val, coerced_expected_val)
            self.logger.debug(f"Criterion '{criterion}' evaluation: '{actual_val}' {comparator} '{coerced_expected_val}' -> {result}")
            return result

        except Exception as e:
            self.logger.warning(f"Error evaluating criterion '{criterion}': {e}. Evaluates to FALSE.", exc_info=True)
            return False

    async def _check_success_criteria(
        self, 
        stage_name: str, 
        stage_spec: MasterStageSpec, 
        context: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Checks all success_criteria for a given stage. Returns (all_passed, list_of_failed_criteria)."""
        if not stage_spec.success_criteria:
            return True, [] # No criteria means success

        all_passed = True
        failed_criteria: List[str] = []
        stage_outputs = context.get('outputs', {}).get(stage_name, {})

        self.logger.info(f"Checking {len(stage_spec.success_criteria)} success criteria for stage '{stage_name}'.")
        for criterion_str in stage_spec.success_criteria:
            if not self._evaluate_criterion(criterion_str, stage_outputs):
                all_passed = False
                failed_criteria.append(criterion_str)
        
        if not all_passed:
            self.logger.warning(f"Stage '{stage_name}' failed success criteria check. Failed criteria: {failed_criteria}")
        else:
            self.logger.info(f"All success criteria passed for stage '{stage_name}'.")
        return all_passed, failed_criteria

    def _parse_condition(self, condition_str: str, context: Dict[str, Any]) -> bool:
        if not condition_str:
            return True
        self.logger.debug(f"Parsing condition: {condition_str}")
        try:
            parts = []
            comparator = None
            # Order matters for multi-character operators
            if '>=' in condition_str:
                parts = condition_str.split('>=', 1)
                comparator = '>='
            elif '<=' in condition_str:
                parts = condition_str.split('<=', 1)
                comparator = '<='
            elif '==' in condition_str: # Should be before single char '=' if that were supported
                parts = condition_str.split('==', 1)
                comparator = '=='
            elif '!=' in condition_str:
                parts = condition_str.split('!=', 1)
                comparator = '!='
            elif '>' in condition_str:
                parts = condition_str.split('>', 1)
                comparator = '>'
            elif '<' in condition_str:
                parts = condition_str.split('<', 1)
                comparator = '<'
            else:
                self.logger.error(f"Unsupported condition format or unknown operator: {condition_str}")
                return False

            if len(parts) != 2:
                self.logger.error(f"Invalid condition structure: {condition_str}")
                return False

            var_path_str = parts[0].strip()
            expected_value_str = parts[1].strip()

            current_val = context
            for key in var_path_str.split('.'):
                if isinstance(current_val, dict) and key in current_val:
                    current_val = current_val[key]
                elif isinstance(current_val, list) and key.isdigit():
                    try:
                        current_val = current_val[int(key)]
                    except IndexError:
                        self.logger.warning(f"Index out of bounds for '{key}' in path '{var_path_str}'.")
                        return False
                else:
                    self.logger.warning(f"Condition variable path '{var_path_str}' (key: '{key}') not fully found in context.")
                    return False
            
            numeric_comparators = ['>', '<', '>=', '<=']
            is_numeric_comparison = comparator in numeric_comparators

            if is_numeric_comparison:
                try:
                    val1 = float(current_val) # Attempt to convert current_val to float
                    val2 = float(expected_value_str.strip("'\"")) # Attempt to convert expected_value_str to float
                    
                    self.logger.debug(f"Numeric condition check: {val1} {comparator} {val2}")
                    if comparator == '>': return val1 > val2
                    if comparator == '<': return val1 < val2
                    if comparator == '>=': return val1 >= val2
                    if comparator == '<=': return val1 <= val2
                except ValueError:
                    self.logger.warning(f"Type mismatch for numeric comparison: '{current_val}' vs '{expected_value_str}'. Condition evaluates to False.")
                    return False # If conversion to float fails for numeric comparison
            else: # Handling '==' and '!='
                try:
                    coerced_expected_value = expected_value_str.strip("'\"") # Default to string
                    if isinstance(current_val, bool):
                        coerced_expected_value = expected_value_str.lower() in ['true', '1', 'yes']
                    elif isinstance(current_val, int):
                        coerced_expected_value = int(expected_value_str.strip("'\""))
                    elif isinstance(current_val, float):
                        coerced_expected_value = float(expected_value_str.strip("'\""))
                    # If current_val is a string, coerced_expected_value remains a string as per default

                    self.logger.debug(f"Equality condition check: '{current_val}' ({type(current_val)}) {comparator} '{coerced_expected_value}' ({type(coerced_expected_value)})")
                    if comparator == '==':
                        return current_val == coerced_expected_value
                    elif comparator == '!=':
                        return current_val != coerced_expected_value
                except ValueError as e:
                    self.logger.error(f"Type conversion error for expected value in '=='/'=' condition '{condition_str}': {e}. Treating as unequal.")
                    return True if comparator == '!=' else False # Default to not equal on conversion error for ==/!=
            
            return False # Fallback, should ideally be covered by above

        except Exception as e:
            self.logger.exception(f"Error evaluating condition '{condition_str}': {e}")
            return False

    def _resolve_input_values(self, inputs_spec: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves input values specified in inputs_spec against the context_data.
        Supports literal values and path-based lookups (e.g., "context.some.value").
        """
        resolved_inputs = {}
        if not isinstance(inputs_spec, dict):
            self.logger.warning(f"_resolve_input_values expected inputs_spec to be a dict, got {type(inputs_spec)}. Returning empty resolved inputs.")
            return {}

        for input_name, context_path in inputs_spec.items():
            resolved_value = context_path # Default to literal

            # Ensure context_path is a string before attempting string operations or path resolution
            if not isinstance(context_path, str):
                self.logger.warning(
                    f"Input '{input_name}' for stage has a non-string context_path value: {context_path} (type: {type(context_path)}). Using it as a literal."
                )
                resolved_inputs[input_name] = context_path
                continue

            # Now we are sure context_path is a string.
            context_path_lower = context_path.lower()
            is_special_previous_output = (input_name.lower() == "previous_stage_outputs" and \
                                          not context_path_lower.startswith("context."))

            if is_special_previous_output:
                # This is the special directive to fetch previous outputs by ID
                if context_path in context_data.get('outputs', {}):
                    resolved_value = context_data['outputs'][context_path]
                    self.logger.info(f"Made outputs of previous stage '{context_path}' available as 'resolved_previous_stage_output_data' for stage '{input_name}'.")
                else:
                    self.logger.warning(f"Outputs for previous stage ID '{context_path}' (specified in inputs for key '{input_name}') not found in context.outputs for stage '{input_name}'. 'resolved_previous_stage_output_data' will be None.")
            else:
                # Resolve the context path against the context_data
                current_val = context_data
                valid_path = True
                path_parts_to_resolve = context_path.split('.')
                
                # If path starts with "context.", actual resolution path is from the second part onwards
                if path_parts_to_resolve and path_parts_to_resolve[0] == "context":
                    path_parts_to_resolve = path_parts_to_resolve[1:]
                
                try:
                    for part in path_parts_to_resolve:
                        if isinstance(current_val, dict):
                            current_val = current_val[part]
                        elif isinstance(current_val, list) and part.isdigit():
                            current_val = current_val[int(part)]
                        elif hasattr(current_val, part): # For object attribute access
                            current_val = getattr(current_val, part)
                        else:
                            self.logger.warning(f"Path part '{part}' not found or invalid access in '{context_path}' for key '{input_name}'.")
                            current_val = context_path # Fallback to literal string
                            valid_path = False
                            break
                    resolved_value = current_val
                except (KeyError, IndexError, AttributeError, TypeError) as e: # Added TypeError for non-subscriptable/non-getattrable
                    self.logger.warning(f"Error resolving context path '{context_path}' for input key '{input_name}': {e}. Using literal value.")
                    resolved_value = context_path # Fallback to literal string
            
            resolved_inputs[input_name] = resolved_value
        return resolved_inputs

    async def _execute_loop(self, start_stage_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        current_stage_name: Optional[str] = start_stage_name
        visited_stages: List[str] = []  # For loop detection or logging
        max_hops = len(self.pipeline_def.stages) + 10 # Safety break for complex loops

        if 'outputs' not in context:
            context['outputs'] = {}
        if 'errors' not in context:
            context['errors'] = {}
        if 'status_updates' not in context:
            context['status_updates'] = []

        run_id_for_status = context.get("run_id", str(uuid.uuid4())) # Get or generate run_id

        while current_stage_name and current_stage_name != "FINAL_STEP" and len(visited_stages) < max_hops:
            if current_stage_name in visited_stages:
                self.logger.error(f"Loop detected: Stage '{current_stage_name}' visited again. Aborting flow.")
                context['errors'][current_stage_name] = "Loop detected"
                context['final_status'] = "ERROR_LOOP_DETECTED"
                
                # Get the spec of the stage that is causing the loop detection to report its correct number
                failing_stage_spec = self.pipeline_def.stages.get(current_stage_name)
                correct_stage_number_for_failure = failing_stage_spec.number if failing_stage_spec else None

                self._update_run_status_with_stage_result(
                    stage_name=current_stage_name,
                    stage_number=correct_stage_number_for_failure, # Use the correct stage number
                    status=StageStatus.FAILURE,
                    reason=f"Loop detected: Stage '{current_stage_name}' visited again. Aborting flow.",
                    error_details={"short_error": "Loop detected", "details": f"Loop detected: Stage '{current_stage_name}' visited again. Aborting flow."}
                )
                break # Stop execution
            visited_stages.append(current_stage_name)

            current_master_stage_spec = self.pipeline_def.stages.get(current_stage_name)

            if not current_master_stage_spec:
                self.logger.error(f"Stage '{current_stage_name}' not found in MasterExecutionPlan. Aborting.")
                context['errors'][current_stage_name] = "Stage not found in plan"
                context['final_status'] = "ERROR_STAGE_NOT_FOUND"
                break
            
            # Log current stage execution attempt
            self.logger.info(f"Orchestrator: Attempting to execute stage: {current_stage_name} (Number: {current_master_stage_spec.number})")
            self.logger.debug(f"Orchestrator._execute_loop: About to access self.run_status_updates. Type: {type(getattr(self, 'run_status_updates', None))}, Exists: {hasattr(self, 'run_status_updates')}")
            # Record RUNNING status for this stage for this specific run
            self.run_status_updates.append({
                "stage_name": current_stage_name,
                "stage_number": current_master_stage_spec.number,
                "status": StageStatus.RUNNING.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            self.logger.info(f"Executing stage: {current_stage_name} (Agent: {current_master_stage_spec.agent_id}) for run_id: {run_id_for_status}")
            context['current_stage_name'] = current_stage_name # Make current stage name available in context

            # --- Prepare Agent Inputs from MasterStageSpec --- 
            # current_agent_inputs: Dict[str, Any] = {} # Old way
            # if current_master_stage_spec.inputs: # Old way
            # current_agent_inputs = current_master_stage_spec.inputs.copy() # Old way
            
            # Start with a fresh copy of the main orchestrator context for this stage
            # This context already contains 'original_request' from the run() method.
            base_context_for_stage_resolution = copy.deepcopy(context)

            # Special handling for "previous_stage_outputs": "<stage_id>"
            # This makes the *entire output object* of the previous stage available 
            # under 'resolved_previous_stage_output_data' in the context used for path resolution.
            resolved_prev_outputs_data = None
            raw_inputs_spec = current_master_stage_spec.inputs if current_master_stage_spec.inputs else {}
            
            prev_stage_id_to_fetch_key = "previous_stage_outputs"
            if prev_stage_id_to_fetch_key in raw_inputs_spec:
                prev_stage_id_value = raw_inputs_spec[prev_stage_id_to_fetch_key]
                if isinstance(prev_stage_id_value, str) and not prev_stage_id_value.startswith("context."):
                    # This is the special directive to fetch previous outputs by ID
                    if prev_stage_id_value in context.get('outputs', {}):
                        resolved_prev_outputs_data = context['outputs'][prev_stage_id_value]
                        self.logger.info(f"Made outputs of previous stage '{prev_stage_id_value}' available as 'resolved_previous_stage_output_data' for stage '{current_stage_name}'.")
                    else:
                        self.logger.warning(f"Outputs for previous stage ID '{prev_stage_id_value}' (specified in inputs for key '{prev_stage_id_to_fetch_key}') not found in context.outputs for stage '{current_stage_name}'. 'resolved_previous_stage_output_data' will be None.")
                elif isinstance(prev_stage_id_value, str) and prev_stage_id_value.startswith("context."):
                    # If it's a context path, _resolve_input_values will handle it.
                    # No special injection into 'resolved_previous_stage_output_data'.
                    pass
                else:
                    self.logger.warning(f"Value for '{prev_stage_id_to_fetch_key}' in stage '{current_stage_name}' inputs is not a string ID or context path: {prev_stage_id_value}. It will be treated as a literal if not a path by _resolve_input_values.")

            if resolved_prev_outputs_data is not None:
                base_context_for_stage_resolution['resolved_previous_stage_output_data'] = resolved_prev_outputs_data
            
            # Resolve all inputs specified in MasterStageSpec.inputs using path resolution
            # against the base_context_for_stage_resolution (which now includes original_request and resolved_previous_stage_output_data)
            stage_specific_resolved_inputs = self._resolve_input_values(
                raw_inputs_spec, 
                base_context_for_stage_resolution 
            )
            
            # Prepare final context for the agent: 
            # Start with a clean copy of the base context that was used for resolution 
            # (so agent has access to original_request, resolved_previous_stage_output_data etc.)
            # Then, update/overlay it with the specifically resolved inputs for the current stage.
            agent_call_context = copy.deepcopy(base_context_for_stage_resolution) 
            agent_call_context.update(stage_specific_resolved_inputs)
            
            # Clean up: remove the orchestrator's special 'previous_stage_outputs' key 
            # from the agent's direct inputs if its value was a string ID (and not a context path),
            # as its purpose was fulfilled by populating 'resolved_previous_stage_output_data'.
            # If it was a "context.path", it would have been resolved by _resolve_input_values already.
            if isinstance(raw_inputs_spec.get(prev_stage_id_to_fetch_key), str) and \
               not raw_inputs_spec.get(prev_stage_id_to_fetch_key, "").startswith("context."):
                agent_call_context.pop(prev_stage_id_to_fetch_key, None)
            
            # old: stage_context_for_agent = current_agent_inputs 
            # old: agent_call_context = copy.deepcopy(stage_context_for_agent)

            stage_status_to_report = StageStatus.RUNNING
            agent_error_details: Optional[AgentErrorDetails] = None
            agent_output_data = None # Standardized variable name

            try:
                agent_id = current_master_stage_spec.agent_id
                agent_callable = await self.agent_provider.get(agent_id)

                self.logger.debug(f"Orchestrator: Retrieved agent_callable: {agent_callable} (type: {type(agent_callable)}) for agent ID: {agent_id}")
                # Intensive debugging for AsyncMock type check
                # Try importing AsyncMock locally to see if it resolves type comparison issues
                callable_type_name = type(agent_callable).__name__
                # is_async_mock_by_name = callable_type_name == 'AsyncMock' # Old check
                is_async_mock_by_instance_check = isinstance(agent_callable, AsyncMock)
                self.logger.debug(f"Orchestrator: callable_type_name='{callable_type_name}', is_async_mock_by_instance_check={is_async_mock_by_instance_check} (using module-level AsyncMock)")

                if is_async_mock_by_instance_check:
                    self.logger.debug(f"Agent {agent_id} is AsyncMock (by module-level isinstance). Trying to call and await.")
                    try:
                        agent_output_data = await agent_callable(agent_call_context)
                        self.logger.debug(f"AsyncMock {agent_id} call awaited successfully.")
                    except Exception as am_exc:
                        self.logger.error(f"Exception directly from awaiting AsyncMock {agent_id}: {am_exc}", exc_info=True)
                        raise # Re-raise to be caught by the outer try-except
                elif inspect.iscoroutinefunction(agent_callable): # Handles 'async def func' as a function object
                    self.logger.debug(f"Agent {agent_id} is coroutine function. Calling and awaiting.")
                    agent_output_data = await agent_callable(agent_call_context)
                elif callable(agent_callable) and inspect.iscoroutinefunction(getattr(agent_callable, '__call__', None)): # Handles instance with 'async def __call__'
                    self.logger.debug(f"Agent {agent_id} is callable with async __call__. Calling and awaiting.")
                    agent_output_data = await agent_callable(agent_call_context)
                # Removed the general awaitable/iscoroutine check that was here.
                # If it's not an AsyncMock or a known async callable type, assume sync.
                else:
                    # For sync agents, run in a thread pool to avoid blocking orchestrator
                    self.logger.debug(f"Agent {agent_id} appears to be synchronous. Running in thread. Type: {type(agent_callable)}")
                    agent_output_data = await asyncio.to_thread(agent_callable, agent_call_context)
    
                context['outputs'][current_stage_name] = agent_output_data if agent_output_data is not None else {}
                
                # --- Check for agent-reported failure via _mcp_status ---
                if isinstance(agent_output_data, dict) and \
                   agent_output_data.get("_mcp_status") == StageStatus.FAILURE.value:
                    self.logger.warning(f"Agent '{agent_id}' for stage '{current_stage_name}' reported failure via _mcp_status.")
                    stage_status_to_report = StageStatus.FAILURE
                    mcp_error_details_data = agent_output_data.get("_mcp_error_details", {})
                    if isinstance(mcp_error_details_data, dict):
                        agent_error_details = AgentErrorDetails(
                            error_type=mcp_error_details_data.get("error_type", "AgentReportedFailure"),
                            message=mcp_error_details_data.get("message", "Agent reported failure."),
                            traceback=mcp_error_details_data.get("traceback"),
                            agent_id=agent_id, # Use the actual agent_id
                            stage_id=current_stage_name
                        )
                    else: # Fallback if _mcp_error_details is not a dict
                        agent_error_details = AgentErrorDetails(
                            error_type="AgentReportedFailureMalformed",
                            message="Agent reported failure with malformed _mcp_error_details.",
                            agent_id=agent_id,
                            stage_id=current_stage_name
                        )
                    # Store the full agent output (which includes the error report) in context
                    context['errors'][current_stage_name] = agent_output_data 
                # --- Check for agent-requested clarification via _mcp_status ---
                elif isinstance(agent_output_data, dict) and \
                     agent_output_data.get("_mcp_status") == "NEEDS_CLARIFICATION": # Custom status string
                    self.logger.info(f"Agent '{agent_id}' for stage '{current_stage_name}' requested clarification.")
                    # This isn't a failure, but it's not a standard success either. It's a pause point.
                    # We will set a specific context flag and let the failure handling block save the state if autonomous.
                    stage_status_to_report = StageStatus.RUNNING # Keep as RUNNING, but it will pause
                    context['_agent_requested_clarification'] = True
                    context['_clarification_details_from_agent'] = agent_output_data.get("_mcp_clarification_details", {})
                    # No agent_error_details here as it's not an error
                else:
                    # If not an agent-reported failure or clarification request, assume success for now (exception handling is separate)
                    stage_status_to_report = StageStatus.SUCCESS
                    self.logger.info(f"Executing agent '{current_master_stage_spec.agent_id}' for stage '{current_stage_name}' (Number: {current_master_stage_spec.number}) with inputs: {agent_call_context}")

                # --- End of agent-reported failure check ---

                if stage_status_to_report == StageStatus.SUCCESS: 
                    # --- Check Success Criteria --- 
                    criteria_passed, failed_criteria_list = await self._check_success_criteria(
                        current_stage_name, current_master_stage_spec, context
                    )
                    if not criteria_passed:
                        self.logger.warning(f"Stage '{current_stage_name}' completed by agent but FAILED success criteria.")
                        stage_status_to_report = StageStatus.FAILURE
                        # Populate agent_error_details for criteria failure
                        agent_error_details = AgentErrorDetails(
                            error_type="SuccessCriteriaFailed",
                            message=f"Stage failed success criteria: {'; '.join(failed_criteria_list)}",
                            agent_id=current_master_stage_spec.agent_id,
                            stage_id=current_stage_name,
                            details={"failed_criteria": failed_criteria_list, "stage_outputs": context.get('outputs',{}).get(current_stage_name,{})}
                        )
                        context['errors'][current_stage_name] = agent_error_details.model_dump()
                    else:
                        # Original success path if criteria passed
                        generated_artifact_paths = []
                        if isinstance(agent_output_data, dict):
                            generated_artifact_paths = agent_output_data.get("_mcp_generated_artifacts_relative_paths_", [])
                            if not isinstance(generated_artifact_paths, list):
                                self.logger.warning(
                                    f"Agent '{current_master_stage_spec.agent_id}' output key '_mcp_generated_artifacts_relative_paths_' for stage '{current_stage_name}' was not a list (got {type(generated_artifact_paths)}). Treating as no artifacts."
                                )
                                generated_artifact_paths = []
                            else:
                                valid_paths = []
                                for p_idx, p_val in enumerate(generated_artifact_paths):
                                    if isinstance(p_val, str):
                                        valid_paths.append(p_val)
                                    else:
                                        self.logger.warning(
                                            f"Agent '{current_master_stage_spec.agent_id}' for stage '{current_stage_name}' provided a non-string path in '_mcp_generated_artifacts_relative_paths_' at index {p_idx} (type: {type(p_val)}). Skipping this path."
                                        )
                                generated_artifact_paths = valid_paths
                        
                        self.logger.info(f"Agent '{current_master_stage_spec.agent_id}' for stage '{current_stage_name}' (Number: {current_master_stage_spec.number}) completed successfully. Reported {len(generated_artifact_paths)} artifacts.")
                        self.state_manager.update_status(
                            pipeline_run_id=self.current_pipeline_run_id,
                            stage_name=current_stage_name,
                            stage_number=current_master_stage_spec.number,
                            status=StageStatus.SUCCESS.value,
                            reason="Stage completed successfully",
                            error_info=None,
                            artifacts=generated_artifact_paths
                        )
                        self.run_status_updates.append({
                            "stage_id": current_stage_name,
                            "stage_number": current_master_stage_spec.number,
                            "status": StageStatus.SUCCESS.value, 
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "outputs_summary": str(agent_output_data)[:200],
                            "num_artifacts_reported": len(generated_artifact_paths)
                        })

                        # --- Check for DSL-defined Clarification Checkpoint ---
                        if isinstance(self.pipeline_def, MasterExecutionPlan) and \
                           current_master_stage_spec.clarification_checkpoint and \
                           not context.get('_agent_requested_clarification'): # Don't pause if agent already requested it
                            
                            self.logger.info(f"Stage '{current_stage_name}' has a DSL-defined clarification checkpoint. Pausing autonomous flow.")
                            if not context.get('_autonomous_flow_paused_state_saved'):
                                try:
                                    current_run_id = self.current_pipeline_run_id
                                    checkpoint_details = current_master_stage_spec.clarification_checkpoint
                                    
                                    paused_details_args = {
                                        "run_id": current_run_id,
                                        "flow_id": self.pipeline_def.id,
                                        "paused_at_stage_id": current_stage_name,
                                        "status": FlowPauseStatus.PAUSED_CLARIFICATION_NEEDED_AT_DSL_CHECKPOINT,
                                        "context_snapshot": copy.deepcopy(context),
                                        "error_details": None, # Not an error
                                        "clarification_request": checkpoint_details
                                    }
                                    self.state_manager.save_paused_flow_state(**paused_details_args)
                                    self.logger.info(f"Autonomous flow run_id '{current_run_id}' paused for DSL-defined clarification at stage '{current_stage_name}'.")
                                    context['_autonomous_flow_paused_state_saved'] = True
                                    context['_autonomous_flow_paused_for_dsl_checkpoint'] = True 
                                    # This will cause the loop to break due to _autonomous_flow_paused_state_saved being true
                                except Exception as save_exc:
                                    self.logger.exception(f"Critical error: Failed to save paused state for DSL clarification checkpoint: {save_exc}")
                            # If successfully paused for DSL checkpoint, status is RUNNING but will break the loop
                            # The generic `_autonomous_flow_paused_state_saved` check later will break.

            except KeyError as e: # This handles agent_id not found by agent_provider
                self.logger.error(f"Agent '{agent_id}' not found for stage '{current_stage_name}': {e}", exc_info=True)
                context['errors'][current_stage_name] = f"Agent '{agent_id}' not found: {e}"
                stage_status_to_report = StageStatus.FAILURE
                agent_error_details = AgentErrorDetails(error_type="AgentNotFound", message=str(e), agent_id=agent_id, stage_id=current_stage_name)
            except Exception as e:
                self.logger.error(f"Error executing agent for stage '{current_stage_name}': {e}", exc_info=True)
                context['errors'][current_stage_name] = f"Agent execution error: {e}"
                agent_output_data = {"error": str(e), "traceback": traceback.format_exc()}
                context['outputs'][current_stage_name] = agent_output_data # Store error in output
                stage_status_to_report = StageStatus.FAILURE
                # Ensure agent_id is defined; it should be from current_master_stage_spec.agent_id if spec was found
                agent_id_for_error = current_master_stage_spec.agent_id if current_master_stage_spec else "unknown_agent"
                agent_error_details = AgentErrorDetails(error_type="AgentExecutionError", message=str(e), traceback=traceback.format_exc(), agent_id=agent_id_for_error, stage_id=current_stage_name)

                # P2.4.1: If autonomous flow, save paused state with specific autonomous status
                if isinstance(self.pipeline_def, MasterExecutionPlan):
                    self.logger.info(f"Autonomous flow (MasterExecutionPlan) encountered an unhandled agent exception in stage '{current_stage_name}'. Saving paused state.")
                    try:
                        current_run_id = self.current_pipeline_run_id 
                        paused_details_args = {
                            "run_id": current_run_id,
                            "flow_id": self.pipeline_def.id,
                            "paused_at_stage_id": current_stage_name,
                            "status": FlowPauseStatus.PAUSED_AUTONOMOUS_FAILURE_UNHANDLED_EXCEPTION,
                            "context_snapshot": copy.deepcopy(context),
                            "error_details": agent_error_details,
                        }
                        self.state_manager.save_paused_flow_state(**paused_details_args)
                        self.logger.info(f"Autonomous flow run_id '{current_run_id}' paused due to unhandled agent exception in stage '{current_stage_name}'. Status: {FlowPauseStatus.PAUSED_AUTONOMOUS_FAILURE_UNHANDLED_EXCEPTION.value}")
                        # Use a more specific flag to indicate this path saved the state
                        context['_autonomous_flow_paused_state_saved_by_exception_handler'] = True 
                    except Exception as save_exc:
                        self.logger.exception(f"Critical error: Failed to save paused state for autonomous flow run_id '{current_run_id}' after agent error: {save_exc}")
            
            context['status_updates'].append({
                "stage_id": current_stage_name,
                "status": stage_status_to_report.value,
                "agent_id": current_master_stage_spec.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "outputs_summary": str(agent_output_data)[:100] if agent_output_data else None, # Summary
                "error": agent_error_details.model_dump() if agent_error_details else None
            })

            if stage_status_to_report == StageStatus.FAILURE:
                self.logger.warning(f"Stage '{current_stage_name}' failed. Determining next step.")

                # Flag to see if reviewer was invoked and a suggestion was made
                reviewer_suggestion_processed = False

                # Check if it's an autonomous flow and if state hasn't been saved by unhandled exception handler
                if isinstance(self.pipeline_def, MasterExecutionPlan) and \
                   not context.get('_autonomous_flow_paused_state_saved_by_exception_handler'): 
                    self.logger.info(f"Autonomous flow (MasterExecutionPlan) encountered an agent-reported failure or other non-exception failure in stage '{current_stage_name}'. Saving paused state before potential review.")
                    
                    if not agent_error_details: 
                        # This block would be hit if success criteria failed, or potentially other future non-exception failures
                        # that don't set agent_error_details directly before this point.
                        agent_id_for_error_fallback = current_master_stage_spec.agent_id if current_master_stage_spec else "unknown_agent"
                        # If agent_error_details is None here, it means it wasn't an agent-reported error and criteria didn't fail it YET
                        # OR it *was* a criteria failure, and agent_error_details was set just above this `if stage_status_to_report == StageStatus.FAILURE:` block.
                        # We need to ensure agent_error_details is correctly populated for PAUSED_AUTONOMOUS_FAILURE_AGENT_REPORTED / CRITERIA
                        # This specific log and creation might be redundant if criteria failure already populated it.
                        # Let's assume agent_error_details *is* populated if it's a criteria failure.
                        # If it's another type of failure that lands here without agent_error_details, create a generic one.
                        if not agent_error_details: # Double check if it's still None
                            agent_error_details = AgentErrorDetails(
                                error_type="GenericStageFailure", # This might be if success_criteria was false
                                message=f"Stage '{current_stage_name}' failed. Reason not specified by agent or criteria check prior to this point.",
                                agent_id=agent_id_for_error_fallback,
                                stage_id=current_stage_name
                            )
                            self.logger.warning("agent_error_details was None when attempting to save non-exception failure state. Created a generic one.")

                    try:
                        current_run_id = self.current_pipeline_run_id
                        # Determine correct pause status for failure
                        failure_pause_status = FlowPauseStatus.PAUSED_AUTONOMOUS_FAILURE_AGENT_REPORTED
                        if agent_error_details and agent_error_details.error_type == "SuccessCriteriaFailed":
                            failure_pause_status = FlowPauseStatus.PAUSED_AUTONOMOUS_FAILURE_CRITERIA
                        
                        paused_details_args = {
                            "run_id": current_run_id,
                            "flow_id": self.pipeline_def.id,
                            "paused_at_stage_id": current_stage_name,
                            "status": failure_pause_status, 
                            "context_snapshot": copy.deepcopy(context),
                            "error_details": agent_error_details,
                            # No clarification_request here for failures
                        }
                        self.state_manager.save_paused_flow_state(**paused_details_args)
                        self.logger.info(f"Autonomous flow run_id '{current_run_id}' paused due to failure in stage '{current_stage_name}'. Status: {failure_pause_status.value}")
                        context['_autonomous_flow_paused_state_saved'] = True 
                    except Exception as save_exc:
                        self.logger.exception(f"Critical error: Failed to save paused state for autonomous flow run_id '{current_run_id}' after non-exception error: {save_exc}")
            
            # --- Handle Agent-Requested Clarification Pause for Autonomous Flows ---
            elif context.get('_agent_requested_clarification') and isinstance(self.pipeline_def, MasterExecutionPlan):
                self.logger.info(f"Autonomous flow (MasterExecutionPlan) requires clarification at stage '{current_stage_name}' as requested by agent.")
                if not context.get('_autonomous_flow_paused_state_saved'): # Ensure state isn't already saved by some preceding logic
                    try:
                        current_run_id = self.current_pipeline_run_id
                        clarification_details = context.get('_clarification_details_from_agent', {})
                        
                        paused_details_args = {
                            "run_id": current_run_id,
                            "flow_id": self.pipeline_def.id,
                            "paused_at_stage_id": current_stage_name,
                            "status": FlowPauseStatus.PAUSED_CLARIFICATION_NEEDED_BY_AGENT,
                            "context_snapshot": copy.deepcopy(context),
                            "error_details": None, # Not an error
                            "clarification_request": clarification_details
                        }
                        self.state_manager.save_paused_flow_state(**paused_details_args)
                        self.logger.info(f"Autonomous flow run_id '{current_run_id}' paused for agent-requested clarification at stage '{current_stage_name}'.")
                        context['_autonomous_flow_paused_state_saved'] = True # Use the general flag
                        context['_autonomous_flow_paused_for_clarification'] = True # Specific flag if needed elsewhere
                    except Exception as save_exc:
                        self.logger.exception(f"Critical error: Failed to save paused state for autonomous flow clarification: {save_exc}")
                
                # If paused for clarification, we should break the loop. 
                # The status_updates should reflect a PAUSED status ideally, not FAILURE.
                # The current logic will fall through to the `if context.get('_autonomous_flow_paused_state_saved'):` block below which breaks.
                # No need to set final_status to error here.
                self.logger.info(f"Autonomous flow stage '{current_stage_name}' paused for clarification. Halting MasterExecutionPlan execution.")
                # The `context.get('_autonomous_flow_paused_state_saved')` check later will break the loop.

            # This is where an unhandled exception in an autonomous flow would have saved state
            elif isinstance(self.pipeline_def, MasterExecutionPlan) and \
                 context.get('_autonomous_flow_paused_state_saved_by_exception_handler'):
                 self.logger.info(f"Autonomous flow was already paused by unhandled exception handler for stage '{current_stage_name}'. Proceeding to reviewer (if applicable for failure).)")
                 context['_autonomous_flow_paused_state_saved'] = True 

            # --- Invoke MasterPlannerReviewerAgent if autonomous flow is now paused ---
            if isinstance(self.pipeline_def, MasterExecutionPlan) and context.get('_autonomous_flow_paused_state_saved'):
                self.logger.info(f"Invoking MasterPlannerReviewerAgent for failed autonomous stage: {current_stage_name}")
                try:
                    reviewer_agent_callable = await self.agent_provider.get(MasterPlannerReviewerAgent.AGENT_ID)
                    
                    # Ensure agent_error_details is not None
                    if not agent_error_details:
                        # This should ideally not happen if a failure status is set
                        agent_id_for_error_fallback = current_master_stage_spec.agent_id if current_master_stage_spec else "unknown_agent"
                        agent_error_details = AgentErrorDetails(
                            error_type="UnknownFailureForReviewer",
                            message=f"Stage '{current_stage_name}' failed but no specific error details were available for reviewer.",
                            agent_id=agent_id_for_error_fallback, # Use fallback
                            stage_id=current_stage_name
                        )
                        self.logger.warning("MasterPlannerReviewer invocation: agent_error_details was None. Created a default.")
                    
                    reviewer_input = MasterPlannerReviewerInput(
                        original_goal=str(self.pipeline_def.original_request.goal_description) if self.pipeline_def.original_request else "Unknown original goal",
                        failed_master_plan_json=self.pipeline_def.model_dump_json(indent=2),
                        failed_stage_id=current_stage_name,
                        error_details=agent_error_details, # This should be populated from either exception or _mcp_status
                        full_context_snapshot=copy.deepcopy(context)
                    )
                    
                    reviewer_output: MasterPlannerReviewerOutput = await reviewer_agent_callable.async_invoke(reviewer_input)
                    self.logger.info(f"MasterPlannerReviewerAgent suggested: {reviewer_output.suggested_action.value}")
                    self.logger.debug(f"Reviewer justification: {reviewer_output.justification}")
                    context['last_reviewer_suggestion'] = reviewer_output.model_dump()
                    reviewer_suggestion_processed = True

                    # For P2.4.3, we only log. Actual processing of suggestions (retry, modify plan) is P2.4.4+
                    if reviewer_output.suggested_action == ReviewerActionType.ESCALATE_TO_USER:
                        self.logger.info(f"Reviewer advised escalation for stage '{current_stage_name}'. Autonomous flow remains paused.")
                    # Add other conditions here in the future for RETRY, MODIFY_PLAN etc.

                except Exception as reviewer_exc:
                    self.logger.exception(f"Error invoking or processing MasterPlannerReviewerAgent for stage '{current_stage_name}': {reviewer_exc}")
                    # Even if reviewer fails, the flow is already paused. Log and proceed to break.
                    context['errors']["MasterPlannerReviewerAgent_invocation"] = str(reviewer_exc)
            
            # --- End of Reviewer Invocation ---

            # This existing block handles logging the final state / breaking the loop
            if context.get('_autonomous_flow_paused_state_saved'):
                self.logger.info(f"Autonomous flow stage '{current_stage_name}' failed and state was saved. Halting MasterExecutionPlan execution.")
            else:
                # Default behavior for non-autonomous flows or if state saving failed for autonomous
                self.logger.error(f"Stage '{current_stage_name}' failed. Halting flow execution. Consider implementing PAUSE_FOR_INTERVENTION for this plan type if applicable.")
            
            context['final_status'] = f"ERROR_STAGE_FAILED: {current_stage_name}"
            # Update the main run status in StateManager to FAILED if not already paused
            if not context.get('_autonomous_flow_paused_state_saved'):
                 self.state_manager.update_status(
                    pipeline_run_id=self.current_pipeline_run_id,
                    # stage_name and stage_number are for specific stage, here we update overall run
                    status=StageStatus.FAILURE.value, # Or a more specific overall run failure status if available
                    reason=f"Flow failed at stage {current_stage_name}",
                    error_info=agent_error_details.model_dump() if agent_error_details else None
                )
            break # Stop execution

            # Determine next stage based on MasterStageSpec's simple next_stage link
            next_stage_name_from_spec = current_master_stage_spec.next_stage
            if not next_stage_name_from_spec or next_stage_name_from_spec == "FINAL_STEP":
                self.logger.info(f"Reached end of plan or FINAL_STEP at stage '{current_stage_name}'.")
                context['final_status'] = "COMPLETED"
                current_stage_name = None # End loop
            elif next_stage_name_from_spec not in self.pipeline_def.stages:
                self.logger.error(f"Next stage '{next_stage_name_from_spec}' from stage '{current_stage_name}' not found in plan. Aborting.")
                context['errors'][current_stage_name] = f"Next stage '{next_stage_name_from_spec}' invalid."
                context['final_status'] = "ERROR_INVALID_NEXT_STAGE"
                current_stage_name = None # End loop
            else:
                current_stage_name = next_stage_name_from_spec
        
        if not context.get('final_status'): # If loop exited due to max_hops
            if len(visited_stages) >= max_hops:
                self.logger.warning(f"MasterExecutionPlan execution stopped due to max hops reached ({max_hops}).")
                context['final_status'] = "ERROR_MAX_HOPS_REACHED"
            elif not current_stage_name: # Should be covered by FINAL_STEP or completion logic
                 self.logger.info("MasterExecutionPlan completed normally (current_stage_name is None).")
                 if "final_status" not in context: context['final_status'] = "COMPLETED_UNKNOWN_EXIT" # Should not happen

        return context

    async def run(self, pipeline_def: Union[PipelineDefinition, MasterExecutionPlan], initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the given MasterExecutionPlan.
        Args:
            plan: The MasterExecutionPlan to execute.
            context: Initial context for the execution.
        Returns:
            The final context after execution, including outputs and errors.
        """
        self.logger.info(f"AsyncOrchestrator starting execution of MasterExecutionPlan: {pipeline_def.id}")
        if not pipeline_def.start_stage:
            self.logger.error(f"MasterExecutionPlan {pipeline_def.id} has no start_stage defined. Cannot execute.")
            return {"error": "No start_stage defined in plan", "outputs": {}, "final_status": "ERROR_NO_START_STAGE"}

        if self.pipeline_def.id != pipeline_def.id:
             self.logger.warning(f"Running a new plan {pipeline_def.id} on an orchestrator initialized with {self.pipeline_def.id}. Re-assigning internal plan.")
             self.pipeline_def = pipeline_def

        if "run_id" not in initial_context:
            initial_context["run_id"] = f"mep_run_{pipeline_def.id}_{str(uuid.uuid4())[:8]}"

        self.current_stage_outputs: Dict[str, Any] = {}
        self.current_stage_errors: Dict[str, Any] = {}
        self.visited_stages_in_current_run: List[str] = []
        self.run_status_updates: List[Dict[str, Any]] = []
        self.logger.debug(f"Orchestrator.run: Initialized self.run_status_updates. Type: {type(self.run_status_updates)}, Exists: {hasattr(self, 'run_status_updates')}")

        initial_context.setdefault('outputs', {})
        initial_context.setdefault('errors', {})
        initial_context.setdefault('status_updates', [])
        initial_context.setdefault('visited', [])

        run_id_for_status = str(self.state_manager.get_or_create_current_run_id() or uuid.uuid4())
        initial_context['run_id'] = run_id_for_status 
        initial_context['flow_id'] = pipeline_def.id 
        self.current_pipeline_run_id = run_id_for_status # Ensure this is set on the instance
        
        if hasattr(pipeline_def, 'original_request') and pipeline_def.original_request is not None:
            initial_context['original_request'] = pipeline_def.original_request
        else:
            self.logger.warning(f"MasterExecutionPlan '{pipeline_def.id}' does not have an 'original_request' attribute or it's None. 'original_request' will be None in context.")
            initial_context['original_request'] = None 
            
        # Determine start stage
        start_stage_name = pipeline_def.start_stage

        final_context = await self._execute_loop(start_stage_name, initial_context)
        self.logger.info(f"AsyncOrchestrator finished execution of MasterExecutionPlan: {pipeline_def.id}. Final status: {final_context.get('final_status')}")
        return final_context

    async def resume_flow(
        self,
        run_id: str,
        action: str,
        action_data: Optional[Dict[str, Any]] = None,
        # clarification_response: Optional[Dict[str, Any]] = None # Decided to pass via action_data
    ) -> Dict[str, Any]:
        """Resumes a paused MASTER flow based on the specified action."""
        self.logger.info(f"Attempting to resume MASTER flow run_id: {run_id} with action: {action}")
        action_data = action_data or {} # Ensure action_data is a dict

        try:
            paused_details = self.state_manager.load_paused_flow_state(run_id)
            if not paused_details:
                self.logger.error(f"No paused run found for run_id: {run_id}")
                return {"error": f"No paused run found for run_id: {run_id}"}
            
            if not isinstance(self.pipeline_def, MasterExecutionPlan) or self.pipeline_def.id != paused_details.flow_id:
                 self.logger.error(f"Orchestrator's current plan ('{self.pipeline_def.id}' type: {type(self.pipeline_def)}) doesn't match paused flow_id ('{paused_details.flow_id}'). Cannot resume safely.")
                 return {"error": "Mismatched flow definition during resume attempt."}

            context = paused_details.context_snapshot 
            if not context:
                self.logger.warning(f"Context snapshot in paused details is empty for run_id: {run_id}. Proceeding with empty context.")
                context = {}

            start_stage_name: Optional[str] = None
            resume_context = context
            # action_data = action_data or {} # Moved up

            # --- Handle Resumption based on Paused Status FIRST if it's a clarification pause ---
            if paused_details.status in [
                FlowPauseStatus.PAUSED_CLARIFICATION_NEEDED_BY_AGENT,
                FlowPauseStatus.PAUSED_CLARIFICATION_NEEDED_AT_DSL_CHECKPOINT,
                FlowPauseStatus.PAUSED_CLARIFICATION_NEEDED_BY_ORCHESTRATOR # Future use
            ]:
                self.logger.info(f"[Resume] Paused status is '{paused_details.status.value}'. Expecting 'provide_clarification' action or forced override.")
                clarification_response_data = action_data.get('clarification_response')

                if action == "provide_clarification":
                    if clarification_response_data is not None and isinstance(clarification_response_data, dict):
                        self.logger.info(f"[Resume] Processing 'provide_clarification' action with response: {clarification_response_data}")
                        # Update context with clarification response
                        # Place it where subsequent agent/logic can find it, e.g., under the paused stage's outputs.
                        resume_context.setdefault('outputs', {}).setdefault(paused_details.paused_at_stage_id, {})['clarification_response'] = clarification_response_data
                        self.logger.debug(f"Context updated with clarification_response at outputs.{paused_details.paused_at_stage_id}.clarification_response")
                    else:
                        self.logger.warning("'provide_clarification' action called, but 'clarification_response' missing or not a dict in action_data. Resuming without new clarification input.")

                    if paused_details.status == FlowPauseStatus.PAUSED_CLARIFICATION_NEEDED_BY_AGENT:
                        self.logger.info(f"[Resume] Resuming by re-executing stage '{paused_details.paused_at_stage_id}' that requested clarification.")
                        start_stage_name = paused_details.paused_at_stage_id
                    elif paused_details.status == FlowPauseStatus.PAUSED_CLARIFICATION_NEEDED_AT_DSL_CHECKPOINT:
                        self.logger.info(f"[Resume] Resuming after DSL checkpoint at stage '{paused_details.paused_at_stage_id}'. Determining next stage.")
                        paused_stage_spec_for_dsl_checkpoint: Optional[MasterStageSpec] = self.pipeline_def.stages.get(paused_details.paused_at_stage_id)
                        if not paused_stage_spec_for_dsl_checkpoint:
                            self.logger.error(f"Cannot determine next stage; DSL checkpoint stage '{paused_details.paused_at_stage_id}' not found in plan.")
                            return {"error": f"Fatal: Paused stage '{paused_details.paused_at_stage_id}' for DSL checkpoint not found in Master Flow."}
                        
                        # Logic similar to skip_stage to find next stage after a completed one
                        if paused_stage_spec_for_dsl_checkpoint.condition:
                            condition_met = self._parse_condition(paused_stage_spec_for_dsl_checkpoint.condition, resume_context)
                            start_stage_name = paused_stage_spec_for_dsl_checkpoint.next_stage_true if condition_met else paused_stage_spec_for_dsl_checkpoint.next_stage_false
                        else:
                            start_stage_name = paused_stage_spec_for_dsl_checkpoint.next_stage
                        
                        if start_stage_name:
                            self.logger.info(f"Resuming from stage: '{start_stage_name}' after DSL checkpoint.")
                        else:
                            self.logger.info(f"DSL checkpoint stage '{paused_details.paused_at_stage_id}' was the last stage or led to no next stage. Master Flow considered complete.")
                            # Clean up state and return context if flow is now complete
                            try:
                                delete_success = self.state_manager.delete_paused_flow_state(run_id)
                                if not delete_success: self.logger.warning(f"Failed to delete paused state for {run_id} after DSL checkpoint completion.")
                            except Exception as del_err: self.logger.error(f"Error deleting state for {run_id}: {del_err}")
                            resume_context['final_status'] = "COMPLETED_AFTER_CLARIFICATION_CHECKPOINT"
                            return resume_context # Early exit if flow complete
                    else: # e.g. PAUSED_CLARIFICATION_NEEDED_BY_ORCHESTRATOR (if that becomes a thing)
                        self.logger.warning(f"Resume logic for clarification status '{paused_details.status.value}' not fully defined yet. Defaulting to retrying stage '{paused_details.paused_at_stage_id}'.")
                        start_stage_name = paused_details.paused_at_stage_id
                
                elif action in ["retry", "retry_with_inputs", "skip_stage", "force_branch", "abort"]:
                     self.logger.info(f"[Resume] Flow was paused for clarification ('{paused_details.status.value}'), but received a standard error-handling action '{action}'. Proceeding with '{action}'.")
                     # Fall through to existing error handling actions below
                else:
                    self.logger.error(f"Flow paused for clarification ('{paused_details.status.value}'), but received unsupported action: '{action}'. Required: 'provide_clarification' or a standard override action.")
                    return {"error": f"Unsupported action '{action}' for clarification pause. Use 'provide_clarification' or a standard override."}

            # --- Standard Resume Actions (mostly for error pauses, or if clarification pause is overridden by one of these) ---
            if start_stage_name is None: # Only enter if not already set by clarification logic above
                if action == "retry":
                    self.logger.info(f"[Resume] Action: Retry MASTER stage '{paused_details.paused_at_stage_id}'")
                    start_stage_name = paused_details.paused_at_stage_id
            
            if start_stage_name:
                try:
                    clear_success = self.state_manager.delete_paused_flow_state(run_id)
                    if not clear_success:
                        self.logger.warning(f"Failed to clear paused state for run_id={run_id}. Proceeding with execution.")
                except Exception as e:
                    self.logger.exception(f"Failed to clear paused state for run_id={run_id}. Aborting resume. Error: {e}")
                    return {"error": f"Failed to clear paused state for run_id={run_id}. Resume aborted."}

                self.logger.info(f"Resuming MASTER execution loop for run_id '{run_id}' from stage '{start_stage_name}'")
                final_context = await self._execute_loop(start_stage_name=start_stage_name, context=resume_context)
                return final_context
            else:
                self.logger.error(f"Resume logic failed to determine a start stage for action '{action}' or did not handle flow completion correctly.")
                return {"error": "Internal error: Failed to determine resume start stage or handle flow completion."}

        except Exception as e:
            self.logger.exception(f"Error resuming flow: {e}")
            return {"error": f"Error resuming flow: {e}"}

    def _update_run_status_with_stage_result(
        self,
        stage_name: str,
        stage_number: Optional[float],
        status: StageStatus,
        artifacts: Optional[List[str]] = None,
        error_details: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None
    ):
        """Helper to update both the in-memory run_status_updates and the StateManager."""
        timestamp = datetime.now(timezone.utc).isoformat()
        status_entry = {
            "stage_name": stage_name,
            "stage_number": stage_number,
            "status": status.value,
            "reason": reason,
            "artifacts": artifacts or [],
            "error_details": error_details,
            "timestamp": timestamp
        }
        self.run_status_updates.append(status_entry)

        # Persist to StateManager if available
        if self.state_manager:
            try:
                # ADD DEBUG LOGGING BEFORE THIS CALL
                self.logger.debug(
                    f"[ORCH_DEBUG] Calling state_manager.update_status. Args: "
                    f"pipeline_run_id={self.current_pipeline_run_id}, "
                    f"stage_name='{stage_name}', stage_number={stage_number}, status='{status.value}', "
                    f"reason='{reason}', error_info={error_details}, artifacts={artifacts or []}"
                )
                self.state_manager.update_status(
                    pipeline_run_id=self.current_pipeline_run_id,
                    stage_name=stage_name,
                    stage_number=stage_number,
                    status=status.value,
                    reason=reason,
                    error_info=error_details,
                    artifacts=artifacts or []
                )
            except Exception as e:
                self.logger.error(f"Error updating state manager: {e}")


# ---------------------------------------------------------------------------
# DSL Validation Helpers (optional, requires jsonschema)
# ---------------------------------------------------------------------------

def _validate_dsl(data: dict) -> None:  # type: ignore[override]
    pass # No-op validation