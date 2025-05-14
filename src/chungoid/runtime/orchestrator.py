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
from chungoid.schemas.common_enums import StageStatus
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.flows import PausedRunDetails
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec

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
                try:
                    for part in context_path.split('.'):
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
                self._update_run_status_with_stage_result( # Restore original call
                    stage_name=current_stage_name,
                    stage_number=current_master_stage_spec.number,
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
                stage_status_to_report = StageStatus.SUCCESS # Use SUCCESS enum member
                self.logger.info(f"Executing agent '{current_master_stage_spec.agent_id}' for stage '{current_stage_name}' (Number: {current_master_stage_spec.number}) with inputs: {agent_call_context}")
                
                # Extract artifact paths from agent output if provided using the conventional key
                generated_artifact_paths = []
                if isinstance(agent_output_data, dict):
                    # Standardized key for artifact paths list
                    generated_artifact_paths = agent_output_data.get("_mcp_generated_artifacts_relative_paths_", [])
                    if not isinstance(generated_artifact_paths, list):
                        self.logger.warning(
                            f"Agent '{current_master_stage_spec.agent_id}' output key '_mcp_generated_artifacts_relative_paths_' for stage '{current_stage_name}' was not a list (got {type(generated_artifact_paths)}). Treating as no artifacts."
                        )
                        generated_artifact_paths = []
                    else:
                        # Ensure all paths in the list are strings
                        valid_paths = []
                        for p_idx, p_val in enumerate(generated_artifact_paths):
                            if isinstance(p_val, str):
                                valid_paths.append(p_val)
                            else:
                                self.logger.warning(
                                    f"Agent '{current_master_stage_spec.agent_id}' for stage '{current_stage_name}' provided a non-string path in '_mcp_generated_artifacts_relative_paths_' at index {p_idx} (type: {type(p_val)}). Skipping this path."
                                )
                        generated_artifact_paths = valid_paths
                
                self.logger.info(f"Agent '{current_master_stage_spec.agent_id}' for stage '{current_stage_name}' (Number: {current_master_stage_spec.number}) completed. Reported {len(generated_artifact_paths)} artifacts.")
                # Update status to SUCCESS (formerly PASS)
                self.state_manager.update_status(
                    pipeline_run_id=self.current_pipeline_run_id,
                    stage_name=current_stage_name,
                    stage_number=current_master_stage_spec.number,
                    status=StageStatus.SUCCESS.value,
                    reason="Stage completed successfully",
                    error_info=None,
                    artifacts=generated_artifact_paths # Pass extracted/validated paths
                )
                self.run_status_updates.append({
                    "stage_id": current_stage_name,
                    "stage_number": current_master_stage_spec.number,
                    "status": StageStatus.SUCCESS.value, 
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "outputs_summary": str(agent_output_data)[:200], # Optional summary
                    "num_artifacts_reported": len(generated_artifact_paths)
                })

            except KeyError as e:
                self.logger.error(f"Agent '{agent_id}' not found for stage '{current_stage_name}': {e}", exc_info=True)
                context['errors'][current_stage_name] = f"Agent '{agent_id}' not found: {e}"
                stage_status_to_report = StageStatus.FAILURE
                agent_error_details = AgentErrorDetails(error_type="AgentNotFound", message=str(e))
            except Exception as e:
                self.logger.error(f"Error executing agent for stage '{current_stage_name}': {e}", exc_info=True)
                context['errors'][current_stage_name] = f"Agent execution error: {e}"
                agent_output_data = {"error": str(e), "traceback": traceback.format_exc()}
                context['outputs'][current_stage_name] = agent_output_data # Store error in output
                stage_status_to_report = StageStatus.FAILURE
                agent_error_details = AgentErrorDetails(error_type="AgentExecutionError", message=str(e), traceback=traceback.format_exc())
            
            # --- StateManager Update --- (Optional, if MasterExecutionPlan needs live status updates)
            # if self.state_manager and hasattr(self.state_manager, 'update_master_plan_stage_status'):
            #     await asyncio.to_thread( # Assuming update_master_plan_stage_status might be sync
            #         self.state_manager.update_master_plan_stage_status, 
            #         plan_id=self.pipeline_def.id, 
            #         stage_id=current_stage_name, 
            #         status=stage_status_to_report,
            #         outputs=agent_output, # Could be large, consider summarizing
            #         error_details=agent_error_details
            #     )
            context['status_updates'].append({
                "stage_id": current_stage_name,
                "status": stage_status_to_report.value,
                "agent_id": current_master_stage_spec.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "outputs_summary": str(agent_output_data)[:100] if agent_output_data else None, # Summary
                "error": agent_error_details.model_dump() if agent_error_details else None
            })

            if stage_status_to_report == StageStatus.FAILURE:
                self.logger.warning(f"Stage '{current_stage_name}' failed. Determining next step based on error handling strategy.")
                # Default behavior: halt plan execution on first failure
                self.logger.error(f"Stage '{current_stage_name}' failed. Halting MasterExecutionPlan execution.")
                context['final_status'] = f"ERROR_STAGE_FAILED: {current_stage_name}"
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
        action_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resumes a paused MASTER flow based on the specified action."""
        self.logger.info(f"Attempting to resume MASTER flow run_id: {run_id} with action: {action}")

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

        except Exception as e:
            self.logger.exception(f"Error loading state for run_id {run_id}: {e}")
            return {"error": f"Failed to load state for run_id: {run_id}, {e}"}

        self.logger.debug(f"State loaded for run_id: {run_id}. Paused at MASTER stage: {paused_details.paused_at_stage_id}")
        self.logger.debug(f"Resume action: {action}, Data: {action_data}")

        start_stage_name: Optional[str] = None
        resume_context = context
        action_data = action_data or {}

        if action == "retry":
            self.logger.info(f"[Resume] Action: Retry MASTER stage '{paused_details.paused_at_stage_id}'")
            start_stage_name = paused_details.paused_at_stage_id

        elif action == "retry_with_inputs":
            self.logger.info(f"[Resume] Action: Retry MASTER stage '{paused_details.paused_at_stage_id}' with new inputs.")
            start_stage_name = paused_details.paused_at_stage_id
            new_inputs = action_data.get('inputs')
            if isinstance(new_inputs, dict):
                self.logger.debug(f"Applying new inputs to context: {new_inputs}")
                resume_context = copy.deepcopy(context)
                resume_context.update(new_inputs)
            else:
                self.logger.warning("Action 'retry_with_inputs' called without valid 'inputs' dictionary.")
                return {"error": "Action 'retry_with_inputs' requires a dictionary under the 'inputs' key in action_data."}

        elif action == "skip_stage":
            self.logger.info(f"[Resume] Action: Skip MASTER stage '{paused_details.paused_at_stage_id}'")
            paused_stage_spec: Optional[MasterStageSpec] = self.pipeline_def.stages.get(paused_details.paused_at_stage_id)
            if not paused_stage_spec:
                 self.logger.error(f"Cannot determine next stage to skip to; paused MASTER stage '{paused_details.paused_at_stage_id}' not found in plan.")
                 return {"error": f"Fatal: Paused stage '{paused_details.paused_at_stage_id}' not found in Master Flow definition."}

            next_stage_in_master_flow: Optional[str] = None
            if paused_stage_spec.condition:
                condition_met = self._parse_condition(paused_stage_spec.condition, resume_context)
                next_stage_in_master_flow = paused_stage_spec.next_stage_true if condition_met else paused_stage_spec.next_stage_false
            else:
                next_stage_in_master_flow = paused_stage_spec.next_stage
            
            start_stage_name = next_stage_in_master_flow

            if start_stage_name:
                self.logger.info(f"Will attempt to resume MASTER execution from stage: '{start_stage_name}' after skipping.")
            else:
                self.logger.info(f"Skipped MASTER stage '{paused_details.paused_at_stage_id}' was the last stage or led to no next stage. Master Flow considered complete.")
                try:
                    delete_success = self.state_manager.delete_paused_flow_state(run_id)
                    if not delete_success:
                        self.logger.warning(f"Failed to delete paused state file for run_id {run_id} after determining skip leads to completion.")
                except Exception as del_err:
                    self.logger.error(f"Error deleting paused state file for run_id {run_id}: {del_err}")
                return resume_context

        elif action == "force_branch":
            target_stage_id = action_data.get('target_stage_id')
            self.logger.info(f"[Resume] Action: Force branch to MASTER stage '{target_stage_id}'")
            if target_stage_id and isinstance(target_stage_id, str) and target_stage_id in self.pipeline_def.stages:
                start_stage_name = target_stage_id
            else:
                self.logger.error(f"Invalid or missing target_stage_id ('{target_stage_id}') for force_branch action. Must be a valid MASTER stage ID.")
                return {"error": f"Invalid target_stage_id for force_branch: '{target_stage_id}'. It must be a valid stage ID string present in the Master Flow."}
        
        elif action == "abort":
            self.logger.info(f"[Resume] Action: Abort MASTER flow for run_id={run_id}")
            try:
                delete_success = self.state_manager.delete_paused_flow_state(run_id)
                if not delete_success:
                     self.logger.warning(f"Failed to delete paused state file for run_id {run_id} during abort action.")
            except Exception as del_err:
                self.logger.error(f"Error deleting paused state file for run_id {run_id} during abort: {del_err}")
            resume_context["status"] = "ABORTED"
            self.logger.info(f"Paused state cleared for run_id={run_id}. Master Flow aborted by user action.")
            return resume_context

        else:
            self.logger.error(f"Unsupported resume action: '{action}'")
            return {"error": f"Unsupported resume action: {action}"}
        
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