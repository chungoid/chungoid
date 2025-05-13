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

        # Validate against JSON schema (if jsonschema is installed)
        try:
            _validate_dsl(data)
        except Exception as exc:
            # Rewrap to avoid leaking jsonschema dependency to callers
            raise ValueError(f"Flow DSL validation error: {exc}") from exc

        # Fallback check (redundant once schema validated but keeps mypy happy)
        if "stages" not in data or "start_stage" not in data:
            raise ValueError("Flow YAML missing required 'stages' or 'start_stage' key")

        return cls(
            id=flow_id or "<unknown>",
            start_stage=data["start_stage"],
            stages=data["stages"],
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
        # Example: "outputs.stage_a.result == 'go_left'"
        # This needs to be evaluated in the context of `context`
        try:
            # Simplified evaluation logic, vulnerable to injection if condition_str is untrusted.
            # In a real scenario, use a safer evaluation method like ast.literal_eval or a sandbox.
            
            # Split by common comparators, check for variable existence before eval
            parts = []
            comparator = None
            if '==' in condition_str:
                parts = condition_str.split('==', 1)
                comparator = '=='
            elif '!=' in condition_str:
                parts = condition_str.split('!=', 1)
                comparator = '!='
            # Add more comparators as needed (>, <, >=, <=, in, not in)
            else:
                self.logger.error(f"Unsupported condition format: {condition_str}")
                return False # Default to false on parse error

            if len(parts) != 2:
                self.logger.error(f"Invalid condition structure: {condition_str}")
                return False

            var_path_str = parts[0].strip()
            expected_value_str = parts[1].strip()

            # Resolve var_path_str from context (e.g., outputs.stage_a.result)
            current_val = context
            for key in var_path_str.split('.'):
                if isinstance(current_val, dict) and key in current_val:
                    current_val = current_val[key]
                else:
                    self.logger.warning(f"Condition variable path '{var_path_str}' not fully found in context.")
                    return False # Path not found, condition is false
            
            # Convert expected_value_str to the type of current_val for comparison
            # This is a simplification; robust type conversion is needed.
            try:
                if isinstance(current_val, bool):
                    expected_value = expected_value_str.lower() in ['true', '1']
                elif isinstance(current_val, int):
                    expected_value = int(expected_value_str.strip("'\""))
                elif isinstance(current_val, float):
                    expected_value = float(expected_value_str.strip("'\""))
                else: # Assume string
                    expected_value = expected_value_str.strip("'\"") # Remove quotes for string comparison
            except ValueError as e:
                self.logger.error(f"Type conversion error for expected value in condition '{condition_str}': {e}")
                return False

            self.logger.debug(f"Condition check: '{current_val}' {comparator} '{expected_value}'")
            if comparator == '==':
                return current_val == expected_value
            elif comparator == '!=':
                return current_val != expected_value
            
            return False # Should not reach here if comparator is supported

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
    """Orchestrates pipeline execution asynchronously, invoking agents."""

    def __init__(
        self,
        pipeline_def: MasterExecutionPlan,
        config: Dict[str, Any],
        agent_provider: AgentProvider,
        state_manager: StateManager
    ):
        self.pipeline_def = pipeline_def
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._agent_provider = agent_provider
        self._state_manager = state_manager

    def _parse_condition(self, condition_str: str, context: Dict[str, Any]) -> bool:
        if not condition_str:
            return True # No condition means proceed

        self.logger.debug(f"Parsing condition: {condition_str}")
        try:
            parts = []
            comparator = None
            if '==' in condition_str:
                parts = condition_str.split('==', 1)
                comparator = '=='
            elif '!=' in condition_str:
                parts = condition_str.split('!=', 1)
                comparator = '!='
            else:
                self.logger.error(f"Unsupported condition format: {condition_str}")
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
                else:
                    self.logger.warning(f"Condition variable path '{var_path_str}' not fully found in context.")
                    return False 
            
            try:
                if isinstance(current_val, bool):
                    expected_value = expected_value_str.lower() in ['true', '1']
                elif isinstance(current_val, int):
                    expected_value = int(expected_value_str.strip("'\""))
                elif isinstance(current_val, float):
                    expected_value = float(expected_value_str.strip("'\""))
                else: 
                    expected_value = expected_value_str.strip("'\"") 
            except ValueError as e:
                self.logger.error(f"Type conversion error for expected value in condition '{condition_str}': {e}")
                return False

            self.logger.debug(f"Condition check: '{current_val}' {comparator} '{expected_value}'")
            if comparator == '==':
                return current_val == expected_value
            elif comparator == '!=':
                return current_val != expected_value
            
            return False

        except Exception as e:
            self.logger.exception(f"Error evaluating condition '{condition_str}': {e}")
            return False

    async def _execute_loop(self, start_stage_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Core execution loop logic for Master Flows."""
        self.logger.info(f"Starting MASTER execution loop from stage: {start_stage_name}")
        current_stage_name = start_stage_name
        if 'outputs' not in context:
            context['outputs'] = {}
        max_hops = len(self.pipeline_def.stages) + 5 
        hops = 0

        while current_stage_name and hops < max_hops:
            hops += 1
            
            stage: Optional[MasterStageSpec] = self.pipeline_def.stages.get(current_stage_name)
            if not stage:
                self.logger.error(f"MASTER Stage '{current_stage_name}' not found in plan. Aborting loop.")
                fallback_idx = -1.0 
                await self._state_manager.update_status(
                    stage=fallback_idx, 
                    status=StageStatus.FAILURE.value, 
                    artifacts=[], 
                    reason=f"Master Stage '{current_stage_name}' not found in plan"
                )
                break

            current_stage_idx = stage.number if stage.number is not None else -1.0
            if current_stage_idx == -1.0:
                 self.logger.warning(f"MASTER Stage '{current_stage_name}' (Agent: {stage.agent_id}) does not have a numeric identifier. Using -1.0.")

            if hops >= max_hops:
                self.logger.warning(f"Max hops ({max_hops}) reached, breaking MASTER execution at stage '{current_stage_name}'.")
                await self._state_manager.update_status(
                    stage=current_stage_idx, 
                    status=StageStatus.FAILURE.value, 
                    artifacts=[], 
                    reason="Max hops reached"
                )
                break
            
            if stage.condition:
                self.logger.info(f"Evaluating Condition for MASTER Stage: '{current_stage_name}' (Definition: {stage.condition})")
                condition_met = self._parse_condition(stage.condition, context) 
                next_stage_name = stage.next_stage_true if condition_met else stage.next_stage_false
                self.logger.info(f"Condition result: {condition_met}, next MASTER stage: {next_stage_name}")
                
                current_stage_name = next_stage_name
                if not current_stage_name:
                     self.logger.info(f"Conditional branch in MASTER flow leads nowhere. Orchestration complete.")
                     break 
                else:
                     self.logger.debug(f"Continuing MASTER loop to process next stage: '{current_stage_name}'")
                     continue

            self.logger.info(f"Processing Agent for MASTER Stage: {current_stage_name} (Agent ID: {stage.agent_id})")
            try:
                if not stage.agent_id:
                     self.logger.error(f"MASTER Stage '{current_stage_name}' is missing agent_id. Aborting.")
                     await self._state_manager.update_status(
                         stage=current_stage_idx, 
                         status=StageStatus.FAILURE.value, 
                         artifacts=[], 
                         reason="Master Stage missing agent_id"
                     )
                     break

                agent_callable = await self._agent_provider.get(stage.agent_id)
                if agent_callable is None:
                    self.logger.error(f"Agent '{stage.agent_id}' not found for MASTER Stage '{current_stage_name}'. Aborting.")
                    await self._state_manager.update_status(
                        stage=current_stage_idx, 
                        status=StageStatus.FAILURE.value, 
                        artifacts=[], 
                        reason=f"Agent '{stage.agent_id}' not found"
                    )
                    break
                
                agent_input_context = copy.deepcopy(context) 
                if stage.inputs:
                    self.logger.debug(f"Merging inputs from MasterStageSpec '{current_stage_name}': {stage.inputs.keys()}")
                    agent_input_context.update(stage.inputs)
                    resolved_master_inputs = {}
                    for key, value in stage.inputs.items():
                        if isinstance(value, str) and value.startswith("context."):
                            path = value.split('.')[1:]
                            resolved_value = context
                            try:
                                for p_item in path:
                                    if isinstance(resolved_value, list) and p_item.isdigit():
                                        resolved_value = resolved_value[int(p_item)]
                                    elif isinstance(resolved_value, dict):
                                        resolved_value = resolved_value[p_item]
                                    else:
                                        raise KeyError
                            except (KeyError, TypeError, IndexError, ValueError):
                                self.logger.warning(f"Could not resolve MASTER input path '{value}' for stage '{current_stage_name}'. Using original string.")
                                resolved_value = value
                            resolved_master_inputs[key] = resolved_value
                        else:
                            resolved_master_inputs[key] = value
                    agent_input_context.update(resolved_master_inputs)

                self.logger.debug(f"Executing Agent '{stage.agent_id}' for MASTER Stage '{current_stage_name}' with input context keys: {list(agent_input_context.keys())}")
                
                stage_output = await agent_callable(agent_input_context)
                
                context['outputs'][current_stage_name] = stage_output
                
                self.logger.info(f"Agent '{stage.agent_id}' completed MASTER Stage '{current_stage_name}'.")
                await self._state_manager.update_status(
                    stage=current_stage_idx, 
                    status=StageStatus.SUCCESS.value, 
                    artifacts=[]
                )
            
            except Exception as e:
                self.logger.exception(f"Agent '{stage.agent_id}' failed during execution of MASTER stage '{current_stage_name}'. Attempting to pause state.")
                error_details = AgentErrorDetails(
                    error_type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                    agent_id=stage.agent_id,
                    stage_id=current_stage_name
                )
                run_id = self._state_manager.get_or_create_current_run_id()
                if run_id is None:
                    self.logger.error("Failed to get run_id from StateManager. Cannot save paused state.")
                    await self._state_manager.update_status(
                        stage=current_stage_idx, 
                        status=StageStatus.FAILURE.value, 
                        artifacts=[], 
                        reason=f"Agent '{stage.agent_id}' failed (could not get run_id): {type(e).__name__}", 
                        error_details=error_details
                    )
                    break
                try:
                    paused_details = PausedRunDetails(
                        run_id=str(run_id),
                        flow_id=self.pipeline_def.id,
                        paused_at_stage_id=current_stage_name,
                        timestamp=datetime.now(timezone.utc),
                        context_snapshot=context,
                        error_details=error_details,
                        reason="Paused due to agent error in master stage"
                    )
                    save_success = self._state_manager.save_paused_flow_state(paused_details)
                    final_reason = f"PAUSED_ON_ERROR{ ' (Save Failed)' if not save_success else '' }: Agent '{stage.agent_id}' failed: {type(e).__name__}"
                except Exception as pause_create_e:
                    self.logger.exception(f"Failed to create/save PausedRunDetails: {pause_create_e}")
                    final_reason = f"Agent '{stage.agent_id}' failed (pause state save failed): {type(e).__name__}"
                
                await self._state_manager.update_status(
                    stage=current_stage_idx, 
                    status=StageStatus.FAILURE.value, 
                    artifacts=[], 
                    reason=final_reason, 
                    error_details=error_details
                )
                break

            if not stage.condition:
                current_stage_name = stage.next_stage
                if not current_stage_name:
                    self.logger.info(f"MASTER Stage '{stage.agent_id}' completed, no next stage defined. Orchestration complete.")
                    break
                else:
                    self.logger.debug(f"Proceeding to next MASTER stage: '{current_stage_name}'")

        self.logger.info("MASTER execution loop finished.")
        return context

    async def run(self, plan: MasterExecutionPlan, context: Dict[str, Any]) -> Dict[str, Any]:
        """Starts a fresh execution of the Master Flow plan."""
        self.pipeline_def = plan 
        self.logger.info(f"AsyncOrchestrator starting MASTER flow: {self.pipeline_def.id} ({self.pipeline_def.name or 'No Name'})" )
        if 'outputs' not in context:
            context['outputs'] = {}
        return await self._execute_loop(start_stage_name=self.pipeline_def.start_stage, context=context)

    async def resume_flow(
        self,
        run_id: str,
        action: str,
        action_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resumes a paused MASTER flow based on the specified action."""
        self.logger.info(f"Attempting to resume MASTER flow run_id: {run_id} with action: {action}")

        try:
            paused_details = self._state_manager.load_paused_flow_state(run_id)
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
                    delete_success = self._state_manager.delete_paused_flow_state(run_id)
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
                delete_success = self._state_manager.delete_paused_flow_state(run_id)
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
                clear_success = self._state_manager.delete_paused_flow_state(run_id)
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


# ---------------------------------------------------------------------------
# DSL Validation Helpers (optional, requires jsonschema)
# ---------------------------------------------------------------------------

def _validate_dsl(data: dict) -> None:  # type: ignore[override]
    pass # No-op validation