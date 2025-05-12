"""Minimal execution runtime skeleton (Phase-6 scaffold).

This module will gradually evolve into the production‐grade orchestrator that
executes Flow YAML graphs.  For now we expose a *SyncOrchestrator* that can run
very simple flows in-process so we can start adding tests and build up the API
surface incrementally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Callable, Awaitable

import datetime as _dt
from datetime import datetime, timezone
import yaml
from pydantic import BaseModel, Field, ConfigDict
from chungoid.utils.agent_resolver import AgentProvider, RegistryAgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.common_enums import StageStatus
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.flows import PausedRunDetails

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

    def run(self, plan: ExecutionPlan, context: Dict[str, Any]) -> Dict[str, Any]:
        # This is a placeholder for the synchronous orchestrator logic
        self.logger.info("SyncOrchestrator.run called (placeholder)")
        # Simple sequential execution for now, ignoring conditions
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
            # Placeholder for actual agent execution and context update
            # context['outputs'][current_stage_name] = {"message": f"Output from {current_stage_name}"}
            
            if stage.condition:
                if self._parse_condition(stage.condition, context):
                    current_stage_name = stage.next_stage_true
                else:
                    current_stage_name = stage.next_stage_false
            else:
                current_stage_name = stage.next_stage
        
        return context


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
        pipeline_def: ExecutionPlan,
        config: Dict[str, Any],
        agent_provider: AgentProvider,
        state_manager: StateManager
    ):
        super().__init__(pipeline_def, config)
        self._agent_provider = agent_provider
        self._state_manager = state_manager

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

    async def run(self, plan: ExecutionPlan, context: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"AsyncOrchestrator running plan starting with: {plan.start_stage}")
        current_stage_name = plan.start_stage
        if 'outputs' not in context:
            context['outputs'] = {}
        max_hops = len(plan.stages) + 5
        hops = 0
        while current_stage_name and hops < max_hops:
            hops += 1
            
            # Get stage definition first
            stage = plan.stages.get(current_stage_name)
            if not stage:
                self.logger.error(f"Stage '{current_stage_name}' not found in plan. Aborting.")
                # Attempt to use previous stage index or a default if start stage is missing
                fallback_idx = -1.0 # Or potentially retrieve last known good index
                await self._state_manager.update_status(stage=fallback_idx, status=StageStatus.FAILURE, artifacts=[], reason=f"Stage '{current_stage_name}' not found in plan")
                break

            # Get stage number reliably
            current_stage_idx = stage.number if stage.number is not None else -1.0
            if current_stage_idx == -1.0:
                 self.logger.warning(f"Stage '{current_stage_name}' does not have a numeric identifier. Using -1.0. Status tracking might be affected.")

            # Check max hops *after* getting stage info
            if hops >= max_hops:
                self.logger.warning(f"Max hops ({max_hops}) reached, breaking execution at stage '{current_stage_name}'.")
                await self._state_manager.update_status(stage=current_stage_idx, status=StageStatus.FAILURE, artifacts=[], reason="Max hops reached")
                break
            
            # --- Check if this is a purely conditional stage ---
            # A conditional stage might not have an agent_id if its sole purpose is routing
            # is_conditional_routing_stage = stage.condition is not None and not stage.agent_id 
            # Even if it has an agent_id, we might treat stages with conditions differently later,
            # but for now, let's assume if there's a condition, we evaluate it first.
            
            if stage.condition:
                self.logger.info(f"Evaluating Condition for Stage: '{current_stage_name}' (Definition: {stage.condition})")
                condition_met = self._parse_condition(stage.condition, context) # Evaluate against current context
                next_stage_name = None
                if condition_met:
                    self.logger.info(f"Condition TRUE for stage '{current_stage_name}', next: {stage.next_stage_true}")
                    next_stage_name = stage.next_stage_true
                else:
                    self.logger.info(f"Condition FALSE for stage '{current_stage_name}', next: {stage.next_stage_false}")
                    next_stage_name = stage.next_stage_false
                
                # Set the next stage name and continue the loop immediately
                # This skips any agent execution attempt for the current conditional stage itself
                current_stage_name = next_stage_name
                if not current_stage_name:
                     self.logger.info(f"Condition evaluated for '{stage.agent_id}', but no next stage defined. Orchestration complete.")
                     break # End loop if condition leads nowhere
                else:
                     self.logger.debug(f"Continuing loop to process next stage: '{current_stage_name}'")
                     continue # Go to next loop iteration

            # --- If not conditional stage, proceed with agent execution ---
            self.logger.info(f"Processing Agent for Stage: {current_stage_name} (Agent: {stage.agent_id})")
            try:
                # If agent_id is missing here, it's an invalid plan for a non-conditional stage
                if not stage.agent_id:
                     self.logger.error(f"Stage '{current_stage_name}' is missing agent_id. Aborting.")
                     await self._state_manager.update_status(stage=current_stage_idx, status=StageStatus.FAILURE, artifacts=[], reason="Stage missing agent_id")
                     break

                agent_callable = await self._agent_provider.get(stage.agent_id)
                if agent_callable is None:
                    self.logger.error(f"Agent '{stage.agent_id}' not found for Stage '{current_stage_name}'. Aborting.")
                    await self._state_manager.update_status(stage=current_stage_idx, status=StageStatus.FAILURE, artifacts=[], reason="Agent not found")
                    break
                
                # Prepare the specific context for this agent invocation
                # Use deepcopy to prevent issues with nested mutable objects like context['outputs']
                agent_input_context = copy.deepcopy(context) 
                if stage.inputs:
                    resolved_inputs = {}
                    for key, value in stage.inputs.items():
                        if isinstance(value, str) and value.startswith("context."):
                            path = value.split('.')[1:]
                            resolved_value = context # Resolve from the main context
                            try:
                                for p_item in path:
                                    resolved_value = resolved_value[p_item]
                            except (KeyError, TypeError):
                                self.logger.warning(f"Could not resolve input path '{value}' for stage '{current_stage_name}'. Using original string.")
                                resolved_value = value
                            resolved_inputs[key] = resolved_value
                        else:
                            resolved_inputs[key] = value
                    # Merge resolved inputs, overwriting keys from global context if necessary
                    agent_input_context.update(resolved_inputs)
                
                self.logger.debug(f"Executing Agent '{stage.agent_id}' for Stage '{current_stage_name}' with input context keys: {list(agent_input_context.keys())}")
                
                # Execute the agent with its specific input context
                stage_output = await agent_callable(agent_input_context)
                
                # Merge output back into the main context
                context['outputs'][current_stage_name] = stage_output
                
                self.logger.info(f"Agent '{stage.agent_id}' completed Stage '{current_stage_name}'.")
                await self._state_manager.update_status(stage=current_stage_idx, status=StageStatus.SUCCESS, artifacts=[])
            except KeyError as e:
                self.logger.error(f"Agent '{stage.agent_id}' not resolved (KeyError) for Stage '{current_stage_name}': {e}. Aborting.")
                await self._state_manager.update_status(stage=current_stage_idx, status=StageStatus.FAILURE, artifacts=[], reason="Agent resolution failed")
                break
            except Exception as e:
                self.logger.exception(f"Agent '{stage.agent_id}' failed during execution of stage '{current_stage_name}'. Attempting to pause state.")

                # 1. Create error details
                error_details = AgentErrorDetails(
                    error_type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                    agent_id=stage.agent_id,
                    stage_id=current_stage_name
                )

                # 2. Get current run_id
                run_id = self._state_manager.get_or_create_current_run_id()
                if run_id is None:
                    self.logger.error("Failed to get run_id from StateManager. Cannot save paused state.")
                    # Fallback: Log failure and break
                    self._state_manager.update_status(
                        stage=current_stage_name,
                        status=StageStatus.FAILURE.value,
                        artifacts=[],
                        reason=f"Agent execution failed (could not get run_id): {type(e).__name__}",
                        error_details=error_details
                    )
                    break

                # 3. Create PausedRunDetails
                try:
                    paused_details = PausedRunDetails(
                        run_id=str(run_id),
                        paused_at_stage_id=current_stage_name,
                        timestamp=datetime.now(timezone.utc),
                        context_snapshot=context,
                        error_details=error_details,
                        reason="Paused due to agent error"
                    )
                except Exception as pause_create_e:
                    self.logger.exception(f"Failed to create PausedRunDetails: {pause_create_e}")
                    # Fallback: Update status and break
                    self._state_manager.update_status(
                        stage=current_stage_name,
                        status=StageStatus.FAILURE.value,
                        artifacts=[],
                        reason=f"Agent execution failed (could not create pause details): {type(e).__name__}",
                        error_details=error_details
                    )
                    break

                # 4. Save paused state
                save_success = self._state_manager.save_paused_flow_state(paused_details)
                if not save_success:
                    self.logger.error(f"Failed to save paused state for run {run_id} stage {current_stage_name}.")
                    # Optionally add note to reason?
                    final_reason = f"PAUSED_ON_ERROR (Save Failed): Agent execution failed: {type(e).__name__}"
                else:
                    final_reason = f"PAUSED_ON_ERROR: Agent execution failed: {type(e).__name__}"

                # 5. Update main status (indicate pause/failure)
                # Ensure this is awaited regardless of save success
                await self._state_manager.update_status(
                    stage=current_stage_idx, # Use the parsed stage index
                    status=StageStatus.FAILURE.value,
                    artifacts=[],
                    reason=final_reason,
                    error_details=error_details
                )
                # 6. Break execution loop
                break
            
            # Determine next stage ONLY if not handled by condition evaluation above
            # (This block is now only reached by non-conditional stages)
            current_stage_name = stage.next_stage
            if not current_stage_name:
                 self.logger.info(f"No next stage defined after non-conditional stage '{stage.agent_id}'. Orchestration complete.")

        self.logger.info(f"AsyncOrchestrator run finished. Final context output keys: {list(context.get('outputs', {}).keys())}")
        return context


try:
    import jsonschema
    from functools import lru_cache
    from pathlib import Path

    _SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "execution_dsl.json"

    @lru_cache(maxsize=1)
    def _load_schema() -> dict:  # pragma: no cover
        with _SCHEMA_PATH.open("r", encoding="utf-8") as fp:
            import json

            return json.load(fp)

    def _validate_dsl(data: dict) -> None:  # pragma: no cover
        schema = _load_schema()
        jsonschema.validate(instance=data, schema=schema)


except ModuleNotFoundError:  # jsonschema not available – skip runtime validation

    def _validate_dsl(data: dict) -> None:  # type: ignore[override]
        return 