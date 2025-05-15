"""Minimal execution runtime skeleton (Phase-6 scaffold).

This module will gradually evolve into the production‐grade orchestrator that
executes Flow YAML graphs.  For now we expose a *SyncOrchestrator* that can run
very simple flows in-process so we can start adding tests and build up the API
surface incrementally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import datetime as _dt
from datetime import datetime, timezone
import yaml
from pydantic import BaseModel, Field, ConfigDict
from chungoid.utils.agent_resolver import AgentProvider, RegistryAgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.common_enums import StageStatus, FlowPauseStatus
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec

# Import for reviewer agent and its schemas
from chungoid.runtime.agents.system_master_planner_reviewer_agent import (
    MasterPlannerReviewerAgent,
)  # Assuming AGENT_ID is on class

# New import for Metrics
from chungoid.utils.metrics_store import MetricsStore
from chungoid.schemas.metrics import MetricEvent, MetricEventType

import logging
import traceback
import copy
import inspect
import asyncio
import uuid

# New imports for category resolution errors
from chungoid.utils.agent_resolver import NoAgentFoundForCategoryError, AmbiguousAgentCategoryError

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
    created: _dt.datetime = Field(default_factory=_dt.datetime.utcnow)
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
                        expected_value_str.strip("'\"")
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

    # Maximum number of hops to prevent infinite loops if not otherwise caught
    MAX_HOPS = 100
    # Default number of retries for certain recoverable agent errors, if not specified in stage
    DEFAULT_AGENT_RETRIES = 1

    ARTIFACT_OUTPUT_KEY = "_mcp_generated_artifacts_relative_paths_"  # ADDED

    _current_run_id: Optional[str] = (
        None  # Stores the current run_id for the active flow
    )
    _last_successful_stage_output: Optional[Any] = (
        None  # Stores the output of the last successfully completed stage in the current run
    )

    def __init__(
        self,
        config: Dict[str, Any],
        agent_provider: AgentProvider,
        state_manager: StateManager,
        metrics_store: MetricsStore,
        master_planner_reviewer_agent_id: str = MasterPlannerReviewerAgent.AGENT_ID,
    ):
        """
        Initializes the AsyncOrchestrator.

        Args:
            config: Project configuration.
            agent_provider: Provider for resolving and invoking agents.
            state_manager: Manager for flow state persistence and retrieval.
            metrics_store: Store for recording execution metrics.
            master_planner_reviewer_agent_id: The agent ID for the Master Planner Reviewer.
        """
        self.config = config
        self.agent_provider = agent_provider
        self.state_manager = state_manager
        self.metrics_store = metrics_store
        self.logger = logging.getLogger(__name__)
        self.master_planner_reviewer_agent_id = master_planner_reviewer_agent_id

        self.current_plan: Optional[MasterExecutionPlan] = None
        self._current_run_id: Optional[str] = None

        self.logger.info(
            f"AsyncOrchestrator initialized. Reviewer Agent ID: {self.master_planner_reviewer_agent_id}"
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
        # Ensure common fields are not duplicated if passed in kwargs
        metric_data = {
            "flow_id": flow_id,
            "run_id": run_id,
            **kwargs,  # stage_id, agent_id, etc.
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
                f"Failed to emit metric event {event_type} for run {run_id}: {e}",
                exc_info=True,
            )

    def _evaluate_criterion(
        self, criterion: str, stage_outputs: Dict[str, Any]
    ) -> bool:
        """Evaluates a single success criterion string against stage_outputs."""
        self.logger.debug(
            f"Evaluating criterion: '{criterion}' against outputs: {stage_outputs}"
        )

        # EXISTS Check (e.g., "outputs.some_key EXISTS")
        if criterion.upper().endswith(" EXISTS"):
            path_to_check = criterion[: criterion.upper().rfind(" EXISTS")].strip()
            current_val = stage_outputs
            try:
                for key_part in path_to_check.split("."):
                    if isinstance(current_val, dict):
                        current_val = current_val[key_part]
                    elif isinstance(current_val, list) and key_part.isdigit():
                        current_val = current_val[int(key_part)]
                    else:
                        self.logger.debug(
                            f"Criterion '{criterion}' FAILED: Path '{path_to_check}' not fully found (part: {key_part})."
                        )
                        return False  # Path part not found
                self.logger.debug(
                    f"Criterion '{criterion}' PASSED (EXISTS check). Value at path: {current_val}"
                )
                return True  # Path exists
            except (KeyError, IndexError, TypeError):
                self.logger.debug(
                    f"Criterion '{criterion}' FAILED: Path '{path_to_check}' not found (exception)."
                )
                return False  # Path does not exist

        # Simple Comparison (e.g., "outputs.metric == true", "outputs.count > 0")
        # Re-use parts of _parse_condition logic but scoped to stage_outputs
        try:
            parts = []
            comparator = None
            supported_comparators = {
                ">=": lambda a, b: a >= b,
                "<=": lambda a, b: a <= b,
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
                ">": lambda a, b: a > b,
                "<": lambda a, b: a < b,
            }
            for op_str in supported_comparators.keys():  # Check longer ops first
                if op_str in criterion:
                    parts = criterion.split(op_str, 1)
                    comparator = op_str
                    break

            if not comparator or len(parts) != 2:
                self.logger.warning(
                    f"Unsupported criterion format or unknown operator: {criterion}. Evaluates to FALSE."
                )
                return False

            path_str = parts[0].strip()
            expected_value_literal_str = parts[1].strip()

            actual_val = stage_outputs
            for key in path_str.split("."):
                if isinstance(actual_val, dict) and key in actual_val:
                    actual_val = actual_val[key]
                elif isinstance(actual_val, list) and key.isdigit():
                    actual_val = actual_val[int(key)]
                else:
                    self.logger.debug(
                        f"Criterion '{criterion}' FAILED: Path '{path_str}' not fully found in outputs."
                    )
                    return False

            # Attempt to coerce expected_value_literal_str to type of actual_val
            coerced_expected_val: Any
            if isinstance(actual_val, bool):
                coerced_expected_val = expected_value_literal_str.lower() in [
                    "true",
                    "1",
                    "yes",
                ]
            elif isinstance(actual_val, int):
                coerced_expected_val = int(expected_value_literal_str.strip("'\""))
            elif isinstance(actual_val, float):
                coerced_expected_val = float(expected_value_literal_str.strip("'\""))
            else:  # Default to string comparison
                coerced_expected_val = expected_value_literal_str.strip("'\"")

            result = supported_comparators[comparator](actual_val, coerced_expected_val)
            self.logger.debug(
                f"Criterion '{criterion}' evaluation: '{actual_val}' {comparator} '{coerced_expected_val}' -> {result}"
            )
            return result

        except Exception as e:
            self.logger.warning(
                f"Error evaluating criterion '{criterion}': {e}. Evaluates to FALSE.",
                exc_info=True,
            )
            return False

    async def _check_success_criteria(
        self, stage_name: str, stage_spec: MasterStageSpec, context: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Checks all success_criteria for a given stage. Returns (all_passed, list_of_failed_criteria)."""
        if not stage_spec.success_criteria:
            return True, []  # No criteria means success

        all_passed = True
        failed_criteria: List[str] = []
        stage_outputs = context.get("outputs", {}).get(stage_name, {})

        self.logger.info(
            f"Checking {len(stage_spec.success_criteria)} success criteria for stage '{stage_name}'."
        )
        for criterion_str in stage_spec.success_criteria:
            if not self._evaluate_criterion(criterion_str, stage_outputs):
                all_passed = False
                failed_criteria.append(criterion_str)

        if not all_passed:
            self.logger.warning(
                f"Stage '{stage_name}' failed success criteria check. Failed criteria: {failed_criteria}"
            )
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
                        expected_value_str.strip("'\"")
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
            return False

    def _resolve_input_values(
        self, inputs_spec: Dict[str, Any], context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.logger.critical(
            f"RESOLVE_INPUT_ENTRY: inputs_spec='{str(inputs_spec)}', context_keys={list(context_data.keys())}"
        )  # NEW CRITICAL LOG

        resolved_inputs = {}
        if not isinstance(inputs_spec, dict):
            self.logger.warning(
                f"_resolve_input_values expected inputs_spec to be a dict, got {type(inputs_spec)}. Returning empty resolved inputs."
            )
            return {}

        temp_context_for_this_stage = copy.deepcopy(context_data)

        prev_stage_outputs_spec_val = inputs_spec.get("previous_stage_outputs")
        if (
            isinstance(prev_stage_outputs_spec_val, str)
            and not prev_stage_outputs_spec_val.lower().startswith("context.")
            and prev_stage_outputs_spec_val.lower() != "previous_output"
        ):
            prev_stage_id_to_load = prev_stage_outputs_spec_val
            outputs_data = temp_context_for_this_stage.setdefault("outputs", {})
            prev_stage_output_data = outputs_data.get(prev_stage_id_to_load)

            if prev_stage_output_data is not None:
                resolved_inputs["previous_stage_outputs"] = prev_stage_output_data
                temp_context_for_this_stage["resolved_previous_stage_output_data"] = (
                    prev_stage_output_data
                )
            else:
                resolved_inputs["previous_stage_outputs"] = None

        temp_inputs_spec_items = list(inputs_spec.items())
        for input_name, context_path in temp_inputs_spec_items:
            if (
                input_name == "previous_stage_outputs"
                and isinstance(context_path, str)
                and not context_path.lower().startswith("context.")
                and context_path.lower() != "previous_output"
            ):
                continue

            resolved_value = context_path
            if isinstance(context_path, str):
                context_path_lower = context_path.lower()
                if context_path_lower == "previous_output":
                    if self._last_successful_stage_output is not None:
                        resolved_value = self._last_successful_stage_output
                    else:
                        resolved_value = context_path
                elif context_path_lower.startswith("context."):
                    path_parts = context_path.split(".")[1:]
                    current_val = temp_context_for_this_stage
                    valid_path = True
                    try:
                        for part_idx, part in enumerate(path_parts):
                            if isinstance(current_val, dict):
                                next_val = current_val.get(part, _SENTINEL)
                                if next_val is not _SENTINEL:
                                    current_val = next_val
                                else:
                                    valid_path = False
                                    break
                            elif isinstance(current_val, list) and part.isdigit():
                                current_val = current_val[int(part)]
                            elif hasattr(current_val, part):
                                current_val = getattr(current_val, part)
                            else:
                                valid_path = False
                                break
                        if valid_path:
                            resolved_value = current_val
                    except (KeyError, IndexError, AttributeError, TypeError) as e:
                        self.logger.warning(
                            f"Error resolving '{context_path}' from context: {e}. Using literal value '{context_path}'."
                        )
                else:
                    try:
                        if (
                            "." not in context_path
                            and context_path in temp_context_for_this_stage
                        ):
                            resolved_value = temp_context_for_this_stage[context_path]
                    except (KeyError, IndexError, AttributeError, TypeError) as e:
                        self.logger.warning(
                            f"Error attempting to resolve '{context_path}' as a direct path from context: {e}. Using literal value."
                        )
                        resolved_value = context_path
            elif (
                not isinstance(context_path, (str, int, float, bool, list, dict))
                and context_path is not None
            ):
                self.logger.warning(
                    f"Input '{input_name}' has an unexpected literal type: {type(context_path)}. Using it as is."
                )

            resolved_inputs[input_name] = resolved_value
        return resolved_inputs

    async def _invoke_reviewer_and_get_suggestion(
        self,
        run_id: str,
        flow_id: str,
        current_stage_name: str,
        agent_error_details: Optional[AgentErrorDetails],
        current_context: Dict[str, Any],
        # current_plan: MasterExecutionPlan # available as self.current_plan
    ) -> Optional[MasterPlannerReviewerOutput]:
        """Invokes the MasterPlannerReviewerAgent and returns its suggestion."""
        if not self.master_planner_reviewer_agent_id:
            self.logger.warning(f"[RunID: {run_id}] MasterPlannerReviewerAgent ID not configured. Skipping review.")
            return None

        self.logger.info(f"[RunID: {run_id}] Stage '{current_stage_name}' failed. Invoking MasterPlannerReviewerAgent.")
        try:
            reviewer_agent_callable = self.agent_provider.get(self.master_planner_reviewer_agent_id)
            
            synthetic_paused_run_details = {
                "run_id": run_id,
                "flow_id": flow_id,
                "paused_at_stage_id": current_stage_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": FlowPauseStatus.AGENT_ERROR.value,
                "context_snapshot_ref": None, 
                "error_details": agent_error_details.model_dump() if agent_error_details else None,
                "clarification_request": None
            }

            reviewer_input = MasterPlannerReviewerInput(
                current_master_plan=self.current_plan,
                paused_run_details=synthetic_paused_run_details,
                pause_status=FlowPauseStatus.AGENT_ERROR,
                paused_stage_id=current_stage_name,
                triggering_error_details=agent_error_details,
                full_context_at_pause=copy.deepcopy(current_context)
            )

            self.logger.debug(f"[RunID: {run_id}] MasterPlannerReviewerAgent input: {reviewer_input.model_dump_json(indent=2)}")
            
            reviewer_output: MasterPlannerReviewerOutput # Type hint
            if inspect.iscoroutinefunction(reviewer_agent_callable) or inspect.iscoroutinefunction(getattr(reviewer_agent_callable, '__call__', None)):
                reviewer_output = await reviewer_agent_callable(reviewer_input, full_context=current_context)
            elif callable(reviewer_agent_callable):
                reviewer_output = await asyncio.to_thread(reviewer_agent_callable, reviewer_input, full_context=current_context)
            else:
                raise TypeError(f"Reviewer agent {self.master_planner_reviewer_agent_id} is not callable.")

            self.logger.info(f"[RunID: {run_id}] MasterPlannerReviewerAgent suggested: {reviewer_output.suggestion_type.value}. Reasoning: {reviewer_output.reasoning}")
            return reviewer_output

        except Exception as reviewer_exc:
            self.logger.error(f"[RunID: {run_id}] Error during MasterPlannerReviewerAgent invocation: {reviewer_exc}", exc_info=True)
            return None

    async def _execute_master_flow_loop(
        self, start_stage_name: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.logger.critical(
            f"EXEC_LOOP_ENTRY: stage='{start_stage_name}', context_keys={list(context.keys())}"
        )  # NEW CRITICAL LOG

        if not self.current_plan:
            self.logger.error(
                "Current plan not set in _execute_master_flow_loop. This indicates a programming error."
            )
            return {"_flow_error": "Orchestrator internal error: current_plan not set."}

        flow_id = self.current_plan.id
        run_id = self._current_run_id

        if not run_id:  # Defensive check
            self.logger.error(
                "Run ID not set in _execute_master_flow_loop. This indicates a programming error."
            )
            # This might occur if resume_flow or run didn't set it properly.
            return {"_flow_error": "Orchestrator internal error: run_id not set."}

        current_stage_name: Optional[str] = start_stage_name
        current_context = copy.deepcopy(context)
        if "outputs" not in current_context or not isinstance(
            current_context["outputs"], dict
        ):
            current_context["outputs"] = {}
        visited_stages: List[str] = []
        max_hops_for_flow = self.MAX_HOPS

        while current_stage_name and current_stage_name != "FINAL_STEP":
            if current_stage_name in visited_stages:
                self.logger.warning(
                    f"Loop detected: Stage '{current_stage_name}' visited again in plan '{flow_id}', run '{run_id}'. Aborting."
                )
                self._emit_metric(
                    event_type=MetricEventType.ORCHESTRATOR_INFO,
                    flow_id=flow_id,
                    run_id=run_id,  # Use flow_id and run_id from self
                    data={
                        "level": "WARNING",
                        "message": "Loop detected, aborting flow.",
                        "stage": current_stage_name,
                        "visited_log": visited_stages,
                    },
                )
                current_context["_flow_error"] = {
                    "message": "Loop detected, execution aborted.",
                    "stage": current_stage_name,
                }
                return current_context
            visited_stages.append(current_stage_name)

            if len(visited_stages) > max_hops_for_flow:
                self.logger.warning(
                    f"Max hops ({max_hops_for_flow}) reached for plan '{flow_id}', run '{run_id}'. Aborting."
                )
                self._emit_metric(
                    event_type=MetricEventType.ORCHESTRATOR_INFO,
                    flow_id=flow_id,
                    run_id=run_id,  # Use flow_id and run_id from self
                    data={
                        "level": "WARNING",
                        "message": "Max hops reached, aborting flow.",
                        "stage": current_stage_name,
                        "max_hops": max_hops_for_flow,
                    },
                )
                current_context["_flow_error"] = {
                    "message": "Max hops reached, execution aborted.",
                    "stage": current_stage_name,
                }
                return current_context

            current_context["_current_stage_name"] = current_stage_name

            stage_spec = self.current_plan.stages.get(
                current_stage_name
            )  # Use self.current_plan
            if not stage_spec:
                self.logger.error(
                    f"Stage '{current_stage_name}' not found in plan '{flow_id}', run '{run_id}'. Aborting."
                )
                self._emit_metric(
                    event_type=MetricEventType.ORCHESTRATOR_INFO,
                    flow_id=flow_id,
                    run_id=run_id,  # Use flow_id and run_id from self
                    data={
                        "level": "ERROR",
                        "message": "Stage not found in plan, aborting.",
                        "stage_name": current_stage_name,
                    },
                )
                current_context["_flow_error"] = {
                    "message": f"Stage '{current_stage_name}' not found in plan.",
                    "stage": current_stage_name,
                }
                return current_context

            self.logger.info(
                f"[RunID: {self._current_run_id}] Executing master stage: '{current_stage_name}' (Number: {current_stage_spec.number}) using agent_id: '{current_stage_spec.agent_id if current_stage_spec.agent_id else 'N/A - Category based'}"
            )
            # Emit metric for stage start
            self._emit_metric(
                event_type=MetricEventType.STAGE_START,
                flow_id=self.current_plan.id,
                run_id=self._current_run_id,
                data={
                    "stage_name": current_stage_name,
                    "stage_number": current_stage_spec.number,
                    "agent_id": current_stage_spec.agent_id, # Log original spec
                    "agent_category": current_stage_spec.agent_category # Log original spec
                },
            )

            agent_callable = None
            resolved_agent_id_for_stage = current_stage_spec.agent_id # Default to specified agent_id
            stage_execution_error = None

            try:
                if current_stage_spec.agent_id:
                    agent_callable = self.agent_provider.get(current_stage_spec.agent_id)
                    # resolved_agent_id_for_stage is already set
                elif current_stage_spec.agent_category:
                    self.logger.info(f"[RunID: {self._current_run_id}] Resolving agent by category: {current_stage_spec.agent_category} with preferences: {current_stage_spec.agent_selection_preferences}")
                    resolved_agent_id_for_stage, agent_callable = await self.agent_provider.resolve_agent_by_category(
                        current_stage_spec.agent_category,
                        current_stage_spec.agent_selection_preferences
                    )
                    self.logger.info(f"[RunID: {self._current_run_id}] Resolved category '{current_stage_spec.agent_category}' to agent_id: '{resolved_agent_id_for_stage}'")
                else:
                    self.logger.error(f"[RunID: {self._current_run_id}] Stage '{current_stage_name}' has neither agent_id nor agent_category specified.")
                    stage_execution_error = AgentErrorDetails(
                        error_code="INVALID_STAGE_DEFINITION",
                        message=f"Stage '{current_stage_name}' must have agent_id or agent_category.",
                        can_retry=False
                    )
            except (NoAgentFoundForCategoryError, AmbiguousAgentCategoryError) as cat_err:
                self.logger.error(f"[RunID: {self._current_run_id}] Failed to resolve agent for category '{current_stage_spec.agent_category}': {cat_err}")
                stage_execution_error = AgentErrorDetails(
                    error_code="AGENT_CATEGORY_RESOLUTION_FAILED",
                    message=str(cat_err),
                    details={"category": current_stage_spec.agent_category, "preferences": current_stage_spec.agent_selection_preferences},
                    can_retry=False # Typically, category resolution issues are not retriable without plan change
                )
            except KeyError as e: # From agent_provider.get if agent_id not found
                self.logger.error(f"[RunID: {self._current_run_id}] Agent '{current_stage_spec.agent_id}' not found: {e}")
                stage_execution_error = AgentErrorDetails(
                    error_code="AGENT_NOT_FOUND",
                    message=f"Agent ID '{current_stage_spec.agent_id}' not found in provider.",
                    details={"agent_id": current_stage_spec.agent_id},
                    can_retry=False
                )
            except Exception as e: # Catch any other unexpected error during agent resolution
                self.logger.error(f"[RunID: {self._current_run_id}] Unexpected error resolving agent for stage '{current_stage_name}': {e}")
                stage_execution_error = AgentErrorDetails(
                    error_code="AGENT_RESOLUTION_UNEXPECTED_ERROR",
                    message=f"An unexpected error occurred while resolving the agent: {traceback.format_exc()}",
                    can_retry=False
                )

            # If agent_callable is still None due to an error caught above, or if stage_execution_error is set
            if stage_execution_error or not agent_callable:
                # ... (existing error handling logic for stage_execution_error, like invoking reviewer or pausing) ...
                # This part will be complex and depends on the existing error handling flow for stage_execution_error
                # For now, ensure status is FAIL and proceed to error handling section of the loop.
                current_stage_result_status = StageStatus.FAIL
                if not stage_execution_error: # Should not happen if agent_callable is None, but as a safeguard
                    stage_execution_error = AgentErrorDetails(error_code="INTERNAL_ERROR", message="Agent callable not resolved but no specific error captured.")
                
                self._update_run_status_with_stage_result(
                    stage_name=current_stage_name,
                    stage_number=current_stage_spec.number,
                    status=StageStatus.FAIL,
                    error_details=stage_execution_error,
                    reason=stage_execution_error.message
                )
                # Transition to error handling logic of the loop (e.g., reviewer agent invocation or pause)
                # This will likely involve setting next_master_stage_key based on on_failure or pausing.
                # The existing error handling path for `stage_execution_error` should be leveraged.
                # For now, this simplified path assumes it will fall into the existing error handling logic correctly.
                
                # Update: Need to explicitly set path for error handling based on existing logic
                effective_on_failure = current_stage_spec.on_failure
                if not effective_on_failure: # Default behavior if on_failure is not specified
                    self.logger.warning(f"[RunID: {self._current_run_id}] Stage '{current_stage_name}' failed (agent resolution) and no on_failure policy defined. Defaulting to PAUSE_FOR_INTERVENTION.")
                    paused_state_details = self._create_paused_state_details(
                        master_flow_id=self.current_plan.id,
                        paused_stage_id=current_stage_name,
                        execution_context=copy.deepcopy(context), # Save a copy
                        error_details_model=stage_execution_error,
                        status_reason=f"Agent resolution failed: {stage_execution_error.message}",
                        clarification_request=None,
                        pause_status=StageStatus.PAUSED_FOR_AGENT_FAILURE # Generic pause for agent issues
                    )
                    await self.state_manager.save_paused_flow_state(self._current_run_id, paused_state_details)
                    self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=self.current_plan.id, run_id=self._current_run_id, data={"reason": "Agent resolution failure"})
                    return context # End execution for this run, requires resume

                if effective_on_failure.action == "FAIL_MASTER_FLOW":
                    self.logger.error(f"[RunID: {self._current_run_id}] Stage '{current_stage_name}' failed (agent resolution). Master flow configured to FAIL_MASTER_FLOW. Message: {effective_on_failure.log_message}")
                    current_master_stage_key = "_END_FAILURE_" # Signal loop to terminate
                    # Final context will be returned by the loop
                elif effective_on_failure.action == "GOTO_MASTER_STAGE":
                    next_master_stage_key = effective_on_failure.target_master_stage_key
                    self.logger.info(f"[RunID: {self._current_run_id}] Stage '{current_stage_name}' failed (agent resolution). Transitioning to on_failure stage: '{next_master_stage_key}'.")
                    if next_master_stage_key not in self.current_plan.stages and next_master_stage_key not in ["_END_SUCCESS_", "_END_FAILURE_"]:
                        self.logger.error(f"[RunID: {self._current_run_id}] on_failure target_master_stage_key '{next_master_stage_key}' not found in plan. Terminating.")
                        current_master_stage_key = "_END_FAILURE_"
                    else:
                        current_master_stage_key = next_master_stage_key
                elif effective_on_failure.action == "PAUSE_FOR_INTERVENTION":
                    self.logger.info(f"[RunID: {self._current_run_id}] Stage '{current_stage_name}' failed (agent resolution). Pausing for human intervention as per on_failure policy.")
                    paused_state_details = self._create_paused_state_details(
                        master_flow_id=self.current_plan.id,
                        paused_stage_id=current_stage_name,
                        execution_context=copy.deepcopy(context),
                        error_details_model=stage_execution_error,
                        status_reason=f"Agent resolution failed (on_failure policy): {stage_execution_error.message}",
                        clarification_request=None,
                        pause_status=StageStatus.PAUSED_FOR_INTERVENTION
                    )
                    await self.state_manager.save_paused_flow_state(self._current_run_id, paused_state_details)
                    self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=self.current_plan.id, run_id=self._current_run_id, data={"reason": "Agent resolution failure, on_failure policy"})
                    return context # End execution
                else: # Should not happen with validated MasterStageFailurePolicy
                    self.logger.error(f"[RunID: {self._current_run_id}] Unknown on_failure action: {effective_on_failure.action}. Terminating.")
                    current_master_stage_key = "_END_FAILURE_"
                
                # If we didn't return (pause) or set to _END_FAILURE_, we continue the loop to the next stage (error or normal)
                if current_master_stage_key != "_END_FAILURE_" and current_master_stage_key not in ["_END_SUCCESS_", "_END_FAILURE_"]:
                    context["_last_master_stage_hop_info"] = {
                        "from": current_stage_name,
                        "to": current_master_stage_key,
                        "reason": "Agent resolution failure, on_failure transition"
                    }
                    continue # To the next iteration of the while loop with the new current_master_stage_key
                else: # Reached a terminal state due to error handling
                    break # Exit the while loop, to be handled by outer return logic

            # If we are here, agent_callable should be resolved.
            # Original agent invocation logic follows...
            # ... (rest of the agent invocation using agent_callable and resolved_agent_id_for_stage, 
            #      success criteria checking, context updates, and transition to next stage or error handling for *agent execution* errors) ...
            
            # Placeholder for where the original agent execution call was:
            # stage_output_payload_or_error = await self._execute_single_agent_stage(
            #    agent_callable, 
            #    resolved_agent_id_for_stage, # Use the resolved ID
            #    current_stage_name, 
            #    current_stage_spec, 
            #    invokable_context
            # )

            stage_start_time = datetime.now(timezone.utc)
            agent_invocation_succeeded = False
            stage_result_payload: Optional[Any] = None
            agent_error_details: Optional[AgentErrorDetails] = None

            try:
                agent_id_to_invoke = resolved_agent_id_for_stage
                agent_inputs_spec = stage_spec.inputs or {}

                resolved_inputs = self._resolve_input_values(
                    agent_inputs_spec, current_context
                )

                self.logger.debug(
                    f"Invoking agent '{agent_id_to_invoke}' for stage '{current_stage_name}' with resolved inputs: {resolved_inputs}"
                )
                self._emit_metric(
                    event_type=MetricEventType.AGENT_INVOCATION_START,
                    flow_id=flow_id,
                    run_id=run_id,
                    stage_id=current_stage_name,
                    master_stage_id=current_stage_name,
                    agent_id=agent_id_to_invoke,
                    data={"resolved_inputs": copy.deepcopy(resolved_inputs)},
                )

                # Get the agent callable (can be sync or async)
                agent_callable = self.agent_provider.get(agent_id_to_invoke)

                self.logger.debug(
                    f"Retrieved agent_callable: {agent_callable} (type: {type(agent_callable)}) for agent ID: {agent_id_to_invoke}"
                )

                # Inspect and call appropriately
                if inspect.iscoroutinefunction(
                    agent_callable
                ) or inspect.iscoroutinefunction(
                    getattr(agent_callable, "__call__", None)
                ):
                    self.logger.debug(
                        f"Agent '{agent_id_to_invoke}' is async. Awaiting call."
                    )
                    # The agent_callable from RegistryAgentProvider for MCP tools now expects (stage_dict, full_context)
                    # For other async agents, they might just take inputs or (inputs, full_context).
                    # We need to be careful. The original `invoke_agent_async` took (inputs, full_context).
                    # Let's assume for now that async callables obtained via .get() will conform to a signature
                    # that accepts the resolved_agent_inputs. If they need full_context, that's an extension.
                    # The MCP one from RegistryAgentProvider was changed to accept (stage_dict, full_context), where stage_dict has {"inputs": ...}
                    # This is slightly mismatched with the previous direct invoke_agent_async.
                    # For now, let's try passing resolved_agent_inputs directly, assuming it matches agent's expectation.
                    # If we use the MCP agent from RegistryAgentProvider, it expects a dict like {"inputs": resolved_agent_inputs}.
                    # This requires a slight wrapper if agent_callable is the _async_invoke from RegistryAgentProvider.
                    if (
                        hasattr(agent_callable, "__name__")
                        and agent_callable.__name__ == "_async_invoke"
                        and isinstance(self.agent_provider, RegistryAgentProvider)
                    ):
                        # This is the specific MCP tool wrapper from RegistryAgentProvider
                        stage_dict_for_mcp = {"inputs": resolved_inputs}
                        stage_result_payload = await agent_callable(
                            stage_dict_for_mcp, full_context=current_context
                        )
                    else:
                        # Generic async callable, assume it takes resolved inputs directly
                        # Or, if it has a specific signature it adheres to (e.g. Pydantic model input)
                        stage_result_payload = await agent_callable(resolved_inputs)
                elif callable(agent_callable):
                    self.logger.debug(
                        f"Agent '{agent_id_to_invoke}' is synchronous. Running in thread."
                    )
                    stage_result_payload = await asyncio.to_thread(
                        agent_callable, resolved_inputs
                    )
                else:
                    # This case should ideally not be reached if agent_provider.get always returns a callable or raises
                    self.logger.error(
                        f"Agent '{agent_id_to_invoke}' (type: {type(agent_callable)}) is not callable. This is unexpected."
                    )
                    raise TypeError(f"Agent '{agent_id_to_invoke}' is not callable.")

                self._emit_metric(
                    event_type=MetricEventType.AGENT_INVOCATION_END,
                    flow_id=flow_id,
                    run_id=run_id,
                    stage_id=current_stage_name,
                    master_stage_id=current_stage_name,
                    agent_id=agent_id_to_invoke,
                    data={
                        "status": "SUCCESS",
                        "output_type": type(stage_result_payload).__name__,
                    },
                )
                agent_invocation_succeeded = True
                self.logger.debug(
                    f"Agent '{agent_id_to_invoke}' for stage '{current_stage_name}' completed. Result type: {type(stage_result_payload).__name__}"
                )

                if isinstance(stage_result_payload, AgentErrorDetails):
                    self.logger.warning(
                        f"Agent '{agent_id_to_invoke}' for stage '{current_stage_name}' returned an error object: {stage_result_payload.message}"
                    )
                    agent_invocation_succeeded = False
                    agent_error_details = stage_result_payload

            except Exception as agent_exc:
                tb_str = traceback.format_exc()
                self.logger.error(
                    f"Exception during agent '{stage_spec.agent_id}' invocation for stage '{current_stage_name}': {agent_exc}\n{tb_str}"
                )
                agent_invocation_succeeded = False
                agent_error_details = AgentErrorDetails(
                    error_type=type(
                        agent_exc
                    ).__name__,  # Use the actual exception type name
                    message=f"Agent invocation failed: {str(agent_exc)}",
                    agent_id=stage_spec.agent_id,  # Add agent_id
                    stage_id=current_stage_name,  # Add stage_id
                    traceback=tb_str,  # Pass traceback string to 'traceback' field
                    details={
                        "exception_details": str(agent_exc)
                    },  # Optional: pass basic exception string as dict
                )
                self._emit_metric(
                    event_type=MetricEventType.AGENT_INVOCATION_END,
                    flow_id=flow_id,
                    run_id=run_id,
                    stage_id=current_stage_name,
                    master_stage_id=current_stage_name,
                    agent_id=stage_spec.agent_id,
                    data={
                        "status": "FAILURE",
                        "error": str(agent_exc),
                        "exception_type": type(agent_exc).__name__,
                    },
                )

            stage_duration = (
                datetime.now(timezone.utc) - stage_start_time
            ).total_seconds()
            next_stage_for_loop: Optional[str] = None
            stage_final_status: StageStatus = (
                StageStatus.FAILURE
            )  # Default, will be updated
            stage_reason: Optional[str] = None

            if not agent_invocation_succeeded or agent_error_details:
                stage_final_status = (
                    StageStatus.FAILURE
                )  # Explicitly set for this block
                stage_reason = f"Stage failed: {agent_error_details.message if agent_error_details else 'Unknown reason'}"
                current_context["_flow_error"] = {
                    "message": f"Stage '{current_stage_name}' failed.",
                    "details": (
                        agent_error_details.model_dump()
                        if agent_error_details
                        else "Unknown agent error"
                    ),
                }
                self.logger.error(
                    f"Stage '{current_stage_name}' failed in plan '{flow_id}', run '{run_id}'. Details: {agent_error_details}"
                )
                self._emit_metric(
                    event_type=MetricEventType.STAGE_END,
                    flow_id=flow_id,
                    run_id=run_id,
                    stage_id=current_stage_name,
                    master_stage_id=current_stage_name, # Use current_stage_name for master_stage_id
                    agent_id=stage_spec.agent_id,
                    data={
                        "status": StageStatus.FAILURE.value,
                        "duration_seconds": stage_duration,
                        "error_details": (
                            agent_error_details.model_dump_json()
                            if agent_error_details
                            else None
                        ),
                    },
                )

                # ---- BEGIN REVIEWER INVOCATION & HANDLING ----
                reviewer_output: Optional[MasterPlannerReviewerOutput] = None
                should_call_reviewer = True # Or determine based on stage_spec.on_failure policy

                if should_call_reviewer:
                    reviewer_output = await self._invoke_reviewer_and_get_suggestion(
                        run_id=run_id,
                        flow_id=flow_id,
                        current_stage_name=current_stage_name,
                        agent_error_details=agent_error_details,
                        current_context=current_context
                    )

                if reviewer_output:
                    # Handle reviewer suggestions (Start with a few)
                    if reviewer_output.suggestion_type == ReviewerActionType.RETRY_STAGE_AS_IS:
                        self.logger.info(f"[RunID: {run_id}] Reviewer suggested RETRY_STAGE_AS_IS for stage '{current_stage_name}'. Retrying.")
                        context["_last_master_stage_hop_info"] = {
                            "from": current_stage_name, "to": current_stage_name, # Retry
                            "reason": "Reviewer suggested RETRY_STAGE_AS_IS"
                        }
                        hops += 1 
                        if "_flow_error" in current_context: del current_context["_flow_error"]
                        if "_agent_error_details_for_stage_status" in current_context: del current_context["_agent_error_details_for_stage_status"]
                        continue # Retry the current stage

                    elif reviewer_output.suggestion_type == ReviewerActionType.RETRY_STAGE_WITH_MODIFIED_INPUT:
                        self.logger.info(f"[RunID: {run_id}] Reviewer suggested RETRY_STAGE_WITH_MODIFIED_INPUT for stage '{current_stage_name}'.")
                        
                        can_retry_autonomously = False
                        if reviewer_output.suggestion_details and \
                           reviewer_output.suggestion_details.get("target_stage_id") == current_stage_name and \
                           "new_inputs" in reviewer_output.suggestion_details and \
                           isinstance(reviewer_output.suggestion_details["new_inputs"], dict):
                            
                            new_inputs_from_reviewer = reviewer_output.suggestion_details["new_inputs"]
                            self.logger.info(f"[RunID: {run_id}] Attempting to apply new inputs for stage '{current_stage_name}': {new_inputs_from_reviewer}")
                            
                            if current_stage_name in self.current_plan.stages:
                                self.current_plan.stages[current_stage_name].inputs = new_inputs_from_reviewer
                                stage_spec.inputs = new_inputs_from_reviewer # Update current stage_spec view too
                                
                                self.logger.info(f"[RunID: {run_id}] Inputs for stage '{current_stage_name}' updated in plan for this run. Retrying stage.")
                                context["_last_master_stage_hop_info"] = {
                                    "from": current_stage_name, "to": current_stage_name, # Retry
                                    "reason": "Reviewer suggested RETRY_STAGE_WITH_MODIFIED_INPUT (applied automatically)"
                                }
                                hops += 1
                                if "_flow_error" in current_context: del current_context["_flow_error"]
                                if "_agent_error_details_for_stage_status" in current_context: del current_context["_agent_error_details_for_stage_status"]
                                can_retry_autonomously = True
                                continue # Retry with new inputs
                            else:
                                self.logger.error(f"[RunID: {run_id}] Stage '{current_stage_name}' (target_stage_id from reviewer) not found in plan. Cannot apply new inputs. Escalating.")
                        else:
                            self.logger.warning(f"[RunID: {run_id}] Reviewer suggested RETRY_STAGE_WITH_MODIFIED_INPUT but 'new_inputs' were missing, malformed, or for the wrong stage. Details: {reviewer_output.suggestion_details}. Escalating for user to provide inputs.")

                        if not can_retry_autonomously:
                            # Pause for user to provide inputs
                            user_message = "Reviewer suggested retrying stage '{current_stage_name}' with modified inputs, but could not automatically determine them."
                            reviewer_analysis = reviewer_output.suggestion_details.get("modification_needed", "No specific analysis provided by reviewer.")
                            original_inputs = stage_spec.inputs if stage_spec else {}

                            current_context["_flow_error"] = { 
                                "message": user_message,
                                "stage": current_stage_name,
                                "reviewer_suggestion_type": reviewer_output.suggestion_type.value,
                                "reviewer_reasoning": reviewer_output.reasoning,
                                "reviewer_analysis": reviewer_analysis,
                                "original_inputs": original_inputs,
                                "original_agent_error": agent_error_details.model_dump() if agent_error_details else None
                            }
                            
                            paused_state_details = self._create_paused_state_details(
                                master_flow_id=flow_id,
                                paused_stage_id=current_stage_name,
                                execution_context=copy.deepcopy(current_context),
                                error_details_model=agent_error_details, 
                                status_reason=user_message,
                                clarification_request={
                                    "type": "INPUT_MODIFICATION_REQUIRED", 
                                    "message": user_message,
                                    "reviewer_analysis": reviewer_analysis,
                                    "original_stage_inputs": original_inputs,
                                    "reviewer_output": reviewer_output.model_dump(),
                                    "action_required": f"Please provide the complete, corrected set of inputs for stage '{current_stage_name}' to retry.",
                                    "resume_hint": f"Use 'chungoid flow resume {run_id} --action retry_with_inputs --action-data \'{{\"target_stage_id\": \"{current_stage_name}\", \"inputs\": {{...}}}}\'"                                    
                                },
                                pause_status=FlowPauseStatus.USER_INPUT_REQUIRED 
                            )
                            await self.state_manager.save_paused_flow_state(run_id, paused_state_details)
                            self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, 
                                              data={"reason": "Awaiting user input for RETRY_STAGE_WITH_MODIFIED_INPUT", "stage_id": current_stage_name})
                            current_context["_autonomous_flow_paused_state_saved"] = True
                            return current_context # End execution for this run

                    elif reviewer_output.suggestion_type == ReviewerActionType.MODIFY_MASTER_PLAN:
                        self.logger.info(f"[RunID: {run_id}] Reviewer suggested MODIFY_MASTER_PLAN for stage '{current_stage_name}'. Escalating for user review.")
                        # Simple Escalation: Pause the flow and provide reviewer's suggestion to the user.
                        user_message = reviewer_output.suggestion_details.get("message_to_user", 
                                                                              f"Reviewer suggested modifying the master plan due to issues at stage '{current_stage_name}'. Reasoning: {reviewer_output.reasoning}")
                        if "suggested_plan_change" in reviewer_output.suggestion_details:
                            user_message += f" Suggested change: {reviewer_output.suggestion_details['suggested_plan_change']}"

                        current_context["_flow_error"] = { # Keep this for context, but primary info via PausedRunDetails
                            "message": f"Flow paused for user review: Reviewer suggested MODIFY_MASTER_PLAN for stage '{current_stage_name}'.",
                            "stage": current_stage_name,
                            "reviewer_suggestion_type": reviewer_output.suggestion_type.value,
                            "reviewer_reasoning": reviewer_output.reasoning,
                            "reviewer_message_to_user": user_message,
                            "original_agent_error": agent_error_details.model_dump() if agent_error_details else None
                        }
                        
                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=flow_id,
                            paused_stage_id=current_stage_name,
                            execution_context=copy.deepcopy(current_context),
                            error_details_model=agent_error_details, # Original agent error
                            status_reason=f"Reviewer suggested MODIFY_MASTER_PLAN. {user_message}",
                            clarification_request={"type": "PLAN_MODIFICATION_REVIEW", "message": user_message, "reviewer_output": reviewer_output.model_dump()},
                            pause_status=FlowPauseStatus.REVIEWER_ACTION_REQUIRED 
                        )
                        await self.state_manager.save_paused_flow_state(run_id, paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, 
                                          data={"reason": "Reviewer suggested MODIFY_MASTER_PLAN", "stage_id": current_stage_name})
                        current_context["_autonomous_flow_paused_state_saved"] = True # Signal to outer loop
                        return current_context # End execution for this run

                    elif reviewer_output.suggestion_type == ReviewerActionType.ESCALATE_TO_USER:
                        self.logger.info(f"[RunID: {run_id}] Reviewer suggested ESCALATE_TO_USER for stage '{current_stage_name}'. Pausing for user intervention.")
                        user_message = reviewer_output.suggestion_details.get("message_to_user",
                                                                              f"Reviewer escalated stage '{current_stage_name}' for user intervention. Reasoning: {reviewer_output.reasoning}")

                        current_context["_flow_error"] = { # Keep this for context
                            "message": f"Flow paused: Reviewer escalated stage '{current_stage_name}'.",
                            "stage": current_stage_name,
                            "reviewer_suggestion_type": reviewer_output.suggestion_type.value,
                            "reviewer_reasoning": reviewer_output.reasoning,
                            "reviewer_message_to_user": user_message,
                            "original_agent_error": agent_error_details.model_dump() if agent_error_details else None
                        }

                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=flow_id,
                            paused_stage_id=current_stage_name,
                            execution_context=copy.deepcopy(current_context),
                            error_details_model=agent_error_details,
                            status_reason=f"Reviewer escalated for user intervention. {user_message}",
                            clarification_request={"type": "USER_ESCALATION", "message": user_message, "reviewer_output": reviewer_output.model_dump()},
                            pause_status=FlowPauseStatus.USER_INTERVENTION_REQUIRED
                        )
                        await self.state_manager.save_paused_flow_state(run_id, paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, 
                                          data={"reason": "Reviewer suggested ESCALATE_TO_USER", "stage_id": current_stage_name})
                        current_context["_autonomous_flow_paused_state_saved"] = True # Signal to outer loop
                        return current_context # End execution

                    elif reviewer_output.suggestion_type == ReviewerActionType.PROCEED_AS_IS:
                        self.logger.info(f"[RunID: {run_id}] Reviewer suggested PROCEED_AS_IS for stage '{current_stage_name}'. Determining next stage.")
                        stage_final_status = StageStatus.COMPLETED_WITH_WARNINGS
                        stage_reason = f"Stage originally failed but reviewer suggested PROCEED_AS_IS. Failure: {agent_error_details.message if agent_error_details else 'Unknown'}"
                        self._update_run_status_with_stage_result(
                            stage_name=current_stage_name, stage_number=stage_spec.number,
                            status=stage_final_status, stage_output_payload=stage_result_payload,
                            error_details=agent_error_details, reason=stage_reason
                        )
                        
                        next_stage_name_after_proceed: Optional[str] = None
                        if stage_spec.condition:
                            if self._parse_condition(stage_spec.condition, current_context):
                                next_stage_name_after_proceed = stage_spec.next_stage_true
                            else:
                                next_stage_name_after_proceed = stage_spec.next_stage_false
                        else:
                            next_stage_name_after_proceed = stage_spec.next_stage

                        if next_stage_name_after_proceed:
                            current_master_stage_key = next_stage_name_after_proceed
                            context["_last_master_stage_hop_info"] = {
                                "from": current_stage_name, "to": current_master_stage_key,
                                "reason": "Reviewer suggested PROCEED_AS_IS"
                            }
                            hops += 1
                            if "_flow_error" in current_context: del current_context["_flow_error"]
                            continue
                        else:
                            self.logger.info(f"[RunID: {run_id}] Reviewer suggested PROCEED_AS_IS, but no next stage for '{current_stage_name}'. Flow ends.")
                            self.logger.warning(
                                f"Item at index {idx} in artifact paths list for stage '{stage_name}' "
                                f"is not a string (type: {type(p_item).__name__}), skipping."
                            )
                    elif reviewer_output.suggestion_type == ReviewerActionType.NO_ACTION_SUGGESTED:
                        self.logger.warning(f"[RunID: {run_id}] Reviewer provided NO_ACTION_SUGGESTED for stage '{current_stage_name}'. Escalating for user review.")
                        user_message = (f"Reviewer was invoked for stage '{current_stage_name}' but provided no specific recovery action. "
                                        f"Original error: {agent_error_details.message if agent_error_details else 'N/A'}. "
                                        f"Reviewer reasoning: {reviewer_output.reasoning}")
                        
                        current_context["_flow_error"] = { # Keep this for context
                            "message": f"Flow paused: Reviewer provided NO_ACTION_SUGGESTED for stage '{current_stage_name}'.",
                            "stage": current_stage_name,
                            "reviewer_suggestion_type": reviewer_output.suggestion_type.value,
                            "reviewer_reasoning": reviewer_output.reasoning,
                            "reviewer_message_to_user": user_message,
                            "original_agent_error": agent_error_details.model_dump() if agent_error_details else None
                        }

                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=flow_id,
                            paused_stage_id=current_stage_name,
                            execution_context=copy.deepcopy(current_context),
                            error_details_model=agent_error_details, 
                            status_reason=f"Reviewer provided NO_ACTION_SUGGESTED. {user_message}",
                            clarification_request={"type": "NO_REVIEWER_SUGGESTION", "message": user_message, "reviewer_output": reviewer_output.model_dump()},
                            pause_status=FlowPauseStatus.REVIEWER_ACTION_REQUIRED 
                        )
                        await self.state_manager.save_paused_flow_state(run_id, paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, 
                                          data={"reason": "Reviewer provided NO_ACTION_SUGGESTED", "stage_id": current_stage_name})
                        current_context["_autonomous_flow_paused_state_saved"] = True # Signal to outer loop
                        return current_context # End execution

                    # Fallback if no specific reviewer suggestion was handled above and we didn't retry
                    self.logger.warning(f"[RunID: {run_id}] Reviewer suggestion type '{reviewer_output.suggestion_type.value}' not fully handled or led to fallback. Defaulting to PAUSE_FOR_INTERVENTION for stage '{current_stage_name}'.")
                    current_context["_flow_error"] = { # Keep this for context
                        "message": f"Flow paused for intervention: Unhandled or fallback reviewer suggestion for stage '{current_stage_name}'.",
                        "stage": current_stage_name,
                        "reviewer_suggestion_type": reviewer_output.suggestion_type.value,
                        "reviewer_reasoning": reviewer_output.reasoning,
                        "original_agent_error": agent_error_details.model_dump() if agent_error_details else None
                    }
                    paused_state_details = self._create_paused_state_details(
                        master_flow_id=flow_id,
                        paused_stage_id=current_stage_name,
                        execution_context=copy.deepcopy(current_context),
                        error_details_model=agent_error_details,
                        status_reason=f"Unhandled reviewer suggestion ({reviewer_output.suggestion_type.value}) or fallback. Reasoning: {reviewer_output.reasoning}",
                        clarification_request={"type": "UNHANDLED_REVIEWER_SUGGESTION", "message": f"Reviewer suggested {reviewer_output.suggestion_type.value}, which led to a fallback pause.", "reviewer_output": reviewer_output.model_dump()},
                        pause_status=FlowPauseStatus.PAUSED_FOR_INTERVENTION # Generic pause
                    )
                    await self.state_manager.save_paused_flow_state(run_id, paused_state_details)
                    self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, data={"reason": "Unhandled reviewer suggestion", "stage_id": current_stage_name})
                    current_context["_autonomous_flow_paused_state_saved"] = True # Signal to outer loop
                    return current_context # End execution

                else: # No reviewer_output (e.g., reviewer not configured or failed)
                    # Original error handling: Use on_failure policy or default pause
                    effective_on_failure = stage_spec.on_failure
                    if not effective_on_failure: # Default behavior if on_failure is not specified
                        self.logger.warning(f"[RunID: {self._current_run_id}] Stage '{current_stage_name}' failed (agent resolution) and no on_failure policy defined. Defaulting to PAUSE_FOR_INTERVENTION.")
                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=self.current_plan.id,
                            paused_stage_id=current_stage_name,
                            execution_context=copy.deepcopy(context),
                            error_details_model=stage_execution_error,
                            status_reason=f"Agent resolution failed (on_failure policy): {stage_execution_error.message}",
                            clarification_request=None,
                            pause_status=StageStatus.PAUSED_FOR_INTERVENTION
                        )
                        await self.state_manager.save_paused_flow_state(self._current_run_id, paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=self.current_plan.id, run_id=self._current_run_id, data={"reason": "Agent resolution failure, on_failure policy"})
                        return context # End execution

                    if effective_on_failure.action == "FAIL_MASTER_FLOW":
                        self.logger.error(f"[RunID: {self._current_run_id}] Stage '{current_stage_name}' failed (agent resolution). Master flow configured to FAIL_MASTER_FLOW. Message: {effective_on_failure.log_message}")
                        current_master_stage_key = "_END_FAILURE_" # Signal loop to terminate
                        # Final context will be returned by the loop
                    elif effective_on_failure.action == "GOTO_MASTER_STAGE":
                        next_master_stage_key = effective_on_failure.target_master_stage_key
                        self.logger.info(f"[RunID: {self._current_run_id}] Stage '{current_stage_name}' failed (agent resolution). Transitioning to on_failure stage: '{next_master_stage_key}'.")
                        if next_master_stage_key not in self.current_plan.stages and next_master_stage_key not in ["_END_SUCCESS_", "_END_FAILURE_"]:
                            self.logger.error(f"[RunID: {self._current_run_id}] on_failure target_master_stage_key '{next_master_stage_key}' not found in plan. Terminating.")
                            current_master_stage_key = "_END_FAILURE_"
                        else:
                            current_master_stage_key = next_master_stage_key
                    elif effective_on_failure.action == "PAUSE_FOR_INTERVENTION":
                        self.logger.info(f"[RunID: {self._current_run_id}] Stage '{current_stage_name}' failed (agent resolution). Pausing for human intervention as per on_failure policy.")
                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=self.current_plan.id,
                            paused_stage_id=current_stage_name,
                            execution_context=copy.deepcopy(context),
                            error_details_model=stage_execution_error,
                            status_reason=f"Agent resolution failed (on_failure policy): {stage_execution_error.message}",
                            clarification_request=None,
                            pause_status=StageStatus.PAUSED_FOR_INTERVENTION
                        )
                        await self.state_manager.save_paused_flow_state(self._current_run_id, paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=self.current_plan.id, run_id=self._current_run_id, data={"reason": "Agent resolution failure, on_failure policy"})
                        return context # End execution
                    else: # Should not happen with validated MasterStageFailurePolicy
                        self.logger.error(f"[RunID: {self._current_run_id}] Unknown on_failure action: {effective_on_failure.action}. Terminating.")
                        current_master_stage_key = "_END_FAILURE_"
                    
                    # If we didn't return (pause) or set to _END_FAILURE_, we continue the loop to the next stage (error or normal)
                    if current_master_stage_key != "_END_FAILURE_" and current_master_stage_key not in ["_END_SUCCESS_", "_END_FAILURE_"]:
                        context["_last_master_stage_hop_info"] = {
                            "from": current_stage_name,
                            "to": current_master_stage_key,
                            "reason": "Agent resolution failure, on_failure transition"
                        }
                        continue # To the next iteration of the while loop with the new current_master_stage_key
                    else: # Reached a terminal state due to error handling
                        break # Exit the while loop, to be handled by outer return logic

        final_reason = reason if reason else f"Stage {status.value}"

        # TODO: Reconcile error_details type: AgentErrorDetails vs Dict
        # StateManager.update_status expects Optional[AgentErrorDetails]
        actual_error_details_for_sm: Optional[AgentErrorDetails] = None
        if error_details:
            if isinstance(error_details, AgentErrorDetails):
                actual_error_details_for_sm = error_details
            elif isinstance(error_details, dict):
                # Attempt to reconstruct if it was a dict from model_dump()
                try:
                    actual_error_details_for_sm = AgentErrorDetails(**error_details)
                except Exception as e:
                    self.logger.error(
                        f"Could not reconstruct AgentErrorDetails from dict for stage '{stage_name}': {e}"
                    )
            else:
                self.logger.warning(
                    f"error_details for stage '{stage_name}' is of unexpected type: {type(error_details)}"
                )

        self.state_manager.update_status(
            run_id=current_run_id,
            stage=stage_number if stage_number is not None else -1.0,
            status=status.value,
            artifacts=extracted_artifacts or [],
            reason=final_reason,
            error_details=actual_error_details_for_sm,
        )

    async def run(
        self, plan: MasterExecutionPlan, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Runs the entire execution plan from its start_stage.

        Args:
            plan: The MasterExecutionPlan to execute.
            context: The initial context for the flow.

        Returns:
            The final context after execution, or a context containing error information.
        """
        self.current_plan = plan
        start_stage_name = plan.start_stage

        run_id_from_state_manager = self.state_manager.get_or_create_current_run_id()
        if run_id_from_state_manager is None:
            self.logger.error(
                "Failed to get or create a run_id from StateManager for a new run."
            )
            self._current_run_id = str(uuid.uuid4())
            self.logger.warning(
                f"Using fallback UUID for run_id: {self._current_run_id}"
            )
        else:
            self._current_run_id = str(run_id_from_state_manager)

        self.logger.info(
            f"Starting execution of plan '{self.current_plan.id}', run_id '{self._current_run_id}', from stage '{start_stage_name}'."
        )
        self.logger.critical(f"RUN_DEBUG: self.current_plan.id = {self.current_plan.id}")

        # Only try to access original_request and save if it's a MasterExecutionPlan
        if isinstance(self.current_plan, MasterExecutionPlan):
            current_plan_original_request = getattr(self.current_plan, 'original_request', None)
            if current_plan_original_request:
                self.logger.critical(f"RUN_DEBUG: MasterExecutionPlan has original_request: {current_plan_original_request}")
                try:
                    self.state_manager.save_master_execution_plan(self.current_plan)
                    self.logger.info(f"Saved MasterExecutionPlan ID {self.current_plan.id} to StateManager.")
                except Exception as e_save_plan:
                    self.logger.error(f"RUN_DEBUG: Failed to save master plan with original_request: {e_save_plan}", exc_info=True)
            else:
                self.logger.critical("RUN_DEBUG: MasterExecutionPlan has NO original_request attribute (or it's None).")
        else:
            self.logger.critical(f"RUN_DEBUG: self.current_plan (type: {type(self.current_plan)}) is not a MasterExecutionPlan, skipping original_request check.")

        # Initialize overall flow status metric event
        flow_start_time = datetime.now(timezone.utc)
        self._emit_metric(
            event_type=MetricEventType.FLOW_START,
            flow_id=self.current_plan.id,
            run_id=self._current_run_id,
            data={
                "plan_name": getattr(self.current_plan, 'name', None),
                "start_stage": start_stage_name,
                "initial_context_keys": list(context.keys()),
            },
        )

        # Initialize final_context_for_loop before the try block
        final_context_for_loop: Dict[str, Any] = {} 

        current_context_for_loop = copy.deepcopy(context)
        current_context_for_loop["run_id"] = self._current_run_id
        current_context_for_loop["flow_id"] = self.current_plan.id
        current_context_for_loop.setdefault("global_flow_state", {})[
            "start_time"] = flow_start_time.isoformat()

        # If the plan is a MasterExecutionPlan and has an original_request, add it to the context
        if isinstance(self.current_plan, MasterExecutionPlan):
            plan_original_request = getattr(self.current_plan, 'original_request', None)
            if plan_original_request:
                current_context_for_loop["original_request"] = plan_original_request
                self.logger.debug(f"Added original_request from MasterExecutionPlan to run context for {self._current_run_id}")

        try:
            final_context_for_loop = await self._execute_master_flow_loop(
                start_stage_name, current_context_for_loop
            )

            if "_flow_error" in final_context_for_loop:
                flow_final_status = StageStatus.FAILURE
                error_info_for_state = final_context_for_loop["_flow_error"]
                reason_for_state = f"Flow failed: {final_context_for_loop['_flow_error'].get('message', 'Unknown error')}"
                self.logger.error(
                    f"Plan '{self.current_plan.id}', run '{self._current_run_id}' finished with error: {reason_for_state}"
                )
            elif final_context_for_loop.get("_autonomous_flow_paused_state_saved"):
                flow_final_status = StageStatus.UNKNOWN
                reason_for_state = f"Flow paused. Run ID: {self._current_run_id}. See PausedRunDetails for specific pause status."
                self.logger.info(reason_for_state)
                # No call to state_manager.record_flow_end() here, as it's paused, not ended.
            else:
                flow_final_status = StageStatus.SUCCESS  # Corrected
                reason_for_state = "Flow completed successfully."
                self.logger.info(
                    f"Plan '{self.current_plan.id}', run '{self._current_run_id}' completed successfully."
                )
                status_message_for_run = (
                    f"Flow {flow_final_status.value}: {reason_for_state}"
                )
                ctx_to_save = {
                    **final_context_for_loop,
                    "_flow_end_reason": reason_for_state,
                }
                self.state_manager.update_status(
                    stage=-1.0,  # Signify run-level update
                    status=status_message_for_run,
                    artifacts=[],
                    reason=reason_for_state,
                    # final_context=ctx_to_save # Not a param of update_status
                )

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.exception(
                f"Unhandled exception during execution of plan '{self.current_plan.id}', run '{self._current_run_id}': {e}"
            )
            flow_final_status = StageStatus.FAILURE
            error_info_for_state = {
                "error_type": type(e).__name__,
                "message": str(e),
                "traceback": tb_str,
            }
            reason_for_state = f"Flow failed with unhandled exception: {str(e)}"
            final_context_for_loop["_flow_error"] = error_info_for_state
            try:
                status_message_for_run_ex = (
                    f"Flow {flow_final_status.value}: {reason_for_state}"
                )
                ctx_to_save_ex = {
                    **final_context_for_loop,
                    "_flow_end_reason": reason_for_state,
                    "_flow_end_error_info": error_info_for_state,
                }
                self.state_manager.update_status(
                    stage=-1.0,  # Signify run-level update
                    status=status_message_for_run_ex,
                    artifacts=[],
                    reason=reason_for_state,
                    # final_context=ctx_to_save_ex # Not a param of update_status
                )
            except Exception as sm_exc:
                self.logger.error(
                    f"Failed to record flow end state after unhandled exception: {sm_exc}"
                )
        # ... code ...

        finally:
            # ... code in finally: emit FLOW_END metric, cleanup self.current_plan, self._current_run_id ...
            self.logger.debug(
                f"Clearing current_plan and _current_run_id for orchestrator instance after run of plan '{self.current_plan.id}', run '{self._current_run_id}'."
            )
            self.current_plan = None
            self._current_run_id = None

        return final_context_for_loop

    async def resume_flow(
        self,
        run_id: str,
        action: str,  # e.g., "retry", "skip_stage", "force_branch", "abort"
        action_data: Optional[Dict[str, Any]] = None,
        # reviewer_decision: Optional[ReviewerDecision] = None # Deprecated for now
    ) -> Dict[str, Any]:
        """
        Resumes a previously paused flow run based on a specified action.

        Args:
            run_id: The ID of the paused flow run to resume.
            action: The action to take (e.g., "retry", "skip_stage").
            action_data: Additional data required for the action (e.g., new inputs for retry).

        Returns:
            The final context after resumption, or error information.
        """
        self.logger.info(
            f"Attempting to resume flow for run_id '{run_id}' with action '{action}'."
        )
        if action_data:
            self.logger.info(f"Action data: {action_data}")

        paused_details = self.state_manager.load_paused_flow_state(run_id)
        if not paused_details:
            self.logger.error(f"No paused run found for run_id '{run_id}'. Cannot resume.")
            return {"error": f"No paused run found for run_id '{run_id}'"}

        if not self.current_plan or self.current_plan.id != paused_details.flow_id:
            self.logger.warning(
                f"Orchestrator's current_plan (id: {self.current_plan.id if self.current_plan else 'None'}) "
                f"does not match paused_details.flow_id ({paused_details.flow_id}). "
                "This might lead to issues if not correctly handled by the calling context or test setup."
            )
            # In a real system, you'd load the plan:
            # self.current_plan = self.load_plan_by_id(paused_details.flow_id)
            # For now, we rely on test setup to provide basic_plan via orchestrator_for_resume.pipeline_def

        if not self.current_plan: # Still no plan after warning
             self.logger.error(f"Cannot resume run_id '{run_id}': MasterExecutionPlan with id '{paused_details.flow_id}' not loaded in orchestrator.")
             return {"error": f"MasterExecutionPlan '{paused_details.flow_id}' for run '{run_id}' not loaded."}


        self._current_run_id = run_id # Set run_id for this resumption

        context_to_resume_with = paused_details.context_snapshot
        if context_to_resume_with is None: # Should not happen with current PausedRunDetails
            self.logger.warning(f"Context snapshot for run_id '{run_id}' is None. Using empty context.")
            context_to_resume_with = {}
        
        # Ensure critical keys are present, similar to .run()
        context_to_resume_with.setdefault("outputs", {})
        context_to_resume_with["run_id"] = self._current_run_id
        context_to_resume_with["flow_id"] = self.current_plan.id
        if self.current_plan.original_request:
            context_to_resume_with["original_request"] = self.current_plan.original_request
        else:
            context_to_resume_with["original_request"] = None


        next_stage_to_execute: Optional[str] = None
        perform_execution_loop = True

        if action == "retry":
            next_stage_to_execute = paused_details.paused_at_stage_id
            self.logger.info(f"Resuming run_id '{run_id}' by retrying stage '{next_stage_to_execute}'.")

        elif action == "retry_with_inputs":
            if not action_data or "inputs" not in action_data or not isinstance(action_data["inputs"], dict):
                msg = "Action 'retry_with_inputs' requires a dictionary under the 'inputs' key in action_data."
                self.logger.error(msg)
                return {"error": msg}
            
            new_inputs = action_data["inputs"]
            self.logger.info(f"Resuming run_id '{run_id}' by retrying stage '{paused_details.paused_at_stage_id}' with new inputs: {new_inputs}.")
            context_to_resume_with.update(new_inputs)
            next_stage_to_execute = paused_details.paused_at_stage_id

        elif action == "skip_stage":
            current_stage_spec = self.current_plan.stages.get(paused_details.paused_at_stage_id)
            if not current_stage_spec:
                msg = f"Cannot skip stage: Paused stage ID '{paused_details.paused_at_stage_id}' not found in current plan '{self.current_plan.id}'."
                self.logger.error(msg)
                return {"error": msg}
            
            next_stage_to_execute = current_stage_spec.next_stage
            if not next_stage_to_execute or next_stage_to_execute == "FINAL_STEP":
                self.logger.info(f"Resuming run_id '{run_id}' by skipping last/final stage '{paused_details.paused_at_stage_id}'. Flow considered complete.")
                perform_execution_loop = False # No more stages to run
            else:
                self.logger.info(f"Resuming run_id '{run_id}' by skipping stage '{paused_details.paused_at_stage_id}' and proceeding to '{next_stage_to_execute}'.")
        
        elif action == "force_branch":
            if not action_data or "target_stage_id" not in action_data:
                msg = "Action 'force_branch' requires 'target_stage_id' in action_data."
                self.logger.error(msg)
                return {"error": msg}
            
            target_stage = action_data["target_stage_id"]
            if target_stage not in self.current_plan.stages:
                msg = f"Invalid target_stage_id '{target_stage}' for force_branch. Not found in plan '{self.current_plan.id}'."
                self.logger.error(msg)
                return {"error": msg}
            
            next_stage_to_execute = target_stage
            self.logger.info(f"Resuming run_id '{run_id}' by forcing branch to stage '{next_stage_to_execute}'.")

        elif action == "abort":
            self.logger.info(f"Aborting flow for run_id '{run_id}'.")
            context_to_resume_with["_flow_status"] = "ABORTED" # Mark as aborted
            context_to_resume_with["_flow_end_reason"] = f"Flow aborted by resume action for run_id {run_id}."
            perform_execution_loop = False
            # TODO: Should this also update state_manager with a final ABORTED status for the run?

        else:
            msg = f"Unknown resume action: '{action}' for run_id '{run_id}'."
            self.logger.error(msg)
            return {"error": msg}

        try:
            if not self.state_manager.delete_paused_flow_state(run_id):
                self.logger.warning(f"Failed to clear paused state for run_id '{run_id}' from StateManager. State might be inconsistent.")
        except Exception as e_clear:
            self.logger.error(f"Exception while clearing paused state for run_id '{run_id}': {e_clear}", exc_info=True)
            return {"error": f"Exception clearing paused state for run_id '{run_id}'. Cannot resume. Details: {e_clear}"}


        if perform_execution_loop and next_stage_to_execute:
            self.logger.info(f"Proceeding to execute loop from stage '{next_stage_to_execute}' for run_id '{run_id}'.")
            self._emit_metric(
                event_type=MetricEventType.FLOW_RESUME,
                flow_id=self.current_plan.id, 
                run_id=run_id,
                data={"resume_action": action, "resume_stage": next_stage_to_execute}
            )
            final_context = await self._execute_master_flow_loop(next_stage_to_execute, context_to_resume_with)
        elif perform_execution_loop and not next_stage_to_execute: 
            self.logger.info(f"No further stages to execute for run_id '{run_id}' after action '{action}'. Flow ends.")
            final_context = context_to_resume_with 
            final_context.setdefault("_flow_end_reason", f"Flow ended after resume action '{action}' with no further stages.")
        else: 
            self.logger.info(f"Execution loop not performed for run_id '{run_id}' due to action '{action}'.")
            final_context = context_to_resume_with
            # Ensure _flow_end_reason is set if aborting or skipping to a definitive end
            if action == "abort":
                final_context.setdefault("_flow_status", "ABORTED") # Already set earlier for abort
                final_context.setdefault("_flow_end_reason", f"Flow aborted by resume action for run_id {run_id}.")
            elif action == "skip_stage" and not next_stage_to_execute: # If skip_stage led to no next stage
                 final_context.setdefault("_flow_end_reason", f"Flow ended after resume action '{action}' with no further stages.") # Aligned wording

        self.current_plan = None
        self._current_run_id = None
        return final_context

# Placeholder for ReviewerDecision schema if it's decided to be defined here.
# class ReviewerDecision(BaseModel):
#     action: ReviewerActionType
#     details: Optional[Dict[str, Any]] = None
#     reasoning: Optional[str] = None
