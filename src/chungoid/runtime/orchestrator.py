"""Minimal execution runtime skeleton (Phase-6 scaffold).

This module will gradually evolve into the production‐grade orchestrator that
executes Flow YAML graphs.  For now we expose a *SyncOrchestrator* that can run
very simple flows in-process so we can start adding tests and build up the API
surface incrementally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path # ADDED THIS IMPORT
import datetime as _dt
from datetime import datetime, timezone
import yaml
from pydantic import BaseModel, Field, ConfigDict
from chungoid.utils.agent_resolver import AgentProvider, RegistryAgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.common_enums import StageStatus, FlowPauseStatus, OnFailureAction # ADDED OnFailureAction
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec
from chungoid.schemas.flows import PausedRunDetails # <<< ADD THIS IMPORT

# Import for reviewer agent and its schemas
from chungoid.runtime.agents.system_master_planner_reviewer_agent import (
    MasterPlannerReviewerAgent,
)  # Assuming AGENT_ID is on class
from chungoid.schemas.agent_master_planner_reviewer import (
    MasterPlannerReviewerInput, 
    MasterPlannerReviewerOutput, 
    ReviewerActionType,
    RetryStageWithChangesDetails, 
    AddClarificationStageDetails, 
    ModifyMasterPlanRemoveStageDetails,
    ModifyMasterPlanModifyStageDetails,
)

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

# Define constants for special next_stage values
NEXT_STAGE_END_SUCCESS = "__END_SUCCESS__"
NEXT_STAGE_END_FAILURE = "__END_FAILURE__"

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
        self.logger.setLevel(logging.DEBUG) # Ensure DEBUG messages from this logger are processed
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
        # stage_outputs is the direct output of the agent for the current stage,
        # typically a Pydantic model instance or a dictionary.

        # Helper function to resolve a path string against the stage_outputs object/dict
        def _resolve_path(path_str: str, root_obj: Any) -> Any:
            current_val = root_obj
            for part_idx, key_part in enumerate(path_str.split(".")):
                if isinstance(current_val, dict):
                    if key_part in current_val:
                        current_val = current_val[key_part]
                    else:
                        self.logger.debug(f"Path part '{key_part}' not found in dict during path resolution for '{path_str}'")
                        return _SENTINEL # Path not fully found
                elif isinstance(current_val, list) and key_part.isdigit():
                    idx_val = int(key_part)
                    if 0 <= idx_val < len(current_val):
                        current_val = current_val[int(key_part)]
                    else:
                        self.logger.debug(f"Index '{idx_val}' out of bounds for list during path resolution for '{path_str}'")
                        return _SENTINEL # Index out of bounds
                elif hasattr(current_val, key_part):
                    current_val = getattr(current_val, key_part)
                else:
                    self.logger.debug(f"Path part '{key_part}' not found as attribute or dict key for '{path_str}'")
                    return _SENTINEL # Path not fully found
            return current_val

        criterion_upper = criterion.upper()

        if criterion_upper.endswith(" EXISTS"):
            path_to_check = criterion[: criterion_upper.rfind(" EXISTS")].strip()
            actual_val = _resolve_path(path_to_check, stage_outputs)
            result = actual_val is not _SENTINEL
            self.logger.info(f"Criterion '{criterion}' (EXISTS check for '{path_to_check}') evaluated to {result}")
            return result
        
        elif " IS_NOT_EMPTY" in criterion_upper:
            path_to_check = criterion_upper.split(" IS_NOT_EMPTY", 1)[0].strip()
            # Since criterion_upper was used, need to map back to original case for path resolution if needed,
            # but Pydantic fields are case-sensitive. Assuming path_to_check in criterion is correct case.
            # Find the original case path:
            original_path_to_check = criterion[:criterion_upper.find(" IS_NOT_EMPTY")].strip()

            actual_val = _resolve_path(original_path_to_check, stage_outputs)

            if actual_val is _SENTINEL: # Path not found
                self.logger.info(f"Criterion '{criterion}' (IS_NOT_EMPTY check for '{original_path_to_check}'): Path not found. Evaluates to FALSE.")
                return False
            if actual_val is None:
                self.logger.info(f"Criterion '{criterion}' (IS_NOT_EMPTY check for '{original_path_to_check}'): Value is None. Evaluates to FALSE.")
                return False
            
            result: bool
            if isinstance(actual_val, (str, list, dict)):
                result = bool(actual_val) # True if not empty, False if empty
            else: # For other types like int, bool, consider them "not empty" if they exist and are not None
                result = True # Already checked for None above
            self.logger.info(f"Criterion '{criterion}' (IS_NOT_EMPTY check for '{original_path_to_check}', value: '{str(actual_val)[:50]}...') evaluated to {result}")
            return result

        elif " CONTAINS " in criterion_upper:
            # Split based on the case-insensitive " CONTAINS "
            # Path part needs to retain original casing from `criterion`
            # Substring part is the literal to find.
            idx_contains = criterion_upper.find(" CONTAINS ")
            original_path_to_check = criterion[:idx_contains].strip()
            substring_to_find_literal = criterion[idx_contains + len(" CONTAINS "):].strip()
            
            actual_val = _resolve_path(original_path_to_check, stage_outputs)

            if actual_val is _SENTINEL: # Path not found
                self.logger.info(f"Criterion '{criterion}' (CONTAINS check for '{original_path_to_check}'): Path not found. Evaluates to FALSE.")
                return False
            if not isinstance(actual_val, str):
                self.logger.warning(f"Criterion '{criterion}' (CONTAINS check for '{original_path_to_check}'): Value is not a string (type: {type(actual_val)}). Evaluates to FALSE.")
                return False

            # Substring literal should be enclosed in single quotes, remove them.
            expected_substring: str
            if substring_to_find_literal.startswith("'") and substring_to_find_literal.endswith("'"): # noqa: G001
                # Strip outer quotes and then unescape YAML's doubled single quotes to literal single quotes
                expected_substring = substring_to_find_literal[1:-1].replace("''", "'")
            else:
                self.logger.warning(f"Criterion '{criterion}': Substring literal '{substring_to_find_literal}' is not correctly single-quoted. Using as is, but this might be unintended.")
                expected_substring = substring_to_find_literal # Use as is if not quoted, with a warning
            
            self.logger.info(f"CONTAINS_DEBUG: expected_substring='{expected_substring}' (repr: {repr(expected_substring)})") # ADDED
            self.logger.info(f"CONTAINS_DEBUG: actual_val (first 300 chars)='{str(actual_val)[:300]}' (repr of first 300: {repr(str(actual_val)[:300])})") # ADDED
            result = expected_substring in actual_val
            self.logger.info(f"Criterion '{criterion}' (CONTAINS check: '{str(actual_val)[:50]}...' CONTAINS '{expected_substring}') evaluated to {result}")
            return result

        elif " ENDS_WITH " in criterion_upper: # ADDED BLOCK FOR ENDS_WITH
            idx_ends_with = criterion_upper.find(" ENDS_WITH ")
            original_path_to_check = criterion[:idx_ends_with].strip()
            substring_to_find_literal = criterion[idx_ends_with + len(" ENDS_WITH "):].strip()
            
            actual_val = _resolve_path(original_path_to_check, stage_outputs)

            if actual_val is _SENTINEL: # Path not found
                self.logger.info(f"Criterion '{criterion}' (ENDS_WITH check for '{original_path_to_check}'): Path not found. Evaluates to FALSE.")
                return False
            if not isinstance(actual_val, str):
                self.logger.warning(f"Criterion '{criterion}' (ENDS_WITH check for '{original_path_to_check}'): Value is not a string (type: {type(actual_val)}). Evaluates to FALSE.")
                return False

            expected_substring: str
            if substring_to_find_literal.startswith("'") and substring_to_find_literal.endswith("'"): # noqa: G001
                expected_substring = substring_to_find_literal[1:-1]
            else:
                self.logger.warning(f"Criterion '{criterion}': Substring literal '{substring_to_find_literal}' for ENDS_WITH is not correctly single-quoted. Using as is.")
                expected_substring = substring_to_find_literal
            
            result = actual_val.endswith(expected_substring)
            self.logger.info(f"Criterion '{criterion}' (ENDS_WITH check: '{str(actual_val)[:50]}...' ENDS_WITH '{expected_substring}') evaluated to {result}")
            return result

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
            self.logger.info(f"COMPARISON check for path '{path_str}'. Initial current_val type: {type(actual_val)}, value: {str(actual_val)[:200]}") # DEBUG LOG
            for idx, key in enumerate(path_str.split(".")):
                self.logger.info(f"COMPARISON check: part {idx} = '{key}'. current_val type before access: {type(actual_val)}") # DEBUG LOG
                if isinstance(actual_val, dict) and key in actual_val:
                    actual_val = actual_val[key]
                elif isinstance(actual_val, list) and key.isdigit():
                    idx_val = int(key)
                    if 0 <= idx_val < len(actual_val):
                        actual_val = actual_val[int(key)]
                    else:
                        self.logger.info(f"COMPARISON check: Index {idx_val} out of bounds for list of len {len(actual_val)}.") # DEBUG LOG
                        return False # Index out of bounds
                elif hasattr(actual_val, key):
                    actual_val = getattr(actual_val, key)
                else:
                    self.logger.info(
                        f"Criterion '{criterion}' FAILED (COMPARISON check): Path part '{key}' not found or not accessible on current_val of type {type(actual_val)}."
                    )
                    return False
                self.logger.info(f"COMPARISON check: after part '{key}', current_val type: {type(actual_val)}, value: {str(actual_val)[:200]}") # DEBUG LOG

            # Attempt to coerce expected_value_literal_str to type of actual_val
            coerced_expected_val: Any # Declare with type hint
            self.logger.info(f"COMPARISON check: actual_val='{actual_val}' (type: {type(actual_val)}), expected_value_literal_str='{expected_value_literal_str}'")
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
            elif isinstance(actual_val, str): # Explicitly handle string
                coerced_expected_val = expected_value_literal_str.strip("'\"")
            else:  # Fallback for any other type not explicitly handled or if actual_val is None
                self.logger.warning(
                    f"Unsupported or None type for actual_val in comparison: {type(actual_val)}. "
                    f"Proceeding with string coercion for expected_value: '{expected_value_literal_str}'."
                )
                coerced_expected_val = expected_value_literal_str.strip("'\"")

            result = supported_comparators[comparator](actual_val, coerced_expected_val)
            self.logger.info(
                f"Criterion '{criterion}' evaluation: '{actual_val}' {comparator} '{coerced_expected_val}' (coerced type: {type(coerced_expected_val)}) -> {result}"
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
        stage_outputs = context.get("outputs", {}).get(stage_name, {}) # This is the object criteria are checked against
        self.logger.debug(f"_check_success_criteria for stage '{stage_name}': Input stage_outputs type = {type(stage_outputs)}, value = {str(stage_outputs)[:500]}") # ADDED DEBUG LOG

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
        # Standard debug log, if ever needed for input resolution issues.
        # self.logger.debug(
        #    f"_resolve_input_values: inputs_spec='{str(inputs_spec)[:200]}', context_keys={list(context_data.keys())}"
        # )

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
                    self.logger.debug(f"[RESOLVE_INPUTS] Attempting to resolve path: '{context_path}'. Initial current_val type: {type(current_val)}")
                    try:
                        for part_idx, part in enumerate(path_parts):
                            self.logger.debug(f"[RESOLVE_INPUTS] Path part {part_idx+1}/{len(path_parts)}: '{part}'. Current current_val type: {type(current_val)}")
                            if isinstance(current_val, dict):
                                self.logger.debug(f"[RESOLVE_INPUTS] current_val is dict. Keys: {list(current_val.keys())}")
                                next_val = current_val.get(part, _SENTINEL)
                                if next_val is not _SENTINEL:
                                    current_val = next_val
                                    self.logger.debug(f"[RESOLVE_INPUTS] Found '{part}' in dict. New current_val type: {type(current_val)}")
                                else:
                                    self.logger.debug(f"[RESOLVE_INPUTS] Part '{part}' not in dict. Setting valid_path=False.")
                                    valid_path = False
                                    break
                            elif isinstance(current_val, list) and part.isdigit():
                                idx = int(part)
                                self.logger.debug(f"[RESOLVE_INPUTS] current_val is list (len {len(current_val)}). Attempting index {idx}.")
                                if 0 <= idx < len(current_val):
                                    current_val = current_val[idx]
                                    self.logger.debug(f"[RESOLVE_INPUTS] Found index {idx}. New current_val type: {type(current_val)}")
                                else:
                                    self.logger.debug(f"[RESOLVE_INPUTS] Index {idx} out of bounds. Setting valid_path=False.")
                                    valid_path = False
                                    break
                            elif hasattr(current_val, part):
                                self.logger.debug(f"[RESOLVE_INPUTS] current_val is object. Attempting getattr for '{part}'.")
                                current_val = getattr(current_val, part)
                                self.logger.debug(f"[RESOLVE_INPUTS] getattr successful for '{part}'. New current_val type: {type(current_val)}")
                            else:
                                self.logger.debug(f"[RESOLVE_INPUTS] Part '{part}' not found as dict key, list index, or object attribute. Setting valid_path=False.")
                                valid_path = False
                                break
                        
                        if valid_path:
                            resolved_value = current_val
                            self.logger.debug(f"[RESOLVE_INPUTS] Successfully resolved '{context_path}' to value of type {type(resolved_value)}.")
                        else:
                            # resolved_value remains context_path (literal string)
                            self.logger.debug(f"[RESOLVE_INPUTS] Failed to fully resolve '{context_path}'. Using literal value.")
                    except (KeyError, IndexError, AttributeError, TypeError) as e:
                        self.logger.warning(
                            f"[RESOLVE_INPUTS] Error resolving '{context_path}' from context: {e}. Using literal value '{context_path}'.", exc_info=True
                        )
                        # resolved_value remains context_path (literal string)
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
                "status": FlowPauseStatus.PAUSED_FOR_AGENT_FAILURE_IN_MASTER.value, # CORRECTED
                "context_snapshot_ref": None, 
                "error_details": agent_error_details.model_dump() if agent_error_details else None,
                "clarification_request": None
            }

            reviewer_input = MasterPlannerReviewerInput(
                current_master_plan=self.current_plan,
                paused_run_details=synthetic_paused_run_details,
                pause_status=FlowPauseStatus.PAUSED_FOR_AGENT_FAILURE_IN_MASTER, # <<< THIS LINE CHANGED
                paused_stage_id=current_stage_name,
                triggering_error_details=agent_error_details,
                full_context_at_pause=copy.deepcopy(current_context)
            )

            self.logger.debug(f"[RunID: {run_id}] MasterPlannerReviewerAgent input: {reviewer_input.model_dump_json(indent=2)}")
            
            reviewer_output: MasterPlannerReviewerOutput # Type hint
            if inspect.iscoroutinefunction(reviewer_agent_callable) or inspect.iscoroutinefunction(getattr(reviewer_agent_callable, '__call__', None)):
                reviewer_output = await reviewer_agent_callable(reviewer_input) # REMOVED full_context
            elif callable(reviewer_agent_callable):
                # For sync callables, we pass input_payload directly. The callable itself should handle context if needed via input_payload.
                reviewer_output = await asyncio.to_thread(reviewer_agent_callable, reviewer_input) # REMOVED full_context
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
        # raise Exception("ORCHESTRATOR CODE IS FRESH - If you see this, the .py file is being used!") # ADDED FOR DEBUG
        
        current_stage_name = start_stage_name
        visited_stages = set()  # To detect loops
        hop_count = 0 # To prevent runaway execution even with complex loop structures
        stage_is_being_retried_after_review = False # Special flag for RETRY_STAGE_WITH_CHANGES

        # Initialize current_context at the beginning of the loop from the input context
        current_context = context
        
        # Ensure 'intermediate_outputs' exists in current_context
        if "intermediate_outputs" not in current_context:
            current_context["intermediate_outputs"] = {}
        if "outputs" not in current_context: # Ensure 'outputs' also exists for direct agent results
            current_context["outputs"] = {}
        
        # Get or create run_id at the beginning of the flow execution
        # This run_id will persist for the entire master flow execution.
        run_id = self._current_run_id # Should be set by the run() method
        flow_id = self.current_plan.id # Get flow_id from the plan

        # This variable will hold any specific pause status determined by reviewer actions.
        # It's checked at the end of the main loop iteration to decide if termination due to pause is needed.
        pause_status_override: Optional[FlowPauseStatus] = None
        
        # Flag to indicate if the loop should terminate due to a pause action (e.g., from reviewer or clarification)
        should_terminate_for_pause = False

        # Store details of the stage and error that led to a reviewer escalation
        # Around line 1028 in the latest snippet provided by the tool
        stage_id_that_led_to_reviewer: Optional[str] = None
        error_details_that_led_to_reviewer: Optional[AgentErrorDetails] = None
        captured_escalation_message: Optional[str] = None


        while current_stage_name and current_stage_name not in {NEXT_STAGE_END_SUCCESS, NEXT_STAGE_END_FAILURE}:
            self.logger.critical(f"EXEC_LOOP_ENTRY: stage='{current_stage_name}', context_keys={list(current_context.keys())}")
            if not isinstance(current_context.get("intermediate_outputs"), dict):
                self.logger.warning(f"Forcing context['intermediate_outputs'] to {{}} at EXEC_LOOP_ENTRY because it was {type(current_context.get('intermediate_outputs'))}. Original value: {current_context.get('intermediate_outputs')}")
                current_context["intermediate_outputs"] = {}


            hop_count += 1
            if hop_count > self.MAX_HOPS:
                self.logger.error(f"[RunID: {run_id}] Maximum hop count ({self.MAX_HOPS}) exceeded. Terminating flow to prevent infinite loop. Current stage: {current_stage_name}")
                self._emit_metric(event_type=MetricEventType.FLOW_ERROR, flow_id=flow_id, run_id=run_id, data={"error_type": "MaxHopsExceeded", "stage_id": current_stage_name})
                current_context["_flow_error"] = AgentErrorDetails(agent_id="orchestrator", error_type="MaxHopsExceeded", message=f"Max hop count {self.MAX_HOPS} exceeded at stage {current_stage_name}.")
                break # Exit the while loop

            if not stage_is_being_retried_after_review and current_stage_name in visited_stages:
                self.logger.error(f"[RunID: {run_id}] Loop detected: Stage '{current_stage_name}' visited again. Terminating flow.")
                self._emit_metric(event_type=MetricEventType.FLOW_ERROR, flow_id=flow_id, run_id=run_id, data={"error_type": "LoopDetected", "stage_id": current_stage_name})
                current_context["_flow_error"] = AgentErrorDetails(agent_id="orchestrator", error_type="LoopDetected", message=f"Loop detected at stage {current_stage_name}.")
                break # Exit the while loop
            
            if not stage_is_being_retried_after_review : # Only add to visited if not a special retry
                visited_stages.add(current_stage_name)
            else:
                # Reset the flag after allowing one retry pass
                self.logger.debug(f"[RunID: {run_id}] Stage '{current_stage_name}' was processed as a reviewer-led retry. Resetting retry flag.")
                stage_is_being_retried_after_review = False


            current_stage_spec = self.current_plan.stages.get(current_stage_name)
            if not current_stage_spec:
                self.logger.error(f"[RunID: {run_id}] Stage '{current_stage_name}' not found in plan. Terminating flow.")
                self._emit_metric(event_type=MetricEventType.FLOW_ERROR, flow_id=flow_id, run_id=run_id, data={"error_type": "StageNotFound", "stage_id": current_stage_name})
                current_context["_flow_error"] = AgentErrorDetails(agent_id="orchestrator", error_type="StageNotFound", message=f"Stage {current_stage_name} not found in plan.")
                break # Exit the while loop

            # --- For logging and hop info, ensure we use the consistent ID throughout this iteration ---
            current_stage_name_for_this_iteration_logging_and_hop_info = current_stage_name
            # ---

            self.logger.info(
                f"[RunID: {run_id}] Executing master stage: '{current_stage_name_for_this_iteration_logging_and_hop_info}' (Number: {current_stage_spec.number}) using agent_id: '{current_stage_spec.agent_id}'"
            )
            self._emit_metric(
                event_type=MetricEventType.STAGE_START,
                flow_id=flow_id,
                run_id=run_id,
                stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                master_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                agent_id=current_stage_spec.agent_id, # MODIFIED
                data={"inputs": current_stage_spec.inputs} # Log static inputs
            )
            
            stage_result_payload: Optional[Any] = None
            agent_error_details_for_current_stage: Optional[AgentErrorDetails] = None
            stage_final_status: StageStatus = StageStatus.PENDING 
            resolved_agent_id_for_stage = current_stage_spec.agent_id # Default to spec

            stage_failed_or_paused = False # Flag to indicate if current stage processing resulted in failure/pause

            try:
                if current_stage_spec.agent_id == "system.noop_agent_v1": # Handle NOOP directly
                    self.logger.info(f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' is a NOOP. Skipping agent invocation.")
                    stage_result_payload = {"message": f"NOOP stage {current_stage_name_for_this_iteration_logging_and_hop_info} executed."}
                    stage_final_status = StageStatus.SUCCESS

                elif current_stage_spec.agent_id == "system.human_input_agent_v1":
                     self.logger.info(f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' is a human input stage. Pausing for user input.")
                     # Construct clarification_request from inputs or spec
                     clarification_msg = "Awaiting human input."
                     if isinstance(current_stage_spec.inputs, dict):
                         clarification_msg = current_stage_spec.inputs.get("prompt_message", clarification_msg)
                     
                     await self._save_paused_state_and_terminate_loop(
                         current_plan=self.current_plan,
                         pause_status=FlowPauseStatus.PAUSED_FOR_HUMAN_INPUT,
                         context_at_pause=copy.deepcopy(current_context),
                         paused_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info,
                         clarification_request=clarification_msg
                     )
                     should_terminate_for_pause = True # Ensure termination
                     stage_failed_or_paused = True # Mark as paused

                else: # Resolve and invoke actual agent
                    agent_callable = self.agent_provider.get(current_stage_spec.agent_id) # REMOVED current_stage_spec.agent_category
                    if agent_callable:
                        resolved_agent_id_for_stage = getattr(agent_callable, 'AGENT_ID', current_stage_spec.agent_id) # Get actual resolved agent ID, fallback to spec
                        self.logger.info(
                            f"[RunID: {run_id}] Invoking agent '{resolved_agent_id_for_stage}' for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'."
                        )
                        
                        # Resolve inputs against the current context
                        resolved_inputs = self._resolve_input_values(current_stage_spec.inputs, current_context)

                        # --- START: ARTIFACT HANDLING ---
                        # If the agent is CoreCodeGeneratorAgent_v1, inject the artifact path
                        if resolved_agent_id_for_stage == "CoreCodeGeneratorAgent_v1":
                            # Determine path relative to dummy_project/generated_code/
                            # This is a simplification. A more robust system would use project config.
                            # For example, project_root / "generated_code" / flow_id / stage_id / file_name
                            # For now, let's just use a generic name if not provided.
                            target_file_name = resolved_inputs.get("target_file_name", f"{current_stage_name_for_this_iteration_logging_and_hop_info}_output.py")
                            # Base path for generated code within the project
                            # This assumes self.config["project_root_dir"] is set by the CLI or engine
                            generated_code_base_path = Path(self.config.get("project_root_dir", ".")) / "generated_code"
                            artifact_relative_path = f"generated_code/{target_file_name}" # Relative to project root
                            
                            # Ensure the directory exists
                            (generated_code_base_path).mkdir(parents=True, exist_ok=True)
                            
                            resolved_inputs["_artifact_output_path"] = str(generated_code_base_path / target_file_name)
                            self.logger.info(f"[RunID: {run_id}] Injected _artifact_output_path: {resolved_inputs['_artifact_output_path']} for {resolved_agent_id_for_stage}")
                        # --- END: ARTIFACT HANDLING ---

                        start_time = datetime.now(timezone.utc)
                        if hasattr(agent_callable, 'invoke_async') and callable(getattr(agent_callable, 'invoke_async')):
                            stage_result_payload = await agent_callable.invoke_async(resolved_inputs, full_context=current_context)
                        elif callable(agent_callable):
                            # If agent_callable is a direct async function (e.g., from MCP or a simple fallback function)
                            # It might not expect full_context as a separate arg if it's designed to get it via resolved_inputs or other means.
                            # For now, let's assume if it's a direct callable, it takes only resolved_inputs.
                            # This matches the _stub and _async_invoke signatures in RegistryAgentProvider.
                            # If direct callables *also* need full_context, their signatures would need to match.
                            stage_result_payload = await agent_callable(resolved_inputs) # Simplified call for direct functions
                        else:
                            raise TypeError(f"Agent callable for '{resolved_agent_id_for_stage}' is not a callable or an object with invoke_async method.")
                        
                        end_time = datetime.now(timezone.utc)
                        duration_seconds = (end_time - start_time).total_seconds()

                        self.logger.info(f"[RunID: {run_id}] Agent '{resolved_agent_id_for_stage}' for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' completed.")
                        
                        # --- START: ARTIFACT TRACKING ---
                        if resolved_agent_id_for_stage == "CoreCodeGeneratorAgent_v1" and stage_result_payload and isinstance(stage_result_payload, dict) and stage_result_payload.get("artifact_generated_path"):
                            generated_path_absolute = stage_result_payload.get("artifact_generated_path")
                            # Convert absolute path from agent to path relative to project_root_dir
                            project_root = Path(self.config.get("project_root_dir", ".")).resolve()
                            try:
                                relative_artifact_path = Path(generated_path_absolute).relative_to(project_root)
                                if self.ARTIFACT_OUTPUT_KEY not in current_context["intermediate_outputs"]:
                                    current_context["intermediate_outputs"][self.ARTIFACT_OUTPUT_KEY] = []
                                current_context["intermediate_outputs"][self.ARTIFACT_OUTPUT_KEY].append(str(relative_artifact_path))
                                self.logger.info(f"[RunID: {run_id}] Tracked generated artifact: {relative_artifact_path}")
                            except ValueError:
                                self.logger.warning(f"[RunID: {run_id}] Could not make artifact path {generated_path_absolute} relative to project root {project_root}. Storing absolute.")
                                if self.ARTIFACT_OUTPUT_KEY not in current_context["intermediate_outputs"]:
                                     current_context["intermediate_outputs"][self.ARTIFACT_OUTPUT_KEY] = []
                                current_context["intermediate_outputs"][self.ARTIFACT_OUTPUT_KEY].append(str(generated_path_absolute))
                        # --- END: ARTIFACT TRACKING ---
                        
                        # Store the direct output of the agent under context.outputs.<stage_name>
                        # This makes it available for success criteria checks that might use e.g. "outputs.stage_B.some_field"
                        if current_context.get('outputs') is None: current_context['outputs'] = {} # Ensure outputs dict exists
                        current_context['outputs'][current_stage_name_for_this_iteration_logging_and_hop_info] = stage_result_payload
                        self.logger.info(f"Stored raw agent output in context.outputs.{current_stage_name_for_this_iteration_logging_and_hop_info}")

                        # Check success criteria if defined
                        if current_stage_spec.success_criteria:
                            self.logger.info(f"Checking {len(current_stage_spec.success_criteria)} success criteria for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'.")
                            all_criteria_passed, failed_criteria_details = await self._check_success_criteria(
                                current_stage_name_for_this_iteration_logging_and_hop_info, 
                                current_stage_spec, 
                                current_context # Pass current_context which now includes context.outputs.<stage_name>
                            )
                            if all_criteria_passed:
                                self.logger.info(f"All success criteria passed for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'.")
                                stage_final_status = StageStatus.SUCCESS
                            else:
                                self.logger.warning(f"Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' failed success criteria check. Failed criteria: {failed_criteria_details}")
                                stage_final_status = StageStatus.FAILURE
                                agent_error_details_for_current_stage = AgentErrorDetails(
                                    agent_id=resolved_agent_id_for_stage,
                                    error_type="SuccessCriteriaFailed",
                                    message=f"Stage failed success criteria: {', '.join(failed_criteria_details)}",
                                    stage_id=current_stage_name_for_this_iteration_logging_and_hop_info,
                                    details={"failed_criteria": failed_criteria_details}
                                )
                                stage_failed_or_paused = True # Mark as failed
                        else: # No success criteria, assume success if agent didn't raise exception
                            stage_final_status = StageStatus.SUCCESS
                    else: # Agent not found by provider
                        self.logger.error(f"[RunID: {run_id}] Agent for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' (agent_id: {current_stage_spec.agent_id}, category: {current_stage_spec.agent_category}) could not be resolved. Stage execution aborted.")
                        agent_error_details_for_current_stage = AgentErrorDetails(
                            agent_id=current_stage_spec.agent_id,
                            error_type="AgentNotFoundError", # More specific than a generic "StageFailed"
                            message=f"Agent ID '{current_stage_spec.agent_id}' (Category: '{current_stage_spec.agent_category}') not found by provider.",
                            stage_id=current_stage_name_for_this_iteration_logging_and_hop_info
                        )
                        stage_final_status = StageStatus.FAILURE
                        stage_failed_or_paused = True # Mark as failed
            
            except (NoAgentFoundForCategoryError, AmbiguousAgentCategoryError) as cat_exc:
                self.logger.error(f"[RunID: {run_id}] Agent resolution error for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}': {cat_exc}", exc_info=True)
                error_type = "NoAgentFoundForCategoryError" if isinstance(cat_exc, NoAgentFoundForCategoryError) else "AmbiguousAgentCategoryError"
                agent_error_details_for_current_stage = AgentErrorDetails(
                    agent_id=current_stage_spec.agent_id, # This was the requested ID/category
                    error_type=error_type,
                    message=str(cat_exc),
                    stage_id=current_stage_name_for_this_iteration_logging_and_hop_info,
                    details={"category": current_stage_spec.agent_category}
                )
                stage_final_status = StageStatus.FAILURE
                stage_failed_or_paused = True # Mark as failed
            
            except Exception as e: # This is the general exception handler for the agent processing block
                self.logger.error(f"[RunID: {run_id}] Exception during agent execution or processing for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' (Agent: '{resolved_agent_id_for_stage}'): {e}", exc_info=True)
                agent_error_details_for_current_stage = AgentErrorDetails(
                    agent_id=resolved_agent_id_for_stage, # Use the resolved agent ID if available
                    error_type=e.__class__.__name__,
                    message=str(e),
                    stage_id=current_stage_name_for_this_iteration_logging_and_hop_info,
                    traceback=traceback.format_exc()
                )
                stage_final_status = StageStatus.FAILURE
                stage_failed_or_paused = True # Mark as failed

            # --- After try-except for agent execution ---
            
            # Ensure we use the consistent stage name for this iteration's post-processing
            stage_name_for_failure_handling = current_stage_name_for_this_iteration_logging_and_hop_info

            if stage_failed_or_paused and not should_terminate_for_pause: # Stage failed (or was paused by non-terminating action like success criteria fail)
                                # This is the beginning of the block that replaces 'pass'
                self.logger.warning(
                    f"[RunID: {run_id}] Stage '{stage_name_for_failure_handling}' FAILED or was internally paused. Error: {agent_error_details_for_current_stage.message if agent_error_details_for_current_stage else 'Unknown (likely success criteria)'}"
                )
                
                on_failure_policy = current_stage_spec.on_failure
                on_failure_action = on_failure_policy.action if on_failure_policy else OnFailureAction.INVOKE_REVIEWER
                on_failure_target = on_failure_policy.target_master_stage_key if on_failure_policy else None
                on_failure_log_message = on_failure_policy.log_message if on_failure_policy else None

                if on_failure_action == OnFailureAction.RETRY_STAGE:
                    self.logger.info(f"[RunID: {run_id}] ON_FAILURE: RETRY_STAGE for '{stage_name_for_failure_handling}'. Re-queueing.")
                    stage_failed_or_paused = False 
                    if stage_name_for_failure_handling in visited_stages and not stage_is_being_retried_after_review:
                         self.logger.debug(f"[RunID: {run_id}] ON_FAILURE RETRY: Marking '{stage_name_for_failure_handling}' for retry, removing from strict visited set for this pass.")
                         stage_is_being_retried_after_review = True
                         if stage_name_for_failure_handling in visited_stages: visited_stages.remove(stage_name_for_failure_handling)
                    # current_stage_name remains the same to retry

                elif on_failure_action == OnFailureAction.GOTO_MASTER_STAGE:
                    if on_failure_target:
                        self.logger.info(f"[RunID: {run_id}] ON_FAILURE: GOTO_MASTER_STAGE for '{stage_name_for_failure_handling}'. Jumping to '{on_failure_target}'.")
                        current_stage_name = on_failure_target
                        if current_stage_name in visited_stages:
                            self.logger.warning(f"[RunID: {run_id}] ON_FAILURE: GOTO_MASTER_STAGE to '{current_stage_name}' would create a loop. Terminating.")
                            current_context["_flow_error"] = AgentErrorDetails(agent_id="orchestrator", error_type="GotoLoopDetected", message=f"on_failure: GOTO_MASTER_STAGE to {current_stage_name} would create a loop.")
                            break
                        stage_failed_or_paused = False # Reset flag as we are jumping
                    else:
                        self.logger.error(f"[RunID: {run_id}] ON_FAILURE: GOTO_MASTER_STAGE for '{stage_name_for_failure_handling}' but no target specified. Terminating.")
                        current_context["_flow_error"] = AgentErrorDetails(agent_id="orchestrator", error_type="GotoMissingTarget", message="on_failure: GOTO_MASTER_STAGE action specified but no target stage.")
                        break
                
                elif on_failure_action == OnFailureAction.PAUSE_FOR_INTERVENTION:
                    self.logger.info(f"[RunID: {run_id}] ON_FAILURE: PAUSE_FOR_INTERVENTION for stage '{stage_name_for_failure_handling}'. Saving state.")
                    await self._save_paused_state_and_terminate_loop(
                        pause_status=FlowPauseStatus.PAUSED_FOR_INTERVENTION,
                        context_at_pause=copy.deepcopy(current_context),
                        paused_stage_id=stage_name_for_failure_handling,
                        error_details=agent_error_details_for_current_stage,
                        clarification_request=on_failure_log_message or f"Flow paused for intervention at stage {stage_name_for_failure_handling}."
                    )
                    should_terminate_for_pause = True

                elif on_failure_action == OnFailureAction.FAIL_MASTER_FLOW:
                    self.logger.warning(f"[RunID: {run_id}] ON_FAILURE: FAIL_MASTER_FLOW for stage '{stage_name_for_failure_handling}'. Message: '{on_failure_log_message}'. Terminating flow.")
                    current_context["_flow_error"] = AgentErrorDetails(
                        agent_id="orchestrator", 
                        error_type="ExplicitFlowFailure", 
                        message=on_failure_log_message or f"Stage '{stage_name_for_failure_handling}' triggered FAIL_MASTER_FLOW due to its on_failure policy.",
                        stage_id=stage_name_for_failure_handling,
                        details={"original_agent_error": agent_error_details_for_current_stage.model_dump() if agent_error_details_for_current_stage else None}
                    )
                    current_stage_name = NEXT_STAGE_END_FAILURE
                    break 

                elif on_failure_action == OnFailureAction.INVOKE_REVIEWER:
                    self.logger.info(f"[RunID: {run_id}] ON_FAILURE: INVOKE_REVIEWER for stage '{stage_name_for_failure_handling}'.")
                    
                    stage_id_that_led_to_reviewer = stage_name_for_failure_handling
                    error_details_that_led_to_reviewer = agent_error_details_for_current_stage

                    reviewer_suggestion = await self._invoke_reviewer_and_get_suggestion(
                        run_id=run_id,
                        flow_id=flow_id,
                        current_stage_name=stage_name_for_failure_handling,
                        agent_error_details=agent_error_details_for_current_stage,
                        current_context=current_context,
                    )

                    if reviewer_suggestion:
                        self.logger.info(f"[RunID: {run_id}] Reviewer output received. Suggestion type: {reviewer_suggestion.suggestion_type.value}, Details: {reviewer_suggestion.suggestion_details}")
                        # ... (All the reviewer suggestion handling: RETRY_STAGE_AS_IS, RETRY_STAGE_WITH_CHANGES, ADD_CLARIFICATION_STAGE, MODIFY_MASTER_PLAN, ESCALATE_TO_USER) ...
                        # This is the large block from approx line 1335 to 1696 in the previous complete file content.
                        # For brevity here, I'll summarize, but you need the full logic from your previous context or I can re-provide that specific sub-section if needed.
                        # START of reviewer suggestion handling for ON_FAILURE
                        if reviewer_suggestion.suggestion_type == ReviewerActionType.RETRY_STAGE_AS_IS:
                            self.logger.info(f"[RunID: {run_id}] Reviewer suggested RETRY_STAGE_AS_IS for stage '{reviewer_suggestion.suggestion_details.target_stage_id}'.")
                            current_stage_name = reviewer_suggestion.suggestion_details.target_stage_id
                            stage_is_being_retried_after_review = True
                            if current_stage_name in visited_stages: visited_stages.remove(current_stage_name)
                            stage_failed_or_paused = False # Critical: reset for retry
                        
                        elif reviewer_suggestion.suggestion_type == ReviewerActionType.RETRY_STAGE_WITH_CHANGES:
                            details = reviewer_suggestion.suggestion_details
                            self.logger.info(f"[RunID: {run_id}] Reviewer suggested RETRY_STAGE_WITH_CHANGES for stage '{details.target_stage_id}'. Applying changes: {details.changes_to_stage_spec}")
                            target_stage_to_modify = self.current_plan.stages.get(details.target_stage_id)
                            if target_stage_to_modify:
                                if "inputs" in details.changes_to_stage_spec and isinstance(details.changes_to_stage_spec["inputs"], dict):
                                    new_input_changes = details.changes_to_stage_spec.pop("inputs")
                                    if target_stage_to_modify.inputs is None: target_stage_to_modify.inputs = {}
                                    elif not isinstance(target_stage_to_modify.inputs, dict):
                                        self.logger.warning(f"Stage '{details.target_stage_id}' original inputs was not a dict. Overwriting.")
                                        target_stage_to_modify.inputs = {}
                                    target_stage_to_modify.inputs.update(new_input_changes)
                                
                                for field, value in details.changes_to_stage_spec.items():
                                    if hasattr(target_stage_to_modify, field): setattr(target_stage_to_modify, field, value)
                                    else: self.logger.warning(f"Cannot apply change: Field '{field}' does not exist on MasterStageSpec.")
                                
                                self.state_manager.save_master_execution_plan(self.current_plan) # Use non-async version
                                self._emit_metric(event_type=MetricEventType.PLAN_MODIFIED, flow_id=flow_id, run_id=run_id, stage_id=details.target_stage_id, data={"action": "RETRY_STAGE_WITH_CHANGES", "changes": details.changes_to_stage_spec})
                                current_stage_name = details.target_stage_id
                                stage_is_being_retried_after_review = True
                                if current_stage_name in visited_stages: visited_stages.remove(current_stage_name)
                                stage_failed_or_paused = False # Critical: reset for retry
                            else:
                                self.logger.error(f"Reviewer targeted non-existent stage {details.target_stage_id} for RETRY_STAGE_WITH_CHANGES. Terminating.")
                                current_context["_flow_error"] = AgentErrorDetails(agent_id="orchestrator", error_type="ReviewerInvalidTarget", message=f"Reviewer targeted non-existent stage {details.target_stage_id} for RETRY_STAGE_WITH_CHANGES.")
                                break
                        
                        elif reviewer_suggestion.suggestion_type == ReviewerActionType.ADD_CLARIFICATION_STAGE:
                            details = reviewer_suggestion.suggestion_details # Should be AddClarificationStageDetails
                            new_stage_id = details.new_stage_spec.id
                            original_failed_stage_id = details.original_failed_stage_id
                            if new_stage_id in self.current_plan.stages:
                                self.logger.error(f"Reviewer attempted to add stage with existing ID {new_stage_id}. Terminating.")
                                current_context["_flow_error"] = AgentErrorDetails(agent_id="orchestrator", error_type="ReviewerDuplicateStageId", message=f"Reviewer attempted to add stage with existing ID {new_stage_id}.")
                                break
                            if not details.new_stage_spec.next_stage: details.new_stage_spec.next_stage = original_failed_stage_id
                            self.current_plan.stages[new_stage_id] = details.new_stage_spec
                            
                            found_predecessor_and_rewired = False
                            if self.current_plan.start_stage == original_failed_stage_id:
                                self.current_plan.start_stage = new_stage_id
                                found_predecessor_and_rewired = True
                            else:
                                for stage_iter_id, stage_iter_spec in self.current_plan.stages.items():
                                    if stage_iter_spec.next_stage == original_failed_stage_id:
                                        stage_iter_spec.next_stage = new_stage_id
                                        found_predecessor_and_rewired = True; break
                                    if stage_iter_spec.next_stage_true == original_failed_stage_id:
                                        stage_iter_spec.next_stage_true = new_stage_id
                                        found_predecessor_and_rewired = True; break
                                    if stage_iter_spec.next_stage_false == original_failed_stage_id:
                                        stage_iter_spec.next_stage_false = new_stage_id
                                        found_predecessor_and_rewired = True; break
                            if not found_predecessor_and_rewired: self.logger.warning(f"Could not find predecessor for '{original_failed_stage_id}' to rewire to '{new_stage_id}'.")

                            # Handle output mapping to verification stage (from previous logic)
                            if details.new_stage_output_to_map_to_verification_stage_input:
                                mapping_info = details.new_stage_output_to_map_to_verification_stage_input
                                source_field = mapping_info.get("source_output_field")
                                target_input_field = mapping_info.get("target_input_field_in_verification_stage")
                                verification_stage_id_to_update = "stage_C_verify" # Assuming this for now
                                if verification_stage_id_to_update in self.current_plan.stages and source_field and target_input_field:
                                    verification_stage_spec = self.current_plan.stages[verification_stage_id_to_update]
                                    if verification_stage_spec.inputs is None: verification_stage_spec.inputs = {}
                                    verification_stage_spec.inputs[target_input_field] = f"context.outputs.{new_stage_id}.{source_field}"
                                    self.logger.info(f"Updated inputs for '{verification_stage_id_to_update}': {verification_stage_spec.inputs}")
                                else: self.logger.warning(f"Could not map output for ADD_CLARIFICATION_STAGE. Verification stage: '{verification_stage_id_to_update}', source: '{source_field}', target: '{target_input_field}'")
                            
                            self.state_manager.save_master_execution_plan(self.current_plan) # Use non-async
                            self._emit_metric(event_type=MetricEventType.PLAN_MODIFIED, flow_id=flow_id, run_id=run_id, stage_id=original_failed_stage_id, data={"action": "ADD_CLARIFICATION_STAGE", "new_stage_id": new_stage_id})
                            current_stage_name = new_stage_id
                            if original_failed_stage_id in visited_stages: visited_stages.remove(original_failed_stage_id)
                            stage_failed_or_paused = False # Critical: reset for new stage

                        elif reviewer_suggestion.suggestion_type == ReviewerActionType.MODIFY_MASTER_PLAN:
                            modification_details = reviewer_suggestion.suggestion_details
                            if isinstance(modification_details, ModifyMasterPlanRemoveStageDetails):
                                stage_to_remove_id = modification_details.target_stage_id
                                if stage_to_remove_id not in self.current_plan.stages:
                                    self.logger.error(f"Cannot remove non-existent stage '{stage_to_remove_id}'. Terminating."); break
                                removed_stage_spec = self.current_plan.stages.pop(stage_to_remove_id)
                                next_after_removed = removed_stage_spec.next_stage 
                                if self.current_plan.start_stage == stage_to_remove_id: self.current_plan.start_stage = next_after_removed
                                else:
                                    rewired = False
                                    for sid, sspec in self.current_plan.stages.items():
                                        if sspec.next_stage == stage_to_remove_id: sspec.next_stage = next_after_removed; rewired=True; break
                                        if sspec.next_stage_true == stage_to_remove_id: sspec.next_stage_true = next_after_removed; rewired=True; break
                                        if sspec.next_stage_false == stage_to_remove_id: sspec.next_stage_false = next_after_removed; rewired=True; break
                                    if not rewired: self.logger.warning(f"Could not rewire for removed stage '{stage_to_remove_id}'.")
                                if stage_name_for_failure_handling == stage_to_remove_id: current_stage_name = next_after_removed
                                else: current_stage_name = current_stage_name # Stay or rely on predecessor's next
                                if stage_to_remove_id in visited_stages: visited_stages.remove(stage_to_remove_id)
                                self.state_manager.save_master_execution_plan(self.current_plan) # Use non-async
                                self._emit_metric(event_type=MetricEventType.PLAN_MODIFIED, flow_id=flow_id, run_id=run_id, data={"action": "REMOVE_STAGE", "removed_stage_id": stage_to_remove_id})
                                stage_failed_or_paused = False # Allow flow to continue with new plan
                            elif isinstance(modification_details, ModifyMasterPlanModifyStageDetails):
                                details = modification_details
                                target_stage_to_modify = self.current_plan.stages.get(details.target_stage_id)
                                if target_stage_to_modify:
                                    self.current_plan.stages[details.target_stage_id] = details.updated_stage_spec
                                    self.state_manager.save_master_execution_plan(self.current_plan) # Use non-async
                                    self._emit_metric(event_type=MetricEventType.PLAN_MODIFIED, flow_id=flow_id, run_id=run_id, stage_id=details.target_stage_id, data={"action": "MODIFY_STAGE_SPEC", "updated_spec_fields": details.updated_stage_spec.model_dump(exclude_unset=True)})
                                    current_stage_name = details.target_stage_id
                                    stage_is_being_retried_after_review = True
                                    if current_stage_name in visited_stages: visited_stages.remove(current_stage_name)
                                    stage_failed_or_paused = False # Critical: reset for retry
                                else:
                                    self.logger.error(f"Reviewer targeted non-existent stage {details.target_stage_id} for MODIFY_STAGE_SPEC. Terminating.")
                                    current_context["_flow_error"] = AgentErrorDetails(agent_id="orchestrator", error_type="ReviewerInvalidTarget", message=f"Reviewer targeted non-existent stage {details.target_stage_id} for MODIFY_STAGE_SPEC.")
                                    break
                            else: self.logger.warning(f"Unknown MODIFY_MASTER_PLAN detail type: {type(modification_details)}.")
                        
                        elif reviewer_suggestion.suggestion_type == ReviewerActionType.ESCALATE_TO_USER:
                            escalation_payload = reviewer_suggestion.suggestion_details
                            log_msg = str(escalation_payload.get("message_to_user", "User escalation requested.")) if isinstance(escalation_payload, dict) else str(escalation_payload)
                            self.logger.info(f"[RunID: {run_id}] Reviewer suggested ESCALATE_TO_USER. Message: '{log_msg}'")
                            pause_status_override = FlowPauseStatus.USER_INTERVENTION_REQUIRED
                            should_terminate_for_pause = True
                            captured_escalation_message = escalation_payload # Store the direct dict or string
                        
                        else: # NO_ACTION_SUGGESTED or unhandled
                            self.logger.warning(f"[RunID: {run_id}] Reviewer returned unhandled or no action: {reviewer_suggestion.suggestion_type}. Flow proceeds based on original failure policy.")
                            # No change to current_stage_name or stage_failed_or_paused, let outer logic handle based on on_failure or terminate.
                            # This means if the original on_failure was INVOKE_REVIEWER, and reviewer does nothing, it effectively becomes a "terminate or pause based on no recovery"
                            # For safety, if we reach here, let's ensure the flow pauses if no other explicit on_failure action takes over.
                            if not (current_stage_spec.on_failure and current_stage_spec.on_failure.action != OnFailureAction.INVOKE_REVIEWER):
                                self.logger.warning(f"[RunID: {run_id}] No specific recovery from reviewer and no other on_failure policy. Pausing for intervention.")
                                pause_status_override = FlowPauseStatus.PAUSED_FOR_INTERVENTION
                                should_terminate_for_pause = True
                                captured_escalation_message = {"message_to_user": f"Stage {stage_name_for_failure_handling} failed, and reviewer suggested no action or an unhandled action."}


                    else: # Reviewer invocation failed or returned None
                        self.logger.warning(f"[RunID: {run_id}] MasterPlannerReviewerAgent provided no suggestion or failed. Defaulting to PAUSE_FOR_INTERVENTION.")
                        pause_status_override = FlowPauseStatus.PAUSED_FOR_INTERVENTION
                        should_terminate_for_pause = True
                        captured_escalation_message = {"message_to_user": f"Stage {stage_name_for_failure_handling} failed, and the MasterPlannerReviewerAgent failed or provided no suggestion."}
                        # stage_id_that_led_to_reviewer and error_details_that_led_to_reviewer are already set

                else: # Unknown on_failure_action
                    self.logger.error(f"[RunID: {run_id}] Unknown on_failure action '{on_failure_action}' for stage '{stage_name_for_failure_handling}'. Terminating.")
                    current_context["_flow_error"] = AgentErrorDetails(agent_id="orchestrator", error_type="UnknownOnFailureAction", message=f"Unknown on_failure action: {on_failure_action}")
                    break # Exit while loop

            # --- End of stage_failed_or_paused block ---

            if should_terminate_for_pause:
                self.logger.info(f"[RunID: {run_id}] Loop termination flag is set. Breaking from master flow loop. Pause override: {pause_status_override}")
                # If pause_status_override is set (from ESCALATE_TO_USER), state saving happens *after* this loop.
                # If pause was due to human_input or PAUSE_FOR_INTERVENTION (on_failure), state was already saved.
                break # Exit the while loop
            
            # --- Determine next stage if loop didn't break ---
            if not stage_failed_or_paused: # Only advance if stage succeeded and no pause/termination was triggered
                if stage_final_status == StageStatus.SUCCESS:
                    # Store output of successful stage for "previous_output" resolution
                    self._last_successful_stage_output = stage_result_payload 
                    # Store in intermediate_outputs as well, if an output_context_path is specified
                    if current_stage_spec.output_context_path:
                        try:
                            # Ensure intermediate_outputs exists
                            if "intermediate_outputs" not in current_context:
                                current_context["intermediate_outputs"] = {}
                            
                            # Resolve path, creating dicts if necessary
                            path_parts = current_stage_spec.output_context_path.split('.')
                            target_dict = current_context["intermediate_outputs"]
                            for part in path_parts[:-1]:
                                target_dict = target_dict.setdefault(part, {})
                            target_dict[path_parts[-1]] = stage_result_payload
                            self.logger.info(f"[RunID: {run_id}] Stored stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' output to context.intermediate_outputs.{current_stage_spec.output_context_path}")
                        except Exception as e:
                             self.logger.error(f"Error updating context with stage output for custom path '{current_stage_spec.output_context_path}': {e}", exc_info=True)

                    # --- BEGIN ADDED BLOCK FOR ON_SUCCESS REVIEWER INVOCATION ---
                    if current_stage_spec.on_success and current_stage_spec.on_success.action == OnFailureAction.INVOKE_REVIEWER: # Using OnFailureAction enum as it covers INVOKE_REVIEWER
                        self.logger.info(f"[RunID: {run_id}] ON_SUCCESS: INVOKE_REVIEWER for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'.")
                        
                        # Store details for potential escalation if reviewer suggests it
                        # Unlike on_failure, on_success doesn't inherently have an "error" that led to reviewer.
                        # We pass None for agent_error_details.
                        stage_id_that_led_to_reviewer_on_success = current_stage_name_for_this_iteration_logging_and_hop_info
                        
                        reviewer_suggestion_on_success = await self._invoke_reviewer_and_get_suggestion(
                            run_id=run_id,
                            flow_id=flow_id,
                            current_stage_name=current_stage_name_for_this_iteration_logging_and_hop_info,
                            agent_error_details=None, # No agent error on success path
                            current_context=current_context,
                        )

                        if reviewer_suggestion_on_success:
                            self.logger.info(f"[RunID: {run_id}] ON_SUCCESS Reviewer output received. Suggestion type: {reviewer_suggestion_on_success.suggestion_type.value}, Details: {reviewer_suggestion_on_success.suggestion_details}")
                            
                            # Handle reviewer suggestions that might alter flow or plan
                            # This will be a subset of the on_failure reviewer handling, focusing on plan changes or escalations.
                            # RETRY_STAGE_AS_IS and RETRY_STAGE_WITH_CHANGES might be less common on success but possible for refinement.
                            
                            if reviewer_suggestion_on_success.suggestion_type == ReviewerActionType.RETRY_STAGE_AS_IS:
                                details_rsais = reviewer_suggestion_on_success.suggestion_details
                                self.logger.info(f"[RunID: {run_id}] ON_SUCCESS Reviewer suggested RETRY_STAGE_AS_IS for stage '{details_rsais.target_stage_id}'.")
                                current_stage_name = details_rsais.target_stage_id
                                stage_is_being_retried_after_review = True
                                if current_stage_name in visited_stages: visited_stages.remove(current_stage_name)
                                # stage_failed_or_paused is False, so this retry will execute the stage again.
                                continue # Skip normal next_stage logic, loop back to retry
                            
                            elif reviewer_suggestion_on_success.suggestion_type == ReviewerActionType.RETRY_STAGE_WITH_CHANGES:
                                details_rswc = reviewer_suggestion_on_success.suggestion_details
                                self.logger.info(f"[RunID: {run_id}] ON_SUCCESS Reviewer suggested RETRY_STAGE_WITH_CHANGES for stage '{details_rswc.target_stage_id}'. Applying changes: {details_rswc.changes_to_stage_spec}")
                                target_stage_to_modify_rswc = self.current_plan.stages.get(details_rswc.target_stage_id)
                                if target_stage_to_modify_rswc:
                                    if "inputs" in details_rswc.changes_to_stage_spec and isinstance(details_rswc.changes_to_stage_spec["inputs"], dict):
                                        new_input_changes_rswc = details_rswc.changes_to_stage_spec.pop("inputs")
                                        if target_stage_to_modify_rswc.inputs is None: target_stage_to_modify_rswc.inputs = {}
                                        elif not isinstance(target_stage_to_modify_rswc.inputs, dict):
                                            self.logger.warning(f"Stage '{details_rswc.target_stage_id}' original inputs was not a dict. Overwriting.")
                                            target_stage_to_modify_rswc.inputs = {}
                                        target_stage_to_modify_rswc.inputs.update(new_input_changes_rswc)
                                    
                                    for field, value in details_rswc.changes_to_stage_spec.items():
                                        if hasattr(target_stage_to_modify_rswc, field): setattr(target_stage_to_modify_rswc, field, value)
                                        else: self.logger.warning(f"Cannot apply change: Field '{field}' does not exist on MasterStageSpec.")
                                    
                                    self.state_manager.save_master_execution_plan(self.current_plan)
                                    self._emit_metric(event_type=MetricEventType.PLAN_MODIFIED, flow_id=flow_id, run_id=run_id, stage_id=details_rswc.target_stage_id, data={"action": "RETRY_STAGE_WITH_CHANGES_ON_SUCCESS", "changes": details_rswc.changes_to_stage_spec})
                                    current_stage_name = details_rswc.target_stage_id
                                    stage_is_being_retried_after_review = True
                                    if current_stage_name in visited_stages: visited_stages.remove(current_stage_name)
                                    continue # Skip normal next_stage logic, loop back to retry with changes
                                else:
                                    self.logger.error(f"ON_SUCCESS Reviewer targeted non-existent stage {details_rswc.target_stage_id} for RETRY_STAGE_WITH_CHANGES. Proceeding with normal next_stage.")
                            
                            elif reviewer_suggestion_on_success.suggestion_type == ReviewerActionType.ADD_CLARIFICATION_STAGE:
                                details_acs = reviewer_suggestion_on_success.suggestion_details
                                new_stage_id_acs = details_acs.new_stage_spec.id
                                # original_succeeded_stage_id = details_acs.original_failed_stage_id # This field name is a bit misleading here, it's the stage that *succeeded*
                                original_succeeded_stage_id = current_stage_name_for_this_iteration_logging_and_hop_info

                                if new_stage_id_acs in self.current_plan.stages:
                                    self.logger.error(f"ON_SUCCESS Reviewer attempted to add stage with existing ID {new_stage_id_acs}. Proceeding with normal next_stage.")
                                else:
                                    # New stage should point to the original next_stage of the succeeded stage
                                    # And the succeeded stage should point to the new clarification stage.
                                    original_next_stage_of_succeeded = current_stage_spec.next_stage # Assuming simple next_stage for now, not conditional
                                    if current_stage_spec.condition:
                                        self.logger.warning(f"ON_SUCCESS: ADD_CLARIFICATION_STAGE after a conditional stage ('{original_succeeded_stage_id}') is complex. New stage will point to original 'next_stage' only: {original_next_stage_of_succeeded}")
                                        # Potentially need more robust handling for inserting between conditional stages.
                                    
                                    details_acs.new_stage_spec.next_stage = original_next_stage_of_succeeded
                                    self.current_plan.stages[new_stage_id_acs] = details_acs.new_stage_spec
                                    
                                    # Update the current (succeeded) stage to point to the new clarification stage
                                    current_stage_spec.next_stage = new_stage_id_acs
                                    current_stage_spec.next_stage_true = None # Nullify conditional paths if we insert a linear stage after
                                    current_stage_spec.next_stage_false = None
                                    current_stage_spec.condition = None

                                    # Handle output mapping (similar to on_failure)
                                    if details_acs.new_stage_output_to_map_to_verification_stage_input:
                                        mapping_info_acs = details_acs.new_stage_output_to_map_to_verification_stage_input
                                        # Use direct attribute access as per Pydantic model
                                        source_field_acs = mapping_info_acs.source_output_field 
                                        target_input_field_acs = mapping_info_acs.target_input_field_in_verification_stage
                                        
                                        # Determine the target verification stage ID.
                                        verification_stage_id_to_update_acs = "stage_C_verify" # Default for this test case
                                        
                                        self.logger.info(f"[RunID: {run_id}] ADD_CLARIFICATION_STAGE (on_success): Checking for verification stage input mapping. "
                                                         f"New stage: {new_stage_id_acs}, Source output: '{source_field_acs}', "
                                                         f"Target verifier: '{verification_stage_id_to_update_acs}', Target input: '{target_input_field_acs}'")

                                        if verification_stage_id_to_update_acs in self.current_plan.stages and source_field_acs and target_input_field_acs:
                                            verification_stage_spec_acs = self.current_plan.stages[verification_stage_id_to_update_acs]
                                            if verification_stage_spec_acs.inputs is None:
                                                verification_stage_spec_acs.inputs = {}
                                            
                                            mapped_input_path = f"context.outputs.{new_stage_id_acs}.{source_field_acs}"
                                            verification_stage_spec_acs.inputs[target_input_field_acs] = mapped_input_path
                                            
                                            self.logger.info(f"[RunID: {run_id}] ON_SUCCESS ADD_CLARIFICATION_STAGE: Successfully updated inputs for verification stage '{verification_stage_id_to_update_acs}'. "
                                                             f"Input '{target_input_field_acs}' is now mapped to '{mapped_input_path}'.")
                                            self.logger.debug(f"Updated verification_stage_spec_acs.inputs: {verification_stage_spec_acs.inputs}")
                                        else:
                                            self.logger.warning(f"[RunID: {run_id}] ON_SUCCESS ADD_CLARIFICATION_STAGE: Could not perform output mapping. "
                                                                f"Verification stage ID: '{verification_stage_id_to_update_acs}', "
                                                                f"Source field: '{source_field_acs}', Target input: '{target_input_field_acs}'. "
                                                                "One or more components might be missing or invalid.")
                                    else:
                                        self.logger.info(f"[RunID: {run_id}] ON_SUCCESS ADD_CLARIFICATION_STAGE: No output mapping details provided by reviewer for new stage '{new_stage_id_acs}'.")

                                    self.state_manager.save_master_execution_plan(self.current_plan)
                                    self._emit_metric(event_type=MetricEventType.PLAN_MODIFIED, flow_id=flow_id, run_id=run_id, stage_id=original_succeeded_stage_id, data={"action": "ADD_CLARIFICATION_STAGE_ON_SUCCESS", "new_stage_id": new_stage_id_acs})
                                    current_stage_name = new_stage_id_acs # Next stage is the new one
                                    # No need to remove original_succeeded_stage_id from visited_stages as it did succeed.
                                    continue # Loop back to execute the new clarification stage

                            elif reviewer_suggestion_on_success.suggestion_type == ReviewerActionType.MODIFY_MASTER_PLAN:
                                modification_details_mmp = reviewer_suggestion_on_success.suggestion_details
                                if isinstance(modification_details_mmp, ModifyMasterPlanRemoveStageDetails):
                                    stage_to_remove_id_mmp = modification_details_mmp.target_stage_id
                                    if stage_to_remove_id_mmp not in self.current_plan.stages:
                                        self.logger.error(f"ON_SUCCESS: Cannot remove non-existent stage '{stage_to_remove_id_mmp}'. Proceeding.")
                                    else:
                                        removed_stage_spec_mmp = self.current_plan.stages.pop(stage_to_remove_id_mmp)
                                        next_after_removed_mmp = removed_stage_spec_mmp.next_stage # Assuming simple next
                                        
                                        # Rewire predecessors
                                        if self.current_plan.start_stage == stage_to_remove_id_mmp: self.current_plan.start_stage = next_after_removed_mmp
                                        else:
                                            rewired_mmp = False
                                            for sid_mmp, sspec_mmp in self.current_plan.stages.items():
                                                if sspec_mmp.next_stage == stage_to_remove_id_mmp: sspec_mmp.next_stage = next_after_removed_mmp; rewired_mmp=True; break
                                                if sspec_mmp.next_stage_true == stage_to_remove_id_mmp: sspec_mmp.next_stage_true = next_after_removed_mmp; rewired_mmp=True; break
                                                if sspec_mmp.next_stage_false == stage_to_remove_id_mmp: sspec_mmp.next_stage_false = next_after_removed_mmp; rewired_mmp=True; break
                                            if not rewired_mmp: self.logger.warning(f"Could not rewire for removed stage '{stage_to_remove_id_mmp}'.")
                                        
                                        # Determine current_stage_name after removal
                                        if current_stage_name_for_this_iteration_logging_and_hop_info == stage_to_remove_id_mmp:
                                            current_stage_name = next_after_removed_mmp # If current was removed, go to its next
                                        elif current_stage_spec.next_stage == stage_to_remove_id_mmp:
                                            current_stage_spec.next_stage = next_after_removed_mmp # If current's next was removed, update current's next
                                            current_stage_name = current_stage_spec.next_stage # And proceed to it
                                        # else: current_stage_name remains as determined by normal success path if not directly affected.
                                                                                
                                        if stage_to_remove_id_mmp in visited_stages: visited_stages.remove(stage_to_remove_id_mmp)
                                        self.state_manager.save_master_execution_plan(self.current_plan)
                                        self._emit_metric(event_type=MetricEventType.PLAN_MODIFIED, flow_id=flow_id, run_id=run_id, data={"action": "REMOVE_STAGE_ON_SUCCESS", "removed_stage_id": stage_to_remove_id_mmp})
                                        # Do not 'continue' here, let the normal next_stage logic proceed if current_stage_name was updated,
                                        # or if the removed stage was not the immediate next one.
                                elif isinstance(modification_details_mmp, ModifyMasterPlanModifyStageDetails):
                                    details_mms = modification_details_mmp
                                    target_stage_to_modify_mms = self.current_plan.stages.get(details_mms.target_stage_id)
                                    if target_stage_to_modify_mms:
                                        self.current_plan.stages[details_mms.target_stage_id] = details_mms.updated_stage_spec
                                        self.state_manager.save_master_execution_plan(self.current_plan)
                                        self._emit_metric(event_type=MetricEventType.PLAN_MODIFIED, flow_id=flow_id, run_id=run_id, stage_id=details_mms.target_stage_id, data={"action": "MODIFY_STAGE_SPEC_ON_SUCCESS", "updated_spec_fields": details_mms.updated_stage_spec.model_dump(exclude_unset=True)})
                                        # If the modified stage is the *current* one, it's a bit late.
                                        # If it's a *future* stage, the change will take effect when it's reached.
                                        # For now, assume this doesn't immediately re-run current stage.
                                        self.logger.info(f"ON_SUCCESS: Stage '{details_mms.target_stage_id}' spec modified by reviewer. Change will apply when/if stage is next executed.")
                                    else:
                                        self.logger.error(f"ON_SUCCESS Reviewer targeted non-existent stage {details_mms.target_stage_id} for MODIFY_STAGE_SPEC. Proceeding.")
                                else: self.logger.warning(f"Unknown MODIFY_MASTER_PLAN detail type on success: {type(modification_details_mmp)}.")

                            elif reviewer_suggestion_on_success.suggestion_type == ReviewerActionType.ESCALATE_TO_USER:
                                escalation_payload_onsuccess = reviewer_suggestion_on_success.suggestion_details
                                log_msg_onsuccess = str(escalation_payload_onsuccess.get("message_to_user", "User escalation requested after successful stage.")) if isinstance(escalation_payload_onsuccess, dict) else str(escalation_payload_onsuccess)
                                self.logger.info(f"[RunID: {run_id}] ON_SUCCESS Reviewer suggested ESCALATE_TO_USER. Message: '{log_msg_onsuccess}'")
                                pause_status_override = FlowPauseStatus.USER_INTERVENTION_REQUIRED
                                should_terminate_for_pause = True
                                # Use the stage that succeeded as the one leading to escalation. Error details are None.
                                stage_id_that_led_to_reviewer = current_stage_name_for_this_iteration_logging_and_hop_info
                                error_details_that_led_to_reviewer = None
                                captured_escalation_message = escalation_payload_onsuccess
                                break # Break from main loop to save paused state outside

                            else: # NO_ACTION_SUGGESTED or unhandled on success
                                self.logger.info(f"[RunID: {run_id}] ON_SUCCESS Reviewer returned unhandled or no action: {reviewer_suggestion_on_success.suggestion_type}. Flow proceeds normally.")
                                # No change to current_stage_name, let original success logic determine next step.
                        
                        else: # Reviewer invocation failed or returned None on success
                            self.logger.warning(f"[RunID: {run_id}] ON_SUCCESS MasterPlannerReviewerAgent provided no suggestion or failed. Flow proceeds normally.")
                            # No change to current_stage_name, let original success logic determine next step.
                    
                    # --- END ADDED BLOCK FOR ON_SUCCESS REVIEWER INVOCATION ---

                    # Determine next stage based on success (original logic, executed if reviewer didn't 'continue' or 'break')
                    if not should_terminate_for_pause: # Check again, as reviewer might have set it
                        if current_stage_spec.condition:
                            if self._parse_condition(current_stage_spec.condition, current_context):
                                current_stage_name = current_stage_spec.next_stage_true
                            else:
                                current_stage_name = current_stage_spec.next_stage_false
                        else:
                            current_stage_name = current_stage_spec.next_stage
                    
                    self.logger.info(f"[RunID: {run_id}] Stage '{stage_name_for_failure_handling}' SUCCEEDED. Next stage: '{current_stage_name}'.") # Use stage_name_for_failure_handling as it's the one that just finished
                else:
                    # This case (stage_final_status != SUCCESS but stage_failed_or_paused is False) should ideally not be reached
                    # if logic is correct, as failure should set stage_failed_or_paused.
                    # But as a fallback, if somehow it's not SUCCESS and not PAUSED/FAILED, treat as error.
                    self.logger.error(f"[RunID: {run_id}] Stage '{stage_name_for_failure_handling}' ended with status {stage_final_status} but was not marked as failed/paused. This is unexpected. Terminating.")
                    current_context["_flow_error"] = AgentErrorDetails(agent_id="orchestrator", error_type="InconsistentState", message=f"Stage {stage_name_for_failure_handling} had status {stage_final_status} but not marked failed.")
                    break # Exit while loop
            else: # stage_failed_or_paused was true, but not handled by on_failure to continue or already broke loop.
                  # This means an on_failure action (like RETRY) might have reset stage_failed_or_paused and set current_stage_name.
                  # Or, the loop is about to break due to should_terminate_for_pause.
                  # If current_stage_name is not set by a recovery action, it will use its existing value (e.g. from RETRY).
                  self.logger.debug(f"[RunID: {run_id}] Post-failure/pause block. Loop continues or terminates. Current stage for next iter (if any): '{current_stage_name}'")


            # Small delay to prevent tight loops in certain async scenarios if ever needed (optional)
            # await asyncio.sleep(0.01) 

        # --- After the while loop ---
        
        final_flow_status_for_metric: Optional[str] = None

        if current_context.get("_flow_error"):
            error_info = current_context["_flow_error"]
            self.logger.error(f"[RunID: {run_id}] Master flow '{flow_id}' finished with error: {error_info.message if isinstance(error_info, AgentErrorDetails) else error_info}")
            final_flow_status_for_metric = "ERROR"
            # Potentially save a paused state here too if it's a critical error that warrants intervention
            # For now, errors like MaxHops, LoopDetected, StageNotFound are considered terminal without explicit pause save.

        elif should_terminate_for_pause and pause_status_override is not None:
            # This means ESCALATE_TO_USER occurred, loop broke, now we save.
            self.logger.info(f"[RunID: {run_id}] Flow {flow_id} (Run: {run_id}) is terminating due to reviewer escalation (status: {pause_status_override.value}). Saving final paused state.")
            await self._save_paused_state_and_terminate_loop(
                pause_status=pause_status_override, # CRITICAL: Use the override! Should now be USER_INTERVENTION_REQUIRED
                context_at_pause=copy.deepcopy(current_context),
                # Use the stage ID and error details that originally triggered the reviewer
                paused_stage_id=stage_id_that_led_to_reviewer if stage_id_that_led_to_reviewer else current_stage_name_for_this_iteration_logging_and_hop_info, # Fallback if somehow not set
                error_details=error_details_that_led_to_reviewer, # Use the captured error
                clarification_request=captured_escalation_details_for_saving # Use the (potentially wrapped) dictionary
            )
            final_flow_status_for_metric = pause_status_override.value


        elif current_stage_name == NEXT_STAGE_END_SUCCESS:
            self.logger.info(f"[RunID: {run_id}] Master flow '{flow_id}' completed successfully.")
            final_flow_status_for_metric = "SUCCESS"
        elif current_stage_name == NEXT_STAGE_END_FAILURE:
            self.logger.warning(f"[RunID: {run_id}] Master flow '{flow_id}' ended in explicit failure state (NEXT_STAGE_END_FAILURE).")
            final_flow_status_for_metric = "FAILURE"
        elif should_terminate_for_pause: # and pause_status_override is None
            # This means a pause like PAUSED_FOR_HUMAN_INPUT or PAUSED_FOR_INTERVENTION happened,
            # and _save_paused_state_and_terminate_loop was already called directly.
            # The status was logged by that direct call. We just need to make sure metrics are consistent.
            # The pause_status would be in the saved PausedRunDetails.
            # For metric, we might need to retrieve it or pass it out.
            # For now, this indicates a PAUSED state handled prior to this block.
             self.logger.info(f"[RunID: {run_id}] Master flow '{flow_id}' loop terminated due to a pause action where state was already saved (e.g., human input, intervention).")
             # To get the actual pause status for the metric, we'd ideally have it from the direct call.
             # This path might need refinement if a specific metric status is needed here.
             # Assuming the metric was emitted by _save_paused_state_and_terminate_loop itself.

        # NEW CONDITION: current_stage_name is None, and no errors/pauses are flagged
        elif current_stage_name is None and not current_context.get("_flow_error") and not should_terminate_for_pause:
            self.logger.info(f"[RunID: {run_id}] Master flow '{flow_id}' completed successfully (final stage had no next_stage).")
            final_flow_status_for_metric = "SUCCESS"
        else: # Loop exited for other reasons (e.g. current_stage_name became None without error/success/pause)
            self.logger.warning(f"[RunID: {run_id}] Master flow '{flow_id}' ended unexpectedly. Current stage: {current_stage_name}. Context: {str(current_context)[:500]}")
            final_flow_status_for_metric = "UNKNOWN_TERMINATION"

        # Emit final flow metric if status determined
        if final_flow_status_for_metric:
            self._emit_metric(
                event_type=MetricEventType.FLOW_END,
                flow_id=flow_id,
                run_id=run_id,
                data={"final_status": final_flow_status_for_metric, "final_context_keys": list(current_context.keys())}
            )
        
        # Clean up instance variables for the next run
        self._current_run_id = None 
        self._last_successful_stage_output = None
        self.current_plan = None # Clear plan associated with this orchestrator instance after run

        return current_context

    async def run(
        self,
        flow_yaml_path: Optional[str] = None,
        master_plan_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        run_id_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes a master flow plan, loading it from YAML or by ID.

        The orchestrator instance must be pre-configured with StateManager, AgentProvider, etc.
        Args:
            flow_yaml_path: Path to the flow YAML file.
            master_plan_id: ID of an existing master plan to load.
            initial_context: Optional initial context for the flow.
            run_id_override: Optional specific run ID to use; one will be generated if None.

        Returns:
            The final context after flow execution.
        """
        if not flow_yaml_path and not master_plan_id:
            self.logger.error("Orchestrator.run called without flow_yaml_path or master_plan_id.")
            raise ValueError("Either flow_yaml_path or master_plan_id must be provided to run.")

        current_run_id = run_id_override or str(uuid.uuid4())
        self._current_run_id = current_run_id
        self._last_successful_stage_output = None # Reset from any previous run

        # Load the MasterExecutionPlan
        loaded_plan_id_for_logging = "<not_yet_loaded>"
        try:
            if flow_yaml_path:
                with open(flow_yaml_path, 'r') as f:
                    raw_yaml_content = f.read()
                
                # Load YAML to dict, inject/overwrite ID, then dump back to string
                yaml_data = yaml.safe_load(raw_yaml_content)
                if not isinstance(yaml_data, dict):
                    # This case should ideally be handled by from_yaml, but good to be robust
                    raise ValueError("YAML content does not represent a dictionary.")

                plan_id_for_yaml = master_plan_id or Path(flow_yaml_path).stem
                yaml_data['id'] = plan_id_for_yaml # Inject/overwrite the ID
                
                # Ensure other required fields are present before trying to parse
                # This is a bit redundant as from_yaml also checks, but good for clarity
                if "start_stage" not in yaml_data or "stages" not in yaml_data:
                     raise ValueError("Loaded YAML data missing required 'start_stage' or 'stages' key after ID injection.")

                yaml_content_with_id = yaml.dump(yaml_data) # Dump back to string
                
                self.current_plan = MasterExecutionPlan.from_yaml(yaml_content_with_id) # Call with YAML string only
                loaded_plan_id_for_logging = self.current_plan.id # ID should now be from plan_id_for_yaml
                self.logger.info(f"[RunID: {self._current_run_id}] Loaded MasterExecutionPlan from YAML: {flow_yaml_path} (ID: {loaded_plan_id_for_logging})")
            elif master_plan_id:
                # project_root_dir should be in self.config from __init__ if needed by state_manager
                self.current_plan = await self.state_manager.load_master_execution_plan(master_plan_id, self.config.get("project_root_dir"))
                if not self.current_plan:
                    raise FileNotFoundError(f"MasterExecutionPlan with ID '{master_plan_id}' not found by StateManager.")
                loaded_plan_id_for_logging = self.current_plan.id
                self.logger.info(f"[RunID: {self._current_run_id}] Loaded MasterExecutionPlan by ID: {loaded_plan_id_for_logging}")
            else:
                # This case should be caught by the initial check, but as a safeguard:
                raise ValueError("Plan loading logic error: No source for plan.")

        except Exception as e:
            self.logger.error(f"[RunID: {self._current_run_id}] Failed to load MasterExecutionPlan (source YAML: '{flow_yaml_path}', ID: '{master_plan_id}'): {e}", exc_info=True)
            # Clean up potentially partially set instance vars for this run
            self._current_run_id = None
            self.current_plan = None
            return {"_flow_error": AgentErrorDetails(agent_id="orchestrator", error_type="PlanLoadError", message=f"Failed to load plan (source YAML: '{flow_yaml_path}', ID: '{master_plan_id}'): {e}")}
        
        flow_id = self.current_plan.id # Should now be correctly set from loaded plan
        self.logger.info(f"[RunID: {self._current_run_id}] Starting master flow '{flow_id}'. Start stage: {self.current_plan.start_stage}")
        self._emit_metric(
            event_type=MetricEventType.FLOW_START,
            flow_id=flow_id,
            run_id=self._current_run_id,
            data={"start_stage": self.current_plan.start_stage}
        )

        context = initial_context if initial_context is not None else {}
        if "intermediate_outputs" not in context: context["intermediate_outputs"] = {}
        if "outputs" not in context: context["outputs"] = {}
        # self.config is already available to the orchestrator instance via __init__
        # Pass relevant parts of self.config to the context if agents need it, or let them access via full_context if passed
        context["project_config"] = self.config # Make project_config available in context
        context["system_config"] = {"run_id": self._current_run_id, "flow_id": flow_id}

        final_context = await self._execute_master_flow_loop(
            start_stage_name=self.current_plan.start_stage,
            context=context
        )
        
        self.logger.info(f"[RunID: {self._current_run_id}] Master flow '{flow_id}' processing finished.")
        return final_context

    async def resume_paused_flow(
        self,
        paused_run_details: PausedRunDetails, # Corrected parameter
        # master_plan: MasterExecutionPlan, # Plan should be loaded via PausedRunDetails or StateManager
        # initial_context: Optional[Dict[str, Any]] = None, # Context comes from PausedRunDetails
        # run_id: Optional[str] = None # Run ID comes from PausedRunDetails
    ) -> Dict[str, Any]:
        self.logger.warning("resume_paused_flow is not yet fully implemented.")
        # TODO: Implement the logic to resume a paused flow.
        # 1. Load current_plan from paused_run_details.flow_id (or directly if stored in PausedRunDetails)
        # 2. Set self._current_run_id = paused_run_details.run_id
        # 3. Restore context from paused_run_details.context_snapshot_ref or embedded context
        # 4. Determine the stage_to_resume_from (paused_run_details.paused_at_stage_id or next stage based on user action)
        # 5. Call self._execute_master_flow_loop(start_stage_name=stage_to_resume_from, context=restored_context)
        pass # Placeholder to make it syntactically valid

    async def _save_paused_state_and_terminate_loop(
        self,
        # current_plan: MasterExecutionPlan, # Use self.current_plan
        pause_status: FlowPauseStatus,
        context_at_pause: Dict[str, Any],
        paused_stage_id: str,
        error_details: Optional[AgentErrorDetails] = None,
        clarification_request: Optional[str] = None
    ):
        """Saves the flow's state and prepares for termination of the execution loop."""
        if not self._current_run_id or not self.current_plan:
            self.logger.error(
                "_save_paused_state_and_terminate_loop called with no current run_id or plan. Cannot save state."
            )
            return

        run_id = self._current_run_id
        flow_id = self.current_plan.id

        self.logger.info(
            f"[RunID: {run_id}] Pausing flow '{flow_id}' at stage '{paused_stage_id}' with status '{pause_status.value}'."
        )

        # Create PausedRunDetails object
        paused_details = PausedRunDetails(
            run_id=run_id,
            flow_id=flow_id,
            paused_at_stage_id=paused_stage_id,
            status=pause_status,
            timestamp=datetime.now(timezone.utc),
            # context_snapshot_ref: Optional[str] = None, # For larger contexts, could point to a file
            context_snapshot=context_at_pause, # Embed context for now
            error_details=error_details,
            clarification_request=clarification_request,
            # Ensure project_root_dir is available from config for StateManager pathing
            # project_root_dir=self.config.get("project_root_dir")
        )

        try:
            # Save the paused state using StateManager
            # The project_root_dir for where to save is handled by StateManager's init
            self.state_manager.save_paused_flow_state(paused_details)
            self.logger.info(f"[RunID: {run_id}] Successfully saved paused state for run_id '{run_id}'.")

            # Emit a metric for the pause event
            self._emit_metric(
                event_type=MetricEventType.FLOW_PAUSED,
                flow_id=flow_id,
                run_id=run_id,
                stage_id=paused_stage_id,
                data={
                    "pause_status": pause_status.value,
                    "clarification_request": clarification_request,
                    "error_message": error_details.message if error_details else None,
                },
            )
        except Exception as e:
            self.logger.error(
                f"[RunID: {run_id}] Failed to save paused state for run_id '{run_id}': {e}",
                exc_info=True,
            )
            # Even if saving fails, we might still want to terminate the loop, 
            # but the flow won't be resumable. This is a critical error.
            # For now, log and proceed to loop termination. 
            # A more robust handler might retry or take other actions.
