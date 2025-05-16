"""Minimal execution runtime skeleton (Phase-6 scaffold).

This module will gradually evolve into the production‐grade orchestrator that
executes Flow YAML graphs.  For now we expose a *SyncOrchestrator* that can run
very simple flows in-process so we can start adding tests and build up the API
surface incrementally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import datetime as _dt
from datetime import datetime, timezone
import yaml
from pydantic import BaseModel, Field, ConfigDict
from chungoid.utils.agent_resolver import AgentProvider, RegistryAgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.common_enums import StageStatus, FlowPauseStatus
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec
from chungoid.schemas.flows import PausedRunDetails # <<< ADD THIS IMPORT

# Import for reviewer agent and its schemas
from chungoid.runtime.agents.system_master_planner_reviewer_agent import (
    MasterPlannerReviewerAgent,
)  # Assuming AGENT_ID is on class
from chungoid.schemas.agent_master_planner_reviewer import MasterPlannerReviewerInput, MasterPlannerReviewerOutput, ReviewerActionType # ADDED MasterPlannerReviewerInput, Output, ActionType

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
                        current_val = current_val[idx_val]
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
                expected_substring = substring_to_find_literal[1:-1]
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
        # raise Exception("ORCHESTRATOR CODE IS FRESH - If you see this, the .py file is being used!") # ADDED FOR DEBUG
        self.logger.critical(
            f"EXEC_LOOP_ENTRY: stage='{start_stage_name}', context_keys={list(context.keys())}"
        )  # NEW CRITICAL LOG

        # === Aggressive fix: Ensure intermediate_outputs is a dict at the start of the loop ===
        if not isinstance(context.get("intermediate_outputs"), dict):
            self.logger.warning(
                f"Forcing context['intermediate_outputs'] to {{}} at EXEC_LOOP_ENTRY because it was {type(context.get("intermediate_outputs"))}. "
                f"Original value: {str(context.get('intermediate_outputs'))[:200]}"
            )
            context["intermediate_outputs"] = {}
        # === End aggressive fix ===

        if not self.current_plan:
            self.logger.error(
                "Current plan not set in _execute_master_flow_loop. This indicates a programming error."
            )
            return {"_flow_error": "Orchestrator internal error: current_plan not set."}

        # --- DEBUG LOGGING FOR STAGE_1_DEFINE_SPEC from self.current_plan ---
        if self.current_plan and "stage_1_define_spec" in self.current_plan.stages:
            stage_1_from_plan = self.current_plan.stages["stage_1_define_spec"]
            self.logger.critical(f"ORCH_DEBUG_PLAN_CHECK stage_1_define_spec.next_stage: {stage_1_from_plan.next_stage}")
        # --- END DEBUG LOGGING ---

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
        self.logger.debug(f"[RunID: {run_id}] Initialized visited_stages (should be empty): {visited_stages}") # ADDED
        max_hops_for_flow = self.MAX_HOPS

        while current_stage_name and current_stage_name not in ["FINAL_STEP", "_END_SUCCESS_", "_END_FAILURE_"]:
            # === START OF SECTION TO PRESERVE current_stage_name for this iteration's perspective ===
            current_stage_name_for_this_iteration_logging_and_hop_info = current_stage_name
            # === END OF SECTION ===

            self.logger.debug(f"[RunID: {run_id}] Loop top: current_stage_name='{current_stage_name_for_this_iteration_logging_and_hop_info}', visited_stages_before_check={visited_stages}") # MODIFIED to use preserved name
            if current_stage_name_for_this_iteration_logging_and_hop_info in visited_stages: # MODIFIED
                self.logger.warning(
                    f"Loop detected: Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' visited again in plan '{flow_id}', run '{run_id}'. Aborting." # MODIFIED
                )
                self._emit_metric(
                    event_type=MetricEventType.ORCHESTRATOR_INFO,
                    flow_id=flow_id,
                    run_id=run_id,
                    data={
                        "level": "WARNING",
                        "message": "Loop detected, aborting flow.",
                        "stage": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                        "visited_log": visited_stages,
                    },
                )
                current_context["_flow_error"] = {
                    "message": "Loop detected, execution aborted.",
                    "stage": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                }
                return current_context
            visited_stages.append(current_stage_name_for_this_iteration_logging_and_hop_info) # MODIFIED
            self.logger.debug(f"[RunID: {run_id}] Appended '{current_stage_name_for_this_iteration_logging_and_hop_info}' to visited_stages: {visited_stages}") # MODIFIED

            if len(visited_stages) > max_hops_for_flow:
                self.logger.warning(
                    f"Max hops ({max_hops_for_flow}) reached for plan '{flow_id}', run '{run_id}'. Aborting."
                )
                self._emit_metric(
                    event_type=MetricEventType.ORCHESTRATOR_INFO,
                    flow_id=flow_id,
                    run_id=run_id,
                    data={
                        "level": "WARNING",
                        "message": "Max hops reached, aborting flow.",
                        "stage": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                        "max_hops": max_hops_for_flow,
                    },
                )
                current_context["_flow_error"] = {
                    "message": "Max hops reached, execution aborted.",
                    "stage": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                }
                return current_context

            current_context["_current_stage_name"] = current_stage_name_for_this_iteration_logging_and_hop_info # MODIFIED

            # --- Stage Pre-computation & Validation ---
            # Ensure current_stage_spec is fetched *before* the check and logging
            current_stage_spec = self.current_plan.stages.get(current_stage_name_for_this_iteration_logging_and_hop_info) # MODIFIED

            # --- DEBUG LOGGING FOR STAGE_1_DEFINE_SPEC from current_stage_spec INSIDE LOOP ---
            if current_stage_name_for_this_iteration_logging_and_hop_info == "stage_1_define_spec" and current_stage_spec: # MODIFIED
                self.logger.critical(f"ORCH_DEBUG_LOOP_FETCH stage_1_define_spec.next_stage: {current_stage_spec.next_stage}")
            # --- END DEBUG LOGGING ---

            if not current_stage_spec:
                self.logger.error(
                    f"[RunID: {self._current_run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' not found in plan. Aborting flow." # MODIFIED
                )
                self._emit_metric( # Ensure metric is emitted for this critical failure
                    event_type=MetricEventType.ORCHESTRATOR_INFO,
                    flow_id=self.current_plan.id if self.current_plan else "UNKNOWN_PLAN",
                    run_id=self._current_run_id,
                    data={
                        "level": "ERROR",
                        "message": "Stage not found in plan, aborting.",
                        "stage_name": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                    },
                )
                context["_flow_error"] = { # Use the context passed into the loop
                    "message": f"Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' not found in plan.", # MODIFIED
                    "stage": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                }
                return context # Return the modified context
            
            # Initialize next_stage_name_after_execution for this iteration
            next_stage_name_after_execution: Optional[str] = None
            # Initialize agent_error_details for this iteration to avoid conflicts if set by agent_callable
            agent_error_details_for_current_stage: Optional[AgentErrorDetails] = None


            self.logger.info(
                f"[RunID: {self._current_run_id}] Executing master stage: '{current_stage_name_for_this_iteration_logging_and_hop_info}' (Number: {current_stage_spec.number}) using agent_id: '{current_stage_spec.agent_id if current_stage_spec.agent_id else 'N/A - Category based'}'" # MODIFIED
            )
            # Emit metric for stage start
            self._emit_metric(
                event_type=MetricEventType.STAGE_START,
                flow_id=self.current_plan.id,
                run_id=self._current_run_id,
                data={
                    "stage_name": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                    "stage_number": current_stage_spec.number,
                    "agent_id": current_stage_spec.agent_id, 
                    "agent_category": current_stage_spec.agent_category 
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
                    self.logger.error(f"[RunID: {self._current_run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' has neither agent_id nor agent_category specified.") # MODIFIED
                    stage_execution_error = AgentErrorDetails(
                        error_type="InvalidStageDefinitionError",
                        message=f"Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' must have agent_id or agent_category.", # MODIFIED
                        agent_id=None,
                        stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                        can_retry=False
                    )
            except (NoAgentFoundForCategoryError, AmbiguousAgentCategoryError) as cat_err:
                self.logger.error(f"[RunID: {self._current_run_id}] Failed to resolve agent for category '{current_stage_spec.agent_category}': {cat_err}")
                stage_execution_error = AgentErrorDetails(
                    error_type=type(cat_err).__name__,
                    message=str(cat_err),
                    agent_id=None,
                    stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                    details={"category": current_stage_spec.agent_category, "preferences": current_stage_spec.agent_selection_preferences},
                    can_retry=False
                )
            except KeyError as e: # From agent_provider.get if agent_id not found
                self.logger.error(f"[RunID: {self._current_run_id}] Agent '{current_stage_spec.agent_id}' not found: {e}")
                stage_execution_error = AgentErrorDetails(
                    error_type="AgentNotFoundError",
                    message=f"Agent ID '{current_stage_spec.agent_id}' not found in provider. Original error: {str(e)}",
                    agent_id=current_stage_spec.agent_id,
                    stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                    can_retry=False
                )
            except Exception as e: # Catch any other unexpected error during agent resolution
                self.logger.error(f"[RunID: {self._current_run_id}] Unexpected error resolving agent for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}': {e}") # MODIFIED
                tb_str = traceback.format_exc()
                stage_execution_error = AgentErrorDetails(
                    error_type=type(e).__name__,
                    message=f"An unexpected error occurred while resolving the agent: {str(e)}",
                    agent_id=current_stage_spec.agent_id if current_stage_spec.agent_id else None,
                    stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                    traceback=tb_str,
                    can_retry=False
                )

            # If agent_callable is still None due to an error caught above, or if stage_execution_error is set
            if stage_execution_error or not agent_callable:
                self.logger.error(f"[RunID: {self._current_run_id}] Agent for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' (agent_id: {current_stage_spec.agent_id}, category: {current_stage_spec.agent_category}) could not be resolved or invoked. Stage execution aborted.") # MODIFIED + added category
                # stage_final_status = StageStatus.FAILURE # This will be set inside the failure block
                # stage_reason = f"Stage failed: {stage_execution_error.message if stage_execution_error else 'Unknown reason for agent resolution failure'}" # MODIFIED
                current_context["_flow_error"] = {
                    "message": f"Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' failed due to agent resolution/invocation issues.", # MODIFIED
                    "details": (
                        stage_execution_error.model_dump()
                        if stage_execution_error
                        else "Unknown agent error during resolution"
                    ),
                }
                # Store the error for status update and reviewer
                agent_error_details_for_current_stage = stage_execution_error
                stage_final_status = StageStatus.FAILURE # Explicitly set failure status here
                
                # Fall through to the common failure handling logic below
                # No direct 'return' here, let shared failure logic handle reviewer/on_failure
                self.logger.error(
                    f"Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' in plan '{flow_id}', run '{run_id}' failed before agent invocation. Details: {agent_error_details_for_current_stage}" # MODIFIED
                )
                # No direct STAGE_END metric here, it will be emitted in the shared failure path or after success path

            else: # Agent resolution was successful, proceed to invocation
                resolved_inputs = self._resolve_input_values(
                    current_stage_spec.inputs or {}, current_context # Ensure inputs is a dict
                )
                self.logger.debug(
                    f"[RunID: {self._current_run_id}] Resolved inputs for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}': {resolved_inputs}" # MODIFIED
                )

                stage_start_time_agent_call = datetime.now(timezone.utc)
                stage_result_payload: Optional[Any] = None
                
                try:
                    self.logger.info(f"[RunID: {run_id}] Invoking agent '{resolved_agent_id_for_stage}' for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'.") # MODIFIED
                    if inspect.iscoroutinefunction(agent_callable) or inspect.iscoroutinefunction(getattr(agent_callable, '__call__', None)):
                        stage_result_payload = await agent_callable(resolved_inputs, full_context=current_context)
                    elif callable(agent_callable):
                        stage_result_payload = await asyncio.to_thread(agent_callable, resolved_inputs, full_context=current_context)
                    else:
                        raise TypeError(f"Agent {resolved_agent_id_for_stage} is not callable.")
                    
                    self.logger.info(f"[RunID: {run_id}] Agent '{resolved_agent_id_for_stage}' for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' completed.") # MODIFIED

                except Exception as agent_exc:
                    tb_str = traceback.format_exc()
                    self.logger.error(
                        f"[RunID: {run_id}] Agent '{resolved_agent_id_for_stage}' for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' raised an exception: {agent_exc}", # MODIFIED
                        exc_info=True,
                    )
                    agent_error_details_for_current_stage = AgentErrorDetails(
                        error_type=type(agent_exc).__name__,
                        message=str(agent_exc),
                        traceback=tb_str,
                        agent_id=resolved_agent_id_for_stage, # Use resolved agent ID
                        stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                        # TODO: Add can_retry from agent spec if available
                    )
                    current_context["_agent_error_details_for_stage_status"] = agent_error_details_for_current_stage.model_dump()


                # If agent execution did not result in an immediate error, 
                # make its payload available in the context for success criteria check.
                if not agent_error_details_for_current_stage:
                    current_context.setdefault("outputs", {})[current_stage_name_for_this_iteration_logging_and_hop_info] = stage_result_payload
                    # Note: self._last_successful_stage_output will be set later only if stage truly PASSES all checks.

                duration_seconds = (datetime.now(timezone.utc) - stage_start_time_agent_call).total_seconds()

                all_success_criteria_passed, failed_criteria_details = await self._check_success_criteria(
                    current_stage_name_for_this_iteration_logging_and_hop_info, current_stage_spec, current_context # MODIFIED
                )
                
                # Determine stage_final_status based on errors or success criteria
                # Error during agent call takes precedence
                if agent_error_details_for_current_stage:
                    stage_final_status = StageStatus.FAILURE
                    if not failed_criteria_details and agent_error_details_for_current_stage.message is None:
                         # If criteria might have passed (or weren't checked due to agent error) but agent had error without message
                         agent_error_details_for_current_stage.message = agent_error_details_for_current_stage.message or "Agent execution failed."
                elif not all_success_criteria_passed:
                    stage_final_status = StageStatus.FAILURE
                    # Populate agent_error_details if not already set by an agent execution exception
                    agent_error_details_for_current_stage = AgentErrorDetails(
                        error_type="SuccessCriteriaFailed",
                        message=f"Stage failed success criteria: {failed_criteria_details}",
                        agent_id=resolved_agent_id_for_stage,
                        stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                        details={"failed_criteria": failed_criteria_details},
                        can_retry=False # Typically, success criteria failure is not a simple retry
                    )
                else:
                    stage_final_status = StageStatus.SUCCESS
                    # Outputs already populated above if no agent error.
                    # Now that all checks passed, confirm _last_successful_stage_output.
                    self._last_successful_stage_output = stage_result_payload


            # --- Unified Stage End Processing (Status Update, Metrics, Next Stage Determination) ---
            # `stage_final_status` is now set (SUCCESS or FAILURE)
            # `agent_error_details_for_current_stage` contains error details if any failure occurred (agent exec, criteria, or resolution)

            stage_reason_for_state_manager: str = f"Stage {stage_final_status.value}"
            if agent_error_details_for_current_stage:
                stage_reason_for_state_manager += f" - {agent_error_details_for_current_stage.message or agent_error_details_for_current_stage.error_type}"


            # TODO: Artifact extraction needs to be robust and happen before this based on stage_result_payload
            extracted_artifacts_for_sm: List[Dict[str, str]] = [] # Placeholder
            if stage_result_payload and isinstance(stage_result_payload, dict):
                 # Example: looking for a conventional key
                raw_artifacts = stage_result_payload.get(self.ARTIFACT_OUTPUT_KEY, [])
                if isinstance(raw_artifacts, list):
                    for art_path in raw_artifacts:
                        if isinstance(art_path, str):
                             # Basic validation, can be expanded
                            extracted_artifacts_for_sm.append({"path": art_path, "type": "file"}) # example structure
                        else:
                            self.logger.warning(f"Non-string artifact path found in {self.ARTIFACT_OUTPUT_KEY}: {art_path}")
            
            self.state_manager.update_status(
                stage=current_stage_spec.number if current_stage_spec and current_stage_spec.number is not None else -1.0,
                status=stage_final_status.value,
                artifacts=extracted_artifacts_for_sm,
                reason=stage_reason_for_state_manager,
                error_details=agent_error_details_for_current_stage 
            )

            self._emit_metric(
                event_type=MetricEventType.STAGE_END,
                flow_id=flow_id,
                run_id=run_id,
                stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                master_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                agent_id=resolved_agent_id_for_stage if 'resolved_agent_id_for_stage' in locals() else current_stage_spec.agent_id, # MODIFIED
                data={
                    "status": stage_final_status.value,
                    "duration_seconds": duration_seconds if 'duration_seconds' in locals() else 0,
                    "error_details": agent_error_details_for_current_stage.model_dump_json() if agent_error_details_for_current_stage else None,
                    "output_keys": list(stage_result_payload.keys()) if isinstance(stage_result_payload, dict) else None, # ADDED
                    "failed_success_criteria": failed_criteria_details if 'failed_criteria_details' in locals() and failed_criteria_details else None # ADDED
                },
            )


            # --- Next Stage Determination & Reviewer Interaction (if needed) ---
            # next_stage_name_after_execution = None # Already initialized at top of loop iteration

            if stage_final_status == StageStatus.SUCCESS:
                if current_stage_spec.clarification_checkpoint and \
                   current_stage_spec.clarification_checkpoint.is_enabled and \
                   not current_context.get("global_flow_state", {}).get(f"_clarification_approved_{current_stage_name_for_this_iteration_logging_and_hop_info}"): # MODIFIED

                    self.logger.info(f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' requires clarification. Pausing.") # MODIFIED
                    paused_state_details = self._create_paused_state_details(
                        master_flow_id=flow_id,
                        paused_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                        execution_context=copy.deepcopy(current_context),
                        error_details_model=None, # Not an error pause
                        status_reason=f"Paused for clarification checkpoint: {current_stage_spec.clarification_checkpoint.message}",
                        clarification_request={
                            "type": "USER_CLARIFICATION_REQUIRED",
                            "message": current_stage_spec.clarification_checkpoint.message,
                            "options": current_stage_spec.clarification_checkpoint.options,
                            "resume_hint": f"Use 'chungoid flow resume {run_id} --action provide_clarification --action-data \'{{\"choice\": \"<USER_CHOICE>\"}}\'"
                        },
                        pause_status=FlowPauseStatus.USER_CLARIFICATION_REQUIRED
                    )
                    self.state_manager.save_paused_flow_state(paused_state_details)
                    self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, data={"reason": "Clarification required", "stage_id": current_stage_name_for_this_iteration_logging_and_hop_info}) # MODIFIED
                    current_context["_autonomous_flow_paused_state_saved"] = True
                    return current_context # Exit execution loop

                else: # No clarification needed or already approved
                    if current_stage_spec.condition:
                        if self._parse_condition(current_stage_spec.condition, current_context):
                            next_stage_name_after_execution = current_stage_spec.next_stage_true
                        else:
                            next_stage_name_after_execution = current_stage_spec.next_stage_false
                        self.logger.info(
                            f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' ({current_stage_spec.number}) condition '{current_stage_spec.condition}' evaluated. Next stage: '{next_stage_name_after_execution}'" # MODIFIED
                        )
                    else: # Stage has no condition, direct next stage
                        next_stage_name_after_execution = current_stage_spec.next_stage
                        self.logger.info(
                            f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' ({current_stage_spec.number}) SUCCESS, no condition. Next stage from spec: '{next_stage_name_after_execution}'" # MODIFIED
                        )

                    # ---- START: Simplified Processing for output_context_path ----
                    if current_stage_spec.output_context_path:
                        self.logger.info(
                            f"[RunID: {run_id}] Stage '{current_stage_name}' has output_context_path defined: {current_stage_spec.output_context_path}"
                        )
                        if stage_result_payload is None:
                            self.logger.error(f"[RunID: {run_id}] CRITICAL_ERROR: agent_result_payload is None before processing output_context_path for stage '{current_stage_name}'. This should not happen.")
                        
                        self.logger.info(
                            f"[RunID: {run_id}] PRE_UPDATE_CONTEXT for stage '{current_stage_name}': Storing entire agent_result_payload to '{current_stage_spec.output_context_path}'. Type: {type(stage_result_payload)}. Value (first 100 chars): {str(stage_result_payload)[:100]}"
                        )
                        self._update_context_with_stage_output(
                            context_data=current_context, # Use current_context
                            stage_name=current_stage_name, 
                            output_context_path=current_stage_spec.output_context_path, 
                            stage_result_payload=stage_result_payload, # Store the entire payload
                        )
                        # Standard INFO log after update, if needed for specific path debugging.
                        # current_intermediate_outputs = current_context.get('intermediate_outputs')
                        # log_intermediate_val = str(current_intermediate_outputs)[:200] + ("..." if len(str(current_intermediate_outputs)) > 200 else "")
                        # self.logger.info(
                        #    f"[RunID: {run_id}] POST_UPDATE_CONTEXT for stage '{current_stage_name}'. Context intermediate_outputs (first 200 chars): {log_intermediate_val}"
                        # )
                    else:
                        self.logger.info(
                            f"[RunID: {run_id}] Stage '{current_stage_name}' has no output_context_path defined. Raw agent output stored in context.outputs.{current_stage_name}."
                        )
                    # ---- END: Simplified Processing for output_context_path ----

            elif stage_final_status == StageStatus.FAILURE:
                self.logger.warning(f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' FAILED. Error: {agent_error_details_for_current_stage.message if agent_error_details_for_current_stage else 'Unknown reason'}") # MODIFIED

                reviewer_output: Optional[MasterPlannerReviewerOutput] = None
                # Determine if reviewer should be called (e.g., based on stage spec or global config)
                should_call_reviewer = True # Default to true, can be made configurable
                
                if should_call_reviewer:
                    reviewer_output = await self._invoke_reviewer_and_get_suggestion(
                        run_id=run_id,
                        flow_id=flow_id,
                        current_stage_name=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                        agent_error_details=agent_error_details_for_current_stage, 
                        current_context=current_context
                    )

                if reviewer_output:
                    # Handle reviewer suggestions
                    if reviewer_output.suggestion_type == ReviewerActionType.RETRY_STAGE_AS_IS:
                        self.logger.info(f"[RunID: {run_id}] Reviewer suggested RETRY_STAGE_AS_IS for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'. Retrying.") # MODIFIED
                        # current_context["_last_master_stage_hop_info"] = { ... } # Not needed as we continue current stage
                        if "_flow_error" in current_context: del current_context["_flow_error"]
                        # No change to current_stage_name, just continue to retry the same stage
                        continue 

                    elif reviewer_output.suggestion_type == ReviewerActionType.RETRY_STAGE_WITH_MODIFIED_INPUT:
                        # ... (logic as before, using current_stage_name_for_this_iteration_logging_and_hop_info) ...
                        # If can_retry_autonomously:
                        #   Update stage_spec.inputs
                        #   continue
                        # Else (pause for user input):
                        #   _create_paused_state_details, save_paused_flow_state, return current_context
                        # Placeholder for existing logic, ensuring current_stage_name_for_this_iteration_logging_and_hop_info is used
                        self.logger.info(f"[RunID: {run_id}] Reviewer suggested RETRY_STAGE_WITH_MODIFIED_INPUT for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'.") # MODIFIED
                        
                        can_retry_autonomously = False
                        if reviewer_output.suggestion_details and \
                           reviewer_output.suggestion_details.get("target_stage_id") == current_stage_name_for_this_iteration_logging_and_hop_info and \
                           "new_inputs" in reviewer_output.suggestion_details and \
                           isinstance(reviewer_output.suggestion_details["new_inputs"], dict):
                            
                            new_inputs_from_reviewer = reviewer_output.suggestion_details["new_inputs"]
                            self.logger.info(f"[RunID: {run_id}] Attempting to apply new inputs for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}': {new_inputs_from_reviewer}") # MODIFIED
                            
                            if current_stage_name_for_this_iteration_logging_and_hop_info in self.current_plan.stages: # MODIFIED
                                self.current_plan.stages[current_stage_name_for_this_iteration_logging_and_hop_info].inputs = new_inputs_from_reviewer # MODIFIED
                                current_stage_spec.inputs = new_inputs_from_reviewer 
                                
                                self.logger.info(f"[RunID: {run_id}] Inputs for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' updated in plan for this run. Retrying stage.") # MODIFIED
                                if "_flow_error" in current_context: del current_context["_flow_error"]
                                can_retry_autonomously = True
                                continue 
                            else:
                                self.logger.error(f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' (target_stage_id from reviewer) not found in plan. Cannot apply new inputs. Escalating.") # MODIFIED
                        else:
                            self.logger.warning(f"[RunID: {run_id}] Reviewer suggested RETRY_STAGE_WITH_MODIFIED_INPUT but 'new_inputs' were missing, malformed, or for the wrong stage. Details: {reviewer_output.suggestion_details}. Escalating for user to provide inputs.")

                        if not can_retry_autonomously:
                            user_message = f"Reviewer suggested retrying stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' with modified inputs, but could not automatically determine them." # MODIFIED
                            reviewer_analysis = reviewer_output.suggestion_details.get("modification_needed", "No specific analysis provided by reviewer.")
                            original_inputs = current_stage_spec.inputs if current_stage_spec else {}

                            current_context["_flow_error"] = { 
                                "message": user_message,
                                "stage": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                                "reviewer_suggestion_type": reviewer_output.suggestion_type.value,
                                "reviewer_reasoning": reviewer_output.reasoning,
                                "reviewer_analysis": reviewer_analysis,
                                "original_inputs": original_inputs,
                                "original_agent_error": agent_error_details_for_current_stage.model_dump() if agent_error_details_for_current_stage else None
                            }
                            
                            paused_state_details = self._create_paused_state_details(
                                master_flow_id=flow_id,
                                paused_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                                execution_context=copy.deepcopy(current_context),
                                error_details_model=agent_error_details_for_current_stage, 
                                status_reason=user_message,
                                clarification_request={
                                    "type": "INPUT_MODIFICATION_REQUIRED", 
                                    "message": user_message,
                                    "reviewer_analysis": reviewer_analysis,
                                    "original_stage_inputs": original_inputs,
                                    "reviewer_output": reviewer_output.model_dump(),
                                    "action_required": f"Please provide the complete, corrected set of inputs for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' to retry.", # MODIFIED
                                    "resume_hint": f"Use 'chungoid flow resume {run_id} --action retry_with_inputs --action-data \'{{\"target_stage_id\": \"{current_stage_name_for_this_iteration_logging_and_hop_info}\", \"inputs\": {{...}}}}\'" # MODIFIED                                    
                                },
                                pause_status=FlowPauseStatus.USER_INPUT_REQUIRED 
                            )
                            self.state_manager.save_paused_flow_state(paused_state_details)
                            self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, 
                                              data={"reason": "Awaiting user input for RETRY_STAGE_WITH_MODIFIED_INPUT", "stage_id": current_stage_name_for_this_iteration_logging_and_hop_info}) # MODIFIED
                            current_context["_autonomous_flow_paused_state_saved"] = True
                            return current_context

                    elif reviewer_output.suggestion_type == ReviewerActionType.MODIFY_MASTER_PLAN:
                        # ... (logic as before, using current_stage_name_for_this_iteration_logging_and_hop_info) ...
                        # _create_paused_state_details, save_paused_flow_state, return current_context
                        self.logger.info(f"[RunID: {run_id}] Reviewer suggested MODIFY_MASTER_PLAN for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'. Escalating for user review.") # MODIFIED
                        user_message = reviewer_output.suggestion_details.get("message_to_user", 
                                                                              f"Reviewer suggested modifying the master plan due to issues at stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'. Reasoning: {reviewer_output.reasoning}") # MODIFIED
                        if "suggested_plan_change" in reviewer_output.suggestion_details:
                            user_message += f" Suggested change: {reviewer_output.suggestion_details['suggested_plan_change']}"

                        current_context["_flow_error"] = { 
                            "message": f"Flow paused for user review: Reviewer suggested MODIFY_MASTER_PLAN for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'.", # MODIFIED
                            "stage": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            "reviewer_suggestion_type": reviewer_output.suggestion_type.value,
                            "reviewer_reasoning": reviewer_output.reasoning,
                            "reviewer_message_to_user": user_message,
                            "original_agent_error": agent_error_details_for_current_stage.model_dump() if agent_error_details_for_current_stage else None
                        }
                        
                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=flow_id,
                            paused_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            execution_context=copy.deepcopy(current_context),
                            error_details_model=agent_error_details_for_current_stage, 
                            status_reason=f"Reviewer suggested MODIFY_MASTER_PLAN. {user_message}",
                            clarification_request={"type": "PLAN_MODIFICATION_REVIEW", "message": user_message, "reviewer_output": reviewer_output.model_dump()},
                            pause_status=FlowPauseStatus.PAUSED_FOR_INTERVENTION 
                        )
                        self.state_manager.save_paused_flow_state(paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, 
                                          data={"reason": "Reviewer suggested MODIFY_MASTER_PLAN", "stage_id": current_stage_name_for_this_iteration_logging_and_hop_info}) # MODIFIED
                        current_context["_autonomous_flow_paused_state_saved"] = True 
                        return current_context

                    elif reviewer_output.suggestion_type == ReviewerActionType.ESCALATE_TO_USER:
                        # ... (logic as before, using current_stage_name_for_this_iteration_logging_and_hop_info) ...
                        # _create_paused_state_details, save_paused_flow_state, return current_context
                        self.logger.info(f"[RunID: {run_id}] Reviewer suggested ESCALATE_TO_USER for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'. Pausing for user intervention.") # MODIFIED
                        user_message = reviewer_output.suggestion_details.get("message_to_user",
                                                                              f"Reviewer escalated stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' for user intervention. Reasoning: {reviewer_output.reasoning}") # MODIFIED

                        current_context["_flow_error"] = { 
                            "message": f"Flow paused: Reviewer escalated stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'.", # MODIFIED
                            "stage": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            "reviewer_suggestion_type": reviewer_output.suggestion_type.value,
                            "reviewer_reasoning": reviewer_output.reasoning,
                            "reviewer_message_to_user": user_message,
                            "original_agent_error": agent_error_details_for_current_stage.model_dump() if agent_error_details_for_current_stage else None
                        }

                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=flow_id,
                            paused_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            execution_context=copy.deepcopy(current_context),
                            error_details_model=agent_error_details_for_current_stage,
                            status_reason=f"Reviewer escalated for user intervention. {user_message}",
                            clarification_request={"type": "USER_ESCALATION", "message": user_message, "reviewer_output": reviewer_output.model_dump()},
                            pause_status=FlowPauseStatus.PAUSED_FOR_INTERVENTION # CORRECTED ENUM MEMBER
                        )
                        self.state_manager.save_paused_flow_state(paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, 
                                          data={"reason": "Reviewer suggested ESCALATE_TO_USER", "stage_id": current_stage_name_for_this_iteration_logging_and_hop_info}) # MODIFIED
                        current_context["_autonomous_flow_paused_state_saved"] = True 
                        return current_context 

                    elif reviewer_output.suggestion_type == ReviewerActionType.PROCEED_AS_IS:
                        self.logger.info(f"[RunID: {run_id}] Reviewer suggested PROCEED_AS_IS after failure of stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'. Determining next stage.") # MODIFIED
                        # This path should determine the next_stage_name_after_execution based on normal flow (condition/next_stage)
                        # as if the stage had succeeded with warnings.
                        if current_stage_spec.condition:
                            if self._parse_condition(current_stage_spec.condition, current_context):
                                next_stage_name_after_execution = current_stage_spec.next_stage_true
                            else:
                                next_stage_name_after_execution = current_stage_spec.next_stage_false
                        else:
                            next_stage_name_after_execution = current_stage_spec.next_stage
                        
                        self.logger.info(f"[RunID: {run_id}] Reviewer PROCEED_AS_IS: Next stage for '{current_stage_name_for_this_iteration_logging_and_hop_info}' is '{next_stage_name_after_execution}'.") # MODIFIED
                        if "_flow_error" in current_context: del current_context["_flow_error"]
                        # current_stage_name will be updated by the universal logic at the end of the loop using next_stage_name_after_execution
                        # No 'continue' here, let it fall through to universal update logic

                    elif reviewer_output.suggestion_type == ReviewerActionType.NO_ACTION_SUGGESTED:
                        # ... (logic as before, using current_stage_name_for_this_iteration_logging_and_hop_info) ...
                        # _create_paused_state_details, save_paused_flow_state, return current_context
                        self.logger.warning(f"[RunID: {run_id}] Reviewer provided NO_ACTION_SUGGESTED for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'. Escalating for user review.") # MODIFIED
                        user_message = (f"Reviewer was invoked for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' but provided no specific recovery action. " # MODIFIED
                                        f"Original error: {agent_error_details_for_current_stage.message if agent_error_details_for_current_stage else 'N/A'}. "
                                        f"Reviewer reasoning: {reviewer_output.reasoning}")
                        
                        current_context["_flow_error"] = { 
                            "message": f"Flow paused: Reviewer provided NO_ACTION_SUGGESTED for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'.", # MODIFIED
                            "stage": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            "reviewer_suggestion_type": reviewer_output.suggestion_type.value,
                            "reviewer_reasoning": reviewer_output.reasoning,
                            "reviewer_message_to_user": user_message,
                            "original_agent_error": agent_error_details_for_current_stage.model_dump() if agent_error_details_for_current_stage else None
                        }

                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=flow_id,
                            paused_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            execution_context=copy.deepcopy(current_context),
                            error_details_model=agent_error_details_for_current_stage, 
                            status_reason=f"Reviewer provided NO_ACTION_SUGGESTED. {user_message}",
                            clarification_request={"type": "NO_REVIEWER_SUGGESTION", "message": user_message, "reviewer_output": reviewer_output.model_dump()},
                            pause_status=FlowPauseStatus.PAUSED_FOR_INTERVENTION 
                        )
                        self.state_manager.save_paused_flow_state(paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, 
                                          data={"reason": "Reviewer provided NO_ACTION_SUGGESTED", "stage_id": current_stage_name_for_this_iteration_logging_and_hop_info}) # MODIFIED
                        current_context["_autonomous_flow_paused_state_saved"] = True 
                        return current_context 

                    else: # Fallback for unhandled reviewer suggestions
                        self.logger.warning(f"[RunID: {run_id}] Reviewer suggestion type '{reviewer_output.suggestion_type.value}' not fully handled or led to fallback. Defaulting to PAUSE_FOR_INTERVENTION for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'.") # MODIFIED
                        # ... (logic as before, using current_stage_name_for_this_iteration_logging_and_hop_info, then return current_context) ...
                        current_context["_flow_error"] = { 
                            "message": f"Flow paused for intervention: Unhandled or fallback reviewer suggestion for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'.", # MODIFIED
                            "stage": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            "reviewer_suggestion_type": reviewer_output.suggestion_type.value,
                            "reviewer_reasoning": reviewer_output.reasoning,
                            "original_agent_error": agent_error_details_for_current_stage.model_dump() if agent_error_details_for_current_stage else None
                        }
                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=flow_id,
                            paused_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            execution_context=copy.deepcopy(current_context),
                            error_details_model=agent_error_details_for_current_stage,
                            status_reason=f"Unhandled reviewer suggestion ({reviewer_output.suggestion_type.value}) or fallback. Reasoning: {reviewer_output.reasoning}",
                            clarification_request={"type": "UNHANDLED_REVIEWER_SUGGESTION", "message": f"Reviewer suggested {reviewer_output.suggestion_type.value}, which led to a fallback pause.", "reviewer_output": reviewer_output.model_dump()},
                            pause_status=FlowPauseStatus.PAUSED_FOR_INTERVENTION 
                        )
                        self.state_manager.save_paused_flow_state(paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, data={"reason": "Unhandled reviewer suggestion", "stage_id": current_stage_name_for_this_iteration_logging_and_hop_info}) # MODIFIED
                        current_context["_autonomous_flow_paused_state_saved"] = True 
                        return current_context 

                else: # No reviewer_output (reviewer not configured or failed to provide suggestion)
                    effective_on_failure = current_stage_spec.on_failure
                    if not effective_on_failure:
                        self.logger.warning(f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' failed and no on_failure policy defined. Defaulting to PAUSE_FOR_INTERVENTION.") # MODIFIED
                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=flow_id, # Use flow_id
                            paused_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            execution_context=copy.deepcopy(current_context), # Use current_context
                            error_details_model=agent_error_details_for_current_stage,
                            status_reason=f"Stage failed (no reviewer action, no on_failure policy): {agent_error_details_for_current_stage.message if agent_error_details_for_current_stage else 'Unknown Error'}",
                            clarification_request=None,
                            pause_status=FlowPauseStatus.PAUSED_FOR_INTERVENTION
                        )
                        self.state_manager.save_paused_flow_state(paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, data={"reason": "Stage failure, no reviewer action, no on_failure policy", "stage_id": current_stage_name_for_this_iteration_logging_and_hop_info}) # MODIFIED
                        current_context["_autonomous_flow_paused_state_saved"] = True
                        return current_context 

                    if effective_on_failure.action == "FAIL_MASTER_FLOW":
                        self.logger.error(f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' failed. Master flow configured to FAIL_MASTER_FLOW. Message: {effective_on_failure.log_message}") # MODIFIED
                        next_stage_name_after_execution = "_END_FAILURE_" 
                    elif effective_on_failure.action == "GOTO_MASTER_STAGE":
                        target_next_stage = effective_on_failure.target_master_stage_key
                        self.logger.info(f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' failed. Transitioning to on_failure stage: '{target_next_stage}'.") # MODIFIED
                        if target_next_stage not in self.current_plan.stages and target_next_stage not in ["_END_SUCCESS_", "_END_FAILURE_", "FINAL_STEP"]:
                            self.logger.error(f"[RunID: {run_id}] on_failure target_master_stage_key '{target_next_stage}' not found in plan. Terminating with failure.") # MODIFIED
                            next_stage_name_after_execution = "_END_FAILURE_"
                        else:
                            next_stage_name_after_execution = target_next_stage
                    elif effective_on_failure.action == "PAUSE_FOR_INTERVENTION":
                        self.logger.info(f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' failed. Pausing for human intervention as per on_failure policy.") # MODIFIED
                        paused_state_details = self._create_paused_state_details(
                            master_flow_id=flow_id, # Use flow_id
                            paused_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            execution_context=copy.deepcopy(current_context), # Use current_context
                            error_details_model=agent_error_details_for_current_stage,
                            status_reason=f"Stage failed (on_failure policy PAUSE): {agent_error_details_for_current_stage.message if agent_error_details_for_current_stage else 'Unknown Error'}",
                            clarification_request=None,
                            pause_status=FlowPauseStatus.PAUSED_FOR_INTERVENTION
                        )
                        self.state_manager.save_paused_flow_state(paused_state_details)
                        self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, data={"reason": "Stage failure, on_failure policy PAUSE", "stage_id": current_stage_name_for_this_iteration_logging_and_hop_info}) # MODIFIED
                        current_context["_autonomous_flow_paused_state_saved"] = True
                        return current_context
                    else: 
                        self.logger.error(f"[RunID: {run_id}] Unknown on_failure action: {effective_on_failure.action} for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'. Terminating with failure.") # MODIFIED
                        next_stage_name_after_execution = "_END_FAILURE_"
            
            elif stage_final_status == StageStatus.PAUSED_FOR_INTERVENTION: # Should ideally not be hit if pause logic returns context
                self.logger.warning(f"[RunID: {run_id}] Stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' was marked PAUSED_FOR_INTERVENTION but orchestrator loop continued. This may be unexpected.") # MODIFIED
                # If flow was paused, current_context["_autonomous_flow_paused_state_saved"] should be True and this path might be skipped by outer logic.
                # If not, this is a fallback to ensure the loop terminates or pauses correctly.
                if not current_context.get("_autonomous_flow_paused_state_saved"):
                    # This indicates a logic error if a stage is PAUSED but didn't save and return.
                    # For safety, treat as a generic pause if we reach here.
                    paused_state_details = self._create_paused_state_details(
                        master_flow_id=flow_id,
                        paused_stage_id=current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                        execution_context=copy.deepcopy(current_context),
                        error_details_model=agent_error_details_for_current_stage, # May or may not be set
                        status_reason="Flow reached PAUSED_FOR_INTERVENTION status without explicit pause handling.",
                        clarification_request=None,
                        pause_status=FlowPauseStatus.PAUSED_FOR_INTERVENTION 
                    )
                    self.state_manager.save_paused_flow_state(paused_state_details)
                    self._emit_metric(event_type=MetricEventType.FLOW_PAUSED, flow_id=flow_id, run_id=run_id, data={"reason": "Fallback PAUSED_FOR_INTERVENTION", "stage_id": current_stage_name_for_this_iteration_logging_and_hop_info}) # MODIFIED
                    current_context["_autonomous_flow_paused_state_saved"] = True
                    return current_context


            # === UNIVERSAL STAGE TRANSITION LOGIC (if not already returned/continued earlier in this iteration) ===
            if current_context.get("_autonomous_flow_paused_state_saved"):
                # Flow was paused in this iteration (e.g. clarification, reviewer action), orchestrator will return current_context.
                # No stage transition should happen here. The `return current_context` above would have exited.
                # This check is more of a safeguard if somehow that return was missed.
                 self.logger.debug(f"[RunID: {run_id}] Flow pause state saved for stage '{current_stage_name_for_this_iteration_logging_and_hop_info}'. Orchestrator loop will return.") # MODIFIED
            else: # Flow was not paused in this iteration, determine next stage for the loop.
                if next_stage_name_after_execution is not None:
                    # A next stage was determined by SUCCESS, or by FAILURE + Reviewer/OnFailure GOTO
                    if next_stage_name_after_execution in ["_END_SUCCESS_", "_END_FAILURE_", "FINAL_STEP"]:
                        self.logger.info(f"[RunID: {run_id}] Transitioning from '{current_stage_name_for_this_iteration_logging_and_hop_info}' to terminal state: {next_stage_name_after_execution}") # MODIFIED
                    elif next_stage_name_after_execution in self.current_plan.stages:
                        current_context["_last_master_stage_hop_info"] = {
                            "from": current_stage_name_for_this_iteration_logging_and_hop_info, # MODIFIED
                            "to": next_stage_name_after_execution,
                            "reason": "Stage progression" 
                        }
                        self.logger.debug(f"[RunID: {run_id}] Advancing from stage '{current_stage_name_for_this_iteration_logging_and_hop_info}' to stage: '{next_stage_name_after_execution}'") # MODIFIED
                    else: 
                        self.logger.error(f"[RunID: {run_id}] Invalid next stage '{next_stage_name_after_execution}' determined from '{current_stage_name_for_this_iteration_logging_and_hop_info}'. Aborting.") # MODIFIED
                        current_context["_flow_error"] = {"message": f"Invalid next stage '{next_stage_name_after_execution}' determined."}
                        next_stage_name_after_execution = "_END_FAILURE_" 
                    
                    current_stage_name = next_stage_name_after_execution # CRITICAL FIX: Update current_stage_name
                
                elif not (current_stage_name_for_this_iteration_logging_and_hop_info in ["_END_SUCCESS_", "_END_FAILURE_", "FINAL_STEP"]): # MODIFIED (check original stage)
                    # No next_stage_name_after_execution defined (e.g. last stage of a sequence with no explicit 'next'),
                    # and not already in a terminal state.
                    self.logger.info(f"[RunID: {run_id}] No next stage determined after '{current_stage_name_for_this_iteration_logging_and_hop_info}'. Flow considered complete.") # MODIFIED
                    current_stage_name = "_END_SUCCESS_" # CRITICAL FIX: Update current_stage_name to terminal state
                # If current_stage_name was already a terminal state and next_stage_name_after_execution is None, current_stage_name remains, and loop will terminate.

            # Cleanup context key that might have been set by agent error handling
            # This pop was originally much earlier. If agent_error_details_for_current_stage is used consistently, this might not be needed here.
            # However, if "_agent_error_details_for_stage_status" was set directly on context by older agent code, this is a safety.
            if "_agent_error_details_for_stage_status" in current_context:
                del current_context["_agent_error_details_for_stage_status"]
            

        # End of the `while current_stage_name and current_stage_name not in ["FINAL_STEP", "_END_SUCCESS_", "_END_FAILURE_"]:` loop
        self.logger.debug(f"[RunID: {run_id}] Exited master flow execution loop. Final effective stage_name for termination was: {current_stage_name}")
        return current_context # Ensure context is returned even if loop finishes normally.

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
                flow_has_ended = True # Mark that the flow has reached a terminal state
                flow_final_status_for_metric = StageStatus.FAILURE
                final_reason_for_metric = f"Flow failed: {final_context_for_loop['_flow_error'].get('message', 'Unknown error')}"
                self.logger.error(
                    f"Plan '{self.current_plan.id}', run '{self._current_run_id}' finished with error: {final_reason_for_metric}"
                ) 
            elif final_context_for_loop.get("_autonomous_flow_paused_state_saved"):
                flow_has_ended = False # Mark that the flow is paused, not terminally ended
                flow_final_status_for_metric = None # No terminal status for metric yet
                final_reason_for_metric = f"Flow paused. Run ID: {self._current_run_id}. See PausedRunDetails."
                self.logger.info(final_reason_for_metric)
                # No call to state_manager.update_status for overall flow here, it's paused.
            else:
                flow_has_ended = True # Mark that the flow has reached a terminal state
                flow_final_status_for_metric = StageStatus.SUCCESS
                final_reason_for_metric = "Flow completed successfully."
                self.logger.info(
                    f"Plan '{self.current_plan.id}', run '{self._current_run_id}' completed successfully."
                )
                # This call to update_status was for overall run, ensure it uses valid enum values
                # and is appropriate for a successful *terminal* state.
                status_message_for_run = f"Flow {flow_final_status_for_metric.value}: {final_reason_for_metric}"
                ctx_to_save = {
                    **final_context_for_loop,
                    "_flow_end_reason": final_reason_for_metric,
                }
                # This specific update_status call for overall success might need to be re-evaluated
                # regarding its parameters, especially if stage = -1.0 means something special.
                # For now, ensuring it uses a valid status. 
                # The original code here also did not pass error_details, which is fine for success.
                self.state_manager.update_status(
                    stage=-1.0,  # Signify run-level update
                    status=flow_final_status_for_metric.value, # Use the Pydantic model's value
                    artifacts=[],
                    reason=final_reason_for_metric,
                    # No error_details for success
                )

        except Exception as e:
            flow_has_ended = True # Unhandled exception means terminal failure
            tb_str = traceback.format_exc()
            self.logger.exception(
                f"Unhandled exception during execution of plan '{self.current_plan.id}', run '{self._current_run_id}': {e}"
            )
            flow_final_status_for_metric = StageStatus.FAILURE
            error_info_for_state = {
                "error_type": type(e).__name__,
                "message": str(e),
                "traceback": tb_str,
            }
            final_reason_for_metric = f"Flow failed with unhandled exception: {str(e)}"
            final_context_for_loop["_flow_error"] = error_info_for_state
            try:
                # This call is for overall run failure due to unhandled exception.
                self.state_manager.update_status(
                    stage=-1.0,  # Signify run-level update
                    status=flow_final_status.value,  # CORRECTED
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

    # ---- Helper for creating PausedRunDetails ----
    def _create_paused_state_details(
        self,
        master_flow_id: str,
        paused_stage_id: str,
        execution_context: Dict[str, Any],
        error_details_model: Optional[AgentErrorDetails],
        status_reason: str, # General reason for the pause status message
        clarification_request: Optional[Dict[str, Any]],
        pause_status: FlowPauseStatus # The specific FlowPauseStatus enum member
    ) -> PausedRunDetails:
        """Helper method to construct a PausedRunDetails object."""
        if not self._current_run_id:
            self.logger.error("CRITICAL: _create_paused_state_details called but _current_run_id is not set.")
            # Fallback or raise? For now, generate a temporary one to avoid None.
            # This indicates a larger issue if hit.
            fallback_run_id = str(uuid.uuid4())
            self.logger.warning(f"Using fallback UUID for run_id in PausedRunDetails: {fallback_run_id}")
            run_id_to_use = fallback_run_id
        else:
            run_id_to_use = self._current_run_id

        return PausedRunDetails(
            run_id=run_id_to_use,
            flow_id=master_flow_id,
            paused_at_stage_id=paused_stage_id,
            timestamp=datetime.now(timezone.utc),
            status=pause_status, # Use the provided FlowPauseStatus enum member
            context_snapshot=copy.deepcopy(execution_context),
            error_details=error_details_model,
            clarification_request=clarification_request
        )

    def _update_context_with_stage_output(
        self,
        context_data: Dict[str, Any],
        stage_name: str,
        output_context_path: Optional[str],
        stage_result_payload: Any,
    ):
        # Always store raw output directly under context.outputs.stage_name
        outputs_dict = context_data.setdefault("outputs", {})
        outputs_dict[stage_name] = stage_result_payload
        # Standard debug log for raw output storage.
        # self.logger.debug(f"Stored raw output for stage '{stage_name}' in context.outputs.{stage_name} (type: {type(stage_result_payload)})")

        if output_context_path:
            # self.logger.info(f"UPDATE_CTX_ENTRY: stage='{stage_name}', output_path='{output_context_path}', payload_type='{type(stage_result_payload)}'")
            
            if output_context_path.startswith("intermediate_outputs."):
                if not isinstance(context_data.get("intermediate_outputs"), dict):
                    self.logger.warning(
                        f"Forcing context_data['intermediate_outputs'] to {{}} because it was {type(context_data.get("intermediate_outputs"))} "
                        f"before processing path '{output_context_path}' for stage '{stage_name}'."
                    )
                    context_data["intermediate_outputs"] = {}
            
            path_parts: List[str]
            # Determine the path parts for placing the output in the context
            if output_context_path.lower().startswith("context."):
                path_parts = output_context_path.split(".")[1:] # Remove "context." prefix
            else:
                # If not starting with "context.", treat the whole string as the path from the root of context_data
                path_parts = output_context_path.split(".") # CORRECTED LOGIC
            
            current_level = context_data
            try:
                for i, part in enumerate(path_parts):
                    if i == len(path_parts) - 1: 
                        current_level[part] = stage_result_payload
                        # Standard INFO log for context update, if needed for specific path debugging.
                        # if output_context_path == "intermediate_outputs.generated_command_code_artifacts":
                        #    log_intermediate_outputs = context_data.get('intermediate_outputs')
                        #    log_generated_artifacts = None
                        #    if isinstance(log_intermediate_outputs, dict):
                        #        log_generated_artifacts = log_intermediate_outputs.get('generated_command_code_artifacts')
                        #    self.logger.info(
                        #        f"UPDATE_CTX_WRITE: Path='{output_context_path}'. intermediate_outputs.generated_command_code_artifacts type: {type(log_generated_artifacts)}."
                        #    )
                    else: # Not the last part, ensure dict and traverse
                        existing_val_for_part = current_level.get(part)
                        if not isinstance(existing_val_for_part, dict):
                            # If part doesn't exist, or exists but isn't a dict, overwrite/create it as a new dict
                            self.logger.warning(f"Path part '{part}' in '{output_context_path}' was not a dict (was {type(existing_val_for_part)}). Forcing to dict to allow nested output.")
                            current_level[part] = {}
                        current_level = current_level[part] # Now current_level points to the (potentially new) dict at current_level[part]
                self.logger.info(f"Output for stage '{stage_name}' (type: {type(stage_result_payload)}) also placed at custom path: {output_context_path}")
            except Exception as e:
                self.logger.error(f"Error updating context with stage output for custom path '{output_context_path}': {e}", exc_info=True)

# Placeholder for ReviewerDecision schema if it's decided to be defined here.
# class ReviewerDecision(BaseModel):
#     action: ReviewerActionType
#     details: Optional[Dict[str, Any]] = None
#     reasoning: Optional[str] = None
