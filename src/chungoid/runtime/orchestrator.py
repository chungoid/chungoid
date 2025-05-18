"""Minimal execution runtime skeleton (Phase-6 scaffold).

This module will gradually evolve into the production‐grade orchestrator that
executes Flow YAML graphs.  For now we expose a *SyncOrchestrator* that can run
very simple flows in-process so we can start adding tests and build up the API
surface incrementally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Callable, cast, ClassVar
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
from chungoid.schemas.orchestration import SharedContext # ADDED SharedContext IMPORT

# Import for reviewer agent and its schemas
from chungoid.runtime.agents.system_master_planner_reviewer_agent import (
    MasterPlannerReviewerAgent,
)  # Assuming AGENT_ID is on class
from chungoid.schemas.agent_master_planner_reviewer import (
    MasterPlannerReviewerInput, 
    MasterPlannerReviewerOutput, 
    ReviewerActionType,
    ReviewerModifyPlanAction, # ADDED THIS IMPORT
    RetryStageWithChangesDetails, 
    AddClarificationStageDetails, 
    ModifyMasterPlanRemoveStageDetails,
    ModifyMasterPlanModifyStageDetails,
    ModifyMasterPlanDetails, # This is the discriminated union
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
import functools
import json

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
    _current_flow_config: Optional[Dict[str, Any]] = None # Added to store flow-specific config

    shared_context: SharedContext # ADDED: Instance of SharedContext

    COMPARATOR_MAP: ClassVar[Dict[str, Callable[[Any, Any], bool]]] = {
        ">=": lambda x, y: x >= y,
        "<=": lambda x, y: x <= y,
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
        ">": lambda x, y: x > y,
        "<": lambda x, y: x < y,
        # Placeholder for future: "in": lambda x, y: x in y, (y would need to be list/dict)
        # Placeholder for future: "notin": lambda x, y: x not in y,
    }

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
        self.master_flow_path: Optional[Path] = None

        # Initialize SharedContext
        project_id_from_config = self.config.get("project_id")
        project_root_path_from_config = self.config.get("project_root_path")

        # Try to get from state_manager if not in config or if state_manager is primary source
        # Assuming state_manager is initialized and has project_id and project_root
        final_project_id = project_id_from_config or getattr(self.state_manager, 'project_id', None)
        
        # For project_root_path, ensure it's a string. state_manager.project_root might be Path.
        project_root_path_from_sm = getattr(self.state_manager, 'project_root', None)
        final_project_root_path_str = None
        if project_root_path_from_config:
            final_project_root_path_str = str(project_root_path_from_config)
        elif project_root_path_from_sm:
            final_project_root_path_str = str(project_root_path_from_sm)

        if not final_project_id:
            raise ValueError("AsyncOrchestrator: 'project_id' could not be determined from config or StateManager.")
        if not final_project_root_path_str:
            raise ValueError("AsyncOrchestrator: 'project_root_path' could not be determined from config or StateManager.")

        self.shared_context = SharedContext(
            project_id=final_project_id,
            project_root_path=final_project_root_path_str
        )
        # Populate other initial shared_context fields if needed
        self.shared_context.global_project_settings = self.config.get("global_project_settings", {})
        # self.logger.debug(f\"AsyncOrchestrator initialized with SharedContext: {self.shared_context.model_dump_json(indent=2)}\")

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

        elif " IS_BOOL" in criterion_upper: # ADDED BLOCK FOR IS_BOOL
            path_to_check = criterion_upper.split(" IS_BOOL", 1)[0].strip()
            original_path_to_check = criterion[:criterion_upper.find(" IS_BOOL")].strip()
            actual_val = _resolve_path(original_path_to_check, stage_outputs)
            
            is_actually_bool = isinstance(actual_val, bool)
            # For "IS_NOT_BOOL", the logic is just the negation
            is_not_bool_operator = " IS_NOT_BOOL" in criterion_upper # Check if "IS_NOT_BOOL" specifically
            
            result = not is_actually_bool if is_not_bool_operator else is_actually_bool
            
            operator_log_str = "IS_NOT_BOOL" if is_not_bool_operator else "IS_BOOL"
            self.logger.info(f"Criterion '{criterion}' ({operator_log_str} check for '{original_path_to_check}', actual type: {type(actual_val)}) evaluated to {result}")
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

    def _parse_condition(self, condition_str: str) -> bool:
        if not condition_str:
            self.logger.debug("_parse_condition called with empty string, evaluating as True (no condition).")
            return True
        try:
            parts = []
            comparator_symbol = None
            # Ensure longer operators are checked before shorter ones (e.g., >= before >)
            sorted_ops = sorted(list(self.COMPARATOR_MAP.keys()), key=len, reverse=True)

            for op_sym in sorted_ops:
                if op_sym in condition_str:
                    temp_parts = condition_str.split(op_sym, 1)
                    if len(temp_parts) == 2: # Found a valid split
                        parts = temp_parts
                        comparator_symbol = op_sym
                        break
            
            if not comparator_symbol or len(parts) != 2:
                self.logger.error(f"Unsupported condition format or unknown/missing operator: '{condition_str}'. Ensure one of {list(self.COMPARATOR_MAP.keys())} is used correctly.")
                return False

            var_path_expression = parts[0].strip()
            expected_value_str = parts[1].strip()

            if not var_path_expression.startswith("@"):
                self.logger.error(f"Condition variable path '{var_path_expression}' in '{condition_str}' must start with a prefix like @outputs, @context, @artifacts, @state, or @config.")
                return False

            # Use _resolve_input_values to get the actual LHS value
            # _resolve_input_values itself uses self.shared_context, self.config, self.state_manager
            resolved_lhs_dict = self._resolve_input_values({"lhs_value": var_path_expression})
            actual_lhs_value = resolved_lhs_dict.get("lhs_value")

            # self.logger.debug(f"Condition LHS '{var_path_expression}' (from '{condition_str}') resolved to: {actual_lhs_value} (type: {type(actual_lhs_value)})")

            typed_expected_value: Any = expected_value_str
            # Coercion logic for RHS based on LHS type or common patterns
            if expected_value_str.lower() in ["null", "none"]:
                typed_expected_value = None
            elif isinstance(actual_lhs_value, bool) or expected_value_str.lower() in ["true", "false"]:
                # If LHS is bool, or RHS looks like a bool, coerce RHS to bool
                if expected_value_str.lower() == "true": typed_expected_value = True
                elif expected_value_str.lower() == "false": typed_expected_value = False
                # If actual_lhs_value is bool but expected_value_str is not 'true'/'false', comparison will likely be a TypeError or logical false.
            elif isinstance(actual_lhs_value, float):
                try: typed_expected_value = float(expected_value_str)
                except ValueError: 
                    self.logger.debug(f"Condition RHS '{expected_value_str}' could not be coerced to float for '{condition_str}'. Comparing as strings or original types.")
                    # Keep expected_value_str as is, or handle as error? For now, allow comparison to proceed.
            elif isinstance(actual_lhs_value, int):
                try: typed_expected_value = int(expected_value_str)
                except ValueError: 
                    self.logger.debug(f"Condition RHS '{expected_value_str}' could not be coerced to int for '{condition_str}'. Comparing as strings or original types.")
            # Note: If actual_lhs_value is None, direct comparison with coerced typed_expected_value (which might also be None) is fine.
            
            # self.logger.debug(f"Condition RHS '{expected_value_str}' (coerced if applicable for '{condition_str}'): {typed_expected_value} (type: {type(typed_expected_value)})")

            eval_func = self.COMPARATOR_MAP[comparator_symbol]
            try:
                result = eval_func(actual_lhs_value, typed_expected_value)
            except TypeError as te:
                # This handles cases like comparing int with a non-numeric string after coercion attempts failed for RHS
                self.logger.warning(f"Type error during condition evaluation for '{condition_str}': comparing '{actual_lhs_value}' ({type(actual_lhs_value)}) with '{typed_expected_value}' ({type(typed_expected_value)}). Error: {te}. Condition evaluates to False.")
                return False 
            
            # self.logger.debug(f"Condition '{condition_str}' evaluated to: {result}")
            return bool(result)

        except Exception as e:
            self.logger.error(f"General error parsing or evaluating condition '{condition_str}': {e}", exc_info=True)
            return False

    def _get_next_stage(self, current_stage_name: str) -> str | None: 
        if not self.current_plan:
            self.logger.error(f"_get_next_stage called for stage '{current_stage_name}' but no current_plan is loaded.")
            return NEXT_STAGE_END_FAILURE

        stage_spec = self.current_plan.stages.get(current_stage_name)

        if stage_spec:
            # Check for conditional branching first
            if stage_spec.condition: # This field comes from the YAML 'next.condition'
                # Ensure both true and false branches are defined if a condition exists
                if stage_spec.next_stage_true is not None and stage_spec.next_stage_false is not None:
                    # _parse_condition now uses self.shared_context internally via _resolve_input_values
                    condition_result = self._parse_condition(stage_spec.condition)
                    if condition_result:
                        self.logger.debug(f"Condition '{stage_spec.condition}' for stage '{current_stage_name}' is TRUE. Next stage: {stage_spec.next_stage_true}")
                        return stage_spec.next_stage_true
                    else:
                        self.logger.debug(f"Condition '{stage_spec.condition}' for stage '{current_stage_name}' is FALSE. Next stage: {stage_spec.next_stage_false}")
                        return stage_spec.next_stage_false
                else:
                    # This is a plan validation issue: condition exists but branches are not properly defined.
                    self.logger.error(f"Stage '{current_stage_name}' has a condition '{stage_spec.condition}' but is missing one or both conditional branches (next_stage_true/next_stage_false). Terminating path.")
                    return NEXT_STAGE_END_FAILURE 
            # If no condition, check for a simple direct next_stage
            elif stage_spec.next_stage is not None:
                self.logger.debug(f"No condition for stage '{current_stage_name}'. Direct next stage: {stage_spec.next_stage}")
                return stage_spec.next_stage
            # If neither conditional nor direct next_stage is present, this path ends successfully.
            else:
                self.logger.info(f"Stage '{current_stage_name}' has no defined next stage (no condition, next_stage, next_stage_true/false). Ending this execution path as successful.")
                return NEXT_STAGE_END_SUCCESS
        else:
            # Stage name not found in the plan, this is a critical plan execution error.
            self.logger.error(f"Stage '{current_stage_name}' not found in current plan '{self.current_plan.id if self.current_plan else 'Unknown Plan'}'. Terminating path.")
            return NEXT_STAGE_END_FAILURE

    async def _handle_stage_error(
        self, 
        current_stage_name: str, 
        stage_spec: MasterStageSpec, 
        agent_id_for_error: str, 
        e: Any, # Can be Exception or AgentErrorDetails
        attempt_number: int,
        max_retries_for_stage: int
        # context: Dict[str, Any] parameter removed from here
    ) -> tuple[Optional[str], Optional[AgentErrorDetails], bool, Optional[FlowPauseStatus]]: 
        """
        Handles errors that occur during stage execution, including retries and escalations.
        Returns a tuple: (next_stage_name, error_details, should_retry_stage, pause_status_override)
        - next_stage_name: If error handling determines a specific next stage (e.g., from reviewer).
        - error_details: The populated AgentErrorDetails.
        - should_retry_stage: True if the current stage should be retried immediately.
        - pause_status_override: If the flow should pause due to this error handling.
        """
        agent_error_details: Optional[AgentErrorDetails]
        if isinstance(e, AgentErrorDetails):
            agent_error_details = e
        elif isinstance(e, Exception):
            agent_error_details = AgentErrorDetails(
                agent_id=agent_id_for_error,
                stage_name=current_stage_name,
                error_type=e.__class__.__name__,
                message=str(e),
                details=traceback.format_exc(), # Include traceback for better debugging
                can_retry=False, # Default, can be overridden by specific error types or stage spec
                can_escalate=True # Default, most errors can be escalated
            )
        else: # Should not ideally happen if 'e' is properly typed Exception or AgentErrorDetails
            agent_error_details = AgentErrorDetails(
                agent_id=agent_id_for_error, 
                stage_name=current_stage_name, 
                error_type="UnknownErrorTypeInHandling", 
                message=str(e),
                can_retry=False,
                can_escalate=True
            )
        
        # --- Retry Logic --- 
        should_retry_now = False
        # Check if error itself is retryable AND stage allows retry AND attempts not exhausted
        if agent_error_details.can_retry and attempt_number < max_retries_for_stage:
            if stage_spec.on_failure in [OnFailureAction.RETRY_THEN_ESCALATE, OnFailureAction.RETRY_THEN_FAIL]:
                should_retry_now = True
        
        if should_retry_now:
            self.logger.info(f"Stage '{current_stage_name}' (agent: {agent_id_for_error}) failed. Retrying (attempt {attempt_number + 1}/{max_retries_for_stage}). Error: {agent_error_details.message}")
            self._emit_metric(
                MetricEventType.STAGE_RETRY,
                flow_id=self.current_plan.id if self.current_plan else "UNKNOWN_FLOW",
                run_id=self._current_run_id or "unknown_run",
                stage_id=current_stage_name,
                master_stage_id=current_stage_name, 
                agent_id=agent_id_for_error,
                data={"attempt": attempt_number + 1, "max_retries": max_retries_for_stage, "error": agent_error_details.message}
            )
            return current_stage_name, agent_error_details, True, None # (next_stage=current, error, retry=True, pause=None)

        # --- Escalation/Reviewer Logic (if not retrying or retries exhausted) ---
        effective_on_failure = stage_spec.on_failure
        # If retries were specified but exhausted, adjust effective_on_failure for escalation/failure
        if stage_spec.on_failure == OnFailureAction.RETRY_THEN_ESCALATE and not should_retry_now:
            effective_on_failure = OnFailureAction.ESCALATE_TO_REVIEWER
        elif stage_spec.on_failure == OnFailureAction.RETRY_THEN_FAIL and not should_retry_now:
            effective_on_failure = OnFailureAction.FAIL_PIPELINE

        # Check if escalation to reviewer is appropriate
        should_escalate = False
        if effective_on_failure == OnFailureAction.ESCALATE_TO_REVIEWER:
            should_escalate = True
        elif agent_error_details.can_escalate and effective_on_failure not in [OnFailureAction.FAIL_PIPELINE, OnFailureAction.CONTINUE_WITH_ERROR]:
            # If error is generically escalatable and stage doesn't explicitly forbid it or demand immediate failure/continuation
            should_escalate = True 

        if should_escalate:
            self.logger.info(f"Escalating error in stage '{current_stage_name}' (agent: {agent_id_for_error}, Run ID: {self._current_run_id}) to reviewer.")
            
            # Construct reviewer_invocation_context from self.shared_context
            # Ensure deep copies for mutable structures passed to reviewer to prevent unintended side-effects.
            reviewer_invocation_context = {
                "project_id": self.shared_context.project_id,
                "current_run_id": self._current_run_id or "unknown_run",
                "current_flow_id": self.current_plan.id if self.current_plan else "UNKNOWN_FLOW",
                "failed_stage_name": current_stage_name,
                "failed_agent_id": agent_id_for_error,
                "error_details_json": agent_error_details.model_dump_json(indent=2) if agent_error_details else None,
                "current_artifact_references": copy.deepcopy(self.shared_context.artifact_references),
                "recent_stage_outputs": { 
                    k: copy.deepcopy(v) for k, v in self.shared_context.previous_stage_outputs.items() 
                    if k not in ["_initial_context_", "_initial_run_inputs_"] # Avoid sending potentially huge initial context dicts
                },
                "current_scratchpad": copy.deepcopy(self.shared_context.scratchpad),
                "current_stage_spec_json": stage_spec.model_dump_json(indent=2) # Provide spec of failed stage
            }

            reviewer_suggestion = await self._invoke_reviewer_and_get_suggestion(
                run_id=self._current_run_id or "unknown_run", 
                flow_id=self.current_plan.id if self.current_plan else "unknown_flow",
                current_stage_name=current_stage_name,
                agent_error_details=agent_error_details, 
                current_context=reviewer_invocation_context # Pass the constructed context
            )
            
            if reviewer_suggestion and reviewer_suggestion.action: # Ensure action is present
                action = reviewer_suggestion.action
                self.logger.info(f"Reviewer responded for stage '{current_stage_name}' with action: {action.action_type.value}")
                # Emit metric for reviewer action taken
                self._emit_metric(
                    MetricEventType.REVIEWER_ACTION_TAKEN,
                    flow_id=self.current_plan.id if self.current_plan else "UNKNOWN_FLOW",
                    run_id=self._current_run_id or "unknown_run",
                    stage_id=current_stage_name,
                    data={"reviewer_action_type": action.action_type.value, "details": action.model_dump()}
                )

                if action.action_type == ReviewerActionType.RETRY_STAGE_WITH_CHANGES:
                    # TODO: Implement application of changes from action.details (e.g., modify inputs_spec)
                    self.logger.info(f"Reviewer suggested to retry stage '{current_stage_name}' (potentially with changes). Proceeding with retry.")
                    return current_stage_name, agent_error_details, True, None 
                
                elif action.action_type == ReviewerActionType.PAUSE_FOR_HUMAN_CLARIFICATION:
                    self.logger.info(f"Reviewer suggested to pause for human clarification for stage '{current_stage_name}'.")
                    # Details for pause (e.g., clarification prompt) should be in action.details
                    # The main loop will use this pause_status to save state correctly.
                    return None, agent_error_details, False, FlowPauseStatus.PAUSED_BY_REVIEWER 
                
                elif action.action_type == ReviewerActionType.FAIL_PIPELINE:
                    self.logger.info(f"Reviewer explicitly suggested to fail the pipeline after stage '{current_stage_name}'.")
                    return None, agent_error_details, False, None 
                
                elif action.action_type == ReviewerActionType.CONTINUE_IGNORE_ERROR:
                     self.logger.info(f"Reviewer suggested to continue and ignore error for stage '{current_stage_name}'. Trying to get next normal stage.")
                     # Attempt to get the normal next stage as if the current one succeeded ignoring error.
                     # This means _handle_stage_error signals that the error is "resolved" by ignoring it.
                     next_stage_after_ignored_error = self._get_next_stage(current_stage_name)
                     # If getting next stage itself results in an end/failure, propagate that.
                     if next_stage_after_ignored_error in [NEXT_STAGE_END_FAILURE, NEXT_STAGE_END_SUCCESS, None]:
                         # If it's a definitive end or failure from _get_next_stage, or None (which might imply an issue or actual end)
                         # We return None as next stage, effectively ending this path or failing if END_FAILURE.
                         # The main loop will then process this (e.g. if None, might go to END_SUCCESS if no error set, or END_FAILURE).
                         # Let's ensure if _get_next_stage returns None it means error or actual end.
                         # NEXT_STAGE_END_SUCCESS from _get_next_stage here means the flow successfully ends by ignoring error.
                         final_next_stage = next_stage_after_ignored_error if next_stage_after_ignored_error != NEXT_STAGE_END_FAILURE else None
                         if final_next_stage == NEXT_STAGE_END_SUCCESS and stage_spec.next_stage is None and stage_spec.condition is None:
                             pass # This is a valid end of flow
                         elif final_next_stage is None and next_stage_after_ignored_error == NEXT_STAGE_END_FAILURE:
                             self.logger.warning(f"Could not determine a valid next stage after ignoring error for '{current_stage_name}'. Failing.")
                         return final_next_stage, agent_error_details, False, None
                     else: # A specific next stage was found
                        return next_stage_after_ignored_error, agent_error_details, False, None
                
                elif action.action_type == ReviewerActionType.MODIFY_MASTER_PLAN:
                    self.logger.info(f"Reviewer suggested to modify the master plan for run {self._current_run_id}.")
                    # This is a complex operation. The orchestrator needs to apply these plan changes.
                    # For now, we will pause the flow, indicating plan modification is required.
                    # The actual modification and re-triggering would be an external process or a more advanced orchestrator feature.
                    # The details of modification are in action.details (e.g., ReviewerModifyPlanAction)
                    # PauseStatus.PAUSED_FOR_PLAN_MODIFICATION could be a new status.
                    self.logger.warning("Plan modification by reviewer is an advanced feature. Pausing flow for manual intervention based on reviewer's plan modification details.")
                    # Store the reviewer's suggested plan modification details in shared_context.scratchpad for inspection
                    if action.details:
                        self.shared_context.set_scratchpad_data("reviewer_suggested_plan_modification", action.details.model_dump())
                    return None, agent_error_details, False, FlowPauseStatus.PAUSED_FOR_PLAN_MODIFICATION 
                
                else:
                    self.logger.warning(f"Reviewer returned unhandled action type: {action.action_type.value}. Defaulting to failing the stage.")
                    return None, agent_error_details, False, None # Default to fail if action unhandled
            else:
                self.logger.warning(f"Reviewer agent '{self.master_planner_reviewer_agent_id}' did not return a suggestion or action for stage '{current_stage_name}'. Defaulting to failing the stage.")
                return None, agent_error_details, False, None # Default to fail if no suggestion

        # --- Final default action if not retried and not escalated (or escalation didn't lead to a resolution) ---
        # This path is taken if on_failure is FAIL_PIPELINE, or if escalation was not triggered/successful.
        self.logger.error(f"Stage '{current_stage_name}' (agent: {agent_id_for_error}, Run ID: {self._current_run_id}) failed permanently after {attempt_number} attempts or due to policy. Error: {agent_error_details.message if agent_error_details else str(e)}")
        self._emit_metric(
            MetricEventType.STAGE_ERROR,
            flow_id=self.current_plan.id if self.current_plan else "UNKNOWN_FLOW",
            run_id=self._current_run_id or "unknown_run",
            stage_id=current_stage_name,
            master_stage_id=current_stage_name, 
            agent_id=agent_id_for_error,
            data=agent_error_details.model_dump() if agent_error_details else {"error": str(e), "on_failure_policy": str(effective_on_failure)}
        )
        # Signal to the main loop that this stage failed definitively. 
        # The main loop will then terminate the flow with a failure status.
        return None, agent_error_details, False, None 

    async def _invoke_reviewer_and_get_suggestion(
        self,
        run_id: str,
        flow_id: str,
        current_stage_name: str,
        agent_error_details: Optional[AgentErrorDetails],
        current_context: Dict[str, Any]
    ) -> Optional[MasterPlannerReviewerOutput]:
        # This method should be implemented to invoke the reviewer agent and get a suggestion
        # For now, we'll just return a placeholder response
        return MasterPlannerReviewerOutput(
            action=ReviewerAction(action_type=ReviewerActionType.CONTINUE_IGNORE_ERROR)
        )

    async def _invoke_agent_for_stage(
        self,
        stage_name: str,
        agent_id: str, 
        agent_callable: Callable, 
        inputs_spec: Optional[Dict[str, Any]],
        max_retries: int
    ) -> Any: # Returns agent_output or raises an error
        
        # self.logger.debug(f"_invoke_agent_for_stage: stage='{stage_name}', agent_id='{agent_id}', inputs_spec={inputs_spec}")
        final_inputs = {}
        if inputs_spec:
            # _resolve_input_values now uses self.shared_context, self.config, self.state_manager internally
            final_inputs = self._resolve_input_values(inputs_spec)
        
        # ... rest of _invoke_agent_for_stage method ...
        