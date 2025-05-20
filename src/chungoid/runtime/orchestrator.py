"""Minimal execution runtime skeleton (Phase-6 scaffold).

This module will gradually evolve into the production‐grade orchestrator that
executes Flow YAML graphs.  For now we expose a *SyncOrchestrator* that can run
very simple flows in-process so we can start adding tests and build up the API
surface incrementally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Callable, cast, ClassVar
from pathlib import Path 
import datetime as _dt
from datetime import datetime, timezone
import yaml
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from chungoid.utils.agent_resolver import AgentProvider, RegistryAgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.common_enums import StageStatus, FlowPauseStatus, OnFailureAction 
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec
from chungoid.schemas.flows import PausedRunDetails 
from chungoid.schemas.orchestration import SharedContext 

# Import for reviewer agent and its schemas
from chungoid.runtime.agents.system_master_planner_reviewer_agent import (
    MasterPlannerReviewerAgent
    # ReviewerAction was removed here
)
from chungoid.schemas.agent_master_planner_reviewer import (
    MasterPlannerReviewerInput,
    MasterPlannerReviewerOutput,
    ReviewerActionType, # Added ReviewerActionType import
    ReviewerModifyPlanAction,
    RetryStageWithChangesDetails,
    AddClarificationStageDetails,
    ModifyMasterPlanRemoveStageDetails,
    ModifyMasterPlanModifyStageDetails,
    ModifyMasterPlanDetails, 
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

# Import for MasterPlannerAgent and its input schema
from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent
from chungoid.schemas.agent_master_planner import MasterPlannerInput, MasterPlannerOutput # ADDED MasterPlannerOutput

# Import for PROJECT_CHUNGOID_DIR
from chungoid.constants import PROJECT_CHUNGOID_DIR

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

    MAX_HOPS = 100
    DEFAULT_AGENT_RETRIES = 1
    ARTIFACT_OUTPUT_KEY = "_mcp_generated_artifacts_relative_paths_"

    _current_run_id: Optional[str] = None
    _last_successful_stage_output: Optional[Any] = None
    _current_flow_config: Optional[Dict[str, Any]] = None
    shared_context: SharedContext

    COMPARATOR_MAP: ClassVar[Dict[str, Callable[[Any, Any], bool]]] = {
        ">=": lambda x, y: x >= y,
        "<=": lambda x, y: x <= y,
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
        ">": lambda x, y: x > y,
        "<": lambda x, y: x < y,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        agent_provider: AgentProvider,
        state_manager: StateManager,
        metrics_store: MetricsStore,
        master_planner_reviewer_agent_id: str = MasterPlannerReviewerAgent.AGENT_ID,
    ):
        self.config = config
        self.agent_provider = agent_provider
        self.state_manager = state_manager
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

        # --- Escalation/Reviewer Logic (if not retrying and retries exhausted) ---
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
            
            if reviewer_suggestion and reviewer_suggestion.suggestion_type: # Ensure suggestion_type is present
                action_type = reviewer_suggestion.suggestion_type # Use suggestion_type
                action_details = reviewer_suggestion.suggestion_details # Get details

                self.logger.info(f"Reviewer responded for stage '{current_stage_name}' with action type: {action_type.value}")
                # Emit metric for reviewer action taken
                self._emit_metric(
                    MetricEventType.REVIEWER_ACTION_TAKEN,
                    flow_id=self.current_plan.id if self.current_plan else "UNKNOWN_FLOW",
                    run_id=self._current_run_id or "unknown_run",
                    stage_id=current_stage_name,
                    data={"reviewer_action_type": action_type.value, "details": action_details.model_dump() if action_details and hasattr(action_details, 'model_dump') else (action_details if isinstance(action_details, dict) else None)}
                )

                if action_type == ReviewerActionType.RETRY_STAGE_WITH_CHANGES:
                    # TODO: Implement application of changes from action_details (e.g., modify inputs_spec)
                    self.logger.info(f"Reviewer suggested to retry stage '{current_stage_name}' (potentially with changes). Proceeding with retry.")
                    return current_stage_name, agent_error_details, True, None 
                
                elif action_type == ReviewerActionType.PAUSE_FOR_HUMAN_CLARIFICATION:
                    self.logger.info(f"Reviewer suggested to pause for human clarification for stage '{current_stage_name}'.")
                    # Details for pause (e.g., clarification prompt) should be in action_details
                    # The main loop will use this pause_status to save state correctly.
                    return None, agent_error_details, False, FlowPauseStatus.PAUSED_BY_REVIEWER 
                
                elif action_type == ReviewerActionType.FAIL_PIPELINE:
                    self.logger.info(f"Reviewer explicitly suggested to fail the pipeline after stage '{current_stage_name}'.")
                    return None, agent_error_details, False, None 
                
                elif action_type == ReviewerActionType.CONTINUE_IGNORE_ERROR:
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
                
                elif action_type == ReviewerActionType.MODIFY_MASTER_PLAN:
                    self.logger.info(f"Reviewer suggested to modify the master plan for run {self._current_run_id}.")
                    # This is a complex operation. The orchestrator needs to apply these plan changes.
                    # For now, we will pause the flow, indicating plan modification is required.
                    # The actual modification and re-triggering would be an external process or a more advanced orchestrator feature.
                    # The details of modification are in action_details (e.g., ReviewerModifyPlanAction)
                    # PauseStatus.PAUSED_FOR_PLAN_MODIFICATION could be a new status.
                    self.logger.warning("Plan modification by reviewer is an advanced feature. Pausing flow for manual intervention based on reviewer's plan modification details.")
                    # Store the reviewer's suggested plan modification details in shared_context.scratchpad for inspection
                    if action_details:
                         # Ensure action_details is serializable. If it's a Pydantic model, model_dump() is good.
                        details_to_store = action_details.model_dump() if hasattr(action_details, 'model_dump') else action_details
                        self.shared_context.set_scratchpad_data("reviewer_suggested_plan_modification", details_to_store)
                    return None, agent_error_details, False, FlowPauseStatus.PAUSED_FOR_PLAN_MODIFICATION 
                
                else:
                    self.logger.warning(f"Reviewer returned unhandled action type: {action_type.value}. Defaulting to failing the stage.")
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
            suggestion_type=ReviewerActionType.CONTINUE_IGNORE_ERROR 
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
        
    async def run(
        self,
        goal_str: Optional[str] = None,
        flow_yaml_path: Optional[str] = None,
        master_plan_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        run_id_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs a master execution plan or generates one from a goal.

        Args:
            goal_str: The user's high-level goal to generate a plan.
            flow_yaml_path: Path to a YAML file defining the MasterExecutionPlan.
            master_plan_id: ID of an existing MasterExecutionPlan to load.
            initial_context: Initial key-value pairs for the shared context.
            run_id_override: Specific run ID to use for this execution.

        Returns:
            A dictionary containing the final shared context outputs.
        """
        current_run_id = run_id_override or f"run_{uuid.uuid4().hex[:16]}"
        self._current_run_id = current_run_id
        self.logger.info(f"Starting AsyncOrchestrator.run with ID: {current_run_id}")

        # Initialize or update shared_context for this run
        # project_id and project_root_path are set in __init__
        # We need to reset/initialize run-specific parts of shared_context
        self.shared_context.run_id = current_run_id
        self.shared_context.flow_id = None # Will be set after plan is loaded/generated
        self.shared_context.initial_inputs = initial_context or {}
        self.shared_context.previous_stage_outputs = {"_initial_context_": self.shared_context.initial_inputs.copy()}
        self.shared_context.artifact_references = {}
        self.shared_context.scratchpad = {}
        self.shared_context.current_stage_id = None
        self.shared_context.current_stage_status = None

        loaded_plan: Optional[MasterExecutionPlan] = None
        plan_source_description: str = "Unknown"

        try:
            if goal_str:
                plan_source_description = f"goal: '{goal_str[:100]}...'"
                self.logger.info(f"Generating MasterExecutionPlan from goal for run {current_run_id}.")
                # Ensure shared_context has initial inputs if any were passed for the planner
                # This might involve merging initial_context into a specific structure for MasterPlannerInput.
                # For now, assuming MasterPlannerInput takes user_goal and project_id directly.

                planner_agent_id = "SystemMasterPlannerAgent_v1" # Or get from config
                
                # Construct input for the MasterPlannerAgent
                try:
                    # agent_callable, _ = self.agent_provider.get(planner_agent_id) # Error if not found
                    agent_callable = self.agent_provider.get(planner_agent_id) # CORRECTED
                    if not agent_callable:
                        self.logger.error(f"Could not resolve agent: {planner_agent_id}")
                    # ... rest of the code ...
                except Exception as e:
                    self.logger.error(f"Error resolving agent for planner: {e}", exc_info=True)
                    return {"_flow_error": f"Failed to resolve agent for planner: {e}"}
                
                # Ensure project_id is available in shared_context or config
                # project_id_for_planner = self.shared_context.project_id # Assuming it's set
                # Fallback if not in shared_context (should be guaranteed by CLI path)
                project_id_for_planner = self.config.get("project_id", "unknown_project")
                if project_id_for_planner == "unknown_project":
                    self.logger.warning("Project ID for planner is unknown, this might cause issues.")


                planner_input = MasterPlannerInput(user_goal=goal_str, project_id=project_id_for_planner) # Use validated schema
                
                # Emit metric for planner invocation
                self._emit_metric(MetricEventType.AGENT_CALL_START, "N/A", current_run_id, agent_id=planner_agent_id, data={"input_summary": planner_input.model_dump(mode='json')}) # Use model_dump for Pydantic v2

                planner_output: Optional[MasterPlannerOutput] = None
                planner_error_details: Optional[AgentErrorDetails] = None

                try:
                    agent_callable = self.agent_provider.get(planner_agent_id) # MODIFIED
                    
                    self.logger.info(f"Invoking MasterPlannerAgent (ID: {planner_agent_id}) with input: {planner_input.model_dump(mode='json')}")
                    
                    # Pass the full_context (self.shared_context) when invoking agents
                    raw_planner_result = await agent_callable(inputs=planner_input, full_context=self.shared_context) # MODIFIED: Pass planner_input object directly
                    
                    if isinstance(raw_planner_result, MasterPlannerOutput):
                        planner_output = raw_planner_result
                    elif isinstance(raw_planner_result, dict): # Handle if agent returns dict instead of schema
                        try:
                            planner_output = MasterPlannerOutput(**raw_planner_result)
                        except Exception as e_dict_parse:
                            self.logger.error(f"Failed to parse MasterPlannerAgent dict output into MasterPlannerOutput: {e_dict_parse}", exc_info=True)
                            planner_error_details = AgentErrorDetails(error_type="OutputParsingError", message=str(e_dict_parse))
                    else: # Unexpected output type
                        self.logger.error(f"MasterPlannerAgent returned unexpected raw type: {type(raw_planner_result)}")
                        planner_error_details = AgentErrorDetails(error_type="UnexpectedOutputType", message=f"Expected MasterPlannerOutput or dict, got {type(raw_planner_result)}")

                except Exception as e:
                    self.logger.error(f"Error invoking MasterPlannerAgent: {e}", exc_info=True)
                    planner_error_details = AgentErrorDetails(error_type="InvocationError", message=str(e), traceback=str(traceback.format_exc()))
                
                self._emit_metric(MetricEventType.AGENT_CALL_END, "N/A", current_run_id, agent_id=planner_agent_id, data={"output_type": str(type(planner_output)), "error": planner_error_details.model_dump(mode='json') if planner_error_details else None})

                master_plan: Optional[MasterExecutionPlan] = None # Ensure master_plan is defined in this scope

                if planner_output:
                    self.logger.info(f"ORCHESTRATOR_DEBUG: Type of planner_output: {type(planner_output)}")
                    if hasattr(planner_output, 'master_plan_json') and planner_output.master_plan_json:
                        self.logger.info("ORCHESTRATOR_DEBUG: planner_output.master_plan_json exists.")
                        self.logger.info(f"ORCHESTRATOR_DEBUG: Type of planner_output.master_plan_json: {type(planner_output.master_plan_json)}")
                        plan_json_str = planner_output.master_plan_json
                        self.logger.info(f"ORCHESTRATOR_DEBUG: Value of planner_output.master_plan_json (first 200 chars): {str(plan_json_str)[:200]}")
                        self.logger.info(f"ORCHESTRATOR_DEBUG: Attempting to parse plan_json_str (len: {len(plan_json_str)}).")
                        try:
                            plan_dict = json.loads(plan_json_str)
                            self.logger.info("ORCHESTRATOR_DEBUG: Successfully loaded plan_json_str into plan_dict.")
                            # Add project_id to plan_dict if not present, from orchestrator's config
                            if 'project_id' not in plan_dict and 'project_id' in self.config:
                                plan_dict['project_id'] = self.config['project_id']
                            master_plan = MasterExecutionPlan(**plan_dict)
                            self.logger.info(f"ORCHESTRATOR_DEBUG: Successfully parsed plan_dict into MasterExecutionPlan. Plan ID: {master_plan.id}")
                        except json.JSONDecodeError as e_json:
                            self.logger.error(f"ORCHESTRATOR_ERROR: JSONDecodeError parsing plan_json_str: {e_json}", exc_info=True)
                            self.logger.error(f"ORCHESTRATOR_ERROR_DETAIL: plan_json_str (first 1000 chars): {plan_json_str[:1000]}")
                            master_plan = None
                        except Exception as e_gen_parse: 
                            self.logger.error(f"ORCHESTRATOR_ERROR: Generic exception parsing plan_json_str or instantiating MasterExecutionPlan: {e_gen_parse}", exc_info=True)
                            self.logger.error(f"ORCHESTRATOR_ERROR_DETAIL: plan_json_str (first 1000 chars): {plan_json_str[:1000]}")
                            master_plan = None 
                    else:
                        self.logger.warning("ORCHESTRATOR_WARN: planner_output.master_plan_json was empty.")
                        master_plan = None 
                else:
                    self.logger.error(f"ORCHESTRATOR_ERROR: MasterPlannerAgent returned an unexpected type: {type(planner_output)}. Expected MasterExecutionPlan, dict, or object with master_plan_json string.", exc_info=True)
                    self._emit_metric(MetricEventType.ORCHESTRATOR_ERROR, "UNKNOWN_FLOW", current_run_id, data={"error": f"MasterPlannerAgent returned unexpected type: {type(planner_output)}"})
                    master_plan = None 

                if master_plan: 
                    self.logger.info(f"ORCHESTRATOR_INFO: Successfully obtained master_plan (ID: {master_plan.id}) from goal_str path.")
                    self.shared_context.flow_id = master_plan.id 
                    loaded_plan = master_plan 
                    plan_source_description = f"goal: '{goal_str[:100]}...' (generated plan ID: {master_plan.id})" 
                else:
                    self.logger.warning(f"ORCHESTRATOR_WARN: master_plan is None after processing planner_output from goal_str. Plan generation or parsing likely failed.")
                
                # Removed the problematic early return that was here.
                # The logic will now fall through to the checks for loaded_plan below.

            elif flow_yaml_path:
                plan_source_description = f"yaml: {flow_yaml_path}"
                self.logger.info(f"Loading MasterExecutionPlan from YAML: {flow_yaml_path} for run {current_run_id}")
                try:
                    loaded_plan = MasterExecutionPlan.from_yaml_file(flow_yaml_path)
                    self.shared_context.flow_id = loaded_plan.id
                    # self.state_manager.save_master_plan(loaded_plan) # Agent should save its own plan if it generates it. If loaded from file, it's already "saved".
                except Exception as e:
                    self.logger.error(f"Failed to load MasterExecutionPlan from YAML '{flow_yaml_path}': {e}", exc_info=True)
                    self._emit_metric(MetricEventType.ORCHESTRATOR_ERROR, "UNKNOWN_FLOW", current_run_id, data={"error": f"Failed to load plan YAML: {e}"})
                    return {"_flow_error": f"Failed to load MasterExecutionPlan from YAML: {e}"}
            
            elif master_plan_id:
                plan_source_description = f"id: {master_plan_id}"
                self.logger.info(f"Loading MasterExecutionPlan by ID: {master_plan_id} for run {current_run_id}")
                try:
                    # Attempt to load from ProjectChromaManagerAgent first
                    project_chroma_manager: Optional[ProjectChromaManagerAgent_v1] = self.agent_provider.get_project_chroma_manager() # type: ignore
                    if project_chroma_manager:
                        retrieved_artifact = await project_chroma_manager.retrieve_artifact(
                            base_collection_name=EXECUTION_PLANS_COLLECTION, # Ensure this constant is available
                            document_id=master_plan_id
                        )
                        if retrieved_artifact.status == "SUCCESS" and isinstance(retrieved_artifact.content, dict):
                            loaded_plan = MasterExecutionPlan(**retrieved_artifact.content)
                            self.shared_context.flow_id = loaded_plan.id
                            self.logger.info(f"Successfully loaded MasterExecutionPlan '{master_plan_id}' from PCMA.")
                        elif retrieved_artifact.status == "SUCCESS" and isinstance(retrieved_artifact.content, str): # If content is string (JSON)
                            try:
                                plan_dict_pcma = json.loads(retrieved_artifact.content)
                                loaded_plan = MasterExecutionPlan(**plan_dict_pcma)
                                self.shared_context.flow_id = loaded_plan.id
                                self.logger.info(f"Successfully loaded and parsed MasterExecutionPlan '{master_plan_id}' (string content) from PCMA.")
                            except json.JSONDecodeError as e_json_pcma:
                                self.logger.error(f"Failed to parse MasterExecutionPlan JSON string from PCMA for ID '{master_plan_id}': {e_json_pcma}")
                        else:
                            self.logger.warning(f"MasterExecutionPlan ID '{master_plan_id}' not found or content invalid in PCMA ({EXECUTION_PLANS_COLLECTION}). Status: {retrieved_artifact.status}. Trying StateManager.")
                    
                    if not loaded_plan: # Fallback to StateManager if not found in PCMA or PCMA not available
                        loaded_plan = self.state_manager.load_master_plan(master_plan_id)
                        if loaded_plan:
                            self.shared_context.flow_id = loaded_plan.id
                            self.logger.info(f"Successfully loaded MasterExecutionPlan '{master_plan_id}' from StateManager.")
                        else:
                             self.logger.error(f"MasterExecutionPlan ID '{master_plan_id}' not found in StateManager after PCMA attempt.")

                except Exception as e:
                    self.logger.error(f"Failed to load MasterExecutionPlan by ID '{master_plan_id}': {e}", exc_info=True)
            
            # This is the consolidated check: if after all attempts, loaded_plan is None.
            if not loaded_plan:
                error_msg = f"No valid MasterExecutionPlan could be determined from goal, YAML, or ID. Plan source description: {plan_source_description}"
                self.logger.error(error_msg)
                self._emit_metric(MetricEventType.ORCHESTRATOR_ERROR, self.shared_context.flow_id or "UNKNOWN_FLOW", current_run_id, data={"error": error_msg})
                return {"_flow_error": error_msg}


            # If we've reached here, loaded_plan should be valid.
            # The original check 'if not loaded_plan or not self.shared_context.flow_id:' is still good.
            if not self.shared_context.flow_id: # flow_id should have been set if loaded_plan is valid
                 warn_msg = "Shared context flow_id is not set despite having a loaded_plan. This is unexpected."
                 self.logger.warning(warn_msg)
                 # Attempt to set it from loaded_plan if possible
                 if hasattr(loaded_plan, 'id') and loaded_plan.id:
                     self.shared_context.flow_id = loaded_plan.id
                     self.logger.info(f"Fallback: Set shared_context.flow_id to loaded_plan.id: {loaded_plan.id}")
                 else: # If plan has no ID, this is a critical issue with the plan itself
                     critical_error_msg = "Loaded plan has no ID, cannot proceed."
                     self.logger.error(critical_error_msg)
                     self._emit_metric(MetricEventType.ORCHESTRATOR_ERROR, "UNKNOWN_FLOW_NO_ID", current_run_id, data={"error": critical_error_msg})
                     return {"_flow_error": critical_error_msg}


            self.current_plan = loaded_plan
            start_stage_name = self.current_plan.start_stage

            self._emit_metric(MetricEventType.FLOW_START, self.shared_context.flow_id, current_run_id, data={"plan_source": plan_source_description, "initial_context_keys": list(initial_context.keys()) if initial_context else []})
            self.state_manager.record_flow_start(current_run_id, self.shared_context.flow_id, initial_context or {})
            
            # Execute the flow starting from the identified start_stage
            final_outputs = await self._execute_flow_loop(
                run_id=current_run_id,
                flow_id=self.shared_context.flow_id,
                start_stage_name=start_stage_name
            )
            return final_outputs

        except Exception as e_outer:
            self.logger.exception(f"Unhandled exception during AsyncOrchestrator.run (run_id: {current_run_id}): {e_outer}")
            flow_id_for_metric = self.shared_context.flow_id or (loaded_plan.id if loaded_plan else "UNKNOWN_FLOW")
            self._emit_metric(MetricEventType.FLOW_END, flow_id_for_metric, current_run_id, data={"status": StageStatus.FAILURE.value, "error": str(e_outer)}) # Changed to StageStatus.FAILURE
            self.state_manager.record_flow_end(current_run_id, flow_id_for_metric, final_status=StageStatus.FAILURE, error_message=str(e_outer)) # Changed to StageStatus.FAILURE
            return {"_flow_error": f"Unhandled orchestrator error: {e_outer}"}
        finally:
            self._current_run_id = None # Clear run ID after execution finishes or fails

    async def _execute_flow_loop(
        self,
        run_id: str,
        flow_id: str,
        start_stage_name: str,
        # initial_shared_context is now assumed to be populated on self.shared_context by the caller (run or resume_flow)
    ) -> Dict[str, Any]:
        """
        The main execution loop for a flow.
        Assumes self.current_plan and self.shared_context (for this run_id) are already set up.
        """
        self.logger.info(f"Entering _execute_flow_loop for run_id: {run_id}, flow_id: {flow_id}, start_stage: {start_stage_name}")
        current_stage_name: Optional[str] = start_stage_name
        hops = 0
        final_status: StageStatus = StageStatus.COMPLETED_SUCCESS # Optimistic default
        flow_error_details: Optional[str] = None

        while current_stage_name and \
              current_stage_name not in [NEXT_STAGE_END_SUCCESS, NEXT_STAGE_END_FAILURE] and \
              hops < self.MAX_HOPS:
            
            hops += 1
            if hops >= self.MAX_HOPS:
                self.logger.warning(f"Max hops ({self.MAX_HOPS}) reached for run {run_id}. Terminating flow with failure.")
                final_status = StageStatus.COMPLETED_FAILURE
                flow_error_details = "Max hops reached, possible infinite loop."
                break

            if not self.current_plan: # Should be set by caller
                self.logger.error(f"No current_plan loaded for run {run_id} in _execute_flow_loop. Critical error.")
                final_status = StageStatus.FAILURE # Changed to StageStatus.FAILURE
                flow_error_details = "Orchestrator critical error: current_plan not set."
                break

            stage_spec = self.current_plan.stages.get(current_stage_name)
            if not stage_spec:
                self.logger.error(f"Stage '{current_stage_name}' not found in plan '{flow_id}' for run {run_id}. Terminating.")
                final_status = StageStatus.FAILURE # Changed to StageStatus.FAILURE
                flow_error_details = f"Stage '{current_stage_name}' not found in plan."
                break

            self.shared_context.current_stage_id = current_stage_name
            self.shared_context.current_stage_status = StageStatus.RUNNING # Update shared context

            self.logger.info(f"Run {run_id}: Executing stage '{current_stage_name}' (Agent: {stage_spec.agent_id})")
            self._emit_metric(MetricEventType.MASTER_STAGE_START, flow_id, run_id, stage_id=current_stage_name, master_stage_id=current_stage_name, agent_id=stage_spec.agent_id)
            self.state_manager.record_stage_start(run_id, flow_id, current_stage_name, stage_spec.agent_id)

            agent_id_to_invoke = stage_spec.agent_category or stage_spec.agent_id
            if not agent_id_to_invoke: # Should be validated by MasterExecutionPlan model
                 self.logger.error(f"Stage '{current_stage_name}' in plan '{flow_id}' has neither agent_id nor agent_category. Terminating.")
                 final_status = StageStatus.FAILURE # Changed to StageStatus.FAILURE
                 flow_error_details = f"Stage '{current_stage_name}' misconfigured: missing agent_id/category."
                 break
            
            max_retries_for_stage = stage_spec.max_retries if stage_spec.max_retries is not None else self.DEFAULT_AGENT_RETRIES
            agent_error_obj: Optional[AgentErrorDetails] = None
            next_stage_after_error_handling: Optional[str] = None
            pause_status_override: Optional[FlowPauseStatus] = None
            
            try:
                # Resolve agent callable (this might raise NoAgentFoundForCategoryError, AmbiguousAgentCategoryError)
                # agent_callable, resolved_agent_id = await self.agent_provider.get_agent_callable(agent_id_to_invoke)
                agent_callable = self.agent_provider.get(agent_id_to_invoke) # Corrected method call
                resolved_agent_id = agent_id_to_invoke # Assuming agent_id_to_invoke is the resolved ID
                
                if stage_spec.agent_category and not stage_spec.agent_id: # If category was used, log resolved agent
                    self.logger.info(f"Stage '{current_stage_name}': Category '{stage_spec.agent_category}' resolved to agent ID '{resolved_agent_id}'.")
                
                agent_id_for_error_handling = resolved_agent_id # Use the specifically resolved agent_id

                # Attempt to execute the stage
                # _invoke_agent_for_stage now handles its own retries internally based on max_retries_for_stage
                stage_output = await self._invoke_agent_for_stage(
                    stage_name=current_stage_name,
                    agent_id=resolved_agent_id, # Pass the resolved agent ID
                    agent_callable=agent_callable,
                    inputs_spec=stage_spec.inputs,
                    max_retries=max_retries_for_stage # Pass max_retries for internal retry loop
                )
                # If _invoke_agent_for_stage completes without raising, it means success after any internal retries.

                # Update shared context with outputs
                output_key = stage_spec.output_context_path or current_stage_name
                self.shared_context.previous_stage_outputs[output_key] = stage_output
                self._last_successful_stage_output = stage_output # Update for potential use by next stages

                # Handle artifact registration from stage_output if necessary
                if isinstance(stage_output, dict) and self.ARTIFACT_OUTPUT_KEY in stage_output:
                    artifact_paths = stage_output[self.ARTIFACT_OUTPUT_KEY]
                    if isinstance(artifact_paths, list):
                        for p_str in artifact_paths:
                            self.shared_context.register_artifact(current_stage_name, Path(p_str))
                    elif isinstance(artifact_paths, str):
                         self.shared_context.register_artifact(current_stage_name, Path(artifact_paths))
                
                # Check success criteria
                criteria_passed, failed_criteria = await self._check_success_criteria(current_stage_name, stage_spec, self.shared_context.previous_stage_outputs)
                if not criteria_passed:
                    self.logger.warning(f"Stage '{current_stage_name}' failed success criteria: {failed_criteria}. Triggering error handling.")
                    # Create an AgentErrorDetails for success criteria failure
                    agent_error_obj = AgentErrorDetails(
                        agent_id=resolved_agent_id,
                        stage_name=current_stage_name,
                        error_type="SuccessCriteriaFailed",
                        message=f"Stage failed success criteria: {', '.join(failed_criteria)}",
                        details={"failed_criteria": failed_criteria},
                        can_retry=False, # Typically, success criteria failure is not directly retryable by the same agent
                        can_escalate=True
                    )
                    # Call _handle_stage_error. attempt_number is effectively > max_retries to prevent direct retry of invoke
                    next_stage_after_error_handling, agent_error_obj, _, pause_status_override = await self._handle_stage_error(
                        current_stage_name, stage_spec, resolved_agent_id, agent_error_obj, 
                        attempt_number=max_retries_for_stage + 1, # Ensure it doesn't retry via _invoke_agent's loop
                        max_retries_for_stage=max_retries_for_stage
                    )
                    if pause_status_override: # If error handling led to a pause
                        # Save paused state and break loop
                        self.state_manager.save_paused_run(run_id, flow_id, current_stage_name, pause_status_override, self.shared_context.model_dump())
                        self.logger.info(f"Run {run_id} paused at stage '{current_stage_name}' due to error handling result: {pause_status_override.value}")
                        return self.shared_context.previous_stage_outputs # Or specific pause info
                    
                    if next_stage_after_error_handling is None and agent_error_obj: # Error handling decided to fail the stage/pipeline
                        final_status = StageStatus.FAILURE # Changed to StageStatus.FAILURE
                        flow_error_details = agent_error_obj.message
                        current_stage_name = NEXT_STAGE_END_FAILURE # Break loop

                if current_stage_name != NEXT_STAGE_END_FAILURE : # If not already failed by criteria
                    self._emit_metric(MetricEventType.MASTER_STAGE_END, flow_id, run_id, stage_id=current_stage_name, master_stage_id=current_stage_name, agent_id=resolved_agent_id, data={"status": StageStatus.COMPLETED_SUCCESS.value})
                    self.state_manager.record_stage_end(run_id, flow_id, current_stage_name, StageStatus.COMPLETED_SUCCESS, outputs=stage_output)
                    self.shared_context.current_stage_status = StageStatus.COMPLETED_SUCCESS
                    current_stage_name = self._get_next_stage(current_stage_name) # Determine next stage

            except (NoAgentFoundForCategoryError, AmbiguousAgentCategoryError) as e_agent_resolve:
                self.logger.error(f"Run {run_id}: Agent resolution failed for stage '{current_stage_name}', agent specifier '{agent_id_to_invoke}': {e_agent_resolve}")
                # This is a structural/configuration error for the stage.
                # It's not an agent execution error, so attempt_number is 1.
                agent_error_obj = AgentErrorDetails(
                    agent_id=agent_id_to_invoke, # This was the specifier
                    stage_name=current_stage_name,
                    error_type=e_agent_resolve.__class__.__name__,
                    message=str(e_agent_resolve),
                    can_retry=False, # Retrying won't help if agent can't be resolved
                    can_escalate=True # Reviewer might suggest fixing plan
                )
                next_stage_after_error_handling, agent_error_obj, _, pause_status_override = await self._handle_stage_error(
                    current_stage_name, stage_spec, agent_id_to_invoke, agent_error_obj, 1, 1 # No retries for resolution failure
                )

            except Exception as e_invoke: # Catch errors from _invoke_agent_for_stage (e.g., AgentErrorDetails if all retries failed)
                self.logger.error(f"Run {run_id}: Exception during agent invocation for stage '{current_stage_name}': {e_invoke}", exc_info=True)
                # _invoke_agent_for_stage should have already tried retries and would raise AgentErrorDetails if it failed permanently
                # If it's another unexpected error, wrap it.
                current_agent_id_for_err = stage_spec.agent_id or stage_spec.agent_category or "UNKNOWN_AGENT"
                
                # Error handling is now primarily managed by _invoke_agent_for_stage which raises AgentErrorDetails on final failure
                # or if a non-retryable error occurs.
                # The _handle_stage_error here is for post-invocation issues or if _invoke_agent itself has an unhandled exception (less likely).
                # For now, assume e_invoke is the AgentErrorDetails from _invoke_agent_for_stage if it failed after retries.
                next_stage_after_error_handling, agent_error_obj, _, pause_status_override = await self._handle_stage_error(
                    current_stage_name, stage_spec, current_agent_id_for_err, e_invoke, 
                    attempt_number=max_retries_for_stage +1, # Signify that agent invocation retries (if any) are done
                    max_retries_for_stage=max_retries_for_stage
                )
            
            # Post-invocation error handling (if any error occurred and was processed by _handle_stage_error)
            if agent_error_obj: # This means an error occurred (either during invoke, or success criteria, or agent resolution)
                self.shared_context.current_stage_status = StageStatus.FAILURE # Changed to StageStatus.FAILURE
                self._emit_metric(MetricEventType.MASTER_STAGE_END, flow_id, run_id, stage_id=current_stage_name, master_stage_id=current_stage_name, agent_id=agent_error_obj.agent_id, data={"status": StageStatus.FAILURE.value, "error": agent_error_obj.message}) # Changed to StageStatus.FAILURE
                self.state_manager.record_stage_end(run_id, flow_id, current_stage_name, StageStatus.FAILURE, error_details=agent_error_obj.model_dump()) # Changed to StageStatus.FAILURE

                if pause_status_override:
                    self.state_manager.save_paused_run(run_id, flow_id, current_stage_name, pause_status_override, self.shared_context.model_dump())
                    self.logger.info(f"Run {run_id} paused at stage '{current_stage_name}' due to error handling result: {pause_status_override.value}")
                    return self.shared_context.previous_stage_outputs # Or specific pause info
                
                if next_stage_after_error_handling is None: # Error handling decided to fail the stage/pipeline
                    final_status = StageStatus.FAILURE # Changed to StageStatus.FAILURE
                    flow_error_details = agent_error_obj.message if agent_error_obj else "Unknown error after handling."
                    current_stage_name = NEXT_STAGE_END_FAILURE # Break loop
                else: # Error handling provided a next stage (e.g., reviewer intervention, or continue_ignore_error)
                    current_stage_name = next_stage_after_error_handling
            
            # Check for user clarification checkpoint
            if stage_spec.clarification_checkpoint and current_stage_name not in [NEXT_STAGE_END_FAILURE, NEXT_STAGE_END_SUCCESS, None]:
                should_pause_for_clarification = True
                if stage_spec.clarification_checkpoint.condition:
                    should_pause_for_clarification = self._parse_condition(stage_spec.clarification_checkpoint.condition)
                
                if should_pause_for_clarification:
                    self.logger.info(f"Run {run_id}: Pausing at stage '{current_stage_name}' for user clarification. Prompt: {stage_spec.clarification_checkpoint.prompt}")
                    pause_details = PausedRunDetails(
                        run_id=run_id,
                        flow_id=flow_id,
                        paused_at_stage_id=current_stage_name, # Pause *before* executing the next stage
                        status=FlowPauseStatus.PAUSED_FOR_CLARIFICATION,
                        timestamp=datetime.now(timezone.utc),
                        required_clarification_prompt=stage_spec.clarification_checkpoint.prompt,
                        # Store current full shared context
                        full_shared_context_snapshot=self.shared_context.model_dump() 
                    )
                    self.state_manager.save_paused_run_details(pause_details) # Use new method
                    self._emit_metric(MetricEventType.FLOW_PAUSED, flow_id, run_id, stage_id=current_stage_name, data={"reason": FlowPauseStatus.PAUSED_FOR_CLARIFICATION.value, "prompt": stage_spec.clarification_checkpoint.prompt})
                    return self.shared_context.previous_stage_outputs # Return current outputs, flow is paused.
            
            self.state_manager.update_run_context(run_id, self.shared_context.previous_stage_outputs, self.shared_context.artifact_references) # Save context progress

        # ---- End of main execution loop ----

        if current_stage_name == NEXT_STAGE_END_SUCCESS:
            final_status = StageStatus.COMPLETED_SUCCESS
            self.logger.info(f"Run {run_id} for flow '{flow_id}' completed successfully.")
        elif current_stage_name == NEXT_STAGE_END_FAILURE:
            final_status = StageStatus.FAILURE # Changed to StageStatus.FAILURE
            self.logger.error(f"Run {run_id} for flow '{flow_id}' ended in failure. Error: {flow_error_details or 'Unknown error'}")
        elif hops >= self.MAX_HOPS: # Already handled inside loop, but as a final status check
            final_status = StageStatus.FAILURE # Changed to StageStatus.FAILURE
            flow_error_details = flow_error_details or "Max hops reached."
            self.logger.error(f"Run {run_id} for flow '{flow_id}' terminated due to max hops. Error: {flow_error_details}")
        elif current_stage_name is None and final_status == StageStatus.COMPLETED_SUCCESS: # Flow ended because no next stage was defined
             self.logger.info(f"Run {run_id} for flow '{flow_id}' completed successfully as no next stage was defined.")
        else: # Other unexpected termination
            final_status = StageStatus.FAILURE # Changed to StageStatus.FAILURE
            flow_error_details = flow_error_details or f"Flow ended unexpectedly at stage '{current_stage_name}'."
            self.logger.error(f"Run {run_id} for flow '{flow_id}' ended unexpectedly. Final Stage: {current_stage_name}. Status: {final_status}. Error: {flow_error_details}")

        self._emit_metric(MetricEventType.FLOW_END, flow_id, run_id, data={"status": final_status.value, "error": flow_error_details})
        self.state_manager.record_flow_end(run_id, flow_id, final_status, error_message=flow_error_details, final_outputs=self.shared_context.previous_stage_outputs)
        
        # Clean up run-specific state from orchestrator instance if necessary,
        # though shared_context is re-initialized per run.
        # self._current_run_id is cleared by the caller (run method)

        if final_status == StageStatus.FAILURE and flow_error_details: # Changed to StageStatus.FAILURE
             self.shared_context.previous_stage_outputs["_flow_error"] = flow_error_details
        
        return self.shared_context.previous_stage_outputs


    def _resolve_input_values(self, inputs_spec: Dict[str, Any]) -> Dict[str, Any]:
        # ... existing code ...
        pass

# Ensuring the file ends with a valid, indented statement.
pass