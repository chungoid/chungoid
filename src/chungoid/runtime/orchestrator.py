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
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec, ClarificationCheckpointSpec
from chungoid.schemas.agent_master_planner import MasterPlannerInput, MasterPlannerOutput
from chungoid.schemas.agent_code_generator import SmartCodeGeneratorAgentInput
from chungoid.schemas.flows import PausedRunDetails
from chungoid.schemas.orchestration import SharedContext, SystemContext
from chungoid.schemas.metrics import MetricEvent, MetricEventType
from chungoid.utils.metrics_store import MetricsStore
from chungoid.runtime.agents.system_file_system_agent import SystemFileSystemAgent_v1, WriteArtifactToFileInput
from chungoid.runtime.agents.system_master_planner_reviewer_agent import MasterPlannerReviewerAgent
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import EXECUTION_PLANS_COLLECTION, ProjectChromaManagerAgent_v1
from chungoid.schemas.project_state import ProjectStateV2, RunRecord, StageRecord
from chungoid.runtime.agents.core_code_generator_agent import CoreCodeGeneratorAgent_v1 # For checking agent_id

# ADDED: Import for the correct collection name
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import GENERATED_CODE_ARTIFACTS_COLLECTION

# Constants for next_stage signals
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

    MAX_HOPS = 100
    DEFAULT_AGENT_RETRIES = 1
    ARTIFACT_OUTPUT_KEY = "_mcp_generated_artifacts_relative_paths_"

    _current_run_id: Optional[str] = None
    _last_successful_stage_output: Optional[Any] = None
    _current_flow_config: Optional[Dict[str, Any]] = None
    shared_context: SharedContext
    initial_goal_str: Optional[str] = None # ADDED: To store the initial goal string for the run

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
        default_on_failure_action: OnFailureAction = OnFailureAction.INVOKE_REVIEWER # ADDED
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
        self.logger = logging.getLogger(__name__) # MOVED/ADDED: Get logger instance here
        self.logger.setLevel(logging.DEBUG) # Optional: set level for this specific logger
        self.master_planner_reviewer_agent_id = master_planner_reviewer_agent_id
        self.default_on_failure_action = default_on_failure_action # ADDED

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

        # Ensure the project directory from orchestrator config is in global_project_settings
        if self.shared_context.global_project_settings is None: # Initialize if None
            self.shared_context.global_project_settings = {}
        # Use self.shared_context.project_root_path which is already resolved and set
        self.shared_context.global_project_settings["project_dir"] = str(self.shared_context.project_root_path) 
        self.logger.info(f"Orchestrator __init__ set shared_context.global_project_settings.project_dir to: {self.shared_context.project_root_path}")

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
        self, criterion: str, stage_outputs: Dict[str, Any], current_stage_name: str
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

            # STRIP QUOTES from path_str if they exist for direct dictionary key access
            if path_str.startswith("'") and path_str.endswith("'"):
                path_str = path_str[1:-1]
            
            # Normalize path if it refers to the current stage's outputs via full context path
            current_stage_output_prefix = f"context.outputs.{current_stage_name}."
            if path_str.startswith(current_stage_output_prefix):
                path_str = path_str[len(current_stage_output_prefix):]
                self.logger.info(f"COMPARISON check: Normalized path_str to '{path_str}' relative to current stage outputs.")

            actual_val = stage_outputs
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
            # Pass current_stage_name for path normalization
            if not self._evaluate_criterion(criterion_str, stage_outputs, stage_name): 
                self.logger.warning(f"Stage '{stage_name}' failed success criterion: {criterion_str}")
                failed_criteria.append(criterion_str)
                all_passed = False # <--- ADDED THIS LINE

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
        """
        Determines the next stage based on the current stage's definition and context.
        This version is for MasterExecutionPlan.
        """
        current_stage_spec = self.shared_context.current_master_plan.stages.get(current_stage_name) # type: ignore
        if not current_stage_spec:
            self.logger.error(f"Current stage '{current_stage_name}' not found in plan.")
            return NEXT_STAGE_END_FAILURE

        # Check for clarification checkpoint first
        if current_stage_spec.clarification_checkpoint:
            # If a checkpoint is defined, it means we might pause here.
            # The actual pause logic is handled in _execute_flow_loop before invoking the agent.
            # For determining the *next* stage after a potential clarification,
            # we assume the clarification will be resolved and we proceed normally.
            pass # Clarification doesn't change the next_stage linkage here.

        # Success criteria are checked *after* agent execution, so they don't influence next_stage decision here.
        
        next_stage_name = current_stage_spec.next_stage

        if not next_stage_name:
            self.logger.error(f"Stage '{current_stage_name}' has no 'next_stage' defined.")
            return NEXT_STAGE_END_FAILURE
        
        if next_stage_name == "FINAL_STEP":
            return NEXT_STAGE_END_SUCCESS

        if next_stage_name not in self.shared_context.current_master_plan.stages: # type: ignore
            self.logger.error(f"Next stage '{next_stage_name}' (from '{current_stage_name}') not found in plan.")
            return NEXT_STAGE_END_FAILURE
            
        return next_stage_name

    async def _handle_stage_error(
        self,
        current_stage_name: str,
        flow_id: str,
        run_id: str,
        current_plan: MasterExecutionPlan, # Added to access stage spec for max_retries
        agent_id_for_error: Optional[str],
        error: Exception,
        attempt_number: int,
        current_shared_context: SharedContext,
        # current_stage_context: Dict[str, Any], # No longer explicitly needed here if passing shared_context
        # project_root: Path # No longer explicitly needed here
    ) -> Tuple[str, AgentErrorDetails, FlowPauseStatus, Optional[FlowPauseStatus]]: # MODIFIED: PauseStatus to FlowPauseStatus
        """Handles errors that occur during stage execution."""
        self.logger.error(
            f"Run {run_id}: Stage '{current_stage_name}' encountered an error on attempt {attempt_number}: {type(error).__name__} - {error}",
            exc_info=True,
        )

        # Ensure agent_id_for_error has a value, even if it's a placeholder
        effective_agent_id = agent_id_for_error or "UNKNOWN_AGENT_DURING_ERROR"

        error_traceback_str = traceback.format_exc()

        # MODIFIED: Correct instantiation of AgentErrorDetails
        agent_error_details = AgentErrorDetails(
            agent_id=effective_agent_id,
            stage_id=current_stage_name,
            error_type=type(error).__name__,
            message=str(error),
            traceback=error_traceback_str, # Correct field for traceback
            details=None # Set details to None, or provide actual structured details if available
        )
        
        # current_shared_context.add_error_to_stage(current_stage_name, agent_error_details) # Commented out as StateManager handles error recording

        stage_spec = current_plan.stages.get(current_stage_name)
        max_retries_for_stage = 0 # Default to no retries
        if stage_spec and stage_spec.max_retries is not None:
            max_retries_for_stage = stage_spec.max_retries
        
        # MODIFIED: Correct retry logic without 'can_retry' field
        if attempt_number < max_retries_for_stage:
            self.logger.info(
                f"Run {run_id}: Stage '{current_stage_name}' attempt {attempt_number} failed. Retrying (max attempts: {max_retries_for_stage})."
            )
            # self._emit_metric(...) or await self._state_manager.record_stage_attempt_failed(...) could go here
            return current_stage_name, agent_error_details, FlowPauseStatus.NOT_PAUSED, None # MODIFIED: Use NOT_PAUSED

        self.logger.error(
            f"Run {run_id}: Stage '{current_stage_name}' failed after {attempt_number} attempts (max: {max_retries_for_stage}). Escalating."
        )

        # --- Escalation/Reviewer Logic (if not retrying or retries exhausted) ---
        # This part handles what happens when retries are exhausted.
        # It might involve invoking a reviewer agent or deciding to pause/fail the pipeline based on stage_spec.on_failure.

        pause_status_after_failure = FlowPauseStatus.CRITICAL_ERROR_REQUIRES_MANUAL_INTERVENTION # MODIFIED: PauseStatus to FlowPauseStatus
        next_stage_if_failed_permanently = NEXT_STAGE_END_FAILURE

        if stage_spec:
            if stage_spec.on_failure == OnFailureAction.FAIL_MASTER_FLOW: # MODIFIED: FAIL_PIPELINE to FAIL_MASTER_FLOW
                self.logger.info(f"Run {run_id}: Stage '{current_stage_name}' policy is FAIL_MASTER_FLOW.")
                pause_status_after_failure = FlowPauseStatus.NOT_PAUSED # MODIFIED: Use NOT_PAUSED
            elif stage_spec.on_failure == OnFailureAction.INVOKE_REVIEWER or \
                 (not stage_spec.on_failure and self.default_on_failure_action == OnFailureAction.INVOKE_REVIEWER): # MODIFIED: ensure ESCALATE_TO_REVIEWER is INVOKE_REVIEWER
                self.logger.info(f"Run {run_id}: Stage '{current_stage_name}' policy is INVOKE_REVIEWER (or default).")
                # Invoke reviewer logic
                # Placeholder for actual reviewer invocation logic:
                # reviewer_suggestion = await self._invoke_reviewer_and_get_suggestion(...)
                # Based on reviewer_suggestion, you might change next_stage_if_failed_permanently or pause_status_after_failure
                pause_status_after_failure = FlowPauseStatus.PAUSED_FOR_INTERVENTION # MODIFIED: PAUSED_FOR_REVIEW to PAUSED_FOR_INTERVENTION
            elif stage_spec.on_failure == OnFailureAction.RETRY_THEN_FAIL and attempt_number >= max_retries_for_stage:
                 self.logger.info(f"Run {run_id}: Stage '{current_stage_name}' policy is RETRY_THEN_FAIL, retries exhausted. Failing.")
                 pause_status_after_failure = FlowPauseStatus.NOT_PAUSED # MODIFIED: Use NOT_PAUSED
            # Add other OnFailureAction handlers as needed

        self.logger.error(
            f"Run {run_id}: Stage '{current_stage_name}' failed permanently. Error: {agent_error_details.message}"
        )
        # self._emit_metric(...) for permanent failure
        
        return next_stage_if_failed_permanently, agent_error_details, pause_status_after_failure, None

    async def _invoke_reviewer_and_get_suggestion(
        self,
        run_id: str,
        flow_id: str,
        current_stage_name: str,
        agent_error_details: Optional[AgentErrorDetails],
        current_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]: # CHANGED Type hint
        # This method should be implemented to invoke the reviewer agent and get a suggestion
        # For now, we'll just return a placeholder response indicating to continue.
        self.logger.warning(f"Run {run_id}: _invoke_reviewer_and_get_suggestion is a placeholder and was called for stage '{current_stage_name}'. Returning a default 'continue' suggestion.")
        # This placeholder no longer attempts to instantiate MasterPlannerReviewerOutput
        return {
            "suggestion_type": "CONTINUE_IGNORE_ERROR", # Placeholder value, assuming ReviewerActionType.CONTINUE_IGNORE_ERROR is a string
            "revised_plan_json": None,
            "notes": "Placeholder response from _invoke_reviewer_and_get_suggestion"
        }

    async def _invoke_agent_for_stage(
        self,
        stage_name: str,
        agent_id: str, 
        agent_callable: Callable, 
        inputs_spec: Optional[Dict[str, Any]],
        max_retries: int # This parameter might be unused if retries are handled elsewhere
    ) -> Any: # Returns agent_output or raises an error
        
        final_inputs_for_agent: Dict[str, Any] # Type hint for clarity

        # ADDED: Inject initial_goal_str for the first stage if 'user_goal' is not defined
        if self.current_plan and stage_name == self.current_plan.start_stage:
            self.logger.debug(f"Stage '{stage_name}' is the start stage. Checking for 'user_goal' injection.")
            current_inputs_spec = inputs_spec or {} # Ensure it's a dict
            if 'user_goal' not in current_inputs_spec and self.shared_context.initial_goal_str:
                self.logger.info(f"Injecting initial_goal_str as 'user_goal' for start stage '{stage_name}'.")
                # Create a new dict to avoid modifying the original plan's stage_spec
                final_inputs_for_agent = current_inputs_spec.copy() 
                final_inputs_for_agent['user_goal'] = self.shared_context.initial_goal_str
            else:
                # If 'user_goal' is already there, or no initial_goal_str, resolve as usual
                final_inputs_for_agent = self._resolve_input_values(current_inputs_spec)
        else:
            # For non-start stages, or if start stage already has user_goal specified (or no initial_goal_str)
            final_inputs_for_agent = self._resolve_input_values(inputs_spec or {})
        # --- END ADDED ---

        # Original logic for resolving inputs_spec (now handled above with conditional injection)
        # final_inputs_for_agent = self._resolve_input_values(inputs_spec or {})

        self.logger.debug(f"_invoke_agent_for_stage: stage='{stage_name}', agent_id='{agent_id}', inputs_spec={final_inputs_for_agent}")
        self._current_run_id = self._current_run_id or f"run_{uuid.uuid4().hex[:16]}"
        
        self.logger.info(f"Run {self._current_run_id}: Invoking agent '{agent_id}' for stage '{stage_name}' with inputs: {final_inputs_for_agent}")
        
        try:
            agent_output: Any
            if agent_id == "SmartCodeGeneratorAgent_v1": # Check for the specific agent
                self.logger.debug(f"Preparing SmartCodeGeneratorAgentInput for agent {agent_id}")
                # project_id is needed by SmartCodeGeneratorAgentInput but might not be in final_inputs directly from plan
                # It should be in the shared_context or potentially the full_context for the agent
                if 'project_id' not in final_inputs_for_agent and self.shared_context.project_id:
                    final_inputs_for_agent['project_id'] = self.shared_context.project_id
                
                # task_id is also required. If not in inputs, generate one.
                if 'task_id' not in final_inputs_for_agent:
                    final_inputs_for_agent['task_id'] = f"{stage_name}_{uuid.uuid4()}"

                # Ensure all required fields for SmartCodeGeneratorAgentInput are present in final_inputs
                # or can be derived. This might need more robust handling.
                # For now, assume final_inputs can be spread into SmartCodeGeneratorAgentInput
                try:
                    task_input_model = SmartCodeGeneratorAgentInput(**final_inputs_for_agent)
                    agent_output = await agent_callable(task_input=task_input_model, full_context=self.shared_context)
                except ValidationError as pydantic_error:
                    self.logger.error(f"Pydantic validation error creating SmartCodeGeneratorAgentInput for stage '{stage_name}': {pydantic_error}. Inputs: {final_inputs_for_agent}")
                    raise # Re-raise the validation error to be caught by the broader try-except
            elif agent_id == "SystemFileSystemAgent_v1" or agent_id == "FileOperationAgent_v1":
                self.logger.debug(f"Special invocation for {agent_id} (stage: {stage_name}).")
                tool_name_val = final_inputs_for_agent.get("tool_name")
                tool_input_val = final_inputs_for_agent.get("tool_input", {}) # Default to empty dict if not present
                
                if not tool_name_val:
                    raise ValueError(f"Agent {agent_id} was called for stage '{stage_name}' without 'tool_name' in inputs.")

                # The SystemFileSystemAgent_v1.invoke_async now handles 'inputs' dict directly
                # and expects project_root to be passed if not in full_context.
                # We ensure project_root is available.
                invoke_kwargs = {
                    "inputs": final_inputs_for_agent, # Pass the resolved inputs
                    "full_context": self.shared_context
                }
                if self.shared_context.project_root_path:
                     invoke_kwargs["project_root"] = self.shared_context.project_root_path

                agent_output = await agent_callable(**invoke_kwargs)
            elif agent_id == "SystemTestRunnerAgent_v1":
                # Example: SystemTestRunnerAgent might not use 'inputs' kwarg directly
                # or might need project_root explicitly.
                # This depends on its invoke_async signature.
                # For now, assuming it follows the default or its __init__ handles project_root.
                invoke_kwargs = {
                    "inputs": final_inputs_for_agent, # Pass the resolved inputs
                    "full_context": self.shared_context
                }
                if self.shared_context.project_root_path:
                     invoke_kwargs["project_root"] = self.shared_context.project_root_path

                agent_output = await agent_callable(**invoke_kwargs)
            elif agent_id == "ArchitectAgent_v1": # ADDED: Specific handling for ArchitectAgent_v1
                self.logger.info(f"Run {self._current_run_id}: Preparing ArchitectAgentInput for ArchitectAgent_v1.")
                try:
                    # --- Prepare ArchitectAgentInput ---
                    # project_id should be available in shared_context
                    project_id_for_architect = self.shared_context.project_id
                    if not project_id_for_architect:
                        raise ValueError("project_id is not set in shared_context for ArchitectAgent_v1.")

                    # 'final_inputs_for_agent' should contain the resolved value for
                    # 'refined_requirements_document_id' based on the plan's input mapping.
                    # The ArchitectAgentInput model expects 'loprd_doc_id'.
                    
                    # The key in final_inputs_for_agent will be what was in the plan's 'inputs' for this stage.
                    # The plan for 'architectural_blueprinting' stage has:
                    # "inputs": { "refined_requirements_document_id": "{context.outputs.goal_analysis_and_requirements_gathering.refined_requirements_document_id}" }
                    # So, after _resolve_input_values, final_inputs_for_agent should have a key 'refined_requirements_document_id'
                    # with the actual ID.

                    loprd_doc_id_from_resolved_inputs = final_inputs_for_agent.get("refined_requirements_document_id")

                    if not loprd_doc_id_from_resolved_inputs:
                        self.logger.error(
                            f"Critical: 'refined_requirements_document_id' was not resolved correctly from plan inputs "
                            f"or was not present in resolved inputs: {final_inputs_for_agent}. "
                            f"ArchitectAgent_v1 will likely fail."
                        )
                        # Set to None, Pydantic validation will catch it if it's a required field.
                        loprd_doc_id_from_resolved_inputs = None
                    else:
                        self.logger.info(f"Successfully resolved 'refined_requirements_document_id': {loprd_doc_id_from_resolved_inputs} for ArchitectAgent_v1.")


                    # Create the ArchitectAgentInput object
                    # final_inputs_for_agent might contain other optional fields like existing_blueprint_doc_id
                    architect_task_input_data = {
                        "project_id": project_id_for_architect,
                        "loprd_doc_id": loprd_doc_id_from_resolved_inputs, # Use the resolved value
                        # Spread other inputs from final_inputs_for_agent,
                        # but be careful not to overwrite project_id or loprd_doc_id
                        **{k: v for k, v in final_inputs_for_agent.items() if k not in ["project_id", "loprd_doc_id", "refined_requirements_document_id"]}
                    }
                    # task_id has a default factory in ArchitectAgentInput

                    # Ensure loprd_doc_id is not None if it's a required field in ArchitectAgentInput
                    if architect_task_input_data.get("loprd_doc_id") is None:
                         self.logger.error("ArchitectAgentInput is being created with loprd_doc_id as None. This will likely fail validation if the field is not Optional.")


                    from chungoid.agents.autonomous_engine.architect_agent import ArchitectAgentInput # Import locally for Pydantic model
                    architect_agent_input_obj = ArchitectAgentInput(**architect_task_input_data)
                    
                    self.logger.info(f"Run {self._current_run_id}: Invoking ArchitectAgent_v1 with ArchitectAgentInput: {architect_agent_input_obj.model_dump_json(indent=2)}")
                    agent_output = await agent_callable(task_input=architect_agent_input_obj, full_context=self.shared_context)
                except ValidationError as ve:
                    self.logger.error(f"Pydantic ValidationError creating ArchitectAgentInput for stage '{stage_name}': {ve}. Data: {architect_task_input_data}", exc_info=True)
                    raise OrchestratorError(f"Failed to create valid input for ArchitectAgent_v1 for stage '{stage_name}': {ve}") from ve
                except Exception as e_arch_prep:
                    self.logger.error(f"Error preparing inputs or invoking ArchitectAgent_v1 for stage '{stage_name}': {e_arch_prep}", exc_info=True)
                    raise OrchestratorError(f"Failed to prepare inputs for ArchitectAgent_v1: {e_arch_prep}") from e_arch_prep
            else:
                # Default invocation for other agents
                agent_output = await agent_callable(inputs=final_inputs_for_agent, full_context=self.shared_context)
            
            self.logger.info(f"Run {self._current_run_id}: Agent '{agent_id}' for stage '{stage_name}' completed. Output type: {type(agent_output)}")
            self.logger.debug(f"Run {self._current_run_id}: Agent '{agent_id}' output for stage '{stage_name}': {str(agent_output)[:500]}...") # Log snippet
            return agent_output
        except Exception as e:
            self.logger.error(f"Run {self._current_run_id}: Error invoking agent '{agent_id}' for stage '{stage_name}': {e}", exc_info=True)
            raise OrchestratorError(f"Agent invocation failed for stage '{stage_name}', agent '{agent_id}'. Original error: {type(e).__name__} - {e}") from e
        
        # REMOVED: The incomplete "... rest of _invoke_agent_for_stage method ..."
        
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
        self.shared_context.initial_goal_str = goal_str # ADDED: Store initial goal string
        self.shared_context.previous_stage_outputs = {"_initial_context_": self.shared_context.initial_inputs.copy()}
        self.shared_context.artifact_references = {}
        self.shared_context.scratchpad = {}
        self.shared_context.current_stage_id = None
        self.shared_context.current_stage_status = None

        # Ensure the project directory from orchestrator config is in global_project_settings
        if self.shared_context.global_project_settings is None: # Initialize if None
            self.shared_context.global_project_settings = {}
        # Use self.shared_context.project_root_path which is already resolved and set
        self.shared_context.global_project_settings["project_dir"] = str(self.shared_context.project_root_path) 
        self.logger.info(f"Orchestrator .run() also set shared_context.global_project_settings.project_dir to: {self.shared_context.project_root_path}")

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

            # Validate the plan structure immediately after it's confirmed to be loaded.
            # This ensures any auto-fixes (like assigning a plan ID if missing) are applied
            # before the ID or other structure is used.
            self._validate_master_plan_structure(loaded_plan)

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
                     self.logger.error(critical_error_msg) # This error should now be less likely due to auto-fix
                     self._emit_metric(MetricEventType.ORCHESTRATOR_ERROR, "UNKNOWN_FLOW_NO_ID", current_run_id, data={"error": critical_error_msg})
                     return {"_flow_error": critical_error_msg}


            self.current_plan = loaded_plan
            self.shared_context.current_master_plan = self.current_plan # ADDED: Ensure shared_context has the plan

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
            
            plan_to_resume = self._state_manager.load_master_plan(paused_details.flow_id)
            if not plan_to_resume:
                self.logger.error(f"CRITICAL RESUME ERROR: Could not load MasterExecutionPlan ID '{paused_details.flow_id}' for resume.")
                return {"error": f"Could not load plan for flow ID {paused_details.flow_id}"}
            self.current_plan = plan_to_resume

            if self.current_plan.id != paused_details.flow_id:
                 self.logger.error(f"Orchestrator's current plan ('{self.current_plan.id}') doesn't match paused flow_id ('{paused_details.flow_id}'). Cannot resume safely.")
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
        action_data = action_data or {}

        self.shared_context = SharedContext(**context) 
        self.shared_context.run_id = run_id
        self.shared_context.flow_id = paused_details.flow_id
        project_id_from_config = self.config.get("project_id")
        project_root_path_from_config = self.config.get("project_root_path")
        if not self.shared_context.project_id and project_id_from_config:
            self.shared_context.project_id = project_id_from_config
        if not self.shared_context.project_root_path and project_root_path_from_config:
            self.shared_context.project_root_path = str(project_root_path_from_config)
        if self.shared_context.global_project_settings is None:
            self.shared_context.global_project_settings = {}
        if self.shared_context.project_root_path:
            self.shared_context.global_project_settings["project_dir"] = str(self.shared_context.project_root_path)
        
        self._current_run_id = run_id

        if action == "retry":
            self.logger.info(f"[Resume] Action: Retry MASTER stage '{paused_details.paused_at_stage_id}'")
            start_stage_name = paused_details.paused_at_stage_id

        elif action == "retry_with_inputs":
            self.logger.info(f"[Resume] Action: Retry MASTER stage '{paused_details.paused_at_stage_id}' with new inputs.")
            start_stage_name = paused_details.paused_at_stage_id
            new_inputs = action_data.get('inputs')
            if isinstance(new_inputs, dict):
                self.logger.debug(f"Applying new inputs to shared_context.initial_inputs: {new_inputs}")
                if self.shared_context.initial_inputs:
                    self.shared_context.initial_inputs.update(new_inputs) 
                else:
                    self.shared_context.initial_inputs = new_inputs
            else:
                self.logger.warning("Action 'retry_with_inputs' called without valid 'inputs' dictionary.")
                return {"error": "Action 'retry_with_inputs' requires a dictionary under the 'inputs' key in action_data."}
        
        elif action == "skip_stage":
            self.logger.info(f"[Resume] Action: Skip MASTER stage '{paused_details.paused_at_stage_id}'")
            if not self.current_plan or self.current_plan.id != paused_details.flow_id:
                 self.logger.error("CRITICAL RESUME SKIP ERROR: current_plan mismatch or not set.")
                 return {"error": "Plan mismatch during skip action."}
            
            paused_stage_spec: Optional[MasterStageSpec] = self.current_plan.stages.get(paused_details.paused_at_stage_id)
            if not paused_stage_spec:
                 self.logger.error(f"Cannot determine next stage to skip to; paused MASTER stage '{paused_details.paused_at_stage_id}' not found in plan '{self.current_plan.id}'.")
                 return {"error": f"Fatal: Paused stage '{paused_details.paused_at_stage_id}' not found in Master Flow definition."}
            start_stage_name = self._get_next_stage(paused_details.paused_at_stage_id)

            if start_stage_name and start_stage_name not in [NEXT_STAGE_END_SUCCESS, NEXT_STAGE_END_FAILURE]:
                self.logger.info(f"Will attempt to resume MASTER execution from stage: '{start_stage_name}' after skipping.")
            else: 
                self.logger.info(f"Skipped MASTER stage '{paused_details.paused_at_stage_id}'. Flow considered complete or next stage is an end signal ('{start_stage_name}').")
                try:
                    delete_success = self._state_manager.delete_paused_flow_state(run_id)
                    if not delete_success:
                        self.logger.warning(f"Failed to delete paused state file for run_id {run_id} after determining skip leads to completion.")
                except Exception as del_err:
                    self.logger.error(f"Error deleting paused state file for run_id {run_id}: {del_err}")
                return self.shared_context.previous_stage_outputs if self.shared_context.previous_stage_outputs else {}

        elif action == "force_branch":
            target_stage_id = action_data.get('target_stage_id')
            self.logger.info(f"[Resume] Action: Force branch to MASTER stage '{target_stage_id}'")
            if not self.current_plan or self.current_plan.id != paused_details.flow_id:
                 self.logger.error("CRITICAL RESUME FORCE_BRANCH ERROR: current_plan mismatch or not set.")
                 return {"error": "Plan mismatch during force_branch action."}

            if target_stage_id and isinstance(target_stage_id, str) and target_stage_id in self.current_plan.stages:
                start_stage_name = target_stage_id
            else:
                self.logger.error(f"Invalid or missing target_stage_id ('{target_stage_id}') for force_branch action. Must be a valid MASTER stage ID in plan '{self.current_plan.id}'.")
                return {"error": f"Invalid target_stage_id for force_branch: '{target_stage_id}'. It must be a valid stage ID string present in the Master Flow."}
        
        elif action == "abort":
            self.logger.info(f"[Resume] Action: Abort MASTER flow for run_id={run_id}")
            try:
                delete_success = self._state_manager.delete_paused_flow_state(run_id)
                if not delete_success:
                     self.logger.warning(f"Failed to delete paused state file for run_id {run_id} during abort action.")
            except Exception as del_err:
                self.logger.error(f"Error deleting paused state file for run_id {run_id} during abort: {del_err}")
            
            if self.shared_context.previous_stage_outputs is None: self.shared_context.previous_stage_outputs = {}
            self.shared_context.previous_stage_outputs["_orchestrator_final_status"] = StageStatus.ABORTED_BY_USER.value
            self.logger.info(f"Paused state cleared for run_id={run_id}. Master Flow aborted by user action.")
            return self.shared_context.previous_stage_outputs

        else:
            self.logger.error(f"Unsupported resume action: '{action}'")
            return {"error": f"Unsupported resume action: {action}"}
        
        if start_stage_name and start_stage_name not in [NEXT_STAGE_END_SUCCESS, NEXT_STAGE_END_FAILURE]:
            try:
                clear_success = self._state_manager.delete_paused_flow_state(run_id)
                if not clear_success:
                    self.logger.warning(f"Failed to clear paused state for run_id={run_id}. Proceeding with execution.")
            except Exception as e:
                self.logger.exception(f"Failed to clear paused state for run_id={run_id}. Aborting resume. Error: {e}")
                return {"error": f"Failed to clear paused state for run_id={run_id}. Resume aborted."}
            self.logger.info(f"Resuming MASTER execution loop for run_id '{run_id}' from stage '{start_stage_name}'")
            
            final_outputs = await self._execute_flow_loop(
                run_id=run_id, 
                flow_id=paused_details.flow_id, 
                start_stage_name=start_stage_name
            )
            return final_outputs
        elif start_stage_name == NEXT_STAGE_END_SUCCESS or start_stage_name == NEXT_STAGE_END_FAILURE :
            self.logger.info(f"Resume action '{action}' determined flow should proceed to an end state: {start_stage_name}")
            final_status_val = StageStatus.COMPLETED_SUCCESS if start_stage_name == NEXT_STAGE_END_SUCCESS else StageStatus.FAILURE
            self._emit_metric(MetricEventType.FLOW_END, paused_details.flow_id, run_id, data={"status": final_status_val.value})
            self.state_manager.record_flow_end(run_id, paused_details.flow_id, final_status_val, final_outputs=self.shared_context.previous_stage_outputs)
            if self.shared_context.previous_stage_outputs is None: self.shared_context.previous_stage_outputs = {}
            self.shared_context.previous_stage_outputs["_orchestrator_final_status"] = final_status_val.value
            return self.shared_context.previous_stage_outputs
        else: 
            if action != "skip_stage": # If it was skip_stage and start_stage_name is None, it means it skipped to the end.
                 self.logger.error(f"Resume logic for action '{action}' failed to determine a start stage or handle flow completion correctly. Start_stage_name is None.")
                 return {"error": "Internal error: Failed to determine resume start stage or handle flow completion."}
            # If it was skip_stage and resulted in no next stage, it means the flow completed.
            # The previous_stage_outputs should be returned as they are.
            return self.shared_context.previous_stage_outputs if self.shared_context.previous_stage_outputs else {}

    async def _execute_flow_loop( # Ensure async def
        self,
        run_id: str,
        flow_id: str,
        start_stage_name: str
        # initial_shared_context is now assumed to be populated on self.shared_context by the caller (run or resume_flow)
    ) -> Dict[str, Any]:
        current_stage_name: Optional[str] = start_stage_name
        final_status: StageStatus = StageStatus.COMPLETED_SUCCESS # Default to success
        flow_error_details: Optional[str] = None
        hops = 0

        self.logger.info(f"Entering _execute_flow_loop for run_id: {run_id}, flow_id: {flow_id}, start_stage: {start_stage_name}")

        while current_stage_name and current_stage_name not in [NEXT_STAGE_END_SUCCESS, NEXT_STAGE_END_FAILURE] and hops < self.MAX_HOPS:
            hops += 1
            self.logger.info(f"Orchestrator loop hop: {hops}, current_stage_name: {current_stage_name}")

            if current_stage_name == "FINAL_STEP": # ADDED: Handle FINAL_STEP explicitly
                self.logger.info(f"Run {run_id}: Reached FINAL_STEP for flow '{flow_id}'. Signalling successful completion.")
                current_stage_name = NEXT_STAGE_END_SUCCESS
                break

            if not self.current_plan:
                self.logger.error(f"Run {run_id}: Orchestrator critical error - current_plan is not set during loop execution. Terminating.")
                final_status = StageStatus.FAILURE
                flow_error_details = "Orchestrator critical error: current_plan not set."
                break

            stage_spec = self.current_plan.stages.get(current_stage_name)
            if not stage_spec:
                self.logger.error(f"Stage '{current_stage_name}' not found in plan '{flow_id}' for run {run_id}. Terminating.")
                final_status = StageStatus.FAILURE
                flow_error_details = f"Stage '{current_stage_name}' not found in plan."
                break

            self.shared_context.current_stage_id = current_stage_name
            self.shared_context.current_stage_status = StageStatus.RUNNING # Update shared context

            self.logger.info(f"Run {run_id}: Executing stage '{current_stage_name}' (Agent: {stage_spec.agent_id})")
            self._emit_metric(MetricEventType.MASTER_STAGE_START, flow_id, run_id, stage_id=current_stage_name, master_stage_id=current_stage_name, agent_id=stage_spec.agent_id)
            self.state_manager.record_stage_start(run_id, flow_id, current_stage_name, stage_spec.agent_id)

            agent_id_to_invoke = stage_spec.agent_category or stage_spec.agent_id
            if not agent_id_to_invoke: 
                 self.logger.error(f"Stage '{current_stage_name}' in plan '{flow_id}' has neither agent_id nor agent_category. Terminating.")
                 final_status = StageStatus.FAILURE 
                 flow_error_details = f"Stage '{current_stage_name}' misconfigured: missing agent_id/category."
                 break
            
            max_retries_for_stage = stage_spec.max_retries if stage_spec.max_retries is not None else self.DEFAULT_AGENT_RETRIES
            agent_error_obj: Optional[AgentErrorDetails] = None
            next_stage_after_error_handling: Optional[str] = None
            pause_status_override: Optional[FlowPauseStatus] = None
            
            try:
                agent_callable = self.agent_provider.get(agent_id_to_invoke) 
                resolved_agent_id = agent_id_to_invoke 
                
                if stage_spec.agent_category and not stage_spec.agent_id: 
                    self.logger.info(f"Stage '{current_stage_name}': Category '{stage_spec.agent_category}' resolved to agent ID '{resolved_agent_id}'.")
                
                agent_id_for_error_handling = resolved_agent_id 

                stage_output = await self._invoke_agent_for_stage(
                    stage_name=current_stage_name,
                    agent_id=resolved_agent_id, 
                    agent_callable=agent_callable,
                    inputs_spec=stage_spec.inputs,
                    max_retries=max_retries_for_stage 
                )

                output_key = stage_spec.output_context_path or current_stage_name
                self.shared_context.previous_stage_outputs[output_key] = stage_output
                self._last_successful_stage_output = stage_output 

                if isinstance(stage_output, dict) and self.ARTIFACT_OUTPUT_KEY in stage_output:
                    artifact_paths = stage_output[self.ARTIFACT_OUTPUT_KEY]
                    if isinstance(artifact_paths, list):
                        for p_str in artifact_paths:
                            self.shared_context.register_artifact(current_stage_name, Path(p_str))
                    elif isinstance(artifact_paths, str):
                         self.shared_context.register_artifact(current_stage_name, Path(artifact_paths))
                
                criteria_passed, failed_criteria = await self._check_success_criteria(current_stage_name, stage_spec, self.shared_context.previous_stage_outputs)
                if not criteria_passed:
                    self.logger.warning(f"Stage '{current_stage_name}' failed success criteria: {failed_criteria}. Triggering error handling.")
                    agent_error_obj = AgentErrorDetails(
                        agent_id=resolved_agent_id,
                        stage_name=current_stage_name,
                        error_type="SuccessCriteriaFailed",
                        message=f"Stage failed success criteria: {', '.join(failed_criteria)}",
                        details={"failed_criteria": failed_criteria},
                        can_retry=False, 
                        can_escalate=True
                    )
                    next_stage_after_error_handling, agent_error_obj, _, pause_status_override = await self._handle_stage_error(
                        current_stage_name=current_stage_name,
                        flow_id=flow_id,
                        run_id=run_id,
                        current_plan=self.current_plan, 
                        agent_id_for_error=resolved_agent_id,
                        error=agent_error_obj, 
                        attempt_number=max_retries_for_stage + 1, 
                        current_shared_context=self.shared_context 
                    )
                    if pause_status_override: 
                        self.state_manager.save_paused_run(run_id, flow_id, current_stage_name, pause_status_override, self.shared_context.model_dump())
                        self.logger.info(f"Run {run_id} paused at stage '{current_stage_name}' due to error handling result: {pause_status_override.value}")
                        return self.shared_context.previous_stage_outputs 
                    
                    if next_stage_after_error_handling is None and agent_error_obj: 
                        final_status = StageStatus.FAILURE 
                        flow_error_details = agent_error_obj.message
                        current_stage_name = NEXT_STAGE_END_FAILURE 

                if current_stage_name != NEXT_STAGE_END_FAILURE : 
                    self._emit_metric(MetricEventType.MASTER_STAGE_END, flow_id, run_id, stage_id=current_stage_name, master_stage_id=current_stage_name, agent_id=resolved_agent_id, data={"status": StageStatus.COMPLETED_SUCCESS.value})
                    self.state_manager.record_stage_end(run_id, flow_id, current_stage_name, StageStatus.COMPLETED_SUCCESS, outputs=stage_output)
                    self.shared_context.current_stage_status = StageStatus.COMPLETED_SUCCESS
                    
                    self.logger.debug(f"Run {run_id}: Stage '{current_stage_name}' - Evaluating output storage. Stage_spec.output_context_path: '{stage_spec.output_context_path if stage_spec else None}'. Stage_output is not None: {stage_output is not None}")

                    if stage_output and stage_spec.output_context_path: 
                        key_to_use_in_outputs_dict = stage_spec.output_context_path
                        if key_to_use_in_outputs_dict.startswith("outputs."):
                            key_to_use_in_outputs_dict = key_to_use_in_outputs_dict[len("outputs."):]
                        
                        self.shared_context.outputs[key_to_use_in_outputs_dict] = stage_output
                        self.logger.debug(f"Run {run_id}: Stored output of stage '{current_stage_name}' into shared_context.outputs['{key_to_use_in_outputs_dict}']")
                        if isinstance(stage_output, BaseModel):
                            self.shared_context.previous_stage_outputs = stage_output.model_dump()
                        elif isinstance(stage_output, dict):
                            self.shared_context.previous_stage_outputs = stage_output.copy()
                        else:
                            self.logger.warning(f"Stage output for {current_stage_name} (with output_context_path) is of unexpected type {type(stage_output)}. Storing as is in previous_stage_outputs.")
                            self.shared_context.previous_stage_outputs = stage_output
                    elif stage_output: 
                        self.logger.debug(f"Run {run_id}: Stored output of stage '{current_stage_name}' into shared_context.previous_stage_outputs (no specific output_context_path).")
                        if isinstance(stage_output, BaseModel):
                            self.shared_context.previous_stage_outputs = stage_output.model_dump()
                        elif isinstance(stage_output, dict):
                            self.shared_context.previous_stage_outputs = stage_output.copy() 
                        else:
                            self.logger.warning(f"Stage output for {current_stage_name} (no output_context_path) is of unexpected type {type(stage_output)}. Storing as is in previous_stage_outputs.")
                            self.shared_context.previous_stage_outputs = stage_output 
                    else: 
                        self.logger.debug(f"Run {run_id}: Stage '{current_stage_name}' had no output. previous_stage_outputs remains unchanged or set to empty dict if first stage.")
                        if self.shared_context.previous_stage_outputs is None:
                             self.shared_context.previous_stage_outputs = {} 

                    if stage_spec.agent_id == "SystemFileSystemAgent_v1" and isinstance(stage_output, dict) and self.ARTIFACT_OUTPUT_KEY in stage_output:
                        artifact_paths = stage_output[self.ARTIFACT_OUTPUT_KEY]
                        if isinstance(artifact_paths, list):
                            for p_str in artifact_paths:
                                self.shared_context.register_artifact(current_stage_name, Path(p_str))
                        elif isinstance(artifact_paths, str):
                             self.shared_context.register_artifact(current_stage_name, Path(artifact_paths))

                    current_stage_name = self._get_next_stage(current_stage_name) 

            except (NoAgentFoundForCategoryError, AmbiguousAgentCategoryError) as e_agent_resolve:
                self.logger.error(f"Run {run_id}: Agent resolution failed for stage '{current_stage_name}', agent specifier '{agent_id_to_invoke}': {e_agent_resolve}")
                agent_error_obj = AgentErrorDetails(
                    agent_id=agent_id_to_invoke, 
                    stage_name=current_stage_name,
                    error_type=e_agent_resolve.__class__.__name__,
                    message=str(e_agent_resolve),
                    can_retry=False, 
                    can_escalate=True 
                )
                next_stage_after_error_handling, agent_error_obj, _, pause_status_override = await self._handle_stage_error(
                    current_stage_name=current_stage_name,
                    flow_id=flow_id,
                    run_id=run_id,
                    current_plan=self.current_plan, 
                    agent_id_for_error=agent_id_to_invoke, 
                    error=agent_error_obj, 
                    attempt_number=1, 
                    current_shared_context=self.shared_context 
                )

            except Exception as e_invoke: 
                self.logger.error(f"Run {run_id}: Exception during agent invocation for stage '{current_stage_name}': {e_invoke}", exc_info=True)
                current_agent_id_for_err = stage_spec.agent_id or stage_spec.agent_category or "UNKNOWN_AGENT"
                
                next_stage_after_error_handling, agent_error_obj, _, pause_status_override = await self._handle_stage_error(
                    current_stage_name=current_stage_name,
                    flow_id=flow_id,
                    run_id=run_id,
                    current_plan=self.current_plan, 
                    agent_id_for_error=current_agent_id_for_err,
                    error=e_invoke, 
                    attempt_number=max_retries_for_stage + 1, 
                    current_shared_context=self.shared_context 
                )
            
            if agent_error_obj: 
                self.shared_context.current_stage_status = StageStatus.FAILURE 
                self._emit_metric(MetricEventType.MASTER_STAGE_END, flow_id, run_id, stage_id=current_stage_name, master_stage_id=current_stage_name, agent_id=agent_error_obj.agent_id, data={"status": StageStatus.FAILURE.value, "error": agent_error_obj.message}) 
                self.state_manager.record_stage_end(run_id, flow_id, current_stage_name, StageStatus.FAILURE, error_details=agent_error_obj.model_dump()) 

                if pause_status_override:
                    self.state_manager.save_paused_run(run_id, flow_id, current_stage_name, pause_status_override, self.shared_context.model_dump())
                    self.logger.info(f"Run {run_id} paused at stage '{current_stage_name}' due to error handling result: {pause_status_override.value}")
                    return self.shared_context.previous_stage_outputs 
                
                if next_stage_after_error_handling is None: 
                    final_status = StageStatus.FAILURE 
                    flow_error_details = agent_error_obj.message if agent_error_obj else "Unknown error after handling."
                    current_stage_name = NEXT_STAGE_END_FAILURE 
                else: 
                    current_stage_name = next_stage_after_error_handling
            
            if stage_spec.clarification_checkpoint and current_stage_name not in [NEXT_STAGE_END_FAILURE, NEXT_STAGE_END_SUCCESS, None]:
                should_pause_for_clarification = True
                if stage_spec.clarification_checkpoint.condition:
                    should_pause_for_clarification = self._parse_condition(stage_spec.clarification_checkpoint.condition)
                
                if should_pause_for_clarification:
                    self.logger.info(f"Run {run_id}: Pausing at stage '{current_stage_name}' for user clarification. Prompt: {stage_spec.clarification_checkpoint.prompt}")
                    pause_details = PausedRunDetails(
                        run_id=run_id,
                        flow_id=flow_id,
                        paused_at_stage_id=current_stage_name, 
                        status=FlowPauseStatus.PAUSED_FOR_CLARIFICATION,
                        timestamp=datetime.now(timezone.utc),
                        required_clarification_prompt=stage_spec.clarification_checkpoint.prompt,
                        full_shared_context_snapshot=self.shared_context.model_dump() 
                    )
                    self.state_manager.save_paused_run_details(pause_details) 
                    self._emit_metric(MetricEventType.FLOW_PAUSED, flow_id, run_id, stage_id=current_stage_name, data={"reason": FlowPauseStatus.PAUSED_FOR_CLARIFICATION.value, "prompt": stage_spec.clarification_checkpoint.prompt})
                    return self.shared_context.previous_stage_outputs 
            
            self.state_manager.update_run_context(run_id, self.shared_context.previous_stage_outputs, self.shared_context.artifact_references) 

        if current_stage_name == NEXT_STAGE_END_SUCCESS:
            final_status = StageStatus.COMPLETED_SUCCESS
            self.logger.info(f"Run {run_id} for flow '{flow_id}' completed successfully.")
        elif current_stage_name == NEXT_STAGE_END_FAILURE:
            final_status = StageStatus.FAILURE 
            self.logger.error(f"Run {run_id} for flow '{flow_id}' ended in failure. Error: {flow_error_details or 'Unknown error'}")
        elif hops >= self.MAX_HOPS: 
            final_status = StageStatus.FAILURE 
            flow_error_details = flow_error_details or "Max hops reached."
            self.logger.error(f"Run {run_id} for flow '{flow_id}' terminated due to max hops. Error: {flow_error_details}")
        elif current_stage_name is None and final_status == StageStatus.COMPLETED_SUCCESS: 
             self.logger.info(f"Run {run_id} for flow '{flow_id}' completed successfully as no next stage was defined.")
        else: 
            final_status = StageStatus.FAILURE 
            flow_error_details = flow_error_details or f"Flow ended unexpectedly at stage '{current_stage_name}'."
            self.logger.error(f"Run {run_id} for flow '{flow_id}' ended unexpectedly. Final Stage: {current_stage_name}. Status: {final_status}. Error: {flow_error_details}")

        self._emit_metric(MetricEventType.FLOW_END, flow_id, run_id, data={"status": final_status.value, "error": flow_error_details})
        self.state_manager.record_flow_end(run_id, flow_id, final_status, error_message=flow_error_details, final_outputs=self.shared_context.previous_stage_outputs)
        
        if self.shared_context.previous_stage_outputs is None:
            self.shared_context.previous_stage_outputs = {}
            self.logger.warning(f"Run {run_id} flow {flow_id}: previous_stage_outputs was None at the end of _execute_flow_loop. Initialized to empty dict before adding status.")

        self.shared_context.previous_stage_outputs["_orchestrator_final_status"] = final_status.value
        if flow_error_details: 
            self.shared_context.previous_stage_outputs["_orchestrator_flow_error_details"] = flow_error_details
        elif "_orchestrator_flow_error_details" in self.shared_context.previous_stage_outputs:
            del self.shared_context.previous_stage_outputs["_orchestrator_flow_error_details"]
        
        return self.shared_context.previous_stage_outputs

    def _resolve_input_values(self, inputs_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves input values based on the input specification.
        This implementation handles direct string passthrough,
        context lookups prefixed with '@', and context lookups
        encapsulated in '{context...}'.
        """
        if not inputs_spec:
            return {}

        # Corrected: Initialize resolved_inputs, not resolved_values
        resolved_inputs: Dict[str, Any] = {} 

        for key, value_spec in inputs_spec.items():
            if isinstance(value_spec, str):
                # Try to match direct context path: {context.path.to.value}
                direct_match = re.match(r"^\{context\.([a-zA-Z0-9_.]+)\}$", value_spec)
                if direct_match:
                    path_str = direct_match.group(1)
                    self.logger.debug(f"Found general context path string: '{value_spec}', extracted path: '{path_str}' for key '{key}'")
                    
                    resolved_value = None
                    path_resolved_from_outputs = False
                    path_resolved_from_previous = False
                    p_str_to_try_for_log = path_str # For logging in case of exception

                    # Attempt 1: Resolve from self.shared_context.outputs if path starts with "outputs."
                    if path_str.startswith("outputs."):
                        try:
                            current_obj = self.shared_context.outputs
                            # path_str is like "outputs.stage_name.key" or "outputs.key"
                            for part in path_str.split('.')[1:]: # Skip "outputs."
                                if isinstance(current_obj, dict) and part in current_obj:
                                    current_obj = current_obj[part]
                                elif hasattr(current_obj, part): # Handles Pydantic models
                                    current_obj = getattr(current_obj, part)
                                else:
                                    self.logger.debug(f"Path part '{part}' not in shared_context.outputs dict or as attr for path '{path_str}' for key '{key}'. Current obj type: {type(current_obj)}")
                                    current_obj = None 
                                    break
                            if current_obj is not None:
                                resolved_value = current_obj
                                path_resolved_from_outputs = True
                                self.logger.debug(f"Resolved '{path_str}' for key '{key}' from shared_context.outputs to: {str(resolved_value)[:100]}")
                        except Exception as e_outputs:
                            self.logger.warning(f"Error resolving path '{path_str}' in self.shared_context.outputs: {e_outputs}")
                    
                    # Attempt 2: If not resolved from outputs, try self.shared_context.previous_stage_outputs
                    if not path_resolved_from_outputs and self.shared_context.previous_stage_outputs:
                        self.logger.debug(f"Path '{path_str}' for key '{key}' not resolved from shared_context.outputs. Trying previous_stage_outputs.")
                        
                        key_to_find_directly_in_previous = path_str # Default for simple {context.key} or if complex path matches an actual key
                        is_complex_outputs_path = False
                        path_parts = path_str.split('.')

                        if path_str.startswith("outputs.") and len(path_parts) == 3:
                            # This is for a path like "outputs.STAGE_NAME.ACTUAL_KEY"
                            # where STAGE_NAME's output (a Pydantic model or dict) is directly in previous_stage_outputs.
                            # We need to find ACTUAL_KEY on that model/dict.
                            key_to_find_directly_in_previous = path_parts[2] # ACTUAL_KEY
                            is_complex_outputs_path = True
                            self.logger.debug(f"Interpreting '{path_str}' as looking for '{key_to_find_directly_in_previous}' on the direct output of stage '{path_parts[1]}' (expected in previous_stage_outputs).")
                        elif path_str.startswith("outputs.") and len(path_parts) == 2:
                            # This is for "outputs.KEY", try finding KEY directly on previous_stage_outputs
                            key_to_find_directly_in_previous = path_parts[1]
                            self.logger.debug(f"Interpreting '{path_str}' as looking for '{key_to_find_directly_in_previous}' directly in previous_stage_outputs.")
                        # Else, for a simple {context.key} or non-standard {context.outputs.something}, key_to_find_directly_in_previous remains path_str
                        
                        current_obj_prev = self.shared_context.previous_stage_outputs

                        try:
                            if isinstance(current_obj_prev, dict) and key_to_find_directly_in_previous in current_obj_prev:
                                resolved_value = current_obj_prev[key_to_find_directly_in_previous]
                                path_resolved_from_previous = True
                            elif hasattr(current_obj_prev, key_to_find_directly_in_previous): # Handles Pydantic models
                                resolved_value = getattr(current_obj_prev, key_to_find_directly_in_previous)
                                path_resolved_from_previous = True
                            
                            # Fallback: If not resolved by direct key access (e.g. for non-outputs.STAGE.KEY patterns or truly nested structures in previous_outputs)
                            # and it wasn't a recognized complex outputs path that should have resolved directly.
                            if not path_resolved_from_previous and not is_complex_outputs_path:
                                self.logger.debug(f"Direct key '{key_to_find_directly_in_previous}' not found or path not complex outputs type. Attempting full traversal of '{path_str}' on previous_stage_outputs.")
                                temp_obj = current_obj_prev
                                successfully_traversed = True
                                for part in path_parts: # Use original path_parts for traversal
                                    if isinstance(temp_obj, dict) and part in temp_obj:
                                        temp_obj = temp_obj[part]
                                    elif hasattr(temp_obj, part):
                                        temp_obj = getattr(temp_obj, part)
                                    else:
                                        successfully_traversed = False
                                        self.logger.debug(f"Full traversal of '{path_str}' on previous_stage_outputs failed at part '{part}'.")
                                        break
                                if successfully_traversed and temp_obj is not None:
                                    resolved_value = temp_obj
                                    path_resolved_from_previous = True

                            if path_resolved_from_previous:
                                self.logger.debug(f"Resolved '{path_str}' (as '{key_to_find_directly_in_previous}' or full traversal) for key '{key}' from previous_stage_outputs to: {str(resolved_value)[:100]}")
                        
                        except Exception as e_previous:
                            self.logger.warning(f"Error resolving path '{path_str}' (interpreted for direct key '{key_to_find_directly_in_previous}') in self.shared_context.previous_stage_outputs: {e_previous}")

                    if path_resolved_from_outputs or path_resolved_from_previous:
                        resolved_inputs[key] = resolved_value # Corrected: resolved_inputs
                    else:
                        self.logger.warning(f"Could not fully resolve general context path '{value_spec}' for key '{key}'. Using original string: '{value_spec}'.")
                        resolved_inputs[key] = value_spec # Corrected: resolved_inputs
                
                elif value_spec.startswith("{context.initial_inputs.") and value_spec.endswith("}"):
                    self.logger.warning(f"Initial inputs not fully implemented for key '{key}'. Using original string: '{value_spec}'.")
                    resolved_inputs[key] = value_spec # Corrected: resolved_inputs
                else:
                    self.logger.info(f"No matching pattern found for key '{key}'. Using original string: '{value_spec}'.")
                    resolved_inputs[key] = value_spec # Corrected: resolved_inputs
            else:
                resolved_inputs[key] = value_spec # Corrected: resolved_inputs
                self.logger.debug(f"Input '{key}' is a literal of type {type(value_spec)}: {value_spec}")
        
        return resolved_inputs # Corrected: resolved_inputs

    def _validate_master_plan_structure(self, plan: MasterExecutionPlan):
        """
        Validates the basic structure of a MasterExecutionPlan.
        Can perform some auto-fixing for common LLM generation omissions.
        """
        self.logger.info(f"Validating structure of MasterExecutionPlan ID: {plan.id}")
        if not plan.start_stage or plan.start_stage not in plan.stages:
            msg = f"MasterExecutionPlan {plan.id} has an invalid or missing start_stage: '{plan.start_stage}'."
            self.logger.error(msg)
            raise ValueError(msg)

        all_stage_names = set(plan.stages.keys())
        referenced_next_stages = set()

        # First pass: basic validation and collection of next_stage references
        for stage_name, stage_spec in plan.stages.items():
            self.logger.debug(f"Validating stage: {stage_name} (Agent: {stage_spec.agent_id})")
            if stage_spec.next_stage:
                referenced_next_stages.add(stage_spec.next_stage)
            
            # AUTO-FIX: If SmartCodeGeneratorAgent_v1 is used and inputs are defined but target_file_path is missing.
            if stage_spec.agent_id == "SmartCodeGeneratorAgent_v1" and stage_spec.inputs and "target_file_path" not in stage_spec.inputs:
                self.logger.critical(f"CRITICAL_VALIDATION_ERROR (AUTO-FIXING): Stage '{stage_name}' ({stage_spec.agent_id}) is missing 'target_file_path' in 'inputs'. Injecting default.")
                stage_spec.inputs["target_file_path"] = f"AUTO-FIXED/placeholder/{stage_name}_missing_path.py"
                self.logger.warning(f"AUTO-FIX APPLIED for {stage_spec.agent_id} stage '{stage_name}'. New inputs: {stage_spec.inputs}")

            # AUTO-FIX: If CoreTestGeneratorAgent_v1 is used and inputs are defined but test_file_path is missing
            if stage_spec.agent_id == "CoreTestGeneratorAgent_v1" and stage_spec.inputs and "test_file_path" not in stage_spec.inputs:
                self.logger.critical(f"CRITICAL_VALIDATION_ERROR (AUTO-FIXING): Stage '{stage_name}' ({stage_spec.agent_id}) is missing 'test_file_path' in 'inputs'. Injecting default.")
                stage_spec.inputs["test_file_path"] = f"AUTO-FIXED/placeholder/{stage_name}_missing_tests.py"
                self.logger.warning(f"AUTO-FIX APPLIED for {stage_spec.agent_id} stage '{stage_name}'. New inputs: {stage_spec.inputs}")


        # Check if all referenced next_stages (that aren't FINAL_STEP) exist
        for next_stage_ref in referenced_next_stages:
            if next_stage_ref != "FINAL_STEP" and next_stage_ref not in all_stage_names:
                msg = f"MasterExecutionPlan {plan.id}: Stage '{next_stage_ref}' is referenced as a next_stage but does not exist in the plan stages."
                self.logger.error(msg)
                # Potentially raise ValueError here, or allow auto-linking to attempt a fix if applicable
                # For now, logging as an error. Stricter validation might be needed.

        # Second pass: Auto-link stages if next_stage is missing, based on sequential numbers
        # Sort stages by number to attempt sequential linking
        sorted_stages_by_number = sorted(plan.stages.items(), key=lambda item: item[1].number if item[1].number is not None else float('inf'))
        
        for i, (stage_name, stage_spec) in enumerate(sorted_stages_by_number):
            if not stage_spec.next_stage and stage_name != "FINAL_STEP": # Check if next_stage is None or empty
                # Try to find the next stage by number
                current_number = stage_spec.number
                if current_number is not None:
                    next_stage_candidate_name = None
                    # Look for the stage with number + 1
                    for next_s_name, next_s_spec in sorted_stages_by_number:
                        if next_s_spec.number == current_number + 1:
                            next_stage_candidate_name = next_s_name
                            break
                    
                    if next_stage_candidate_name:
                        self.logger.warning(
                            f"AUTO-FIX APPLIED: Stage '{stage_name}' (Number: {current_number}) was missing 'next_stage'. "
                            f"Automatically linking to stage '{next_stage_candidate_name}' (Number: {current_number + 1})."
                        )
                        stage_spec.next_stage = next_stage_candidate_name
                    elif i + 1 < len(sorted_stages_by_number):
                        # Fallback: if no direct number+1 match, link to the next in the sorted list if not FINAL_STEP
                        potential_next_name = sorted_stages_by_number[i+1][0]
                        if potential_next_name != "FINAL_STEP": # Avoid linking to FINAL_STEP unless it's truly the last
                             self.logger.warning(
                                f"AUTO-FIX APPLIED (Fallback): Stage '{stage_name}' (Number: {current_number}) was missing 'next_stage'. "
                                f"Automatically linking to next stage in sorted list: '{potential_next_name}'."
                            )
                             stage_spec.next_stage = potential_next_name
                        # If the next in list is FINAL_STEP, and we didn't find a number+1, we might be at the actual end.
                        # Or, if it's the last stage in the list and next_stage is still missing, it might implicitly be FINAL_STEP
                    elif i == len(sorted_stages_by_number) - 1: # Is it the last stage in the plan?
                         self.logger.warning(
                            f"AUTO-FIX APPLIED (Last Stage): Stage '{stage_name}' (Number: {current_number}) was missing 'next_stage' and is the last numbered stage. "
                            f"Setting next_stage to 'FINAL_STEP'."
                        )
                         stage_spec.next_stage = "FINAL_STEP"


        self.logger.info(f"MasterExecutionPlan ID: {plan.id} structure validation successful.")

    # Regex to capture "context.path.to.value" in group 1, and "path.to.value" in group 2
    REGEX_CONTEXT_PATH = re.compile(r"{(context\.((?:[\w\-]+\.?)+))}")

    def _resolve_path_value(self, path_after_context_str: str) -> Any:
        # path_after_context_str is "outputs.stage.key" or "global_config.key" etc.
        # It's the part *after* "context."
        
        key_parts = path_after_context_str.split('.') # e.g. ["outputs", "stage_name", "attribute_key"] or ["global_config", "project_dir"]

        if not key_parts or not key_parts[0]:
            self.logger.warning(f"RESOLVE_PATH_VALUE: Path_after_context '{path_after_context_str}' resulted in invalid key_parts. Returning original: '{{context.{path_after_context_str}}}'")
            return f"{{context.{path_after_context_str}}}"

        # current_value: Any = self.shared_context # This was the old way, direct access to shared_context fields
        first_key_segment = key_parts[0]
        
        if first_key_segment == "outputs":
            # Path is like "outputs.stage_name_as_key.attribute_name_in_model.sub_attribute"
            if len(key_parts) < 2: # Must have at least outputs.stage_name_as_key
                self.logger.warning(f"RESOLVE_PATH_VALUE (Outputs): Path '{path_after_context_str}' is too short. Needs at least one key part after 'outputs'. Returning original: '{{context.{path_after_context_str}}}'")
                return f"{{context.{path_after_context_str}}}"
            
            stage_name_as_key = key_parts[1] # This is the key for the shared_context.outputs dictionary

            if not (hasattr(self.shared_context, 'outputs') and isinstance(self.shared_context.outputs, dict)):
                self.logger.warning(f"RESOLVE_PATH_VALUE (Outputs): shared_context.outputs is not a dict or does not exist for path 'context.{path_after_context_str}'. Returning original.")
                return f"{{context.{path_after_context_str}}}"

            if stage_name_as_key not in self.shared_context.outputs:
                self.logger.warning(f"RESOLVE_PATH_VALUE (Outputs): Stage key '{stage_name_as_key}' not found in shared_context.outputs for path 'context.{path_after_context_str}'. Current output keys: {list(self.shared_context.outputs.keys())}. Returning original.")
                return f"{{context.{path_after_context_str}}}"
            
            # Start with the object/value stored for the stage_name_as_key
            current_resolved_value = self.shared_context.outputs[stage_name_as_key]
            
            # If there are more parts, navigate into the current_resolved_value
            # key_parts[2:] are the attributes to get from current_resolved_value
            if len(key_parts) > 2:
                for i, attribute_part in enumerate(key_parts[2:]):
                    path_being_accessed = ".".join(key_parts[2:i+3]) # For logging: e.g., "attr1" then "attr1.attr2"
                    
                    # --- DETAILED LOGGING START ---
                    self.logger.info(f"RESOLVE_PATH_VALUE_DETAIL (Outputs): Attempting to access attribute_part='{attribute_part}'")
                    self.logger.info(f"RESOLVE_PATH_VALUE_DETAIL (Outputs): current_resolved_value type: {type(current_resolved_value)}")
                    if isinstance(current_resolved_value, BaseModel):
                        self.logger.info(f"RESOLVE_PATH_VALUE_DETAIL (Outputs): current_resolved_value class: {current_resolved_value.__class__.__name__}")
                        try:
                            fields_keys = list(current_resolved_value.model_fields.keys())
                            self.logger.info(f"RESOLVE_PATH_VALUE_DETAIL (Outputs): current_resolved_value.model_fields.keys(): {fields_keys}")
                        except Exception as e_log_fields:
                            self.logger.warning(f"RESOLVE_PATH_VALUE_DETAIL (Outputs): Could not get model_fields.keys(): {e_log_fields}")
                    elif isinstance(current_resolved_value, dict):
                        self.logger.info(f"RESOLVE_PATH_VALUE_DETAIL (Outputs): current_resolved_value keys: {list(current_resolved_value.keys())}")
                    # --- DETAILED LOGGING END ---

                    if isinstance(current_resolved_value, BaseModel):
                        if hasattr(current_resolved_value, attribute_part):
                            current_resolved_value = getattr(current_resolved_value, attribute_part)
                        else:
                            self.logger.warning(f"RESOLVE_PATH_VALUE (Outputs): Attribute '{attribute_part}' (part of '{path_being_accessed}') not found in BaseModel for stage '{stage_name_as_key}'. Full path: '{{context.{path_after_context_str}}}'. Returning original.")
                            return f"{{context.{path_after_context_str}}}"
                    elif isinstance(current_resolved_value, dict):
                        if attribute_part in current_resolved_value:
                            current_resolved_value = current_resolved_value[attribute_part]
                        else:
                            self.logger.warning(f"RESOLVE_PATH_VALUE (Outputs): Key '{attribute_part}' (part of '{path_being_accessed}') not found in dict for stage '{stage_name_as_key}'. Full path: '{{context.{path_after_context_str}}}'. Returning original.")
                            return f"{{context.{path_after_context_str}}}"
                    elif current_resolved_value is None:
                        self.logger.warning(f"RESOLVE_PATH_VALUE (Outputs): Encountered None while trying to access '{attribute_part}' for stage '{stage_name_as_key}'. Full path: '{{context.{path_after_context_str}}}'. Returning original.")
                        return f"{{context.{path_after_context_str}}}"
                    else: # It's a primitive or un-navigable type, but more path parts exist
                        self.logger.warning(f"RESOLVE_PATH_VALUE (Outputs): Cannot navigate '{attribute_part}' in value of type {type(current_resolved_value)} for stage '{stage_name_as_key}'. Full path: '{{context.{path_after_context_str}}}'. Returning original.")
                        return f"{{context.{path_after_context_str}}}"
            
            # After loop (or if len(key_parts) == 2), current_resolved_value is the final value
            self.logger.info(f"RESOLVE_PATH_VALUE (Outputs): Resolved '{{context.{path_after_context_str}}}' to '{str(current_resolved_value)[:100]}...' (type: {type(current_resolved_value)})")
            return current_resolved_value

        elif first_key_segment == "global_config":
            # Path is like "global_config.project_dir"
            if not hasattr(self.shared_context, 'global_config') or self.shared_context.global_config is None:
                self.logger.warning(f"RESOLVE_PATH_VALUE: global_config not found or is None. Path: '{{context.{path_after_context_str}}}'. Returning original.")
                return f"{{context.{path_after_context_str}}}"
            
            current_resolved_value = self.shared_context.global_config
            # key_parts[1:] are attributes to get from global_config
            for i, attribute_part in enumerate(key_parts[1:]):
                path_being_accessed = ".".join(key_parts[1:i+2])
                
                # --- DETAILED LOGGING START (GlobalConfig) ---
                self.logger.info(f"RESOLVE_PATH_VALUE_DETAIL (GlobalConfig): Attempting to access attribute_part='{attribute_part}'")
                self.logger.info(f"RESOLVE_PATH_VALUE_DETAIL (GlobalConfig): current_resolved_value type: {type(current_resolved_value)}")
                if isinstance(current_resolved_value, BaseModel):
                    self.logger.info(f"RESOLVE_PATH_VALUE_DETAIL (GlobalConfig): current_resolved_value class: {current_resolved_value.__class__.__name__}")
                    try:
                        fields_keys = list(current_resolved_value.model_fields.keys())
                        self.logger.info(f"RESOLVE_PATH_VALUE_DETAIL (GlobalConfig): current_resolved_value.model_fields.keys(): {fields_keys}")
                    except Exception as e_log_fields_gc:
                        self.logger.warning(f"RESOLVE_PATH_VALUE_DETAIL (GlobalConfig): Could not get model_fields.keys(): {e_log_fields_gc}")
                elif isinstance(current_resolved_value, dict):
                    self.logger.info(f"RESOLVE_PATH_VALUE_DETAIL (GlobalConfig): current_resolved_value keys: {list(current_resolved_value.keys())}")
                # --- DETAILED LOGGING END (GlobalConfig) ---

                if isinstance(current_resolved_value, BaseModel):
                    if hasattr(current_resolved_value, attribute_part):
                        current_resolved_value = getattr(current_resolved_value, attribute_part)
                    else:
                        self.logger.warning(f"RESOLVE_PATH_VALUE (GlobalConfig): Attribute '{attribute_part}' (part of '{path_being_accessed}') not found in BaseModel. Full path: '{{context.{path_after_context_str}}}'. Returning original.")
                        return f"{{context.{path_after_context_str}}}"
                elif isinstance(current_resolved_value, dict):
                    if attribute_part in current_resolved_value:
                        current_resolved_value = current_resolved_value[attribute_part]
                    else:
                        self.logger.warning(f"RESOLVE_PATH_VALUE (GlobalConfig): Key '{attribute_part}' (part of '{path_being_accessed}') not found in dict. Full path: '{{context.{path_after_context_str}}}'. Returning original.")
                        return f"{{context.{path_after_context_str}}}"
                elif current_resolved_value is None:
                    self.logger.warning(f"RESOLVE_PATH_VALUE (GlobalConfig): Encountered None while trying to access '{attribute_part}'. Full path: '{{context.{path_after_context_str}}}'. Returning original.")
                    return f"{{context.{path_after_context_str}}}"
                else:
                    self.logger.warning(f"RESOLVE_PATH_VALUE (GlobalConfig): Cannot navigate '{attribute_part}' in value of type {type(current_resolved_value)}. Full path: '{{context.{path_after_context_str}}}'. Returning original.")
                    return f"{{context.{path_after_context_str}}}"
            
            self.logger.info(f"RESOLVE_PATH_VALUE (GlobalConfig): Resolved '{{context.{path_after_context_str}}}' to '{str(current_resolved_value)[:100]}...' (type: {type(current_resolved_value)})")
            return current_resolved_value
        
        # Fallback for unknown top-level keys like "context.unknown_key..."
        self.logger.warning(f"RESOLVE_PATH_VALUE: Unknown top-level key '{first_key_segment}' in path 'context.{path_after_context_str}'. Returning original.")
        return f"{{context.{path_after_context_str}}}"

    def _recursively_resolve_values(self, item: Any) -> Any:
        if isinstance(item, str):
            match = self.REGEX_CONTEXT_PATH.fullmatch(item) # item is like "{context.path.to.value}"
            if match:
                # group(1) is "context.path.to.value" (the full string inside braces if needed for logging original)
                # group(2) is "path.to.value" (the part after "context.")
                path_after_context = match.group(2) 
                self.logger.debug(f"RECURSIVE_RESOLVE: Matched context path string: '{item}', extracted path_after_context: '{path_after_context}' for _resolve_path_value.")
                return self._resolve_path_value(path_after_context)
            return item # Not a context path, return as is
        elif isinstance(item, dict):
            return {key: self._recursively_resolve_values(value) for key, value in item.items()}
        elif isinstance(item, list):
            return [self._recursively_resolve_values(elem) for elem in item]
        else:
            return item # Non-string, non-dict, non-list, return as is

    def _resolve_input_values(self, inputs_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolves input values from the inputs_spec,
        handling context path lookups like {context.path.to.value}.
        """
        if not inputs_spec:
            return {}
        
        self.logger.info(f"RESOLVE_INPUTS: Starting recursive resolution for inputs_spec: {inputs_spec}")
        
        resolved_inputs = {}
        for key, value_spec in inputs_spec.items():
            resolved_value = self._recursively_resolve_values(value_spec)
            resolved_inputs[key] = resolved_value
            self.logger.info(f"RESOLVE_INPUTS: Key '{key}' resolved to: {str(resolved_value)[:200]} (type: {type(resolved_value)})")

        self.logger.info(f"RESOLVE_INPUTS: Final recursively resolved_inputs: {resolved_inputs}")
        return resolved_inputs

    def _validate_master_plan_structure(self, plan: MasterExecutionPlan):
        """
        Validates the basic structure of a MasterExecutionPlan.
        Can perform some auto-fixing for common LLM generation omissions.
        """
        self.logger.info(f"Validating structure of MasterExecutionPlan ID: {plan.id}")
        if not plan.start_stage or plan.start_stage not in plan.stages:
            msg = f"MasterExecutionPlan {plan.id} has an invalid or missing start_stage: '{plan.start_stage}'."
            self.logger.error(msg)
            raise ValueError(msg)

        all_stage_names = set(plan.stages.keys())
        referenced_next_stages = set()

        # First pass: basic validation and collection of next_stage references
        for stage_name, stage_spec in plan.stages.items():
            self.logger.debug(f"Validating stage: {stage_name} (Agent: {stage_spec.agent_id})")
            if stage_spec.next_stage:
                referenced_next_stages.add(stage_spec.next_stage)
            
            # AUTO-FIX: If SmartCodeGeneratorAgent_v1 is used and inputs are defined but target_file_path is missing.
            if stage_spec.agent_id == "SmartCodeGeneratorAgent_v1" and stage_spec.inputs and "target_file_path" not in stage_spec.inputs:
                self.logger.critical(f"CRITICAL_VALIDATION_ERROR (AUTO-FIXING): Stage '{stage_name}' ({stage_spec.agent_id}) is missing 'target_file_path' in 'inputs'. Injecting default.")
                stage_spec.inputs["target_file_path"] = f"AUTO-FIXED/placeholder/{stage_name}_missing_path.py"
                self.logger.warning(f"AUTO-FIX APPLIED for {stage_spec.agent_id} stage '{stage_name}'. New inputs: {stage_spec.inputs}")

            # AUTO-FIX: If CoreTestGeneratorAgent_v1 is used and inputs are defined but test_file_path is missing
            if stage_spec.agent_id == "CoreTestGeneratorAgent_v1" and stage_spec.inputs and "test_file_path" not in stage_spec.inputs:
                self.logger.critical(f"CRITICAL_VALIDATION_ERROR (AUTO-FIXING): Stage '{stage_name}' ({stage_spec.agent_id}) is missing 'test_file_path' in 'inputs'. Injecting default.")
                stage_spec.inputs["test_file_path"] = f"AUTO-FIXED/placeholder/{stage_name}_missing_tests.py"
                self.logger.warning(f"AUTO-FIX APPLIED for {stage_spec.agent_id} stage '{stage_name}'. New inputs: {stage_spec.inputs}")


        # Check if all referenced next_stages (that aren't FINAL_STEP) exist
        for next_stage_ref in referenced_next_stages:
            if next_stage_ref != "FINAL_STEP" and next_stage_ref not in all_stage_names:
                msg = f"MasterExecutionPlan {plan.id}: Stage '{next_stage_ref}' is referenced as a next_stage but does not exist in the plan stages."
                self.logger.error(msg)
                # Potentially raise ValueError here, or allow auto-linking to attempt a fix if applicable
                # For now, logging as an error. Stricter validation might be needed.

        # Second pass: Auto-link stages if next_stage is missing, based on sequential numbers
        # Sort stages by number to attempt sequential linking
        sorted_stages_by_number = sorted(plan.stages.items(), key=lambda item: item[1].number if item[1].number is not None else float('inf'))
        
        for i, (stage_name, stage_spec) in enumerate(sorted_stages_by_number):
            if not stage_spec.next_stage and stage_name != "FINAL_STEP": # Check if next_stage is None or empty
                # Try to find the next stage by number
                current_number = stage_spec.number
                if current_number is not None:
                    next_stage_candidate_name = None
                    # Look for the stage with number + 1
                    for next_s_name, next_s_spec in sorted_stages_by_number:
                        if next_s_spec.number == current_number + 1:
                            next_stage_candidate_name = next_s_name
                            break
                    
                    if next_stage_candidate_name:
                        self.logger.warning(
                            f"AUTO-FIX APPLIED: Stage '{stage_name}' (Number: {current_number}) was missing 'next_stage'. "
                            f"Automatically linking to stage '{next_stage_candidate_name}' (Number: {current_number + 1})."
                        )
                        stage_spec.next_stage = next_stage_candidate_name
                    elif i + 1 < len(sorted_stages_by_number):
                        # Fallback: if no direct number+1 match, link to the next in the sorted list if not FINAL_STEP
                        potential_next_name = sorted_stages_by_number[i+1][0]
                        if potential_next_name != "FINAL_STEP": # Avoid linking to FINAL_STEP unless it's truly the last
                             self.logger.warning(
                                f"AUTO-FIX APPLIED (Fallback): Stage '{stage_name}' (Number: {current_number}) was missing 'next_stage'. "
                                f"Automatically linking to next stage in sorted list: '{potential_next_name}'."
                            )
                             stage_spec.next_stage = potential_next_name
                        # If the next in list is FINAL_STEP, and we didn't find a number+1, we might be at the actual end.
                        # Or, if it's the last stage in the list and next_stage is still missing, it might implicitly be FINAL_STEP
                    elif i == len(sorted_stages_by_number) - 1: # Is it the last stage in the plan?
                         self.logger.warning(
                            f"AUTO-FIX APPLIED (Last Stage): Stage '{stage_name}' (Number: {current_number}) was missing 'next_stage' and is the last numbered stage. "
                            f"Setting next_stage to 'FINAL_STEP'."
                        )
                         stage_spec.next_stage = "FINAL_STEP"


        self.logger.info(f"MasterExecutionPlan ID: {plan.id} structure validation successful.")
