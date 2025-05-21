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

                    # loprd_doc_id comes from the output of the 'goal_analysis' stage
                    # This assumes 'goal_analysis' stage ran and its output structure.
                    goal_analysis_output = self.shared_context.previous_stage_outputs.get("goal_analysis")
                    loprd_doc_id_for_architect = None
                    if isinstance(goal_analysis_output, dict):
                        loprd_doc_id_for_architect = goal_analysis_output.get("refined_requirements_document_id")
                    elif hasattr(goal_analysis_output, "refined_requirements_document_id"): # If it's an object
                        loprd_doc_id_for_architect = getattr(goal_analysis_output, "refined_requirements_document_id")
                    
                    if not loprd_doc_id_for_architect:
                        # Fallback or error if loprd_doc_id is crucial and not found
                        # For now, we'll proceed, but the agent might fail if it's required and missing.
                        self.logger.warning(f"loprd_doc_id not found in 'goal_analysis' stage output for ArchitectAgent_v1. Inputs provided to Architect: {final_inputs_for_agent}")
                        # If final_inputs_for_agent (from plan) contains loprd_doc_id, use that as a fallback
                        loprd_doc_id_from_plan = final_inputs_for_agent.get("loprd_doc_id")
                        if loprd_doc_id_from_plan:
                            loprd_doc_id_for_architect = loprd_doc_id_from_plan
                            self.logger.info(f"Using loprd_doc_id from plan inputs as fallback: {loprd_doc_id_for_architect}")
                        else:
                             # If it's absolutely required and not found, we might need to raise an error earlier or
                             # the ArchitectAgent itself will fail. For now, pass None if not found.
                             self.logger.error("Critical: loprd_doc_id could not be determined for ArchitectAgent_v1. The agent will likely fail.")


                    # Create the ArchitectAgentInput object
                    # final_inputs_for_agent might contain other optional fields like existing_blueprint_doc_id
                    architect_task_input_data = {
                        "project_id": project_id_for_architect,
                        "loprd_doc_id": loprd_doc_id_for_architect, # This might be None if not found
                        **final_inputs_for_agent # Spread other inputs from plan (e.g., existing_blueprint_doc_id)
                    }
                    # task_id has a default factory in ArchitectAgentInput

                    # Remove None values if ArchitectAgentInput fields are not Optional and have no defaults
                    # For ArchitectAgentInput, loprd_doc_id is mandatory.
                    if architect_task_input_data.get("loprd_doc_id") is None:
                        # This will cause a validation error if loprd_doc_id is not Optional in the Pydantic model
                        # and is required. The agent's __init__ or invoke will fail.
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

    async def _execute_flow_loop(
        self,
        run_id: str,
        flow_id: str,
        start_stage_name: str,
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
                # The following line was a bug from a previous merge/edit, current_stage_name is not available here if current_plan is None.
                # flow_error_details = f"Stage '{current_stage_name}' not found in plan."
                break

            # --- MOVED: stage_spec definition before its use ---
            stage_spec = self.current_plan.stages.get(current_stage_name)
            if not stage_spec:
                self.logger.error(f"Stage '{current_stage_name}' not found in plan '{flow_id}' for run {run_id}. Terminating.")
                final_status = StageStatus.FAILURE
                flow_error_details = f"Stage '{current_stage_name}' not found in plan."
                break
            # --- END MOVED ---

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
                        current_stage_name=current_stage_name,
                        flow_id=flow_id,
                        run_id=run_id,
                        current_plan=self.current_plan, # Use self.current_plan
                        agent_id_for_error=resolved_agent_id,
                        error=agent_error_obj, # This is the AgentErrorDetails for SuccessCriteriaFailed
                        attempt_number=max_retries_for_stage + 1, # Ensure it doesn't retry via _invoke_agent's loop
                        current_shared_context=self.shared_context # Pass shared_context
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

                    # BEGIN: Automatic artifact materialization for SmartCodeGeneratorAgent_v1
                    if resolved_agent_id == "SmartCodeGeneratorAgent_v1":
                        self.logger.info(f"Run {run_id}: Stage '{current_stage_name}' used SmartCodeGeneratorAgent_v1. Attempting automatic artifact materialization.")
                        # MODIFIED: Check if stage_output is an instance of SmartCodeGeneratorAgentOutput (or has the needed attributes)
                        if hasattr(stage_output, 'generated_code_artifact_doc_id') and hasattr(stage_output, 'target_file_path'):
                            artifact_doc_id = stage_output.generated_code_artifact_doc_id
                            target_file_path_from_output = stage_output.target_file_path

                            if artifact_doc_id and target_file_path_from_output:
                                self.logger.info(f"Run {run_id}: Attempting to write artifact '{artifact_doc_id}' to '{target_file_path_from_output}'.")
                                try:
                                    pcma_agent_instance = self.agent_provider._project_chroma_manager # Relies on RegistryAgentProvider
                                    if not pcma_agent_instance:
                                        raise ValueError("ProjectChromaManagerAgent could not be retrieved from agent_provider for artifact materialization.")

                                    # SystemFileSystemAgent_v1 expects system_context as a dict for BaseAgent init
                                    system_context_for_fs_dict = {
                                        "project_root": Path(self.shared_context.project_root_path),
                                        "logger": self.logger, # Pass orchestrator's logger for now
                                        "run_id": run_id
                                        # llm_provider and prompt_manager are not directly used by file ops
                                    }
                                    
                                    fs_agent = SystemFileSystemAgent_v1(
                                        system_context=system_context_for_fs_dict, 
                                        pcma_agent=pcma_agent_instance
                                    )
                                    
                                    tool_call_input = WriteArtifactToFileInput(
                                        artifact_doc_id=artifact_doc_id,
                                        collection_name=GENERATED_CODE_ARTIFACTS_COLLECTION, # MODIFIED: Use the correct constant
                                        target_file_path=target_file_path_from_output,
                                        overwrite=True # Default to overwrite for generated code
                                    )
                                    
                                    # write_artifact_to_file_tool needs project_root explicitly passed
                                    write_result_dict = await fs_agent.write_artifact_to_file_tool(
                                        artifact_doc_id=tool_call_input.artifact_doc_id,
                                        collection_name=tool_call_input.collection_name,
                                        target_file_path=tool_call_input.target_file_path,
                                        overwrite=tool_call_input.overwrite,
                                        project_root=Path(self.shared_context.project_root_path)
                                    )

                                    if write_result_dict.get("success"):
                                        self.logger.info(f"Run {run_id}: Successfully materialized artifact '{artifact_doc_id}' to '{target_file_path_from_output}'.")
                                    else:
                                        self.logger.error(f"Run {run_id}: Failed to materialize artifact '{artifact_doc_id}' to '{target_file_path_from_output}'. Error: {write_result_dict.get('error')}")
                                except Exception as e_materialize:
                                    self.logger.error(f"Run {run_id}: Exception during artifact materialization for '{artifact_doc_id}': {e_materialize}", exc_info=True)
                            else:
                                self.logger.warning(f"Run {run_id}: SmartCodeGeneratorAgent_v1 output for stage '{current_stage_name}' missing 'generated_code_artifact_doc_id' or 'target_file_path' values. Cannot materialize.")
                        else:
                            self.logger.warning(f"Run {run_id}: SmartCodeGeneratorAgent_v1 output for stage '{current_stage_name}' (type: {type(stage_output)}) does not have expected attributes for materialization. Cannot materialize.")
                    # END: Automatic artifact materialization

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
                    current_stage_name=current_stage_name,
                    flow_id=flow_id,
                    run_id=run_id,
                    current_plan=self.current_plan, # Use self.current_plan
                    agent_id_for_error=agent_id_to_invoke, # This was the specifier
                    error=agent_error_obj, # This is the AgentErrorDetails for resolution failure
                    attempt_number=1, # No retries for resolution failure
                    current_shared_context=self.shared_context # Pass shared_context
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
                    current_stage_name=current_stage_name,
                    flow_id=flow_id,
                    run_id=run_id,
                    current_plan=self.current_plan, # Use self.current_plan
                    agent_id_for_error=current_agent_id_for_err,
                    error=e_invoke, 
                    attempt_number=max_retries_for_stage + 1, # Signify that agent invocation retries (if any) are done
                    current_shared_context=self.shared_context # Pass shared_context
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

        # Ensure previous_stage_outputs is a dict (it should be)
        if self.shared_context.previous_stage_outputs is None:
            self.shared_context.previous_stage_outputs = {}
            self.logger.warning(f"Run {run_id} flow {flow_id}: previous_stage_outputs was None at the end of _execute_flow_loop. Initialized to empty dict before adding status.")

        self.shared_context.previous_stage_outputs["_orchestrator_final_status"] = final_status.value
        if flow_error_details: # flow_error_details might be None if successful
            self.shared_context.previous_stage_outputs["_orchestrator_flow_error_details"] = flow_error_details
        elif "_orchestrator_flow_error_details" in self.shared_context.previous_stage_outputs:
            # Ensure the key is not present if there are no error details
            del self.shared_context.previous_stage_outputs["_orchestrator_flow_error_details"]
        
        return self.shared_context.previous_stage_outputs


    def _resolve_input_values(self, inputs_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves input values based on the input specification.
        This is a basic implementation and will need to be expanded.
        It currently handles direct string passthrough and very simple context lookups.
        """
        if not inputs_spec:
            return {}

        resolved_inputs: Dict[str, Any] = {}
        self.logger.debug(f"Resolving input_spec: {inputs_spec}")

        for key, value_spec in inputs_spec.items():
            if isinstance(value_spec, str):

                # --- ADDED: Handle specific known context path patterns via direct string replacement ---
                # This should run before other prefix checks if those prefixes might be part of the replacement strings.
                temp_value_spec = value_spec
                replacement_made = False

                # Pattern 1: {context.project_root_path}
                if "{context.project_root_path}" in temp_value_spec and hasattr(self.shared_context, 'project_root_path') and self.shared_context.project_root_path:
                    temp_value_spec = temp_value_spec.replace("{context.project_root_path}", str(self.shared_context.project_root_path))
                    self.logger.debug(f"Replaced '{{context.project_root_path}}' in '{value_spec}' with '{self.shared_context.project_root_path}' -> '{temp_value_spec}'")
                    replacement_made = True
                
                # Pattern 2: {context.global_config.project_dir} (maps to project_root_path)
                # This is based on how the MasterPlannerAgent seems to structure its `global_config` output.
                # `global_config` in the plan often corresponds to `global_project_settings` in SharedContext,
                # and `project_dir` within that is typically the main project root path.
                if "{context.global_config.project_dir}" in temp_value_spec and hasattr(self.shared_context, 'project_root_path') and self.shared_context.project_root_path:
                    temp_value_spec = temp_value_spec.replace("{context.global_config.project_dir}", str(self.shared_context.project_root_path))
                    self.logger.debug(f"Replaced '{{context.global_config.project_dir}}' in '{value_spec}' with '{self.shared_context.project_root_path}' -> '{temp_value_spec}'")
                    replacement_made = True
                
                value_spec_after_replacement = temp_value_spec # Use the potentially modified string for subsequent checks
                # --- END ADDED ---

                if value_spec_after_replacement.startswith("@outputs."):
                    # Attempt to resolve from previous stage outputs
                    path = value_spec_after_replacement[len("@outputs."):]
                    current_val = self.shared_context.previous_stage_outputs
                    try:
                        for part in path.split("."):
                            if isinstance(current_val, dict):
                                current_val = current_val[part]
                            elif hasattr(current_val, part): # For Pydantic models
                                current_val = getattr(current_val, part)
                            else:
                                raise KeyError(f"Path part '{part}' not found in outputs context for '{path}'")
                        resolved_inputs[key] = current_val
                        self.logger.debug(f"Resolved input '{key}' from @outputs.{path} to: {current_val}")
                    except Exception as e:
                        self.logger.warning(f"Could not resolve @outputs.{path} for input '{key}': {e}. Using None.")
                        resolved_inputs[key] = None
                elif value_spec_after_replacement.startswith("@context."): # placeholder for other context types
                    self.logger.warning(f"Resolution for context type other than @outputs (e.g., {value_spec_after_replacement}) not fully implemented for key '{key}'. Using None.")
                    resolved_inputs[key] = None # Placeholder
                elif value_spec_after_replacement.startswith("@artifact."):
                     self.logger.warning(f"Resolution for @artifact not implemented for key '{key}'. Using None.")
                     resolved_inputs[key] = None
                elif value_spec_after_replacement.startswith("@config."):
                    # Attempt to resolve from orchestrator's self.config or shared_context.global_project_settings
                    path = value_spec_after_replacement[len("@config."):]
                    config_source = self.config # Default to orchestrator config
                    if path.startswith("global_project_settings."):
                        config_source = self.shared_context.global_project_settings
                        path = path[len("global_project_settings."):]
                    
                    current_val = config_source
                    try:
                        for part in path.split("."):
                            current_val = current_val[part] # Assume dict access
                        resolved_inputs[key] = current_val
                        self.logger.debug(f"Resolved input '{key}' from @config.{path} to: {current_val}")
                    except Exception as e:
                        self.logger.warning(f"Could not resolve @config.{path} for input '{key}': {e}. Using None.")
                        resolved_inputs[key] = None
                else:
                    # Assume it's a literal string value (after potential direct context replacements)
                    resolved_inputs[key] = value_spec_after_replacement
                    self.logger.debug(f"Input '{key}' is a literal string (after replacements): {value_spec_after_replacement}")
            else:
                # If not a string, pass it through as is (e.g., bool, int, list, dict literals)
                resolved_inputs[key] = value_spec
                self.logger.debug(f"Input '{key}' is a literal of type {type(value_spec)}: {value_spec}")
        
        self.logger.debug(f"Resolved inputs: {resolved_inputs}")
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