from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, Union
from pathlib import Path

from pydantic import BaseModel, Field, model_validator # IMPORT model_validator
import yaml # For from_yaml method

from .user_goal_schemas import UserGoalRequest # <<< ADD IMPORT
from .common_enums import OnFailureAction # <<< MODIFIED IMPORT: Removed AgentCategory here
from chungoid.utils.agent_registry_meta import AgentCategory # <<< CORRECTED IMPORT
from chungoid.schemas.common import ArbitraryModel, InputOutputContextPathStr # Ensure InputOutputContextPathStr is defined or imported
from chungoid.schemas.common_enums import FlowPauseStatus, StageStatus # <<< REMOVED RecoveryAction, StageOnFailureAction

# TODO: Potentially reference or reuse parts of StageSpec from orchestrator.py 
# if there's significant overlap and it makes sense.

class ConditionalTransition(BaseModel):
    """Defines a conditional transition to another stage."""
    condition: str = Field(..., description="The condition string to evaluate.")
    next_stage_id: str = Field(..., description="The ID of the stage to transition to if the condition is true.")

class MasterStageFailurePolicy(BaseModel):
    """Defines the policy for handling failures within a master stage."""
    action: OnFailureAction = Field(..., description="Action to take on stage failure.")
    target_master_stage_key: Optional[str] = Field(
        None, 
        description="The key of the master stage to transition to if action is GOTO_MASTER_STAGE."
    )
    log_message: Optional[str] = Field(
        None, 
        description="Optional custom message to log when this failure policy is enacted."
    )

class ClarificationCheckpointSpec(BaseModel):
    """Specification for a user clarification checkpoint."""
    prompt_message_for_user: str = Field(..., description="The message/question to present to the user for clarification.")
    target_context_path: Optional[str] = Field(
        None, 
        description="Optional dot-notation path in the context where the user's input should be placed. E.g., 'stage_inputs.parameter_name'."
    )
    expected_input_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional JSON schema defining the expected structure of the user's input JSON."
    )

class MasterStageSpec(BaseModel):
    """Specification of a single stage within a Master Execution Plan."""
    id: str # Unique identifier for the stage within the plan
    name: str
    description: Optional[str] = None
    agent_id: Optional[str] = Field(None, description="ID of the agent to invoke (e.g., 'CoreStageExecutorAgent')")
    agent_category: Optional[AgentCategory] = Field(None, description="Category of agent to invoke if agent_id is not specified.")
    agent_selection_preferences: Optional[Dict[str, Any]] = Field(None, description="Preferences for selecting an agent from agent_category. Example: {'capability_profile_match': {'language': 'python'}, 'priority_gte': 5}")
    inputs: Optional[Dict[str, Any]] = Field(
        None, 
        description=(
            "Input parameters for the agent. For CoreStageExecutorAgent, "
            "this must include 'stage_definition_path'."
        )
    )
    success_criteria: Optional[List[str]] = Field(
        default_factory=list, 
        description=(
            "List of conditions that must be true after successful agent execution for the stage to be fully successful. "
            "E.g., [\"outputs.file_generated_path EXISTS\", \"outputs.analysis_metric > 0.9\"]"
        )
    )
    clarification_checkpoint: Optional[ClarificationCheckpointSpec] = Field(
        None, 
        description=(
            "If set, the orchestrator will pause after this stage (if successful and criteria pass) "
            "for user clarification."
        )
    )
    conditional_transitions: Optional[List[ConditionalTransition]] = Field(
        None,
        description="A list of conditional transitions. If conditions are met, these take precedence over next_stage_id."
    )
    on_failure: Optional[MasterStageFailurePolicy] = Field(
        None, 
        description="Optional policy to define behavior if this master stage fails (e.g., agent resolution error, or agent execution error not handled by reviewer)."
    )
    on_success: Optional[MasterStageFailurePolicy] = Field(
        None,
        description="Optional policy to define behavior after this master stage succeeds (e.g., invoke reviewer for post-success check)."
    )
    condition: Optional[str] = Field(None, description="Condition for branching in the Master Flow.")
    next_stage_true: Optional[str] = Field(None, description="Next Master Flow stage if condition is true.")
    next_stage_false: Optional[str] = Field(None, description="Next Master Flow stage if condition is false.")
    next_stage: Optional[str] = Field(None, description="Next Master Flow stage (if no condition).")
    number: Optional[float] = Field(None, description="Unique stage number for status tracking within the Master Flow.")
    max_retries: Optional[int] = Field(None, description="Maximum number of retries for this stage if it fails and a retry mechanism is configured (e.g., via on_failure or reviewer action).")
    # on_error: Optional[Any] = Field(None, description="Error handling strategy for this master stage.") # For future?
    output_context_path: Optional[str] = Field(
        None, 
        description="Optional dot-notation path where the agent's entire output dictionary should be placed in the main flow context. E.g., 'stage_outputs.current_stage_name'. If None, output is merged at root (TBD)."
    )

    @model_validator(mode='after')
    def check_agent_id_or_category_provided(cls, data: Any) -> Any:
        if isinstance(data, MasterStageSpec): # Ensure it's already a model instance for attribute access
            if not data.agent_id and not data.agent_category:
                raise ValueError("Either 'agent_id' or 'agent_category' must be provided for a stage.")
            # Orchestrator prioritizes agent_id if both are present, so no specific validation needed for that case here.
        return data

    @model_validator(mode='after')
    def check_conditional_transitions_or_next_stage(cls, data: Any) -> Any:
        if isinstance(data, MasterStageSpec): # Ensure it's already a model instance
            has_conditional_transitions = bool(data.conditional_transitions)
            has_simple_condition = bool(data.condition and (data.next_stage_true or data.next_stage_false))
            has_direct_next = bool(data.next_stage)

            # Warn if old (condition/next_stage_true/false) and new (conditional_transitions) are mixed
            if has_conditional_transitions and has_simple_condition:
                # Consider this a warning, orchestrator will prioritize conditional_transitions
                # Ideally, plans should use one or the other for a given stage.
                # logger.warning(f"Stage '{data.id}' mixes conditional_transitions with condition/next_stage_true/false. Prioritizing conditional_transitions.")
                pass # For now, just allow, orchestrator handles precedence

            # A stage should generally have some way to move forward unless it's explicitly an end stage.
            # This validation might be too strict if null next_stage_id is a valid end-of-flow marker.
            # if not has_conditional_transitions and not has_simple_condition and not has_direct_next:
            #     logger.warning(f"Stage '{data.id}' has no defined next stage or conditional transitions. This might be an intended end of a path.")
        return data

class EnhancedMasterStageSpec(BaseModel):
    """Enhanced stage specification for task-type based autonomous orchestration."""
    id: str
    name: str
    description: str
    
    # NEW: Task-type based specification (CORE TRANSFORMATION)
    task_type: str = Field(..., description="Primary task type from autonomous vocabulary")
    required_capabilities: List[str] = Field(..., description="Required agent capabilities")
    preferred_execution: Literal["autonomous", "concrete", "any"] = Field(default="autonomous", description="Execution mode preference")
    
    # ENHANCED: Agent selection (backward compatibility)
    agent_id: Optional[str] = Field(None, description="Specific agent ID (for concrete agents or fallback)")
    fallback_agent_id: Optional[str] = Field(None, description="Fallback agent if autonomous unavailable")
    
    # STANDARD: Input/output configuration
    inputs: Dict[str, Any] = Field(default_factory=dict)
    output_context_path: Optional[str] = Field(None)
    success_criteria: List[str] = Field(default_factory=list, description="Autonomous completion criteria")
    next_stage: Optional[str] = Field(None)
    
    # COMPATIBILITY: Legacy fields for backward compatibility
    number: Optional[float] = Field(None, description="Unique stage number for status tracking")
    condition: Optional[str] = Field(None, description="Condition for branching")
    next_stage_true: Optional[str] = Field(None, description="Next stage if condition is true")
    next_stage_false: Optional[str] = Field(None, description="Next stage if condition is false")
    clarification_checkpoint: Optional[ClarificationCheckpointSpec] = Field(None)
    conditional_transitions: Optional[List[ConditionalTransition]] = Field(None)
    on_failure: Optional[MasterStageFailurePolicy] = Field(None)
    on_success: Optional[MasterStageFailurePolicy] = Field(None)
    max_retries: Optional[int] = Field(None)

class EnhancedMasterExecutionPlan(BaseModel):
    """Enhanced execution plan for task-type based autonomous orchestration."""
    id: str
    name: str
    description: str
    version: str = Field(default="2.0.0", description="Enhanced autonomous version")
    
    # NEW: Task-type based stages
    stages: Dict[str, EnhancedMasterStageSpec] = Field(...)
    initial_stage: str = Field(..., description="First stage to execute (replaces start_stage)")
    
    # COMPATIBILITY: Legacy fields for backward compatibility
    start_stage: Optional[str] = Field(None, description="Legacy field, use initial_stage instead")
    project_id: Optional[str] = Field(None, description="The ID of the project this plan belongs to")
    global_config: Optional[Dict[str, Any]] = Field(None, description="Global configuration for the plan")
    original_request: Optional[UserGoalRequest] = Field(None, description="The original UserGoalRequest that initiated this plan")
    file_path: Optional[Path] = Field(None, description="Optional path to the file from which this plan was loaded")

    @model_validator(mode='after')
    def ensure_start_stage_compatibility(cls, data: Any) -> Any:
        """Ensure backward compatibility between start_stage and initial_stage."""
        if isinstance(data, EnhancedMasterExecutionPlan):
            if data.start_stage and not data.initial_stage:
                data.initial_stage = data.start_stage
            elif data.initial_stage and not data.start_stage:
                data.start_stage = data.initial_stage
        return data

class MasterExecutionPlan(BaseModel):
    """Validated, structured representation of a Master Flow YAML."""
    id: str = Field(..., description="Unique ID for this Master Execution Plan.")
    project_id: Optional[str] = Field(None, description="The ID of the project this plan belongs to.")
    name: Optional[str] = Field(None, description="Human-readable name for the Master Flow.")
    description: Optional[str] = Field(None, description="Description of the Master Flow's purpose.")
    version: str = Field("1.0.0", description="Version of the Master Flow definition.")
    start_stage: str = Field(..., description="The first stage to execute in the Master Flow.")
    stages: Dict[str, MasterStageSpec] = Field(..., description="Dictionary of stage definitions for the Master Flow.")
    original_request: Optional[UserGoalRequest] = Field(None, description="The original UserGoalRequest that initiated this plan.") # <<< ADD FIELD
    file_path: Optional[Path] = Field(None, description="Optional path to the file from which this plan was loaded.") # ADDED

    @classmethod
    def from_yaml(cls, yaml_text: str) -> MasterExecutionPlan:
        """Parse the *yaml_text* of a Master Flow and convert it to a plan."""
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError("Master Flow YAML must map keys → values")

        if "id" not in data or "start_stage" not in data or "stages" not in data:
            raise ValueError(
                "Master Flow YAML missing required 'id', 'start_stage', or 'stages' key"
            )
        
        # Inject the dictionary key as the 'id' field for each stage spec
        stages_dict = data.get("stages", {})
        if isinstance(stages_dict, dict):
            for stage_key, stage_spec_dict in stages_dict.items():
                if isinstance(stage_spec_dict, dict):
                    stage_spec_dict["id"] = stage_key # Set/overwrite 'id' with the key
            data["stages"] = stages_dict # Ensure data reflects these changes
        
        return cls(**data)

    def to_yaml(self) -> str:
        """Serializes the MasterExecutionPlan to a YAML string."""
        return yaml.dump(self.model_dump(exclude_none=True), sort_keys=False)

__all__ = [
    "MasterStageFailurePolicy",
    "ClarificationCheckpointSpec",
    "MasterStageSpec",
    "MasterExecutionPlan",
    "ConditionalTransition",
    "EnhancedMasterStageSpec",
    "EnhancedMasterExecutionPlan"
]

# Example Usage (for testing or reference)
if __name__ == "__main__":
    example_master_flow_yaml = """
id: basic_project_build
name: Basic Project Build Master Flow
description: Orchestrates Stage 0 and Stage 1 using the CoreStageExecutorAgent.
version: 1.0.1
start_stage: run_stage_0_via_executor

stages:
  run_stage_0_via_executor:
    name: "Execute Project Requirements Stage (Stage 0)"
    agent_id: CoreStageExecutorAgent
    number: 0.0
    inputs:
      stage_definition_path: "chungoid-core/server_prompts/stages/stage0.yaml"
      initial_project_brief: "Build a simple CLI tool for managing tasks."
    next_stage: run_stage_1_via_executor

  run_stage_1_via_executor:
    name: "Execute Project Planning Stage (Stage 1)"
    agent_id: CoreStageExecutorAgent
    number: 1.0
    inputs:
      stage_definition_path: "chungoid-core/server_prompts/stages/stage1.yaml"
      # Context from run_stage_0_via_executor will be available through the orchestrator.
      # Additional static inputs specific to this master stage step can be added here.
      planning_focus: "core_features_only"
    next_stage: evaluate_plan

  evaluate_plan:
    name: "Evaluate Plan Condition"
    agent_id: ConditionEvaluatorAgent # Hypothetical agent for complex conditions
    number: 1.5
    inputs:
      plan_details: "context.outputs.run_stage_1_via_executor.plan_document"
    condition: "outputs.evaluate_plan.evaluation_result == 'APPROVE'"
    next_stage_true: finalize_project
    next_stage_false: revise_plan

  revise_plan:
    name: "Revise Plan (if needed)"
    agent_id: CoreStageExecutorAgent # Could re-run stage 1 with different inputs
    number: 1.6
    inputs:
      stage_definition_path: "chungoid-core/server_prompts/stages/stage1.yaml"
      feedback: "context.outputs.evaluate_plan.feedback_notes"
      revision_mode: True
    next_stage: evaluate_plan # Loop back to re-evaluate

  finalize_project:
    name: "Finalize Project (Placeholder)"
    agent_id: FinalizationAgent
    number: 2.0
    inputs:
      final_plan: "context.outputs.run_stage_1_via_executor.plan_document"
    next_stage: null
"""

    try:
        plan = MasterExecutionPlan.from_yaml(example_master_flow_yaml)
        print("Successfully parsed MasterExecutionPlan:")
        print(plan.model_dump_json(indent=2))

        print("\nSerialized back to YAML:")
        print(plan.to_yaml())
    except ValueError as e:
        print(f"Error parsing Master Flow YAML: {e}") 