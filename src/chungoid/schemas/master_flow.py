from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
import yaml # For from_yaml method

from .user_goal_schemas import UserGoalRequest # <<< ADD IMPORT

# TODO: Potentially reference or reuse parts of StageSpec from orchestrator.py 
# if there's significant overlap and it makes sense.

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
    agent_id: str = Field(..., description="ID of the agent to invoke (e.g., 'CoreStageExecutorAgent')")
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
    condition: Optional[str] = Field(None, description="Condition for branching in the Master Flow.")
    next_stage_true: Optional[str] = Field(None, description="Next Master Flow stage if condition is true.")
    next_stage_false: Optional[str] = Field(None, description="Next Master Flow stage if condition is false.")
    next_stage: Optional[str] = Field(None, description="Next Master Flow stage (if no condition).")
    number: Optional[float] = Field(None, description="Unique stage number for status tracking within the Master Flow.")
    # on_error: Optional[Any] = Field(None, description="Error handling strategy for this master stage.") # For future?
    name: Optional[str] = Field(None, description="Optional human-readable name for this master stage step.")
    output_context_path: Optional[str] = Field(
        None, 
        description="Optional dot-notation path where the agent's entire output dictionary should be placed in the main flow context. E.g., 'stage_outputs.current_stage_name'. If None, output is merged at root (TBD)."
    )

class MasterExecutionPlan(BaseModel):
    """Validated, structured representation of a Master Flow YAML."""
    id: str = Field(..., description="Unique ID for this Master Execution Plan.")
    name: Optional[str] = Field(None, description="Human-readable name for the Master Flow.")
    description: Optional[str] = Field(None, description="Description of the Master Flow's purpose.")
    version: str = Field("1.0.0", description="Version of the Master Flow definition.")
    start_stage: str = Field(..., description="The first stage to execute in the Master Flow.")
    stages: Dict[str, MasterStageSpec] = Field(..., description="Dictionary of stage definitions for the Master Flow.")
    original_request: Optional[UserGoalRequest] = Field(None, description="The original UserGoalRequest that initiated this plan.") # <<< ADD FIELD

    @classmethod
    def from_yaml(cls, yaml_text: str) -> MasterExecutionPlan:
        """Parse the *yaml_text* of a Master Flow and convert it to a plan."""
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError("Master Flow YAML must map keys → values")

        # Basic structural validation (can be enhanced with jsonschema later if needed)
        if "id" not in data or "start_stage" not in data or "stages" not in data:
            raise ValueError(
                "Master Flow YAML missing required 'id', 'start_stage', or 'stages' key"
            )
        
        # TODO: Add more robust validation if a JSON schema for Master Flows is created.
        # For now, rely on Pydantic's validation during model instantiation.
        
        return cls(**data)

    def to_yaml(self) -> str:
        """Serializes the MasterExecutionPlan to a YAML string."""
        return yaml.dump(self.model_dump(exclude_none=True), sort_keys=False)


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