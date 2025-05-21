from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from chungoid.schemas.common_enums import StageStatus
from chungoid.schemas.master_flow import MasterExecutionPlan

class SharedContext(BaseModel):
    """
    A Pydantic model representing the shared context passed between agents/stages
    managed by the AsyncOrchestrator.
    """
    project_id: str = Field(..., description="The unique identifier for the current project.")
    project_root_path: str = Field(..., description="The absolute string path to the project's root directory.")
    
    run_id: Optional[str] = Field(None, description="The unique identifier for the current execution run of a flow.")
    flow_id: Optional[str] = Field(None, description="The unique identifier for the specific flow being executed.")

    initial_goal_str: Optional[str] = Field(None, description="The initial user goal string that triggered the flow run, if applicable.")
    current_master_plan: Optional[MasterExecutionPlan] = Field(None, description="The active MasterExecutionPlan for the current run.")

    current_cycle_id: Optional[str] = Field(None, description="Identifier for the current iteration or operational cycle.")
    current_stage_id: Optional[str] = Field(None, description="ID of the currently executing stage in the workflow.")
    current_stage_status: Optional[StageStatus] = Field(None, description="Status of the currently executing stage.")

    initial_inputs: Dict[str, Any] = Field(default_factory=dict, description="The initial inputs provided to the flow run.")

    previous_stage_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs from previously completed stages. Keys are stage names (or agent IDs), values are their outputs (e.g., artifact IDs, status messages)."
    )
    
    artifact_references: Dict[str, str] = Field(
        default_factory=dict,
        description="Named references to key artifact IDs. E.g., {'current_blueprint_id': 'uuid_xyz'}"
    )

    global_project_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Global settings applicable to the project or all agents (e.g., LLM preferences, verbosity levels)."
    )
    
    scratchpad: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary for temporary data sharing between agents or within a complex stage or cycle. Should be used judiciously."
    )
    
    human_feedback_for_current_cycle: Optional[Any] = Field(
        None, 
        description="Structured human feedback and directives relevant to the current operational cycle, if provided."
    )

    # Consider adding fields for:
    # - overall_project_status (from ProjectStateV2)
    # - access to StateManager instance (though this might be better managed by the orchestrator itself)
    # - access to ProjectChromaManagerAgent instance

    class Config:
        arbitrary_types_allowed = True # If Path objects or other non-standard types are directly used. For string paths, it's not strictly needed.
        validate_assignment = True

    def update_artifact_reference(self, name: str, artifact_id: str):
        """Helper to add or update an artifact reference."""
        self.artifact_references[name] = artifact_id

    def get_artifact_reference(self, name: str) -> Optional[str]:
        """Helper to retrieve an artifact reference."""
        return self.artifact_references.get(name)

    def update_previous_stage_output(self, stage_name: str, output: Any):
        """Helper to record the output of a completed stage."""
        self.previous_stage_outputs[stage_name] = output

    def get_previous_stage_output(self, stage_name: str) -> Optional[Any]:
        """Helper to retrieve the output of a previous stage."""
        return self.previous_stage_outputs.get(stage_name)

    def set_scratchpad_data(self, key: str, value: Any):
        """Sets data in the scratchpad."""
        self.scratchpad[key] = value

    def get_scratchpad_data(self, key: str) -> Optional[Any]:
        """Gets data from the scratchpad."""
        return self.scratchpad.get(key)

class SystemContext(BaseModel):
    """Context providing system-level utilities and information to agents."""
    project_root: Path = Field(..., description="The absolute root path of the project workspace.")
    logger: Any = Field(..., description="A logger instance for agents to use.") # Can be more specific if a base logger type exists
    run_id: Optional[str] = Field(None, description="The unique identifier for the current execution run.")

    class Config:
        arbitrary_types_allowed = True # For Path and logger objects

__all__ = ["SharedContext", "SystemContext"] # Add SystemContext to __all__

# Example of how it might be initialized or used by the orchestrator:
# context = SharedContext(project_id="proj_123", project_root_path="/path/to/project")
# context.current_cycle_id = "cycle_001"
# context.update_artifact_reference("initial_requirements_id", "req_abc") 