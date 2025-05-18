from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

class SharedContext(BaseModel):
    """
    A Pydantic model representing the shared context passed between agents/stages
    managed by the AsyncOrchestrator.
    """
    project_id: str = Field(..., description="The unique identifier for the current project.")
    project_root_path: str = Field(..., description="The absolute string path to the project's root directory.")
    
    current_cycle_id: Optional[str] = Field(None, description="Identifier for the current iteration or operational cycle.")
    current_stage_name: Optional[str] = Field(None, description="Name of the currently executing stage in the workflow.")
    
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

# Example of how it might be initialized or used by the orchestrator:
# context = SharedContext(project_id="proj_123", project_root_path="/path/to/project")
# context.current_cycle_id = "cycle_001"
# context.update_artifact_reference("initial_requirements_id", "req_abc") 