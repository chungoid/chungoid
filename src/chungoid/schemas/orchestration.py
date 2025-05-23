#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, List

from pydantic import BaseModel, Field, ConfigDict

from chungoid.schemas.common_enums import StageStatus, FlowPauseStatus, ResumeActionType
from chungoid.schemas.master_flow import MasterExecutionPlan
from chungoid.schemas.project_status_schema import ProjectStateV2, ArtifactDetails

class SharedContext(BaseModel):
    """
    Holds the shared data and state across stages in a single flow run.
    It's an isolated container for each run, evolving as the flow progresses.
    """
    run_id: str = Field(..., description="Unique identifier for the flow run this context belongs to.")
    flow_id: str = Field(..., description="Identifier of the master flow definition.")
    
    data: Dict[str, Any] = Field(default_factory=dict, description="The core data dictionary where stage outputs and other contextual information are stored. Keys are typically strings (e.g., 'user_query', 'retrieved_documents', 'stage1_output').")
    
    # System-managed state within the context
    current_stage_id: Optional[str] = Field(None, description="The ID of the stage currently being processed or that was last processed before a potential pause.")
    current_stage_status: Optional[StageStatus] = Field(None, description="The status of the stage currently being processed.")
    current_attempt_number_for_stage: Optional[int] = Field(None, description="Current attempt number for the active stage.")
    last_successful_stage_id: Optional[str] = Field(None, description="The ID of the most recent stage that completed successfully.")
    last_successful_stage_output: Optional[Any] = Field(None, description="The direct output of the last successfully completed stage.")
    flow_has_warnings: bool = Field(default=False, description="Indicates if any stage in the flow has completed with warnings (e.g., via PROCEED_AS_IS). Set by the orchestrator.")

    # Method stubs for get/set value, to be implemented by ContextResolutionService primarily
    # Orchestrator will interact with ContextResolutionService, which then uses SharedContext.data
    # def get_value(self, path: str, default: Optional[Any] = None) -> Any:
    #     pass 

    # def set_value(self, path: str, value: Any):
    #     pass
            
    def update_data(self, new_data: Dict[str, Any]):
        """Merges new_data into the existing context data. This is a shallow update."""
        if self.data is None: # Should not happen with default_factory=dict
            self.data = {}
        self.data.update(new_data)

    def update_resolved_inputs_for_current_stage(self, resolved_inputs: Dict[str, Any]):
        """Stores the resolved inputs for the current stage into the context data."""
        if self.data is None:
            self.data = {}
        self.data["_current_stage_resolved_inputs"] = resolved_inputs

    def get_resolved_inputs_for_current_stage(self) -> Optional[Dict[str, Any]]:
        """Retrieves the resolved inputs for the current stage from the context data."""
        if self.data is None:
            return None
        return self.data.get("_current_stage_resolved_inputs")

    def update_current_stage_output(self, stage_output: Any):
        """Stores the output of the most recently completed stage."""
        if self.data is None: # Should not happen with default_factory=dict
            self.data = {}
        self.data["_current_stage_output"] = stage_output

    def get_current_stage_output(self) -> Optional[Any]:
        """Retrieves the output of the most recently completed stage."""
        if self.data is None:
            return None
        return self.data.get("_current_stage_output")

    def add_stage_output_to_history(self, stage_id: str, stage_output: Any):
        """Adds the output of a completed stage to the historical outputs, typically under 'outputs.{stage_id}'."""
        if self.data is None:
            self.data = {}
        if "outputs" not in self.data or not isinstance(self.data["outputs"], dict):
            self.data["outputs"] = {}
        self.data["outputs"][stage_id] = stage_output
        
    def get_historical_stage_output(self, stage_id: str) -> Optional[Any]:
        """Retrieves the historical output of a specific stage."""
        if self.data is None or "outputs" not in self.data or not isinstance(self.data["outputs"], dict):
            return None
        return self.data["outputs"].get(stage_id)

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

class SystemContext(BaseModel):
    """Context providing system-level utilities and information to agents."""
    project_root: Path = Field(..., description="The absolute root path of the project workspace.")
    logger: Any = Field(..., description="A logger instance for agents to use.") # Can be more specific if a base logger type exists
    run_id: Optional[str] = Field(None, description="The unique identifier for the current execution run.")

    class Config:
        arbitrary_types_allowed = True # For Path and logger objects

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

class ResumeContext(BaseModel):
    """
    Defines the necessary information to resume a paused flow.
    This context is typically constructed based on user input or reviewer agent suggestions.
    """
    run_id: str = Field(..., description="The ID of the flow run to resume.")
    resume_action: ResumeActionType = Field(..., description="The action to take for resuming the flow (e.g., RETRY_STAGE, SKIP_STAGE).")
    
    target_stage_id: Optional[str] = Field(None, description="The ID of the stage to target for the resume action (e.g., for retry, skip, or branch). Relevant for most actions except perhaps ABORT_FLOW.")
    
    # For RETRY_STAGE_WITH_CHANGES or similar actions
    modified_stage_inputs: Optional[Dict[str, Any]] = Field(None, description="New or modified inputs specifically for the target_stage_id if the action involves retrying with different inputs.")
    modified_shared_context_data: Optional[Dict[str, Any]] = Field(None, description="Specific key-value pairs to update/add in the SharedContext.data before resuming. Useful for broader context adjustments.")
    
    # For actions involving significant re-planning by a reviewer agent
    new_master_plan: Optional[MasterExecutionPlan] = Field(None, description="A completely new or modified MasterExecutionPlan to use upon resumption.")
    
    # For BRANCH_TO_STAGE action
    branch_to_stage_id: Optional[str] = Field(None, description="If resume_action is BRANCH_TO_STAGE, this specifies the ID of the stage to jump to.")
    
    # For CLARIFICATION_PROVIDED action
    user_provided_clarification_data: Optional[Dict[str, Any]] = Field(None, description="Data provided by a user in response to a clarification checkpoint, to be injected into shared context.")
    
    notes: Optional[str] = Field(None, description="Optional notes from the user or system performing the resume action, for audit or informational purposes.")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True
    )

__all__ = ["ResumeContext", "SharedContext", "ClarificationCheckpointSpec"]

# Example of how it might be initialized or used by the orchestrator:
# context = SharedContext(project_id="proj_123", project_root_path="/path/to/project")
# context.current_cycle_id = "cycle_001"
# context.update_artifact_reference("initial_requirements_id", "req_abc") 