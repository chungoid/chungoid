#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .errors import AgentErrorDetails # Import for use in PausedRunDetails
from chungoid.schemas.common_enums import FlowPauseStatus # Added import
from chungoid.schemas.master_flow import MasterExecutionPlan # Added for current_master_plan_snapshot
from chungoid.schemas.orchestration import ClarificationCheckpointSpec # Assuming ClarificationCheckpointSpec will be in orchestration.py or similar

class StageInput(BaseModel):
    """Represents input mapping for a stage."""
    # Define structure for how inputs are mapped
    pass # Placeholder

class StageOutput(BaseModel):
    """Represents expected output structure for a stage."""
    # Define structure for how outputs are captured
    pass # Placeholder

class StageDefinition(BaseModel):
    """Defines a single stage within a flow."""
    id: str = Field(..., description="Unique identifier for the stage within the flow.")
    agent: str = Field(..., description="Identifier of the agent to execute.")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Static inputs or context mappings for the agent.")
    # Outputs definition might be needed for validation or linking
    # outputs: Optional[Dict[str, str]] = Field(None, description="Mapping of agent output keys to context keys.")
    next_stage: Optional[str] = Field(None, description="ID of the next stage to execute upon successful completion.")
    condition: Optional[str] = Field(None, description="A condition expression (evaluated in context) for branching.")
    next_stage_true: Optional[str] = Field(None, description="ID of the next stage if condition is true.")
    next_stage_false: Optional[str] = Field(None, description="ID of the next stage if condition is false.")
    on_error: Optional[str] = Field(None, description="ID of the stage to execute if an error occurs.") # Simple error handling for now

class FlowDefinition(BaseModel):
    """Defines the structure and execution logic of a workflow."""
    id: str = Field(..., description="Unique identifier for the flow.")
    description: Optional[str] = Field(None, description="A brief description of the flow's purpose.")
    start_stage: str = Field(..., description="The ID of the initial stage to execute.")
    stages: Dict[str, StageDefinition] = Field(..., description="Dictionary mapping stage IDs to their definitions.")

# --- New Model for Paused Runs --- #
class PausedRunDetails(BaseModel):
    """Schema for data saved when a flow run is paused, typically on error."""
    run_id: str = Field(..., description="Unique identifier for this specific execution run.")
    flow_id: str = Field(..., description="Identifier of the FlowDefinition being executed.")
    paused_stage_id: str = Field(..., description="The ID of the stage where execution paused.") # Renamed from paused_at_stage_id
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the pause occurred.")
    status: FlowPauseStatus = Field(FlowPauseStatus.PAUSED_UNKNOWN, description="Structured status indicating why the flow is paused.")
    reason: Optional[str] = Field(None, description="A brief, human-readable reason for the pause.") # Added
    
    context_snapshot_ref: Optional[str] = Field(None, description="Reference to where the full context snapshot is stored, e.g., a file path or DB key.")
    error_details: Optional[AgentErrorDetails] = Field(None, description="Details of the error that caused the pause, if applicable.")
    # Changed clarification_request to use ClarificationCheckpointSpec
    clarification_checkpoint_details: Optional[ClarificationCheckpointSpec] = Field(None, description="Details of the clarification checkpoint that triggered a pause, if applicable.")

    # Added fields based on refactoring plan review
    next_stage_options: Optional[List[str]] = Field(None, description="Suggested next stage IDs the user can choose from if resuming manually.")
    current_master_plan_snapshot: Optional[MasterExecutionPlan] = Field(None, description="A snapshot of the MasterExecutionPlan at the time of pause.")
    audit_trail: Optional[List[Dict[str, Any]]] = Field(None, description="A log of review and resume attempts related to this pause instance. Each entry could be an 'AuditEntry' model later.")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        json_encoders = {
            AgentErrorDetails: lambda v: v.to_dict() if v else None,
        }
    )

class StageRunRecord(BaseModel):
    """Record of a single stage execution within a run."""
    # ... existing code ... 