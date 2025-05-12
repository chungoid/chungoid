#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

from .errors import AgentErrorDetails # Import for use in PausedRunDetails

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
    """Holds information about a paused workflow run, typically due to an error requiring intervention."""
    run_id: str = Field(..., description="The unique ID of the paused run.")
    paused_at_stage_id: str = Field(..., description="The ID of the stage where the execution paused.")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp when the run was paused.")
    context_snapshot: Dict[str, Any] = Field(..., description="Snapshot of the execution context at the time of pausing.")
    error_details: Optional[AgentErrorDetails] = Field(None, description="Details of the agent error that caused the pause, if applicable.")
    reason: Optional[str] = Field(None, description="Optional brief reason for the pause (e.g., 'Manual Pause', 'Agent Error').") 