#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
import datetime

from .errors import AgentErrorDetails # Assuming AgentErrorDetails exists

# --- Enums for Status ---

class A2ATaskStatus(str, Enum):
    """Standard status codes for A2A task progression."""
    RECEIVED = "RECEIVED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED_SUCCESS = "COMPLETED_SUCCESS"
    COMPLETED_FAILURE = "COMPLETED_FAILURE"
    CANCELLED = "CANCELLED"

# --- Core Handshake Schemas ---

class A2ATaskInitiationInput(BaseModel):
    """Input schema for an agent initiating a task with another agent."""
    correlation_id: str = Field(..., description="Unique ID for tracing this entire interaction thread.")
    task_description: str = Field(..., description="Natural language description of the task to be performed.")
    task_input: Dict[str, Any] = Field(default_factory=dict, description="Specific input parameters required by the receiving agent/tool.")
    requesting_agent_id: Optional[str] = Field(None, description="Identifier of the agent making the request (optional but recommended).")
    priority: int = Field(default=0, description="Task priority (higher value means higher priority).")
    deadline: Optional[datetime.datetime] = Field(None, description="Suggested deadline for task completion.")

class A2ATaskInitiationOutput(BaseModel):
    """Output schema sent by the receiving agent upon receiving a task initiation request."""
    correlation_id: str = Field(..., description="The correlation ID received in the initiation request.")
    status: A2ATaskStatus = Field(..., description="Initial status (e.g., RECEIVED, maybe ACCEPTED/REJECTED if decidable immediately).")
    receiving_agent_id: Optional[str] = Field(None, description="Identifier of the agent that received the task.")
    message: Optional[str] = Field(None, description="Optional message, e.g., reason for immediate rejection.")

class A2ATaskUpdateInput(BaseModel):
    """Input schema for an agent providing a progress update on an assigned task."""
    correlation_id: str = Field(..., description="Unique ID linking this update to the original task.")
    status: A2ATaskStatus = Field(..., description="Current status of the task (e.g., IN_PROGRESS).")
    progress_percentage: Optional[float] = Field(None, description="Estimated completion percentage (0.0 to 100.0).")
    current_step: Optional[str] = Field(None, description="Description of the current step being worked on.")
    next_step: Optional[str] = Field(None, description="Description of the anticipated next step.")
    estimated_completion_time: Optional[datetime.datetime] = Field(None, description="Updated estimate for completion.")
    update_message: Optional[str] = Field(None, description="Optional free-text update message.")

class A2ATaskCompletionInput(BaseModel):
    """Input schema for an agent signaling the completion (success or failure) of an assigned task."""
    correlation_id: str = Field(..., description="Unique ID linking this completion to the original task.")
    status: A2ATaskStatus = Field(..., description="Final status (COMPLETED_SUCCESS or COMPLETED_FAILURE).")
    task_output: Optional[Dict[str, Any]] = Field(None, description="The results/output of the task, if successful.")
    error_details: Optional[AgentErrorDetails] = Field(None, description="Details of the error, if the task failed.")
    completion_message: Optional[str] = Field(None, description="Optional final message (e.g., summary, reason for failure).")

# --- Skeletons/Templates for Additional A2A Tools ---

class A2ATaskAcceptanceInput(BaseModel):
    """(Skeleton) Input for explicitly accepting a previously received task."""
    correlation_id: str = Field(..., description="ID of the task being accepted.")
    accepting_agent_id: Optional[str] = Field(None, description="ID of the agent accepting.")
    estimated_start_time: Optional[datetime.datetime] = Field(None)
    message: Optional[str] = Field(None)

class A2ATaskRejectionInput(BaseModel):
    """(Skeleton) Input for explicitly rejecting a previously received task."""
    correlation_id: str = Field(..., description="ID of the task being rejected.")
    rejecting_agent_id: Optional[str] = Field(None, description="ID of the agent rejecting.")
    reason: str = Field(..., description="Reason for rejecting the task.")
    alternative_suggestion: Optional[str] = Field(None) # e.g., suggest a different agent

class A2AQueryTaskStatusInput(BaseModel):
    """(Skeleton) Input for requesting the status of a previously initiated task."""
    correlation_id: str = Field(..., description="ID of the task to query.")
    requesting_agent_id: Optional[str] = Field(None)

class A2AQueryTaskStatusOutput(BaseModel):
    """(Skeleton) Output containing the status of a queried task."""
    correlation_id: str
    status: A2ATaskStatus
    last_update_time: Optional[datetime.datetime]
    details: Optional[Dict[str, Any]] # Could contain progress_percentage, current_step etc.

class A2ACancelTaskInput(BaseModel):
    """(Skeleton) Input for requesting the cancellation of a previously initiated task."""
    correlation_id: str = Field(..., description="ID of the task to cancel.")
    requesting_agent_id: Optional[str] = Field(None)
    reason: Optional[str] = Field(None, description="Reason for cancellation.")

class A2ACancelTaskOutput(BaseModel):
    """(Skeleton) Output confirming the cancellation request."""
    correlation_id: str
    cancellation_acknowledged: bool
    final_status: Optional[A2ATaskStatus] # e.g., was it already completed?
    message: Optional[str]

# --- Example MCP Tool Input (for documentation/AgentCard) ---
# This shows how you might structure the JSON schema for AgentCard.mcp_tool_input_schemas

INITIATE_TASK_SCHEMA = {
    "initiate_a2a_task": {
        "type": "object",
        "properties": {
            "correlation_id": {
                "type": "string",
                "description": "Unique ID for tracing this entire interaction thread."
            },
            "task_description": {
                "type": "string",
                "description": "Natural language description of the task to be performed."
            },
            "task_input": {
                "type": "object",
                "description": "Specific input parameters required by the receiving agent/tool.",
                "additionalProperties": True
            },
            "requesting_agent_id": {
                "type": ["string", "null"],
                "description": "Identifier of the agent making the request (optional but recommended)."
            },
            "priority": {
                "type": "integer",
                "description": "Task priority (higher value means higher priority).",
                "default": 0
            },
            "deadline": {
                "type": ["string", "null"],
                "format": "date-time",
                "description": "Suggested deadline for task completion."
            }
        },
        "required": ["correlation_id", "task_description"]
    }
} 