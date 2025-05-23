#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
import datetime
import uuid

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

class A2ATaskRequest(BaseModel):
    """Schema for an agent to request a task from another agent or agent category."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this specific task request.")
    requesting_agent_id: str = Field(..., description="ID of the agent making the request.")
    target_agent_id: Optional[str] = Field(None, description="Specific ID of the target agent. Use if known.")
    target_agent_category: Optional[str] = Field(None, description="Category of agent to perform the task. Used if target_agent_id is not set.")
    # ^^^ TODO: Add validation: one of target_agent_id or target_agent_category must be set.
    task_type: str = Field(..., description="A descriptor for the type of task, e.g., 'code_review', 'data_analysis', 'file_patch'.")
    task_payload: Dict[str, Any] = Field(..., description="The actual input/data required for the target agent to perform the task.")
    correlation_id: Optional[str] = Field(None, description="ID to correlate this task with a broader workflow or parent task.")
    priority: int = Field(0, description="Priority of the task (higher values mean higher priority). Default is 0.")
    # Optional: deadline, dependencies, etc.

class A2ATaskStatusReport(BaseModel):
    """Schema for an agent to report the status of an A2A task it is handling or has handled."""
    task_id: str = Field(..., description="The unique ID of the task this status report pertains to.")
    reporting_agent_id: str = Field(..., description="ID of the agent providing this status update (usually the one performing the task).")
    status: str = Field(..., description="Current status of the task, e.g., 'ACCEPTED', 'REJECTED', 'IN_PROGRESS', 'COMPLETED_SUCCESS', 'COMPLETED_FAILURE', 'PENDING_CLARIFICATION'.")
    # TODO: Standardize status values into an Enum (e.g., A2ATaskLifecycleStatus)
    progress_percent: Optional[float] = Field(None, description="Estimated completion percentage (0.0 to 100.0).")
    status_message: Optional[str] = Field(None, description="Human-readable message accompanying the status.")
    result_payload: Optional[Dict[str, Any]] = Field(None, description="The output/results of the task, if status is COMPLETED_SUCCESS.")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Details of any error, if status is COMPLETED_FAILURE.")
    clarification_needed_payload: Optional[Dict[str, Any]] = Field(None, description="Details if clarification is needed from the requesting agent.")
    correlation_id: Optional[str] = Field(None, description="ID to correlate this task with a broader workflow.")

# --- Schemas for MCP Tools facilitating A2A Tasking ---

class PostA2ATaskMCPInput(BaseModel):
    """Input schema for an MCP tool that posts/submits an A2A task request."""
    request: A2ATaskRequest

class PostA2ATaskMCPOutput(BaseModel):
    """Output schema for an MCP tool after posting/submitting an A2A task request."""
    task_id: str
    submission_status: str # e.g., 'TASK_POSTED_OK', 'ROUTING_ERROR', 'INVALID_REQUEST'
    message: Optional[str] = None

class GetA2ATaskStatusMCPInput(BaseModel):
    """Input schema for an MCP tool to query the status of an A2A task."""
    task_id: str
    requesting_agent_id: Optional[str] = Field(None, description="Optional: ID of the agent requesting the status, for auth/logging.")

class GetA2ATaskStatusMCPOutput(BaseModel):
    """Output schema for an MCP tool that returns the status of an A2A task."""
    task_id: str
    status_report: Optional[A2ATaskStatusReport] = None
    query_status: str # e.g., 'FOUND', 'NOT_FOUND', 'ACCESS_DENIED'
    error_message: Optional[str] = None # If task_id not found or other query error

# Potential future MCP tool schemas:
# - ClaimA2ATaskMCPInput / Output (for an agent to claim a task from a queue/broker)
# - UpdateA2ATaskProgressMCPInput / Output (for an agent to provide intermediate progress)
# - ProvideClarificationForA2ATaskMCPInput / Output (for requester to respond to clarification_needed) 