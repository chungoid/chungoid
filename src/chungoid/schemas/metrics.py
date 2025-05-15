import uuid
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class MetricEventType(str, Enum):
    """Defines the types of metric events that can be recorded."""
    FLOW_START = "FLOW_START"
    FLOW_END = "FLOW_END"
    STAGE_START = "STAGE_START"
    STAGE_END = "STAGE_END" # Covers success, failure, skip, etc. status is in data.
    AGENT_INVOCATION_START = "AGENT_INVOCATION_START"
    AGENT_INVOCATION_END = "AGENT_INVOCATION_END"
    AGENT_REPORTED_METRIC = "AGENT_REPORTED_METRIC" # For custom metrics like token usage, cost
    ORCHESTRATOR_INFO = "ORCHESTRATOR_INFO" # General info/logs from orchestrator not tied to specific stage cycle
    FLOW_RESUME = "FLOW_RESUME" # ADDED for when a flow is resumed
    # Add more specific event types as needed


class MetricEvent(BaseModel):
    """Schema for a single metric event recorded by the system."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this event log entry.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp when the event occurred or was recorded.")
    event_type: MetricEventType = Field(..., description="The type of metric event.")
    
    # Contextual Identifiers
    flow_id: Optional[str] = Field(None, description="Identifier of the flow definition being executed.")
    run_id: Optional[str] = Field(None, description="Unique identifier for the specific execution run of the flow.")
    stage_id: Optional[str] = Field(None, description="Identifier of the stage within the flow, if applicable.")
    master_stage_id: Optional[str] = Field(None, description="Identifier of the master stage, if the event is related to a master flow execution.")
    agent_id: Optional[str] = Field(None, description="Identifier of the agent involved, if applicable.")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing related operations.")
    
    # Payload
    data: Dict[str, Any] = Field(default_factory=dict, description="Specific metrics data associated with the event. E.g., duration_seconds, status, token_count, cost, error_message.")

    class Config:
        use_enum_values = True # Ensures enum values are used in serialization 