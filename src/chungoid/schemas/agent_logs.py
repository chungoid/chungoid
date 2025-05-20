from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ARCALogEntry(BaseModel):
    """
    Represents a structured log entry for AutomatedRefinementCoordinatorAgent (ARCA) operations.
    """
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this log entry.")
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc), description="Timestamp of when the log event occurred.")
    
    arca_task_id: str = Field(..., description="The ARCA task_id this log entry pertains to.")
    project_id: str = Field(..., description="Identifier for the current project.")
    cycle_id: str = Field(..., description="The ID of the current refinement cycle.")
    
    event_type: Literal[
        "ARCA_INVOCATION_START",
        "ARCA_DECISION_MADE",
        "SUB_AGENT_INVOCATION_START",
        "SUB_AGENT_INVOCATION_END",
        "MAX_DEBUG_ATTEMPTS_REACHED",
        "STATE_UPDATE_ATTEMPT", # For logging attempts to update project_state.json
        "STATE_UPDATE_SUCCESS",
        "STATE_UPDATE_FAILURE",
        "ARCA_INTERNAL_ERROR"
    ] = Field(..., description="The type of event being logged.")
    
    event_details: Dict[str, Any] = Field(..., description="A dictionary containing specific details about the event.")
    
    related_artifact_doc_ids: List[str] = Field(default_factory=list, description="List of ChromaDB document IDs for artifacts relevant to this log event.")
    
    severity: Literal["INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Severity level of the log event.")

    class Config:
        # Example for event_details based on event_type (for documentation/validation if needed later)
        # This is more for conceptual clarity, direct validation of event_details based on event_type
        # within Pydantic v2 would require more complex logic (e.g. discriminated unions or validators)
        schema_extra = {
            "examples": [
                {
                    "log_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                    "timestamp": "2023-10-26T10:30:00Z",
                    "arca_task_id": "task_123",
                    "project_id": "proj_abc",
                    "cycle_id": "cycle_xyz",
                    "event_type": "ARCA_DECISION_MADE",
                    "event_details": {
                        "artifact_type": "LOPRD",
                        "artifact_doc_id": "doc_loprd_001",
                        "decision": "ACCEPT_ARTIFACT",
                        "reasoning": "Confidence scores above threshold.",
                        "confidence": 0.95
                    },
                    "related_artifact_doc_ids": ["doc_loprd_001"],
                    "severity": "INFO"
                },
                {
                    "log_id": "b2c3d4e5-f6a7-8901-2345-67890abcdef0",
                    "timestamp": "2023-10-26T10:35:00Z",
                    "arca_task_id": "task_123",
                    "project_id": "proj_abc",
                    "cycle_id": "cycle_xyz",
                    "event_type": "SUB_AGENT_INVOCATION_START",
                    "event_details": {
                        "invoked_agent_id": "CodeDebuggingAgent_v1",
                        "input_summary": {"faulty_code_path": "/path/to/file.py", "num_failed_tests": 3}
                    },
                    "related_artifact_doc_ids": ["doc_code_module_002"],
                    "severity": "INFO"
                },
                {
                    "log_id": "c3d4e5f6-a7b8-9012-3456-7890abcdef01",
                    "timestamp": "2023-10-26T10:40:00Z",
                    "arca_task_id": "task_123",
                    "project_id": "proj_abc",
                    "cycle_id": "cycle_xyz",
                    "event_type": "MAX_DEBUG_ATTEMPTS_REACHED",
                    "event_details": {
                        "module_path": "/path/to/file.py",
                        "attempts": 5
                    },
                    "related_artifact_doc_ids": ["doc_code_module_002"],
                    "severity": "WARNING"
                }
            ]
        } 