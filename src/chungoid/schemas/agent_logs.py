from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Literal, Optional
from enum import Enum

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


class QALogEntry(BaseModel):
    """Schema for a Quality Assurance log entry."""
    log_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for this QA log entry.")
    timestamp_utc: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc), description="Timestamp of when the QA check was performed.")
    project_id: str = Field(..., description="Identifier of the project this QA log pertains to.")
    cycle_id: str = Field(..., description="Identifier of the autonomous cycle during which this QA check occurred.")
    
    qa_agent_id: str = Field(..., description="ID of the agent that performed the QA check.")
    target_artifact_id: str = Field(..., description="Document ID of the artifact that was assessed.")
    target_artifact_type: str = Field(..., description="The type of the artifact that was assessed (e.g., ProjectBlueprint_MD, LOPRD_JSON).")
    
    qa_check_type: str = Field(..., description="Specific type of QA check performed (e.g., 'BlueprintReview', 'CodeStandardAdherence', 'DocCoverage').")
    status: Literal["PASSED", "FAILED", "WARNINGS_FOUND", "NOT_APPLICABLE", "ERROR_DURING_CHECK"] = Field(..., description="Outcome of the QA check.")
    
    summary: str = Field(..., description="A concise summary of the QA findings.")
    details: Optional[Dict[str, Any]] = Field(None, description="Structured details of the findings (e.g., list of issues, specific metrics).")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in this QA assessment/finding, if applicable.")
    
    # Optional: Link to a more detailed report if findings are extensive
    detailed_report_doc_id: Optional[str] = Field(None, description="Optional document ID of a more detailed QA report artifact, if one was generated.")

    class Config:
        validate_assignment = True
        # Ensure enums are handled correctly if any string fields become enums later 


class LLMCallDetails(BaseModel):
    """Details about a specific LLM call made by an agent."""
    model_name: str = Field(..., description="Name of the LLM used.")
    prompt_template_id: Optional[str] = Field(None, description="Identifier for the prompt template used, if applicable.")
    # prompt_content_summary: Optional[str] = Field(None, description="A brief summary or hash of the rendered prompt sent to the LLM.") # Potentially too verbose
    system_prompt_summary: Optional[str] = Field(None, description="Summary or hash of the system prompt used.")
    user_prompt_summary: Optional[str] = Field(None, description="Summary or hash of the user prompt used.")
    temperature: Optional[float] = Field(None, description="Temperature setting used for the call.")
    max_tokens: Optional[int] = Field(None, description="Max tokens setting used for the call.")
    tokens_used_prompt: Optional[int] = Field(None, description="Tokens used by the prompt.")
    tokens_used_completion: Optional[int] = Field(None, description="Tokens used by the completion.")
    tokens_total: Optional[int] = Field(None, description="Total tokens used for the call.")
    # response_summary: Optional[str] = Field(None, description="A brief summary or hash of the LLM's response.") # Potentially too verbose
    stop_reason: Optional[str] = Field(None, description="The reason the LLM stopped generating tokens.")


class ToolCallDetails(BaseModel):
    """Details about a specific external tool call made by an agent (via MCP or other means)."""
    tool_name: str = Field(..., description="Name of the tool called.")
    tool_input: Dict[str, Any] = Field(..., description="Input parameters provided to the tool.")
    tool_output_summary: Optional[str] = Field(None, description="A brief summary of the tool's output or key results.")
    status: Literal["SUCCESS", "FAILURE", "PENDING"] = Field(..., description="Status of the tool call.")
    error_message: Optional[str] = Field(None, description="Error message if the tool call failed.")
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    end_time: Optional[datetime.datetime] = None
    duration_seconds: Optional[float] = None


class GenericAgentReflection(BaseModel):
    """
    A generic reflection log entry for any agent, detailing its thought process, 
    key decisions, LLM interactions, and tool usage for a specific task.
    To be stored in AGENT_REFLECTIONS_AND_LOGS_COLLECTION.
    """
    reflection_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this reflection entry.")
    timestamp_utc: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc), description="Timestamp of when the reflection was recorded.")
    
    project_id: str = Field(..., description="Identifier of the project this reflection pertains to.")
    cycle_id: Optional[str] = Field(None, description="Identifier of the autonomous cycle during which this activity occurred, if applicable.")
    
    agent_id: str = Field(..., description="ID of the agent that generated this reflection.")
    agent_version: Optional[str] = Field(None, description="Version of the agent.")
    source_task_id: str = Field(..., description="The ID of the task the agent was performing when this reflection was generated.")
    
    summary_of_activity: str = Field(..., description="A brief summary of the agent's activity or decision covered by this reflection.")
    
    # Inputs
    input_artifact_ids_used: List[str] = Field(default_factory=list, description="List of main input artifact document IDs used for this activity.")
    # input_parameters_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of key input parameters received by the agent for this task.") # Potentially too verbose
    
    # Outputs
    output_artifact_ids_generated: List[str] = Field(default_factory=list, description="List of primary output artifact document IDs generated from this activity.")
    
    # Decision Making & Rationale
    key_decision_points: Optional[List[str]] = Field(default_factory=list, description="Description of key decision points encountered.")
    decision_rationale: Optional[str] = Field(None, description="Detailed rationale behind the primary decision or outcome of this activity.")
    
    # Confidence (if the agent's primary output doesn't have a dedicated confidence score, or for overall process confidence)
    process_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Agent's confidence in the process/outcome described in this reflection.")
    
    # LLM & Tool Usage
    llm_calls: List[LLMCallDetails] = Field(default_factory=list, description="Details of LLM calls made during this activity.")
    tool_calls: List[ToolCallDetails] = Field(default_factory=list, description="Details of external tool calls made during this activity.")
    
    # Adherence & Challenges
    contextual_adherence_explanation: Optional[str] = Field(None, description="Explanation of how the agent's actions/outputs adhered to provided context and instructions.")
    challenges_encountered: Optional[List[str]] = Field(default_factory=list, description="Any challenges or difficulties encountered.")
    
    # Miscellaneous
    custom_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Any custom metrics the agent wishes to log for this activity.")
    
    class Config:
        validate_assignment = True 


# --- New Schemas for Quality Assurance Logs ---

class QAEventType(str, Enum):
    ARCA_ARTIFACT_ASSESSMENT = "ARCA_ARTIFACT_ASSESSMENT"
    HUMAN_REVIEW_QA_COMPLETED = "HUMAN_REVIEW_QA_COMPLETED"
    AUTOMATED_QUALITY_CHECK = "AUTOMATED_QUALITY_CHECK"
    # Add other specific QA event types as needed

class OverallQualityStatus(str, Enum):
    APPROVED_PASSED = "APPROVED_PASSED" # Meets all quality criteria
    APPROVED_WITH_RESERVATIONS = "APPROVED_WITH_RESERVATIONS" # Acceptable, but with minor issues noted
    REJECTED_NEEDS_REFINEMENT = "REJECTED_NEEDS_REFINEMENT" # Does not meet quality criteria, requires rework
    FLAGGED_FOR_MANUAL_REVIEW = "FLAGGED_FOR_MANUAL_REVIEW" # Automated QA cannot make a determination
    ERROR_IN_QA_PROCESS = "ERROR_IN_QA_PROCESS" # The QA process itself failed

class QualityAssuranceLogEntry(BaseModel):
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this QA log entry.")
    project_id: str = Field(..., description="Identifier for the project context.")
    cycle_id: Optional[str] = Field(None, description="Identifier for the current development or refinement cycle.")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    artifact_doc_id_assessed: Optional[str] = Field(None, description="ChromaDB document ID of the artifact being assessed. Optional if QA log is about a process rather than a specific artifact.")
    artifact_type_assessed: Optional[str] = Field(None, description="Type of the artifact being assessed (e.g., LOPRD, Blueprint, CodeModule).")
    
    qa_event_type: QAEventType = Field(..., description="The type of QA event being logged.")
    assessing_entity_id: str = Field(..., description="Identifier of the agent (e.g., ARCA_v1) or human reviewer performing the QA.")
    
    summary_of_assessment: str = Field(..., description="A concise summary of the QA findings or assessment.")
    overall_quality_status: OverallQualityStatus = Field(..., description="The overall quality status determined by this QA event.")
    
    key_metrics_or_findings: Optional[Dict[str, Any]] = Field(None, description="Structured data for key metrics or specific findings (e.g., {'test_pass_rate': 0.95, 'critical_defects_found': 0}).")
    detailed_report_link_or_id: Optional[str] = Field(None, description="Link or ID to a more detailed report if applicable (e.g., a full test summary report ID in ChromaDB).")
    
    confidence_in_assessment: Optional[float] = Field(None, description="Confidence in this QA assessment itself (0.0 to 1.0), if applicable.")
    action_taken_or_recommended: Optional[str] = Field(None, description="Brief description of any action taken as a result of this QA event (e.g., 'Triggered refinement cycle') or recommended next steps.")

    class Config:
        use_enum_values = True # Ensures enum values are used in serialization 