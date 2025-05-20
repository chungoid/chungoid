from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
import uuid

# Schemas based on: chungoid-core/docs/design_documents/autonomous_engine/code_debugging_agent_design.md

class FailedTestReport(BaseModel):
    test_name: str = Field(..., description="Name of the failed test.")
    error_message: str = Field(..., description="Error message from the test failure.")
    stack_trace: str = Field(..., description="Stack trace associated with the failure.")
    expected_behavior_summary: Optional[str] = Field(None, description="Optional summary of what the test expected.")

class PreviousDebuggingAttempt(BaseModel):
    attempted_fix_summary: str = Field(..., description="Summary of the fix that was attempted.")
    outcome: str = Field(..., description="e.g., 'tests_still_failed', 'new_errors_introduced'")

class DebuggingTaskInput(BaseModel):
    project_id: str = Field(..., description="Identifier for the project.")
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this debugging task.")
    
    faulty_code_path: str = Field(
        ..., 
        description="Path to the code file needing debugging (retrieved by ProjectChromaManagerAgent)."
    )
    faulty_code_content_doc_id: str = Field(
        ..., 
        description="Document ID of the faulty code content in ChromaDB."
    )
    faulty_code_snippet: Optional[str] = Field(
        None, 
        description="(Optional) The specific code snippet if already localized by ARCA or a previous process."
    )
    failed_test_reports: List[FailedTestReport] = Field(
        ..., 
        description="List of structured test failure objects (from ProjectChromaManagerAgent, originally from SystemTestRunnerAgent)."
    )
    relevant_loprd_requirements_ids: List[str] = Field(
        ..., 
        description="List of LOPRD requirement IDs (e.g., FRs, ACs) that the faulty code was intended to satisfy (from ProjectChromaManagerAgent)."
    )
    relevant_blueprint_section_ids: Optional[List[str]] = Field(
        None,
        description="List of Blueprint section IDs relevant to the code's design (from ProjectChromaManagerAgent)."
    )
    previous_debugging_attempts: Optional[List[PreviousDebuggingAttempt]] = Field(
        None,
        description="(Optional) List of previous fixes attempted for this issue in the current cycle, to avoid loops and provide history to the LLM."
    )
    max_iterations_for_this_call: Optional[int] = Field(
        None,
        description="(Optional) A limit set by ARCA for this specific debugging invocation's internal reasoning if applicable."
    )
    # Added for consistency with other agents and potential direct invocation context
    additional_context: Optional[Dict[str, Any]] = Field(None, description="Generic dictionary for any additional context ARCA might want to pass.")


class DebuggingTaskOutput(BaseModel):
    task_id: str = Field(..., description="Identifier of the debugging task this output corresponds to.")
    proposed_solution_type: Literal["CODE_PATCH", "MODIFIED_SNIPPET", "NO_FIX_IDENTIFIED", "NEEDS_MORE_CONTEXT"] = Field(
        ..., 
        description="Type of solution being proposed."
    )
    proposed_code_changes: Optional[str] = Field(
        None, 
        description="The actual patch (e.g., diff format using `diff -u`) or the full modified code snippet. Null if no fix is identified."
    )
    explanation_of_fix: Optional[str] = Field(
        None, 
        description="LLM-generated explanation of the diagnosed bug and the proposed fix. Null if no fix is identified."
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Likelihood the proposed fix resolves the issue. If NO_FIX_IDENTIFIED, this might reflect confidence in that assessment."
    )
    areas_of_uncertainty: Optional[List[str]] = Field(
        None, 
        description="(Optional) Any parts of the code, problem, or context the agent is unsure about."
    )
    suggestions_for_ARCA: Optional[str] = Field(
        None, 
        description="(Optional) E.g., 'Consider broader refactoring if this pattern repeats,' or 'Unable to fix without X specific context from Y module.'"
    )
    status: Literal["SUCCESS_FIX_PROPOSED", "FAILURE_NO_FIX_IDENTIFIED", "FAILURE_NEEDS_CLARIFICATION", "ERROR_INTERNAL"] = Field(
        ..., 
        description="Overall status of the debugging attempt."
    )
    # Added for consistency
    original_input_summary: Optional[Dict[str, Any]] = Field(None, description="A brief summary or key fields from the input for traceability.") 