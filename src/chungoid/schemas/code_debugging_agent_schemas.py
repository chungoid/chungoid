from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class FailedTestReport(BaseModel):
    test_name: str
    error_message: str
    stack_trace: str
    expected_behavior_summary: Optional[str] = Field(None, description="Summary of what the test expected.")

class PreviousDebuggingAttempt(BaseModel):
    attempted_fix_summary: str
    outcome: str = Field(description="e.g., 'tests_still_failed', 'new_errors_introduced'")

class DebuggingTaskInput(BaseModel):
    faulty_code_path: str = Field(..., description="Path to the code file needing debugging.")
    faulty_code_content: str = Field(..., description="The full string content of the faulty code file.")
    faulty_code_snippet: Optional[str] = Field(
        None, description="(Optional) The specific code snippet if already localized by ARCA or a previous process."
    )
    failed_test_reports: List[FailedTestReport] = Field(
        ..., description="List of structured test failure objects."
    )
    relevant_loprd_requirements_ids: List[str] = Field(
        default_factory=list, description="List of LOPRD requirement IDs (e.g., FRs, ACs) that the faulty code was intended to satisfy."
    )
    relevant_loprd_requirements_details_json: str = Field(
        default="[]", description="JSON string containing details of the LOPRD requirements (e.g., [{id: 'FR1', text: '...'}, ...])."
    )
    relevant_blueprint_section_ids: List[str] = Field(
        default_factory=list, description="List of Blueprint section IDs relevant to the code's design."
    )
    relevant_blueprint_sections_details_text: str = Field(
        default="", description="Concatenated text of relevant blueprint sections."
    )
    previous_debugging_attempts: List[PreviousDebuggingAttempt] = Field(
        default_factory=list, 
        description="(Optional) List of previous fixes attempted for this issue in the current cycle, to avoid loops and provide history to the LLM."
    )
    max_iterations_for_this_call: Optional[int] = Field(None, description="(Optional) A limit set by ARCA for this specific debugging invocation's internal reasoning if applicable.")

class DebuggingTaskOutput(BaseModel):
    proposed_solution_type: Literal["CODE_PATCH", "MODIFIED_SNIPPET", "NO_FIX_IDENTIFIED", "NEEDS_MORE_CONTEXT"] = Field(description="Type of solution being proposed.")
    proposed_code_changes: Optional[str] = Field(None, description="The actual patch (e.g., diff format using `diff -u`) or the full modified code snippet. Null if no fix is identified.")
    explanation_of_fix: Optional[str] = Field(None, description="LLM-generated explanation of the diagnosed bug and the proposed fix. Null if no fix is identified.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Likelihood the proposed fix resolves the issue. If NO_FIX_IDENTIFIED, this might reflect confidence in that assessment.")
    areas_of_uncertainty: Optional[List[str]] = Field(None, description="(Optional) Any parts of the code, problem, or context the agent is unsure about.")
    suggestions_for_ARCA: Optional[str] = Field(None, description="(Optional) E.g., 'Consider broader refactoring if this pattern repeats,' or 'Unable to fix without X specific context from Y module.'")
    status: Literal["SUCCESS_FIX_PROPOSED", "FAILURE_NO_FIX_IDENTIFIED", "FAILURE_NEEDS_CLARIFICATION", "ERROR_INTERNAL"] = Field(description="Overall status of the debugging attempt.") 