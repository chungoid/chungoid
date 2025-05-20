from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator

# Forward declaration for types used in ARCA integration if needed later
# class ARCAReviewInput(BaseModel): ...
# class ARCAOutput(BaseModel): ...


class ProjectOverallStatus(str, Enum):
    """Overall status of the project."""
    NOT_STARTED = "NOT_STARTED"
    INITIALIZING = "INITIALIZING"
    ANALYZING_GOAL = "ANALYZING_GOAL"
    LOPRD_GENERATION = "LOPRD_GENERATION" # First major artifact generation phase
    ARCHITECTING = "ARCHITECTING"
    PLANNING = "PLANNING"
    CODING_IMPLEMENTATION = "CODING_IMPLEMENTATION"
    TESTING_VALIDATION = "TESTING_VALIDATION"
    DOCUMENTATION_GENERATION = "DOCUMENTATION_GENERATION"
    CYCLE_COMPLETED_PENDING_REVIEW = "CYCLE_COMPLETED_PENDING_REVIEW"
    HUMAN_REVIEW_IN_PROGRESS = "HUMAN_REVIEW_IN_PROGRESS"
    AWAITING_NEXT_CYCLE_START = "AWAITING_NEXT_CYCLE_START" # After human review confirms to proceed
    REFINEMENT_CYCLE_IN_PROGRESS = "REFINEMENT_CYCLE_IN_PROGRESS" # General status for any subsequent cycle
    PROJECT_COMPLETED_SUCCESSFULLY = "PROJECT_COMPLETED_SUCCESSFULLY"
    PROJECT_PAUSED_BY_USER = "PROJECT_PAUSED_BY_USER"
    PROJECT_FAILED = "PROJECT_FAILED"
    PROJECT_ARCHIVED = "PROJECT_ARCHIVED"


class CycleStatus(str, Enum):
    """Status of an individual autonomous cycle."""
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED_SUCCESS = "COMPLETED_SUCCESS" # All objectives met, high confidence
    COMPLETED_WITH_ISSUES_FOR_REVIEW = "COMPLETED_WITH_ISSUES_FOR_REVIEW" # Objectives met, but ARCA flagged items
    FAILED_INTERNAL_ERROR = "FAILED_INTERNAL_ERROR" # Agent or system error
    FAILED_UNRESOLVABLE_ISSUES = "FAILED_UNRESOLVABLE_ISSUES" # ARCA could not resolve issues within its capability for this cycle
    TERMINATED_BY_USER = "TERMINATED_BY_USER"


class ArtifactLink(BaseModel):
    """Link to a specific version of an artifact stored in ChromaDB."""
    artifact_type: str = Field(..., description="Type of the artifact (e.g., 'LOPRD', 'Blueprint', 'CodeModule:main.py').")
    artifact_doc_id: str = Field(..., description="ChromaDB document ID.")
    version_identifier: Optional[str] = Field(None, description="Version number or cycle ID associated with this artifact instance.")
    description: Optional[str] = Field(None, description="Brief description or notes about this artifact version.")
    created_at_utc: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata specific to this artifact link.")


class KeyDecision(BaseModel):
    """Record of a key decision made during a cycle."""
    decision_id: str = Field(default_factory=lambda: f"kd_{uuid.uuid4().hex[:8]}", description="Unique ID for the decision.")
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
    deciding_agent_id: Optional[str] = Field(None, description="Agent ID that made the decision (e.g., ARCA).")
    decision_summary: str = Field(..., description="Brief summary of the decision made.")
    rationale_doc_id: Optional[str] = Field(None, description="Link to a ChromaDB document with detailed rationale, if any.")
    inputs_considered_doc_ids: Optional[List[str]] = Field(None, description="Links to key input artifacts considered for this decision.")
    output_artifacts_doc_ids: Optional[List[str]] = Field(None, description="Links to key output artifacts resulting from this decision.")


class CycleInfo(BaseModel):
    """Detailed information about a single autonomous or refinement cycle."""
    cycle_id: str = Field(..., description="Unique identifier for this cycle (e.g., 'cycle_001_loprd', 'refinement_002_codefix').")
    cycle_number: int = Field(..., ge=0, description="Sequential number of the cycle (0 for initial, 1+ for refinements).")
    cycle_objective: str = Field(..., description="Primary goal for this specific cycle.")
    status: CycleStatus = Field(..., description="Current status of this cycle.")
    start_time_utc: datetime = Field(default_factory=datetime.utcnow)
    end_time_utc: Optional[datetime] = Field(None, description="Timestamp when the cycle concluded or was last active.")
    
    # Artifacts generated or significantly modified *during* this cycle
    cycle_produced_artifacts: List[ArtifactLink] = Field(default_factory=list, description="Key artifacts generated or updated in this cycle.")
    
    # Summary from ARCA at the end of this cycle (if applicable)
    arca_cycle_summary_doc_id: Optional[str] = Field(None, description="ChromaDB ID of ARCA's summary report for this cycle's autonomous operations.")
    arca_decision_at_cycle_end: Optional[str] = Field(None, description="ARCA's decision, e.g., 'PROCEED_TO_BLUEPRINT', 'FLAGGED_FOR_HUMAN_REVIEW'.")
    issues_flagged_by_arca_in_cycle: Optional[List[str]] = Field(None, description="List of specific issues ARCA could not resolve in this cycle.")
    key_decisions_in_cycle: List[KeyDecision] = Field(default_factory=list, description="Key decisions made by agents (especially ARCA) during this cycle.")


class HumanReviewRecord(BaseModel):
    """Record of a human review and gating decision."""
    review_id: str = Field(default_factory=lambda: f"hr_{uuid.uuid4().hex[:8]}", description="Unique ID for this human review instance.")
    review_time_utc: datetime = Field(default_factory=datetime.utcnow)
    reviewer_id: Optional[str] = Field(None, description="Identifier for the human reviewer (e.g., username, email).")
    
    # Documents reviewed by the human
    reviewed_cycle_id: str = Field(..., description="The cycle_id whose outputs were reviewed.")
    reviewed_artifact_doc_ids: List[str] = Field(default_factory=list, description="List of primary ChromaDB artifact IDs reviewed.")
    
    # Human's feedback and decision
    feedback_summary: str = Field(..., description="Summary of the human's feedback and observations.")
    detailed_feedback_doc_id: Optional[str] = Field(None, description="ChromaDB ID for a more detailed feedback document, if provided.")
    decision_for_next_step: Literal[
        "PROCEED_TO_NEXT_AUTONOMOUS_PHASE", # e.g., from LOPRD to Blueprint gen
        "INITIATE_REFINEMENT_CYCLE",         # Trigger another autonomous cycle with this feedback
        "MODIFY_PROJECT_GOAL",               # Requires restarting analysis from goal
        "PAUSE_PROJECT",
        "ARCHIVE_PROJECT_SUCCESS",
        "ARCHIVE_PROJECT_FAILURE"
    ] = Field(..., description="The human's decision on how to proceed.")
    next_cycle_objective_override: Optional[str] = Field(None, description="If initiating refinement, human can specify/override the next cycle's objective.")
    additional_notes: Optional[str] = Field(None, description="Any other notes from the human reviewer.")


class ProjectStateV2(BaseModel):
    """
    Schema for project_status.json, version 2.
    Tracks the overall state, cycle history, key artifacts, and human review gates
    for an autonomous project.
    """
    project_id: str = Field(..., description="Unique identifier for the project.")
    project_name: Optional[str] = Field(None, description="User-defined name for the project.")
    initial_user_goal_summary: str = Field(..., description="A concise summary of the initial user goal for the project.")
    initial_user_goal_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the full initial user goal document.")
    refined_user_goal_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the latest refined user goal document.")
    
    schema_version: Literal["2.0.0"] = Field("2.0.0", description="Version of this project status schema.")
    created_at_utc: datetime = Field(default_factory=datetime.utcnow)
    last_updated_utc: datetime = Field(default_factory=datetime.utcnow)

    overall_status: ProjectOverallStatus = Field(ProjectOverallStatus.NOT_STARTED, description="The current high-level status of the project.")
    
    # --- Current State & Cycle ---
    current_cycle_id: Optional[str] = Field(None, description="ID of the cycle currently in progress or most recently completed.")
    current_cycle_number: int = Field(0, ge=0, description="The number of the current/latest cycle.")
    
    # --- Latest Accepted Artifacts (Single source of truth for "current best") ---
    # These are updated by ARCA or human review decisions.
    latest_accepted_loprd_doc_id: Optional[str] = Field(None)
    latest_accepted_blueprint_doc_id: Optional[str] = Field(None)
    latest_accepted_master_plan_doc_id: Optional[str] = Field(None)
    # For code, this might be a manifest or a link to a "live_codebase_collection" snapshot ID
    latest_accepted_code_snapshot_id: Optional[str] = Field(None, description="ID representing the latest accepted version of the codebase (e.g., a commit hash, a collection snapshot ID).")
    latest_test_report_summary_doc_id: Optional[str] = Field(None)
    latest_project_readme_doc_id: Optional[str] = Field(None)
    latest_full_documentation_bundle_id: Optional[str] = Field(None, description="ID for a comprehensive documentation bundle, if generated.")

    # --- ARCA's view of the "Best Autonomously Achievable State" (typically before human review) ---
    # This might be populated at the end of an autonomous cycle by ARCA.
    arca_best_state_summary_doc_id: Optional[str] = Field(None, description="ChromaDB ID of ARCA's summary of the best state achieved autonomously in the last cycle.")
    arca_overall_confidence_in_best_state: Optional[float] = Field(None, ge=0.0, le=1.0, description="ARCA's overall confidence in this best state.")
    arca_issues_pending_human_review: List[Dict[str, Any]] = Field(default_factory=list, description="List of specific issues or items ARCA flagged for mandatory human review. Each item could be a structured dict.")

    # --- Human Review & Gating ---
    last_human_review: Optional[HumanReviewRecord] = Field(None, description="Details of the most recent human review.")
    next_action_determined_by: Literal["AUTONOMOUS_AGENT", "HUMAN_REVIEWER", "SYSTEM_INITIALIZATION"] = Field("SYSTEM_INITIALIZATION")
    
    # --- History ---
    # Stores information about *completed* or *failed* cycles. The current cycle's details are built up then moved here.
    historical_cycles: List[CycleInfo] = Field(default_factory=list, description="Record of all past autonomous and refinement cycles.")

    # --- Error and Debugging State ---
    error_count_current_cycle: int = Field(0, description="Number of errors encountered in the current cycle.")
    last_error_message: Optional[str] = Field(None, description="Brief message of the last significant error.")
    last_error_timestamp_utc: Optional[datetime] = Field(None)

    # --- Configuration & Metadata ---
    project_configuration_doc_id: Optional[str] = Field(None, description="ChromaDB ID of project-specific configurations used by agents.")
    tags: Optional[List[str]] = Field(None, description="User-defined tags for categorizing or filtering projects.")

    @model_validator(mode='before')
    @classmethod
    def ensure_project_id_is_uuid_like(cls, data: Any) -> Any:
        # Basic check, can be more robust if a strict UUID format is enforced elsewhere
        if isinstance(data, dict) and 'project_id' in data:
            project_id = data['project_id']
            if not isinstance(project_id, str) or len(project_id) < 8: # Arbitrary minimum length
                raise ValueError('project_id must be a string of reasonable length.')
        return data

    def update_last_updated(self):
        self.last_updated_utc = datetime.utcnow()

    def start_new_cycle(self, cycle_id: str, cycle_objective: str) -> CycleInfo:
        self.current_cycle_number += 1
        self.current_cycle_id = cycle_id
        new_cycle_info = CycleInfo(
            cycle_id=cycle_id,
            cycle_number=self.current_cycle_number,
            cycle_objective=cycle_objective,
            status=CycleStatus.IN_PROGRESS,
            start_time_utc=datetime.utcnow()
        )
        # The active cycle's details are managed externally and then added to historical_cycles upon completion/failure.
        # Or, we can have a current_cycle_details field directly.
        # Let's opt for a current_cycle_details field for easier direct updates.
        # This means `historical_cycles` will store *previous* completed cycles.
        # (Adjusting schema definition now to reflect this)

        self.overall_status = ProjectOverallStatus.REFINEMENT_CYCLE_IN_PROGRESS # Or more specific based on cycle type
        if self.current_cycle_number == 0: # Initial cycle might have a more specific status
            self.overall_status = ProjectOverallStatus.LOPRD_GENERATION # Example
        
        self.update_last_updated()
        # Caller should store the new_cycle_info and update it as the cycle progresses.
        return new_cycle_info

    def complete_cycle(
        self, 
        completed_cycle_info: CycleInfo, # The fully populated info for the cycle that just finished
        new_overall_status: ProjectOverallStatus,
        arca_summary_doc_id: Optional[str] = None,
        arca_decision: Optional[str] = None,
        issues_flagged: Optional[List[str]] = None
    ):
        if not self.current_cycle_id or self.current_cycle_id != completed_cycle_info.cycle_id:
            raise ValueError("Mismatch between current_cycle_id and completed_cycle_info.cycle_id.")

        completed_cycle_info.end_time_utc = datetime.utcnow()
        # completed_cycle_info.status is set by the caller based on outcome
        completed_cycle_info.arca_cycle_summary_doc_id = arca_summary_doc_id
        completed_cycle_info.arca_decision_at_cycle_end = arca_decision
        completed_cycle_info.issues_flagged_by_arca_in_cycle = issues_flagged
        
        self.historical_cycles.append(completed_cycle_info)
        
        self.overall_status = new_overall_status
        self.current_cycle_id = None # Ready for a new cycle to be planned or for human review
        
        if new_overall_status == ProjectOverallStatus.CYCLE_COMPLETED_PENDING_REVIEW:
            self.next_action_determined_by = "AUTONOMOUS_AGENT" # ARCA has finished, awaiting human
        
        self.update_last_updated()

    def record_human_review(self, review_record: HumanReviewRecord):
        self.last_human_review = review_record
        self.overall_status = ProjectOverallStatus.HUMAN_REVIEW_IN_PROGRESS # Initial state for review
        
        # Based on human decision, update overall_status and next_action_determined_by
        if review_record.decision_for_next_step == "PROCEED_TO_NEXT_AUTONOMOUS_PHASE":
            self.overall_status = ProjectOverallStatus.AWAITING_NEXT_CYCLE_START # Or a more specific phase status
        elif review_record.decision_for_next_step == "INITIATE_REFINEMENT_CYCLE":
            self.overall_status = ProjectOverallStatus.AWAITING_NEXT_CYCLE_START # Objective will be set by human or next planner
        elif review_record.decision_for_next_step == "PAUSE_PROJECT":
            self.overall_status = ProjectOverallStatus.PROJECT_PAUSED_BY_USER
        elif review_record.decision_for_next_step == "ARCHIVE_PROJECT_SUCCESS":
            self.overall_status = ProjectOverallStatus.PROJECT_COMPLETED_SUCCESSFULLY
        elif review_record.decision_for_next_step == "ARCHIVE_PROJECT_FAILURE":
            self.overall_status = ProjectOverallStatus.PROJECT_FAILED # Or ARCHIVED_FAILURE
        # etc. for other decisions

        self.next_action_determined_by = "HUMAN_REVIEWER"
        self.update_last_updated()

# Example Usage (Conceptual - this would be in StateManager or an orchestrator)
# if __name__ == "__main__":
#     # Initialize project status
#     project_status = ProjectStateV2(
#         project_id="proj_sample_001",
#         initial_user_goal_summary="Create a simple web app for task management."
#     )
#     print(f"Initial Status: {project_status.model_dump_json(indent=2)}")

#     # Start the first cycle (LOPRD generation)
#     current_cycle = project_status.start_new_cycle(
#         cycle_id="cycle_000_loprd_gen",
#         cycle_objective="Generate LOPRD from refined user goal."
#     )
#     project_status.overall_status = ProjectOverallStatus.LOPRD_GENERATION
#     current_cycle.status = CycleStatus.IN_PROGRESS 
#     # ... cycle runs ...
#     # Assume LOPRD generated
#     loprd_artifact = ArtifactLink(artifact_type="LOPRD", artifact_doc_id="loprd_doc_abc123", version_identifier="v1.0")
#     current_cycle.cycle_produced_artifacts.append(loprd_artifact)
#     project_status.latest_accepted_loprd_doc_id = loprd_artifact.artifact_doc_id
    
#     # Complete the LOPRD cycle
#     current_cycle.status = CycleStatus.COMPLETED_SUCCESS
#     project_status.complete_cycle(
#         completed_cycle_info=current_cycle,
#         new_overall_status=ProjectOverallStatus.CYCLE_COMPLETED_PENDING_REVIEW,
#         arca_summary_doc_id="arca_summary_loprd_xyz789",
#         arca_decision="LOPRD generated, meets initial criteria, recommend proceeding to Blueprint.",
#     )
#     project_status.arca_best_state_summary_doc_id = "arca_summary_loprd_xyz789" # Update ARCA's view
#     print(f"Status after LOPRD cycle: {project_status.model_dump_json(indent=2)}")

#     # Human Review
#     review = HumanReviewRecord(
#         reviewed_cycle_id=current_cycle.cycle_id,
#         feedback_summary="LOPRD looks good. Approved for blueprinting.",
#         decision_for_next_step="PROCEED_TO_NEXT_AUTONOMOUS_PHASE"
#     )
#     project_status.record_human_review(review)
#     print(f"Status after human review: {project_status.model_dump_json(indent=2)}")

#     # Start next cycle (Blueprint generation)
#     current_cycle = project_status.start_new_cycle(
#         cycle_id="cycle_001_blueprint_gen",
#         cycle_objective="Generate Project Blueprint from approved LOPRD."
#     )
#     project_status.overall_status = ProjectOverallStatus.ARCHITECTING
#     # ... and so on ... 