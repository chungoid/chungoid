from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ARCAReviewArtifactType(str, Enum):
    LOPRD = "LOPRD"
    PROJECT_BLUEPRINT = "PROJECT_BLUEPRINT"
    MASTER_EXECUTION_PLAN = "MASTER_EXECUTION_PLAN"
    CODE_MODULE = "CODE_MODULE"
    TEST_SUITE = "TEST_SUITE"
    OPTIMIZATION_SUGGESTION_REPORT = "OPTIMIZATION_SUGGESTION_REPORT"
    REQUIREMENTS_DOCUMENT = "REQUIREMENTS_DOCUMENT"
    GENERATE_PROJECT_DOCUMENTATION = "GENERATE_PROJECT_DOCUMENTATION"
    CODE_MODULE_TEST_FAILURE = "CODE_MODULE_TEST_FAILURE"
    RISK_ASSESSMENT_REPORT = "RISK_ASSESSMENT_REPORT"
    TRACEABILITY_REPORT = "TRACEABILITY_REPORT"
    PROJECT_DOCUMENTATION = "PROJECT_DOCUMENTATION"
    # Add other types as they become relevant for ARCA review
    # Example: USER_STORY, DESIGN_DOCUMENT, etc.

class ARCAReviewInput(BaseModel):
    project_id: str = Field(..., description="ID of the project this review pertains to.")
    cycle_id: str = Field(..., description="ID of the current cycle.")
    artifact_type: ARCAReviewArtifactType = Field(..., description="The type of artifact being submitted for ARCA review.")
    artifact_doc_id: str = Field(..., description="ChromaDB document ID of the artifact to be reviewed.")
    artifact_name: Optional[str] = Field(None, description="Specific name of the artifact, e.g., a filename for CODE_MODULE.")
    generator_agent_id: str = Field(..., description="ID of the agent that generated/modified the artifact.")
    generator_agent_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score from the generating agent about the artifact's quality/completeness.")
    # Optional field for providing the content directly if not relying solely on ChromaDB retrieval by ARCA
    # artifact_content_override: Optional[str] = Field(None, description="Optional direct content of the artifact, bypassing ChromaDB retrieval if provided.")


class ARCAEvaluationDecision(str, Enum):
    APPROVE_ARTIFACT_AS_IS = "APPROVE_ARTIFACT_AS_IS"
    REQUEST_REVISIONS_TO_ARTIFACT = "REQUEST_REVISIONS_TO_ARTIFACT"
    REJECT_ARTIFACT_AND_FLAG_FOR_HUMAN_REVIEW = "REJECT_ARTIFACT_AND_FLAG_FOR_HUMAN_REVIEW"
    MODIFY_PLAN_NEW_TASK_FROM_SUGGESTION = "MODIFY_PLAN_NEW_TASK_FROM_SUGGESTION" # Specific to OPTIMIZATION_SUGGESTION_REPORT review
    MODIFY_PLAN_UPDATE_EXISTING_TASK = "MODIFY_PLAN_UPDATE_EXISTING_TASK" # Specific to OPTIMIZATION_SUGGESTION_REPORT review
    FLAG_FOR_HUMAN_REVIEW_ESCALATION = "FLAG_FOR_HUMAN_REVIEW_ESCALATION" # General escalation

class ARCAOutput(BaseModel):
    decision: ARCAEvaluationDecision = Field(..., description="The primary decision made by ARCA after reviewing the artifact.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="ARCA's confidence in its decision.")
    summary: str = Field(..., description="A human-readable summary of ARCA's findings and rationale.")
    
    # Optional fields depending on the decision
    revision_requests: Optional[List[str]] = Field(None, description="Specific revision requests if decision is REQUEST_REVISIONS_TO_ARTIFACT.")
    human_review_reason: Optional[str] = Field(None, description="Reason for flagging for human review if decision involves escalation.")
    
    # Specific fields for plan modification decisions (from OPTIMIZATION_SUGGESTION_REPORT review)
    new_master_plan_doc_id: Optional[str] = Field(None, description="If plan is modified, the ChromaDB doc ID of the new MasterExecutionPlan.")
    newly_added_stage_id: Optional[str] = Field(None, description="If a new task/stage was added, its ID in the new plan.")
    modified_existing_stage_id: Optional[str] = Field(None, description="If an existing task/stage was modified, its ID.")
    next_stage_id: Optional[str] = Field(None, description="Suggested next stage_id to execute in the master plan (could be the new/modified one or subsequent).")

    # Fallback for any other structured data ARCA might want to return
    additional_structured_data: Optional[Dict[str, Any]] = Field(None, description="Any other structured data relevant to the decision.")

__all__ = [
    "ARCAReviewArtifactType",
    "ARCAReviewInput",
    "ARCAEvaluationDecision",
    "ARCAOutput"
] 