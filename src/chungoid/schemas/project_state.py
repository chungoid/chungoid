from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Assuming CommonStatus from common_enums.py or similar
from .common_enums import StageStatus # Added import for StageStatus

class StageRecord(BaseModel):
    stage_id: str = Field(..., description="Identifier of the stage within the run.")
    agent_id: str = Field(..., description="Agent ID used for this stage.")
    start_time: datetime = Field(default_factory=datetime.now, description="Timestamp when the stage started.")
    end_time: Optional[datetime] = Field(None, description="Timestamp when the stage ended.")
    status: StageStatus = Field(..., description="Final status of the stage.")
    outputs_summary: Optional[str] = Field(None, description="A brief summary or reference to the stage outputs. Full outputs might be too large to store here directly.") # Avoid storing large outputs
    error_details: Optional[Dict[str, Any]] = Field(None, description="Details of any error that occurred during the stage.")


class RunRecord(BaseModel):
    run_id: str = Field(..., description="Unique identifier for this execution run.")
    flow_id: str = Field(..., description="Identifier of the flow/plan being executed.")
    start_time: datetime = Field(default_factory=datetime.now, description="Timestamp when the run started.")
    end_time: Optional[datetime] = Field(None, description="Timestamp when the run ended.")
    status: StageStatus = Field(..., description="Final status of the run.")
    initial_context_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of the initial context provided for the run.")
    final_outputs_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of the final outputs of the run.") # Avoid storing large outputs
    error_message: Optional[str] = Field(None, description="Overall error message if the run failed.")
    stages: List[StageRecord] = Field(default_factory=list, description="Chronological record of stages executed in this run.")


class CycleHistoryItem(BaseModel):
    cycle_id: str = Field(..., description="Unique identifier for this cycle.")
    start_time: Optional[datetime] = Field(None, description="Timestamp when the cycle started.")
    end_time: Optional[datetime] = Field(None, description="Timestamp when the cycle ended.")
    goal_for_cycle: Optional[str] = Field(None, description="Brief description of the cycle's goal, or reference to a goal artifact ID.")
    
    # Using List[str] for artifact IDs, assuming they are strings (e.g., UUIDs or ChromaDB document IDs)
    key_artifacts_generated_or_modified_ids: List[str] = Field(default_factory=list, description="ChromaDB IDs of key artifacts generated or modified during this cycle.")
    
    arca_summary_of_cycle_outcome: Optional[str] = Field(None, description="ARCA's summary of what was achieved or attempted in the cycle.")
    
    issues_flagged_for_human_review: List[Dict[str, Any]] = Field(default_factory=list, description="List of issues flagged by ARCA for human review at the end of the cycle. Each dict might contain 'issue_id', 'description', 'relevant_artifact_ids'.")
    
    human_feedback_and_directives_for_next_cycle: Optional[str] = Field(None, description="Human feedback or reference to directives for the next cycle (e.g., path to a new user goal file or a ChromaDB artifact ID).")

class ProjectStateV2(BaseModel):
    project_id: str = Field(..., description="Globally unique identifier for this project instance.")
    project_name: Optional[str] = Field(None, description="User-defined name for the project.")
    # Consider defining an Enum for overall_project_status if specific states are known
    # e.g., class OverallProjectStatus(str, Enum): INITIALIZING = "initializing"; CYCLE_IN_PROGRESS = "cycle_in_progress"; ...
    overall_project_status: str = Field(..., description="Overall status of the project (e.g., initializing, cycle_in_progress, pending_human_review, project_complete).")
    
    current_cycle_id: Optional[str] = Field(None, description="Identifier of the currently active cycle, if any.")
    
    cycle_history: List[CycleHistoryItem] = Field(default_factory=list, description="A list of all completed and the current in-progress cycle.")
    
    run_history: Dict[str, RunRecord] = Field(default_factory=dict, description="A dictionary of all execution runs, keyed by run_id.") # Added run_history

    master_loprd_id: Optional[str] = Field(None, description="ChromaDB ID of the latest master/approved LOPRD for the project.")
    master_blueprint_id: Optional[str] = Field(None, description="ChromaDB ID of the latest master/approved Project Blueprint.")
    master_execution_plan_id: Optional[str] = Field(None, description="ChromaDB ID of the latest master/approved Master Execution Plan.")
    
    # This could be a specific version tag, a collection name suffix, or a direct reference
    # depending on how live codebase snapshots are managed in ChromaDB.
    link_to_live_codebase_collection_snapshot: Optional[str] = Field(None, description="Reference to the current live codebase snapshot in ChromaDB.")

    # To ensure compatibility with existing StateManager which might manage 'runs' and 'master_plans'
    # We can include them here or decide if ProjectStateV2 fully replaces the old structure.
    # For now, let's assume V2 is a distinct, new top-level structure.
    # If StateManager needs to manage both old and new, further thought is needed on structure.

    # Example of how a run (from existing StateManager) might be incorporated if needed,
    # but for now, focusing on the V2 schema as per blueprint for P3.1.1
    # runs: List[Dict[str, Any]] = Field(default_factory=list, description="History of individual execution runs (stages, statuses, artifacts). From older StateManager schema.")
    # master_plans: Dict[str, Any] = Field(default_factory=dict, description="Storage for MasterExecutionPlan definitions. From older StateManager schema.")

    # Metadata for the state file itself
    schema_version: str = Field(default="2.0", description="Version of this project state schema.")
    last_updated: Optional[datetime] = Field(None, description="Timestamp of when this state file was last updated.")

    class Config:
        # Pydantic V2 config
        json_schema_extra = {
            "examples": [
                {
                    "project_id": "proj_abc123",
                    "overall_project_status": "pending_human_review",
                    "current_cycle_id": "cycle_002_dev_phase",
                    "cycle_history": [
                        {
                            "cycle_id": "cycle_001_loprd_refinement",
                            "start_time": "2024-05-20T10:00:00Z",
                            "end_time": "2024-05-20T18:00:00Z",
                            "goal_for_cycle": "Refine LOPRD for initial user feedback.",
                            "key_artifacts_generated_or_modified_ids": ["loprd_v1_final_doc_id", "praa_report_loprd_v1_id"],
                            "arca_summary_of_cycle_outcome": "LOPRD refined based on PRAA feedback. Key risks addressed. Ready for blueprinting.",
                            "issues_flagged_for_human_review": [],
                            "human_feedback_and_directives_for_next_cycle": "Approved. Proceed to blueprinting with focus on modular design."
                        },
                        {
                            "cycle_id": "cycle_002_dev_phase",
                            "start_time": "2024-05-21T09:00:00Z",
                            "goal_for_cycle": "Develop core features A, B, C.",
                            "key_artifacts_generated_or_modified_ids": ["blueprint_v1_id", "mep_v1_id", "module_A_code_id", "module_B_code_id"],
                            "arca_summary_of_cycle_outcome": "Features A & B developed and unit tested. Feature C encountered integration issues.",
                            "issues_flagged_for_human_review": [
                                {"issue_id": "issue_001", "description": "Feature C integration with external API failed. Needs config review.", "relevant_artifact_ids": ["module_C_code_id", "api_docs_id"]}
                            ],
                        }
                    ],
                    "master_loprd_id": "loprd_v1_final_doc_id",
                    "master_blueprint_id": "blueprint_v1_id",
                    "master_execution_plan_id": "mep_v1_id",
                    "link_to_live_codebase_collection_snapshot": "code_snapshot_cycle002_tag",
                    "schema_version": "2.0",
                    "last_updated": "2024-05-21T17:00:00Z"
                }
            ]
        }

# It's good practice to add __all__ if this file grows
__all__ = ["CycleHistoryItem", "ProjectStateV2", "RunRecord", "StageRecord"] # Added new models 