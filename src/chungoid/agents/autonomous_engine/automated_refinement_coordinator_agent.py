from __future__ import annotations

import logging
import datetime
import uuid
import json
from typing import Any, Dict, Optional, Literal, Union, ClassVar, List, Type, get_args
from enum import Enum
import yaml

from pydantic import BaseModel, Field, model_validator, PrivateAttr

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider # Assuming direct LLM call for complex decisions might be an option
from chungoid.utils.prompt_manager import PromptManager # If decision logic uses prompts
from chungoid.schemas.common import ConfidenceScore
from chungoid.schemas.arca_request_and_response import ARCAReviewArtifactType # ADDED IMPORT
# Mocked Agent Inputs for agents ARCA might call
# from .product_analyst_agent import ProductAnalystInput # For LOPRD refinement - OLD
from . import product_analyst_agent as pa_module # For LOPRD refinement - NEW
from .architect_agent import ArchitectAgentInput # For blueprint refinement
from chungoid.schemas.agent_master_planner import MasterPlannerInput # For instructing plan refinement
# Import the new documentation agent's input schema
from .project_documentation_agent import ProjectDocumentationAgentInput, ProjectDocumentationAgent_v1 

# NEW: Import StateManager and related schemas
from chungoid.utils.state_manager import StateManager, StatusFileError
from chungoid.schemas.project_status_schema import ProjectStateV2, CycleStatus, ProjectOverallStatus, CycleInfo # Corrected import

from chungoid.utils.agent_registry import AgentCard # For AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1, 
    GENERATED_CODE_ARTIFACTS_COLLECTION,
    RetrieveArtifactOutput, # Added
    RISK_ASSESSMENT_REPORTS_COLLECTION, # Added
    OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION, # Added
    TRACEABILITY_REPORTS_COLLECTION, # Added
    REVIEW_REPORTS_COLLECTION # Added
)

# Added import for FailedTestReport
from chungoid.schemas.code_debugging_agent_schemas import FailedTestReport
# NEW Imports for Debugging Logic
from chungoid.schemas.code_debugging_agent_schemas import DebuggingTaskInput, DebuggingTaskOutput
from chungoid.agents.autonomous_engine.code_debugging_agent import CodeDebuggingAgent_v1

# NEW: Import utility schemas for CodeIntegration and TestRunner
from chungoid.schemas.autonomous_engine_utility_schemas import (
    CodeIntegrationTaskInput,
    CodeIntegrationTaskOutput,
    TestRunnerTaskInput,
    TestRunnerTaskOutput
)

# NEW: Import for ARCA Logging
from chungoid.schemas.agent_logs import ARCALogEntry
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, LogStorageConfirmation # For type hinting PCMA
# NEW QA LOG IMPORTS
from chungoid.schemas.agent_logs import QualityAssuranceLogEntry, QAEventType, OverallQualityStatus
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    QUALITY_ASSURANCE_LOGS_COLLECTION, 
    ARTIFACT_TYPE_QA_LOG_ENTRY_JSON,
    StoreArtifactInput # Ensure this is imported for store_artifact
)

# Import for plan modification
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec

logger = logging.getLogger(__name__)

ARCA_PROMPT_NAME = "automated_refinement_coordinator_agent_v1.yaml" # If LLM-based decision making is used
ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME = "arca_optimization_evaluator_v1_prompt.yaml" # NEW PROMPT

# Constants for ARCA behavior
MAX_DEBUGGING_ATTEMPTS_PER_MODULE: ClassVar[int] = 3 # Added ClassVar
DEFAULT_ACCEPTANCE_THRESHOLD: ClassVar[float] = 0.85 # MODIFIED: Added ClassVar


# --- Input and Output Schemas for ARCA --- #
class ARCAReviewInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this ARCA review task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    cycle_id: str = Field(..., description="The ID of the current refinement cycle.")
    artifact_type: ARCAReviewArtifactType = Field(..., description="The type of artifact ARCA needs to review or act upon.")
    artifact_doc_id: Optional[str] = Field(
        None, description="The document ID of the primary artifact in ChromaDB."
    )
    # New fields for CodeModule_TestFailure
    code_module_file_path: Optional[str] = Field(
        None, description="Path to the code file that has failed tests. Required if artifact_type is CodeModule_TestFailure."
    )
    failed_test_report_details: Optional[List[FailedTestReport]] = Field(
        None, description="List of failed test report objects. Required if artifact_type is CodeModule_TestFailure."
    )
    generator_agent_id: str = Field(..., description="ID of the agent that generated the artifact.")
    generator_agent_confidence: Optional[ConfidenceScore] = Field(None, description="Confidence score from the generating agent.")
    
    # Specific inputs based on artifact_type
    praa_risk_report_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the PRAA Risk Assessment Report for the artifact.")
    praa_optimization_report_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the PRAA Optimization Suggestions Report.")
    # praa_confidence_score: Optional[ConfidenceScore] = Field(None, description="Confidence score from PRAA.")

    rta_report_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the RTA Traceability Report (e.g., LOPRD to Blueprint, Blueprint to Plan).")
    # rta_confidence_score: Optional[ConfidenceScore] = Field(None, description="Confidence score from RTA.")

    # For Blueprint Reviewer feedback if applicable before planning
    blueprint_reviewer_report_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the Blueprint Reviewer Agent's report.")

    # Inputs specifically for initiating documentation generation after a cycle completes
    # These would be populated by the orchestrator when calling ARCA to trigger documentation
    # Or ARCA itself sets them up when transitioning to documentation.
    # For clarity, let's assume if artifact_type is e.g. MasterExecutionPlan and ACCEPTED,
    # ARCA will construct the input for ProjectDocumentationAgent.
    final_loprd_doc_id_for_docs: Optional[str] = Field(None, description="Doc ID of the final LOPRD for documentation.")
    final_blueprint_doc_id_for_docs: Optional[str] = Field(None, description="Doc ID of the final Blueprint for documentation.")
    final_plan_doc_id_for_docs: Optional[str] = Field(None, description="Doc ID of the final MasterExecutionPlan for documentation.")
    final_code_root_path_for_docs: Optional[str] = Field(None, description="Path to generated code root for documentation.")
    final_test_summary_doc_id_for_docs: Optional[str] = Field(None, description="Doc ID of the final test summary for documentation.")

    # Ensure that relevant report IDs are provided based on artifact type
    @model_validator(mode='after')
    def check_report_ids(self) -> 'ARCAReviewInput':
        if self.artifact_type in ["LOPRD", "Blueprint", "MasterExecutionPlan", "CodeModule"]:
            if not self.praa_risk_report_doc_id or not self.praa_optimization_report_doc_id:
                logger.warning(f"PRAA report IDs missing for {self.artifact_type} review in ARCA. This might limit decision quality.")
        if self.artifact_type in ["Blueprint", "MasterExecutionPlan", "CodeModule"]:
            if not self.rta_report_doc_id:
                logger.warning(f"RTA report ID missing for {self.artifact_type} review in ARCA. This might limit decision quality.")
        return self

    @model_validator(mode='after')
    def check_artifact_specific_fields(cls, values: 'ARCAReviewInput') -> 'ARCAReviewInput':
        artifact_type = values.artifact_type
        artifact_doc_id = values.artifact_doc_id
        code_module_file_path = values.code_module_file_path
        failed_test_report_details = values.failed_test_report_details

        if artifact_type == "LOPRD":
            if not artifact_doc_id:
                raise ValueError("artifact_doc_id is required for LOPRD artifact_type")
        elif artifact_type == "ProjectBlueprint":
            if not artifact_doc_id:
                raise ValueError("artifact_doc_id is required for ProjectBlueprint artifact_type")
            # Potentially add checks for LOPRD related IDs if needed for blueprint context
        elif artifact_type == "MasterExecutionPlan":
            if not artifact_doc_id:
                raise ValueError("artifact_doc_id is required for MasterExecutionPlan artifact_type")
        elif artifact_type == "CodeModule":
            if not artifact_doc_id:
                raise ValueError("artifact_doc_id is required for CodeModule artifact_type")
        elif artifact_type == "CodeModule_TestFailure": # New validation block
            if not code_module_file_path:
                raise ValueError("code_module_file_path is required for CodeModule_TestFailure artifact_type")
            if not failed_test_report_details:
                raise ValueError("failed_test_report_details are required for CodeModule_TestFailure artifact_type")
            # artifact_doc_id might still be relevant here if it refers to the CodeModule's ID
            # For example, if the CodeModule itself is an artifact, and test failures are *about* it.
            # If artifact_doc_id is always the code_module_file_path for this type, ensure consistency or clarify.
            # For now, not making artifact_doc_id mandatory for CodeModule_TestFailure but it might be needed for context.

        # Add checks for other artifact_types as they are defined and have specific field requirements
        # e.g., RiskAssessmentReport, TraceabilityReport etc.
        elif artifact_type in ["RiskAssessmentReport", "TraceabilityReport", "OptimizationSuggestionReport", "ProjectDocumentation"]:
            if not artifact_doc_id:
                raise ValueError(f"artifact_doc_id is required for {artifact_type}")
        
        # Ensure that generator_agent_id and generator_agent_confidence are present if the artifact_type
        # implies a generated artifact that needs quality assessment (e.g. CodeModule, LOPRD, Blueprint)
        if artifact_type in ["LOPRD", "ProjectBlueprint", "MasterExecutionPlan", "CodeModule"]:
            if values.generator_agent_id is None or values.generator_agent_confidence is None:
                # Allowing these to be None for now as per original schema, but consider if they should be mandatory
                pass # raise ValueError(f"generator_agent_id and generator_agent_confidence are required for {artifact_type}")

        return values

class ARCAOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    reviewed_artifact_doc_id: str = Field(..., description="ChromaDB ID of the artifact that was reviewed.")
    reviewed_artifact_type: ARCAReviewArtifactType = Field(..., description="Type of the artifact reviewed.")
    decision: Literal["ACCEPT_ARTIFACT", "REFINEMENT_REQUIRED", "ESCALATE_TO_USER", "TEST_FAILURE_HANDOFF", "PROCEED_TO_DOCUMENTATION", "PLAN_MODIFIED_NEW_TASKS_ADDED", "ERROR"] = Field(..., description="The final decision of the ARCA review.")
    reasoning: str = Field(..., description="The detailed reasoning behind the decision.")
    confidence_in_decision: Optional[ConfidenceScore] = Field(None, description="ARCA's confidence in its own decision.")
    
    # If decision is REFINEMENT_REQUIRED, these fields provide details for the orchestrator
    next_agent_id_for_refinement: Optional[str] = Field(None, description="The agent_id to call for refinement (e.g., ProductAnalystAgent_v1, ArchitectAgent_v1, SystemMasterPlannerAgent_v1).")
    next_agent_input: Optional[Union[pa_module.ProductAnalystAgentInput, ArchitectAgentInput, MasterPlannerInput, ProjectDocumentationAgentInput, Dict[str, Any]]] = Field(None, description="The full input payload for the next agent if refinement is needed.")
    
    # If decision is TEST_FAILURE_HANDOFF
    debugging_task_input: Optional[DebuggingTaskInput] = Field(None, description="Input for the CodeDebuggingAgent if a test failure is being handed off.")

    # If decision is ACCEPT_ARTIFACT or PLAN_MODIFIED_NEW_TASKS_ADDED
    final_artifact_doc_id: Optional[str] = Field(None, description="The doc_id of the artifact if accepted (usually same as input artifact_doc_id). For PLAN_MODIFIED_NEW_TASKS_ADDED, this will be the new MasterExecutionPlan doc_id.")
    
    # If decision is PLAN_MODIFIED_NEW_TASKS_ADDED
    new_master_plan_doc_id: Optional[str] = Field(None, description="The document ID of the newly created/modified MasterExecutionPlan if new tasks were added.")

    error_message: Optional[str] = Field(None, description="Error message if ARCA encountered an issue.")


class AutomatedRefinementCoordinatorAgent_v1(BaseAgent[ARCAReviewInput, ARCAOutput]):
    AGENT_ID: ClassVar[str] = "AutomatedRefinementCoordinatorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Automated Refinement Coordinator Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Coordinates the iterative refinement of project artifacts, invoking specialist agents as needed."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "automated_refinement_coordinator_agent_v1.yaml" # Points to server_prompts/autonomous_engine/
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.AUTONOMOUS_COORDINATION # MODIFIED
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL # Usually internal, invoked by higher orchestrator
    INPUT_SCHEMA: ClassVar[Type[ARCAReviewInput]] = ARCAReviewInput
    OUTPUT_SCHEMA: ClassVar[Type[ARCAOutput]] = ARCAOutput

    # Declare private attributes using PrivateAttr
    _llm_provider: Optional[LLMProvider] = PrivateAttr(default=None)
    _prompt_manager: Optional[PromptManager] = PrivateAttr(default=None)
    _project_chroma_manager: Optional[ProjectChromaManagerAgent_v1] = PrivateAttr(default=None)
    _code_debugging_agent_instance: Optional[CodeDebuggingAgent_v1] = PrivateAttr(default=None)
    _logger_instance: logging.Logger = PrivateAttr() # No default, will be set in __init__
    _state_manager: Optional[StateManager] = PrivateAttr(default=None)
    _current_debug_attempts_for_module: int = PrivateAttr(default=0)
    _last_feedback_doc_id: Optional[str] = PrivateAttr(default=None)

    # Thresholds for decision making (can be made configurable later)
    MIN_GENERATOR_CONFIDENCE: ClassVar[float] = 0.6
    MIN_PRAA_CONFIDENCE: ClassVar[float] = 0.5
    MIN_RTA_CONFIDENCE: ClassVar[float] = 0.6
    MIN_DOC_AGENT_CONFIDENCE_FOR_ACCEPT: ClassVar[float] = 0.5

    DEFAULT_ACCEPTANCE_THRESHOLD: ClassVar[float] = 0.75 # Default acceptance threshold

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        prompt_manager: Optional[PromptManager] = None,
        project_chroma_manager: Optional[ProjectChromaManagerAgent_v1] = None,
        system_context: Optional[Dict[str, Any]] = None,
        state_manager: Optional[StateManager] = None,
        **kwargs  # Pydantic will populate model fields from here
    ):
        super().__init__(**kwargs) # Initialize Pydantic model fields first

        # Now initialize PrivateAttrs
        if system_context and "logger" in system_context:
            self._logger_instance = system_context["logger"]
        else:
            # Ensure AGENT_ID is accessible, might need to be self.AGENT_ID if BaseAgent sets it up
            # or AutomatedRefinementCoordinatorAgent_v1.AGENT_ID if accessed as class variable
            self._logger_instance = logging.getLogger(AutomatedRefinementCoordinatorAgent_v1.AGENT_ID)

        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        self._project_chroma_manager = project_chroma_manager
        self._state_manager = state_manager
        self._current_debug_attempts_for_module = 0 # Explicitly initialize, though PrivateAttr has default
        self._last_feedback_doc_id = None # Explicitly initialize

        # Initialize CodeDebuggingAgent instance
        if self._llm_provider and self._prompt_manager and self._project_chroma_manager:
            try:
                # Ensure CodeDebuggingAgent_v1 is correctly imported
                self._code_debugging_agent_instance = CodeDebuggingAgent_v1(
                    llm_provider=self._llm_provider,
                    prompt_manager=self._prompt_manager,
                    project_chroma_manager=self._project_chroma_manager, # Pass the PCMA instance
                    system_context={"logger": self._logger_instance.getChild("CodeDebuggingAgent_v1")}
                )
            except Exception as e:
                self._logger_instance.error(f"Failed to initialize CodeDebuggingAgent_v1 within ARCA: {e}", exc_info=True)
                self._code_debugging_agent_instance = None
        else:
            self._logger_instance.warning("LLMProvider, PromptManager, or ProjectChromaManager not available for CodeDebuggingAgent initialization.")
            self._code_debugging_agent_instance = None

    async def _log_event_to_pcma(
        self,
        pcma_agent: ProjectChromaManagerAgent_v1,
        project_id: str,
        cycle_id: str,
        arca_task_id: str,
        event_type: Literal[
            "ARCA_INVOCATION_START",
            "ARCA_DECISION_MADE",
            "SUB_AGENT_INVOCATION_START",
            "SUB_AGENT_INVOCATION_END",
            "MAX_DEBUG_ATTEMPTS_REACHED",
            "STATE_UPDATE_ATTEMPT",
            "STATE_UPDATE_SUCCESS",
            "STATE_UPDATE_FAILURE",
            "ARCA_INTERNAL_ERROR"
        ],
        event_details: Dict[str, Any],
        severity: Literal["INFO", "WARNING", "ERROR"] = "INFO",
        related_doc_ids: Optional[List[str]] = None
    ):
        """Helper method to log an ARCA event to ProjectChromaManagerAgent."""
        log_entry = ARCALogEntry(
            agent_id=self.AGENT_ID, # Added agent_id
            # log_id is generated by default_factory in ARCALogEntry
            timestamp=datetime.datetime.now(datetime.timezone.utc), # Ensure timestamp is set here
            arca_task_id=arca_task_id,
            project_id=project_id,
            cycle_id=cycle_id,
            event_type=event_type,
            event_details=event_details,
            severity=severity,
            related_artifact_doc_ids=related_doc_ids or []
        )
        try:
            # Assuming pcma_agent is already resolved and available
            confirmation: LogStorageConfirmation = await pcma_agent.log_arca_event(
                project_id=project_id,
                cycle_id=cycle_id,
                log_entry=log_entry
            )
            if confirmation.status != "SUCCESS":
                self._logger_instance.error(f"PCMA failed to log ARCA event ({event_type}): {confirmation.message}. Log ID: {log_entry.log_id}")
            else:
                # self._logger_instance.info(f"ARCA event ({event_type}) logged to PCMA. Log ID: {confirmation.log_id}") # Can be too verbose
                pass
        except Exception as e:
            self._logger_instance.error(f"Exception during ARCA event logging to PCMA for event type {event_type}, Log ID {log_entry.log_id}: {e}", exc_info=True)


    async def invoke_async(
        self, 
        task_input: ARCAReviewInput, 
        full_context: Optional[Dict[str, Any]] = None,
    ) -> ARCAOutput:
        # --- Initialization and Setup ---
        self._logger_instance.info(f"ARCA invoked for task_id: {task_input.task_id}, project_id: {task_input.project_id}, artifact_type: {task_input.artifact_type.value}")
        
        if not self._project_chroma_manager:
            self._logger_instance.error("ProjectChromaManagerAgent not initialized in ARCA.")
            return ARCAOutput(
                task_id=task_input.task_id, project_id=task_input.project_id,
                reviewed_artifact_doc_id=task_input.artifact_doc_id or "N/A",
                reviewed_artifact_type=task_input.artifact_type, decision="ERROR",
                reasoning="ARCA internal configuration error: PCMA not available.",
                error_message="ARCA internal configuration error: PCMA not available."
            )
        
        # Removed StateManager check from critical path here, will check where needed.

        await self._log_event_to_pcma(
            pcma_agent=self._project_chroma_manager, project_id=task_input.project_id,
            cycle_id=task_input.cycle_id, arca_task_id=task_input.task_id,
            event_type="ARCA_INVOCATION_START",
            event_details={
                "artifact_type": task_input.artifact_type.value,
                "artifact_doc_id": task_input.artifact_doc_id,
                "generator_agent_id": task_input.generator_agent_id
            }
        )

        praa_structured_suggestions: List[Dict[str, Any]] = []
        all_structured_suggestions_for_llm: List[Dict[str, Any]] = []
        praa_optimization_report_content_md: Optional[str] = None
        blueprint_reviewer_report_content_md: Optional[str] = None
        praa_risk_report_content_md: Optional[str] = None
        rta_report_content_md: Optional[str] = None
        new_tasks_to_add_to_plan: List[Dict[str, Any]] = [] # Initialize here

        # --- NEW: Handle OPTIMIZATION_SUGGESTION_REPORT as primary input for suggestions ---
        if task_input.artifact_type == ARCAReviewArtifactType.OPTIMIZATION_SUGGESTION_REPORT and task_input.artifact_doc_id and self._project_chroma_manager:
            self._logger_instance.info(f"ARCA: Artifact type is OPTIMIZATION_SUGGESTION_REPORT. Attempting to parse suggestions from its content (doc_id: {task_input.artifact_doc_id}).")
            try:
                retrieved_opt_report_artifact: RetrieveArtifactOutput = await self._project_chroma_manager.retrieve_artifact(
                    base_collection_name=OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION,
                    document_id=task_input.artifact_doc_id
                )
                # Check if retrieval was successful and content exists
                if retrieved_opt_report_artifact.status == "SUCCESS" and retrieved_opt_report_artifact.content: # CORRECTED .artifact_content to .content
                    # Content should be the JSON string of the report
                    report_content_str = retrieved_opt_report_artifact.content
                    if isinstance(report_content_str, str):
                        try:
                            # Assuming the content of OPTIMIZATION_SUGGESTION_REPORT (like the mock) is a JSON string
                            # with a "suggestions" key.
                            report_content_json = json.loads(report_content_str)
                            suggestions_from_input_report = report_content_json.get("suggestions", [])
                            
                            if isinstance(suggestions_from_input_report, list):
                                for sugg in suggestions_from_input_report:
                                    if isinstance(sugg, dict): # Ensure suggestion is a dict
                                        sugg['source_report'] = 'OPTIMIZATION_SUGGESTION_REPORT_INPUT'
                                        all_structured_suggestions_for_llm.append(sugg)
                                    else:
                                        self._logger_instance.warning(f"Suggestion in OPTIMIZATION_SUGGESTION_REPORT was not a dictionary: {sugg}")
                                self._logger_instance.info(f"Successfully parsed {len(suggestions_from_input_report)} suggestions from OPTIMIZATION_SUGGESTION_REPORT artifact doc_id: {task_input.artifact_doc_id}.")
                            else:
                                self._logger_instance.warning(f"Content of OPTIMIZATION_SUGGESTION_REPORT (doc_id: {task_input.artifact_doc_id}) had 'suggestions' field, but it was not a list.")
                        except json.JSONDecodeError as e:
                            self._logger_instance.error(f"Failed to parse JSON from OPTIMIZATION_SUGGESTION_REPORT (doc_id: {task_input.artifact_doc_id}): {e}. Content: {report_content_str[:500]}")
                        except Exception as e_parse: # Catch other potential errors during parsing/processing
                            self._logger_instance.error(f"Error processing suggestions from OPTIMIZATION_SUGGESTION_REPORT (doc_id: {task_input.artifact_doc_id}): {e_parse}", exc_info=True)
                    else:
                        self._logger_instance.warning(f"Failed to retrieve content for OPTIMIZATION_SUGGESTION_REPORT (doc_id: {task_input.artifact_doc_id}). Status: {retrieved_opt_report_artifact.status}, Message: {retrieved_opt_report_artifact.error_message or 'Unknown error'}") # MODIFIED
                else:
                    self._logger_instance.warning(f"Failed to retrieve content for OPTIMIZATION_SUGGESTION_REPORT (doc_id: {task_input.artifact_doc_id}). Status: {retrieved_opt_report_artifact.status}, Message: {retrieved_opt_report_artifact.error_message or 'Unknown error'}") # MODIFIED
            except Exception as e_retrieve:
                self._logger_instance.error(f"Exception retrieving OPTIMIZATION_SUGGESTION_REPORT (doc_id: {task_input.artifact_doc_id}): {e_retrieve}", exc_info=True)
        # --- END NEW BLOCK ---

        # --- Retrieve and Parse PRAA Optimization Report ---
        # Only run this if suggestions weren't already populated from a primary OPTIMIZATION_SUGGESTION_REPORT input,
        # or if we want to aggregate (for now, let's make it conditional to avoid double processing if not intended)
        if not all_structured_suggestions_for_llm and task_input.praa_optimization_report_doc_id and self._project_chroma_manager:
            try:
                retrieved_praa_opt_report = await self._project_chroma_manager.retrieve_artifact(
                    doc_id=task_input.praa_optimization_report_doc_id,
                    collection_name=OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION
                )
                self._logger_instance.info(f"Retrieving PRAA optimization report: {task_input.praa_optimization_report_doc_id}")
                retrieved_praa_opt_report: RetrieveArtifactOutput = await self._project_chroma_manager.retrieve_artifact(
                    doc_id=task_input.praa_optimization_report_doc_id, # MODIFIED: doc_id directly
                    collection_name=OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION 
                )
                if retrieved_praa_opt_report.status == "SUCCESS" and retrieved_praa_opt_report.artifact_content:
                    try:
                        praa_output_json = json.loads(retrieved_praa_opt_report.artifact_content)
                        praa_optimization_report_content_md = praa_output_json.get("optimization_opportunities_report_md")
                        
                        suggestions_from_praa = praa_output_json.get("structured_optimization_suggestions_json", [])
                        if isinstance(suggestions_from_praa, list):
                            praa_structured_suggestions = suggestions_from_praa
                            for sugg in praa_structured_suggestions: # MODIFIED: Enrich with source_report
                                sugg['source_report'] = 'PRAA'
                                all_structured_suggestions_for_llm.append(sugg)
                        else:
                            self._logger_instance.warning("PRAA 'structured_optimization_suggestions_json' was not a list.")
                            
                        self._logger_instance.info(f"Successfully parsed PRAA optimization report. Found {len(praa_structured_suggestions)} structured suggestions.")
                    except json.JSONDecodeError as e:
                        self._logger_instance.error(f"Failed to parse JSON from PRAA optimization report {task_input.praa_optimization_report_doc_id}: {e}")
                        praa_optimization_report_content_md = retrieved_praa_opt_report.artifact_content 
                else:
                    self._logger_instance.warning(f"Failed to retrieve PRAA optimization report content: {retrieved_praa_opt_report.message or retrieved_praa_opt_report.error_message}")
            except Exception as e:
                self._logger_instance.error(f"Exception retrieving PRAA optimization report: {e}", exc_info=True)

        # --- Retrieve and Parse Blueprint Reviewer Report ---
        if task_input.blueprint_reviewer_report_doc_id and self._project_chroma_manager:
            try:
                self._logger_instance.info(f"Retrieving Blueprint Reviewer report: {task_input.blueprint_reviewer_report_doc_id}")
                retrieved_blueprint_review_report: RetrieveArtifactOutput = await self._project_chroma_manager.retrieve_artifact(
                    doc_id=task_input.blueprint_reviewer_report_doc_id, # MODIFIED: doc_id directly
                    collection_name=REVIEW_REPORTS_COLLECTION 
                )
                if retrieved_blueprint_review_report.status == "SUCCESS" and retrieved_blueprint_review_report.artifact_content:
                    try:
                        blueprint_reviewer_output_json = json.loads(retrieved_blueprint_review_report.artifact_content)
                        blueprint_reviewer_report_content_md = blueprint_reviewer_output_json.get("blueprint_optimization_report_md")
                        
                        suggestions_from_br = blueprint_reviewer_output_json.get("structured_optimization_suggestions_json", [])
                        if isinstance(suggestions_from_br, list):
                            blueprint_reviewer_structured_suggestions = suggestions_from_br
                            for sugg in blueprint_reviewer_structured_suggestions: # MODIFIED: Enrich with source_report
                                sugg['source_report'] = 'BlueprintReviewer'
                                all_structured_suggestions_for_llm.append(sugg)
                        else:
                             self._logger_instance.warning("Blueprint Reviewer 'structured_optimization_suggestions_json' was not a list.")

                        self._logger_instance.info(f"Successfully parsed Blueprint Reviewer report. Found {len(blueprint_reviewer_structured_suggestions)} structured suggestions.")
                    except json.JSONDecodeError as e:
                        self._logger_instance.error(f"Failed to parse JSON from Blueprint Reviewer report {task_input.blueprint_reviewer_report_doc_id}: {e}")
                        blueprint_reviewer_report_content_md = retrieved_blueprint_review_report.artifact_content
                else:
                    self._logger_instance.warning(f"Failed to retrieve Blueprint Reviewer report content: {retrieved_blueprint_review_report.message or retrieved_blueprint_review_report.error_message}")
            except Exception as e:
                self._logger_instance.error(f"Exception retrieving Blueprint Reviewer report: {e}", exc_info=True)

        # --- Retrieve PRAA Risk Report (assuming it's still direct Markdown or handled by a different logic if JSON) ---
        if task_input.praa_risk_report_doc_id and self._project_chroma_manager:
            try:
                self._logger_instance.info(f"Retrieving PRAA risk report: {task_input.praa_risk_report_doc_id}")
                retrieved_praa_risk_report: RetrieveArtifactOutput = await self._project_chroma_manager.retrieve_artifact(
                    doc_id=task_input.praa_risk_report_doc_id,
                    collection_name=RISK_ASSESSMENT_REPORTS_COLLECTION
                )
                if retrieved_praa_risk_report.status == "SUCCESS" and retrieved_praa_risk_report.artifact_content:
                    # If PRAA Risk Report also became JSON, similar parsing to above would be needed.
                    # For now, assuming it's direct Markdown as per current EXECUTION_PLAN focus.
                    praa_risk_report_content_md = retrieved_praa_risk_report.artifact_content
                    self._logger_instance.info(f"Successfully retrieved PRAA risk report content.")
                else:
                    self._logger_instance.warning(f"Failed to retrieve PRAA risk report content: {retrieved_praa_risk_report.message or retrieved_praa_risk_report.error_message}")
            except Exception as e:
                self._logger_instance.error(f"Exception retrieving PRAA risk report: {e}", exc_info=True)
        
        # --- Retrieve RTA Report (assuming it's still direct Markdown or handled by a different logic if JSON) ---
        if task_input.rta_report_doc_id and self._project_chroma_manager:
            try:
                self._logger_instance.info(f"Retrieving RTA report: {task_input.rta_report_doc_id}")
                retrieved_rta_report: RetrieveArtifactOutput = await self._project_chroma_manager.retrieve_artifact(
                    doc_id=task_input.rta_report_doc_id,
                    collection_name=TRACEABILITY_REPORTS_COLLECTION
                )
                if retrieved_rta_report.status == "SUCCESS" and retrieved_rta_report.artifact_content:
                    # If RTA Report also became JSON, similar parsing would be needed.
                    # For now, assuming it's direct Markdown.
                    rta_report_content_md = retrieved_rta_report.artifact_content
                    self._logger_instance.info(f"Successfully retrieved RTA report content.")
                else:
                    self._logger_instance.warning(f"Failed to retrieve RTA report content: {retrieved_rta_report.message or retrieved_rta_report.error_message}")
            except Exception as e:
                self._logger_instance.error(f"Exception retrieving RTA report: {e}", exc_info=True)


        # ARCA's main decision logic starts here. 
        # It will use praa_risk_report_content_md, praa_optimization_report_content_md, 
        # blueprint_reviewer_report_content_md, rta_report_content_md, 
        # and the new 'all_structured_suggestions_for_llm' list.

        # Log collected suggestions
        if all_structured_suggestions_for_llm: # MODIFIED: Use new list name
            self._logger_instance.info(f"ARCA collected a total of {len(all_structured_suggestions_for_llm)} structured optimization suggestions for LLM evaluation.")
        else:
            self._logger_instance.info("ARCA found no structured optimization suggestions from PRAA or Blueprint Reviewer reports to send for LLM evaluation.")

        # Based on task_input.artifact_type, ARCA decides the next steps.
        # This is a simplified placeholder for the complex decision tree.

        # --- Handle GenerateProjectDocumentation Task ---
        
        if task_input.artifact_type == ARCAReviewArtifactType.GENERATE_PROJECT_DOCUMENTATION: # Corrected typo: GenerateProjectDocumentation -> GENERATE_PROJECT_DOCUMENTATION
            self._logger_instance.info(f"ARCA: Received GenerateProjectDocumentation task for project {task_input.project_id}, cycle {task_input.cycle_id}.")
            if not (task_input.final_loprd_doc_id_for_docs and \
                    task_input.final_blueprint_doc_id_for_docs and \
                    task_input.final_plan_doc_id_for_docs and \
                    task_input.final_code_root_path_for_docs):
                decision = "ERROR"
                reasoning = "Cannot proceed with documentation generation: Missing critical input document IDs (LOPRD, Blueprint, Plan) or code root path in ARCAReviewInput."
                self._logger_instance.error(reasoning)
            else:
                doc_agent_input = ProjectDocumentationAgentInput(
                    project_id=task_input.project_id,
                    refined_user_goal_doc_id=task_input.final_loprd_doc_id_for_docs, # Assuming LOPRD acts as refined goal for docs
                    project_blueprint_doc_id=task_input.final_blueprint_doc_id_for_docs,
                    master_execution_plan_doc_id=task_input.final_plan_doc_id_for_docs,
                    generated_code_root_path=task_input.final_code_root_path_for_docs,
                    test_summary_doc_id=task_input.final_test_summary_doc_id_for_docs, # Optional
                    cycle_id=task_input.cycle_id # Pass the cycle_id
                )
                decision = "PROCEED_TO_DOCUMENTATION"
                reasoning = "Proceeding to generate project documentation based on provided final artifacts."
                next_agent_for_refinement = ProjectDocumentationAgent_v1.AGENT_ID
                next_agent_refinement_input = doc_agent_input
                arca_confidence = ConfidenceScore(value=0.95, level="HIGH", method="ProgrammaticDecision", reasoning="Documentation generation triggered by explicit request.")
                self._logger_instance.info(f"ARCA: Prepared input for ProjectDocumentationAgent_v1. Cycle ID: {task_input.cycle_id}")
            
            # Early exit for documentation task
            # No further review or state updates related to a *reviewed artifact* are needed here.
            # StateManager updates for cycle completion might still occur if this is the end of a cycle.
            
            # Log ARCA Decision Made
            if self._project_chroma_manager:
                await self._log_event_to_pcma(
                    pcma_agent=self._project_chroma_manager, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                    arca_task_id=task_input.task_id, event_type="ARCA_DECISION_MADE",
                    event_details={
                        "reviewed_artifact_doc_id": "N/A - Documentation Generation Task",
                        "reviewed_artifact_type": task_input.artifact_type.value,
                        "decision": decision,
                        "reasoning_summary": reasoning[:200],
                        "next_agent_id": next_agent_for_refinement,
                    },
                    severity="INFO"
                )

            return ARCAOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                reviewed_artifact_doc_id=task_input.artifact_doc_id or "N/A", # No specific artifact was "reviewed"
                reviewed_artifact_type=task_input.artifact_type,
                decision=decision,
                reasoning=reasoning, # Changed from decision_reasoning
                confidence_in_decision=arca_confidence,
                next_agent_id_for_refinement=next_agent_for_refinement,
                next_agent_input=next_agent_refinement_input.model_dump() if next_agent_refinement_input and hasattr(next_agent_refinement_input, 'model_dump') else next_agent_refinement_input,
                final_artifact_doc_id=None # No artifact was accepted here
            )
        # --- End Handle GenerateProjectDocumentation Task ---

        # Existing logic for other artifact types (LOPRD, Blueprint, etc.)
        # ... (rest of the invoke_async method) ...

        # --- Retrieve supporting documents (PRAA, RTA reports, Blueprint Review) ---
        praa_risk_report_content: Optional[str] = None
        praa_optimization_report_content: Optional[str] = None
        rta_report_content: Optional[str] = None
        blueprint_review_content: Optional[str] = None
        
        # NEW: Retrieve the main artifact content for summary
        main_artifact_content: Optional[str] = None
        main_artifact_collection_name: Optional[str] = None

        if task_input.artifact_doc_id:
            # Determine collection based on artifact_type (this is a simplified mapping)
            # More robust mapping might be needed if artifact types to collection names are complex
            if task_input.artifact_type == ARCAReviewArtifactType.LOPRD:
                # Assuming LOPRDs are in a collection like 'loprd_collection' - this needs to be accurate
                # For now, let's assume a generic 'project_artifacts_collection' or specific ones
                # This part requires knowing where LOPRDs, Blueprints, etc. are stored by ProductAnalyst/Architect.
                # Let's placeholder with a direct reference if available in full_context or a specific collection.
                # For now, we will rely on the doc_id being universally unique or collection being passed.
                # For this example, let's assume a generic collection if not specific.
                # This needs to be refined based on where these artifacts are ACTUALLY stored by their generator agents.
                # For MVP, we might need to pass the collection name or have a mapping.
                # For now, this part of fetching main_artifact_content is illustrative and needs correct collections.
                # Let's assume task_input.artifact_doc_id IS the content for simplicity or that a
                # generic retrieval method exists that doesn't strictly need collection for main artifacts.
                # THIS IS A CRITICAL POINT: How does ARCA get the content of the artifact it's reviewing?
                # For now, we'll assume pcma_instance can retrieve it with just the doc_id from a known main collection
                # or the doc_id itself might sometimes be a content placeholder.
                # Given pcma_instance.retrieve_artifact needs a base_collection_name, we must define it.
                # This will be a placeholder and likely incorrect until actual storage collections are known.
                
                # A more robust way: the agent that *created* the artifact_doc_id should also provide its collection.
                # Or ARCA needs a mapping.
                # For now, let's assume a hypothetical default or look it up.
                # This lookup logic is simplified:
                artifact_collection_map = {
                    ARCAReviewArtifactType.LOPRD: "loprds_collection", # Example, needs to be correct
                    ARCAReviewArtifactType.PROJECT_BLUEPRINT: "project_blueprints_collection", # Corrected
                    ARCAReviewArtifactType.MASTER_EXECUTION_PLAN: "master_execution_plans_collection", # Corrected
                    ARCAReviewArtifactType.CODE_MODULE: GENERATED_CODE_ARTIFACTS_COLLECTION,
                    # Add other types if they need to be fetched for summarization
                }
                main_artifact_collection_name = artifact_collection_map.get(task_input.artifact_type)

                if self._project_chroma_manager and main_artifact_collection_name:
                    try:
                        main_artifact_retrieved: RetrieveArtifactOutput = await self._project_chroma_manager.retrieve_artifact(
                            base_collection_name=main_artifact_collection_name,
                            document_id=task_input.artifact_doc_id
                        )
                        if main_artifact_retrieved and main_artifact_retrieved.status == "SUCCESS" and main_artifact_retrieved.content:
                            main_artifact_content = str(main_artifact_retrieved.content)
                            self._logger_instance.info(f"Successfully retrieved main artifact ({task_input.artifact_type.value}) content for doc_id: {task_input.artifact_doc_id}")
                        else:
                            self._logger_instance.warning(f"Failed to retrieve main artifact ({task_input.artifact_type.value}) content for doc_id {task_input.artifact_doc_id}. Status: {main_artifact_retrieved.status if main_artifact_retrieved else 'N/A'}")
                    except Exception as e_main_artifact:
                        self._logger_instance.error(f"Exception retrieving main artifact ({task_input.artifact_type.value}) {task_input.artifact_doc_id}: {e_main_artifact}", exc_info=True)
                elif not main_artifact_collection_name:
                    self._logger_instance.warning(f"No collection mapping found for artifact type {task_input.artifact_type.value} to retrieve its content for summarization.")

        # --- Ensure supporting documents (PRAA, RTA reports, Blueprint Review) are still fetched ---
        # This block was unintentionally removed by a previous edit, restoring it.
        if not self._project_chroma_manager:
            self._logger_instance.error("ARCA: project_chroma_manager is not available for retrieving supporting documents. Cannot proceed effectively.")
        else:
            try:
                if task_input.praa_risk_report_doc_id:
                    doc_output: RetrieveArtifactOutput = await self._project_chroma_manager.retrieve_artifact(
                        doc_id=task_input.praa_risk_report_doc_id,
                        collection_name=RISK_ASSESSMENT_REPORTS_COLLECTION
                    )
                    if doc_output and doc_output.status == "SUCCESS" and doc_output.artifact_content:
                        praa_risk_report_content_md = str(doc_output.artifact_content)
                    else:
                        self._logger_instance.warning(f"PRAA risk report {task_input.praa_risk_report_doc_id} not found or content empty. Status: {doc_output.status if doc_output else 'N/A'}")

                # praa_optimization_report_content_md is already fetched and parsed into structured suggestions and MD above
                # rta_report_content_md is already fetched and parsed into structured suggestions and MD above
                # blueprint_reviewer_report_content_md is already fetched and parsed into structured suggestions and MD above

                if task_input.rta_report_doc_id: # Ensure RTA MD content is still fetched if needed by heuristics
                    doc_output: RetrieveArtifactOutput = await self._project_chroma_manager.retrieve_artifact(
                        doc_id=task_input.rta_report_doc_id,
                        collection_name=TRACEABILITY_REPORTS_COLLECTION
                    )
                    if doc_output and doc_output.status == "SUCCESS" and doc_output.artifact_content:
                        rta_report_content_md = str(doc_output.artifact_content)
                    else:
                        self._logger_instance.warning(f"RTA report {task_input.rta_report_doc_id} not found or content empty. Status: {doc_output.status if doc_output else 'N/A'}")
                
            except Exception as e_reports:
                self._logger_instance.error(f"Exception while retrieving one or more supporting reports (risk/RTA): {e_reports}", exc_info=True)
                pass        

        # --- NEW: Evaluate Optimization Suggestions using LLM (Moved after main_artifact_content retrieval) ---
        evaluated_optimizations_from_llm: Optional[List[Dict[str, Any]]] = None
        llm_optimization_evaluation_summary: Optional[str] = None

        if self._llm_provider and self._prompt_manager and all_structured_suggestions_for_llm: 
            if not main_artifact_content: 
                self._logger_instance.warning("Main artifact content still not available after attempting fetch, skipping LLM optimization evaluation.")
            else:
                try:
                    all_suggestions_json_string = json.dumps(all_structured_suggestions_for_llm)
                    optimization_eval_inputs = {
                        "artifact_type": task_input.artifact_type.value,
                        "artifact_content_summary": (main_artifact_content[:10000] if main_artifact_content else "Main artifact content not available."), 
                        "all_structured_suggestions_json_string": all_suggestions_json_string, 
                        "current_project_goal_summary": full_context.get("project_goal_summary", "No project goal summary provided.") if full_context else "No project goal summary provided."
                    }

                    self._logger_instance.info(f"Invoking LLM for optimization evaluation with prompt: {ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME}")
                    
                    if self._project_chroma_manager:
                        await self._log_event_to_pcma(
                            pcma_agent=self._project_chroma_manager, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                            arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_START",
                            event_details={
                                "invoked_agent_id": "LLM_OptimizationEvaluator",
                                "prompt_name": ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME,
                                "input_summary": {
                                    "artifact_type": optimization_eval_inputs["artifact_type"],
                                    "num_suggestions_to_evaluate": len(all_structured_suggestions_for_llm)
                                }
                            },
                            severity="INFO"
                        )

                    llm_response_raw = await self._llm_provider.instruct_direct_async(
                        prompt_name=ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME,
                        prompt_manager=self._prompt_manager,
                        input_vars=optimization_eval_inputs,
                        # model_name, temperature, etc., could be configured via prompt_definition if needed
                    )

                    if llm_response_raw:
                        self._logger_instance.debug(f"LLM Optimization Evaluation Raw Response: {llm_response_raw}")
                        try:
                            llm_response_json = json.loads(llm_response_raw)
                            evaluated_optimizations_from_llm = llm_response_json.get("evaluated_optimizations")
                            llm_optimization_evaluation_summary = llm_response_json.get("overall_summary_of_actions")
                            self._logger_instance.info(f"Successfully parsed LLM optimization evaluation. Summary: {llm_optimization_evaluation_summary}")
                            if self._project_chroma_manager: 
                                await self._log_event_to_pcma(
                                    pcma_agent=self._project_chroma_manager, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                    arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                                    event_details={
                                        "invoked_agent_id": "LLM_OptimizationEvaluator",
                                        "status": "SUCCESS_PARSED",
                                        "summary": llm_optimization_evaluation_summary,
                                        "num_evaluated_optimizations": len(evaluated_optimizations_from_llm) if evaluated_optimizations_from_llm else 0
                                    },
                                    severity="INFO"
                                )
                        except json.JSONDecodeError as e_json:
                            self._logger_instance.error(f"Failed to parse JSON from LLM optimization evaluation response: {e_json}. Response: {llm_response_raw}", exc_info=True)
                            if self._project_chroma_manager: 
                                await self._log_event_to_pcma(
                                    pcma_agent=self._project_chroma_manager, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                    arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                                    event_details={
                                        "invoked_agent_id": "LLM_OptimizationEvaluator",
                                        "status": "FAILURE_JSON_PARSE_ERROR",
                                        "error": str(e_json)
                                    },
                                    severity="ERROR"
                                )
                    else:
                        self._logger_instance.warning("LLM optimization evaluation returned an empty response.")
                        if self._project_chroma_manager: 
                             await self._log_event_to_pcma(
                                pcma_agent=self._project_chroma_manager, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                                event_details={
                                    "invoked_agent_id": "LLM_OptimizationEvaluator",
                                    "status": "FAILURE_EMPTY_RESPONSE"
                                },
                                severity="WARNING"
                            )

                except Exception as e_llm_opt_eval:
                    self._logger_instance.error(f"Error during LLM optimization evaluation: {e_llm_opt_eval}", exc_info=True)
                    if self._project_chroma_manager: 
                        await self._log_event_to_pcma(
                            pcma_agent=self._project_chroma_manager, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                            arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                            event_details={
                                "invoked_agent_id": "LLM_OptimizationEvaluator",
                                "status": "FAILURE_EXCEPTION",
                                "error": str(e_llm_opt_eval)
                            },
                            severity="ERROR"
                        )
        elif not all_structured_suggestions_for_llm: 
            self._logger_instance.info("No structured optimization suggestions were available, skipping LLM optimization evaluation.")
        elif not self._llm_provider or not self._prompt_manager:
            self._logger_instance.warning("LLMProvider or PromptManager not available, skipping LLM optimization evaluation.")

        # --- Decision Logic --- #
        # Initialize decision variables
        decision: Literal["ACCEPT_ARTIFACT", "REFINEMENT_REQUIRED", "ESCALATE_TO_USER", "TEST_FAILURE_HANDOFF", "PROCEED_TO_DOCUMENTATION", "PLAN_MODIFIED_NEW_TASKS_ADDED", "ERROR"] = "ACCEPT_ARTIFACT" 
        reasoning = f"Initial assessment for {task_input.artifact_type.value} ID: {task_input.artifact_doc_id or task_input.code_module_file_path}."
        next_agent_for_refinement: Optional[str] = None
        next_agent_refinement_input: Optional[Union[pa_module.ProductAnalystAgentInput, ArchitectAgentInput, MasterPlannerInput, ProjectDocumentationAgentInput, DebuggingTaskInput, Dict[str, Any]]] = None 
        final_doc_id_on_accept: Optional[str] = None
        debugging_input_for_output: Optional[DebuggingTaskInput] = None 
        arca_error_message: Optional[str] = None

        # Incorporate LLM evaluation summary into overall reasoning
        if llm_optimization_evaluation_summary:
            reasoning += f" LLM Optimization Evaluation Summary: {llm_optimization_evaluation_summary}."
        
        llm_recommended_incorporations: List[Dict[str, Any]] = []
        if evaluated_optimizations_from_llm:
            for opt_eval in evaluated_optimizations_from_llm:
                if opt_eval.get("recommendation") == "INCORPORATE" and opt_eval.get("incorporation_instructions_for_next_agent"):
                    llm_recommended_incorporations.append(opt_eval)
                elif isinstance(opt_eval, dict) and opt_eval.get("recommendation") == "NEW_TASK_FOR_PLAN":
                    if opt_eval.get("new_task_details_for_plan"):
                        new_tasks_to_add_to_plan.append(opt_eval["new_task_details_for_plan"])
                        reasoning += f"\n- LLM recommended NEW_TASK_FOR_PLAN for suggestion: '{opt_eval.get('suggestion_id', 'N/A')}'. Details: {opt_eval['new_task_details_for_plan']}"
                    else:
                        reasoning += f"\n- LLM recommended NEW_TASK_FOR_PLAN for suggestion: '{opt_eval.get('suggestion_id', 'N/A')}' but new_task_details_for_plan was missing."
                        self._logger_instance.warning(f"ARCA: LLM recommended NEW_TASK_FOR_PLAN but new_task_details_for_plan was missing for suggestion '{opt_eval.get('suggestion_id', 'N/A')}'.")

        if llm_recommended_incorporations:
            decision = "REFINEMENT_REQUIRED"
            # Update reasoning to clearly state LLM forced the refinement
            reasoning += f" Decision automatically set to REFINEMENT_REQUIRED due to {len(llm_recommended_incorporations)} LLM-evaluated optimization(s) recommended for incorporation."
            self._logger_instance.info(f"ARCA: Decision forced to REFINEMENT_REQUIRED by LLM based on {len(llm_recommended_incorporations)} recommended incorporations.")

        # Calculate combined metric (heuristic)
        generator_confidence_val = task_input.generator_agent_confidence.score if task_input.generator_agent_confidence else self.MIN_GENERATOR_CONFIDENCE 
        praa_confidence_val_num = 0.0 
        if praa_risk_report_content_md: 
            praa_confidence_val_num = 0.7 
        rta_confidence_val_num = 0.0 
        if rta_report_content_md: 
            rta_confidence_val_num = 0.7

        combined_metric = 0.0
        if task_input.artifact_type == ARCAReviewArtifactType.LOPRD:
            combined_metric = (generator_confidence_val * 0.6) + (praa_confidence_val_num * 0.4)
        elif task_input.artifact_type == ARCAReviewArtifactType.PROJECT_BLUEPRINT: # Corrected
            # For blueprint, RTA becomes more important, PRAA (risk) also.
            combined_metric = (generator_confidence_val * 0.5) + (praa_confidence_val_num * 0.25) + (rta_confidence_val_num * 0.25)
        elif task_input.artifact_type == ARCAReviewArtifactType.MASTER_EXECUTION_PLAN: # Corrected
            # For plan, RTA (to blueprint) is critical. Generator confidence also high.
            combined_metric = (generator_confidence_val * 0.4) + (praa_confidence_val_num * 0.2) + (rta_confidence_val_num * 0.4)
        # No combined_metric for CodeModule or CodeModule_TestFailure if it needs specific handling,
        # or can add a general case if appropriate.
        elif task_input.artifact_type == ARCAReviewArtifactType.CODE_MODULE: # Corrected (was already correct but good to be explicit)
            # For code, RTA (to plan/requirements) is important.
            combined_metric = (generator_confidence_val * 0.5) + (praa_confidence_val_num * 0.2) + (rta_confidence_val_num * 0.3)
        else: # Default for other types or if specific metrics aren't defined
            combined_metric = generator_confidence_val # Fallback to generator's confidence

        # Store heuristic reasoning separately for clarity, might append later if relevant
        heuristic_reasoning_part = f" Heuristic confidence scores: Gen_Conf={generator_confidence_val:.2f}, PRAA_Risk_Conf(h)={praa_confidence_val_num:.2f}, RTA_Conf(h)={rta_confidence_val_num:.2f}. Combined_Heuristic_Metric={combined_metric:.2f}. Default_Accept_Threshold={self.DEFAULT_ACCEPTANCE_THRESHOLD:.2f}."

        # Decision logic based on artifact type
        if task_input.artifact_type == ARCAReviewArtifactType.CODE_MODULE_TEST_FAILURE: # Corrected
            self._logger_instance.info(f"ARCA: Handling CodeModule_TestFailure for {task_input.code_module_file_path}.")
            # This block sets its own 'decision' and 'reasoning'. 
            # The initial 'decision' and 'reasoning' (including LLM eval for other artifacts) are effectively bypassed for this path.
            
            faulty_code_path_for_input = task_input.code_module_file_path
            faulty_code_content_for_input: Optional[str] = None
            failed_test_reports = task_input.failed_test_report_details

            pcma_for_code_retrieval = self._project_chroma_manager 
            if not pcma_for_code_retrieval:
                self._logger_instance.error(f"ARCA: PCMA instance is not available for CodeModule_TestFailure handling of {task_input.code_module_file_path or task_input.artifact_doc_id}. Cannot proceed.")
                decision = "ERROR" # Override
                reasoning = "PCMA instance not available for code retrieval during debugging." # Override
                arca_error_message = reasoning
            else:
                if not faulty_code_content_for_input and task_input.artifact_doc_id:
                    try:
                        code_doc_output: RetrieveArtifactOutput = await pcma_for_code_retrieval.retrieve_artifact(
                            doc_id=task_input.artifact_doc_id, 
                            collection_name=GENERATED_CODE_ARTIFACTS_COLLECTION 
                        )
                        if code_doc_output and code_doc_output.status == "SUCCESS" and code_doc_output.artifact_content: 
                            faulty_code_content_for_input = str(code_doc_output.artifact_content) 
                            self._logger_instance.info(f"Retrieved faulty code content from doc_id: {task_input.artifact_doc_id}")
                            if not faulty_code_path_for_input:
                                self._logger_instance.warning(f"Code content retrieved for {task_input.artifact_doc_id} but no file path provided to ARCA. Debugger might be limited.")
                        else:
                            self._logger_instance.error(f"Failed to retrieve faulty code content for doc_id {task_input.artifact_doc_id}. Status: {code_doc_output.status if code_doc_output else 'N/A'}")
                            decision = "ERROR" # Override
                            reasoning = f"Failed to retrieve source code for debugging from doc_id {task_input.artifact_doc_id}. Status: {code_doc_output.status if code_doc_output else 'N/A'}" # Override
                            arca_error_message = reasoning
                    except Exception as e_fetch_code:
                        self._logger_instance.error(f"Exception fetching faulty code {task_input.artifact_doc_id}: {e_fetch_code}", exc_info=True)
                        decision = "ERROR" # Override
                        reasoning = f"Exception fetching source code for debugging from doc_id {task_input.artifact_doc_id}: {e_fetch_code}" # Override
                        arca_error_message = reasoning
                
                if decision != "ERROR" and not faulty_code_content_for_input and not faulty_code_path_for_input:
                    err_msg = "Cannot proceed with debugging: Neither code content nor a valid file path for the faulty code is available."
                    self._logger_instance.error(err_msg)
                    decision = "ERROR"; reasoning = err_msg; arca_error_message = err_msg # Override

                if decision != "ERROR": # If no errors so far in fetching code for debugging
                    relevant_loprd_ids = full_context.get("relevant_loprd_ids_for_debug", []) if full_context else []
                    relevant_blueprint_ids = full_context.get("relevant_blueprint_ids_for_debug", []) if full_context else []
                    previous_debugging_attempts_from_context = full_context.get("previous_debugging_attempts", []) if full_context else []

                    debugging_input_for_output = DebuggingTaskInput( 
                        project_id=task_input.project_id,
                        faulty_code_path=faulty_code_path_for_input,
                        faulty_code_content=faulty_code_content_for_input,
                        failed_test_reports=failed_test_reports or [], 
                        relevant_loprd_requirements_ids=relevant_loprd_ids,
                        relevant_blueprint_section_ids=relevant_blueprint_ids,
                        previous_debugging_attempts=previous_debugging_attempts_from_context,
                        max_iterations_for_this_call=full_context.get("max_debug_iterations_per_call", 3) if full_context else 3,
                        cycle_id=task_input.cycle_id
                    )
                    current_attempt_count = len(debugging_input_for_output.previous_debugging_attempts)
                    max_attempts_for_module = (full_context.get("max_total_debugging_attempts_for_module", self.MAX_DEBUGGING_ATTEMPTS_PER_MODULE)
                                              if full_context else self.MAX_DEBUGGING_ATTEMPTS_PER_MODULE)

                    if current_attempt_count >= max_attempts_for_module:
                        decision = "ESCALATE_TO_USER" # Override
                        reasoning = f"Max debugging attempts ({max_attempts_for_module}) reached for module {faulty_code_path_for_input or task_input.artifact_doc_id}. Escalating. Last failed tests: {failed_test_reports}" # Override
                        self._logger_instance.warning(reasoning)
                    else:
                        decision = "TEST_FAILURE_HANDOFF" # Override
                        reasoning = f"Handing off to CodeDebuggingAgent for {faulty_code_path_for_input or task_input.artifact_doc_id}. Attempt {current_attempt_count + 1}." # Override
                        self._logger_instance.info(reasoning)
                        # next_agent_for_refinement and next_agent_refinement_input are not set here;
                        # The orchestrator handles TEST_FAILURE_HANDOFF by invoking the debugger with debugging_input_for_output.
            
        elif decision != "ERROR" and decision != "ESCALATE_TO_USER": 
            # ... (logic for non-CodeModule_TestFailure artifacts as applied in Chunk 3.3, this should be mostly correct now)
            # ... This block relies on 'decision' potentially being REFINEMENT_REQUIRED due to LLM eval.
            if decision == "REFINEMENT_REQUIRED": 
                self._logger_instance.info(f"Decision is REFINEMENT_REQUIRED (possibly due to LLM). Appending heuristic info: {heuristic_reasoning_part}")
                if heuristic_reasoning_part not in reasoning:
                    reasoning += heuristic_reasoning_part
            else: # LLM did not force refinement, so evaluate heuristic scores
                if combined_metric >= self.DEFAULT_ACCEPTANCE_THRESHOLD and \
                   generator_confidence_val >= self.MIN_GENERATOR_CONFIDENCE and \
                   praa_confidence_val_num >= self.MIN_PRAA_CONFIDENCE and \
                   (task_input.artifact_type == ARCAReviewArtifactType.LOPRD or rta_confidence_val_num >= self.MIN_RTA_CONFIDENCE):
                    decision = "ACCEPT_ARTIFACT"
                    if heuristic_reasoning_part not in reasoning: reasoning += heuristic_reasoning_part 
                    reasoning += f" Artifact ACCEPTED based on heuristic confidence thresholds meeting/exceeding {self.DEFAULT_ACCEPTANCE_THRESHOLD:.2f}."
                    final_doc_id_on_accept = task_input.artifact_doc_id
                    self._logger_instance.info(f"ARCA Decision: ACCEPT_ARTIFACT for {task_input.artifact_doc_id or task_input.code_module_file_path} ({task_input.artifact_type.value}). Final Reasoning: {reasoning}")
                else: 
                    decision = "REFINEMENT_REQUIRED"
                    if heuristic_reasoning_part not in reasoning: reasoning += heuristic_reasoning_part
                    reasoning += f" Refinement indicated by heuristic confidence scores failing to meet thresholds (Combined: {combined_metric:.2f} vs {self.DEFAULT_ACCEPTANCE_THRESHOLD:.2f}, or individual scores low)."
                    self._logger_instance.info(f"ARCA Decision: REFINEMENT_REQUIRED for {task_input.artifact_doc_id or task_input.code_module_file_path} ({task_input.artifact_type.value}). Final Reasoning: {reasoning}")
            
            if decision == "REFINEMENT_REQUIRED":
                # ... (refinement_instructions_for_agent and next_agent selection logic as applied in Chunk 3.2, assumed correct)
                refinement_instructions_list = []
                if llm_recommended_incorporations:
                    refinement_instructions_list.append("Based on LLM evaluation, please incorporate the following optimizations:")
                    for opt_eval in llm_recommended_incorporations:
                        refinement_instructions_list.append(
                            f"  - Opt ID {opt_eval.get('optimization_id', 'N/A')} (Source: {opt_eval.get('source_report', 'N/A')}): {opt_eval['incorporation_instructions_for_next_agent']}"
                        )
                
                if not llm_recommended_incorporations or (combined_metric < self.DEFAULT_ACCEPTANCE_THRESHOLD):
                    general_instruction = f"General Refinement Note: Review overall quality for {task_input.artifact_type.value} (ID: {task_input.artifact_doc_id or task_input.code_module_file_path}). " \
                                          f"Heuristic scores: Gen: {generator_confidence_val:.2f}, PRAA_h: {praa_confidence_val_num:.2f}, RTA_h: {rta_confidence_val_num:.2f}. Combined_h: {combined_metric:.2f}."
                    if not llm_recommended_incorporations:
                        refinement_instructions_list.append(general_instruction)
                    else: 
                        refinement_instructions_list.append(f"Additionally, {general_instruction}")
                
                if not refinement_instructions_list:
                    refinement_instructions_list.append(
                        f"General Refinement requested for {task_input.artifact_type.value} (ID: {task_input.artifact_doc_id or task_input.code_module_file_path}) due to ARCA internal logic. Please review for quality."
                    )

                refinement_instructions_for_agent = "\n".join(refinement_instructions_list)
                self._logger_instance.debug(f"Final refinement instructions for next agent: {refinement_instructions_for_agent}")

                initial_user_goal_for_paa = "User goal not available in ARCA context for refinement input." 
                if full_context and full_context.get("intermediate_outputs", {}).get("initial_goal_setup", {}).get("initial_user_goal"):
                    initial_user_goal_for_paa = full_context["intermediate_outputs"]["initial_goal_setup"]["initial_user_goal"]
                elif full_context and full_context.get("initial_user_goal"):
                     initial_user_goal_for_paa = full_context["initial_user_goal"]
                
                if task_input.artifact_type == ARCAReviewArtifactType.LOPRD:
                    next_agent_for_refinement = pa_module.ProductAnalystAgent_v1.AGENT_ID
                    next_agent_refinement_input = pa_module.ProductAnalystAgentInput(
                        project_id=task_input.project_id,
                        existing_loprd_doc_id=task_input.artifact_doc_id, 
                        refinement_directives=refinement_instructions_for_agent, 
                        initial_user_goal=initial_user_goal_for_paa, 
                        cycle_id=task_input.cycle_id
                    )
                elif task_input.artifact_type == ARCAReviewArtifactType.PROJECT_BLUEPRINT: # Corrected
                    next_agent_for_refinement = ArchitectAgentInput.AGENT_ID 
                    loprd_id_for_architect = "unknown_loprd_id_for_blueprint_refinement" 
                    if full_context and full_context.get("latest_accepted_loprd_doc_id"): 
                        loprd_id_for_architect = full_context["latest_accepted_loprd_doc_id"]
                    next_agent_refinement_input = ArchitectAgentInput(
                        project_id=task_input.project_id,
                        loprd_doc_id=loprd_id_for_architect, 
                        existing_blueprint_doc_id=task_input.artifact_doc_id, 
                        refinement_instructions=refinement_instructions_for_agent,
                        cycle_id=task_input.cycle_id
                    )
                elif task_input.artifact_type == ARCAReviewArtifactType.MASTER_EXECUTION_PLAN: # Corrected
                    next_agent_for_refinement = "SystemMasterPlannerAgent_v1" 
                    blueprint_id_for_planner = "unknown_blueprint_id_for_plan_refinement"
                    if full_context and full_context.get("latest_accepted_blueprint_doc_id"):
                         blueprint_id_for_planner = full_context["latest_accepted_blueprint_doc_id"]
                    next_agent_refinement_input = MasterPlannerInput( 
                        project_id=task_input.project_id,
                        blueprint_doc_id=blueprint_id_for_planner,
                        refinement_instructions=refinement_instructions_for_agent,
                        existing_plan_doc_id=task_input.artifact_doc_id 
                    )
                elif task_input.artifact_type == ARCAReviewArtifactType.CODE_MODULE:
                    next_agent_for_refinement = "CoreCodeGeneratorAgent_v1" 
                    code_spec_doc_id_for_refinement = "unknown_spec_for_code_refinement" 
                    if full_context and full_context.get("current_task_specification_doc_id"): 
                        code_spec_doc_id_for_refinement = full_context["current_task_specification_doc_id"]
                    next_agent_refinement_input = {
                        "project_id": task_input.project_id,
                        "code_specification_doc_id": code_spec_doc_id_for_refinement, 
                        "existing_code_doc_id": task_input.artifact_doc_id, 
                        "refinement_instructions": refinement_instructions_for_agent,
                        "cycle_id": task_input.cycle_id
                    }
                    self._logger_instance.warning(f"ARCA: CodeModule refinement input is a generic dict. Ensure {next_agent_for_refinement} can handle it.")
                else: 
                    current_reasoning_for_no_agent = " However, no specific refinement agent path is configured for this artifact type."
                    if current_reasoning_for_no_agent not in reasoning:
                        reasoning += current_reasoning_for_no_agent
                    next_agent_for_refinement = None 
                    next_agent_refinement_input = None
                    self._logger_instance.warning(f"ARCA: Refinement decided for {task_input.artifact_type.value}, but no agent path defined. Final Reasoning: {reasoning}")

        # AGENT DIAGNOSTIC: Right before final escalation check (L988)
        self._logger_instance.warning(f"AGENT DIAGNOSTIC (L988_PRE_ESCALATE_CHECK): decision == {decision}")
        self._logger_instance.warning(f"AGENT DIAGNOSTIC (L988_PRE_ESCALATE_CHECK): next_agent_for_refinement is None == {next_agent_for_refinement is None}")
        self._logger_instance.warning(f"AGENT DIAGNOSTIC (L988_PRE_ESCALATE_CHECK): task_input.artifact_type != ARCAReviewArtifactType.CODE_MODULE_TEST_FAILURE == {task_input.artifact_type != ARCAReviewArtifactType.CODE_MODULE_TEST_FAILURE}")
        self._logger_instance.warning(f"AGENT DIAGNOSTIC (L988_PRE_ESCALATE_CHECK): new_tasks_to_add_to_plan is empty == {not new_tasks_to_add_to_plan}")
        if new_tasks_to_add_to_plan:
            try:
                self._logger_instance.warning(f"AGENT DIAGNOSTIC (L988_PRE_ESCALATE_CHECK): new_tasks_to_add_to_plan content: {json.dumps(new_tasks_to_add_to_plan, indent=2)}")
            except TypeError as e_json:
                self._logger_instance.warning(f"AGENT DIAGNOSTIC (L988_PRE_ESCALATE_CHECK): Could not serialize new_tasks_to_add_to_plan for logging: {e_json}. Content: {str(new_tasks_to_add_to_plan)}")
        # END AGENT DIAGNOSTIC (L988)

        # Final fallback for escalation if refinement was needed but no agent path found (outside CodeModule_TestFailure)
        if decision == "REFINEMENT_REQUIRED" and not next_agent_for_refinement and task_input.artifact_type != ARCAReviewArtifactType.CODE_MODULE_TEST_FAILURE and not new_tasks_to_add_to_plan: # MODIFIED: Added 'and not new_tasks_to_add_to_plan'
            decision = "ESCALATE_TO_USER"
            escalation_reason = " Escalating: Refinement required by ARCA, but no automated agent refinement path was identified for this artifact type (and no new tasks were suggested by LLM to take precedence)."
            if escalation_reason not in reasoning:
                reasoning += escalation_reason
            self._logger_instance.warning(f"ARCA: Changing decision to ESCALATE_TO_USER for {task_input.artifact_doc_id or task_input.code_module_file_path} as no refinement agent identified (and no new tasks from LLM). Final Reasoning: {reasoning}")

        # --- Handle New Task Generation if any were identified ---
        # Initialize new_plan_doc_id here for the ARCAOutput
        new_plan_doc_id: Optional[str] = None
        if new_tasks_to_add_to_plan and decision != "ERROR": # Only proceed if not already in an error state from prior logic
            self._logger_instance.info(f"ARCA: Attempting to add {len(new_tasks_to_add_to_plan)} new tasks to the MasterExecutionPlan.")
            
            current_master_plan_doc_id: Optional[str] = None
            try:
                if self._state_manager:
                    project_state = await self._state_manager.get_project_state_v2(task_input.project_id)
                    if project_state and project_state.master_execution_plan_doc_id: # CHECKING FOR CORRECT FIELD
                        current_master_plan_doc_id = project_state.master_execution_plan_doc_id
                        self._logger_instance.info(f"ARCA: Retrieved current master plan doc ID: {current_master_plan_doc_id} from project state.")
                    else:
                        reasoning += "\n- Failed to add new tasks: MasterExecutionPlan doc ID not found in project state."
                        self._logger_instance.error(f"ARCA: MasterExecutionPlan doc ID not found in project state for project {task_input.project_id} when trying to add new tasks.")
                        decision = "ERROR"
                        arca_error_message = "MasterExecutionPlan doc ID not found in project state."
                else:
                    reasoning += "\n- Failed to add new tasks: StateManager not available."
                    self._logger_instance.error("ARCA: StateManager not available when trying to add new tasks.")
                    decision = "ERROR"
                    arca_error_message = "StateManager not available."
            except Exception as e:
                reasoning += f"\n- Failed to retrieve master plan doc id from state: {str(e)}"
                self._logger_instance.error(f"ARCA: Error retrieving master plan doc id from state: {str(e)}", exc_info=True)
                decision = "ERROR"
                arca_error_message = f"Error retrieving master plan doc id from state: {str(e)}"

            # AGENT DIAGNOSTIC: Check before new task handling
            self._logger_instance.warning(f"AGENT DIAGNOSTIC: Before new task handling - decision == {decision}")
            self._logger_instance.warning(f"AGENT DIAGNOSTIC: Before new task handling - new_tasks_to_add_to_plan is empty == {not new_tasks_to_add_to_plan}")
            if new_tasks_to_add_to_plan:
                try:
                    self._logger_instance.warning(f"AGENT DIAGNOSTIC: Before new task handling - new_tasks_to_add_to_plan content: {json.dumps(new_tasks_to_add_to_plan, indent=2)}")
                except TypeError as e_json: # Handle cases where content might not be directly JSON serializable for logging
                    self._logger_instance.warning(f"AGENT DIAGNOSTIC: Could not serialize new_tasks_to_add_to_plan for logging: {e_json}. Content: {str(new_tasks_to_add_to_plan)}")
            # END AGENT DIAGNOSTIC

            if current_master_plan_doc_id and decision != "ERROR":
                # Call the new helper method
                decision, reasoning, new_plan_doc_id_from_helper, arca_error_message = await self._handle_new_task_generation_and_plan_modification(
                    task_input=task_input,
                    new_tasks_to_add_to_plan=new_tasks_to_add_to_plan,
                    current_master_plan_doc_id=current_master_plan_doc_id,
                    initial_reasoning=reasoning,
                    initial_decision=decision # Corrected parameter name back
                )
                if new_plan_doc_id_from_helper:
                    new_plan_doc_id = new_plan_doc_id_from_helper # Update the main new_plan_doc_id
            elif decision != "ERROR":
                reasoning += "\n- Could not proceed with adding new tasks: Master plan doc ID could not be retrieved, and not already in an ERROR state."
                self._logger_instance.warning("ARCA: Could not proceed with adding new tasks as master plan doc ID was not retrieved (and not already in ERROR state from prior steps).")

        # --- Final Output Preparation ---
        arca_confidence = ConfidenceScore(value=0.9, reasoning="ARCA decision process completed.") # Placeholder, corrected score to value

        code_module_path_for_output = task_input.code_module_file_path if task_input.artifact_type == ARCAReviewArtifactType.CODE_MODULE_TEST_FAILURE else None
        output_reviewed_doc_id = task_input.artifact_doc_id or code_module_path_for_output or "N/A"

        # Prepare final_artifact_doc_id for ARCAOutput
        # If plan was modified, this should be the new plan's doc_id. Otherwise, if artifact was accepted, it's that artifact's doc_id.
        output_final_artifact_doc_id = final_doc_id_on_accept # Initial value from acceptance path
        if decision == "PLAN_MODIFIED_NEW_TASKS_ADDED" and new_plan_doc_id:
            output_final_artifact_doc_id = new_plan_doc_id
        elif decision == "PLAN_MODIFIED_NEW_TASKS_ADDED" and not new_plan_doc_id:
             # This case should ideally not happen if decision is set correctly only after successful storage.
             # If it does, it's an error state.
            self._logger_instance.error("ARCA: Decision is PLAN_MODIFIED_NEW_TASKS_ADDED but new_plan_doc_id is not set. This indicates an issue.")
            # Potentially override decision to ERROR here. For now, log and proceed with None.

        final_arca_output = ARCAOutput(
            task_id=task_input.task_id,
            project_id=task_input.project_id,
            reviewed_artifact_doc_id=output_reviewed_doc_id, 
            reviewed_artifact_type=task_input.artifact_type,
            decision=decision, 
            reasoning=reasoning, 
            confidence_in_decision=arca_confidence,
            next_agent_id_for_refinement=next_agent_for_refinement,
            next_agent_input=next_agent_refinement_input.model_dump() if next_agent_refinement_input and hasattr(next_agent_refinement_input, 'model_dump') else next_agent_refinement_input,
            debugging_task_input=debugging_input_for_output, 
            final_artifact_doc_id=output_final_artifact_doc_id, # Use the determined value
            new_master_plan_doc_id=new_plan_doc_id if decision == "PLAN_MODIFIED_NEW_TASKS_ADDED" else None, # Populate this correctly
            error_message=arca_error_message if decision == "ERROR" else None # Ensure arca_error_message is used
        )

        # PCMA Log for ARCA Decision (already correct)
        if self._project_chroma_manager:
            await self._log_event_to_pcma(
                pcma_agent=self._project_chroma_manager, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                arca_task_id=task_input.task_id, event_type="ARCA_DECISION_MADE",
                event_details={
                    "reviewed_artifact_doc_id": output_final_artifact_doc_id or output_reviewed_doc_id, # Use new plan ID if available
                    "reviewed_artifact_type": task_input.artifact_type.value,
                    "decision": final_arca_output.decision,
                    "reasoning_summary": final_arca_output.reasoning[:200] if final_arca_output.reasoning else "N/A",
                    "next_agent_id": final_arca_output.next_agent_id_for_refinement,
                    "new_plan_doc_id": final_arca_output.new_master_plan_doc_id
                },
                severity="ERROR" if final_arca_output.decision == "ERROR" else "INFO"
            )

        # --- QA Log Entry Storing --- 
        if self._project_chroma_manager:
            try:
                # ... (quality_status_mapping and action_taken_summary as before) ...

                # AGENT DIAGNOSTIC PRINTS FOR OverallQualityStatus
                self._logger_instance.info(f"AGENT DIAGNOSTIC (OverallQualityStatus): __module__ == {OverallQualityStatus.__module__}")
                self._logger_instance.info(f"AGENT DIAGNOSTIC (OverallQualityStatus): __name__ == {OverallQualityStatus.__name__}")
                self._logger_instance.info(f"AGENT DIAGNOSTIC (OverallQualityStatus): members == {[member.name for member in OverallQualityStatus]}")
                self._logger_instance.info(f"AGENT DIAGNOSTIC (OverallQualityStatus): final_arca_output.decision == {final_arca_output.decision}")
                # END AGENT DIAGNOSTIC PRINTS

                quality_status_mapping = {
                    "ACCEPT_ARTIFACT": OverallQualityStatus.APPROVED_PASSED,
                    "PROCEED_TO_DOCUMENTATION": OverallQualityStatus.APPROVED_PASSED, 
                    "REFINEMENT_REQUIRED": OverallQualityStatus.REJECTED_NEEDS_REFINEMENT,
                    "ESCALATE_TO_USER": OverallQualityStatus.FLAGGED_FOR_MANUAL_REVIEW,
                    "TEST_FAILURE_HANDOFF": OverallQualityStatus.UNDER_REMEDIATION, 
                    "ERROR": OverallQualityStatus.ERROR_IN_QA_PROCESS 
                }
                overall_quality_status = quality_status_mapping.get(final_arca_output.decision, OverallQualityStatus.FLAGGED_FOR_MANUAL_REVIEW)

                action_taken_summary = f"ARCA decision: {final_arca_output.decision}."
                if final_arca_output.decision == "REFINEMENT_REQUIRED" and final_arca_output.next_agent_id_for_refinement:
                    action_taken_summary += f" Next agent: {final_arca_output.next_agent_id_for_refinement}."
                elif final_arca_output.decision == "TEST_FAILURE_HANDOFF":
                    action_taken_summary += " Handoff to CodeDebuggingAgent."

                qa_log_entry_content = QualityAssuranceLogEntry(
                    project_id=task_input.project_id,
                    cycle_id=task_input.cycle_id,
                    artifact_doc_id_assessed=final_arca_output.reviewed_artifact_doc_id,
                    artifact_type_assessed=task_input.artifact_type.value, 
                    qa_event_type=QAEventType.ARCA_ARTIFACT_ASSESSMENT,
                    assessing_entity_id=self.AGENT_ID,
                    summary_of_assessment=final_arca_output.reasoning[:1000] if final_arca_output.reasoning else "No reasoning provided.", 
                    overall_quality_status=overall_quality_status,
                    confidence_in_assessment=final_arca_output.confidence_in_decision.value if final_arca_output.confidence_in_decision else 0.0,
                    action_taken_or_recommended=action_taken_summary,
                    key_metrics_or_findings={
                        "generator_agent_id": task_input.generator_agent_id,
                        "generator_confidence": task_input.generator_agent_confidence.score if task_input.generator_agent_confidence else None,
                        "arca_decision_confidence": final_arca_output.confidence_in_decision.value if final_arca_output.confidence_in_decision else None,
                        "combined_heuristic_metric": combined_metric if task_input.artifact_type != ARCAReviewArtifactType.CODE_MODULE_TEST_FAILURE else None,
                        "llm_eval_summary": llm_optimization_evaluation_summary, # Ensure this is populated
                        "num_llm_recommended_incorporations": len(llm_recommended_incorporations) # Ensure this is populated
                    }
                )
                await self._project_chroma_manager.store_artifact(
                    StoreArtifactInput(
                        base_collection_name=QUALITY_ASSURANCE_LOGS_COLLECTION,
                        artifact_content=qa_log_entry_content.model_dump(mode='json'),
                        metadata={
                            "artifact_type": ARTIFACT_TYPE_QA_LOG_ENTRY_JSON,
                            "project_id": task_input.project_id,
                            "cycle_id": task_input.cycle_id,
                            "assessed_artifact_id": final_arca_output.reviewed_artifact_doc_id,
                            "assessed_artifact_type": task_input.artifact_type.value
                        },
                        cycle_id=task_input.cycle_id, # Pass cycle_id for lineage
                        source_agent_id=self.AGENT_ID # ARCA is the source of this log
                    )
                )
                self._logger_instance.info(f"Successfully stored QA log entry: {qa_log_entry_content.log_id} for artifact {final_arca_output.reviewed_artifact_doc_id}")
            except Exception as e_qa_log:
                self._logger_instance.error(f"Exception during QA log storing: {e_qa_log}", exc_info=True)

        return final_arca_output

    async def _handle_new_task_generation_and_plan_modification(
        self,
        task_input: ARCAReviewInput,
        new_tasks_to_add_to_plan: List[Dict[str, Any]],
        current_master_plan_doc_id: Optional[str], # Made Optional, as it might not always exist
        initial_reasoning: str,
        initial_decision: Literal[
            "REFINEMENT_REQUIRED", "ESCALATE_TO_USER", "NO_ACTION_NEEDED", "ERROR_INTERNAL", "PLAN_MODIFIED_NEW_TASKS_ADDED", "PROJECT_QA_REVIEW_COMPLETE_PROCEED", "PROJECT_QA_REVIEW_COMPLETE_HALT"
        ]
    ) -> tuple[
        Literal[
            "REFINEMENT_REQUIRED", "ESCALATE_TO_USER", "NO_ACTION_NEEDED", "ERROR_INTERNAL", "PLAN_MODIFIED_NEW_TASKS_ADDED", "PROJECT_QA_REVIEW_COMPLETE_PROCEED", "PROJECT_QA_REVIEW_COMPLETE_HALT"
        ],
        str, # reasoning
        Optional[str], # new_master_plan_doc_id
        Optional[str]  # arca_error_message
    ]:
        """Orchestrates fetching, modifying, and storing the MasterExecutionPlan if new tasks are to be added."""
        reasoning = initial_reasoning
        decision = initial_decision 
        new_plan_doc_id: Optional[str] = None
        arca_error_message: Optional[str] = None

        if not new_tasks_to_add_to_plan:
            self._logger_instance.info("ARCA Orchestrator (plan_mod): No new tasks to add to plan. Skipping plan modification.")
            reasoning += "\\n- Orchestrator (plan_mod): No new tasks were identified for addition to the Master Execution Plan."
            # Decision and other return values remain as they were passed in initially
            return decision, reasoning, new_plan_doc_id, arca_error_message

        if not current_master_plan_doc_id:
            self._logger_instance.error("ARCA Orchestrator (plan_mod): current_master_plan_doc_id is missing. Cannot proceed with plan modification.")
            reasoning += "\\n- Orchestrator (plan_mod): Cannot modify plan - current_master_plan_doc_id is missing."
            # Update decision to ERROR_INTERNAL if it wasn't already, or keep it if it's a more specific error.
            decision = "ERROR_INTERNAL" if decision not in ["ERROR_INTERNAL"] else decision
            arca_error_message = "Cannot modify plan: current_master_plan_doc_id is missing."
            return decision, reasoning, new_plan_doc_id, arca_error_message

        self._logger_instance.info(f"ARCA Orchestrator (plan_mod): Starting plan modification process. Current plan_doc_id: {current_master_plan_doc_id}")
        reasoning += "\\n- Orchestrator (plan_mod): Attempting to modify MasterExecutionPlan to add new tasks."

        # Step 1: Fetch and Parse the current Master Plan
        parsed_master_plan, reasoning, fetch_parse_error = await self._fetch_and_parse_master_plan(
            current_master_plan_doc_id=current_master_plan_doc_id,
            existing_reasoning=reasoning
        )

        if fetch_parse_error or not parsed_master_plan:
            self._logger_instance.error(f"ARCA Orchestrator (plan_mod): Failed to fetch/parse plan. Error: {fetch_parse_error}")
            decision = "ERROR_INTERNAL"
            arca_error_message = fetch_parse_error or "Failed to fetch or parse the master plan."
            # reasoning is already updated by the helper
            return decision, reasoning, new_plan_doc_id, arca_error_message
        
        self._logger_instance.info(f"ARCA Orchestrator (plan_mod): Plan fetched and parsed successfully. Plan ID: {parsed_master_plan.id}")

        # Step 2: Apply new tasks to the parsed plan object
        plan_was_modified, reasoning = await self._apply_new_tasks_to_parsed_plan(
            parsed_master_plan=parsed_master_plan, # This will be mutated
            new_tasks_to_add_to_plan=new_tasks_to_add_to_plan,
            existing_reasoning=reasoning
        )

        if not plan_was_modified:
            self._logger_instance.info("ARCA Orchestrator (plan_mod): Plan was not modified by _apply_new_tasks_to_parsed_plan (e.g., no tasks applicable, or internal logic decided not to modify). No further action to store plan.")
            reasoning += "\\n- Orchestrator (plan_mod): No modifications were applied to the plan structure by the task application logic."
            # Decision and other return values remain as they were (unless an error occurred inside _apply_new_tasks_to_parsed_plan that should have been bubbled up differently, but its signature returns bool)
            # If _apply_new_tasks_to_parsed_plan encounters critical errors, it should log them and potentially the reasoning would reflect it.
            return decision, reasoning, new_plan_doc_id, arca_error_message

        self._logger_instance.info("ARCA Orchestrator (plan_mod): Plan was modified with new tasks. Proceeding to serialize and store.")

        # Step 3: Serialize, Store the modified plan, and Update StateManager
        stored_new_plan_doc_id, reasoning, serialize_store_error = await self._serialize_store_and_update_state_for_plan(
            modified_plan=parsed_master_plan, # The mutated plan object
            task_input=task_input,
            original_plan_doc_id=current_master_plan_doc_id,
            existing_reasoning=reasoning
        )

        if serialize_store_error or not stored_new_plan_doc_id:
            self._logger_instance.error(f"ARCA Orchestrator (plan_mod): Failed to serialize/store/update state for modified plan. Error: {serialize_store_error}")
            decision = "ERROR_INTERNAL"
            arca_error_message = serialize_store_error or "Failed to save the modified master plan or update state."
            # reasoning is already updated by the helper
            return decision, reasoning, new_plan_doc_id, arca_error_message # new_plan_doc_id is still None here

        # Success Case for Plan Modification
        new_plan_doc_id = stored_new_plan_doc_id
        decision = "PLAN_MODIFIED_NEW_TASKS_ADDED"
        reasoning += f"\\n- Orchestrator (plan_mod): Successfully modified MasterExecutionPlan. New plan document ID: {new_plan_doc_id}. Decision set to PLAN_MODIFIED_NEW_TASKS_ADDED."
        self._logger_instance.info(f"ARCA Orchestrator (plan_mod): Successfully processed plan modification. New plan ID: {new_plan_doc_id}. Decision: {decision}")
        
        return decision, reasoning, new_plan_doc_id, arca_error_message

    async def _apply_new_tasks_to_parsed_plan(
        self,
        parsed_master_plan: MasterExecutionPlan, # This object will be mutated
        new_tasks_to_add_to_plan: List[Dict[str, Any]],
        existing_reasoning: str
    ) -> tuple[bool, str]:
        """Applies new tasks to the parsed MasterExecutionPlan object, mutating it directly."""
        reasoning = existing_reasoning
        plan_was_modified = False

        if not parsed_master_plan:
            self._logger_instance.error("ARCA Helper (apply_tasks): parsed_master_plan is None. Cannot apply tasks.")
            reasoning += "\\n- Helper (apply_tasks): parsed_master_plan was None, cannot apply tasks."
            return False, reasoning

        for task_details in new_tasks_to_add_to_plan:
            self._logger_instance.info(f"ARCA Helper (apply_tasks): Processing new task details for plan modification: {task_details}")
            new_stage_id = task_details.get("new_stage_id", f"stage_{uuid.uuid4().hex[:8]}")
            
            if not task_details.get("name") or not task_details.get("agent_id"):
                self._logger_instance.error(f"ARCA Helper (apply_tasks): Skipping new task due to missing 'name' or 'agent_id': {task_details}")
                reasoning += f"\\n- Helper (apply_tasks): Skipped adding new task (ID: {new_stage_id}) due to missing critical details (name/agent_id)."
                continue
            try:
                new_stage = MasterStageSpec(
                    id=new_stage_id,
                    name=task_details["name"],
                    description=task_details.get("description"),
                    agent_id=task_details["agent_id"],
                    agent_category=AgentCategory(task_details["agent_category"]) if task_details.get("agent_category") else None,
                    inputs=task_details.get("inputs", {}),
                    success_criteria=task_details.get("success_criteria", []),
                    output_context_path=task_details.get("output_context_path_suggestion")
                )
                self._logger_instance.debug(f"ARCA Helper (apply_tasks): Created new MasterStageSpec: {new_stage.model_dump_json(indent=2)}")
                
                placement_hint = task_details.get("placement_hint", {})
                inserted = False

                if placement_hint.get("insert_as_initial_stage"):
                    self._logger_instance.info(f"ARCA Helper (apply_tasks): Attempting to insert new stage {new_stage_id} as initial stage.")
                    old_initial_stage_id = parsed_master_plan.initial_stage
                    new_stage.next_stage = old_initial_stage_id
                    new_stage.number = "1.0"
                    if old_initial_stage_id and old_initial_stage_id in parsed_master_plan.stages:
                        try:
                            old_initial_stage_obj = parsed_master_plan.stages[old_initial_stage_id]
                            if old_initial_stage_obj.number and old_initial_stage_obj.number.startswith("1"):
                                 old_initial_stage_obj.number = "2.0"
                        except Exception as e_renum:
                            self._logger_instance.warning(f"ARCA Helper (apply_tasks): Could not re-number old initial stage {old_initial_stage_id}: {e_renum}")
                    else:
                        new_stage.next_stage = "FINAL_STEP"
                    parsed_master_plan.initial_stage = new_stage_id
                    parsed_master_plan.stages[new_stage_id] = new_stage
                    plan_was_modified = True
                    inserted = True
                    reasoning += f"\\n- Helper (apply_tasks): New task {new_stage_id} inserted as initial stage."
                elif placement_hint.get("insert_after_stage_id"):
                    target_stage_id = placement_hint["insert_after_stage_id"]
                    self._logger_instance.info(f"ARCA Helper (apply_tasks): Attempting to insert new stage {new_stage_id} after stage {target_stage_id}.")
                    if target_stage_id in parsed_master_plan.stages:
                        target_stage = parsed_master_plan.stages[target_stage_id]
                        new_stage.next_stage = target_stage.next_stage
                        target_stage.next_stage = new_stage_id
                        base_number = target_stage.number or "0"
                        sub_level = 1
                        while f"{base_number}.{sub_level}" in [s.number for s in parsed_master_plan.stages.values() if s.number]: # Added check for s.number not None
                            sub_level += 1
                        new_stage.number = f"{base_number}.{sub_level}"
                        parsed_master_plan.stages[new_stage_id] = new_stage
                        plan_was_modified = True
                        inserted = True
                        reasoning += f"\\n- Helper (apply_tasks): New task {new_stage_id} inserted after {target_stage_id}. New number: {new_stage.number}."
                    else:
                        self._logger_instance.error(f"ARCA Helper (apply_tasks): Target stage_id '{target_stage_id}' for insertion not found. Skipping {new_stage_id}.")
                        reasoning += f"\\n- Helper (apply_tasks): Failed to insert new task {new_stage_id}: target '{target_stage_id}' not found."
                
                if not inserted:
                    # Default placement: append to the end of the plan
                    # This requires finding the current last stage.
                    # For simplicity, if no specific placement and not initial, we can log and skip complex end-append for now, or implement it.
                    # Let's try a simple end-append if no other hints matched.
                    self._logger_instance.warning(f"ARCA Helper (apply_tasks): New stage {new_stage_id} not inserted by specific hints. Attempting to append to end.")
                    if not parsed_master_plan.stages: # If plan has no stages yet
                        new_stage.number = "1.0"
                        new_stage.next_stage = "FINAL_STEP"
                        parsed_master_plan.initial_stage = new_stage_id
                        parsed_master_plan.stages[new_stage_id] = new_stage
                        plan_was_modified = True
                        inserted = True
                        reasoning += f"\\n- Helper (apply_tasks): New task {new_stage_id} inserted as the first and only stage."
                    else:
                        # Find the last stage by iterating or if there's a known tail pointer (not in current schema)
                        # Simplified: find stage that points to FINAL_STEP or has no next_stage and highest number
                        last_stage_id: Optional[str] = None
                        highest_num = -1.0
                        current_stage_id = parsed_master_plan.initial_stage
                        visited_stages = set()
                        while current_stage_id and current_stage_id != "FINAL_STEP" and current_stage_id not in visited_stages:
                            visited_stages.add(current_stage_id)
                            stage = parsed_master_plan.stages.get(current_stage_id)
                            if not stage:
                                break # Should not happen in a valid plan
                            if stage.next_stage == "FINAL_STEP" or not stage.next_stage:
                                last_stage_id = current_stage_id
                                break
                            current_stage_id = stage.next_stage
                        
                        if last_stage_id and last_stage_id in parsed_master_plan.stages:
                            last_stage_obj = parsed_master_plan.stages[last_stage_id]
                            last_stage_obj.next_stage = new_stage_id
                            new_stage.next_stage = "FINAL_STEP"
                            try:
                                new_stage.number = str(float(last_stage_obj.number or "0") + 1.0) # Basic numbering for appended stage
                            except ValueError:
                                new_stage.number = "999" # Fallback if last stage number is not float-convertible
                            parsed_master_plan.stages[new_stage_id] = new_stage
                            plan_was_modified = True
                            inserted = True
                            reasoning += f"\\n- Helper (apply_tasks): New task {new_stage_id} appended to the end of the plan."
                        else:
                            self._logger_instance.error(f"ARCA Helper (apply_tasks): Could not determine last stage to append {new_stage_id}. Manual placement might be needed.")
                            reasoning += f"\\n- Helper (apply_tasks): New task {new_stage_id} ({new_stage.name}) prepared, but could not append to end automatically."

            except Exception as e_stage_create_place: # Catches errors in MasterStageSpec creation or placement logic
                self._logger_instance.error(f"ARCA Helper (apply_tasks): Failed to create/place MasterStageSpec for task {task_details}: {e_stage_create_place}", exc_info=True)
                reasoning += f"\\n- Helper (apply_tasks): Failed to process new task into a stage: {e_stage_create_place}"
        
        return plan_was_modified, reasoning

    async def _fetch_and_parse_master_plan(
        self,
        current_master_plan_doc_id: str,
        existing_reasoning: str
    ) -> tuple[Optional[MasterExecutionPlan], str, Optional[str]]:
        """Fetches plan content from PCMA and parses it into a MasterExecutionPlan object."""
        reasoning = existing_reasoning
        parsed_master_plan: Optional[MasterExecutionPlan] = None
        error_message: Optional[str] = None

        if not self._project_chroma_manager:
            self._logger_instance.error("ARCA Helper (_fetch_and_parse_master_plan): ProjectChromaManager not available.")
            reasoning += "\\n- Helper (fetch/parse): ProjectChromaManager not available to fetch plan."
            return None, reasoning, "Helper (fetch/parse): ProjectChromaManager not available."

        try:
            self._logger_instance.info(f"ARCA Helper (fetch/parse): Fetching MasterExecutionPlan content for doc_id: {current_master_plan_doc_id}")
            retrieved_plan_artifact: RetrieveArtifactOutput = await self._project_chroma_manager.retrieve_artifact(
                doc_id=current_master_plan_doc_id,
                collection_name=None # Assuming doc_id is unique enough or PCMA has defaults
            )

            if retrieved_plan_artifact.status == "SUCCESS" and retrieved_plan_artifact.artifact_content:
                active_master_plan_content_str = retrieved_plan_artifact.artifact_content
                self._logger_instance.info(f"ARCA Helper (fetch/parse): Successfully fetched MasterExecutionPlan content (length: {len(active_master_plan_content_str)} chars).")
                try:
                    parsed_master_plan = MasterExecutionPlan.from_yaml(active_master_plan_content_str)
                    self._logger_instance.info(f"ARCA Helper (fetch/parse): Successfully parsed MasterExecutionPlan YAML for plan ID: {parsed_master_plan.id}")
                except yaml.YAMLError as e_yaml_parse:
                    self._logger_instance.error(f"ARCA Helper (fetch/parse): Failed to parse MasterExecutionPlan YAML: {e_yaml_parse}", exc_info=True)
                    reasoning += f"\\n- Helper (fetch/parse): Failed to parse MasterExecutionPlan YAML: {e_yaml_parse}"
                    error_message = f"Helper (fetch/parse): Failed to parse plan YAML: {e_yaml_parse}"
                    parsed_master_plan = None # Ensure it's None on error
                except Exception as e_parse_generic: # Catch other Pydantic validation errors etc.
                    self._logger_instance.error(f"ARCA Helper (fetch/parse): Error during MasterExecutionPlan parsing (e.g. Pydantic): {e_parse_generic}", exc_info=True)
                    reasoning += f"\\n- Helper (fetch/parse): Error parsing MasterExecutionPlan: {e_parse_generic}"
                    error_message = f"Helper (fetch/parse): Error parsing MasterExecutionPlan: {e_parse_generic}"
                    parsed_master_plan = None # Ensure it's None on error
            else:
                self._logger_instance.error(f"ARCA Helper (fetch/parse): Failed to retrieve plan from PCMA. Status: {retrieved_plan_artifact.status}, Msg: {retrieved_plan_artifact.message or retrieved_plan_artifact.error_message}")
                reasoning += f"\\n- Helper (fetch/parse): Failed to retrieve plan. PCMA Status: {retrieved_plan_artifact.status}"
                error_message = f"Helper (fetch/parse): Failed to retrieve plan: {retrieved_plan_artifact.message or retrieved_plan_artifact.error_message}"
        except Exception as e_pcma_retrieve_outer: # Catches error in the initial PCMA retrieve_artifact call
            self._logger_instance.error(f"ARCA Helper (fetch/parse): Exception retrieving plan from PCMA: {e_pcma_retrieve_outer}", exc_info=True)
            reasoning += f"\\n- Helper (fetch/parse): Exception retrieving plan from PCMA: {e_pcma_retrieve_outer}"
            error_message = f"Helper (fetch/parse): Exception retrieving plan: {e_pcma_retrieve_outer}"
            
        return parsed_master_plan, reasoning, error_message

    async def _serialize_store_and_update_state_for_plan(
        self,
        modified_plan: MasterExecutionPlan,
        task_input: ARCAReviewInput, # For project_id, cycle_id
        original_plan_doc_id: str, # For metadata reference
        existing_reasoning: str
    ) -> tuple[Optional[str], str, Optional[str]]:
        """Serializes the plan, stores it in PCMA, and updates StateManager."""
        reasoning = existing_reasoning
        new_plan_doc_id: Optional[str] = None
        error_message: Optional[str] = None

        if not self._project_chroma_manager or not self._state_manager:
            self._logger_instance.error("ARCA Helper (serialize_store): PCMA or StateManager not available.")
            reasoning += "\\n- Helper (serialize/store): PCMA or StateManager not available."
            return None, reasoning, "Helper (serialize/store): PCMA or StateManager not available."

        try:
            updated_plan_yaml = modified_plan.to_yaml()
            self._logger_instance.info(f"ARCA Helper (serialize/store): Successfully serialized modified plan (ID: {modified_plan.id}, Len: {len(updated_plan_yaml)}). Attempting to store in PCMA.")
            
            store_output = await self._project_chroma_manager.store_artifact(
                content=updated_plan_yaml,
                artifact_type="MasterExecutionPlan",
                project_id=task_input.project_id,
                name=f"MasterExecutionPlan_cycle{task_input.cycle_id}_modified_from_{original_plan_doc_id}.yaml",
                description=f"Master Execution Plan modified by ARCA in cycle {task_input.cycle_id} based on original {original_plan_doc_id}. New tasks added.",
                metadata={"source_plan_doc_id": original_plan_doc_id, "modification_cycle": task_input.cycle_id, "modified_by_agent": self.AGENT_ID}
            )

            if store_output.status == "SUCCESS" and store_output.doc_id:
                new_plan_doc_id = store_output.doc_id
                self._logger_instance.info(f"ARCA Helper (serialize/store): Successfully stored modified plan. New doc_id: {new_plan_doc_id}")
                reasoning += f"\\n- Helper (serialize/store): Stored modified plan. New doc_id: {new_plan_doc_id}"
                
                # Update StateManager with the new plan doc ID
                try:
                    await self._state_manager.update_project_state_v2(
                        project_id=task_input.project_id,
                        update_data=ProjectStateDataV2(current_master_plan_doc_id=new_plan_doc_id)
                    )
                    self._logger_instance.info(f"ARCA Helper (serialize/store): StateManager updated with new plan_doc_id: {new_plan_doc_id}")
                    reasoning += f"\\n- Helper (serialize/store): StateManager updated with new plan_doc_id: {new_plan_doc_id}."
                except Exception as e_sm_update:
                    self._logger_instance.error(f"ARCA Helper (serialize/store): Failed to update StateManager with new plan_doc_id: {e_sm_update}", exc_info=True)
                    reasoning += f"\\n- Helper (serialize/store): Failed to update StateManager: {e_sm_update}"
                    error_message = f"Helper (serialize/store): Failed to update StateManager: {e_sm_update}" 
                    # Note: Plan is stored, but state update failed. This is a partial success/failure.
                    # ARCA's main logic should decide if this constitutes a full error for the operation.
            else:
                self._logger_instance.error(f"ARCA Helper (serialize/store): Failed to store modified plan in PCMA. Status: {store_output.status}, Msg: {store_output.message}")
                reasoning += f"\\n- Helper (serialize/store): Failed to store modified plan in PCMA. Status: {store_output.status}"
                error_message = f"Helper (serialize/store): Failed to store modified plan: {store_output.message}"

        except yaml.YAMLError as e_yaml_serialize: # Should be less common for to_yaml but possible
            self._logger_instance.error(f"ARCA Helper (serialize/store): Failed to serialize modified plan to YAML: {e_yaml_serialize}", exc_info=True)
            reasoning += f"\\n- Helper (serialize/store): Failed to serialize modified plan: {e_yaml_serialize}"
            error_message = f"Helper (serialize/store): Failed to serialize plan: {e_yaml_serialize}"
        except Exception as e_serialize_store_generic:
            self._logger_instance.error(f"ARCA Helper (serialize/store): Generic error during plan serialization or storage: {e_serialize_store_generic}", exc_info=True)
            reasoning += f"\\n- Helper (serialize/store): Generic error during plan serialization/storage: {e_serialize_store_generic}"
            error_message = f"Helper (serialize/store): Generic error in serialization/storage: {e_serialize_store_generic}"
            
        return new_plan_doc_id, reasoning, error_message

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=AutomatedRefinementCoordinatorAgent_v1.AGENT_ID,
            name=AutomatedRefinementCoordinatorAgent_v1.AGENT_NAME,
            description=AutomatedRefinementCoordinatorAgent_v1.AGENT_DESCRIPTION,
            version=AutomatedRefinementCoordinatorAgent_v1.VERSION,
            input_schema=ARCAReviewInput.model_json_schema(),
            output_schema=ARCAOutput.model_json_schema(),
            categories=[cat.value for cat in [AutomatedRefinementCoordinatorAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=AutomatedRefinementCoordinatorAgent_v1.VISIBILITY.value,
            capability_profile={
                "coordinates_workflows": ["LOPRD_Refinement", "Blueprint_Refinement", "MasterExecutionPlan_Refinement", "CodeModule_Refinement", "Documentation_Triggering"],
                "consumes_reports": ["PRAAReport", "RTAReport", "TestReportSummary"],
                "makes_decisions": ["AcceptArtifact", "RequestRefinement", "ProceedToDocumentation"],
                "primary_function": "Quality gate, refinement loop management, and documentation handoff"
            },
            metadata={
                "callable_fn_path": f"{AutomatedRefinementCoordinatorAgent_v1.__module__}.{AutomatedRefinementCoordinatorAgent_v1.__name__}"
            }
        ) 