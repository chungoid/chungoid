from __future__ import annotations

import logging
import datetime
import uuid
import json
from typing import Any, Dict, Optional, Literal, Union, ClassVar, List, Type, get_args
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider # Assuming direct LLM call for complex decisions might be an option
from chungoid.utils.prompt_manager import PromptManager # If decision logic uses prompts
from chungoid.schemas.common import ConfidenceScore
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

logger = logging.getLogger(__name__)

ARCA_PROMPT_NAME = "automated_refinement_coordinator_agent_v1.yaml" # If LLM-based decision making is used
ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME = "arca_optimization_evaluator_v1_prompt.yaml" # NEW PROMPT

# Constants for ARCA behavior
MAX_DEBUGGING_ATTEMPTS_PER_MODULE: ClassVar[int] = 3 # Added ClassVar
DEFAULT_ACCEPTANCE_THRESHOLD: ClassVar[float] = 0.85 # MODIFIED: Added ClassVar

# Define the Literal for artifact types including the new one
class ARCAReviewArtifactType(Enum):
    LOPRD = "LOPRD"
    ProjectBlueprint = "ProjectBlueprint"
    MasterExecutionPlan = "MasterExecutionPlan"
    CodeModule = "CodeModule"  # Generic code module
    RiskAssessmentReport = "RiskAssessmentReport"
    TraceabilityReport = "TraceabilityReport"
    OptimizationSuggestionReport = "OptimizationSuggestionReport"
    ProjectDocumentation = "ProjectDocumentation"  # For reviewing existing documentation
    CodeModule_TestFailure = "CodeModule_TestFailure"
    QA_Summary_Report = "QA_Summary_Report" # Placeholder, as in the original Literal
    CodeModule_Generated = "CodeModule_Generated"
    GenerateProjectDocumentation = "GenerateProjectDocumentation" # Signal to generate new documentation
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
    decision: Literal["ACCEPT_ARTIFACT", "REFINEMENT_REQUIRED", "ESCALATE_TO_USER", "TEST_FAILURE_HANDOFF", "PROCEED_TO_DOCUMENTATION", "ERROR"] = Field(..., description="The final decision of the ARCA review.")
    reasoning: str = Field(..., description="The detailed reasoning behind the decision.")
    confidence_in_decision: Optional[ConfidenceScore] = Field(None, description="ARCA's confidence in its own decision.")
    
    # If decision is REFINEMENT_REQUIRED, these fields provide details for the orchestrator
    next_agent_id_for_refinement: Optional[str] = Field(None, description="The agent_id to call for refinement (e.g., ProductAnalystAgent_v1, ArchitectAgent_v1, SystemMasterPlannerAgent_v1).")
    next_agent_input: Optional[Union[pa_module.ProductAnalystAgentInput, ArchitectAgentInput, MasterPlannerInput, ProjectDocumentationAgentInput, Dict[str, Any]]] = Field(None, description="The full input payload for the next agent if refinement is needed.")
    
    # If decision is TEST_FAILURE_HANDOFF
    debugging_task_input: Optional[DebuggingTaskInput] = Field(None, description="Input for the CodeDebuggingAgent if a test failure is being handed off.")

    # If decision is ACCEPT_ARTIFACT
    final_artifact_doc_id: Optional[str] = Field(None, description="The doc_id of the artifact if accepted (usually same as input artifact_doc_id).")

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

    _llm_provider: Optional[LLMProvider]
    _prompt_manager: Optional[PromptManager]
    _project_chroma_manager: Optional[ProjectChromaManagerAgent_v1] # Added instance variable
    _code_debugging_agent_instance: Optional[CodeDebuggingAgent_v1] = None # Instance for debugger
    _logger_instance: logging.Logger # Use instance logger
    _state_manager: Optional[StateManager] = None # Ensure this is typed and initialized
    _current_debug_attempts_for_module: int = 0 # Instance variable for tracking
    _last_feedback_doc_id: Optional[str] = None # Instance variable for refinement loops

    # Thresholds for decision making (can be made configurable later)
    MIN_GENERATOR_CONFIDENCE: ClassVar[float] = 0.6
    MIN_PRAA_CONFIDENCE: ClassVar[float] = 0.5
    MIN_RTA_CONFIDENCE: ClassVar[float] = 0.6
    MIN_DOC_AGENT_CONFIDENCE_FOR_ACCEPT: ClassVar[float] = 0.5

    def __init__(
        self, 
        llm_provider: Optional[LLMProvider] = None,
        prompt_manager: Optional[PromptManager] = None,
        project_chroma_manager: Optional[ProjectChromaManagerAgent_v1] = None, # Added
        system_context: Optional[Dict[str, Any]] = None,
        state_manager: Optional[StateManager] = None,
        **kwargs
    ):
        # Initialize logger first
        if system_context and "logger" in system_context:
            self._logger_instance = system_context["logger"]
        else:
            self._logger_instance = logging.getLogger(self.AGENT_ID)

        super().__init__(**kwargs) # Ensure superclass is called if it has an __init__

        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        self._project_chroma_manager = project_chroma_manager # Stored
        self._state_manager = state_manager # Initialize instance variable
        
        # Initialize CodeDebuggingAgent instance
        if self._llm_provider and self._prompt_manager and self._project_chroma_manager:
            try:
                self._code_debugging_agent_instance = CodeDebuggingAgent_v1(
                    llm_provider=self._llm_provider,
                    prompt_manager=self._prompt_manager,
                    project_chroma_manager=self._project_chroma_manager,
                    system_context={"logger": self._logger_instance.getChild("CodeDebuggingAgent_v1")}
                )
                self._logger_instance.info("CodeDebuggingAgent_v1 instance successfully created and configured within ARCA.")
            except Exception as e:
                self._code_debugging_agent_instance = None
                self._logger_instance.error(f"Failed to initialize CodeDebuggingAgent_v1 within ARCA: {e}", exc_info=True)
        else:
            self._code_debugging_agent_instance = None
            missing_deps = []
            if not self._llm_provider: missing_deps.append("LLMProvider")
            if not self._prompt_manager: missing_deps.append("PromptManager")
            if not self._project_chroma_manager: missing_deps.append("ProjectChromaManagerAgent_v1")
            self._logger_instance.warning(
                f"CodeDebuggingAgent_v1 could not be instantiated in ARCA due to missing dependencies: {', '.join(missing_deps)}."
            )

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
        logger_instance = self._logger_instance
        llm_provider_instance = self._llm_provider
        prompt_manager_instance = self._prompt_manager
        pcma_instance = self._project_chroma_manager # Use the instance variable
        state_manager_instance = self._state_manager

        # Log ARCA Invocation Start
        if pcma_instance:
            await self._log_event_to_pcma(
                pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                arca_task_id=task_input.task_id, event_type="ARCA_INVOCATION_START",
                event_details={"task_input_summary": task_input.model_dump_json(indent=2, exclude_none=True)[:500]}, # Log summary
                severity="INFO"
            )

        # Decision variables
        decision: Literal["ACCEPT_ARTIFACT", "REFINEMENT_REQUIRED", "ESCALATE_TO_USER", "TEST_FAILURE_HANDOFF", "PROCEED_TO_DOCUMENTATION", "ERROR"] = "ERROR"
        reasoning = "Initialization error or unsupported artifact type."
        next_agent_for_refinement: Optional[str] = None
        next_agent_refinement_input: Optional[Union[pa_module.ProductAnalystAgentInput, ArchitectAgentInput, MasterPlannerInput, ProjectDocumentationAgentInput, DebuggingTaskInput, Dict[str, Any]]] = None
        arca_confidence: Optional[ConfidenceScore] = None
        issues_for_human_review_list: List[Dict[str, str]] = []
        final_doc_id_on_accept: Optional[str] = task_input.artifact_doc_id # Default to input if accepted
        code_module_path_for_output: Optional[str] = task_input.code_module_file_path

        # --- Handle GenerateProjectDocumentation Task ---
        if task_input.artifact_type == ARCAReviewArtifactType.GenerateProjectDocumentation:
            logger_instance.info(f"ARCA: Received GenerateProjectDocumentation task for project {task_input.project_id}, cycle {task_input.cycle_id}.")
            if not (task_input.final_loprd_doc_id_for_docs and \
                    task_input.final_blueprint_doc_id_for_docs and \
                    task_input.final_plan_doc_id_for_docs and \
                    task_input.final_code_root_path_for_docs):
                decision = "ERROR"
                reasoning = "Cannot proceed with documentation generation: Missing critical input document IDs (LOPRD, Blueprint, Plan) or code root path in ARCAReviewInput."
                logger_instance.error(reasoning)
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
                logger_instance.info(f"ARCA: Prepared input for ProjectDocumentationAgent_v1. Cycle ID: {task_input.cycle_id}")
            
            # Early exit for documentation task
            # No further review or state updates related to a *reviewed artifact* are needed here.
            # StateManager updates for cycle completion might still occur if this is the end of a cycle.
            
            # Log ARCA Decision Made
            if pcma_instance:
                await self._log_event_to_pcma(
                    pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
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
                # or the doc_id itself might sometimes be a path or a content placeholder.
                # Given pcma_instance.retrieve_artifact needs a base_collection_name, we must define it.
                # This will be a placeholder and likely incorrect until actual storage collections are known.
                
                # A more robust way: the agent that *created* the artifact_doc_id should also provide its collection.
                # Or ARCA needs a mapping.
                # For now, let's assume a hypothetical default or look it up.
                # This lookup logic is simplified:
                artifact_collection_map = {
                    ARCAReviewArtifactType.LOPRD: "loprds_collection", # Example, needs to be correct
                    ARCAReviewArtifactType.ProjectBlueprint: "project_blueprints_collection", # Example
                    ARCAReviewArtifactType.MasterExecutionPlan: "master_execution_plans_collection", # Example
                    ARCAReviewArtifactType.CodeModule: GENERATED_CODE_ARTIFACTS_COLLECTION,
                    # Add other types if they need to be fetched for summarization
                }
                main_artifact_collection_name = artifact_collection_map.get(task_input.artifact_type)

                if pcma_instance and main_artifact_collection_name:
                    try:
                        main_artifact_retrieved: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                            base_collection_name=main_artifact_collection_name,
                            document_id=task_input.artifact_doc_id
                        )
                        if main_artifact_retrieved and main_artifact_retrieved.status == "SUCCESS" and main_artifact_retrieved.content:
                            main_artifact_content = str(main_artifact_retrieved.content)
                            logger_instance.info(f"Successfully retrieved main artifact ({task_input.artifact_type.value}) content for doc_id: {task_input.artifact_doc_id}")
                        else:
                            logger_instance.warning(f"Failed to retrieve main artifact ({task_input.artifact_type.value}) content for doc_id {task_input.artifact_doc_id}. Status: {main_artifact_retrieved.status if main_artifact_retrieved else 'N/A'}")
                    except Exception as e_main_artifact:
                        logger_instance.error(f"Exception retrieving main artifact ({task_input.artifact_type.value}) {task_input.artifact_doc_id}: {e_main_artifact}", exc_info=True)
                elif not main_artifact_collection_name:
                    logger_instance.warning(f"No collection mapping found for artifact type {task_input.artifact_type.value} to retrieve its content for summarization.")


        # Ensure pcma_instance is valid before proceeding
        if not pcma_instance:
            logger_instance.error("ARCA: pcma_instance is not available for retrieving supporting documents. Cannot proceed effectively.")
            # Potentially return an error ARCAOutput here, or allow decision logic to handle missing content
            # For now, we will let it proceed and the decision logic will have to cope with None content.
        else:
            try:
                if task_input.praa_risk_report_doc_id:
                    doc_output: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                        base_collection_name=RISK_ASSESSMENT_REPORTS_COLLECTION,
                        document_id=task_input.praa_risk_report_doc_id
                    )
                    if doc_output and doc_output.status == "SUCCESS" and doc_output.content:
                        praa_risk_report_content = str(doc_output.content)
                    else:
                        logger_instance.warning(f"PRAA risk report {task_input.praa_risk_report_doc_id} not found or content empty. Status: {doc_output.status if doc_output else 'N/A'}")

                if task_input.praa_optimization_report_doc_id:
                    doc_output: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                        base_collection_name=OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION,
                        document_id=task_input.praa_optimization_report_doc_id
                    )
                    if doc_output and doc_output.status == "SUCCESS" and doc_output.content:
                        praa_optimization_report_content = str(doc_output.content)
                    else:
                        logger_instance.warning(f"PRAA optimization report {task_input.praa_optimization_report_doc_id} not found or content empty. Status: {doc_output.status if doc_output else 'N/A'}")

                if task_input.rta_report_doc_id:
                    doc_output: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                        base_collection_name=TRACEABILITY_REPORTS_COLLECTION,
                        document_id=task_input.rta_report_doc_id
                    )
                    if doc_output and doc_output.status == "SUCCESS" and doc_output.content:
                        rta_report_content = str(doc_output.content)
                    else:
                        logger_instance.warning(f"RTA report {task_input.rta_report_doc_id} not found or content empty. Status: {doc_output.status if doc_output else 'N/A'}")
                
                if task_input.blueprint_reviewer_report_doc_id:
                    doc_output: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                        base_collection_name=REVIEW_REPORTS_COLLECTION,
                        document_id=task_input.blueprint_reviewer_report_doc_id
                    )
                    if doc_output and doc_output.status == "SUCCESS" and doc_output.content:
                        blueprint_review_content = str(doc_output.content)
                    else:
                        logger_instance.warning(f"Blueprint reviewer report {task_input.blueprint_reviewer_report_doc_id} not found or content empty. Status: {doc_output.status if doc_output else 'N/A'}")

            except Exception as e_reports:
                logger_instance.error(f"Exception while retrieving one or more supporting reports: {e_reports}", exc_info=True)
                # ARCA will proceed with potentially missing info; decision logic must handle None content.
                pass

        # --- NEW: Evaluate Optimization Suggestions using LLM ---
        evaluated_optimizations_from_llm: Optional[List[Dict[str, Any]]] = None
        llm_optimization_evaluation_summary: Optional[str] = None

        if llm_provider_instance and prompt_manager_instance and (praa_optimization_report_content or blueprint_review_content):
            if not main_artifact_content:
                logger_instance.warning("Main artifact content not available, skipping LLM optimization evaluation as context is missing.")
            else:
                try:
                    optimization_eval_inputs = {
                        "artifact_type": task_input.artifact_type.value,
                        "artifact_content_summary": main_artifact_content[:4000], # Truncate for context window
                        "praa_optimization_report": praa_optimization_report_content,
                        "blueprint_reviewer_optimization_report": blueprint_review_content,
                        "current_project_goal_summary": full_context.get("project_goal_summary", "No project goal summary provided.") if full_context else "No project goal summary provided."
                    }

                    logger_instance.info(f"Invoking LLM for optimization evaluation with prompt: {ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME}")
                    
                    # Log Sub-Agent (LLM call for optimization eval) Invocation Start
                    if pcma_instance:
                        await self._log_event_to_pcma(
                            pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                            arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_START",
                            event_details={
                                "invoked_agent_id": "LLM_OptimizationEvaluator",
                                "prompt_name": ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME,
                                "input_summary": {
                                    "artifact_type": optimization_eval_inputs["artifact_type"],
                                    "has_praa_report": bool(optimization_eval_inputs["praa_optimization_report"]),
                                    "has_blueprint_reviewer_report": bool(optimization_eval_inputs["blueprint_reviewer_optimization_report"])
                                }
                            },
                            severity="INFO"
                        )

                    llm_response_raw = await llm_provider_instance.instruct_direct_async(
                        prompt_name=ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME,
                        prompt_manager=prompt_manager_instance,
                        input_vars=optimization_eval_inputs,
                        # model_name, temperature, etc., could be configured here if needed
                    )

                    if llm_response_raw:
                        logger_instance.debug(f"LLM Optimization Evaluation Raw Response: {llm_response_raw}")
                        try:
                            llm_response_json = json.loads(llm_response_raw)
                            evaluated_optimizations_from_llm = llm_response_json.get("evaluated_optimizations")
                            llm_optimization_evaluation_summary = llm_response_json.get("overall_summary_of_actions")
                            logger_instance.info(f"Successfully parsed LLM optimization evaluation. Summary: {llm_optimization_evaluation_summary}")
                            if pcma_instance: # Log success
                                await self._log_event_to_pcma(
                                    pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                    arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                                    event_details={
                                        "invoked_agent_id": "LLM_OptimizationEvaluator",
                                        "status": "SUCCESS_PARSED",
                                        "summary": llm_optimization_evaluation_summary
                                    },
                                    severity="INFO"
                                )
                        except json.JSONDecodeError as e_json:
                            logger_instance.error(f"Failed to parse JSON from LLM optimization evaluation response: {e_json}. Response: {llm_response_raw}", exc_info=True)
                            if pcma_instance: # Log failure
                                await self._log_event_to_pcma(
                                    pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                    arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                                    event_details={
                                        "invoked_agent_id": "LLM_OptimizationEvaluator",
                                        "status": "FAILURE_JSON_PARSE_ERROR",
                                        "error": str(e_json)
                                    },
                                    severity="ERROR"
                                )
                    else:
                        logger_instance.warning("LLM optimization evaluation returned an empty response.")
                        if pcma_instance: # Log empty response
                             await self._log_event_to_pcma(
                                pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                                event_details={
                                    "invoked_agent_id": "LLM_OptimizationEvaluator",
                                    "status": "FAILURE_EMPTY_RESPONSE"
                                },
                                severity="WARNING"
                            )

                except Exception as e_llm_opt_eval:
                    logger_instance.error(f"Error during LLM optimization evaluation: {e_llm_opt_eval}", exc_info=True)
                    if pcma_instance: # Log general error
                        await self._log_event_to_pcma(
                            pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                            arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                            event_details={
                                "invoked_agent_id": "LLM_OptimizationEvaluator",
                                "status": "FAILURE_EXCEPTION",
                                "error": str(e_llm_opt_eval)
                            },
                            severity="ERROR"
                        )
        elif not (praa_optimization_report_content or blueprint_review_content):
            logger_instance.info("No optimization reports provided (PRAA or Blueprint Reviewer), skipping LLM optimization evaluation.")
        elif not llm_provider_instance or not prompt_manager_instance:
            logger_instance.warning("LLMProvider or PromptManager not available, skipping LLM optimization evaluation.")


        # --- Decision Logic --- #
        # This is where the original decision logic based on confidence scores starts.
        # It will now be enhanced by the evaluated_optimizations_from_llm and llm_optimization_evaluation_summary.
        # For example, the `reasoning` can be augmented, and `refinement_instructions_for_agent` can be made more specific.
        
        # Calculate combined metric (existing logic - can be kept or adapted)
        # ... (existing confidence score calculation logic) ...

        # Initialize with existing reasoning or modify if LLM provided a summary
        if llm_optimization_evaluation_summary:
            reasoning = f"Optimization Evaluation: {llm_optimization_evaluation_summary}. " + reasoning
        
        # ... (rest of the existing decision logic: if task_input.artifact_type == "CodeModule_TestFailure": ...)
        # The key change will be within the "else" block where LOPRD, Blueprint, Plan, CodeModule are handled
        # for refinement based on combined_metric.

        logger_instance.info(f"ARCA handling CodeModule_TestFailure for: {task_input.code_module_file_path or task_input.artifact_doc_id}")
        
        faulty_code_path_for_input = task_input.code_module_file_path
        faulty_code_content_for_input: Optional[str] = None
        failed_test_reports = task_input.failed_test_report_details

        # Determine which PCMA instance to use for fetching code
        pcma_for_code_retrieval = self._project_chroma_manager if self._project_chroma_manager else pcma_instance
        try:
            if task_input.praa_risk_report_doc_id:
                doc_output: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                    base_collection_name=RISK_ASSESSMENT_REPORTS_COLLECTION,
                    document_id=task_input.praa_risk_report_doc_id
                )
                if doc_output and doc_output.status == "SUCCESS" and doc_output.content:
                    praa_risk_report_content = str(doc_output.content)
                else:
                    logger_instance.warning(f"PRAA risk report {task_input.praa_risk_report_doc_id} not found/empty. Status: {doc_output.status if doc_output else 'N/A'}")

            if task_input.praa_optimization_report_doc_id:
                doc_output: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                    base_collection_name=OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION,
                    document_id=task_input.praa_optimization_report_doc_id
                )
                if doc_output and doc_output.status == "SUCCESS" and doc_output.content:
                    praa_optimization_report_content = str(doc_output.content)
                else:
                    logger_instance.warning(f"PRAA optimization report {task_input.praa_optimization_report_doc_id} not found/empty. Status: {doc_output.status if doc_output else 'N/A'}")

            if task_input.rta_report_doc_id:
                doc_output: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                    base_collection_name=TRACEABILITY_REPORTS_COLLECTION,
                    document_id=task_input.rta_report_doc_id
                )
                if doc_output and doc_output.status == "SUCCESS" and doc_output.content:
                    rta_report_content = str(doc_output.content)
                else:
                    logger_instance.warning(f"RTA report {task_input.rta_report_doc_id} not found/empty. Status: {doc_output.status if doc_output else 'N/A'}")
            
            if task_input.blueprint_reviewer_report_doc_id:
                doc_output: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                    base_collection_name=REVIEW_REPORTS_COLLECTION,
                    document_id=task_input.blueprint_reviewer_report_doc_id
                )
                if doc_output and doc_output.status == "SUCCESS" and doc_output.content:
                    blueprint_review_content = str(doc_output.content)
                else:
                    logger_instance.warning(f"Blueprint reviewer report {task_input.blueprint_reviewer_report_doc_id} not found/empty. Status: {doc_output.status if doc_output else 'N/A'}")

        except Exception as e_reports:
            logger_instance.error(f"Failed to retrieve one or more supporting reports: {e_reports}", exc_info=True)
            pass # ARCA proceeds with missing info, decision logic handles None content.

        # --- Decision Logic --- #
        logger_instance.info(f"ARCA handling CodeModule_TestFailure for: {task_input.code_module_file_path or task_input.artifact_doc_id}")
        
        faulty_code_path_for_input = task_input.code_module_file_path
        faulty_code_content_for_input: Optional[str] = None
        failed_test_reports = task_input.failed_test_report_details

        # Determine which PCMA instance to use for fetching code
        pcma_for_code_retrieval = self._project_chroma_manager if self._project_chroma_manager else pcma_instance
        if not pcma_for_code_retrieval:
            logger_instance.error(f"ARCA: PCMA instance (pcma_for_code_retrieval) is not available for CodeModule_TestFailure handling of {task_input.code_module_file_path or task_input.artifact_doc_id}. Cannot proceed.")
            # ... (error handling as before)
            pass # Return error ARCAOutput
        
        # Fetch faulty code if path is not directly available but doc_id is
        if not faulty_code_content_for_input and task_input.artifact_doc_id:
            try:
                code_doc_output: RetrieveArtifactOutput = await pcma_for_code_retrieval.retrieve_artifact(
                    base_collection_name=GENERATED_CODE_ARTIFACTS_COLLECTION,
                    document_id=task_input.artifact_doc_id
                )
                if code_doc_output and code_doc_output.status == "SUCCESS" and code_doc_output.content:
                    faulty_code_content_for_input = str(code_doc_output.content)
                    logger_instance.info(f"Retrieved faulty code content from doc_id: {task_input.artifact_doc_id}")
                    if not faulty_code_path_for_input:
                         logger_instance.warning(f"Code content retrieved for {task_input.artifact_doc_id} but no file path provided to ARCA. Debugger might be limited.")
                else:
                    logger_instance.error(f"Failed to retrieve faulty code content for doc_id {task_input.artifact_doc_id}. Status: {code_doc_output.status if code_doc_output else 'N/A'}")
                    return ARCAOutput( # Error Output
                        task_id=task_input.task_id,
                        project_id=task_input.project_id,
                        reviewed_artifact_doc_id=task_input.artifact_doc_id or "unknown_artifact_id",
                        reviewed_artifact_type="CodeModule",
                        decision="ERROR",
                        reasoning=f"Failed to retrieve source code for debugging from doc_id {task_input.artifact_doc_id}. Status: {code_doc_output.status if code_doc_output else 'N/A'}",
                        error_message=f"Failed to retrieve source code for debugging. Status: {code_doc_output.status if code_doc_output else 'N/A'}"
                    )
            except Exception as e_fetch_code:
                logger_instance.error(f"Exception fetching faulty code {task_input.artifact_doc_id}: {e_fetch_code}", exc_info=True)
                return ARCAOutput( # Error Output
                    task_id=task_input.task_id,
                    project_id=task_input.project_id,
                    reviewed_artifact_doc_id=task_input.artifact_doc_id or "unknown_artifact_id",
                    reviewed_artifact_type="CodeModule",
                    decision="ERROR",
                    reasoning=f"Exception fetching source code for debugging from doc_id {task_input.artifact_doc_id}: {e_fetch_code}",
                    error_message=str(e_fetch_code)
                )
        
        # If after attempting fetch, we still don't have content or path, it's an error for debugging
        if not faulty_code_content_for_input and not faulty_code_path_for_input:
            err_msg = "Cannot proceed with debugging: Neither code content nor a valid file path for the faulty code is available."
            logger_instance.error(err_msg)
            return ARCAOutput(
                task_id=task_input.task_id, project_id=task_input.project_id,
                reviewed_artifact_doc_id=task_input.artifact_doc_id or "unknown_code_module",
                reviewed_artifact_type="CodeModule_TestFailure",
                decision="ERROR", reasoning=err_msg, error_message=err_msg
            )

        # Get LOPRD and Blueprint IDs from full_context (these will be passed to CodeDebuggingAgent)
        relevant_loprd_ids = full_context.get("relevant_loprd_ids_for_debug", [])
        relevant_blueprint_ids = full_context.get("relevant_blueprint_ids_for_debug", [])
        logger_instance.info(f"LOPRD IDs for debugger: {relevant_loprd_ids}, Blueprint IDs for debugger: {relevant_blueprint_ids}")

        previous_debugging_attempts_from_context = full_context.get("previous_debugging_attempts", [])
        logger_instance.info(f"Received previous_debugging_attempts from context: {previous_debugging_attempts_from_context}")

        debugging_task_input = DebuggingTaskInput(
            project_id=task_input.project_id, # Pass project_id
            faulty_code_path=faulty_code_path_for_input,
            faulty_code_content=faulty_code_content_for_input,
            failed_test_reports=failed_test_reports,
            relevant_loprd_requirements_ids=relevant_loprd_ids, # Pass IDs only
            relevant_blueprint_section_ids=relevant_blueprint_ids, # Pass IDs only
            previous_debugging_attempts=previous_debugging_attempts_from_context,
            max_iterations_for_this_call=full_context.get("max_debug_iterations_per_call", 3),
            cycle_id=task_input.cycle_id # ADDED THIS LINE
        )

        # --- Implement Task 3.3.6: Max Debugging Attempts Logic ---
        current_attempt_count = len(debugging_task_input.previous_debugging_attempts)
        max_attempts_for_module = full_context.get("max_total_debugging_attempts_for_module", self.MAX_DEBUGGING_ATTEMPTS_PER_MODULE)

        if current_attempt_count >= max_attempts_for_module:
            decision = "REFINEMENT_REQUIRED" # Or a more specific status like "ESCALATION_MAX_ATTEMPTS_REACHED"
            reasoning = f"Max debugging attempts ({max_attempts_for_module}) reached for module {faulty_code_path_for_input}. Escalating for human review. Last known failed tests: {failed_test_reports}"
            logger_instance.warning(reasoning)
            
            if pcma_instance:
                await self._log_event_to_pcma(
                    pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                    arca_task_id=task_input.task_id, event_type="MAX_DEBUG_ATTEMPTS_REACHED",
                    event_details={"module_path": faulty_code_path_for_input, "attempts": current_attempt_count, "max_attempts": max_attempts_for_module},
                    severity="WARNING", related_doc_ids=[faulty_code_path_for_input]
                )
            
            # The `issues_for_human_review_list` will be populated later if state_manager is used.
            # Skip debugger invocation and proceed to output generation for ARCA.
            # Ensure code_debugging_agent_output remains None or is handled appropriately if this path is taken before its assignment.
            code_debugging_agent_output = None # Explicitly set to None as we are bypassing debugger
        else:
            logger_instance.info(f"Debugging attempt {current_attempt_count + 1} of {max_attempts_for_module} for {faulty_code_path_for_input}.")
            # Proceed with debugger invocation (Task 3.3.4)
        
        # Task 3.3.4: Implement invocation of CodeDebuggingAgent_v1
        # This block will only run if max_attempts hasn't been reached.
        # -------------------------------------------------------
        if current_attempt_count < max_attempts_for_module: # Check again before invoking
            code_debugging_agent_output: Optional[DebuggingTaskOutput] = None
            try:
                # Use the pre-initialized CodeDebuggingAgent_v1 instance
                if not self._code_debugging_agent_instance:
                    logger_instance.error("CodeDebuggingAgent_v1 instance not initialized in ARCA. Cannot proceed with debugging.")
                    # Log failure to PCMA if pcma_instance is available
                    if pcma_instance:
                        await self._log_event_to_pcma(
                            pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                            arca_task_id=task_input.task_id, event_type="ARCA_INTERNAL_ERROR",
                            event_details={"error": "CodeDebuggingAgent_v1 instance not available in ARCA"}, severity="ERROR"
                        )
                    raise ValueError("CodeDebuggingAgent_v1 instance not available in ARCA. Debugging cannot be performed.")

                code_debugging_agent_to_use = self._code_debugging_agent_instance
                
                # Task 3.3.4.2: Asynchronously call invoke_async
                logger_instance.info(f"Invoking {code_debugging_agent_to_use.AGENT_ID} with input: {debugging_task_input.model_dump_json(indent=2)}")
                if pcma_instance:
                    await self._log_event_to_pcma(
                        pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_START",
                        event_details={
                            "invoked_agent_id": code_debugging_agent_to_use.AGENT_ID,
                            "input_summary": {
                                "faulty_code_path": debugging_task_input.faulty_code_path,
                                "num_failed_tests": len(debugging_task_input.failed_test_reports),
                                "num_prev_attempts": len(debugging_task_input.previous_debugging_attempts)
                            }
                        },
                        related_doc_ids=[debugging_task_input.faulty_code_path]
                    )
                
                code_debugging_agent_output = await code_debugging_agent_to_use.invoke_async(
                    task_input=debugging_task_input,
                    full_context=full_context
                )
                logger_instance.info(f"{code_debugging_agent_to_use.AGENT_ID} output: {code_debugging_agent_output.model_dump_json(indent=2)}")

                if pcma_instance:
                    await self._log_event_to_pcma(
                        pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                        event_details={
                            "invoked_agent_id": code_debugging_agent_to_use.AGENT_ID,
                            "output_status": code_debugging_agent_output.status,
                            "output_confidence": code_debugging_agent_output.confidence_score,
                            "proposed_solution_type": code_debugging_agent_output.proposed_solution_type
                        },
                        severity="INFO" if "SUCCESS" in code_debugging_agent_output.status else "WARNING"
                    )

            except Exception as e:
                # Task 3.3.4.3: Implement try/except block for debugger invocation
                logger_instance.error(f"Error invoking CodeDebuggingAgent_v1: {e}", exc_info=True)
                if pcma_instance:
                    await self._log_event_to_pcma(
                        pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END", # Still end of invocation attempt
                        event_details={"invoked_agent_id": code_debugging_agent_to_use.AGENT_ID, "error": str(e)},
                        severity="ERROR"
                    )
                decision = "REFINEMENT_REQUIRED" # Default decision on error
                reasoning = f"Error during invocation of CodeDebuggingAgent_v1: {e}. Debugging attempt failed."
                # code_debugging_agent_output will remain None or be the partial error output from agent if it handled it.
                # If the agent itself had an internal error and returned a DebuggingTaskOutput with ERROR_INTERNAL,
                # that will be handled in the next step (3.3.5).
                # This catch block is for errors in ARCA trying to call the debugger.

        # Placeholder for Task 3.3.5 (Output Processing)
        if code_debugging_agent_output: # If invocation was successful (even if debugger reported an issue)
            # Process code_debugging_agent_output here (Task 3.3.5)
            if code_debugging_agent_output.status == "SUCCESS_FIX_PROPOSED" and \
               code_debugging_agent_output.proposed_code_changes and \
               (code_debugging_agent_output.confidence_score or 0.0) >= full_context.get("min_debug_fix_confidence", 0.7): # Configurable threshold
                
                logger_instance.info(f"ARCA: CodeDebuggingAgent_v1 proposed a fix with confidence {code_debugging_agent_output.confidence_score}. Attempting integration.")
                
                # 1. Prepare input for SmartCodeIntegrationAgent_v1
                integration_input = CodeIntegrationTaskInput(
                    project_id=task_input.project_id,
                    cycle_id=task_input.cycle_id,
                    target_file_path=debugging_task_input.faulty_code_path, # from the debugger's input
                    code_changes=code_debugging_agent_output.proposed_code_changes,
                    solution_type=code_debugging_agent_output.proposed_solution_type
                )
                
                integration_output: Optional[CodeIntegrationTaskOutput] = None
                try:
                    if not agent_resolver:
                        logger_instance.error("AgentResolver not found for SmartCodeIntegrationAgent_v1.")
                        raise ValueError("AgentResolver not available.")
                    
                    integration_agent_id = "SmartCodeIntegrationAgent_v1" # Assume this is the ID
                    integration_agent = await agent_resolver.resolve_agent_async(integration_agent_id, full_context=full_context)
                    if not integration_agent:
                        logger_instance.error(f"Failed to resolve {integration_agent_id}")
                        if pcma_instance:
                            await self._log_event_to_pcma(
                                pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                arca_task_id=task_input.task_id, event_type="ARCA_INTERNAL_ERROR",
                                event_details={"error": f"Failed to resolve agent: {integration_agent_id}"}, severity="ERROR"
                            )
                        raise ValueError(f"Could not resolve {integration_agent_id}")

                    logger_instance.info(f"Invoking {integration_agent_id} with: {integration_input.model_dump_json(indent=2)}")
                    if pcma_instance:
                         await self._log_event_to_pcma(
                            pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                            arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_START",
                            event_details={"invoked_agent_id": integration_agent_id, "target_file": integration_input.target_file_path},
                            related_doc_ids=[integration_input.target_file_path]
                        )
                    integration_output = await integration_agent.invoke_async(integration_input, full_context)
                    logger_instance.info(f"{integration_agent_id} output: {integration_output.model_dump_json(indent=2)}")
                    if pcma_instance:
                        await self._log_event_to_pcma(
                            pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                            arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                            event_details={"invoked_agent_id": integration_agent_id, "output_status": integration_output.status, "integrated_file_path": integration_output.integrated_file_path},
                            severity="INFO" if integration_output.status == "SUCCESS_APPLIED" else "ERROR"
                        )

                except Exception as e:
                    logger_instance.error(f"Error invoking {integration_agent_id}: {e}", exc_info=True)
                    if pcma_instance:
                         await self._log_event_to_pcma(
                            pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                            arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                            event_details={"invoked_agent_id": integration_agent_id, "error": str(e)}, severity="ERROR"
                        )
                    # integration_output will be None, handled below

                if integration_output and integration_output.status == "SUCCESS_APPLIED":
                    logger_instance.info("ARCA: Code integration successful. Proceeding to re-test.")
                    
                    # 2. Prepare input for SystemTestRunnerAgent_v1
                    test_runner_input = TestRunnerTaskInput(
                        project_id=task_input.project_id,
                        cycle_id=task_input.cycle_id,
                        code_module_file_path=integration_output.integrated_file_path or debugging_task_input.faulty_code_path,
                        # Potentially run only the previously failed tests or all relevant ones
                        specific_tests_to_run=[ft.test_name for ft in debugging_task_input.failed_test_reports],
                        run_all_tests_for_module=False # Be specific first
                    )
                    
                    test_runner_output: Optional[TestRunnerTaskOutput] = None
                    try:
                        if not agent_resolver:
                            logger_instance.error("AgentResolver not found for SystemTestRunnerAgent_v1.")
                            raise ValueError("AgentResolver not available.")

                        test_runner_agent_id = "SystemTestRunnerAgent_v1" # Assume this is the ID
                        test_runner_agent = await agent_resolver.resolve_agent_async(test_runner_agent_id, full_context=full_context)
                        if not test_runner_agent:
                            logger_instance.error(f"Failed to resolve {test_runner_agent_id}")
                            if pcma_instance:
                                await self._log_event_to_pcma(
                                    pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                    arca_task_id=task_input.task_id, event_type="ARCA_INTERNAL_ERROR",
                                    event_details={"error": f"Failed to resolve agent: {test_runner_agent_id}"}, severity="ERROR"
                                )
                            raise ValueError(f"Could not resolve {test_runner_agent_id}")
                        
                        logger_instance.info(f"Invoking {test_runner_agent_id} with: {test_runner_input.model_dump_json(indent=2)}")
                        if pcma_instance:
                            await self._log_event_to_pcma(
                                pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_START",
                                event_details={"invoked_agent_id": test_runner_agent_id, "module_path": test_runner_input.code_module_file_path},
                                related_doc_ids=[test_runner_input.code_module_file_path]
                            )
                        test_runner_output = await test_runner_agent.invoke_async(test_runner_input, full_context)
                        logger_instance.info(f"{test_runner_agent_id} output: {test_runner_output.model_dump_json(indent=2)}")
                        if pcma_instance:
                            await self._log_event_to_pcma(
                                pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                                event_details={
                                    "invoked_agent_id": test_runner_agent_id, 
                                    "output_status": test_runner_output.status, 
                                    "tests_passed": test_runner_output.passed_tests_count,
                                    "tests_failed": test_runner_output.failed_tests_count
                                },
                                severity="INFO" if test_runner_output.status == "SUCCESS_ALL_PASSED" else "WARNING"
                            )

                    except Exception as e:
                        logger_instance.error(f"Error invoking {test_runner_agent_id}: {e}", exc_info=True)
                        if pcma_instance:
                            await self._log_event_to_pcma(
                                pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                                event_details={"invoked_agent_id": test_runner_agent_id, "error": str(e)}, severity="ERROR"
                            )
                        # test_runner_output will be None, handled below
                    
                    if test_runner_output and test_runner_output.status == "SUCCESS_ALL_PASSED":
                        decision = "ACCEPT_ARTIFACT" # The code module is now considered fixed
                        reasoning = f"CodeDebuggingAgent proposed a fix for {debugging_task_input.faulty_code_path}, SmartCodeIntegrationAgent applied it, and SystemTestRunnerAgent confirmed all relevant tests passed."
                        final_doc_id_on_accept = debugging_task_input.faulty_code_path # Or a new version ID if PCMA versions artifacts
                        logger_instance.info(reasoning)
                    elif test_runner_output and test_runner_output.status == "FAILURE_TESTS_FAILED":
                        decision = "REFINEMENT_REQUIRED"
                        # TODO: ARCA should log these new failures to PCMA and potentially pass them to CodeDebuggingAgent in a subsequent attempt
                        new_failures_count = test_runner_output.failed_tests_count
                        reasoning = f"Applied fix for {debugging_task_input.faulty_code_path}, but {new_failures_count} test(s) still failed. Further debugging needed. New failed reports: {test_runner_output.failed_test_reports}"
                        # Potentially update task_input.failed_test_report_details for a retry loop if ARCA handles that.
                        # For now, just reporting.
                        logger_instance.warning(reasoning)
                    else: # Test runner failed or other error
                        decision = "REFINEMENT_REQUIRED"
                        reasoning = f"Applied fix for {debugging_task_input.faulty_code_path}, but SystemTestRunnerAgent failed or had an issue: {test_runner_output.message if test_runner_output else 'Unknown runner error'}. Manual review or retry needed."
                        logger_instance.error(reasoning)
                
                elif integration_output: # Integration failed
                    decision = "REFINEMENT_REQUIRED"
                    reasoning = f"CodeDebuggingAgent proposed a fix, but SmartCodeIntegrationAgent failed to apply it: {integration_output.message}. Debugging attempt failed. Error: {integration_output.error_details}"
                    logger_instance.error(reasoning)
                else: # integration_agent invocation failed
                    decision = "REFINEMENT_REQUIRED"
                    reasoning = "CodeDebuggingAgent proposed a fix, but SmartCodeIntegrationAgent invocation failed. Debugging attempt aborted."
                    logger_instance.error(reasoning)

            elif not reasoning: # If an error occurred above and reasoning wasn't set (debugger invocation failed)
                 decision = "REFINEMENT_REQUIRED"
                 reasoning = "CodeDebuggingAgent_v1 invocation failed before returning output."
                 logger_instance.error(reasoning)

            # Fall through to the confidence calculation and state update logic below

        elif combined_metric >= self.DEFAULT_ACCEPTANCE_THRESHOLD and \
           generator_confidence_val >= self.MIN_GENERATOR_CONFIDENCE and \
           praa_confidence_val >= self.MIN_PRAA_CONFIDENCE and \
           (task_input.artifact_type == "LOPRD" or rta_confidence_val >= self.MIN_RTA_CONFIDENCE):
            decision = "ACCEPT_ARTIFACT"
            reasoning = f"Artifact accepted based on confidence scores exceeding threshold ({self.DEFAULT_ACCEPTANCE_THRESHOLD:.2f}). Combined: {combined_metric:.2f}."
            final_doc_id_on_accept = task_input.artifact_doc_id
            logger_instance.info(f"ARCA Decision: ACCEPT_ARTIFACT for {task_input.artifact_doc_id} ({task_input.artifact_type}). Reasoning: {reasoning}")
        else:
            decision = "REFINEMENT_REQUIRED"
            reasoning = f"Refinement deemed necessary. Combined metric {combined_metric:.2f} or individual scores below thresholds."
            logger_instance.info(f"ARCA Decision: REFINEMENT_REQUIRED for {task_input.artifact_doc_id} ({task_input.artifact_type}). Reasoning: {reasoning}")
            
            # --- Determine which agent to call for refinement ---
            # This is a simplified placeholder. Actual logic would be more nuanced, 
            # potentially looking at which report (PRAA, RTA) triggered the low score.
            refinement_instructions_for_agent = f"Refinement requested by ARCA for {task_input.artifact_type.value} (ID: {task_input.artifact_doc_id}). Key concerns based on PRAA/RTA reports and overall confidence. Please review and address."
            if generator_agent_confidence_val is not None: # Add generator confidence if available
                refinement_instructions_for_agent += f" Initial generator confidence: {generator_agent_confidence_val:.2f}."
            if praa_confidence_val is not None:
                refinement_instructions_for_agent += f" PRAA confidence: {praa_confidence_val:.2f}."
            if rta_confidence_val is not None and task_input.artifact_type != ARCAReviewArtifactType.LOPRD:
                refinement_instructions_for_agent += f" RTA confidence: {rta_confidence_val:.2f}."
            
            # --- NEW: Incorporate specific optimization instructions from LLM --- #
            specific_optimization_instructions_list = []
            if evaluated_optimizations_from_llm:
                for opt_eval in evaluated_optimizations_from_llm:
                    if opt_eval.get("recommendation") == "INCORPORATE" and opt_eval.get("incorporation_instructions_for_next_agent"):
                        specific_optimization_instructions_list.append(
                            f"- Incorporate Optimization (ID: {opt_eval.get('optimization_id', 'N/A')} from {opt_eval.get('source_report', 'N/A')}): {opt_eval['incorporation_instructions_for_next_agent']}"
                        )
            
            if specific_optimization_instructions_list:
                specific_instructions_str = "\n\nSpecific Optimizations to Incorporate:\n" + "\n".join(specific_optimization_instructions_list)
                refinement_instructions_for_agent += specific_instructions_str
                reasoning += f" Identified {len(specific_optimization_instructions_list)} specific optimization(s) for incorporation."
            elif evaluated_optimizations_from_llm: # Optimizations were evaluated but none to incorporate
                reasoning += " LLM evaluated optimization reports, but no specific 'INCORPORATE' actions were advised at this stage."
            # If evaluated_optimizations_from_llm is None, it means LLM opt eval didn't run or failed, so no change to reasoning here.

            # Update the main reasoning variable for ARCA's output
            reasoning = reasoning

            logger_instance.info(f"ARCA Decision: REFINEMENT_REQUIRED for {task_input.artifact_doc_id} ({task_input.artifact_type.value}). Reasoning: {reasoning}")
            logger_instance.debug(f"Final refinement instructions for next agent: {refinement_instructions_for_agent}")

            # Mock: Get user_goal from full_context if available, needed by PAA
            initial_user_goal_for_paa = "User goal not available in ARCA context for refinement input."
            if full_context and full_context.get("intermediate_outputs", {}).get("initial_goal_setup", {}).get("initial_user_goal"):
                initial_user_goal_for_paa = full_context["intermediate_outputs"]["initial_goal_setup"]["initial_user_goal"]
            elif full_context and full_context.get("initial_user_goal"): # If it was in root context
                 initial_user_goal_for_paa = full_context["initial_user_goal"]

            if task_input.artifact_type == "LOPRD":
                next_agent_for_refinement = pa_module.ProductAnalystAgent_v1.AGENT_ID
                next_agent_refinement_input = pa_module.ProductAnalystAgentInput(
                    project_id=task_input.project_id,
                    refined_user_goal_doc_id=task_input.artifact_doc_id, # Using the LOPRD being reviewed as the base for refinement
                    assumptions_and_ambiguities_doc_id=task_input.assumptions_doc_id, # This was retrieved earlier
                    arca_feedback_doc_id=task_input.feedback_doc_id, # This was set earlier
                    shared_context=task_input.shared_context, # Pass along
                    cycle_id=task_input.cycle_id # ADDED THIS LINE
                )
            elif task_input.artifact_type == "Blueprint":
                next_agent_for_refinement = ArchitectAgentInput.AGENT_ID
                # Architect needs LOPRD ID. Assume it's retrievable or passed through context correctly if not already refined.
                # For this mock, we'll assume the LOPRD that led to this blueprint is somehow known or we re-use the artifact_doc_id if it implies a prior step output.
                # This highlights a dependency: ARCA needs to know the LOPRD for blueprint refinement by Architect.
                # This is a gap if ARCA only gets the blueprint_doc_id. The orchestrator's context flow is key here.
                # For MVP, we assume Architect agent is smart enough or LOPRD_ID is passed in full_context if it was an earlier stage.
                # Let's assume for the MasterPlan, the LOPRD id would be available in context.
                loprd_id_for_architect = "unknown_loprd_id_for_blueprint_refinement" # Placeholder
                if full_context and full_context.get("intermediate_outputs", {}).get("arca_loprd_coordination_output", {}).get("final_artifact_doc_id"):
                    loprd_id_for_architect = full_context["intermediate_outputs"]["arca_loprd_coordination_output"]["final_artifact_doc_id"]
                elif full_context and full_context.get("current_loprd_doc_id"): # Fallback if in root
                    loprd_id_for_architect = full_context["current_loprd_doc_id"]

                next_agent_refinement_input = ArchitectAgentInput(
                    project_id=task_input.project_id,
                    loprd_doc_id=loprd_id_for_architect, 
                    existing_blueprint_doc_id=task_input.artifact_doc_id,
                    refinement_instructions=refinement_instructions_for_agent,
                    cycle_id=task_input.cycle_id # ADDED THIS LINE
                )
            elif task_input.artifact_type == "MasterExecutionPlan":
                next_agent_for_refinement = "SystemMasterPlannerAgent_v1" # Or its specific sub-capability ID
                # MasterPlanner needs blueprint ID. Similar to above, assume it's available.
                blueprint_id_for_planner = "unknown_blueprint_id_for_plan_refinement"
                if full_context and full_context.get("intermediate_outputs", {}).get("arca_blueprint_coordination_output", {}).get("final_artifact_doc_id"):
                     blueprint_id_for_planner = full_context["intermediate_outputs"]["arca_blueprint_coordination_output"]["final_artifact_doc_id"]
                elif full_context and full_context.get("current_blueprint_doc_id"): # Fallback
                     blueprint_id_for_planner = full_context["current_blueprint_doc_id"]
                
                next_agent_refinement_input = MasterPlannerInput(
                    project_id=task_input.project_id,
                    blueprint_doc_id=blueprint_id_for_planner,
                    # user_goal = None, # Not primary for plan refinement from blueprint
                    refinement_instructions=refinement_instructions_for_agent,
                    # existing_plan_doc_id = task_input.artifact_doc_id # If planner supports refining existing plan doc
                )
            elif task_input.artifact_type == "CodeModule":
                # This implies refinement for CoreCodeGeneratorAgent or similar
                # For now, we don't have a direct input schema for that in ARCA's imports
                # So we'll use a dict and assume orchestrator/agent handles it.
                next_agent_for_refinement = "CoreCodeGeneratorAgent_v1" # Or SmartCodeGeneratorAgent_v1
                next_agent_refinement_input = {
                    "project_id": task_input.project_id,
                    "code_specification_doc_id": "REFINED_SPEC_FROM_ARCA_OR_PREVIOUS_STEP", # Needs context
                    "existing_code_doc_id": task_input.artifact_doc_id,
                    "refinement_instructions": refinement_instructions_for_agent,
                    # Add other fields CoreCodeGeneratorAgentInput might need for refinement
                }
                logger_instance.warning(f"ARCA: CodeModule refinement input is a generic dict. Ensure {next_agent_for_refinement} can handle it.")
            else:
                reasoning += " But no specific refinement agent configured for this artifact type."
                # No agent to call, so it's effectively an error or unhandled state for refinement
                logger_instance.error(f"ARCA: Refinement needed for {task_input.artifact_type} but no refinement path defined.")
                # This might lead to an error state in the orchestrator if not handled properly.
                # For now, we will still say REFINEMENT_REQUIRED but not specify an agent.

        # --- Confidence in ARCA's own decision --- 
        # Can be heuristic or LLM-based if ARCA used an LLM for decision.
        # For MVP rule-based, it's fairly high if rules are clear.
        arca_confidence_val = 0.9 if decision == "ACCEPT_ARTIFACT" else 0.75
        arca_confidence = ConfidenceScore(value=arca_confidence_val, level="High" if arca_confidence_val > 0.8 else "Medium", method="RuleBasedHeuristic_ARCA_MVP", reasoning=f"ARCA decision based on pre-defined confidence thresholds. Metric: {combined_metric:.2f}")

        # --- NEW: Update Project State at the end of a cycle/review point ---
        # This logic assumes an ARCA invocation can signify a point where human review is appropriate,
        # or a sub-part of a cycle completes.
        # The definition of what constitutes a full "cycle end" to be recorded in StateManager
        # might also be orchestrated by the calling process (e.g. AsyncOrchestrator)
        # For now, ARCA will attempt to update the *current* cycle if a StateManager is available.

        issues_for_human_review_list = []
        cycle_summary_for_state_manager = f"ARCA review complete for artifact {task_input.artifact_doc_id} ({task_input.artifact_type}). Decision: {decision}. Reasoning: {reasoning}."

        if decision == "REFINEMENT_REQUIRED" and not next_agent_for_refinement:
            # This case means refinement is needed but ARCA doesn't know who to send it to.
            # This is a clear issue for human review.
            issues_for_human_review_list.append({
                "issue_id": f"arca_unhandled_refinement_{task_input.artifact_doc_id}",
                "description": f"ARCA determined refinement is required for {task_input.artifact_type} (ID: {task_input.artifact_doc_id}), but no automated refinement path is defined. {reasoning}",
                "relevant_artifact_ids": [task_input.artifact_doc_id]
            })
            cycle_summary_for_state_manager += " Escalation: No automated refinement path available."


        # Condition to update state: if a state_manager is present and
        # EITHER the decision is to accept (implying a sub-task completion)
        # OR if there are issues explicitly flagged for human review by ARCA.
        # More sophisticated logic for WHEN to update state can be added.
        # For P3.1.1, we assume ARCA's processing of an artifact can be a point to update cycle status.
        
        # Let's assume for now that an ARCA run that results in ACCEPT_ARTIFACT
        # or a REFINEMENT_REQUIRED that cannot be automated should trigger a state update
        # and mark the cycle for review if there are such issues.
        
        update_state_for_review = False
        if decision == "ACCEPT_ARTIFACT" or (decision == "REFINEMENT_REQUIRED" and not next_agent_for_refinement) or decision == "PROCEED_TO_DOCUMENTATION" or (decision == "TEST_FAILURE_HANDOFF" and "Max debugging attempts" in reasoning) or decision == "ESCALATE_TO_USER": # Added escalate_to_user
            update_state_for_review = True # Good point to potentially pend review or record progress.

        if state_manager_instance and update_state_for_review:
            try:
                # Removed direct project_state manipulation here
                # Calls to state_manager methods will handle this.
                log_event_details_base = {"decision": decision, "artifact_type": task_input.artifact_type.value, "artifact_doc_id": task_input.artifact_doc_id}

                if pcma_instance:
                    await self._log_event_to_pcma(
                        pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, event_type="STATE_UPDATE_ATTEMPT",
                        event_details={**log_event_details_base, "target_operation": "complete_current_cycle & update_after_arca_review"}
                    )

                # 1. Complete the current cycle
                cycle_final_status: CycleStatus = CycleStatus.COMPLETED_WITH_ISSUES_FOR_REVIEW # Default
                arca_decision_for_cycle_completion = decision # Could be more nuanced

                if decision == "ACCEPT_ARTIFACT":
                    if task_input.artifact_type == ARCAReviewArtifactType.ProjectDocumentation:
                        cycle_final_status = CycleStatus.COMPLETED_SUCCESS 
                        # Overall project status will be handled by update_project_state_after_arca_review or orchestrator
                    else:
                        cycle_final_status = CycleStatus.COMPLETED_SUCCESS # Or a more nuanced "ARTIFACT_ACCEPTED_CYCLE_CONTINUES" if that exists
                elif decision == "ESCALATE_TO_USER":
                    cycle_final_status = CycleStatus.COMPLETED_WITH_ISSUES_FOR_REVIEW
                elif decision == "REFINEMENT_REQUIRED" and not next_agent_for_refinement:
                    cycle_final_status = CycleStatus.COMPLETED_WITH_ISSUES_FOR_REVIEW
                elif decision == "TEST_FAILURE_HANDOFF" and "Max debugging attempts" in reasoning: # Max debug attempts reached
                     cycle_final_status = CycleStatus.COMPLETED_WITH_ISSUES_FOR_REVIEW
                # Other decisions like REFINEMENT_REQUIRED (with agent) or TEST_FAILURE_HANDOFF (before max attempts)
                # might not mean the *entire* cycle is complete, but rather a sub-loop.
                # For now, if update_state_for_review is true, we assume a significant cycle milestone.

                # ARCA might generate its own summary artifact, for now, use None.
                arca_summary_doc_id_for_cycle: Optional[str] = None 

                # Ensure current_cycle_id is valid before proceeding
                project_state_check = state_manager_instance.get_project_state()
                if not project_state_check.current_cycle_id:
                    logger_instance.error("ARCA: Attempted to complete cycle, but no current_cycle_id is set in StateManager. Aborting state update for this part.")
                    raise StatusFileError("No current_cycle_id in project state for cycle completion.")

                state_manager_instance.complete_current_cycle(
                    final_status=cycle_final_status,
                    arca_summary_doc_id=arca_summary_doc_id_for_cycle,
                    arca_decision=arca_decision_for_cycle_completion,
                    issues_flagged_by_arca=issues_for_human_review_list # This was populated earlier
                )
                logger_instance.info(f"ARCA: Called state_manager.complete_current_cycle for cycle {project_state_check.current_cycle_id} with status {cycle_final_status.value}.")
                
                # 2. Update project state with ARCA's overall review (if cycle completion means pending human review)
                # This might be redundant if complete_current_cycle already sets the overall project status appropriately.
                # However, ProjectStateV2 has specific fields like arca_best_state_summary_doc_id.
                
                # For arca_overall_confidence, use ARCA's confidence in its *own current decision*
                arca_overall_confidence_for_state = arca_confidence.value if arca_confidence else None
                
                # For arca_best_state_summary_doc_id, this would be a link to an artifact ARCA itself produces summarizing the cycle.
                # If ARCA doesn't produce such a document directly, this could be None or link to the primary reviewed artifact if "accepted".
                # For now, let's use the reviewed artifact_doc_id if accepted, or None.
                summary_doc_id_for_arca_review = task_input.artifact_doc_id if decision == "ACCEPT_ARTIFACT" else None

                state_manager_instance.update_project_state_after_arca_review(
                    arca_best_state_summary_doc_id=summary_doc_id_for_arca_review,
                    arca_overall_confidence=arca_overall_confidence_for_state,
                    arca_issues_pending_human_review=issues_for_human_review_list
                )
                logger_instance.info(f"ARCA: Called state_manager.update_project_state_after_arca_review for project {task_input.project_id}.")

                # 3. Update latest accepted artifact IDs if applicable
                if decision == "ACCEPT_ARTIFACT" and task_input.artifact_doc_id:
                    artifact_type_val = task_input.artifact_type.value
                    accepted_doc_id = task_input.artifact_doc_id
                    if artifact_type_val == ARCAReviewArtifactType.LOPRD.value:
                        state_manager_instance.update_latest_accepted_loprd(accepted_doc_id)
                    elif artifact_type_val == ARCAReviewArtifactType.Blueprint.value:
                        state_manager_instance.update_latest_accepted_blueprint(accepted_doc_id)
                    elif artifact_type_val == ARCAReviewArtifactType.MasterExecutionPlan.value:
                        state_manager_instance.update_latest_accepted_master_plan(accepted_doc_id)
                    # Add cases for CodeModule (snapshot) and ProjectDocumentation if they have specific latest_accepted fields
                    # For CodeModule, it might be a snapshot ID or a path, depending on how it's tracked.
                    # For now, assume ProjectDocumentation updates a specific readme field.
                    elif artifact_type_val == ARCAReviewArtifactType.ProjectDocumentation.value:
                         state_manager_instance.update_latest_project_readme(accepted_doc_id) # Assuming this method exists
                    # else: Not all accepted artifacts might have a top-level latest_accepted_ field.
                    logger_instance.info(f"ARCA: Updated latest accepted ID for {artifact_type_val} to {accepted_doc_id}")

                update_state_log_details = {**log_event_details_base, "reason": "Cycle completion and ARCA review update.", "outcome": "SUCCESS"}
                
                if pcma_instance: # Log success
                     await self._log_event_to_pcma(
                        pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, 
                        event_type="STATE_UPDATE_SUCCESS",
                        event_details=update_state_log_details,
                        severity="INFO"
                    )

            except StatusFileError as e:
                logger_instance.error(f"ARCA: Failed to update project state due to StatusFileError: {e}", exc_info=True)
                if pcma_instance:
                    await self._log_event_to_pcma(
                        pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, event_type="STATE_UPDATE_FAILURE",
                        event_details={"error": f"StatusFileError: {str(e)}"}, severity="ERROR"
                    )
            except Exception as e: # Catch any other unexpected errors
                logger_instance.error(f"ARCA: Unexpected error updating project state: {e}", exc_info=True)
                if pcma_instance:
                    await self._log_event_to_pcma(
                        pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, event_type="STATE_UPDATE_FAILURE",
                        event_details={"error": f"Unexpected error: {str(e)}"}, severity="ERROR"
                    )
        elif not state_manager_instance and update_state_for_review:
            logger_instance.warning(
                "ARCA: StateManager instance not available, but an update was flagged. "
                "Skipping project state update."
                )
            if pcma_instance:
                await self._log_event_to_pcma(
                    pcma_agent=pcma_instance, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                    arca_task_id=task_input.task_id, event_type="STATE_UPDATE_FAILURE",
                    event_details={"reason": "StateManager instance not available for a flagged update", "outcome": "SKIPPED"},
                    severity="WARNING"
                    )
        elif not update_state_for_review and state_manager_instance: # If an update wasn't flagged
            logger_instance.info(
                f"ARCA: No project state update was flagged for this review of "
                f"artifact {task_input.artifact_doc_id or task_input.code_module_file_path} ({task_input.artifact_type}). "
                f"State remains unchanged by ARCA for this reason."
                )
            # If state_manager_instance is None AND update_state_for_review is False,
            # it means no update was intended and no manager was present; no specific log here.
        
        output_reviewed_doc_id = task_input.artifact_doc_id
        if task_input.artifact_type == "CodeModule_TestFailure" and task_input.code_module_file_path:
            output_reviewed_doc_id = task_input.artifact_doc_id or task_input.code_module_file_path
        # Fix linter error: output_reviewed__doc_id -> output_reviewed_doc_id
        output_reviewed_doc_id = output_reviewed_doc_id or code_module_path_for_output or "N/A"

        final_arca_output = ARCAOutput(
            task_id=task_input.task_id,
            project_id=task_input.project_id,
            reviewed_artifact_doc_id=output_reviewed_doc_id,
            reviewed_artifact_type=task_input.artifact_type,
            decision=decision,
            decision_reasoning=reasoning,
            confidence_in_decision=arca_confidence,
            next_agent_id_for_refinement=next_agent_for_refinement,
            next_agent_input=next_agent_refinement_input.model_dump() if next_agent_refinement_input and hasattr(next_agent_refinement_input, 'model_dump') else next_agent_refinement_input,
            final_artifact_doc_id=final_doc_id_on_accept
        )

        if pcma_agent_for_logging:
            await self._log_event_to_pcma(
                pcma_agent=pcma_agent_for_logging,
                project_id=task_input.project_id,
                cycle_id=task_input.cycle_id,
                arca_task_id=task_input.task_id,
                event_type="ARCA_DECISION_MADE",
                event_details={
                    "reviewed_artifact_doc_id": final_arca_output.reviewed_artifact_doc_id,
                    "reviewed_artifact_type": final_arca_output.reviewed_artifact_type,
                    "decision": final_arca_output.decision,
                    "reasoning_summary": final_arca_output.decision_reasoning[:200] + "..." if len(final_arca_output.decision_reasoning) > 200 else final_arca_output.decision_reasoning,
                    "next_agent_id": final_arca_output.next_agent_id_for_refinement,
                    "confidence_in_decision": final_arca_output.confidence_in_decision.model_dump() if final_arca_output.confidence_in_decision else None
                },
                related_doc_ids=[final_arca_output.reviewed_artifact_doc_id] if final_arca_output.reviewed_artifact_doc_id else [],
                severity="INFO" if final_arca_output.decision == "ACCEPT_ARTIFACT" else "WARNING"
            )
        
        # --- Construct Final ARCAOutput ---
        arca_output = ARCAOutput(
            task_id=task_input.task_id,
            project_id=task_input.project_id,
            reviewed_artifact_doc_id=task_input.artifact_doc_id if task_input.artifact_doc_id else "N/A_Generated_Doc_Task",
            reviewed_artifact_type=task_input.artifact_type, # Use the enum member directly
            decision=final_decision,
            reasoning=final_reasoning,
            confidence_in_decision=ConfidenceScore(score=decision_confidence, reasoning="ARCA internal confidence assessment."), # Example
            next_agent_id_for_refinement=next_agent_id,
            next_agent_input=next_agent_input_payload,
            debugging_task_input=debugging_input_for_output,
            final_artifact_doc_id=final_doc_id_on_accept,
            error_message=arca_error_message
        )

        # --- NEW: Store Quality Assurance Log Entry ---
        if self._project_chroma_manager:
            try:
                # Map ARCA decision to OverallQualityStatus
                quality_status_mapping = {
                    "ACCEPT_ARTIFACT": OverallQualityStatus.APPROVED_PASSED,
                    "PROCEED_TO_DOCUMENTATION": OverallQualityStatus.APPROVED_PASSED,
                    "REFINEMENT_REQUIRED": OverallQualityStatus.REJECTED_NEEDS_REFINEMENT,
                    "ESCALATE_TO_USER": OverallQualityStatus.FLAGGED_FOR_MANUAL_REVIEW,
                    "TEST_FAILURE_HANDOFF": OverallQualityStatus.REJECTED_NEEDS_REFINEMENT,
                    "ERROR": OverallQualityStatus.ERROR_IN_QA_PROCESS 
                }
                overall_quality_status = quality_status_mapping.get(final_decision, OverallQualityStatus.FLAGGED_FOR_MANUAL_REVIEW)

                action_taken_summary = f"ARCA decision: {final_decision}."
                if final_decision == "REFINEMENT_REQUIRED" and next_agent_id:
                    action_taken_summary += f" Next agent: {next_agent_id}."
                elif final_decision == "TEST_FAILURE_HANDOFF":
                    action_taken_summary += " Handoff to CodeDebuggingAgent."
                
                qa_log_entry_content = QualityAssuranceLogEntry(
                    project_id=task_input.project_id,
                    cycle_id=task_input.cycle_id,
                    artifact_doc_id_assessed=arca_output.reviewed_artifact_doc_id,
                    artifact_type_assessed=task_input.artifact_type.value, # Use .value for Enum
                    qa_event_type=QAEventType.ARCA_ARTIFACT_ASSESSMENT,
                    assessing_entity_id=self.AGENT_ID,
                    summary_of_assessment=final_reasoning[:1000], # Truncate if too long
                    overall_quality_status=overall_quality_status,
                    confidence_in_assessment=decision_confidence, # Assuming decision_confidence is ARCA's confidence
                    action_taken_or_recommended=action_taken_summary,
                    key_metrics_or_findings={
                        "generator_agent_id": task_input.generator_agent_id,
                        "generator_confidence": task_input.generator_agent_confidence.score if task_input.generator_agent_confidence else None,
                        "arca_decision_confidence": decision_confidence
                    }
                )
                
                # Use StoreArtifactInput for PCMA
                store_qa_log_input = StoreArtifactInput(
                    base_collection_name=QUALITY_ASSURANCE_LOGS_COLLECTION,
                    artifact_content=qa_log_entry_content.model_dump(mode='json'),
                    metadata={
                        "artifact_type": ARTIFACT_TYPE_QA_LOG_ENTRY_JSON,
                        "project_id": task_input.project_id,
                        "cycle_id": task_input.cycle_id,
                        "assessed_artifact_id": arca_output.reviewed_artifact_doc_id,
                        "assessed_artifact_type": task_input.artifact_type.value
                    },
                    cycle_id=task_input.cycle_id, # Pass cycle_id for lineage
                    source_agent_id=self.AGENT_ID # ARCA is the source of this log
                )

                qa_log_store_result = await self._project_chroma_manager.store_artifact(store_qa_log_input)
                if qa_log_store_result.status == "SUCCESS":
                    self._logger_instance.info(f"Successfully stored QA log entry: {qa_log_store_result.document_id} for artifact {arca_output.reviewed_artifact_doc_id}")
                else:
                    self._logger_instance.warning(f"Failed to store QA log entry for artifact {arca_output.reviewed_artifact_doc_id}. Error: {qa_log_store_result.error_message}")

            except Exception as e_qa_log:
                self._logger_instance.error(f"Exception during QA log storing: {e_qa_log}", exc_info=True)
        # --- End of NEW QA Log Storing Logic ---

        # Log ARCA Decision Made
        if self._project_chroma_manager:
            await self._log_event_to_pcma(
                pcma_agent=pcma_agent_for_logging,
                project_id=task_input.project_id,
                cycle_id=task_input.cycle_id,
                arca_task_id=task_input.task_id,
                event_type="ARCA_DECISION_MADE",
                event_details={
                    "reviewed_artifact_doc_id": final_arca_output.reviewed_artifact_doc_id,
                    "reviewed_artifact_type": final_arca_output.reviewed_artifact_type,
                    "decision": final_arca_output.decision,
                    "reasoning_summary": final_arca_output.decision_reasoning[:200] + "..." if len(final_arca_output.decision_reasoning) > 200 else final_arca_output.decision_reasoning,
                    "next_agent_id": final_arca_output.next_agent_id_for_refinement,
                    "confidence_in_decision": final_arca_output.confidence_in_decision.model_dump() if final_arca_output.confidence_in_decision else None
                },
                related_doc_ids=[final_arca_output.reviewed_artifact_doc_id] if final_arca_output.reviewed_artifact_doc_id else [],
                severity="INFO" if final_arca_output.decision == "ACCEPT_ARTIFACT" else "WARNING"
            )
        
        return final_arca_output

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