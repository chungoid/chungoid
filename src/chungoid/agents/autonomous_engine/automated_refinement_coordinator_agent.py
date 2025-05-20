from __future__ import annotations

import logging
import datetime
import uuid
import json
from typing import Any, Dict, Optional, Literal, Union, ClassVar, List

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
from chungoid.schemas.project_state import ProjectStateV2, CycleHistoryItem # CycleHistoryItem might not be directly used if modifying existing

from chungoid.utils.agent_registry import AgentCard # For AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, GENERATED_CODE_ARTIFACTS_COLLECTION # NEW - Corrected import

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

logger = logging.getLogger(__name__)

ARCA_PROMPT_NAME = "automated_refinement_coordinator_agent_v1.yaml" # If LLM-based decision making is used

# Constants for ARCA behavior
MAX_DEBUGGING_ATTEMPTS_PER_MODULE: ClassVar[int] = 3 # Added ClassVar
DEFAULT_ACCEPTANCE_THRESHOLD: ClassVar[float] = 0.85 # MODIFIED: Added ClassVar

# Define the Literal for artifact types including the new one
ARCAReviewArtifactType = Literal[
    "LOPRD",
    "ProjectBlueprint",
    "MasterExecutionPlan",
    "CodeModule",
    "RiskAssessmentReport",
    "TraceabilityReport",
    "OptimizationSuggestionReport",
    "ProjectDocumentation",
    "CodeModule_TestFailure", # Added new type
    "QA_Summary_Report" # Assuming this might exist or be added later
]

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
    reviewed_artifact_type: Literal["LOPRD", "Blueprint", "MasterExecutionPlan", "CodeModule", "ProjectDocumentation"] = Field(..., description="Type of the artifact reviewed.")
    decision: Literal["ACCEPT_ARTIFACT", "REFINEMENT_REQUIRED", "ESCALATE_TO_USER", "TEST_FAILURE_HANDOFF", "ERROR"] = Field(..., description="The final decision of the ARCA review.")
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
    _state_manager: Optional[StateManager]
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
        self._state_manager = state_manager
        
        # Initialize CodeDebuggingAgent instance
        if self._llm_provider and self._prompt_manager and self._project_chroma_manager:
            try:
                self._code_debugging_agent_instance = CodeDebuggingAgent_v1(
                    llm_provider=self._llm_provider,
                    prompt_manager=self._prompt_manager,
                    project_chroma_manager=self._project_chroma_manager,
                    system_context=system_context # Pass along system_context for logger, etc.
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
        task_input: ARCAReviewInput, # Changed from inputs: Dict[str, Any]
        full_context: Optional[Dict[str, Any]] = None,
    ) -> ARCAOutput:
        llm_provider = self._llm_provider
        prompt_manager = self._prompt_manager
        logger_instance = self._logger_instance
        state_manager_instance = self._state_manager # NEW: Get StateManager instance

        if full_context:
            # No direct LLM/Prompt use in MVP, but provision for future
            if not llm_provider and "llm_provider" in full_context: llm_provider = full_context["llm_provider"]
            if not prompt_manager and "prompt_manager" in full_context: prompt_manager = full_context["prompt_manager"]
            # NEW: Get StateManager from full_context if not already set
            if not state_manager_instance and "state_manager" in full_context:
                state_manager_instance = full_context["state_manager"]
            if "system_context" in full_context and "logger" in full_context["system_context"]:
                if full_context["system_context"]["logger"] != self._logger_instance: 
                    logger_instance = full_context["system_context"]["logger"]
            # Resolve AgentResolver if available in full_context for dynamic agent invocation
            agent_resolver = full_context.get("agent_resolver") # For resolving CodeDebuggingAgent_v1 etc.
        
        pcma_agent_for_logging: Optional[ProjectChromaManagerAgent_v1] = None
        if agent_resolver:
            try:
                # Resolve PCMA instance for logging
                # Assuming PCMA is registered with its AGENT_ID if it were a BaseAgent,
                # or we have a known way to get it if it's a direct class instantiation managed by resolver.
                # For now, let's assume it might be directly in full_context or resolved if it were a full agent.
                # If PCMA is not a BaseAgent, this resolution might differ.
                # Fallback: Check if an instance is directly passed in full_context.
                if "project_chroma_manager_agent_instance" in full_context:
                    pcma_agent_for_logging = full_context["project_chroma_manager_agent_instance"]
                elif agent_resolver: # Attempt to resolve if it's registered like other agents
                     # This assumes PCMA has an AGENT_ID and is resolvable like other BaseAgents.
                     # This might need adjustment based on how PCMA is actually managed/provided.
                    pcma_instance_candidate = await agent_resolver.resolve_agent_async(
                        ProjectChromaManagerAgent_v1.AGENT_ID if hasattr(ProjectChromaManagerAgent_v1, 'AGENT_ID') else "ProjectChromaManagerAgent_v1",
                        full_context=full_context
                    )
                    if isinstance(pcma_instance_candidate, ProjectChromaManagerAgent_v1):
                        pcma_agent_for_logging = pcma_instance_candidate
                    else:
                        logger_instance.warning("Could not resolve ProjectChromaManagerAgent_v1 via AgentResolver for logging.")
                
                if not pcma_agent_for_logging:
                    logger_instance.warning("PCMA instance not available for ARCA logging.")

            except Exception as e_pcma_resolve:
                logger_instance.error(f"Failed to resolve PCMA for logging: {e_pcma_resolve}", exc_info=True)
                pcma_agent_for_logging = None # Ensure it's None if resolution fails
        else:
            logger_instance.warning("AgentResolver not available in full_context. PCMA for logging cannot be resolved.")


        # Initial Log
        if pcma_agent_for_logging:
            await self._log_event_to_pcma(
                pcma_agent=pcma_agent_for_logging,
                project_id=task_input.project_id,
                cycle_id=task_input.cycle_id,
                arca_task_id=task_input.task_id,
                event_type="ARCA_INVOCATION_START",
                event_details={"artifact_type": task_input.artifact_type, "artifact_doc_id": task_input.artifact_doc_id or task_input.code_module_file_path, "generator_agent_id": task_input.generator_agent_id},
                related_doc_ids=[task_input.artifact_doc_id] if task_input.artifact_doc_id else ([task_input.code_module_file_path] if task_input.code_module_file_path else [])
            )

        try:
            # parsed_inputs = ARCAReviewInput(**inputs) # Removed this line
            pass # Placeholder for removed line, original logic used `parsed_inputs`
        except Exception as e:
            logger_instance.error(f"Input parsing failed: {e}", exc_info=True)
            # Ensure task_id and project_id are accessed safely if inputs was the source
            _task_id = getattr(task_input, 'task_id', 'parse_err_during_direct_use')
            _project_id = getattr(task_input, 'project_id', 'parse_err_during_direct_use')
            
            if pcma_agent_for_logging:
                await self._log_event_to_pcma(
                    pcma_agent=pcma_agent_for_logging, project_id=_project_id, cycle_id=getattr(task_input, 'cycle_id', 'unknown'),
                    arca_task_id=_task_id, event_type="ARCA_INTERNAL_ERROR",
                    event_details={"error": "Input parsing failed", "detail": str(e)}, severity="ERROR"
                )
            return ARCAOutput(task_id=_task_id, project_id=_project_id, reviewed_artifact_doc_id="parse_err", reviewed_artifact_type="LOPRD", decision="REFINEMENT_REQUIRED", decision_reasoning=f"Input parsing error: {e}", error_message=str(e))

        logger_instance.info(f"{self.AGENT_ID} invoked for task {task_input.task_id}, artifact {task_input.artifact_doc_id} ({task_input.artifact_type}) in project {task_input.project_id}")

        # --- MOCK: Retrieve PRAA/RTA report summaries (actual would involve PCMA calls) ---
        # These would be summaries or key findings, not full content for decision making here.
        # For MVP, we'll use confidence scores directly.
        praa_confidence_val = task_input.praa_confidence_score.value if task_input.praa_confidence_score else self.MIN_PRAA_CONFIDENCE
        rta_confidence_val = task_input.rta_confidence_score.value if task_input.rta_confidence_score else self.MIN_RTA_CONFIDENCE
        generator_confidence_val = task_input.generator_agent_confidence.value if task_input.generator_agent_confidence else self.MIN_GENERATOR_CONFIDENCE

        # --- Decision Logic (MVP: Rule-based based on confidence scores) ---
        decision: Literal["ACCEPT_ARTIFACT", "REFINEMENT_REQUIRED", "ESCALATE_TO_USER", "TEST_FAILURE_HANDOFF", "ERROR"] = "REFINEMENT_REQUIRED"
        reasoning = "Defaulting to refinement due to initial MVP logic or low confidence."
        next_agent_for_refinement: Optional[str] = None
        next_agent_refinement_input: Optional[Union[pa_module.ProductAnalystAgentInput, ArchitectAgentInput, MasterPlannerInput, ProjectDocumentationAgentInput, Dict[str,Any]]] = None
        final_doc_id_on_accept: Optional[str] = None

        # Simplified confidence calculation
        # In a real scenario, ARCA might look at specific content in PRAA/RTA reports.
        # For example, if PRAA reports critical risks, refinement is needed regardless of scores.
        combined_metric = generator_confidence_val * praa_confidence_val
        if task_input.artifact_type in ["Blueprint", "MasterExecutionPlan", "CodeModule"]:
            combined_metric *= rta_confidence_val

        logger_instance.info(f"ARCA Decision Logic: Artifact Type: {task_input.artifact_type}, GenConf: {generator_confidence_val:.2f}, PRAAConf: {praa_confidence_val:.2f}, RTAConf: {rta_confidence_val:.2f}, CombinedMetric: {combined_metric:.2f}")

        if task_input.artifact_type == "CodeModule_TestFailure":
            logger_instance.info(f"ARCA handling CodeModule_TestFailure for: {task_input.code_module_file_path or task_input.artifact_doc_id}")
            
            faulty_code_path_for_input = task_input.code_module_file_path
            faulty_code_content_for_input: Optional[str] = None
            failed_test_reports = task_input.failed_test_report_details

            # Determine which PCMA instance to use for fetching code
            pcma_for_code_retrieval = self._project_chroma_manager if self._project_chroma_manager else pcma_agent_for_logging

            if task_input.artifact_doc_id and pcma_for_code_retrieval:
                try:
                    # Use retrieve_artifact instead of get_document_by_id
                    retrieved_artifact_output = await pcma_for_code_retrieval.retrieve_artifact(
                        collection_name=GENERATED_CODE_ARTIFACTS_COLLECTION, # Pass collection_name first
                        document_id=task_input.artifact_doc_id
                    )
                    if retrieved_artifact_output and retrieved_artifact_output.content:
                        faulty_code_content_for_input = retrieved_artifact_output.content
                        logger_instance.info(f"Fetched faulty_code_content for doc_id: {task_input.artifact_doc_id}. Length: {len(faulty_code_content_for_input)}")
                    else:
                        logger_instance.error(f"Failed to fetch faulty_code_content: doc_id {task_input.artifact_doc_id} not found or content empty using {'self._project_chroma_manager' if self._project_chroma_manager else 'pcma_agent_for_logging'}.")
                        raise ValueError(f"Faulty code content for doc_id {task_input.artifact_doc_id} could not be retrieved.")
                except Exception as e_fc:
                    logger_instance.error(f"Error fetching faulty_code_content for doc_id {task_input.artifact_doc_id}: {e_fc}", exc_info=True)
                    # Return an error ARCAOutput
                    return ARCAOutput(
                        task_id=task_input.task_id, project_id=task_input.project_id,
                        reviewed_artifact_doc_id=task_input.artifact_doc_id or task_input.code_module_file_path or "unknown_artifact",
                        reviewed_artifact_type=task_input.artifact_type, # Use the dynamic artifact_type
                        decision="ERROR", 
                        reasoning=f"Failed to retrieve faulty code content for debugging: {e_fc}",
                        error_message=f"Failed to retrieve faulty code content for debugging: {e_fc}"
                    )
            elif not task_input.artifact_doc_id:
                logger_instance.error(f"artifact_doc_id is missing for CodeModule_TestFailure, cannot fetch faulty code content.")
                return ARCAOutput( 
                    task_id=task_input.task_id, project_id=task_input.project_id,
                    reviewed_artifact_doc_id=task_input.code_module_file_path or "unknown_artifact",
                    reviewed_artifact_type=task_input.artifact_type,
                    decision="ERROR",
                    reasoning="artifact_doc_id missing for CodeModule_TestFailure, cannot fetch code.",
                    error_message="artifact_doc_id missing for CodeModule_TestFailure, cannot fetch code."
                )
            elif not pcma_for_code_retrieval: # Check the chosen PCMA instance
                logger_instance.error("PCMA agent not available (neither self._project_chroma_manager nor context-resolved pcma_agent_for_logging) to fetch faulty_code_content.")
                return ARCAOutput( 
                    task_id=task_input.task_id, project_id=task_input.project_id,
                    reviewed_artifact_doc_id=task_input.artifact_doc_id or task_input.code_module_file_path or "unknown_artifact",
                    reviewed_artifact_type=task_input.artifact_type,
                    decision="ERROR",
                    reasoning="PCMA agent not available for debugging context retrieval.",
                    error_message="PCMA agent not available for debugging context retrieval."
                )
            # Ensure faulty_code_content_for_input is not None if we proceed
            if faulty_code_content_for_input is None:
                logger_instance.error(f"Critical: faulty_code_content_for_input is None for {task_input.artifact_doc_id} after retrieval attempt. Cannot proceed with debugging.")
                return ARCAOutput(
                    task_id=task_input.task_id, project_id=task_input.project_id,
                    reviewed_artifact_doc_id=task_input.artifact_doc_id or "unknown_artifact",
                    reviewed_artifact_type=task_input.artifact_type,
                    decision="ERROR",
                    reasoning="Failed to obtain faulty code content, though no direct exception occurred.",
                    error_message="Failed to obtain faulty code content for debugging."
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
                max_iterations_for_this_call=full_context.get("max_debug_iterations_per_call", 3)
            )

            # --- Implement Task 3.3.6: Max Debugging Attempts Logic ---
            current_attempt_count = len(debugging_task_input.previous_debugging_attempts)
            max_attempts_for_module = full_context.get("max_total_debugging_attempts_for_module", self.MAX_DEBUGGING_ATTEMPTS_PER_MODULE)

            if current_attempt_count >= max_attempts_for_module:
                decision = "REFINEMENT_REQUIRED" # Or a more specific status like "ESCALATION_MAX_ATTEMPTS_REACHED"
                reasoning = f"Max debugging attempts ({max_attempts_for_module}) reached for module {faulty_code_path_for_input}. Escalating for human review. Last known failed tests: {failed_test_reports}"
                logger_instance.warning(reasoning)
                
                if pcma_agent_for_logging:
                    await self._log_event_to_pcma(
                        pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
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
                        # Log failure to PCMA if pcma_agent_for_logging is available
                        if pcma_agent_for_logging:
                            await self._log_event_to_pcma(
                                pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                arca_task_id=task_input.task_id, event_type="ARCA_INTERNAL_ERROR",
                                event_details={"error": "CodeDebuggingAgent_v1 instance not available in ARCA"}, severity="ERROR"
                            )
                        raise ValueError("CodeDebuggingAgent_v1 instance not available in ARCA. Debugging cannot be performed.")

                    code_debugging_agent_to_use = self._code_debugging_agent_instance
                    
                    # Task 3.3.4.2: Asynchronously call invoke_async
                    logger_instance.info(f"Invoking {code_debugging_agent_to_use.AGENT_ID} with input: {debugging_task_input.model_dump_json(indent=2)}")
                    if pcma_agent_for_logging:
                        await self._log_event_to_pcma(
                            pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
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

                    if pcma_agent_for_logging:
                        await self._log_event_to_pcma(
                            pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
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
                    if pcma_agent_for_logging:
                        await self._log_event_to_pcma(
                            pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
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
                            if pcma_agent_for_logging:
                                await self._log_event_to_pcma(
                                    pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                    arca_task_id=task_input.task_id, event_type="ARCA_INTERNAL_ERROR",
                                    event_details={"error": f"Failed to resolve agent: {integration_agent_id}"}, severity="ERROR"
                                )
                            raise ValueError(f"Could not resolve {integration_agent_id}")

                        logger_instance.info(f"Invoking {integration_agent_id} with: {integration_input.model_dump_json(indent=2)}")
                        if pcma_agent_for_logging:
                             await self._log_event_to_pcma(
                                pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_START",
                                event_details={"invoked_agent_id": integration_agent_id, "target_file": integration_input.target_file_path},
                                related_doc_ids=[integration_input.target_file_path]
                            )
                        integration_output = await integration_agent.invoke_async(integration_input, full_context)
                        logger_instance.info(f"{integration_agent_id} output: {integration_output.model_dump_json(indent=2)}")
                        if pcma_agent_for_logging:
                            await self._log_event_to_pcma(
                                pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_END",
                                event_details={"invoked_agent_id": integration_agent_id, "output_status": integration_output.status, "integrated_file_path": integration_output.integrated_file_path},
                                severity="INFO" if integration_output.status == "SUCCESS_APPLIED" else "ERROR"
                            )

                    except Exception as e:
                        logger_instance.error(f"Error invoking {integration_agent_id}: {e}", exc_info=True)
                        if pcma_agent_for_logging:
                             await self._log_event_to_pcma(
                                pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
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
                                if pcma_agent_for_logging:
                                    await self._log_event_to_pcma(
                                        pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                        arca_task_id=task_input.task_id, event_type="ARCA_INTERNAL_ERROR",
                                        event_details={"error": f"Failed to resolve agent: {test_runner_agent_id}"}, severity="ERROR"
                                    )
                                raise ValueError(f"Could not resolve {test_runner_agent_id}")
                            
                            logger_instance.info(f"Invoking {test_runner_agent_id} with: {test_runner_input.model_dump_json(indent=2)}")
                            if pcma_agent_for_logging:
                                await self._log_event_to_pcma(
                                    pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                                    arca_task_id=task_input.task_id, event_type="SUB_AGENT_INVOCATION_START",
                                    event_details={"invoked_agent_id": test_runner_agent_id, "module_path": test_runner_input.code_module_file_path},
                                    related_doc_ids=[test_runner_input.code_module_file_path]
                                )
                            test_runner_output = await test_runner_agent.invoke_async(test_runner_input, full_context)
                            logger_instance.info(f"{test_runner_agent_id} output: {test_runner_output.model_dump_json(indent=2)}")
                            if pcma_agent_for_logging:
                                await self._log_event_to_pcma(
                                    pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
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
                            if pcma_agent_for_logging:
                                await self._log_event_to_pcma(
                                    pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
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

                elif code_debugging_agent_output.status == "FAILURE_NO_FIX_IDENTIFIED":
                    decision = "REFINEMENT_REQUIRED"
                    reasoning = f"CodeDebuggingAgent_v1 could not identify a fix for {debugging_task_input.faulty_code_path}. Confidence: {code_debugging_agent_output.confidence_score}. Suggestions: {code_debugging_agent_output.suggestions_for_ARCA}"
                    logger_instance.warning(reasoning)
                elif code_debugging_agent_output.status == "FAILURE_NEEDS_CLARIFICATION":
                    decision = "REFINEMENT_REQUIRED"
                    reasoning = f"CodeDebuggingAgent_v1 needs clarification for {debugging_task_input.faulty_code_path}. Suggestions: {code_debugging_agent_output.suggestions_for_ARCA}. ARCA needs to implement logic to gather this clarification or escalate."
                    # TODO: ARCA could try to use PCMA to fetch what debugger asked for, or re-invoke planner.
                    logger_instance.warning(reasoning)
                elif code_debugging_agent_output.status == "ERROR_INTERNAL":
                    decision = "REFINEMENT_REQUIRED"
                    reasoning = f"CodeDebuggingAgent_v1 encountered an internal error for {debugging_task_input.faulty_code_path}. Details: {code_debugging_agent_output.explanation_of_fix or 'No details provided.'}"
                    logger_instance.error(reasoning)
                else: # Unknown status from debugger
                    decision = "REFINEMENT_REQUIRED"
                    reasoning = f"CodeDebuggingAgent_v1 returned an unexpected status: {code_debugging_agent_output.status} for {debugging_task_input.faulty_code_path}."
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
            refinement_instructions_for_agent = f"Refinement requested by ARCA for {task_input.artifact_type} (ID: {task_input.artifact_doc_id}). Key concerns based on PRAA/RTA reports. Please review and address. Original combined metric: {combined_metric:.2f}. Generator Confidence: {generator_confidence_val:.2f}, PRAA Confidence: {praa_confidence_val:.2f}, RTA Confidence: {rta_confidence_val:.2f}."
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
                    # This needs careful thought: PAA normally takes a *goal* doc id.
                    # If ARCA is refining an *existing LOPRD*, we might need to pass the LOPRD itself
                    # or structure PAA to accept an LOPRD for refinement.
                    # For now, let's assume PAA is flexible or this implies a specific refinement prompt variant.
                    refined_user_goal_doc_id=task_input.artifact_doc_id, # Using the LOPRD being reviewed as the base for refinement
                    assumptions_and_ambiguities_doc_id=task_input.assumptions_doc_id, # Pass along if available
                    arca_feedback_doc_id=task_input.feedback_doc_id, # The feedback just generated
                    shared_context=task_input.shared_context
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
                    refinement_instructions=refinement_instructions_for_agent
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
        if decision == "ACCEPT_ARTIFACT" or (decision == "REFINEMENT_REQUIRED" and not next_agent_for_refinement) or decision == "PROCEED_TO_DOCUMENTATION":
            update_state_for_review = True # Good point to potentially pend review or record progress.

        if state_manager_instance and update_state_for_review:
            try:
                project_state = state_manager_instance.get_project_state()
                current_cycle_id = project_state.current_cycle_id

                if pcma_agent_for_logging:
                    await self._log_event_to_pcma(
                        pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, event_type="STATE_UPDATE_ATTEMPT",
                        event_details={"target_status": "pending_human_review/project_complete", "current_cycle_id": current_cycle_id or "None"}
                    )

                if current_cycle_id:
                    current_cycle_item = next((c for c in project_state.cycle_history if c.cycle_id == current_cycle_id), None)
                    if current_cycle_item:
                        logger_instance.info(f"Updating cycle {current_cycle_id} in ProjectStateV2 via ARCA.")
                        current_cycle_item.arca_summary_of_cycle_outcome = (current_cycle_item.arca_summary_of_cycle_outcome or "") + f"\n- {cycle_summary_for_state_manager}"
                        
                        if issues_for_human_review_list:
                             current_cycle_item.issues_flagged_for_human_review.extend(issues_for_human_review_list)
                        
                        if decision == "ACCEPT_ARTIFACT" and task_input.artifact_type == "ProjectDocumentation": # Example: Project complete
                            project_state.overall_project_status = "project_complete" # Or a new status like "pending_final_review"
                            current_cycle_item.end_time = datetime.datetime.now(datetime.timezone.utc)
                            logger_instance.info(f"Project {project_state.project_id} marked as complete after documentation acceptance.")
                        elif (decision == "REFINEMENT_REQUIRED" and (not next_agent_for_refinement and task_input.artifact_type != "CodeModule_TestFailure")) or \
                             (decision == "REFINEMENT_REQUIRED" and task_input.artifact_type == "CodeModule_TestFailure" and "Max debugging attempts" in reasoning) or \
                             (decision == "ACCEPT_ARTIFACT" and task_input.artifact_type != "ProjectDocumentation") or \
                             decision == "PROCEED_TO_DOCUMENTATION": # Assuming these are points for review
                            project_state.overall_project_status = "pending_human_review"
                            current_cycle_item.end_time = datetime.datetime.now(datetime.timezone.utc)
                            logger_instance.info(f"Project state for {project_state.project_id} set to 'pending_human_review' by ARCA.")

                        state_manager_instance._save_project_state() # Use the private method to save the modified state
                        logger_instance.info(f"ARCA updated ProjectStateV2 for project {task_input.project_id}, cycle {current_cycle_id}.")
                        update_state_log_details = {"reason": "Cycle progression/completion", "outcome": "SUCCESS", "updated_cycle_id": current_cycle_id}
                    else:
                        logger_instance.warning(f"ARCA: Current cycle ID {current_cycle_id} not found in project state history. Cannot update cycle details.")
                        update_state_log_details = {"reason": "Cycle ID not found", "outcome": "FAILURE", "target_cycle_id": current_cycle_id}
                else:
                    logger_instance.warning("ARCA: No current_cycle_id set in ProjectStateV2. Cannot update cycle details.")
                    update_state_log_details = {"reason": "No current cycle ID in state", "outcome": "FAILURE"}
                
                if pcma_agent_for_logging: # Log success/specific failure of state update
                     await self._log_event_to_pcma(
                        pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, 
                        event_type="STATE_UPDATE_SUCCESS" if update_state_log_details["outcome"] == "SUCCESS" else "STATE_UPDATE_FAILURE",
                        event_details=update_state_log_details,
                        severity="INFO" if update_state_log_details["outcome"] == "SUCCESS" else "WARNING"
                    )

            except StatusFileError as e:
                logger_instance.error(f"ARCA: Failed to update project state due to StatusFileError: {e}", exc_info=True)
                if pcma_agent_for_logging:
                    await self._log_event_to_pcma(
                        pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, event_type="STATE_UPDATE_FAILURE",
                        event_details={"error": f"StatusFileError: {str(e)}"}, severity="ERROR"
                    )
            except Exception as e: # Catch any other unexpected errors
                logger_instance.error(f"ARCA: Unexpected error updating project state: {e}", exc_info=True)
                if pcma_agent_for_logging:
                    await self._log_event_to_pcma(
                        pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
                        arca_task_id=task_input.task_id, event_type="STATE_UPDATE_FAILURE",
                        event_details={"error": f"Unexpected error: {str(e)}"}, severity="ERROR"
                    )
        elif not state_manager_instance and update_state_for_review:
            logger_instance.warning(
                "ARCA: StateManager instance not available, but an update was flagged. "
                "Skipping project state update."
                )
            if pcma_agent_for_logging:
                await self._log_event_to_pcma(
                    pcma_agent=pcma_agent_for_logging, project_id=task_input.project_id, cycle_id=task_input.cycle_id,
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
        
        final_arca_output = ARCAOutput(
            task_id=task_input.task_id,
            project_id=task_input.project_id,
            reviewed_artifact_doc_id=task_input.artifact_doc_id or (task_input.code_module_file_path if task_input.artifact_type == "CodeModule_TestFailure" else "unknown_reviewed_doc_id"),
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