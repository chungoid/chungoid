from __future__ import annotations

import logging
import datetime
import uuid
import json
class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

from typing import Any, Dict, Optional, Literal, Union, ClassVar, List, Type, get_args
from enum import Enum
import yaml

from pydantic import BaseModel, Field, model_validator, PrivateAttr

from ..protocol_aware_agent import ProtocolAwareAgent
from ...protocols.base.protocol_interface import ProtocolPhase
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
# COMMENTED OUT: Defunct project_chroma_manager_agent imports
# from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
#     ProjectChromaManagerAgent_v1, 
#     GENERATED_CODE_ARTIFACTS_COLLECTION,
#     RetrieveArtifactOutput, # Added
#     RISK_ASSESSMENT_REPORTS_COLLECTION, # Added
#     OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION, # Added
#     TRACEABILITY_REPORTS_COLLECTION, # Added
#     REVIEW_REPORTS_COLLECTION, # Added
#     EXECUTION_PLANS_COLLECTION # ADDED IMPORT
# )

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
# COMMENTED OUT: Defunct project_chroma_manager_agent imports
# from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, LogStorageConfirmation # For type hinting PCMA
# NEW QA LOG IMPORTS
from chungoid.schemas.agent_logs import QualityAssuranceLogEntry, QAEventType, OverallQualityStatus
# COMMENTED OUT: Defunct project_chroma_manager_agent imports
# from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
#     QUALITY_ASSURANCE_LOGS_COLLECTION, 
#     ARTIFACT_TYPE_QA_LOG_ENTRY_JSON,
#     StoreArtifactInput # Ensure this is imported for store_artifact
# )

# Import for plan modification
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec

# Registry-first architecture import
from chungoid.registry import register_autonomous_engine_agent

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


@register_autonomous_engine_agent(capabilities=["autonomous_coordination", "quality_gates", "refinement_orchestration"])
class AutomatedRefinementCoordinatorAgent_v1(ProtocolAwareAgent):
    AGENT_ID: ClassVar[str] = "AutomatedRefinementCoordinatorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Automated Refinement Coordinator Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Coordinates the iterative refinement of project artifacts, invoking specialist agents as needed."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "automated_refinement_coordinator_agent_v1.yaml" # Points to server_prompts/autonomous_engine/
    AGENT_VERSION: ClassVar[str] = "0.1.0"
    CAPABILITIES: ClassVar[List[str]] = ["autonomous_coordination", "quality_gates", "refinement_orchestration"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.AUTONOMOUS_COORDINATION # MODIFIED
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL # Usually internal, invoked by higher orchestrator
    INPUT_SCHEMA: ClassVar[Type[ARCAReviewInput]] = ARCAReviewInput
    OUTPUT_SCHEMA: ClassVar[Type[ARCAOutput]] = ARCAOutput

    # Declare private attributes using PrivateAttr
    _llm_provider: Optional[LLMProvider] = PrivateAttr(default=None)
    _prompt_manager: Optional[PromptManager] = PrivateAttr(default=None)
    # COMMENTED OUT: Defunct project_chroma_manager_agent references
    # _project_chroma_manager: Optional[ProjectChromaManagerAgent_v1] = PrivateAttr(default=None)
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
    # ADDED: Protocol definitions following AI agent best practices
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["autonomous_team_formation", "enhanced_deep_planning"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["tool_validation", "goal_tracking"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'error_recovery']


    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        prompt_manager: Optional[PromptManager] = None,
        # COMMENTED OUT: Defunct project_chroma_manager_agent parameter
        # project_chroma_manager: Optional[ProjectChromaManagerAgent_v1] = None,
        system_context: Optional[Dict[str, Any]] = None,
        state_manager: Optional[StateManager] = None,
        **kwargs  # Pydantic will populate model fields from here
    ):
        super().__init__(**kwargs) # Initialize Pydantic model fields first

        # Now initialize PrivateAttrs
        if system_context and "logger" in system_context:
            self._logger_instance = system_context["logger"]
        else:
            # Ensure AGENT_ID is accessible, might need to be self.AGENT_ID if ProtocolAwareAgent sets it up
            # or AutomatedRefinementCoordinatorAgent_v1.AGENT_ID if accessed as class variable
            self._logger_instance = logging.getLogger(AutomatedRefinementCoordinatorAgent_v1.AGENT_ID)

        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        self._state_manager = state_manager
        self._current_debug_attempts_for_module = 0 # Explicitly initialize, though PrivateAttr has default
        self._last_feedback_doc_id = None # Explicitly initialize

        # Initialize CodeDebuggingAgent instance
        if self._llm_provider and self._prompt_manager:
            try:
                # Ensure CodeDebuggingAgent_v1 is correctly imported
                self._code_debugging_agent_instance = CodeDebuggingAgent_v1(
                    llm_provider=self._llm_provider,
                    prompt_manager=self._prompt_manager,
                    system_context={"logger": self._logger_instance.getChild("CodeDebuggingAgent_v1")}
                )
            except Exception as e:
                self._logger_instance.error(f"Failed to initialize CodeDebuggingAgent_v1 within ARCA: {e}", exc_info=True)
                self._code_debugging_agent_instance = None
        else:
            self._logger_instance.warning("LLMProvider or PromptManager not available for CodeDebuggingAgent initialization.")
            self._code_debugging_agent_instance = None

    # COMMENTED OUT: Method that depends on defunct ProjectChromaManagerAgent_v1
    # async def _log_event_to_pcma(
    #     self,
    #     pcma_agent: ProjectChromaManagerAgent_v1,
    #     project_id: str,
    #     cycle_id: str,
    #     arca_task_id: str,
    #     event_type: Literal[
    #         "ARCA_INVOCATION_START",
    #         "ARCA_DECISION_MADE",
    #         "SUB_AGENT_INVOCATION_START",
    #         "SUB_AGENT_INVOCATION_END",
    #         "MAX_DEBUG_ATTEMPTS_REACHED",
    #         "STATE_UPDATE_ATTEMPT",
    #         "STATE_UPDATE_SUCCESS",
    #         "STATE_UPDATE_FAILURE",
    #         "ARCA_INTERNAL_ERROR"
    #     ],
    #     event_details: Dict[str, Any],
    #     severity: Literal["INFO", "WARNING", "ERROR"] = "INFO",
    #     related_doc_ids: Optional[List[str]] = None
    # ):
    #     """Helper method to log an ARCA event to ProjectChromaManagerAgent.
    # 
    # ✨ PURE PROTOCOL ARCHITECTURE - No backward compatibility, clean execution paths only."""
    #     log_entry = ARCALogEntry(
    #         agent_id=self.AGENT_ID, # Added agent_id
    #         # log_id is generated by default_factory in ARCALogEntry
    #         timestamp=datetime.datetime.now(datetime.timezone.utc), # Ensure timestamp is set here
    #         arca_task_id=arca_task_id,
    #         project_id=project_id,
    #         cycle_id=cycle_id,
    #         event_type=event_type,
    #         event_details=event_details,
    #         severity=severity,
    #         related_artifact_doc_ids=related_doc_ids or []
    #     )
    #     try:
    #         # Assuming pcma_agent is already resolved and available
    #         confirmation: LogStorageConfirmation = await pcma_agent.log_arca_event(
    #             project_id=project_id,
    #             cycle_id=cycle_id,
    #             log_entry=log_entry
    #         )
    #         if confirmation.status != "SUCCESS":
    #             self._logger_instance.error(f"PCMA failed to log ARCA event ({event_type}): {confirmation.message}. Log ID: {log_entry.log_id}")
    #         else:
    #             # self._logger_instance.info(f"ARCA event ({event_type}) logged to PCMA. Log ID: {confirmation.log_id}") # Can be too verbose
    #             pass
    #     except Exception as e:
    #         self._logger_instance.error(f"Exception during ARCA event logging to PCMA for event type {event_type}, Log ID {log_entry.log_id}: {e}", exc_info=True)

    async def execute(self, task_input, full_context: Optional[Dict[str, Any]] = None):
        """
        Execute using pure protocol architecture.
        No fallback - protocol execution only for clean, maintainable code.
        """
        try:
            # Determine primary protocol for this agent
            primary_protocol = self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "simple_operations"
            
            protocol_task = {
                "task_input": task_input.dict() if hasattr(task_input, 'dict') else task_input,
                "full_context": full_context,
                "goal": f"Execute {self.AGENT_NAME} specialized task"
            }
            
            protocol_result = self.execute_with_protocol(protocol_task, primary_protocol)
            
            if protocol_result["overall_success"]:
                return self._extract_output_from_protocol_result(protocol_result, task_input)
            else:
                # Enhanced error handling instead of fallback
                error_msg = f"Protocol execution failed for {self.AGENT_NAME}: {protocol_result.get('error', 'Unknown error')}"
                self._logger.error(error_msg)
                raise ProtocolExecutionError(error_msg)
                
        except Exception as e:
            error_msg = f"Pure protocol execution failed for {self.AGENT_NAME}: {e}"
            self._logger.error(error_msg)
            raise ProtocolExecutionError(error_msg)
            
            protocol_result = self.execute_with_protocol(protocol_task, primary_protocol)
            
            if protocol_result["overall_success"]:
                return self._extract_output_from_protocol_result(protocol_result, task_input)
            else:
                self._logger.warning("Protocol execution failed, falling back to traditional method")
                raise ProtocolExecutionError("Pure protocol execution failed")
                
        except Exception as e:
            self._logger.warning(f"Protocol execution error: {e}, falling back to traditional method")
            raise ProtocolExecutionError("Pure protocol execution failed")

    # ADDED: Protocol phase execution logic
    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute agent-specific logic for each protocol phase."""
        
        # Generic phase handling - can be overridden by specific agents
        if phase.name in ["discovery", "analysis", "planning", "execution", "validation"]:
            return self._execute_generic_phase(phase)
        else:
            self._logger.warning(f"Unknown protocol phase: {phase.name}")
            return {"phase_completed": True, "method": "fallback"}

    def _execute_generic_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute generic phase logic suitable for most agents."""
        return {
            "phase_name": phase.name,
            "status": "completed", 
            "outputs": {"generic_result": f"Phase {phase.name} completed"},
            "method": "generic_protocol_execution"
        }

    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], task_input) -> Any:
        """Extract agent output from protocol execution results."""
        # Generic extraction - should be overridden by specific agents
        return {
            "status": "SUCCESS",
            "message": "Task completed via protocol execution",
            "protocol_used": protocol_result.get("protocol_name"),
            "execution_time": protocol_result.get("execution_time", 0),
            "phases_completed": len([p for p in protocol_result.get("phases", []) if p.get("success", False)])
        }


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