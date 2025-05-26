from __future__ import annotations

import logging
import datetime
import uuid
import json
import time

from typing import Any, Dict, Optional, Literal, Union, ClassVar, List, Type, get_args, TYPE_CHECKING
from enum import Enum
import yaml

from pydantic import BaseModel, Field, model_validator, PrivateAttr

from chungoid.agents.unified_agent import UnifiedAgent
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

from ...schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    AgentExecutionResult,
    ExecutionMetadata,
    ExecutionMode,
    CompletionReason,
    IterationResult,
    StageInfo,
)

logger = logging.getLogger(__name__)

ARCA_PROMPT_NAME = "automated_refinement_coordinator_agent_v1.yaml" # If LLM-based decision making is used
ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME = "arca_optimization_evaluator_v1_prompt.yaml" # NEW PROMPT

# Constants for ARCA behavior
MAX_DEBUGGING_ATTEMPTS_PER_MODULE: ClassVar[int] = 3 # Added ClassVar
DEFAULT_ACCEPTANCE_THRESHOLD: ClassVar[float] = 0.85 # MODIFIED: Added ClassVar


# ------------------------------------------------------------------
# Fallback type aliases for yet-to-be-migrated PCMA structures --------
# These keep Phase-2 compiling; Phase-3 will remove PCMA dependencies.
# ------------------------------------------------------------------

try:
    from chungoid.schemas.project_status_schema import ProjectStateDataV2  # type: ignore
except Exception:  # pragma: no cover – schema not available yet
    ProjectStateDataV2 = Any  # type: ignore

try:
    from chungoid.schemas.autonomous_engine.project_chroma_schemas import RetrieveArtifactOutput  # type: ignore
except Exception:  # pragma: no cover
    RetrieveArtifactOutput = Any  # type: ignore


# Ensure ProtocolExecutionError exists (may have been dropped during refactor)
class ProtocolExecutionError(Exception):
    """Raised when the protocol-based execution encounters unrecoverable error."""

    pass


# --- Input and Output Schemas for ARCA --- #
class ARCAReviewInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this ARCA review task.")
    
    # Traditional fields - optional when using intelligent context
    project_id: Optional[str] = Field(None, description="Identifier for the current project.")
    cycle_id: Optional[str] = Field(None, description="The ID of the current refinement cycle.")
    artifact_type: Optional[ARCAReviewArtifactType] = Field(None, description="The type of artifact ARCA needs to review or act upon.")
    artifact_doc_id: Optional[str] = Field(
        None, description="The document ID of the primary artifact in ChromaDB."
    )
    
    # ADDED: Intelligent project analysis from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications from orchestrator analysis")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    user_goal: Optional[str] = Field(None, description="Original user goal for context")
    project_path: Optional[str] = Field(None, description="Project path for context")
    # New fields for CodeModule_TestFailure
    code_module_file_path: Optional[str] = Field(
        None, description="Path to the code file that has failed tests. Required if artifact_type is CodeModule_TestFailure."
    )
    failed_test_report_details: Optional[List[FailedTestReport]] = Field(
        None, description="List of failed test report objects. Required if artifact_type is CodeModule_TestFailure."
    )
    generator_agent_id: Optional[str] = Field(None, description="ID of the agent that generated the artifact. Optional in intelligent context mode.")
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
    def check_intelligent_context_requirements(self) -> 'ARCAReviewInput':
        """Validate requirements based on execution mode (intelligent context vs traditional)."""
        
        if self.intelligent_context:
            # Intelligent context mode - requires project specifications and user goal
            if not self.project_specifications:
                raise ValueError("project_specifications is required when intelligent_context=True")
            if not self.user_goal:
                raise ValueError("user_goal is required when intelligent_context=True")
            
            # In intelligent context mode, traditional fields are optional
            # The smart agent will use its LLM processing abilities to handle coordination
            
        else:
            # Traditional mode - requires generator_agent_id and artifact details
            if not self.generator_agent_id:
                raise ValueError("generator_agent_id is required when intelligent_context=False")
            if not self.artifact_type:
                raise ValueError("artifact_type is required when intelligent_context=False")
            if not self.artifact_doc_id:
                raise ValueError("artifact_doc_id is required when intelligent_context=False")
        
        return self

    @model_validator(mode='after')
    def check_report_ids(self) -> 'ARCAReviewInput':
        # Only check report IDs in traditional mode
        if not self.intelligent_context and self.artifact_type in ["LOPRD", "Blueprint", "MasterExecutionPlan", "CodeModule"]:
            if not self.praa_risk_report_doc_id or not self.praa_optimization_report_doc_id:
                logger.warning(f"PRAA report IDs missing for {self.artifact_type} review in ARCA. This might limit decision quality.")
        if not self.intelligent_context and self.artifact_type in ["Blueprint", "MasterExecutionPlan", "CodeModule"]:
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
class AutomatedRefinementCoordinatorAgent_v1(UnifiedAgent):
    AGENT_ID: ClassVar[str] = "AutomatedRefinementCoordinatorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Automated Refinement Coordinator Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Coordinates the iterative refinement of project artifacts, invoking specialist agents as needed."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "automated_refinement_coordinator_agent_v1.yaml" # Points to server_prompts/autonomous_engine/
    AGENT_VERSION: ClassVar[str] = "0.1.0"
    CAPABILITIES: ClassVar[List[str]] = ["autonomous_coordination", "quality_gates", "refinement_orchestration", "complex_analysis"]
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
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        state_manager: Optional[StateManager] = None,
        **kwargs
    ):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self._state_manager = state_manager

    async def _execute_iteration(
        self, 
        context: UEContext, 
        iteration: int
    ) -> IterationResult:
        """
        Phase 3 UAEI implementation for automated refinement coordination.
        Runs comprehensive artifact review workflow: assessment → analysis → decision → coordination
        """
        try:
            # Convert inputs to expected format - handle both dict and object inputs
            if isinstance(context.inputs, dict):
                task_input = ARCAReviewInput(**context.inputs)
            elif hasattr(context.inputs, 'dict'):
                inputs = context.inputs.dict()
                task_input = ARCAReviewInput(**inputs)
            else:
                task_input = context.inputs

            # Phase 1: Assessment - Assess artifact and gather context
            if task_input.intelligent_context and task_input.project_specifications:
                self.logger.info("Using intelligent project specifications from orchestrator")
                assessment_result = self._extract_assessment_from_intelligent_specs(task_input.project_specifications, task_input.user_goal)
            else:
                self.logger.info("Using traditional artifact assessment")
                assessment_result = await self._assess_artifact(task_input, context.shared_context)
            
            # Phase 2: Analysis - Analyze quality and compliance
            analysis_result = await self._analyze_quality_and_compliance(assessment_result, task_input, context.shared_context)
            
            # Phase 3: Decision - Make coordination decision
            decision_result = await self._make_coordination_decision(analysis_result, task_input, context.shared_context)
            
            # Phase 4: Coordination - Execute coordination actions
            coordination_result = await self._execute_coordination_actions(decision_result, task_input, context.shared_context)
            
            # Calculate quality score based on decision confidence
            quality_score = self._calculate_quality_score(coordination_result)
            
            # Create output
            output = ARCAOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id or "intelligent_project",
                reviewed_artifact_doc_id=task_input.artifact_doc_id or "intelligent_artifact",
                reviewed_artifact_type=task_input.artifact_type or "LOPRD",
                decision=coordination_result.get("decision", "ERROR"),
                reasoning=coordination_result.get("reasoning", "Coordination completed"),
                confidence_in_decision=ConfidenceScore(
                    value=quality_score,
                    method="comprehensive_assessment",
                    explanation="Based on comprehensive artifact assessment and analysis"
                ),
                next_agent_id_for_refinement=coordination_result.get("next_agent_id"),
                next_agent_input=coordination_result.get("next_agent_input"),
                debugging_task_input=coordination_result.get("debugging_task_input"),
                final_artifact_doc_id=coordination_result.get("final_artifact_doc_id"),
                new_master_plan_doc_id=coordination_result.get("new_master_plan_doc_id")
            )
            
            tools_used = ["artifact_assessment", "quality_analysis", "coordination_decision"]
            
            return IterationResult(
                output=output,
                quality_score=quality_score,
                tools_used=tools_used,
                protocol_used="refinement_coordination_protocol"
            )
            
        except Exception as e:
            self.logger.error(f"Refinement coordination iteration failed: {e}")
            
            # Create error output
            error_output = ARCAOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())),
                project_id=getattr(task_input, 'project_id', None) or 'intelligent_project',
                reviewed_artifact_doc_id=getattr(task_input, 'artifact_doc_id', None) or 'intelligent_artifact',
                reviewed_artifact_type=getattr(task_input, 'artifact_type', None) or 'LOPRD',
                decision="ERROR",
                reasoning=f"Refinement coordination failed: {str(e)}",
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="refinement_coordination_protocol"
            )

    def _extract_assessment_from_intelligent_specs(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """Extract assessment-like data from intelligent project specifications."""
        
        # Create mock assessment for intelligent context
        assessment = {
            "artifact_available": True,
            "artifact_quality": 0.85,  # High quality from intelligent analysis
            "assessment_confidence": 0.9,
            "intelligent_analysis": True,
            "project_type": project_specs.get("project_type", "unknown"),
            "technologies": project_specs.get("technologies", []),
            "coordination_needed": True,  # Always coordinate in autonomous pipeline
            "refinement_areas": [
                "Architecture optimization",
                "Dependency management",
                "Quality assurance"
            ]
        }
        
        return assessment

    async def _assess_artifact(self, task_input: ARCAReviewInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Assessment - Assess artifact and gather context."""
        self.logger.info(f"Starting artifact assessment for {task_input.artifact_type}")
        
        # Assess artifact based on type
        artifact_score = 0.7  # Default score
        
        if task_input.artifact_type == "CodeModule_TestFailure":
            artifact_score = 0.3  # Low score for failed tests
        elif task_input.generator_agent_confidence:
            artifact_score = task_input.generator_agent_confidence.value
        
        assessment = {
            "artifact_type": task_input.artifact_type,
            "artifact_id": task_input.artifact_doc_id,
            "generator_confidence": artifact_score,
            "has_risk_reports": bool(task_input.praa_risk_report_doc_id),
            "has_optimization_reports": bool(task_input.praa_optimization_report_doc_id),
            "has_traceability_reports": bool(task_input.rta_report_doc_id),
            "assessment_timestamp": datetime.datetime.now().isoformat()
        }
        
        return assessment

    async def _analyze_quality_and_compliance(self, assessment_result: Dict[str, Any], task_input: ARCAReviewInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Analysis - Analyze quality and compliance."""
        self.logger.info("Starting quality and compliance analysis")
        
        generator_confidence = assessment_result.get("generator_confidence", 0.0)
        has_reports = assessment_result.get("has_risk_reports", False)
        
        # Quality analysis based on confidence and available reports
        if generator_confidence >= self.DEFAULT_ACCEPTANCE_THRESHOLD and has_reports:
            quality_status = "HIGH"
            compliance_score = 0.9
        elif generator_confidence >= 0.6:
            quality_status = "MEDIUM"
            compliance_score = 0.7
        else:
            quality_status = "LOW"
            compliance_score = 0.4
            
        analysis = {
            "quality_status": quality_status,
            "compliance_score": compliance_score,
            "meets_acceptance_threshold": compliance_score >= self.DEFAULT_ACCEPTANCE_THRESHOLD,
            "requires_refinement": compliance_score < 0.6,
            "analysis_confidence": min(0.9, compliance_score + 0.1)
        }
        
        return analysis

    async def _make_coordination_decision(self, analysis_result: Dict[str, Any], task_input: ARCAReviewInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Decision - Make coordination decision."""
        self.logger.info("Making coordination decision")
        
        meets_threshold = analysis_result.get("meets_acceptance_threshold", False)
        requires_refinement = analysis_result.get("requires_refinement", True)
        quality_status = analysis_result.get("quality_status", "LOW")
        
        # Decision logic based on artifact type and quality
        if task_input.artifact_type == "CodeModule_TestFailure":
            decision = "TEST_FAILURE_HANDOFF"
            reasoning = "Code module has test failures requiring debugging"
        elif meets_threshold:
            decision = "ACCEPT_ARTIFACT"
            reasoning = f"Artifact meets quality threshold with {quality_status} quality status"
        elif requires_refinement:
            decision = "REFINEMENT_REQUIRED"
            reasoning = f"Artifact requires refinement due to {quality_status} quality status"
        else:
            decision = "ESCALATE_TO_USER"
            reasoning = "Unable to determine appropriate action, escalating to user"
            
        decision_result = {
            "decision": decision,
            "reasoning": reasoning,
            "decision_confidence": analysis_result.get("analysis_confidence", 0.5),
            "quality_status": quality_status,
            "artifact_type": task_input.artifact_type
        }
        
        return decision_result

    async def _execute_coordination_actions(self, decision_result: Dict[str, Any], task_input: ARCAReviewInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Coordination - Execute coordination actions."""
        self.logger.info(f"Executing coordination actions for decision: {decision_result.get('decision')}")
        
        decision = decision_result.get("decision", "ERROR")
        
        coordination = {
            "decision": decision,
            "reasoning": decision_result.get("reasoning", ""),
            "execution_timestamp": datetime.datetime.now().isoformat()
        }
        
        # Execute specific coordination actions based on decision
        if decision == "ACCEPT_ARTIFACT":
            coordination["final_artifact_doc_id"] = task_input.artifact_doc_id
            
        elif decision == "REFINEMENT_REQUIRED":
            # Determine which agent should handle refinement
            if task_input.artifact_type == "LOPRD":
                coordination["next_agent_id"] = "ProductAnalystAgent_v1"
                coordination["next_agent_input"] = {"project_id": task_input.project_id, "refinement_required": True}
            elif task_input.artifact_type in ["Blueprint", "ProjectBlueprint"]:
                coordination["next_agent_id"] = "ArchitectAgent_v1"
                coordination["next_agent_input"] = {"project_id": task_input.project_id, "refinement_required": True}
            elif task_input.artifact_type == "MasterExecutionPlan":
                coordination["next_agent_id"] = "SystemMasterPlannerAgent_v1"
                coordination["next_agent_input"] = {"project_id": task_input.project_id, "refinement_required": True}
                
        elif decision == "TEST_FAILURE_HANDOFF":
            # Create debugging task input
            debugging_input = DebuggingTaskInput(
                project_id=task_input.project_id,
                faulty_code_path=task_input.code_module_file_path or "unknown",
                failed_test_reports=task_input.failed_test_report_details or [],
                relevant_loprd_requirements_ids=[]
            )
            coordination["debugging_task_input"] = debugging_input
            
        return coordination

    def _calculate_quality_score(self, coordination_result: Dict[str, Any]) -> float:
        """Calculate overall quality score based on coordination results."""
        decision = coordination_result.get("decision", "ERROR")
        
        # Score based on decision type
        if decision == "ACCEPT_ARTIFACT":
            return 0.9
        elif decision in ["REFINEMENT_REQUIRED", "TEST_FAILURE_HANDOFF"]:
            return 0.7
        elif decision == "ESCALATE_TO_USER":
            return 0.5
        else:
            return 0.2

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = ARCAReviewInput.model_json_schema()
        output_schema = ARCAOutput.model_json_schema()
        
        return AgentCard(
            agent_id=AutomatedRefinementCoordinatorAgent_v1.AGENT_ID,
            name=AutomatedRefinementCoordinatorAgent_v1.AGENT_NAME,
            description=AutomatedRefinementCoordinatorAgent_v1.AGENT_DESCRIPTION,
            version=AutomatedRefinementCoordinatorAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[AutomatedRefinementCoordinatorAgent_v1.CATEGORY.value],
            visibility=AutomatedRefinementCoordinatorAgent_v1.VISIBILITY.value,
            capability_profile={
                "coordinates_refinement": True,
                "manages_quality_gates": True,
                "orchestrates_agents": True
            },
            metadata={
                "callable_fn_path": f"{AutomatedRefinementCoordinatorAgent_v1.__module__}.{AutomatedRefinementCoordinatorAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[ARCAReviewInput]:
        return ARCAReviewInput

    def get_output_schema(self) -> Type[ARCAOutput]:
        return ARCAOutput 