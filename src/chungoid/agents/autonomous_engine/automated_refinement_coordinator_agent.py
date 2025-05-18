from __future__ import annotations

import logging
import datetime
import uuid
from typing import Any, Dict, Optional, Literal, Union, ClassVar

from pydantic import BaseModel, Field, model_validator

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider # Assuming direct LLM call for complex decisions might be an option
from chungoid.utils.prompt_manager import PromptManager # If decision logic uses prompts
from chungoid.schemas.common import ConfidenceScore
# Mocked Agent Inputs for agents ARCA might call
from .product_analyst_agent import ProductAnalystInput # For LOPRD refinement
from .architect_agent import ArchitectAgentInput # For blueprint refinement
from chungoid.schemas.agent_master_planner import MasterPlannerInput # For instructing plan refinement
# Import the new documentation agent's input schema
from .project_documentation_agent import ProjectDocumentationAgentInput, ProjectDocumentationAgent_v1 

# NEW: Import StateManager and related schemas
from chungoid.utils.state_manager import StateManager, StatusFileError
from chungoid.schemas.project_state import ProjectStateV2, CycleHistoryItem # CycleHistoryItem might not be directly used if modifying existing

from chungoid.utils.agent_registry import AgentCard # For AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard

logger = logging.getLogger(__name__)

ARCA_PROMPT_NAME = "automated_refinement_coordinator_agent_v1.yaml" # If LLM-based decision making is used

# --- Input and Output Schemas for ARCA --- #

class ARCAReviewInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this ARCA review task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    artifact_doc_id: str = Field(..., description="ChromaDB ID of the artifact to be reviewed (LOPRD, Blueprint, or MasterExecutionPlan).")
    artifact_type: Literal["LOPRD", "Blueprint", "MasterExecutionPlan", "CodeModule", "ProjectDocumentation"] = Field(..., description="Type of the artifact.")
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

class ARCAOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    reviewed_artifact_doc_id: str = Field(..., description="ChromaDB ID of the artifact that was reviewed.")
    reviewed_artifact_type: Literal["LOPRD", "Blueprint", "MasterExecutionPlan", "CodeModule", "ProjectDocumentation"] = Field(..., description="Type of the artifact reviewed.")
    decision: Literal["ACCEPT_ARTIFACT", "REFINEMENT_REQUIRED", "PROCEED_TO_DOCUMENTATION"] = Field(..., description="Decision made by ARCA.")
    decision_reasoning: str = Field(..., description="Brief explanation for the decision.")
    confidence_in_decision: Optional[ConfidenceScore] = Field(None, description="ARCA's confidence in its own decision.")
    
    # If decision is REFINEMENT_REQUIRED, these fields provide details for the orchestrator
    next_agent_id_for_refinement: Optional[str] = Field(None, description="The agent_id to call for refinement (e.g., ProductAnalystAgent_v1, ArchitectAgent_v1, SystemMasterPlannerAgent_v1).")
    next_agent_input: Optional[Union[ProductAnalystInput, ArchitectAgentInput, MasterPlannerInput, ProjectDocumentationAgentInput, Dict[str, Any]]] = Field(None, description="The full input payload for the next agent if refinement is needed.")
    
    # If decision is ACCEPT_ARTIFACT
    final_artifact_doc_id: Optional[str] = Field(None, description="The doc_id of the artifact if accepted (usually same as input artifact_doc_id).")

    error_message: Optional[str] = Field(None, description="Error message if ARCA encountered an issue.")


class AutomatedRefinementCoordinatorAgent_v1(BaseAgent):
    AGENT_ID: ClassVar[str] = "AutomatedRefinementCoordinatorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Automated Refinement Coordinator Agent v1"
    DESCRIPTION: ClassVar[str] = "Coordinates the refinement loop for LOPRDs, Blueprints, and MasterExecutionPlans based on PRAA and RTA feedback."
    VERSION: ClassVar[str] = "0.1.1" # Version bump
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.ORCHESTRATION_LOGIC # Or custom
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC

    _llm_provider: Optional[LLMProvider]
    _prompt_manager: Optional[PromptManager]
    _logger: logging.Logger
    _state_manager: Optional[StateManager]

    # Thresholds for decision making (can be made configurable)
    DEFAULT_ACCEPTANCE_THRESHOLD = 0.85  # e.g., If generator_confidence * praa_confidence * rta_confidence > threshold
    MIN_GENERATOR_CONFIDENCE = 0.6
    MIN_PRAA_CONFIDENCE = 0.5 # PRAA might be more critical
    MIN_RTA_CONFIDENCE = 0.6
    MIN_DOC_AGENT_CONFIDENCE_FOR_ACCEPT = 0.5 # If we were to review docs

    def __init__(
        self, 
        llm_provider: Optional[LLMProvider] = None, # May not be needed if rule-based
        prompt_manager: Optional[PromptManager] = None, # May not be needed if rule-based
        system_context: Optional[Dict[str, Any]] = None,
        state_manager: Optional[StateManager] = None, # NEW: Add state_manager parameter
        **kwargs
    ):
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        self._state_manager = state_manager
        if system_context and "logger" in system_context:
            self._logger = system_context["logger"]
        else:
            self._logger = logging.getLogger(self.AGENT_ID)

    async def invoke_async(
        self,
        inputs: Dict[str, Any],
        full_context: Optional[Dict[str, Any]] = None,
    ) -> ARCAOutput:
        llm_provider = self._llm_provider
        prompt_manager = self._prompt_manager
        logger_instance = self._logger
        state_manager_instance = self._state_manager # NEW: Get StateManager instance

        if full_context:
            # No direct LLM/Prompt use in MVP, but provision for future
            if not llm_provider and "llm_provider" in full_context: llm_provider = full_context["llm_provider"]
            if not prompt_manager and "prompt_manager" in full_context: prompt_manager = full_context["prompt_manager"]
            # NEW: Get StateManager from full_context if not already set
            if not state_manager_instance and "state_manager" in full_context:
                state_manager_instance = full_context["state_manager"]
            if "system_context" in full_context and "logger" in full_context["system_context"]:
                if full_context["system_context"]["logger"] != self._logger: 
                    logger_instance = full_context["system_context"]["logger"]
        
        # For MVP, ARCA is primarily rule-based, so LLM/PromptManager aren't strictly needed yet.
        # if not llm_provider or not prompt_manager: # Uncomment if they become essential
        #     # ... handle missing dependency error ...

        try:
            parsed_inputs = ARCAReviewInput(**inputs)
        except Exception as e:
            logger_instance.error(f"Input parsing failed: {e}", exc_info=True)
            return ARCAOutput(task_id=inputs.get("task_id", "parse_err"), project_id=inputs.get("project_id", "parse_err"), reviewed_artifact_doc_id="parse_err", reviewed_artifact_type="LOPRD", decision="REFINEMENT_REQUIRED", decision_reasoning=f"Input parsing error: {e}", error_message=str(e))

        logger_instance.info(f"{self.AGENT_ID} invoked for task {parsed_inputs.task_id}, artifact {parsed_inputs.artifact_doc_id} ({parsed_inputs.artifact_type}) in project {parsed_inputs.project_id}")

        # --- MOCK: Retrieve PRAA/RTA report summaries (actual would involve PCMA calls) ---
        # These would be summaries or key findings, not full content for decision making here.
        # For MVP, we'll use confidence scores directly.
        praa_confidence_val = parsed_inputs.praa_confidence_score.value if parsed_inputs.praa_confidence_score else self.MIN_PRAA_CONFIDENCE
        rta_confidence_val = parsed_inputs.rta_confidence_score.value if parsed_inputs.rta_confidence_score else self.MIN_RTA_CONFIDENCE
        generator_confidence_val = parsed_inputs.generator_agent_confidence.value if parsed_inputs.generator_agent_confidence else self.MIN_GENERATOR_CONFIDENCE

        # --- Decision Logic (MVP: Rule-based based on confidence scores) ---
        decision: Literal["ACCEPT_ARTIFACT", "REFINEMENT_REQUIRED", "PROCEED_TO_DOCUMENTATION"] = "REFINEMENT_REQUIRED"
        reasoning = "Defaulting to refinement due to initial MVP logic or low confidence."
        next_agent_for_refinement: Optional[str] = None
        next_agent_refinement_input: Optional[Union[ProductAnalystInput, ArchitectAgentInput, MasterPlannerInput, ProjectDocumentationAgentInput, Dict[str,Any]]] = None
        final_doc_id_on_accept: Optional[str] = None

        # Simplified confidence calculation
        # In a real scenario, ARCA might look at specific content in PRAA/RTA reports.
        # For example, if PRAA reports critical risks, refinement is needed regardless of scores.
        combined_metric = generator_confidence_val * praa_confidence_val
        if parsed_inputs.artifact_type in ["Blueprint", "MasterExecutionPlan", "CodeModule"]:
            combined_metric *= rta_confidence_val

        logger_instance.info(f"ARCA Decision Logic: Artifact Type: {parsed_inputs.artifact_type}, GenConf: {generator_confidence_val:.2f}, PRAAConf: {praa_confidence_val:.2f}, RTAConf: {rta_confidence_val:.2f}, CombinedMetric: {combined_metric:.2f}")

        if combined_metric >= self.DEFAULT_ACCEPTANCE_THRESHOLD and \
           generator_confidence_val >= self.MIN_GENERATOR_CONFIDENCE and \
           praa_confidence_val >= self.MIN_PRAA_CONFIDENCE and \
           (parsed_inputs.artifact_type == "LOPRD" or rta_confidence_val >= self.MIN_RTA_CONFIDENCE):
            decision = "ACCEPT_ARTIFACT"
            reasoning = f"Artifact accepted based on confidence scores exceeding threshold ({self.DEFAULT_ACCEPTANCE_THRESHOLD:.2f}). Combined: {combined_metric:.2f}."
            final_doc_id_on_accept = parsed_inputs.artifact_doc_id
            logger_instance.info(f"ARCA Decision: ACCEPT_ARTIFACT for {parsed_inputs.artifact_doc_id} ({parsed_inputs.artifact_type}). Reasoning: {reasoning}")
        else:
            decision = "REFINEMENT_REQUIRED"
            reasoning = f"Refinement deemed necessary. Combined metric {combined_metric:.2f} or individual scores below thresholds."
            logger_instance.info(f"ARCA Decision: REFINEMENT_REQUIRED for {parsed_inputs.artifact_doc_id} ({parsed_inputs.artifact_type}). Reasoning: {reasoning}")
            
            # --- Determine which agent to call for refinement ---
            # This is a simplified placeholder. Actual logic would be more nuanced, 
            # potentially looking at which report (PRAA, RTA) triggered the low score.
            refinement_instructions_for_agent = f"Refinement requested by ARCA for {parsed_inputs.artifact_type} (ID: {parsed_inputs.artifact_doc_id}). Key concerns based on PRAA/RTA reports. Please review and address. Original combined metric: {combined_metric:.2f}. Generator Confidence: {generator_confidence_val:.2f}, PRAA Confidence: {praa_confidence_val:.2f}, RTA Confidence: {rta_confidence_val:.2f}."
            # Mock: Get user_goal from full_context if available, needed by PAA
            initial_user_goal_for_paa = "User goal not available in ARCA context for refinement input."
            if full_context and full_context.get("intermediate_outputs", {}).get("initial_goal_setup", {}).get("initial_user_goal"):
                initial_user_goal_for_paa = full_context["intermediate_outputs"]["initial_goal_setup"]["initial_user_goal"]
            elif full_context and full_context.get("initial_user_goal"): # If it was in root context
                 initial_user_goal_for_paa = full_context["initial_user_goal"]

            if parsed_inputs.artifact_type == "LOPRD":
                next_agent_for_refinement = ProductAnalystAgent_v1.AGENT_ID
                next_agent_refinement_input = ProductAnalystInput(
                    project_id=parsed_inputs.project_id,
                    user_goal=initial_user_goal_for_paa, # PAA needs the original goal, or ARCA needs to be smarter about evolving it
                    existing_loprd_doc_id=parsed_inputs.artifact_doc_id,
                    refinement_instructions=refinement_instructions_for_agent
                )
            elif parsed_inputs.artifact_type == "Blueprint":
                next_agent_for_refinement = ArchitectAgent_v1.AGENT_ID
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
                    project_id=parsed_inputs.project_id,
                    loprd_doc_id=loprd_id_for_architect, 
                    existing_blueprint_doc_id=parsed_inputs.artifact_doc_id,
                    refinement_instructions=refinement_instructions_for_agent
                )
            elif parsed_inputs.artifact_type == "MasterExecutionPlan":
                next_agent_for_refinement = "SystemMasterPlannerAgent_v1" # Or its specific sub-capability ID
                # MasterPlanner needs blueprint ID. Similar to above, assume it's available.
                blueprint_id_for_planner = "unknown_blueprint_id_for_plan_refinement"
                if full_context and full_context.get("intermediate_outputs", {}).get("arca_blueprint_coordination_output", {}).get("final_artifact_doc_id"):
                     blueprint_id_for_planner = full_context["intermediate_outputs"]["arca_blueprint_coordination_output"]["final_artifact_doc_id"]
                elif full_context and full_context.get("current_blueprint_doc_id"): # Fallback
                     blueprint_id_for_planner = full_context["current_blueprint_doc_id"]
                
                next_agent_refinement_input = MasterPlannerInput(
                    project_id=parsed_inputs.project_id,
                    blueprint_doc_id=blueprint_id_for_planner,
                    # user_goal = None, # Not primary for plan refinement from blueprint
                    refinement_instructions=refinement_instructions_for_agent,
                    # existing_plan_doc_id = parsed_inputs.artifact_doc_id # If planner supports refining existing plan doc
                )
            elif parsed_inputs.artifact_type == "CodeModule":
                # This implies refinement for CoreCodeGeneratorAgent or similar
                # For now, we don't have a direct input schema for that in ARCA's imports
                # So we'll use a dict and assume orchestrator/agent handles it.
                next_agent_for_refinement = "CoreCodeGeneratorAgent_v1" # Or SmartCodeGeneratorAgent_v1
                next_agent_refinement_input = {
                    "project_id": parsed_inputs.project_id,
                    "code_specification_doc_id": "REFINED_SPEC_FROM_ARCA_OR_PREVIOUS_STEP", # Needs context
                    "existing_code_doc_id": parsed_inputs.artifact_doc_id,
                    "refinement_instructions": refinement_instructions_for_agent,
                    # Add other fields CoreCodeGeneratorAgentInput might need for refinement
                }
                logger_instance.warning(f"ARCA: CodeModule refinement input is a generic dict. Ensure {next_agent_for_refinement} can handle it.")
            else:
                reasoning += " But no specific refinement agent configured for this artifact type."
                # No agent to call, so it's effectively an error or unhandled state for refinement
                logger_instance.error(f"ARCA: Refinement needed for {parsed_inputs.artifact_type} but no refinement path defined.")
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
        cycle_summary_for_state_manager = f"ARCA review complete for artifact {parsed_inputs.artifact_doc_id} ({parsed_inputs.artifact_type}). Decision: {decision}. Reasoning: {reasoning}."

        if decision == "REFINEMENT_REQUIRED" and not next_agent_for_refinement:
            # This case means refinement is needed but ARCA doesn't know who to send it to.
            # This is a clear issue for human review.
            issues_for_human_review_list.append({
                "issue_id": f"arca_unhandled_refinement_{parsed_inputs.artifact_doc_id}",
                "description": f"ARCA determined refinement is required for {parsed_inputs.artifact_type} (ID: {parsed_inputs.artifact_doc_id}), but no automated refinement path is defined. {reasoning}",
                "relevant_artifact_ids": [parsed_inputs.artifact_doc_id]
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

                if current_cycle_id:
                    current_cycle_item = next((c for c in project_state.cycle_history if c.cycle_id == current_cycle_id), None)
                    if current_cycle_item:
                        logger_instance.info(f"Updating cycle {current_cycle_id} in ProjectStateV2 via ARCA.")
                        current_cycle_item.arca_summary_of_cycle_outcome = (current_cycle_item.arca_summary_of_cycle_outcome or "") + f"\n- {cycle_summary_for_state_manager}"
                        
                        if issues_for_human_review_list:
                             current_cycle_item.issues_flagged_for_human_review.extend(issues_for_human_review_list)
                        
                        # If this ARCA decision implies the end of the current cycle processing (e.g., final acceptance or unrecoverable issue)
                        # set it to pending_human_review.
                        # This is a simplification; a more complex orchestrator might make this decision.
                        if decision == "ACCEPT_ARTIFACT" and parsed_inputs.artifact_type == "ProjectDocumentation": # Example: Project complete
                            project_state.overall_project_status = "project_complete" # Or a new status like "pending_final_review"
                            current_cycle_item.end_time = datetime.datetime.now(datetime.timezone.utc)
                            logger_instance.info(f"Project {project_state.project_id} marked as complete after documentation acceptance.")
                        elif (decision == "REFINEMENT_REQUIRED" and not next_agent_for_refinement) or \
                             (decision == "ACCEPT_ARTIFACT" and parsed_inputs.artifact_type != "ProjectDocumentation") or \
                             decision == "PROCEED_TO_DOCUMENTATION": # Assuming these are points for review
                            project_state.overall_project_status = "pending_human_review"
                            # We might not set end_time here if the cycle isn't truly "over" but just needs review to continue.
                            # However, if this ARCA step is the *last* thing in a cycle before review, then end_time is appropriate.
                            # For now, let's set end_time if it's pending review.
                            current_cycle_item.end_time = datetime.datetime.now(datetime.timezone.utc)
                            logger_instance.info(f"Project state for {project_state.project_id} set to 'pending_human_review' by ARCA.")

                        state_manager_instance._save_project_state() # Use the private method to save the modified state
                        logger_instance.info(f"ARCA updated ProjectStateV2 for project {parsed_inputs.project_id}, cycle {current_cycle_id}.")
                    else:
                        logger_instance.warning(f"ARCA: Current cycle ID {current_cycle_id} not found in project state history. Cannot update cycle details.")
                else:
                    logger_instance.warning("ARCA: No current_cycle_id set in ProjectStateV2. Cannot update cycle details.")
            except StatusFileError as e:
                logger_instance.error(f"ARCA: Failed to update project state due to StatusFileError: {e}", exc_info=True)
            except Exception as e: # Catch any other unexpected errors
                logger_instance.error(f"ARCA: Unexpected error updating project state: {e}", exc_info=True)
        elif not state_manager_instance:
            logger_instance.warning("ARCA: StateManager instance not available. Skipping project state update.")


        return ARCAOutput(
            task_id=parsed_inputs.task_id,
            project_id=parsed_inputs.project_id,
            reviewed_artifact_doc_id=parsed_inputs.artifact_doc_id,
            reviewed_artifact_type=parsed_inputs.artifact_type,
            decision=decision,
            decision_reasoning=reasoning,
            confidence_in_decision=arca_confidence,
            next_agent_id_for_refinement=next_agent_for_refinement,
            next_agent_input=next_agent_refinement_input.model_dump() if next_agent_refinement_input and hasattr(next_agent_refinement_input, 'model_dump') else next_agent_refinement_input,
            final_artifact_doc_id=final_doc_id_on_accept
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=AutomatedRefinementCoordinatorAgent_v1.AGENT_ID,
            name=AutomatedRefinementCoordinatorAgent_v1.AGENT_NAME,
            description=AutomatedRefinementCoordinatorAgent_v1.DESCRIPTION,
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