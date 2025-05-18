from __future__ import annotations

import logging
import datetime # For potential timestamping
import uuid
import json # For parsing LLM output if it's a JSON string
from typing import Any, Dict, Optional, Literal, ClassVar

from pydantic import BaseModel, Field

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
# from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, PLANNING_ARTIFACTS_COLLECTION, RISK_REPORTS_COLLECTION, OPTIMIZATION_REPORTS_COLLECTION, AGENT_LOGS_COLLECTION
from chungoid.utils.agent_registry import AgentCard # For AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard categories/visibility

logger = logging.getLogger(__name__)

PRAA_PROMPT_NAME = "proactive_risk_assessor_agent_v1.yaml" # In server_prompts/autonomous_engine/

# --- Input and Output Schemas for the Agent --- #

class ProactiveRiskAssessorInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this assessment task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    artifact_id: str = Field(..., description="ChromaDB ID of the artifact (LOPRD or Blueprint) to be assessed.")
    artifact_type: Literal["LOPRD", "Blueprint", "MasterExecutionPlan"] = Field(..., description="Type of the artifact being assessed.")
    # Optional: Specific areas to focus on, or context about previous reviews
    focus_areas: Optional[List[str]] = Field(None, description="Optional list of specific areas to focus the risk assessment on.")
    # previous_assessment_ids: Optional[List[str]] = Field(None, description="IDs of previous assessment reports for context, if re-assessing.")

class ProactiveRiskAssessorOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    risk_assessment_report_doc_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated Risk Assessment Report (Markdown) is stored.")
    optimization_suggestions_report_doc_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated Optimization Suggestions Report (Markdown) is stored.")
    status: str = Field(..., description="Status of the assessment (e.g., SUCCESS, FAILURE_LLM, FAILURE_ARTIFACT_RETRIEVAL).")
    message: str = Field(..., description="A message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the thoroughness and accuracy of the assessment.")
    error_message: Optional[str] = Field(None, description="Error message if status is not SUCCESS.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging or deeper analysis.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")

class ProactiveRiskAssessorAgent_v1(BaseAgent):
    AGENT_ID: ClassVar[str] = "ProactiveRiskAssessorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Proactive Risk Assessor Agent v1"
    DESCRIPTION: ClassVar[str] = "Analyzes LOPRDs, Blueprints, or Plans for potential risks, issues, and optimization opportunities."
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.PLANNING_ANALYSIS # Or custom category
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC

    _llm_provider: Optional[LLMProvider]
    _prompt_manager: Optional[PromptManager]
    _logger: logging.Logger

    def __init__(
        self, 
        llm_provider: Optional[LLMProvider] = None, 
        prompt_manager: Optional[PromptManager] = None, 
        system_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        if system_context and "logger" in system_context:
            self._logger = system_context["logger"]
        else:
            self._logger = logging.getLogger(self.AGENT_ID)

    async def invoke_async(
        self,
        inputs: Dict[str, Any],
        full_context: Optional[Dict[str, Any]] = None,
    ) -> ProactiveRiskAssessorOutput:
        llm_provider = self._llm_provider
        prompt_manager = self._prompt_manager
        logger_instance = self._logger

        if full_context:
            if not llm_provider and "llm_provider" in full_context:
                llm_provider = full_context["llm_provider"]
            if not prompt_manager and "prompt_manager" in full_context:
                prompt_manager = full_context["prompt_manager"]
            if "system_context" in full_context and "logger" in full_context["system_context"]:
                if full_context["system_context"]["logger"] != self._logger:
                    logger_instance = full_context["system_context"]["logger"]
        
        if not llm_provider or not prompt_manager:
            err_msg = "LLMProvider or PromptManager not available."
            logger_instance.error(f"{err_msg} for {self.AGENT_ID}")
            task_id_fb = inputs.get("task_id", "unknown_task_dep_fail")
            proj_id_fb = inputs.get("project_id", "unknown_proj_dep_fail")
            return ProactiveRiskAssessorOutput(task_id=task_id_fb, project_id=proj_id_fb, status="FAILURE_CONFIGURATION", message=err_msg, error_message=err_msg)

        try:
            parsed_inputs = ProactiveRiskAssessorInput(**inputs)
        except Exception as e:
            logger_instance.error(f"Input parsing failed: {e}", exc_info=True)
            return ProactiveRiskAssessorOutput(task_id=inputs.get("task_id", "parse_err"), project_id=inputs.get("project_id", "parse_err"), status="FAILURE_INPUT_VALIDATION", message=str(e), error_message=str(e))

        logger_instance.info(f"{self.AGENT_ID} invoked for task {parsed_inputs.task_id}, artifact {parsed_inputs.artifact_id} ({parsed_inputs.artifact_type}) in project {parsed_inputs.project_id}")

        # --- MOCK: Retrieve artifact content from PCMA ---
        artifact_content_str: Optional[str] = None
        # if pcma_instance:
        #     retrieved = await pcma.retrieve_artifact(PLANNING_ARTIFACTS_COLLECTION, parsed_inputs.artifact_id)
        #     if retrieved.status == "SUCCESS": artifact_content_str = retrieved.content
        #     else: # Handle error
        artifact_content_str = f"Mock content for {parsed_inputs.artifact_type} ID {parsed_inputs.artifact_id}. Contains details about XYZ."
        if not artifact_content_str:
            msg = f"Failed to retrieve content for artifact_id {parsed_inputs.artifact_id}."
            logger_instance.error(msg)
            return ProactiveRiskAssessorOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_ARTIFACT_RETRIEVAL", message=msg, error_message=msg)
        
        # --- Prompt Rendering ---
        prompt_render_data = {
            "artifact_type": parsed_inputs.artifact_type,
            "artifact_content_str": artifact_content_str,
            "focus_areas_list": parsed_inputs.focus_areas or []
        }
        try:
            rendered_prompts = prompt_manager.render_prompt_template(PRAA_PROMPT_NAME, prompt_render_data, sub_dir="autonomous_engine")
            system_prompt = rendered_prompts.get("system_prompt")
            main_prompt = rendered_prompts.get("prompt_details")
            if not system_prompt or not main_prompt:
                raise PromptRenderError("Missing system or main prompt content after rendering.")
        except PromptRenderError as e:
            logger_instance.error(f"Prompt rendering failed: {e}", exc_info=True)
            return ProactiveRiskAssessorOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_PROMPT_RENDERING", message=str(e), error_message=str(e))

        # --- LLM Interaction ---
        llm_response_json_str: Optional[str] = None
        try:
            logger_instance.info(f"Sending request to LLM for {parsed_inputs.artifact_type} assessment.")
            llm_response_json_str = await llm_provider.generate(prompt=main_prompt, system_prompt=system_prompt, temperature=0.4)
            if not llm_response_json_str or not llm_response_json_str.strip():
                raise ValueError("LLM returned empty or whitespace-only response.")
            
            # The prompt asks for JSON with two markdown strings
            parsed_llm_output = json.loads(llm_response_json_str)
            risk_report_md = parsed_llm_output.get("risk_assessment_report_markdown")
            opt_report_md = parsed_llm_output.get("optimization_suggestions_report_markdown")
            if not risk_report_md or not opt_report_md:
                raise ValueError("LLM JSON output missing required markdown report fields.")
            logger_instance.info("Successfully received and parsed assessment reports from LLM.")
        except Exception as e:
            logger_instance.error(f"LLM interaction or output parsing failed: {e}", exc_info=True)
            return ProactiveRiskAssessorOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_LLM", message=str(e), error_message=str(e), llm_full_response=llm_response_json_str)

        # --- MOCK: Store reports in ChromaDB (via PCMA) ---
        risk_doc_id = f"mock_risk_report_{parsed_inputs.project_id}_{uuid.uuid4()}_doc_id"
        opt_doc_id = f"mock_opt_report_{parsed_inputs.project_id}_{uuid.uuid4()}_doc_id"
        # Actual PCMA storage calls would go here.

        confidence = ConfidenceScore(value=0.7, level="Medium", method="LLMGeneration_MVPHeuristic", reasoning="Assessment generated by LLM based on artifact content.")

        return ProactiveRiskAssessorOutput(
            task_id=parsed_inputs.task_id,
            project_id=parsed_inputs.project_id,
            risk_assessment_report_doc_id=risk_doc_id,
            optimization_suggestions_report_doc_id=opt_doc_id,
            status="SUCCESS",
            message=f"Assessment reports generated and (mock) stored for {parsed_inputs.artifact_type} ID {parsed_inputs.artifact_id}.",
            confidence_score=confidence,
            llm_full_response=llm_response_json_str # Store the JSON string here
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=ProactiveRiskAssessorAgent_v1.AGENT_ID,
            name=ProactiveRiskAssessorAgent_v1.AGENT_NAME,
            description=ProactiveRiskAssessorAgent_v1.DESCRIPTION,
            version=ProactiveRiskAssessorAgent_v1.VERSION,
            input_schema=ProactiveRiskAssessorInput.model_json_schema(),
            output_schema=ProactiveRiskAssessorOutput.model_json_schema(),
            categories=[cat.value for cat in [ProactiveRiskAssessorAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=ProactiveRiskAssessorAgent_v1.VISIBILITY.value,
            capability_profile={
                "analyzes_artifacts": ["LOPRD", "Blueprint", "MasterExecutionPlan"],
                "generates_reports": ["RiskAssessmentReport_Markdown", "OptimizationSuggestionsReport_Markdown"],
                "primary_function": "Artifact Quality Assurance and Risk Identification"
            },
            metadata={
                "callable_fn_path": f"{ProactiveRiskAssessorAgent_v1.__module__}.{ProactiveRiskAssessorAgent_v1.__name__}"
            }
        ) 