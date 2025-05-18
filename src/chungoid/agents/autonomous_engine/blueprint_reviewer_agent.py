from __future__ import annotations

import logging
import datetime
import uuid
from typing import Any, Dict, Optional, ClassVar, List

from pydantic import BaseModel, Field

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
# from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, PLANNING_ARTIFACTS_COLLECTION, AGENT_LOGS_COLLECTION, OPTIMIZATION_REPORTS_COLLECTION # When PCMA is integrated
from chungoid.utils.agent_registry import AgentCard # For AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard

logger = logging.getLogger(__name__)

BLUEPRINT_REVIEWER_PROMPT_NAME = "blueprint_reviewer_agent_v1.yaml"

# --- Input and Output Schemas for the Agent --- #

class BlueprintReviewerInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this review task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    blueprint_doc_id: str = Field(..., description="ChromaDB ID of the Project Blueprint (Markdown) to be reviewed.")
    # Optional context for the review
    previous_review_doc_ids: Optional[List[str]] = Field(None, description="ChromaDB IDs of any previous review reports for this blueprint, for context.")
    specific_focus_areas: Optional[List[str]] = Field(None, description="List of specific areas or concerns to focus the review on.")

class BlueprintReviewerOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    review_report_doc_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated Blueprint Review Report (Markdown, detailing optimizations, alternatives, flaws) is stored.")
    status: str = Field(..., description="Status of the review (e.g., SUCCESS, FAILURE_LLM, FAILURE_INPUT_RETRIEVAL).")
    message: str = Field(..., description="A message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the thoroughness and insightfulness of its review.")
    error_message: Optional[str] = Field(None, description="Error message if status is not SUCCESS.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")

class BlueprintReviewerAgent_v1(BaseAgent):
    AGENT_ID: ClassVar[str] = "BlueprintReviewerAgent_v1"
    AGENT_NAME: ClassVar[str] = "Blueprint Reviewer Agent v1"
    DESCRIPTION: ClassVar[str] = "Performs an advanced review of a Project Blueprint, suggesting optimizations, architectural alternatives, and identifying subtle flaws."
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.DESIGN_REVIEW # Or custom
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
    ) -> BlueprintReviewerOutput:
        llm_provider = self._llm_provider
        prompt_manager = self._prompt_manager
        logger_instance = self._logger

        if full_context:
            if not llm_provider and "llm_provider" in full_context: llm_provider = full_context["llm_provider"]
            if not prompt_manager and "prompt_manager" in full_context: prompt_manager = full_context["prompt_manager"]
            if "system_context" in full_context and "logger" in full_context["system_context"]:
                if full_context["system_context"]["logger"] != self._logger: 
                    logger_instance = full_context["system_context"]["logger"]
        
        if not llm_provider or not prompt_manager:
            err_msg = "LLMProvider or PromptManager not available."
            logger_instance.error(f"{err_msg} for {self.AGENT_ID}")
            task_id_fb = inputs.get("task_id", "unknown_task_dep_fail")
            proj_id_fb = inputs.get("project_id", "unknown_proj_dep_fail")
            return BlueprintReviewerOutput(task_id=task_id_fb, project_id=proj_id_fb, status="FAILURE_CONFIGURATION", message=err_msg, error_message=err_msg)

        try:
            parsed_inputs = BlueprintReviewerInput(**inputs)
        except Exception as e:
            logger_instance.error(f"Input parsing failed: {e}", exc_info=True)
            return BlueprintReviewerOutput(task_id=inputs.get("task_id", "parse_err"), project_id=inputs.get("project_id", "parse_err"), status="FAILURE_INPUT_VALIDATION", message=str(e), error_message=str(e))

        logger_instance.info(f"{self.AGENT_ID} invoked for task {parsed_inputs.task_id}, blueprint {parsed_inputs.blueprint_doc_id} in project {parsed_inputs.project_id}")

        # --- MOCK: Retrieve Blueprint content and previous reviews from PCMA ---
        blueprint_md_content: Optional[str] = f"### Mock Blueprint for {parsed_inputs.blueprint_doc_id}\nThis is a detailed blueprint..."
        # In real implementation, handle retrieval failures
        if not blueprint_md_content:
            msg = f"Failed to retrieve content for blueprint_doc_id {parsed_inputs.blueprint_doc_id}."
            logger_instance.error(msg)
            return BlueprintReviewerOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_INPUT_RETRIEVAL", message=msg, error_message=msg)
        
        previous_reviews_content_list = []
        if parsed_inputs.previous_review_doc_ids:
            for rev_id in parsed_inputs.previous_review_doc_ids:
                # Mock retrieval
                previous_reviews_content_list.append(f"Mock review content for {rev_id}: Suggests refactoring X.")
        previous_reviews_combined_str = "\n\n---\n\n".join(previous_reviews_content_list) if previous_reviews_content_list else None

        # --- Prompt Rendering ---
        prompt_render_data = {
            "project_blueprint_markdown": blueprint_md_content,
            "previous_reviews_markdown_str": previous_reviews_combined_str,
            "specific_focus_areas_list": parsed_inputs.specific_focus_areas or []
        }
        try:
            rendered_prompts = prompt_manager.render_prompt_template(BLUEPRINT_REVIEWER_PROMPT_NAME, prompt_render_data, sub_dir="autonomous_engine")
            system_prompt = rendered_prompts.get("system_prompt")
            main_prompt = rendered_prompts.get("prompt_details")
            if not system_prompt or not main_prompt:
                raise PromptRenderError("Missing system or main prompt content after rendering for BlueprintReviewer.")
        except PromptRenderError as e:
            logger_instance.error(f"Prompt rendering failed: {e}", exc_info=True)
            return BlueprintReviewerOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_PROMPT_RENDERING", message=str(e), error_message=str(e))

        # --- LLM Interaction ---
        generated_review_report_md: Optional[str] = None
        try:
            logger_instance.info("Sending request to LLM for Blueprint review.")
            generated_review_report_md = await llm_provider.generate(prompt=main_prompt, system_prompt=system_prompt, temperature=0.6) # Higher temp for insightful review
            if not generated_review_report_md or not generated_review_report_md.strip():
                raise ValueError("LLM returned empty or whitespace-only review report.")
            logger_instance.info("Successfully received review report from LLM.")
        except Exception as e:
            logger_instance.error(f"LLM interaction failed: {e}", exc_info=True)
            return BlueprintReviewerOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_LLM", message=str(e), error_message=str(e), llm_full_response=str(e))

        # --- MOCK: Store Review Report in ChromaDB (via PCMA) ---
        report_doc_id = f"mock_blueprint_review_{parsed_inputs.project_id}_{uuid.uuid4()}_doc_id"
        # Actual PCMA storage calls would go here.

        confidence = ConfidenceScore(value=0.75, level="High", method="LLMGeneration_MVPHeuristic", reasoning="Blueprint review generated by LLM. Subjective quality depends on LLM's architectural insight.")

        return BlueprintReviewerOutput(
            task_id=parsed_inputs.task_id,
            project_id=parsed_inputs.project_id,
            review_report_doc_id=report_doc_id,
            status="SUCCESS",
            message=f"Blueprint review report generated. Stored as doc_id: {report_doc_id}",
            confidence_score=confidence,
            llm_full_response=generated_review_report_md
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=BlueprintReviewerAgent_v1.AGENT_ID,
            name=BlueprintReviewerAgent_v1.AGENT_NAME,
            description=BlueprintReviewerAgent_v1.DESCRIPTION,
            version=BlueprintReviewerAgent_v1.VERSION,
            input_schema=BlueprintReviewerInput.model_json_schema(),
            output_schema=BlueprintReviewerOutput.model_json_schema(),
            categories=[cat.value for cat in [BlueprintReviewerAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=BlueprintReviewerAgent_v1.VISIBILITY.value,
            capability_profile={
                "reviews_artifacts": ["ProjectBlueprint_Markdown"],
                "generates_reports": ["BlueprintReviewReport_Markdown"],
                "focus": ["AdvancedOptimizations", "ArchitecturalAlternatives", "DesignFlaws"],
                "primary_function": "Expert Architectural Review"
            },
            metadata={
                "callable_fn_path": f"{BlueprintReviewerAgent_v1.__module__}.{BlueprintReviewerAgent_v1.__name__}"
            }
        ) 