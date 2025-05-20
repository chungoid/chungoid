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
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1,
    StoreArtifactInput,
    BLUEPRINT_ARTIFACTS_COLLECTION,
    REVIEW_REPORTS_COLLECTION,
    ARTIFACT_TYPE_BLUEPRINT_REVIEW_REPORT_MD,
    RetrieveArtifactOutput
)
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility

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

class BlueprintReviewerAgent_v1(BaseAgent[BlueprintReviewerInput, BlueprintReviewerOutput]):
    AGENT_ID: ClassVar[str] = "BlueprintReviewerAgent_v1"
    AGENT_NAME: ClassVar[str] = "Blueprint Reviewer Agent v1"
    DESCRIPTION: ClassVar[str] = "Performs an advanced review of a Project Blueprint, suggesting optimizations, architectural alternatives, and identifying subtle flaws."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "blueprint_reviewer_agent_v1.yaml"
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.QUALITY_ASSURANCE # Or custom
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar = BlueprintReviewerInput # No type hint needed for ClassVar with direct assignment

    _llm_provider: Optional[LLMProvider]
    _prompt_manager: Optional[PromptManager]
    _project_chroma_manager: Optional[ProjectChromaManagerAgent_v1]
    _logger: logging.Logger

    def __init__(
        self, 
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        project_chroma_manager: ProjectChromaManagerAgent_v1,
        system_context: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None
    ):
        super().__init__(config=config, system_context=system_context, agent_id=agent_id)
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        self._project_chroma_manager = project_chroma_manager
        if system_context and "logger" in system_context:
            self._logger = system_context["logger"]
        else:
            self._logger = logging.getLogger(self.AGENT_ID)

    async def invoke_async(
        self,
        task_input: BlueprintReviewerInput,
        full_context: Optional[Dict[str, Any]] = None,
    ) -> BlueprintReviewerOutput:
        llm_provider = self._llm_provider
        prompt_manager = self._prompt_manager
        pcma_agent = self._project_chroma_manager
        logger_instance = self._logger

        if full_context:
            if "llm_provider" in full_context and full_context["llm_provider"] != llm_provider:
                 llm_provider = full_context["llm_provider"]
                 logger_instance.info("Using LLMProvider from full_context for BlueprintReviewerAgent.")
            if "prompt_manager" in full_context and full_context["prompt_manager"] != prompt_manager:
                 prompt_manager = full_context["prompt_manager"]
                 logger_instance.info("Using PromptManager from full_context for BlueprintReviewerAgent.")
            if "project_chroma_manager_agent_instance" in full_context and full_context["project_chroma_manager_agent_instance"] != pcma_agent:
                 pcma_agent = full_context["project_chroma_manager_agent_instance"]
                 logger_instance.info("Using ProjectChromaManagerAgent_v1 from full_context for BlueprintReviewerAgent.")
            if "system_context" in full_context and "logger" in full_context["system_context"]:
                if full_context["system_context"]["logger"] != self._logger: 
                    logger_instance = full_context["system_context"]["logger"]
        
        if not llm_provider or not prompt_manager or not pcma_agent:
            err_msg = "LLMProvider, PromptManager, or ProjectChromaManagerAgent not available."
            logger_instance.error(f"{err_msg} for {self.AGENT_ID}")
            _task_id = getattr(task_input, 'task_id', "unknown_task_dep_fail")
            _project_id = getattr(task_input, 'project_id', "unknown_proj_dep_fail")
            return BlueprintReviewerOutput(task_id=_task_id, project_id=_project_id, status="FAILURE_CONFIGURATION", message=err_msg, error_message=err_msg)

        try:
            # parsed_inputs = BlueprintReviewerInput(**inputs) # Input is already parsed
            pass
        except Exception as e: # Should not happen if task_input is already BlueprintReviewerInput
            logger_instance.error(f"Input validation/access failed: {e}", exc_info=True)
            _task_id = getattr(task_input, 'task_id', "parse_err")
            _project_id = getattr(task_input, 'project_id', "parse_err")
            return BlueprintReviewerOutput(task_id=_task_id, project_id=_project_id, status="FAILURE_INPUT_VALIDATION", message=str(e), error_message=str(e))

        logger_instance.info(f"{self.AGENT_ID} invoked for task {task_input.task_id}, blueprint {task_input.blueprint_doc_id} in project {task_input.project_id}")

        # --- Retrieve Blueprint content and previous reviews from PCMA (Conceptual) ---
        blueprint_md_content: Optional[str] = None
        previous_reviews_content_list: List[str] = []

        try:
            # Actual PCMA call for Blueprint:
            doc_output: RetrieveArtifactOutput = await pcma_agent.retrieve_artifact(
                base_collection_name=BLUEPRINT_ARTIFACTS_COLLECTION,
                document_id=task_input.blueprint_doc_id
            )
            if doc_output and doc_output.status == "SUCCESS" and doc_output.content:
                blueprint_md_content = str(doc_output.content)
                logger_instance.debug(f"Retrieved blueprint_doc_id: {task_input.blueprint_doc_id}")
            else:
                raise ValueError(f"Blueprint document with ID {task_input.blueprint_doc_id} not found, content empty, or retrieval failed. Status: {doc_output.status if doc_output else 'N/A'}")

            if task_input.previous_review_doc_ids:
                for rev_id in task_input.previous_review_doc_ids:
                    # Actual PCMA call for previous review:
                    review_doc_output: RetrieveArtifactOutput = await pcma_agent.retrieve_artifact(
                        base_collection_name=REVIEW_REPORTS_COLLECTION, # Previous reviews are in REVIEW_REPORTS_COLLECTION
                        document_id=rev_id
                    )
                    if review_doc_output and review_doc_output.status == "SUCCESS" and review_doc_output.content:
                        previous_reviews_content_list.append(str(review_doc_output.content))
                        logger_instance.debug(f"Retrieved previous_review_doc_id: {rev_id}")
                    else:
                        logger_instance.warning(f"Previous review with ID {rev_id} from {REVIEW_REPORTS_COLLECTION} not found or content empty for project {task_input.project_id}. Status: {review_doc_output.status if review_doc_output else 'N/A'}")
            
        except Exception as e_pcma_fetch:
            msg = f"Failed to retrieve input artifacts via PCMA: {e_pcma_fetch}"
            logger_instance.error(msg, exc_info=True)
            return BlueprintReviewerOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_INPUT_RETRIEVAL", message=msg, error_message=str(e_pcma_fetch))

        previous_reviews_combined_str = "\\n\\n---\\n\\n".join(previous_reviews_content_list) if previous_reviews_content_list else None

        # --- Prompt Rendering ---
        prompt_render_data = {
            "project_blueprint_markdown": blueprint_md_content,
            "previous_reviews_markdown_str": previous_reviews_combined_str,
            "specific_focus_areas_list": task_input.specific_focus_areas or []
        }
        try:
            rendered_prompts = prompt_manager.render_prompt_template(BLUEPRINT_REVIEWER_PROMPT_NAME, prompt_render_data, sub_dir="autonomous_engine")
            # system_prompt = rendered_prompts.get("system_prompt") # Will be handled by generate_text_async_with_prompt_manager
            # main_prompt = rendered_prompts.get("prompt_details") # Will be handled by generate_text_async_with_prompt_manager
            # if not system_prompt or not main_prompt:
            #     raise PromptRenderError("Missing system or main prompt content after rendering for BlueprintReviewer.")
        except PromptRenderError as e_prompt:
            logger_instance.error(f"Prompt rendering preparation failed: {e_prompt}", exc_info=True)
            return BlueprintReviewerOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_PROMPT_RENDERING", message=str(e_prompt), error_message=str(e_prompt))

        # --- LLM Interaction using generate_text_async_with_prompt_manager ---
        generated_review_report_md: Optional[str] = None
        llm_usage_metadata: Optional[Dict[str, Any]] = None

        try:
            logger_instance.info(f"Sending request to LLM via PromptManager for Blueprint review (prompt: {BLUEPRINT_REVIEWER_PROMPT_NAME})...")
            
            generated_review_report_md = await llm_provider.generate_text_async_with_prompt_manager(
                prompt_manager=prompt_manager,
                prompt_name=BLUEPRINT_REVIEWER_PROMPT_NAME,
                prompt_data=prompt_render_data,
                sub_dir="autonomous_engine",
                temperature=0.6, 
                # model_id="gpt-4-turbo-preview" # Or from config/LLMProvider default
            )
            # llm_usage_metadata = usage_dict # If generate_text_async_with_prompt_manager returns it

            if not generated_review_report_md or not generated_review_report_md.strip():
                raise ValueError("LLM returned empty or whitespace-only review report.")
            logger_instance.info("Successfully received review report from LLM.")
        except Exception as e_llm: 
            logger_instance.error(f"LLM interaction failed: {e_llm}", exc_info=True)
            return BlueprintReviewerOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_LLM", message=str(e_llm), error_message=str(e_llm), llm_full_response=str(e_llm) if not generated_review_report_md else generated_review_report_md)

        # --- Store Review Report in ChromaDB (via PCMA) ---
        report_doc_id: Optional[str] = None
        storage_success = True
        storage_error_message = None
        
        try:
            review_metadata = {
                "artifact_type": ARTIFACT_TYPE_BLUEPRINT_REVIEW_REPORT_MD,
                "source_blueprint_doc_id": task_input.blueprint_doc_id,
                "related_previous_review_ids": task_input.previous_review_doc_ids or [],
                "focus_areas_input": task_input.specific_focus_areas or [],
                "generated_by_agent": self.AGENT_ID,
                "project_id": task_input.project_id,
                "task_id": task_input.task_id,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            # If LLM provides a confidence score for the review itself, it should be parsed and added here.
            # For now, the agent_output_confidence is separate.

            store_input_review = StoreArtifactInput(
                base_collection_name=REVIEW_REPORTS_COLLECTION,
                artifact_content=generated_review_report_md,
                metadata=review_metadata,
                source_agent_id=self.AGENT_ID,
                source_task_id=task_input.task_id,
                cycle_id=full_context.get("cycle_id") if full_context else None
            )
            store_result = await pcma_agent.store_artifact(args=store_input_review)

            if store_result.status == "SUCCESS" and store_result.document_id:
                report_doc_id = store_result.document_id
                logger_instance.info(f"Blueprint review report stored with doc_id: {report_doc_id}")
            else:
                storage_success = False
                storage_error_message = store_result.error_message or f"Failed to store {ARTIFACT_TYPE_BLUEPRINT_REVIEW_REPORT_MD}"
                logger_instance.error(f"PCMA review report storage failed: {storage_error_message}")
            
        except Exception as e_pcma_store:
            storage_success = False
            storage_error_message = str(e_pcma_store)
            logger_instance.error(f"Exception during PCMA review report storage: {e_pcma_store}", exc_info=True)

        final_status = "SUCCESS" if storage_success and report_doc_id else "FAILURE_ARTIFACT_STORAGE"
        final_message = f"Blueprint review report generated. Stored as doc_id: {report_doc_id}" if storage_success and report_doc_id else f"Failed to store generated review report: {storage_error_message}"

        agent_output_confidence = ConfidenceScore(value=0.70, level="MediumHigh", method="LLMGeneration_MVPHeuristic", reasoning="Blueprint review generated by LLM. Quality depends on LLM's insight and prompt adherence.")
        if not storage_success: agent_output_confidence.value = 0.3 # Lower if storage failed

        return BlueprintReviewerOutput(
            task_id=task_input.task_id,
            project_id=task_input.project_id,
            review_report_doc_id=report_doc_id,
            status=final_status,
            message=final_message,
            confidence_score=agent_output_confidence,
            llm_full_response=generated_review_report_md,
            usage_metadata=llm_usage_metadata,
            error_message=storage_error_message if not storage_success else None
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