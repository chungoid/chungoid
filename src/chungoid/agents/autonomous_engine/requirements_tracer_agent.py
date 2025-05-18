from __future__ import annotations

import logging
import datetime # For potential timestamping
import uuid
from typing import Any, Dict, Optional, Literal, ClassVar

from pydantic import BaseModel, Field

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
# from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, PLANNING_ARTIFACTS_COLLECTION, TRACEABILITY_REPORTS_COLLECTION
from chungoid.utils.agent_registry import AgentCard # For AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard

logger = logging.getLogger(__name__)

RTA_PROMPT_NAME = "requirements_tracer_agent_v1.yaml" # In server_prompts/autonomous_engine/

# --- Input and Output Schemas for the Agent --- #

class RequirementsTracerInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this traceability task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    source_artifact_doc_id: str = Field(..., description="ChromaDB ID of the source artifact (e.g., LOPRD, previous plan).")
    source_artifact_type: Literal["LOPRD", "Blueprint", "UserStories"] = Field(..., description="Type of the source artifact.")
    target_artifact_doc_id: str = Field(..., description="ChromaDB ID of the target artifact (e.g., Blueprint, MasterExecutionPlan).")
    target_artifact_type: Literal["Blueprint", "MasterExecutionPlan", "CodeModules"] = Field(..., description="Type of the target artifact.")
    # Optional: Specific aspects to trace or previous reports for context
    # focus_aspects: Optional[List[str]] = Field(None, description="Specific aspects or requirement categories to focus the trace on.")

class RequirementsTracerOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    traceability_report_doc_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated Traceability Report (Markdown) is stored.")
    status: str = Field(..., description="Status of the traceability analysis (e.g., SUCCESS, FAILURE_LLM, FAILURE_ARTIFACT_RETRIEVAL).")
    message: str = Field(..., description="A message detailing the outcome.")
    agent_confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the completeness and accuracy of the traceability report.")
    error_message: Optional[str] = Field(None, description="Error message if status is not SUCCESS.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")

class RequirementsTracerAgent_v1(BaseAgent):
    AGENT_ID: ClassVar[str] = "RequirementsTracerAgent_v1"
    AGENT_NAME: ClassVar[str] = "Requirements Tracer Agent v1"
    DESCRIPTION: ClassVar[str] = "Generates a traceability report (Markdown) between two development artifacts (e.g., LOPRD to Blueprint)."
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS # Or custom
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
    ) -> RequirementsTracerOutput:
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
            return RequirementsTracerOutput(task_id=task_id_fb, project_id=proj_id_fb, status="FAILURE_CONFIGURATION", message=err_msg, error_message=err_msg)

        try:
            parsed_inputs = RequirementsTracerInput(**inputs)
        except Exception as e:
            logger_instance.error(f"Input parsing failed: {e}", exc_info=True)
            return RequirementsTracerOutput(task_id=inputs.get("task_id", "parse_err"), project_id=inputs.get("project_id", "parse_err"), status="FAILURE_INPUT_VALIDATION", message=str(e), error_message=str(e))

        logger_instance.info(f"{self.AGENT_ID} invoked for task {parsed_inputs.task_id} in project {parsed_inputs.project_id}. Tracing {parsed_inputs.source_artifact_type} ({parsed_inputs.source_artifact_doc_id}) to {parsed_inputs.target_artifact_type} ({parsed_inputs.target_artifact_doc_id}).")

        # --- MOCK: Retrieve artifact contents from PCMA ---
        source_artifact_content: Optional[str] = f"Mock source content for {parsed_inputs.source_artifact_type} ID {parsed_inputs.source_artifact_doc_id}."
        target_artifact_content: Optional[str] = f"Mock target content for {parsed_inputs.target_artifact_type} ID {parsed_inputs.target_artifact_doc_id}."
        # In real implementation, handle retrieval failures
        if not source_artifact_content or not target_artifact_content:
            msg = "Failed to retrieve content for source or target artifact."
            logger_instance.error(msg)
            return RequirementsTracerOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_ARTIFACT_RETRIEVAL", message=msg, error_message=msg)

        # --- Prompt Rendering ---
        prompt_render_data = {
            "source_artifact_type": parsed_inputs.source_artifact_type,
            "source_artifact_content": source_artifact_content,
            "target_artifact_type": parsed_inputs.target_artifact_type,
            "target_artifact_content": target_artifact_content
        }
        try:
            rendered_prompts = prompt_manager.render_prompt_template(RTA_PROMPT_NAME, prompt_render_data, sub_dir="autonomous_engine")
            system_prompt = rendered_prompts.get("system_prompt")
            main_prompt = rendered_prompts.get("prompt_details")
            if not system_prompt or not main_prompt:
                raise PromptRenderError("Missing system or main prompt content after rendering for RTA.")
        except PromptRenderError as e:
            logger_instance.error(f"Prompt rendering failed: {e}", exc_info=True)
            return RequirementsTracerOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_PROMPT_RENDERING", message=str(e), error_message=str(e))

        # --- LLM Interaction ---
        generated_trace_report_md: Optional[str] = None
        try:
            logger_instance.info("Sending request to LLM for traceability report generation.")
            generated_trace_report_md = await llm_provider.generate(prompt=main_prompt, system_prompt=system_prompt, temperature=0.3)
            if not generated_trace_report_md or not generated_trace_report_md.strip():
                raise ValueError("LLM returned empty or whitespace-only traceability report.")
            logger_instance.info("Successfully received traceability report from LLM.")
        except Exception as e:
            logger_instance.error(f"LLM interaction failed: {e}", exc_info=True)
            return RequirementsTracerOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_LLM", message=str(e), error_message=str(e), llm_full_response=str(e))

        # --- MOCK: Store Traceability Report in ChromaDB (via PCMA) ---
        report_doc_id = f"mock_trace_report_{parsed_inputs.project_id}_{uuid.uuid4()}_doc_id"
        # Actual PCMA storage calls would go here.

        confidence = ConfidenceScore(value=0.65, level="Medium", method="LLMGeneration_MVPHeuristic", reasoning="Traceability report generated by LLM. Manual review recommended for completeness.")

        return RequirementsTracerOutput(
            task_id=parsed_inputs.task_id,
            project_id=parsed_inputs.project_id,
            traceability_report_doc_id=report_doc_id,
            status="SUCCESS",
            message=f"Traceability report generated. Stored as doc_id: {report_doc_id}",
            agent_confidence_score=confidence,
            llm_full_response=generated_trace_report_md
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=RequirementsTracerAgent_v1.AGENT_ID,
            name=RequirementsTracerAgent_v1.AGENT_NAME,
            description=RequirementsTracerAgent_v1.DESCRIPTION,
            version=RequirementsTracerAgent_v1.VERSION,
            input_schema=RequirementsTracerInput.model_json_schema(),
            output_schema=RequirementsTracerOutput.model_json_schema(),
            categories=[cat.value for cat in [RequirementsTracerAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=RequirementsTracerAgent_v1.VISIBILITY.value,
            capability_profile={
                "analyzes_artifacts_relationship": ["LOPRD-Blueprint", "Blueprint-MasterExecutionPlan"],
                "generates_reports": ["TraceabilityReport_Markdown"],
                "primary_function": "Requirements Traceability Verification"
            },
            metadata={
                "callable_fn_path": f"{RequirementsTracerAgent_v1.__module__}.{RequirementsTracerAgent_v1.__name__}"
            }
        ) 