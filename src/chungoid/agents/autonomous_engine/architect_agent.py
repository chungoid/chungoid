from __future__ import annotations

import logging
import datetime # For potential timestamping
import uuid
import json # For LOPRD content if it's retrieved as JSON string
from typing import Any, Dict, Optional, ClassVar

from pydantic import BaseModel, Field

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
# from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, PLANNING_ARTIFACTS_COLLECTION, AGENT_LOGS_COLLECTION
from chungoid.utils.agent_registry import AgentCard # For AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard

logger = logging.getLogger(__name__)

ARCHITECT_AGENT_PROMPT_NAME = "architect_agent_v1.yaml" # In server_prompts/autonomous_engine/

# --- Input and Output Schemas for the Agent --- #

class ArchitectAgentInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this Blueprint generation task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    loprd_doc_id: str = Field(..., description="ChromaDB ID of the LOPRD (JSON artifact) to be used as input.")
    existing_blueprint_doc_id: Optional[str] = Field(None, description="ChromaDB ID of an existing Blueprint to refine, if any.")
    refinement_instructions: Optional[str] = Field(None, description="Specific instructions for refining an existing Blueprint.")
    # target_technologies: Optional[List[str]] = Field(None, description="Preferred technologies or constraints for the architecture.")

class ArchitectAgentOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    blueprint_document_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated/updated Project Blueprint (Markdown) is stored.")
    status: str = Field(..., description="Status of the Blueprint generation (e.g., SUCCESS, FAILURE_LLM, FAILURE_INPUT_RETRIEVAL).")
    message: str = Field(..., description="A message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the quality and completeness of the Blueprint.")
    error_message: Optional[str] = Field(None, description="Error message if status is not SUCCESS.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")

class ArchitectAgent_v1(BaseAgent):
    AGENT_ID: ClassVar[str] = "ArchitectAgent_v1"
    AGENT_NAME: ClassVar[str] = "Architect Agent v1"
    DESCRIPTION: ClassVar[str] = "Generates and refines a Project Blueprint (Markdown) from an LOPRD (JSON)."
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.DESIGN_ARCHITECTURE # Or custom
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
    ) -> ArchitectAgentOutput:
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
            return ArchitectAgentOutput(task_id=task_id_fb, project_id=proj_id_fb, status="FAILURE_CONFIGURATION", message=err_msg, error_message=err_msg)

        try:
            parsed_inputs = ArchitectAgentInput(**inputs)
        except Exception as e:
            logger_instance.error(f"Input parsing failed: {e}", exc_info=True)
            return ArchitectAgentOutput(task_id=inputs.get("task_id", "parse_err"), project_id=inputs.get("project_id", "parse_err"), status="FAILURE_INPUT_VALIDATION", message=str(e), error_message=str(e))

        logger_instance.info(f"{self.AGENT_ID} invoked for task {parsed_inputs.task_id}, LOPRD ID {parsed_inputs.loprd_doc_id} in project {parsed_inputs.project_id}")

        # --- MOCK: Retrieve LOPRD content & existing Blueprint (if any) from PCMA ---
        loprd_json_content_str: Optional[str] = None
        # if pcma_instance:
        #     retrieved_loprd = await pcma.retrieve_artifact(PLANNING_ARTIFACTS_COLLECTION, parsed_inputs.loprd_doc_id)
        #     if retrieved_loprd.status == "SUCCESS": loprd_json_content_str = retrieved_loprd.content
        #     else: # Handle error
        loprd_json_content_str = f'{{"mock_loprd_for_architect": true, "loprd_id": "{parsed_inputs.loprd_doc_id}", "requirements": ["req1", "req2"]}}'
        if not loprd_json_content_str:
            msg = f"Failed to retrieve LOPRD content for doc_id {parsed_inputs.loprd_doc_id}."
            logger_instance.error(msg)
            return ArchitectAgentOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_INPUT_RETRIEVAL", message=msg, error_message=msg)
        
        existing_blueprint_md_str: Optional[str] = None
        if parsed_inputs.existing_blueprint_doc_id:
            existing_blueprint_md_str = f"### Mock Existing Blueprint for {parsed_inputs.existing_blueprint_doc_id}\nDetails to be refined."

        # --- Prompt Rendering ---
        prompt_render_data = {
            "loprd_json_string": loprd_json_content_str,
            "existing_blueprint_markdown_string": existing_blueprint_md_str,
            "refinement_instructions_string": parsed_inputs.refinement_instructions
        }
        try:
            rendered_prompts = prompt_manager.render_prompt_template(ARCHITECT_AGENT_PROMPT_NAME, prompt_render_data, sub_dir="autonomous_engine")
            system_prompt = rendered_prompts.get("system_prompt")
            main_prompt = rendered_prompts.get("prompt_details") # Or user_prompt
            if not system_prompt or not main_prompt:
                raise PromptRenderError("Missing system or main prompt content after rendering for ArchitectAgent.")
        except PromptRenderError as e:
            logger_instance.error(f"Prompt rendering failed: {e}", exc_info=True)
            return ArchitectAgentOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_PROMPT_RENDERING", message=str(e), error_message=str(e))

        # --- LLM Interaction ---
        generated_blueprint_md: Optional[str] = None
        try:
            logger_instance.info(f"Sending request to LLM for Blueprint generation/refinement.")
            generated_blueprint_md = await llm_provider.generate(prompt=main_prompt, system_prompt=system_prompt, temperature=0.5) # Higher temp for creative architecture
            if not generated_blueprint_md or not generated_blueprint_md.strip():
                raise ValueError("LLM returned empty or whitespace-only Blueprint content.")
            logger_instance.info("Successfully received Blueprint content from LLM.")
        except Exception as e:
            logger_instance.error(f"LLM interaction failed: {e}", exc_info=True)
            return ArchitectAgentOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_LLM", message=str(e), error_message=str(e), llm_full_response=str(e))

        # --- MOCK: Store Blueprint in ChromaDB (via PCMA) ---
        blueprint_doc_id = f"mock_blueprint_{parsed_inputs.project_id}_{uuid.uuid4()}_doc_id"
        # Actual PCMA storage calls would go here.

        confidence = ConfidenceScore(value=0.65, level="Medium", method="LLMGeneration_MVPHeuristic", reasoning="Blueprint generated by LLM from LOPRD. Further review and validation recommended.")
        if parsed_inputs.existing_blueprint_doc_id: confidence.value = 0.7 # Slightly higher if refined

        return ArchitectAgentOutput(
            task_id=parsed_inputs.task_id,
            project_id=parsed_inputs.project_id,
            blueprint_document_id=blueprint_doc_id,
            status="SUCCESS",
            message=f"Project Blueprint generated/refined. Stored as doc_id: {blueprint_doc_id}",
            confidence_score=confidence,
            llm_full_response=generated_blueprint_md # Store the generated markdown for debugging
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=ArchitectAgent_v1.AGENT_ID,
            name=ArchitectAgent_v1.AGENT_NAME,
            description=ArchitectAgent_v1.DESCRIPTION,
            version=ArchitectAgent_v1.VERSION,
            input_schema=ArchitectAgentInput.model_json_schema(),
            output_schema=ArchitectAgentOutput.model_json_schema(),
            categories=[cat.value for cat in [ArchitectAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=ArchitectAgent_v1.VISIBILITY.value,
            capability_profile={
                "generates_artifacts": ["ProjectBlueprint_Markdown"],
                "consumes_artifacts": ["LOPRD_JSON", "ExistingBlueprint_Markdown", "RefinementInstructions"],
                "primary_function": "Architectural Design and Blueprint Generation"
            },
            metadata={
                "callable_fn_path": f"{ArchitectAgent_v1.__module__}.{ArchitectAgent_v1.__name__}"
            }
        ) 