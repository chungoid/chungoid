from __future__ import annotations

import logging
import datetime # For potential timestamping
import uuid
import json # For LOPRD content if it's retrieved as JSON string
from typing import Any, Dict, Optional, ClassVar, Type

from pydantic import BaseModel, Field

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1
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

class ArchitectAgent_v1(BaseAgent[ArchitectAgentInput, ArchitectAgentOutput]):
    AGENT_ID: ClassVar[str] = "ArchitectAgent_v1"
    AGENT_NAME: ClassVar[str] = "Architect Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Generates a technical blueprint based on an LOPRD and project context."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "architect_agent_v1.yaml"
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.PLANNING_AND_DESIGN # MODIFIED
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[ArchitectAgentInput]] = ArchitectAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[ArchitectAgentOutput]] = ArchitectAgentOutput

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
        task_input: ArchitectAgentInput,
        full_context: Optional[Dict[str, Any]] = None,
    ) -> ArchitectAgentOutput:
        llm_provider = self._llm_provider
        prompt_manager = self._prompt_manager
        logger_instance = self._logger
        pcma_agent = self._project_chroma_manager

        if full_context:
            if "llm_provider" in full_context and full_context["llm_provider"] != llm_provider:
                 llm_provider = full_context["llm_provider"]
                 logger_instance.info("Using LLMProvider from full_context for ArchitectAgent.")
            if "prompt_manager" in full_context and full_context["prompt_manager"] != prompt_manager:
                 prompt_manager = full_context["prompt_manager"]
                 logger_instance.info("Using PromptManager from full_context for ArchitectAgent.")
            if "project_chroma_manager_agent_instance" in full_context and full_context["project_chroma_manager_agent_instance"] != pcma_agent:
                 pcma_agent = full_context["project_chroma_manager_agent_instance"]
                 logger_instance.info("Using ProjectChromaManagerAgent_v1 from full_context for ArchitectAgent.")
            if "system_context" in full_context and "logger" in full_context["system_context"]:
                if full_context["system_context"]["logger"] != self._logger: 
                    logger_instance = full_context["system_context"]["logger"]
        
        if not llm_provider or not prompt_manager or not pcma_agent:
            err_msg = "LLMProvider, PromptManager, or ProjectChromaManagerAgent not available."
            logger_instance.error(f"{err_msg} for {self.AGENT_ID}")
            _task_id = getattr(task_input, 'task_id', "unknown_task_dep_fail")
            _project_id = getattr(task_input, 'project_id', "unknown_proj_dep_fail")
            return ArchitectAgentOutput(task_id=_task_id, project_id=_project_id, status="FAILURE_CONFIGURATION", message=err_msg, error_message=err_msg)

        try:
            # parsed_inputs = ArchitectAgentInput(**inputs) # Removed this line
            pass # Placeholder
        except Exception as e:
            logger_instance.error(f"Input parsing failed: {e}", exc_info=True)
            _task_id = getattr(task_input, 'task_id', "parse_err")
            _project_id = getattr(task_input, 'project_id', "parse_err")
            return ArchitectAgentOutput(task_id=_task_id, project_id=_project_id, status="FAILURE_INPUT_VALIDATION", message=str(e), error_message=str(e))

        logger_instance.info(f"{self.AGENT_ID} invoked for task {task_input.task_id}, LOPRD ID {task_input.loprd_doc_id} in project {task_input.project_id}")

        # --- Retrieve LOPRD content & existing Blueprint (if any) from PCMA ---
        loprd_json_content_str: Optional[str] = None
        existing_blueprint_md_str: Optional[str] = None

        try:
            # Conceptual PCMA call for LOPRD:
            # loprd_artifact = await pcma_agent.get_document_content_by_id(project_id=task_input.project_id, doc_id=task_input.loprd_doc_id, collection_name="loprds_collection") # Example
            # if loprd_artifact:
            #     loprd_json_content_str = loprd_artifact # Assuming content is already stringified JSON
            # else:
            #     raise ValueError(f"LOPRD document with ID {task_input.loprd_doc_id} not found.")
            loprd_json_content_str = f'{{"mock_loprd_for_architect": true, "loprd_id": "{task_input.loprd_doc_id}", "requirements": ["req1", "req2"], "description": "This is a conceptual LOPRD content that would be fetched by PCMA."}}'
            logger_instance.info(f"Conceptual fetch for loprd_doc_id: {task_input.loprd_doc_id}")

            if task_input.existing_blueprint_doc_id:
                # existing_blueprint_artifact = await pcma_agent.get_document_content_by_id(project_id=task_input.project_id, doc_id=task_input.existing_blueprint_doc_id, collection_name="blueprints_collection") # Example
                # if existing_blueprint_artifact:
                #     existing_blueprint_md_str = existing_blueprint_artifact
                # else:
                #     logger_instance.warning(f"Existing blueprint with ID {task_input.existing_blueprint_doc_id} not found, proceeding with new generation.")
                existing_blueprint_md_str = f"### Mock Existing Blueprint for {task_input.existing_blueprint_doc_id}\nThis is conceptual existing blueprint content that would be fetched by PCMA. Details to be refined by LLM."
                logger_instance.info(f"Conceptual fetch for existing_blueprint_doc_id: {task_input.existing_blueprint_doc_id}")

        except Exception as e_pcma_fetch:
            msg = f"Failed to retrieve input artifacts via PCMA: {e_pcma_fetch}"
            logger_instance.error(msg, exc_info=True)
            return ArchitectAgentOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_INPUT_RETRIEVAL", message=msg, error_message=str(e_pcma_fetch))
        
        # --- Prompt Rendering ---
        prompt_render_data = {
            "loprd_json_string": loprd_json_content_str,
            "existing_blueprint_markdown_string": existing_blueprint_md_str,
            "refinement_instructions_string": task_input.refinement_instructions
        }
        try:
            rendered_prompts = prompt_manager.render_prompt_template(ARCHITECT_AGENT_PROMPT_NAME, prompt_render_data, sub_dir="autonomous_engine")
            system_prompt = rendered_prompts.get("system_prompt")
            main_prompt = rendered_prompts.get("prompt_details") # Or user_prompt
            if not system_prompt or not main_prompt:
                raise PromptRenderError("Missing system or main prompt content after rendering for ArchitectAgent.")
        except PromptRenderError as e_prompt:
            logger_instance.error(f"Prompt rendering failed for {ARCHITECT_AGENT_PROMPT_NAME}: {e_prompt}", exc_info=True)
            return ArchitectAgentOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_PROMPT_RENDERING", message=str(e_prompt), error_message=str(e_prompt))

        # --- LLM Interaction using generate_text_async_with_prompt_manager ---
        generated_blueprint_md: Optional[str] = None
        llm_usage_metadata: Optional[Dict[str, Any]] = None # To store potential usage data from LLM call

        try:
            logger_instance.info(f"Sending request to LLM via PromptManager for Blueprint generation/refinement (prompt: {ARCHITECT_AGENT_PROMPT_NAME})...")
            
            # Assuming generate_text_async_with_prompt_manager can return a more detailed response object
            # or we adapt it. For now, let's assume it returns the text directly.
            llm_response_text = await llm_provider.generate_text_async_with_prompt_manager(
                prompt_manager=prompt_manager,
                prompt_name=ARCHITECT_AGENT_PROMPT_NAME,
                prompt_data=prompt_render_data,
                sub_dir="autonomous_engine",
                temperature=0.5, # Higher temp for creative architecture
                # model_id="gpt-4-turbo-preview" # Or from config/LLMProvider default
            )

            # TODO: If generate_text_async_with_prompt_manager returns a richer object with usage, capture it:
            # e.g., if it returns a tuple (text, usage_dict) or an object response.text, response.usage
            # For now, assuming it just returns the text content.
            generated_blueprint_md = llm_response_text 
            # llm_usage_metadata = usage_dict # If available

            if not generated_blueprint_md or not generated_blueprint_md.strip():
                raise ValueError("LLM returned empty or whitespace-only Blueprint content.")
            logger_instance.info("Successfully received Blueprint content from LLM.")
        except Exception as e_llm: # Catch other LLM call related errors
            logger_instance.error(f"LLM interaction failed: {e_llm}", exc_info=True)
            return ArchitectAgentOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_LLM", message=str(e_llm), error_message=str(e_llm), llm_full_response=str(e_llm) if not generated_blueprint_md else generated_blueprint_md) # Store what we have

        # --- Store Blueprint in ChromaDB (via PCMA) ---
        blueprint_doc_id: Optional[str] = None
        try:
            # Conceptual PCMA call:
            # stored_artifact_response = await pcma_agent.store_text_artifact(
            #     project_id=task_input.project_id,
            #     artifact_content=generated_blueprint_md,
            #     artifact_name=f"project_blueprint_{task_input.project_id}_{uuid.uuid4().hex[:8]}.md",
            #     collection_name="blueprints_collection", # Example
            #     metadata={"loprd_source_doc_id": task_input.loprd_doc_id, "agent_id": self.AGENT_ID}
            # )
            # if stored_artifact_response and stored_artifact_response.doc_id:
            #     blueprint_doc_id = stored_artifact_response.doc_id
            #     logger_instance.info(f"Blueprint artifact (conceptual) stored with doc_id: {blueprint_doc_id}")
            # else:
            #     raise ValueError("PCMA failed to return a document ID for the stored blueprint.")
            blueprint_doc_id = f"blueprint_md_{task_input.project_id}_{uuid.uuid4().hex[:8]}.md_doc_id"
            logger_instance.info(f"Blueprint artifact (conceptual) stored with doc_id: {blueprint_doc_id}. Content length: {len(generated_blueprint_md)}")

        except Exception as e_pcma_store:
            logger_instance.error(f"Conceptual PCMA Blueprint storage failed: {e_pcma_store}", exc_info=True)
            return ArchitectAgentOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                blueprint_document_id=None, # Storage failed
                status="FAILURE_ARTIFACT_STORAGE",
                message=f"Failed to store generated Blueprint: {e_pcma_store}",
                error_message=str(e_pcma_store),
                llm_full_response=generated_blueprint_md # Still return the content if generated
            )

        confidence = ConfidenceScore(value=0.65, level="Medium", method="LLMGeneration_MVPHeuristic", reasoning="Blueprint generated by LLM from LOPRD. Further review and validation recommended.")
        if task_input.existing_blueprint_doc_id: confidence.value = 0.7 # Slightly higher if refined

        return ArchitectAgentOutput(
            task_id=task_input.task_id,
            project_id=task_input.project_id,
            blueprint_document_id=blueprint_doc_id,
            status="SUCCESS",
            message=f"Project Blueprint generated/refined. Stored as doc_id: {blueprint_doc_id}",
            confidence_score=confidence,
            llm_full_response=generated_blueprint_md, # Store the generated markdown for debugging
            usage_metadata=llm_usage_metadata # Store LLM usage if available
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=ArchitectAgent_v1.AGENT_ID,
            name=ArchitectAgent_v1.AGENT_NAME,
            description=ArchitectAgent_v1.AGENT_DESCRIPTION,
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