from __future__ import annotations

import logging
from typing import Any, Dict, Optional, ClassVar, TYPE_CHECKING
import json
from pathlib import Path
import uuid

from pydantic import BaseModel, Field, ValidationError

from chungoid.runtime.agents.agent_base import BaseAgent, InputSchema, OutputSchema
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptDefinition
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
# Conditionally import AgentProvider
if TYPE_CHECKING:
    from chungoid.utils.agent_resolver import AgentProvider
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1,
    LOPRD_ARTIFACTS_COLLECTION,
    ARTIFACT_TYPE_LOPRD_JSON
)
from chungoid.schemas.chroma_agent_io_schemas import StoreArtifactInput

logger = logging.getLogger(__name__)

class SystemRequirementsGatheringInput(BaseModel):
    user_goal: str = Field(..., description="The high-level user goal.")
    project_context_summary: Optional[str] = Field(None, description="Optional summary of the existing project context.")
    # Add other fields as necessary, e.g., existing requirements, constraints

class SystemRequirementsGatheringOutput(BaseModel):
    refined_requirements_document_id: Optional[str] = Field(None, description="ID of the document artifact containing the refined requirements (e.g., in ChromaDB).")
    requirements_summary: str = Field(..., description="A textual summary of the gathered and refined requirements.")
    # Add other fields, e.g., structured requirements data

class SystemRequirementsGatheringAgent_v1(BaseAgent[SystemRequirementsGatheringInput, SystemRequirementsGatheringOutput]):
    AGENT_ID: ClassVar[str] = "SystemRequirementsGatheringAgent_v1"
    AGENT_NAME: ClassVar[str] = "System Requirements Gathering Agent"
    VERSION: ClassVar[str] = "1.0.0"
    DESCRIPTION: ClassVar[str] = "Gathers and refines system requirements based on an initial user goal or problem statement."
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL # Or PUBLIC if it can be invoked directly

    # Declare fields for dependencies injected in __init__
    llm_provider: LLMProvider
    prompt_manager: PromptManager
    agent_provider: Optional['AgentProvider'] = None
    loprd_generation_prompt_template_obj: Optional[PromptDefinition] = None
    loprd_generation_inline_fallback_prompt: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, agent_provider: Optional['AgentProvider'] = None, system_context: Optional[Dict[str, Any]] = None):
        # Pass llm_provider and prompt_manager to super().__init__ for Pydantic validation
        super().__init__(
            system_context=system_context, 
            llm_provider=llm_provider, 
            prompt_manager=prompt_manager,
            agent_provider=agent_provider
        )
        if not llm_provider:
            raise ValueError("LLMProvider is required for SystemRequirementsGatheringAgent_v1")
        if not prompt_manager:
            raise ValueError("PromptManager is required for SystemRequirementsGatheringAgent_v1")
        # self.llm_provider, self.prompt_manager, self.agent_provider are now set by Pydantic

        # Load the LOPRD generation prompt using PromptManager
        try:
            self.loprd_generation_prompt_template_obj = self.prompt_manager.get_prompt_definition(
                prompt_name="system_requirements_gathering_v1",
                prompt_version="1.0",
                sub_path="autonomous_engine"
            )
            self.loprd_generation_inline_fallback_prompt = None
        except FileNotFoundError:
            logger.error("LOPRD prompt file 'autonomous_engine/system_requirements_gathering_agent_v1_prompt.yaml' not found by PromptManager.")
            # Fallback to a basic inline prompt if file not found, though this indicates a setup issue.
            self.loprd_generation_prompt_template_obj = None # Mark as None to use inline string later
            self.loprd_generation_inline_fallback_prompt = """
            You are an expert requirements analyst. Generate a JSON LOPRD for: {user_goal}. Context: {project_context_summary}.
            JSON should include: loprd_id, document_version, user_goal_received, execution_summary, detailed_requirements, acceptance_criteria, relevant_technologies, potential_risks.
            """
            logger.warning("Using a basic inline fallback prompt for LOPRD generation.")
        except Exception as e:
            logger.error(f"Error loading LOPRD prompt: {e}", exc_info=True)
            self.loprd_generation_prompt_template_obj = None
            self.loprd_generation_inline_fallback_prompt = "Error loading prompt. Goal: {user_goal}."
            logger.warning("Using error fallback prompt for LOPRD generation.")

    async def _get_project_chroma_manager(self, project_id: str, mcp_root_path: Path) -> ProjectChromaManagerAgent_v1:
        if not self.agent_provider:
            logger.error(f"{self.AGENT_ID}: AgentProvider not available, cannot get ProjectChromaManagerAgent_v1.")
            raise RuntimeError("AgentProvider not available to resolve ProjectChromaManagerAgent_v1")
        
        if not mcp_root_path:
             logger.error("MCP root path not provided to _get_project_chroma_manager. This is a critical error.")
             raise ValueError("Valid MCP root path is required for ProjectChromaManagerAgent_v1 instantiation.")

        try:
            pcma_instance = ProjectChromaManagerAgent_v1(
                project_root_workspace_path=str(mcp_root_path),
                project_id=project_id
            )
            return pcma_instance
        except Exception as e:
            logger.error(f"Failed to instantiate ProjectChromaManagerAgent_v1 with mcp_root_path='{mcp_root_path}': {e}", exc_info=True)
            raise

    async def invoke_async(
        self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None
    ) -> SystemRequirementsGatheringOutput:
        try:
            parsed_inputs = SystemRequirementsGatheringInput(**inputs)
        except ValidationError as e:
            logger.error(f"Input validation error for {self.AGENT_ID} ({self.AGENT_NAME}): {e}", exc_info=True)
            raise

        logger.info(f"{self.AGENT_NAME} ({self.AGENT_ID}) invoked with goal: {parsed_inputs.user_goal}")

        # 1. Render prompt using inputs
        rendered_prompt = ""
        if self.loprd_generation_prompt_template_obj:
            prompt_render_data = {
                "user_goal": parsed_inputs.user_goal,
                "project_context_summary": parsed_inputs.project_context_summary or "N/A"
            }
            try:
                rendered_prompt = self.prompt_manager.get_rendered_prompt_template(
                    self.loprd_generation_prompt_template_obj.user_prompt_template,
                    prompt_render_data
                )
            except Exception as e_render:
                logger.error(f"Error rendering LOPRD prompt from PromptDefinition: {e_render}", exc_info=True)
                if self.loprd_generation_inline_fallback_prompt:
                    rendered_prompt = self.loprd_generation_inline_fallback_prompt.format(
                        user_goal=parsed_inputs.user_goal,
                        project_context_summary=parsed_inputs.project_context_summary or "N/A"
                    )
                else:
                    rendered_prompt = f"Error rendering. Goal: {parsed_inputs.user_goal}"

        elif self.loprd_generation_inline_fallback_prompt:
            rendered_prompt = self.loprd_generation_inline_fallback_prompt.format(
                user_goal=parsed_inputs.user_goal,
                project_context_summary=parsed_inputs.project_context_summary or "N/A"
            )
        else:
            logger.error("Critical: No LOPRD prompt template object and no inline fallback prompt available.")
            rendered_prompt = f"No prompt configured. Goal: {parsed_inputs.user_goal}"

        # 2. Call LLM to generate LOPRD JSON content
        try:
            # Assuming the LLM is prompted to return a JSON string that can be parsed into a Dict
            logger.info(f"Generating LOPRD content for goal: {parsed_inputs.user_goal}")

            llm_call_args = {
                "prompt": rendered_prompt, # User-facing prompt
                # System prompt is defined by OpenAILLMProvider or can be passed if needed
                # For now, relying on OpenAILLMProvider's default or a generic one passed here.
                # TODO: Standardize how system prompts from PromptDefinition are passed to LLMProvider.generate
                "system_prompt": "You are an AI assistant that generates detailed project requirements documents in JSON format." # Placeholder
            }

            if self.loprd_generation_prompt_template_obj and self.loprd_generation_prompt_template_obj.model_settings:
                model_settings = self.loprd_generation_prompt_template_obj.model_settings
                llm_call_args["model_id"] = model_settings.model_name
                llm_call_args["temperature"] = model_settings.temperature
                if model_settings.max_tokens:
                    llm_call_args["max_tokens"] = model_settings.max_tokens
                # Add any other specific provider_kwargs from model_settings if necessary
                # For example, if model_settings has a field like provider_specific_params: dict

            loprd_content_str = await self.llm_provider.generate(**llm_call_args) # CORRECTED: generate instead of generate_text and unpacked args
            
            loprd_content_json = json.loads(loprd_content_str) # Parse the JSON string from LLM
            logger.info(f"LOPRD JSON content generated successfully.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LOPRD JSON from LLM response: {e}. Response was: {loprd_content_str[:500]}...", exc_info=True)
            # Fallback or error handling: create a simple summary
            requirements_summary_placeholder = f"Failed to generate structured LOPRD. Basic goal: '{parsed_inputs.user_goal}'. Context: {parsed_inputs.project_context_summary or 'N/A'}."
            return SystemRequirementsGatheringOutput(
                refined_requirements_document_id=None,
                requirements_summary=requirements_summary_placeholder
            )
        except Exception as e:
            logger.error(f"Error during LLM call for LOPRD generation: {e}", exc_info=True)
            requirements_summary_placeholder = f"Error in LOPRD generation. Basic goal: '{parsed_inputs.user_goal}'. Context: {parsed_inputs.project_context_summary or 'N/A'}."
            return SystemRequirementsGatheringOutput(
                refined_requirements_document_id=None,
                requirements_summary=requirements_summary_placeholder
            )

        # 3. Store LOPRD artifact
        stored_document_id: Optional[str] = None
        try:
            project_id_from_context: Optional[str] = None
            mcp_root_path_str_from_context: Optional[str] = None

            if full_context:
                project_id_from_context = full_context.project_id
                
                # Get mcp_root_workspace_path from initial_inputs
                if full_context.initial_inputs:
                    mcp_root_path_str_from_context = full_context.initial_inputs.get('mcp_root_workspace_path')
                
                # Fallback if not in initial_inputs (should ideally not be needed if CLI provides it)
                if not mcp_root_path_str_from_context:
                    if hasattr(full_context, 'mcp_root_workspace_path') and full_context.mcp_root_workspace_path:
                        logger.warning("mcp_root_workspace_path not found in full_context.initial_inputs, using direct full_context.mcp_root_workspace_path as fallback.")
                        mcp_root_path_str_from_context = full_context.mcp_root_workspace_path
                    # The fallback to global_project_settings.project_dir was incorrect for MCP root, removing it as primary.
                    # If still not found, the error below will be raised.
            else:
                logger.error("'full_context' is None. Cannot retrieve project_id or mcp_root_workspace_path.")
                # This will lead to ValueError below if these are essential.

            if not project_id_from_context:
                logger.error("project_id could not be determined from full_context.")
                raise ValueError("project_id is required from full_context.")
            
            if not mcp_root_path_str_from_context:
                logger.error("mcp_root_workspace_path could not be determined from full_context.initial_inputs or a direct attribute. This is required to instantiate ProjectChromaManagerAgent.")
                raise ValueError("mcp_root_workspace_path is required to store LOPRD artifact.")

            mcp_root_workspace_path_for_pcma = Path(mcp_root_path_str_from_context)
            
            logger.info(f"Attempting to get ProjectChromaManager for project_id: {project_id_from_context} using MCP root: {mcp_root_workspace_path_for_pcma}")
            project_chroma_manager = await self._get_project_chroma_manager(project_id_from_context, mcp_root_workspace_path_for_pcma)

            # Prepare metadata for the artifact
            loprd_metadata = {
                "artifact_type": ARTIFACT_TYPE_LOPRD_JSON,
                "created_by_agent": self.AGENT_ID,
                "user_goal": parsed_inputs.user_goal,
                "project_context_summary": parsed_inputs.project_context_summary or "",
                "document_title": f"LOPRD for {parsed_inputs.user_goal[:50]}..."
                # Add other relevant metadata
            }
            
            # Ensure collection exists (idempotent)
            await project_chroma_manager.ensure_collection_exists(LOPRD_ARTIFACTS_COLLECTION)
            
            store_input = StoreArtifactInput(
                base_collection_name=LOPRD_ARTIFACTS_COLLECTION,
                artifact_content=loprd_content_json, # Store the parsed JSON object
                metadata=loprd_metadata,
                document_id=str(uuid.uuid4()), # Generate a new ID for the LOPRD
                source_agent_id=self.AGENT_ID,
                # cycle_id and source_task_id could come from full_context if available
                cycle_id=full_context.current_cycle_id if full_context else None,
                # TODO: Consider if source_task_id is available/relevant from full_context
            )
            logger.info(f"Storing LOPRD artifact in collection '{LOPRD_ARTIFACTS_COLLECTION}' with generated ID: {store_input.document_id}")
            store_output = await project_chroma_manager.store_artifact(args=store_input)

            if store_output.status == "SUCCESS" and store_output.document_id:
                stored_document_id = store_output.document_id
                logger.info(f"LOPRD artifact stored successfully with ID: {stored_document_id}")
            else:
                logger.error(f"Failed to store LOPRD artifact. PCMA Status: {store_output.status}, Error: {store_output.error_message}")
        
        except Exception as e:
            logger.error(f"Error storing LOPRD artifact: {e}", exc_info=True)
            # Continue without a stored document ID, but log the failure.

        # Use the generated LOPRD JSON as the summary, or part of it.
        # For now, a simple textual summary from the placeholder or LLM response if structured differently.
        # If loprd_content_json is rich, extract a summary from it.
        final_summary = f"LOPRD generated for goal: '{parsed_inputs.user_goal}'. Document ID: {stored_document_id or 'N/A'}."
        if isinstance(loprd_content_json, dict) and "summary" in loprd_content_json:
            final_summary = loprd_content_json["summary"] # Assuming LLM provides a summary key
        elif isinstance(loprd_content_json, dict):
            # Fallback summary if no specific 'summary' key
            final_summary = f"LOPRD (JSON) generated for goal: '{parsed_inputs.user_goal}'. Fields: {list(loprd_content_json.keys())}. Document ID: {stored_document_id or 'N/A'}."

        logger.info(f"Requirements summary: {final_summary[:100]}...")
        
        return SystemRequirementsGatheringOutput(
            refined_requirements_document_id=stored_document_id, # This will be None if storage failed
            requirements_summary=final_summary
        )

    @classmethod
    def get_agent_card_static(cls) -> AgentCard:
        return AgentCard(
            agent_id=cls.AGENT_ID,
            name=cls.AGENT_NAME,
            version=cls.VERSION,
            description=cls.DESCRIPTION,
            category=cls.CATEGORY.value,
            visibility=cls.VISIBILITY.value,
            input_schema=SystemRequirementsGatheringInput.model_json_schema(),
            output_schema=SystemRequirementsGatheringOutput.model_json_schema(),
            # Add capability_profile, tags, etc. as needed
        )

# Example of how it might be used (for testing this file directly)
# async def main():
#     # Mock dependencies
#     class MockLLMProvider(LLMProvider):
#         async def generate_text(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
#             return f"LLM mock response for: {user_prompt}"
#         async def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
#             return {"summary": f"LLM mock JSON for: {user_prompt}"}

#     class MockPromptManager(PromptManager):
#         def __init__(self): super().__init__(Path(".")) # Dummy path
#         def get_prompt_template(self, template_name: str):
#             # return a mock template
#             class MockTemplate:
#                 def render(self, **kwargs) -> str: return f"Rendered prompt with {kwargs}"
#             return MockTemplate()

#     llm_provider = MockLLMProvider()
#     prompt_manager = MockPromptManager()
#     agent = SystemRequirementsGatheringAgent_v1(llm_provider=llm_provider, prompt_manager=prompt_manager)
    
#     test_input = SystemRequirementsGatheringInput(user_goal="Build a todo app.")
#     output = await agent.invoke_async(test_input)
#     print(f"Agent Output: {output.model_dump_json(indent=2)}")
#     print(f"Agent Card: {agent.get_agent_card_static().model_dump_json(indent=2)}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main()) 