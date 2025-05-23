from __future__ import annotations

import logging
from typing import Any, Dict, Optional, ClassVar, TYPE_CHECKING
import json
from pathlib import Path
import uuid
import traceback

from pydantic import BaseModel, Field, ValidationError

from chungoid.runtime.agents.agent_base import BaseAgent, InputSchema, OutputSchema
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptDefinition
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
# REMOVED: Conditional import of AgentProvider as it's no longer used directly by this agent
# if TYPE_CHECKING:
#     from chungoid.utils.agent_resolver import AgentProvider 
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1,
    LOPRD_ARTIFACTS_COLLECTION,
    ARTIFACT_TYPE_LOPRD_JSON,
    StoreArtifactInput # Ensure StoreArtifactInput is imported
)
# from chungoid.schemas.chroma_agent_io_schemas import StoreArtifactInput # Redundant if imported above

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
    project_chroma_manager: ProjectChromaManagerAgent_v1 # ADDED
    # REMOVED: agent_provider field
    # agent_provider: Optional['AgentProvider'] = None 
    loprd_generation_prompt_template_obj: Optional[PromptDefinition] = None
    loprd_generation_inline_fallback_prompt: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, 
                 llm_provider: LLMProvider, 
                 prompt_manager: PromptManager, 
                 project_chroma_manager: ProjectChromaManagerAgent_v1, # ADDED
                 system_context: Optional[Dict[str, Any]] = None):
        # Pass llm_provider, prompt_manager, and project_chroma_manager to super().__init__
        init_kwargs = {
            "llm_provider": llm_provider,
            "prompt_manager": prompt_manager,
            "project_chroma_manager": project_chroma_manager
        }
        if system_context is not None:
            init_kwargs["system_context"] = system_context
        
        super().__init__(**init_kwargs)
        # Pydantic now handles assignment of llm_provider, prompt_manager, project_chroma_manager
        if not self.llm_provider: # Should be caught by Pydantic if not optional
            raise ValueError("LLMProvider is required for SystemRequirementsGatheringAgent_v1")
        if not self.prompt_manager: # Should be caught by Pydantic if not optional
            raise ValueError("PromptManager is required for SystemRequirementsGatheringAgent_v1")
        if not self.project_chroma_manager: # ADDED check, should be caught by Pydantic
            raise ValueError("ProjectChromaManagerAgent_v1 is required for SystemRequirementsGatheringAgent_v1")

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
            self.loprd_generation_prompt_template_obj = None 
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

    # REMOVED: _get_project_chroma_manager method
    # async def _get_project_chroma_manager(self, project_id: str, mcp_root_path: Path) -> ProjectChromaManagerAgent_v1:
    #     ... (implementation removed) ...

    async def invoke_async(
        self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None
    ) -> SystemRequirementsGatheringOutput:
        """
        Asynchronously processes the user goal to generate a LOPRD.
        """
        # Log entry with goal
        self.logger.info(
            f"System Requirements Gathering Agent ({self.AGENT_ID}) invoked with goal: {inputs.get('user_goal', 'Not specified')}"
        )

        # ----->>> ADD DIAGNOSTIC LOGGING HERE <<<-----
        self.logger.info(f"AGENT DIAGNOSTIC: About to validate inputs. Type: {type(inputs)}, Value: {inputs}, ID: {id(inputs)}")

        try:
            parsed_inputs = SystemRequirementsGatheringInput(**inputs)
        except ValidationError as e:
            self.logger.error(
                f"Input validation error for {self.AGENT_ID} ({self.AGENT_NAME}): {e}\\nTraceback: {traceback.format_exc()}"
            )
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
            logger.info(f"Generating LOPRD content for goal: {parsed_inputs.user_goal}")

            llm_call_args = {
                "prompt": rendered_prompt,
                "system_prompt": "You are an AI assistant that generates detailed project requirements documents in JSON format.",
                "response_format": {"type": "json_object"}
            }

            if self.loprd_generation_prompt_template_obj and self.loprd_generation_prompt_template_obj.model_settings:
                model_settings = self.loprd_generation_prompt_template_obj.model_settings
                llm_call_args["model_id"] = model_settings.model_name
                llm_call_args["temperature"] = model_settings.temperature
                if model_settings.max_tokens:
                    llm_call_args["max_tokens"] = model_settings.max_tokens

            loprd_content_str = await self.llm_provider.generate(**llm_call_args)
            
            loprd_content_json = json.loads(loprd_content_str) 
            logger.info(f"LOPRD JSON content generated successfully.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LOPRD JSON from LLM response: {e}. Response was: {loprd_content_str[:500]}...", exc_info=True)
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
            # project_id and mcp_root_workspace_path are now expected to be correctly set up 
            # and validated in the __init__ of self.project_chroma_manager
            # We just use self.project_chroma_manager directly.

            # No longer need to fetch project_id_from_context or mcp_root_path_str_from_context here
            # as self.project_chroma_manager is already initialized with this info.
            # Ensure self.project_chroma_manager is used for storing.
            
            # REMOVED: Logic to get project_id and mcp_root_path from full_context
            # REMOVED: Call to self._get_project_chroma_manager

            if not self.project_chroma_manager: # Should not happen if __init__ is correct
                 logger.error("Critical: self.project_chroma_manager not initialized in SystemRequirementsGatheringAgent_v1.")
                 raise RuntimeError("ProjectChromaManager not available for storing artifact.")

            # Prepare metadata for the artifact
            loprd_metadata = {
                "artifact_type": ARTIFACT_TYPE_LOPRD_JSON,
                "created_by_agent": self.AGENT_ID,
                "user_goal": parsed_inputs.user_goal,
                "project_context_summary": parsed_inputs.project_context_summary or "",
                "document_title": f"LOPRD for {parsed_inputs.user_goal[:50]}..."
            }
            
            await self.project_chroma_manager.ensure_collection_exists(LOPRD_ARTIFACTS_COLLECTION)
            
            if isinstance(loprd_content_json, dict):
                store_input = StoreArtifactInput(
                    project_id=self.project_chroma_manager.project_id,
                    base_collection_name=LOPRD_ARTIFACTS_COLLECTION,
                    artifact_content=loprd_content_json,
                    metadata=loprd_metadata,
                    cycle_id=full_context.data.get("current_cycle_id") if full_context and full_context.data else None,
                    source_agent_id=self.AGENT_ID
                )
                store_result = await self.project_chroma_manager.store_artifact(store_input)

                doc_id: Optional[str] = None
                error_message_from_store: Optional[str] = None

                if store_result and store_result.status == "SUCCESS":
                    doc_id = store_result.document_id
                    logger.info(f"LOPRD artifact stored successfully. Document ID: {doc_id}")
                else:
                    error_message_from_store = store_result.message or store_result.error_message if store_result else "Unknown error during artifact storage."
                    logger.error(f"Failed to store LOPRD artifact. Message: {error_message_from_store}")
                
                if doc_id:
                    stored_document_id = doc_id
                    logger.info(f"LOPRD artifact stored successfully. Document ID: {stored_document_id}")
                else:
                    logger.error(f"Failed to store LOPRD artifact. Document ID: {stored_document_id or 'N/A'}. Message: {error_message_from_store}")
        
        except Exception as e:
            logger.error(f"Error storing LOPRD artifact: {e}", exc_info=True)
            # Continue to return output even if storage fails, but document_id will be None

        # 4. Prepare and return output
        requirements_summary_str = f"LOPRD (JSON) generated for goal: '{parsed_inputs.user_goal}'. Fields: {list(loprd_content_json.keys())}. Document ID: {stored_document_id or 'N/A'}."
        if parsed_inputs.project_context_summary:
            requirements_summary_str += f" Context: {parsed_inputs.project_context_summary[:100]}..."
        
        logger.info(f"Requirements summary: {requirements_summary_str}")

        return SystemRequirementsGatheringOutput(
            refined_requirements_document_id=stored_document_id,
            requirements_summary=requirements_summary_str
        )

    @classmethod
    def get_agent_card_static(cls) -> AgentCard:
        """Returns the static AgentCard for this agent."""
        return AgentCard(
            agent_id=cls.AGENT_ID,
            name=cls.AGENT_NAME,
            version=cls.VERSION,
            description=cls.DESCRIPTION,
            category=cls.CATEGORY,
            visibility=cls.VISIBILITY,
            input_schema=SystemRequirementsGatheringInput.model_json_schema(),
            output_schema=SystemRequirementsGatheringOutput.model_json_schema(),
            dependencies=["LLMProvider", "PromptManager", "ProjectChromaManagerAgent_v1"],
            init_args=["llm_provider", "prompt_manager", "project_chroma_manager"]
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