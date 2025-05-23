from __future__ import annotations

import logging
import json
import uuid
from typing import Any, Dict, Optional, List, Type, ClassVar
from datetime import datetime, timezone

from chungoid.runtime.agents.agent_base import BaseAgent, InputSchema, OutputSchema
from chungoid.schemas.agent_code_generator import SmartCodeGeneratorAgentInput, SmartCodeGeneratorAgentOutput
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.chroma_agent_io_schemas import StoreArtifactInput, StoreArtifactOutput, RetrieveArtifactOutput
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1,
    LOPRD_ARTIFACTS_COLLECTION,
    BLUEPRINT_ARTIFACTS_COLLECTION,
    GENERATED_CODE_ARTIFACTS_COLLECTION,
    PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION,
    LIVE_CODEBASE_COLLECTION
)

logger = logging.getLogger(__name__)

PROMPT_ID = "smart_code_generator_agent_v1_prompt"
PROMPT_VERSION = "0.2.0"
PROMPT_SUB_DIR = "autonomous_engine"

class CoreCodeGeneratorAgent_v1(BaseAgent[SmartCodeGeneratorAgentInput, SmartCodeGeneratorAgentOutput]):
    AGENT_ID: ClassVar[str] = "SmartCodeGeneratorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Smart Code Generator Agent"
    VERSION: ClassVar[str] = "0.2.0"
    DESCRIPTION: ClassVar[str] = "Generates or modifies code based on detailed specifications and contextual project artifacts, interacting with ChromaDB."
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.CODE_GENERATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[SmartCodeGeneratorAgentInput]] = SmartCodeGeneratorAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[SmartCodeGeneratorAgentOutput]] = SmartCodeGeneratorAgentOutput

    _llm_provider: LLMProvider
    _prompt_manager: PromptManager
    _pcma_agent: ProjectChromaManagerAgent_v1
    _logger: logging.Logger

    def __init__(self, 
                 llm_provider: LLMProvider, 
                 prompt_manager: PromptManager, 
                 pcma_agent: ProjectChromaManagerAgent_v1,
                 system_context: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 agent_id: Optional[str] = None,
                 **kwargs):
        if not llm_provider:
            raise ValueError("LLMProvider is required for SmartCodeGeneratorAgent_v1")
        if not prompt_manager:
            raise ValueError("PromptManager is required for SmartCodeGeneratorAgent_v1")
        if not pcma_agent:
            raise ValueError("ProjectChromaManagerAgent_v1 is required for SmartCodeGeneratorAgent_v1")

        super_kwargs = {
            "llm_provider": llm_provider,
            "prompt_manager": prompt_manager,
            "project_chroma_manager": pcma_agent,
            "agent_id": agent_id or self.AGENT_ID
        }
        if system_context:
            super_kwargs["system_context"] = system_context
        if config:
            super_kwargs["config"] = config
        
        super().__init__(**super_kwargs)

        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        self._pcma_agent = pcma_agent
        
        if system_context and "logger" in system_context and isinstance(system_context["logger"], logging.Logger):
            self._logger = system_context["logger"]
        else:
            self._logger = logging.getLogger(self.AGENT_ID)
        self._logger.info(f"{self.AGENT_ID} (v{self.VERSION}) initialized.")

    async def invoke_async(
        self,
        task_input: SmartCodeGeneratorAgentInput,
        full_context: Optional[Dict[str, Any]] = None,
    ) -> SmartCodeGeneratorAgentOutput:
        parsed_inputs = task_input
        logger_instance = self._logger

        logger_instance.info(f"{self.AGENT_ID} invoked for task {parsed_inputs.task_id} targeting file: {parsed_inputs.target_file_path} in project {parsed_inputs.project_id}")
        logger_instance.debug(f"{self.AGENT_ID} inputs: {parsed_inputs.model_dump_json(indent=2)}")

        code_specification_content: Optional[str] = None
        existing_code_content: Optional[str] = None
        blueprint_context_content: Optional[str] = None
        loprd_requirements_list_content: List[str] = []

        try:
            # Fetch code specification
            if parsed_inputs.code_specification_doc_id:
                retrieved_spec: RetrieveArtifactOutput = await self._pcma_agent.retrieve_artifact(
                    base_collection_name=PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION,
                    document_id=parsed_inputs.code_specification_doc_id
                )
                if retrieved_spec and retrieved_spec.status == "SUCCESS" and retrieved_spec.content:
                    code_specification_content = str(retrieved_spec.content)
                    logger_instance.debug(f"Retrieved code specification: {parsed_inputs.code_specification_doc_id}")
                else:
                    logger_instance.warning(f"Code specification {parsed_inputs.code_specification_doc_id} not found or content empty. Status: {retrieved_spec.status if retrieved_spec else 'N/A'}")
            
            # Fetch existing code if modifying
            if parsed_inputs.existing_code_doc_id:
                retrieved_existing_code: RetrieveArtifactOutput = await self._pcma_agent.retrieve_artifact(
                    base_collection_name=GENERATED_CODE_ARTIFACTS_COLLECTION,
                    document_id=parsed_inputs.existing_code_doc_id
                )
                if retrieved_existing_code and retrieved_existing_code.status == "SUCCESS" and retrieved_existing_code.content:
                    existing_code_content = str(retrieved_existing_code.content)
                    logger_instance.debug(f"Retrieved existing code: {parsed_inputs.existing_code_doc_id}")
                else:
                    logger_instance.warning(f"Existing code {parsed_inputs.existing_code_doc_id} not found or content empty. Status: {retrieved_existing_code.status if retrieved_existing_code else 'N/A'}")

            # Fetch blueprint context
            if parsed_inputs.blueprint_context_doc_id:
                retrieved_blueprint: RetrieveArtifactOutput = await self._pcma_agent.retrieve_artifact(
                    base_collection_name=BLUEPRINT_ARTIFACTS_COLLECTION,
                    document_id=parsed_inputs.blueprint_context_doc_id
                )
                if retrieved_blueprint and retrieved_blueprint.status == "SUCCESS" and retrieved_blueprint.content:
                    blueprint_context_content = str(retrieved_blueprint.content)
                    logger_instance.debug(f"Retrieved blueprint context: {parsed_inputs.blueprint_context_doc_id}")
                else:
                    logger_instance.warning(f"Blueprint context {parsed_inputs.blueprint_context_doc_id} not found or content empty. Status: {retrieved_blueprint.status if retrieved_blueprint else 'N/A'}")

            # Fetch LOPRD requirements
            if parsed_inputs.loprd_requirements_doc_ids:
                for doc_id in parsed_inputs.loprd_requirements_doc_ids:
                    retrieved_loprd: RetrieveArtifactOutput = await self._pcma_agent.retrieve_artifact(
                        base_collection_name=LOPRD_ARTIFACTS_COLLECTION,
                        document_id=doc_id
                    )
                    if retrieved_loprd and retrieved_loprd.status == "SUCCESS" and retrieved_loprd.content:
                        loprd_requirements_list_content.append(str(retrieved_loprd.content))
                        logger_instance.debug(f"Retrieved LOPRD requirement: {doc_id}")
                    else:
                        logger_instance.warning(f"LOPRD requirement {doc_id} not found or content empty. Status: {retrieved_loprd.status if retrieved_loprd else 'N/A'}")
            
            if not code_specification_content:
                logger_instance.info(
                    f"No code_specification_doc_id provided (value: {parsed_inputs.code_specification_doc_id}) "
                    f"or the document content was empty. Using task_description as the primary specification."
                )
                # Use task_description from parsed_inputs if code_specification_content is missing.
                # The prompt should be robust enough to primarily use task_description 
                # if the detailed specification is minimal or absent.
                code_specification_content = parsed_inputs.task_description # Fallback to task_description
                if not code_specification_content: # If task_description itself is somehow empty (should be caught by input validation)
                    raise ValueError("Neither code_specification_content nor task_description is available.")

        except Exception as e:
            logger_instance.error(f"Failed to retrieve context documents via PCMA: {e}", exc_info=True)
            return SmartCodeGeneratorAgentOutput(
                task_id=parsed_inputs.task_id,
                target_file_path=parsed_inputs.target_file_path,
                status="FAILURE_CONTEXT_RETRIEVAL",
                error_message=f"Error retrieving context documents: {e}"
            )

        prompt_render_data = {
            "task_id": parsed_inputs.task_id,
            "project_id": parsed_inputs.project_id,
            "code_specification_content": code_specification_content,
            "target_file_path": parsed_inputs.target_file_path,
            "programming_language": parsed_inputs.programming_language,
            "existing_code_content": existing_code_content or "No existing code provided. Assume new file creation.",
            "blueprint_context_content": blueprint_context_content or "No blueprint context provided.",
            "loprd_requirements_content_list": loprd_requirements_list_content if loprd_requirements_list_content else [],
            "additional_instructions": parsed_inputs.additional_instructions or "Follow standard coding best practices."
        }

        llm_full_response_str: Optional[str] = None
        generated_code_str: Optional[str] = None
        confidence_score_obj: Optional[ConfidenceScore] = None
        usage_metadata_val: Optional[Dict[str, Any]] = None

        try:
            logger_instance.debug(f"Attempting to generate code via LLM for: {parsed_inputs.target_file_path}.")
            
            # Step 1: Get the prompt definition to access model settings
            prompt_def = self._prompt_manager.get_prompt_definition(
                prompt_name=PROMPT_ID, 
                prompt_version=PROMPT_VERSION, 
                sub_path=PROMPT_SUB_DIR
            )

            # Step 2: Render the system and user prompts using PromptManager
            rendered_system_prompt, rendered_user_prompt = await self._prompt_manager.get_rendered_system_and_user_prompts(
                prompt_name=PROMPT_ID, 
                prompt_version=PROMPT_VERSION, 
                prompt_sub_path=PROMPT_SUB_DIR,
                prompt_render_data=prompt_render_data
            )

            if not rendered_user_prompt:
                raise PromptRenderError("PromptManager did not return a user_prompt after rendering.")

            # Step 3: Call the LLMProvider with the rendered prompts and model settings from definition
            llm_full_response_str = await self._llm_provider.generate(
                prompt=rendered_user_prompt,
                system_prompt=rendered_system_prompt, 
                model_id=prompt_def.model_settings.model_name, 
                temperature=prompt_def.model_settings.temperature,
                max_tokens=prompt_def.model_settings.max_tokens,
                response_format={"type": "json_object"}
            )
            
            if not llm_full_response_str:
                raise ValueError("LLM returned an empty response.")

            # We expect the LLM to return a JSON string that matches SmartCodeGeneratorAgentOutput's structure, 
            # or at least contains the key fields like 'generated_code_string'
            llm_response_data = json.loads(llm_full_response_str)
            
            # --- Extract core fields from LLM response based on the prompt's output_schema ---
            generated_code_string = llm_response_data.get("generated_code") # Corrected field name
            confidence_score_data = llm_response_data.get("confidence_score")
            key_decision_rationale = llm_response_data.get("key_decision_rationale")
            contextual_adherence_explanation = llm_response_data.get("contextual_adherence_explanation")
            # --- End extraction ---

            if not generated_code_string or not isinstance(generated_code_string, str):
                logger_instance.error(f"LLM did not return a valid code string in 'generated_code' field.") # Corrected field name in log
                raise ValueError("LLM output missing 'generated_code'.") # Corrected field name in error

            logger_instance.info(f"LLM successfully generated code content for {parsed_inputs.target_file_path}.")

        except PromptRenderError as e_prompt:
            logger_instance.error(f"Prompt rendering failed for {PROMPT_ID}: {e_prompt}", exc_info=True)
            return SmartCodeGeneratorAgentOutput(task_id=parsed_inputs.task_id, target_file_path=parsed_inputs.target_file_path, status="FAILURE_LLM_GENERATION", error_message=f"Prompt rendering error: {e_prompt}", llm_full_response=llm_full_response_str)
        except json.JSONDecodeError as e_json:
            logger_instance.error(f"Failed to decode LLM JSON response: {e_json}. Response: {llm_full_response_str[:500] if llm_full_response_str else 'N/A'}", exc_info=True)
            return SmartCodeGeneratorAgentOutput(task_id=parsed_inputs.task_id, target_file_path=parsed_inputs.target_file_path, status="FAILURE_LLM_GENERATION", error_message=f"LLM response not valid JSON: {e_json}", llm_full_response=llm_full_response_str)
        except ValueError as e_val:
            logger_instance.error(f"Error processing LLM output: {e_val}. Response: {llm_full_response_str[:500] if llm_full_response_str else 'N/A'}", exc_info=True)
            return SmartCodeGeneratorAgentOutput(task_id=parsed_inputs.task_id, target_file_path=parsed_inputs.target_file_path, status="FAILURE_LLM_GENERATION", error_message=str(e_val), llm_full_response=llm_full_response_str)
        except Exception as e_gen:
            logger_instance.error(f"General error during LLM call or processing for {parsed_inputs.target_file_path}: {e_gen}", exc_info=True)
            return SmartCodeGeneratorAgentOutput(
                task_id=parsed_inputs.task_id,
                target_file_path=parsed_inputs.target_file_path,
                status="FAILURE_LLM_GENERATION",
                error_message=f"LLM interaction failed: {str(e_gen)}",
                llm_full_response=llm_full_response_str
            )
        
        generated_code_artifact_doc_id: Optional[str] = None
        stored_in_collection_name: Optional[str] = None

        if generated_code_string and parsed_inputs.project_id:
            try:
                # Prepare metadata, converting Nones to empty strings for Chroma compatibility
                # and JSON-encoding lists.
                
                confidence_val_str = ""
                if confidence_score_data and confidence_score_data.get("value") is not None:
                    confidence_val_str = str(confidence_score_data.get("value"))

                chroma_metadata = {
                    "artifact_type": "GeneratedCodeModule",
                    "target_file_path": parsed_inputs.target_file_path or "",
                    "programming_language": parsed_inputs.programming_language or "",
                    "source_specification_doc_id": parsed_inputs.code_specification_doc_id or "",
                    "source_existing_code_doc_id": parsed_inputs.existing_code_doc_id or "",
                    "source_blueprint_doc_id": parsed_inputs.blueprint_context_doc_id or "",
                    "source_loprd_doc_ids": json.dumps(parsed_inputs.loprd_requirements_doc_ids if parsed_inputs.loprd_requirements_doc_ids else []),
                    "task_id": parsed_inputs.task_id or "",
                    "llm_confidence_value": confidence_val_str,
                    "llm_confidence_explanation": (confidence_score_data.get("explanation") or "") if confidence_score_data else "",
                    "key_decision_rationale": key_decision_rationale or "",
                    "contextual_adherence_explanation": contextual_adherence_explanation or "",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                store_input = StoreArtifactInput(
                    base_collection_name=GENERATED_CODE_ARTIFACTS_COLLECTION,
                    artifact_content=generated_code_string,
                    metadata=chroma_metadata,
                    project_id=parsed_inputs.project_id,
                )
                logger_instance.info(f"Storing generated code for {parsed_inputs.target_file_path} in PCMA collection {GENERATED_CODE_ARTIFACTS_COLLECTION}.")
                
                store_output: StoreArtifactOutput = await self._pcma_agent.store_artifact(args=store_input)
                
                if store_output and store_output.status == "SUCCESS":
                    generated_code_artifact_doc_id = store_output.document_id
                    stored_in_collection_name = GENERATED_CODE_ARTIFACTS_COLLECTION
                    logger_instance.info(f"Generated code stored successfully in PCMA. Doc ID: {generated_code_artifact_doc_id}")
                else:
                    logger_instance.error(f"Failed to store generated code in PCMA. Status: {store_output.status if store_output else 'N/A'}, Message: {store_output.message if store_output else 'N/A'}")
                    return SmartCodeGeneratorAgentOutput(
                        task_id=parsed_inputs.task_id,
                        target_file_path=parsed_inputs.target_file_path,
                        status="FAILURE_OUTPUT_STORAGE",
                        generated_code_string=generated_code_string,
                        confidence_score=ConfidenceScore(value=confidence_score_data.get("value") if confidence_score_data else None, explanation=confidence_score_data.get("explanation") if confidence_score_data else None),
                        llm_full_response=llm_full_response_str,
                        usage_metadata=usage_metadata_val,
                        error_message=f"PCMA storage failed: {store_output.message if store_output else 'Unknown PCMA error'}"
                    )

            except Exception as e_store:
                logger_instance.error(f"Exception during PCMA storage of generated code: {e_store}", exc_info=True)
                return SmartCodeGeneratorAgentOutput(
                    task_id=parsed_inputs.task_id,
                    target_file_path=parsed_inputs.target_file_path,
                    status="FAILURE_OUTPUT_STORAGE",
                    generated_code_string=generated_code_string,
                    confidence_score=ConfidenceScore(value=confidence_score_data.get("value") if confidence_score_data else None, explanation=confidence_score_data.get("explanation") if confidence_score_data else None),
                    llm_full_response=llm_full_response_str,
                    usage_metadata=usage_metadata_val,
                    error_message=f"PCMA storage exception: {e_store}"
                )
        elif not parsed_inputs.project_id:
            logger_instance.warning("project_id not provided in SmartCodeGeneratorAgentInput. Generated code will not be stored in PCMA.")

        return SmartCodeGeneratorAgentOutput(
            task_id=parsed_inputs.task_id,
            target_file_path=parsed_inputs.target_file_path,
            status="SUCCESS",
            generated_code_artifact_doc_id=generated_code_artifact_doc_id,
            stored_in_collection=stored_in_collection_name,
            generated_code_string=generated_code_string,
            confidence_score=ConfidenceScore(value=confidence_score_data.get("value") if confidence_score_data else None, explanation=confidence_score_data.get("explanation") if confidence_score_data else None),
            llm_full_response=llm_full_response_str,
            usage_metadata=usage_metadata_val
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = SmartCodeGeneratorAgentInput.model_json_schema()
        output_schema = SmartCodeGeneratorAgentOutput.model_json_schema()
        
        llm_direct_output_schema = {
            "type": "object",
            "properties": {
                "generated_code_string": {
                    "type": "string",
                    "description": "The complete, syntactically correct code generated by the LLM."
                },
                "confidence_score_obj": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "level": {"type": ["string", "null"], "enum": ["Low", "Medium", "High", None]},
                        "explanation": {"type": "string"},
                        "method": {"type": ["string", "null"]}
                    },
                    "required": ["value", "explanation"],
                    "description": "Structured confidence score from the LLM about the generated code."
                },
                "usage_metadata": {
                    "type": ["object", "null"],
                    "properties": {
                        "prompt_tokens": {"type": "integer"},
                        "completion_tokens": {"type": "integer"},
                        "total_tokens": {"type": "integer"}
                    },
                    "description": "Token usage data from the LLM call."
                }
            },
            "required": ["generated_code_string", "confidence_score_obj"]
        }

        return AgentCard(
            agent_id=CoreCodeGeneratorAgent_v1.AGENT_ID,
            name=CoreCodeGeneratorAgent_v1.AGENT_NAME,
            version=CoreCodeGeneratorAgent_v1.VERSION,
            description=CoreCodeGeneratorAgent_v1.DESCRIPTION,
            categories=[cat.value for cat in [CoreCodeGeneratorAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=CoreCodeGeneratorAgent_v1.VISIBILITY.value,
            input_schema=input_schema,
            output_schema=output_schema,
            llm_direct_output_schema=llm_direct_output_schema,
            capability_profile={
                "generates_code": True,
                "modifies_code": True,
                "uses_context_from_pcma": True,
                "stores_output_to_pcma": True,
                "languages": ["python"],
            },
            metadata={
                "prompt_name": PROMPT_ID,
                "prompt_sub_dir": PROMPT_SUB_DIR,
                "callable_fn_path": f"{CoreCodeGeneratorAgent_v1.__module__}.{CoreCodeGeneratorAgent_v1.__name__}"
            }
        )

async def main_test_smart_code_gen():
    logging.basicConfig(level=logging.DEBUG)
    print("Test stub for SmartCodeGeneratorAgent_v1 needs full environment setup.")

if __name__ == "__main__":
    import asyncio
    print("To test SmartCodeGeneratorAgent_v1, please run through an integration test or a dedicated test script with mocked/real dependencies.") 