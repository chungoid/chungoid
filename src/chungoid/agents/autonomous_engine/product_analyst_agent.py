from __future__ import annotations

import logging
import datetime # Not strictly used, but good for potential timestamping
import uuid
from typing import Any, Dict, Optional, TypeVar, Generic, ClassVar
from pathlib import Path
import json

from pydantic import BaseModel, Field, ValidationError

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
# Assuming ProjectChromaManagerAgent will provide methods to get/store artifacts
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1 # MODIFIED: Actual import
from chungoid.schemas.common import ConfidenceScore # Assuming a common schema exists
from chungoid.schemas.orchestration import SharedContext # ADDED SharedContext IMPORT
from chungoid.schemas.autonomous_engine.loprd_schema import LOPRD # MODIFIED: Import the actual LOPRD Pydantic model
from chungoid.utils.json_schema_loader import load_json_schema_from_file # For LOPRD schema
from chungoid.utils.agent_registry import AgentCard # For AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard categories/visibility

logger = logging.getLogger(__name__)

# --- Input and Output Schemas for the Agent --- #

class ProductAnalystAgentInput(BaseModel):
    refined_user_goal_doc_id: str = Field(..., description="Document ID of the refined_user_goal.md in Chroma.")
    assumptions_and_ambiguities_doc_id: Optional[str] = Field(None, description="Document ID of the assumptions_and_ambiguities.md in Chroma, if available.")
    project_id: str = Field(..., description="The ID of the current project.")
    # Potentially add context from ARCA if this is a refinement loop
    arca_feedback_doc_id: Optional[str] = Field(None, description="Document ID of feedback from ARCA if this is a refinement run.")
    shared_context: Optional[SharedContext] = Field(None, description="The shared context from the orchestrator, providing broader project and workflow information.")

class ProductAnalystAgentOutput(BaseModel):
    loprd_doc_id: str = Field(..., description="Document ID of the generated LOPRD JSON artifact in Chroma.")
    confidence_score: ConfidenceScore = Field(..., description="Confidence score for the generated LOPRD.")
    raw_llm_response: Optional[str] = Field(None, description="The raw JSON string from the LLM before validation, for debugging.")
    validation_errors: Optional[str] = Field(None, description="Validation errors if the LLM output failed schema validation.")

class ProductAnalystAgent_v1(BaseAgent[ProductAnalystAgentInput, ProductAnalystAgentOutput]):
    AGENT_ID: ClassVar[str] = "ProductAnalystAgent_v1"
    AGENT_NAME: ClassVar[str] = "Product Analyst Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Transforms a refined user goal into a detailed LLM-Optimized Product Requirements Document (LOPRD) in JSON format."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "product_analyst_agent_v1.yaml" # In server_prompts/autonomous_engine/
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS # Or custom category
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC

    def __init__(self, 
                 llm_provider: LLMProvider,
                 prompt_manager: PromptManager,
                 project_chroma_manager: ProjectChromaManagerAgent_v1, # MODIFIED: Added PCMA injection
                 config: Optional[Dict[str, Any]] = None,
                 system_context: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, system_context=system_context)
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        self.project_chroma_manager = project_chroma_manager # MODIFIED: Store PCMA instance
        self._logger_instance = system_context.get("logger", logger) if system_context else logger
        self.loprd_json_schema_for_prompt = self._load_loprd_json_schema_for_prompt() # MODIFIED: Renamed for clarity

    def _load_loprd_json_schema_for_prompt(self) -> Optional[Dict[str, Any]]: # MODIFIED: Renamed for clarity
        try:
            # Determine path to loprd_schema.json relative to this file or a known schemas dir
            # This assumes a certain project structure. Adjust path as necessary.
            # chungoid-core/src/chungoid/agents/autonomous_engine/product_analyst_agent.py
            # chungoid-core/src/chungoid/schemas/autonomous_engine/loprd_schema.json
            schema_path = Path(__file__).parent.parent.parent / "schemas" / "autonomous_engine" / "loprd_schema.json"
            if not schema_path.exists():
                self._logger_instance.error(f"LOPRD schema file not found at {schema_path}")
                return None
            # Instead of returning the schema, we return the LOPRD Pydantic model's schema
            return LOPRD.model_json_schema() # MODIFIED: Get schema from Pydantic model
        except Exception as e:
            self._logger_instance.error(f"Failed to load LOPRD JSON schema for prompt: {e}")
            return None

    async def invoke_async(self, task_input: ProductAnalystAgentInput, full_context: Optional[Dict[str, Any]] = None) -> ProductAnalystAgentOutput:
        self._logger_instance.info(f"ProductAnalystAgent_v1 invoked for project {task_input.project_id}.")

        # Resolve LLMProvider and PromptManager from full_context if not already set (standard practice)
        llm_provider = self.llm_provider
        prompt_manager = self.prompt_manager
        pcma_agent = self.project_chroma_manager # Use the injected instance

        if full_context:
            if "llm_provider" in full_context and full_context["llm_provider"] != llm_provider : 
                llm_provider = full_context["llm_provider"]
                self._logger_instance.info("Using LLMProvider from full_context.")
            if "prompt_manager" in full_context and full_context["prompt_manager"] != prompt_manager:
                prompt_manager = full_context["prompt_manager"]
                self._logger_instance.info("Using PromptManager from full_context.")
            if "project_chroma_manager_agent_instance" in full_context and full_context["project_chroma_manager_agent_instance"] != pcma_agent:
                 pcma_agent = full_context["project_chroma_manager_agent_instance"]
                 self._logger_instance.info("Using ProjectChromaManagerAgent_v1 from full_context.")


        if not self.loprd_json_schema_for_prompt:
            return ProductAnalystAgentOutput(
                loprd_doc_id="error_loprd_schema_not_loaded",
                confidence_score=ConfidenceScore(value=0.0, level="Low", method="InternalError", reasoning="LOPRD JSON schema failed to load."),
                validation_errors="LOPRD JSON schema could not be loaded by the agent."
            )

        # 1. Retrieve content for refined_user_goal and assumptions using PCMA
        refined_goal_content: Optional[str] = None
        assumptions_content: Optional[str] = None
        arca_feedback_content: Optional[str] = None
        
        try:
            # Actual PCMA call for refined_user_goal_doc_id
            doc = await pcma_agent.get_document_by_id(project_id=task_input.project_id, doc_id=task_input.refined_user_goal_doc_id)
            if not doc or not doc.document_content:
                raise ValueError(f"Refined user goal document {task_input.refined_user_goal_doc_id} not found or content empty.")
            refined_goal_content = doc.document_content
            self._logger_instance.debug(f"Retrieved refined_user_goal_doc_id: {task_input.refined_user_goal_doc_id}")

            # Actual PCMA call for assumptions_and_ambiguities_doc_id (optional)
            if task_input.assumptions_and_ambiguities_doc_id:
                doc = await pcma_agent.get_document_by_id(project_id=task_input.project_id, doc_id=task_input.assumptions_and_ambiguities_doc_id)
                if doc and doc.document_content:
                    assumptions_content = doc.document_content
                    self._logger_instance.debug(f"Retrieved assumptions_and_ambiguities_doc_id: {task_input.assumptions_and_ambiguities_doc_id}")
                else:
                    self._logger_instance.warning(f"Optional assumptions_and_ambiguities_doc_id {task_input.assumptions_and_ambiguities_doc_id} not found or content empty.")
                    assumptions_content = "Not provided or content empty." # Default if optional and not found
            else:
                assumptions_content = "Not provided."
        
            # Actual PCMA call for arca_feedback_doc_id (optional)
            if task_input.arca_feedback_doc_id:
                doc = await pcma_agent.get_document_by_id(project_id=task_input.project_id, doc_id=task_input.arca_feedback_doc_id)
                if doc and doc.document_content:
                    arca_feedback_content = doc.document_content
                    self._logger_instance.debug(f"Retrieved arca_feedback_doc_id: {task_input.arca_feedback_doc_id}")
                else:
                    self._logger_instance.warning(f"Optional arca_feedback_doc_id {task_input.arca_feedback_doc_id} not found or content empty.")
                    arca_feedback_content = "No specific feedback from ARCA for this iteration or content empty." # Default if optional
            else:
                arca_feedback_content = "No specific feedback from ARCA for this iteration."

        except Exception as e_pcma_fetch:
            self._logger_instance.error(f"PCMA content retrieval failed: {e_pcma_fetch}", exc_info=True)
            # This is a critical failure if the main refined_user_goal cannot be fetched.
            return ProductAnalystAgentOutput(
                loprd_doc_id="error_pcma_input_retrieval_failed",
                confidence_score=ConfidenceScore(value=0.0, level="Low", method="InternalError", reasoning=f"PCMA input retrieval failed: {e_pcma_fetch}"),
                validation_errors=f"Failed to retrieve input documents: {e_pcma_fetch}",
                raw_llm_response=None
            )

        # 2. Prepare prompt data
        prompt_data = {
            "refined_user_goal_md": refined_goal_content,
            "assumptions_and_ambiguities_md": assumptions_content,
            "loprd_json_schema_str": json.dumps(self.loprd_json_schema_for_prompt, indent=2),
            "arca_feedback_md": arca_feedback_content
        }

        # 3. Interact with LLM using generate_text_async_with_prompt_manager
        raw_llm_response_json: Optional[str] = None
        try:
            self._logger_instance.info(f"Sending request to LLM via PromptManager for LOPRD generation (prompt: {self.PROMPT_TEMPLATE_NAME})...")
            
            raw_llm_response_json = await llm_provider.generate_text_async_with_prompt_manager(
                prompt_manager=prompt_manager,
                prompt_name=self.PROMPT_TEMPLATE_NAME, # Class variable
                prompt_data=prompt_data,
                sub_dir="autonomous_engine", # As specified in PROMPT_TEMPLATE_NAME comment
                # model_id="gpt-4-turbo-preview", # Or from config/LLMProvider default
                temperature=0.2, # Example, tune as needed
                # json_response=True # This flag might not be needed if prompt asks for JSON and we parse
            )
            self._logger_instance.info("Received response from LLM.")
            self._logger_instance.debug(f"LLM Raw Response: {raw_llm_response_json[:500] if raw_llm_response_json else 'None'}...")

        except PromptRenderError as e_prompt:
            self._logger_instance.error(f"Failed to render prompt {self.PROMPT_TEMPLATE_NAME}: {e_prompt}", exc_info=True)
            return ProductAnalystAgentOutput(
                loprd_doc_id="error_prompt_render_failed",
                confidence_score=ConfidenceScore(value=0.0, level="Low", method="InternalError", reasoning=f"Prompt rendering failed: {e_prompt}"),
                validation_errors=f"Prompt rendering failed: {e_prompt}"
            )
        except Exception as e_llm: # Catch other LLM call related errors
            self._logger_instance.error(f"LLM interaction failed: {e_llm}", exc_info=True)
            return ProductAnalystAgentOutput(
                loprd_doc_id="error_llm_failed",
                confidence_score=ConfidenceScore(value=0.0, level="Low", method="LLMError", reasoning=f"LLM call failed: {e_llm}"),
                raw_llm_response=str(e_llm) # Store error as raw response for debugging
            )

        if not raw_llm_response_json:
            self._logger_instance.error("LLM returned an empty response.")
            return ProductAnalystAgentOutput(
                loprd_doc_id="error_llm_empty_response",
                confidence_score=ConfidenceScore(value=0.0, level="Low", method="LLMError", reasoning="LLM returned an empty response."),
                raw_llm_response=""
            )

        # 4. Validate LLM JSON output against LOPRD Pydantic schema
        validated_loprd: Optional[LOPRD] = None
        parsed_confidence_score_dict: Optional[Dict[str, Any]] = None # To store the parsed confidence score
        validation_errors_str: Optional[str] = None
        try:
            # The prompt asks for a specific JSON structure.
            # {"loprd_artifact": LOPRD_JSON, "confidence_score": {"value": float, "explanation": "str", ...}}
            full_llm_response_dict = json.loads(raw_llm_response_json)
            
            if "loprd_artifact" not in full_llm_response_dict:
                raise ValueError("LLM response missing 'loprd_artifact' key.")
            if "confidence_score" not in full_llm_response_dict:
                raise ValueError("LLM response missing 'confidence_score' key.")

            loprd_data_dict = full_llm_response_dict["loprd_artifact"]
            parsed_confidence_score_dict = full_llm_response_dict["confidence_score"]

            validated_loprd = LOPRD(**loprd_data_dict) # Pydantic validation
            
            # Validate confidence_score structure (simple check for now, could be a Pydantic model too)
            if not isinstance(parsed_confidence_score_dict.get("value"), (float, int)) or \
               not isinstance(parsed_confidence_score_dict.get("explanation"), str):
                raise ValueError("Confidence score from LLM has incorrect structure or missing fields.")

            self._logger_instance.info("LLM output successfully parsed and validated against LOPRD Pydantic schema and expected structure.")
        except json.JSONDecodeError as e_json_decode:
            self._logger_instance.error(f"Failed to decode LLM response as JSON: {e_json_decode}. Response was: {raw_llm_response_json[:500]}...", exc_info=True)
            validation_errors_str = f"LLM output was not valid JSON: {e_json_decode}"
        except ValidationError as e_pydantic_validation:
            self._logger_instance.error(f"LLM JSON output failed LOPRD Pydantic schema validation: {e_pydantic_validation}", exc_info=True)
            validation_errors_str = str(e_pydantic_validation)
        except Exception as e_unexpected_parse: # Catch any other unexpected error during parsing/validation
            self._logger_instance.error(f"Unexpected error during LLM response parsing or validation: {e_unexpected_parse}", exc_info=True)
            validation_errors_str = f"Unexpected error processing LLM response: {e_unexpected_parse}"
        
        if not validated_loprd:
            return ProductAnalystAgentOutput(
                loprd_doc_id="error_llm_output_invalid",
                confidence_score=ConfidenceScore(value=0.1, level="Low", method="LLMOutputValidation", reasoning="LLM output was not valid LOPRD JSON or failed schema validation."),
                raw_llm_response=raw_llm_response_json,
                validation_errors=validation_errors_str
            )

        # 5. Store LOPRD JSON using PCMA
        stored_loprd_doc_id: Optional[str] = None # Initialize to None
        try:
            if not validated_loprd:
                # This case should ideally be caught earlier, but as a safeguard:
                raise ValueError("Validated LOPRD is None, cannot store.")

            loprd_json_content = validated_loprd.model_dump_json(indent=2)
            
            # Prepare confidence score for metadata
            # Ensure parsed_confidence_score_dict is not None before accessing it
            confidence_metadata = {}
            if parsed_confidence_score_dict:
                confidence_metadata = {
                    "confidence_value": parsed_confidence_score_dict.get("value"),
                    "confidence_method": parsed_confidence_score_dict.get("method", "Agent self-assessment"), # Default if not provided
                    "confidence_explanation": parsed_confidence_score_dict.get("explanation") 
                                              # This explanation should contain adherence & rationale as per prompt
                }
            else: # Fallback if confidence parsing failed but LOPRD was somehow fine (unlikely path)
                 self._logger_instance.warning("parsed_confidence_score_dict was None during metadata preparation. Using default confidence.")
                 confidence_metadata = {
                    "confidence_value": 0.5, # Default placeholder
                    "confidence_method": "FallbackDefault",
                    "confidence_explanation": "Confidence score could not be parsed from LLM output."
                }

            loprd_metadata = {
                "document_type": "loprd.json",
                "source_refined_user_goal_doc_id": task_input.refined_user_goal_doc_id,
                "source_assumptions_doc_id": task_input.assumptions_and_ambiguities_doc_id,
                "source_arca_feedback_doc_id": task_input.arca_feedback_doc_id,
                "generated_by_agent": self.AGENT_ID,
                "project_id": task_input.project_id,
                "task_id_product_analyst": getattr(task_input, 'task_id', str(uuid.uuid4())), # Assuming task_input has task_id
                "timestamp": datetime.datetime.utcnow().isoformat(),
                **confidence_metadata # Merge confidence data into metadata
            }
            
            stored_loprd_doc_id = await pcma_agent.store_document_content(
                project_id=task_input.project_id,
                collection_name="loprd_artifacts_collection", 
                document_content=loprd_json_content,
                metadata=loprd_metadata
            )
            if not stored_loprd_doc_id:
                 raise ValueError("PCMA store_document_content returned no document ID for LOPRD.")
            self._logger_instance.info(f"LOPRD artifact stored with doc_id: {stored_loprd_doc_id}.")
        except Exception as e_pcma_store:
            self._logger_instance.error(f"PCMA LOPRD storage failed: {e_pcma_store}", exc_info=True)
            return ProductAnalystAgentOutput(
                loprd_doc_id=f"error_pcma_store_failed_{uuid.uuid4().hex[:8]}", # Generate a unique error ID
                confidence_score=ConfidenceScore(value=0.1, level="Low", method="PCMAStorageError", reasoning=f"Failed to store LOPRD: {e_pcma_store}"),
                raw_llm_response=raw_llm_response_json,
                validation_errors=validation_errors_str # Carry over previous validation errors if any
            )
            
        # Construct final output
        # Ensure parsed_confidence_score_dict is valid before creating ConfidenceScore
        final_confidence_score_obj = ConfidenceScore(value=0.0, level="Error", method="InternalError", reasoning="Failed before confidence evaluation")
        if parsed_confidence_score_dict and isinstance(parsed_confidence_score_dict.get("value"), (float, int)):
            value = float(parsed_confidence_score_dict.get("value", 0.0))
            level = "High" if value >= 0.75 else ("Medium" if value >= 0.5 else "Low")
            final_confidence_score_obj = ConfidenceScore(
                value=value,
                level=level,
                method=parsed_confidence_score_dict.get("method", "Agent self-assessment"),
                reasoning=parsed_confidence_score_dict.get("explanation", "No explanation provided by LLM.")
            )
        elif validation_errors_str: # If there were validation errors earlier
             final_confidence_score_obj = ConfidenceScore(value=0.1, level="Low", method="LLMOutputValidation", reasoning=validation_errors_str)

        return ProductAnalystAgentOutput(
            loprd_doc_id=stored_loprd_doc_id or "error_loprd_storage_failed",
            confidence_score=final_confidence_score_obj,
            raw_llm_response=raw_llm_response_json,
            validation_errors=validation_errors_str
        ) 

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = ProductAnalystAgentInput.model_json_schema()
        # The agent output is LOPRD doc ID and confidence. The LOPRD itself is an artifact.
        # The prompt output (LLM output) is a structure containing LOPRD and confidence.
        # For the AgentCard, output_schema refers to ProductAnalystAgentOutput.
        output_schema = ProductAnalystAgentOutput.model_json_schema()

        # Prepare LOPRD schema for documentation if needed
        try:
            loprd_artifact_schema_for_docs = LOPRD.model_json_schema()
        except Exception:
            loprd_artifact_schema_for_docs = {"type": "object", "description": "Error loading LOPRD schema for docs."}
        
        # Prepare LLM expected output schema for documentation
        llm_expected_output_schema_for_docs = {
            "type": "object",
            "properties": {
                "loprd_artifact": loprd_artifact_schema_for_docs,
                "confidence_score": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "method": {"type": "string"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["value", "explanation"]
                }
            },
            "required": ["loprd_artifact", "confidence_score"]
        }

        module_path = ProductAnalystAgent_v1.__module__
        class_name = ProductAnalystAgent_v1.__name__

        return AgentCard(
            agent_id=ProductAnalystAgent_v1.AGENT_ID,
            name=ProductAnalystAgent_v1.AGENT_NAME,
            description=ProductAnalystAgent_v1.AGENT_DESCRIPTION,
            version=ProductAnalystAgent_v1.VERSION,
            input_schema=input_schema,
            output_schema=output_schema, # This is ProductAnalystAgentOutput
            # Documenting the primary artifact produced and the direct LLM output structure
            produced_artifacts_schemas={
                "loprd.json (stored_in_chroma)": loprd_artifact_schema_for_docs
            },
            llm_direct_output_schema=llm_expected_output_schema_for_docs, # New field for AgentCard
            project_dependencies=["chungoid-core"],
            categories=[cat.value for cat in [ProductAnalystAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=ProductAnalystAgent_v1.VISIBILITY.value,
            capability_profile={
                "generates_artifacts": ["LOPRD_JSON"],
                "consumes_artifacts": ["UserGoal", "ExistingLOPRD_JSON", "RefinementInstructions"],
                "primary_function": "Requirements Elaboration and Structuring"
            },
            metadata={
                "callable_fn_path": f"{module_path}.{class_name}"
            }
        )

# Example of how to get the card:
# card = ProductAnalystAgent_v1.get_agent_card_static()
# print(card.model_dump_json(indent=2)) 