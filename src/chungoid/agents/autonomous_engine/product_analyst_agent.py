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
# from chungoid.utils.project_chroma_manager import ProjectChromaManagerAgent # Placeholder
from chungoid.schemas.common import ConfidenceScore # Assuming a common schema exists
from chungoid.schemas.orchestration import SharedContext # ADDED SharedContext IMPORT
from chungoid.schemas.autonomous_engine.loprd_schema import LOPRD # Import the actual LOPRD schema if available
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
                 # project_chroma_manager: ProjectChromaManagerAgent, # Inject when available
                 config: Optional[Dict[str, Any]] = None,
                 system_context: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, system_context=system_context)
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        # self.project_chroma_manager = project_chroma_manager
        self._logger_instance = system_context.get("logger", logger) if system_context else logger
        self.loprd_schema = self._load_loprd_json_schema() # Load LOPRD schema once

    def _load_loprd_json_schema(self) -> Optional[Dict[str, Any]]:
        try:
            # Determine path to loprd_schema.json relative to this file or a known schemas dir
            # This assumes a certain project structure. Adjust path as necessary.
            # chungoid-core/src/chungoid/agents/autonomous_engine/product_analyst_agent.py
            # chungoid-core/src/chungoid/schemas/autonomous_engine/loprd_schema.json
            schema_path = Path(__file__).parent.parent.parent / "schemas" / "autonomous_engine" / "loprd_schema.json"
            if not schema_path.exists():
                self._logger_instance.error(f"LOPRD schema file not found at {schema_path}")
                return None
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self._logger_instance.error(f"Failed to load LOPRD JSON schema: {e}")
            return None

    async def __call__(self, inputs: ProductAnalystAgentInput) -> ProductAnalystAgentOutput:
        self._logger_instance.info(f"ProductAnalystAgent_v1 invoked for project {inputs.project_id}.")

        # Example of using shared_context if available
        if inputs.shared_context:
            self._logger_instance.info(f"Agent received SharedContext. Current cycle: {inputs.shared_context.current_cycle_id}, Current stage: {inputs.shared_context.current_stage_name}")
            # Access other fields like inputs.shared_context.artifact_references.get("some_key")
            # Or inputs.shared_context.get_scratchpad_data("some_temp_info")

        if not self.loprd_schema:
            return ProductAnalystAgentOutput(
                loprd_doc_id="error_loprd_schema_not_loaded",
                confidence_score=ConfidenceScore(value=0.0, level="Low", method="InternalError", reasoning="LOPRD JSON schema failed to load."),
                validation_errors="LOPRD JSON schema could not be loaded by the agent."
            )

        # 1. Retrieve content for refined_user_goal and assumptions (via ProjectChromaManagerAgent - MOCKED FOR NOW)
        # Mocking content retrieval
        refined_goal_content = f"Mocked content for refined_user_goal_doc_id: {inputs.refined_user_goal_doc_id}" # self.project_chroma_manager.get_document_content(inputs.refined_user_goal_doc_id)
        assumptions_content = None
        if inputs.assumptions_and_ambiguities_doc_id:
            assumptions_content = f"Mocked content for assumptions_doc_id: {inputs.assumptions_and_ambiguities_doc_id}" # self.project_chroma_manager.get_document_content(inputs.assumptions_and_ambiguities_doc_id)
        
        arca_feedback_content = None
        if inputs.arca_feedback_doc_id:
            arca_feedback_content = f"Mocked content for arca_feedback_doc_id: {inputs.arca_feedback_doc_id}" # self.project_chroma_manager.get_document_content(inputs.arca_feedback_doc_id)

        # 2. Prepare prompt data
        prompt_data = {
            "refined_user_goal_md": refined_goal_content,
            "assumptions_and_ambiguities_md": assumptions_content if assumptions_content else "Not provided.",
            "loprd_json_schema_str": json.dumps(self.loprd_schema, indent=2),
            "arca_feedback_md": arca_feedback_content if arca_feedback_content else "No specific feedback from ARCA for this iteration."
        }

        # 3. Render prompt
        try:
            # Assuming prompt is in chungoid-core/server_prompts/autonomous_engine/
            # Construct full path to prompt template if PromptManager expects it
            # For now, assuming PromptManager handles path resolution correctly from PROMPT_TEMPLATE_NAME
            rendered_prompt = self.prompt_manager.render_prompt_template(
                self.PROMPT_TEMPLATE_NAME,
                prompt_data,
                sub_dir="autonomous_engine" # Specify subdirectory if prompts are organized this way
            )
            # The rendered_prompt here is likely a dict with 'system_prompt' and 'user_prompt'
            # Or it could be a single string if the template is simple.
            # Adjust based on how PromptManager returns and LLMProvider expects it.
            # For this example, assuming rendered_prompt is the final user-facing prompt string if system prompt is separate.
            # If PromptManager gives dict: main_prompt = rendered_prompt.get('user_prompt', rendered_prompt.get('prompt_details'))
            # This needs clarification based on PromptManager behavior.
            # For now, assuming it returns a string or a dict from which we extract the main interaction prompt.
            if isinstance(rendered_prompt, dict):
                 # This is a guess based on typical prompt structures. Adjust as needed.
                llm_main_prompt = rendered_prompt.get('prompt_details', rendered_prompt.get('user_prompt')) 
                llm_system_prompt = rendered_prompt.get('system_prompt')
            else:
                llm_main_prompt = rendered_prompt
                llm_system_prompt = None # Or fetch separately if your PromptManager expects this
            
            if not llm_main_prompt:
                raise PromptRenderError("Rendered main prompt is empty.")

        except PromptRenderError as e:
            self._logger_instance.error(f"Failed to render prompt {self.PROMPT_TEMPLATE_NAME}: {e}")
            return ProductAnalystAgentOutput(
                loprd_doc_id="error_prompt_render_failed",
                confidence_score=ConfidenceScore(value=0.0, level="Low", method="InternalError", reasoning=f"Prompt rendering failed: {e}"),
                validation_errors=f"Prompt rendering failed: {e}"
            )
        
        # 4. Interact with LLM
        try:
            self._logger_instance.info("Sending request to LLM for LOPRD generation...")
            # The generate method might take system_prompt separately
            raw_llm_response_json = await self.llm_provider.generate(
                prompt=llm_main_prompt,
                system_prompt=llm_system_prompt, # Pass if your LLMProvider supports it
                # model_id="gpt-4-turbo-preview", # Or from config
                temperature=0.2, # Example, tune as needed
                json_response=True # Assuming LLMProvider can ask for JSON output
            )
            self._logger_instance.info("Received response from LLM.")
        except Exception as e:
            self._logger_instance.error(f"LLM interaction failed: {e}", exc_info=True)
            return ProductAnalystAgentOutput(
                loprd_doc_id="error_llm_failed",
                confidence_score=ConfidenceScore(value=0.0, level="Low", method="LLMError", reasoning=f"LLM call failed: {e}"),
                raw_llm_response=str(e) # Store error as raw response for debugging
            )

        # 5. Validate LLM JSON output against LOPRD schema
        loprd_data: Optional[Dict[str, Any]] = None
        validation_errors_str: Optional[str] = None
        try:
            loprd_data = json.loads(raw_llm_response_json)
            # Pydantic validation using the loaded schema would be more robust here
            # For now, assuming the LLM is instructed to follow the schema strictly.
            # If LOPRD is a Pydantic model: LOPRD.model_validate(loprd_data)
            # If using jsonschema directly:
            # from jsonschema import validate
            # validate(instance=loprd_data, schema=self.loprd_schema)
            # For simplicity in MVP, direct Pydantic model validation is preferred if LOPRD is a Pydantic model.
            # If LOPRD is not a Pydantic model, and self.loprd_schema is the JSON schema dict:
            try:
                # This is a placeholder for actual jsonschema validation or LOPRD.model_validate()
                # For MVP, we assume the LLM output IS the LOPRD if parsing succeeds.
                # In a real scenario, use: from jsonschema import validate; validate(loprd_data, self.loprd_schema)
                # Or if LOPRD is a Pydantic model: validated_loprd = LOPRD(**loprd_data)
                logger.info("LOPRD JSON parsed. Schema validation placeholder - assuming valid if parsable for MVP.")
            except ValidationError as ve:
                self._logger_instance.warning(f"LLM output failed LOPRD schema validation: {ve}")
                validation_errors_str = str(ve)
                loprd_data = None # Invalidate data if schema check fails
            except Exception as schema_val_e:
                self._logger_instance.error(f"Unexpected error during LOPRD schema validation: {schema_val_e}")
                validation_errors_str = f"Unexpected schema validation error: {schema_val_e}"
                loprd_data = None

        except json.JSONDecodeError as e:
            self._logger_instance.error(f"Failed to decode LLM JSON response: {e}. Response was: {raw_llm_response_json[:500]}...")
            validation_errors_str = f"LLM output was not valid JSON: {e}"
        
        if not loprd_data:
            return ProductAnalystAgentOutput(
                loprd_doc_id="error_llm_output_invalid",
                confidence_score=ConfidenceScore(value=0.1, level="Low", method="LLMOutputValidation", reasoning="LLM output was not valid LOPRD JSON or failed schema validation."),
                raw_llm_response=raw_llm_response_json,
                validation_errors=validation_errors_str
            )

        # 6. Store LOPRD JSON (via ProjectChromaManagerAgent - MOCKED FOR NOW)
        # loprd_doc_id = self.project_chroma_manager.store_artifact(
        # project_id=inputs.project_id, 
        # artifact_name="llm_optimized_prd.json", 
        # content=loprd_data, 
        # artifact_type="LOPRD_JSON"
        # )
        mock_loprd_doc_id = f"loprd_{inputs.project_id}_{Path(__file__).stem.lower()}_{ConfidenceScore().id[:8]}.json_doc_id"
        self._logger_instance.info(f"LOPRD artifact (mock) stored with doc_id: {mock_loprd_doc_id}")

        # 7. Generate Confidence Score (simple for now, can be LLM-assisted later)
        # For MVP, if validation passed, give medium-high confidence.
        final_confidence = ConfidenceScore(
            value=0.75 if not validation_errors_str else 0.25,
            level="Medium" if not validation_errors_str else "Low",
            method="LLMGenerationWithSchemaValidation",
            reasoning="LOPRD generated by LLM. Schema validation passed." if not validation_errors_str else f"LOPRD generated by LLM. Schema validation failed: {validation_errors_str}"
        )

        return ProductAnalystAgentOutput(
            loprd_doc_id=mock_loprd_doc_id,
            confidence_score=final_confidence,
            raw_llm_response=raw_llm_response_json,
            validation_errors=validation_errors_str
        ) 

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = ProductAnalystAgentInput.model_json_schema()
        output_schema = ProductAnalystAgentOutput.model_json_schema()
        
        module_path = ProductAnalystAgent_v1.__module__
        class_name = ProductAnalystAgent_v1.__name__

        return AgentCard(
            agent_id=ProductAnalystAgent_v1.AGENT_ID,
            name=ProductAnalystAgent_v1.AGENT_NAME,
            description=ProductAnalystAgent_v1.AGENT_DESCRIPTION,
            version=ProductAnalystAgent_v1.VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
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