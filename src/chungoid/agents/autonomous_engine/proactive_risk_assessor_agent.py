from __future__ import annotations

import logging
import datetime # For potential timestamping
import uuid
import json # For parsing LLM output if it's a JSON string
from typing import Any, Dict, Optional, Literal, ClassVar

from pydantic import BaseModel, Field

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1, 
    StoreArtifactInput, # Added
    LOPRD_ARTIFACTS_COLLECTION, # For reading LOPRD
    BLUEPRINT_ARTIFACTS_COLLECTION, # For reading Blueprint
    EXECUTION_PLANS_COLLECTION, # For reading MasterExecutionPlan
    RISK_ASSESSMENT_REPORTS_COLLECTION, 
    OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION,
    # Added ARTIFACT_TYPE constants
    ARTIFACT_TYPE_RISK_ASSESSMENT_REPORT_MD,
    ARTIFACT_TYPE_OPTIMIZATION_SUGGESTION_REPORT_MD,
    RetrieveArtifactOutput, # Added
    ARTIFACT_TYPE_AGENT_REFLECTION_JSON
)
from chungoid.utils.agent_registry import AgentCard # For AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard categories/visibility
from chungoid.schemas.agent_logs import GenericAgentReflection, LLMCallDetails, ToolCallDetails

logger = logging.getLogger(__name__)

PRAA_PROMPT_NAME = "proactive_risk_assessor_agent_v1.yaml" # In server_prompts/autonomous_engine/

# --- Input and Output Schemas for the Agent --- #

class ProactiveRiskAssessorInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this assessment task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    artifact_id: str = Field(..., description="ChromaDB ID of the artifact (LOPRD or Blueprint) to be assessed.")
    artifact_type: Literal["LOPRD", "Blueprint", "MasterExecutionPlan"] = Field(..., description="Type of the artifact being assessed.")
    loprd_document_id_for_blueprint_context: Optional[str] = Field(None, description="Optional LOPRD ID if artifact_type is Blueprint, to provide LOPRD context.")
    # Optional: Specific areas to focus on, or context about previous reviews
    focus_areas: Optional[List[str]] = Field(None, description="Optional list of specific areas to focus the risk assessment on.")
    # previous_assessment_ids: Optional[List[str]] = Field(None, description="IDs of previous assessment reports for context, if re-assessing.")

class ProactiveRiskAssessorOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    risk_assessment_report_doc_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated Risk Assessment Report (Markdown) is stored.")
    optimization_suggestions_report_doc_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated Optimization Suggestions Report (Markdown) is stored.")
    status: str = Field(..., description="Status of the assessment (e.g., SUCCESS, FAILURE_LLM, FAILURE_ARTIFACT_RETRIEVAL).")
    message: str = Field(..., description="A message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the thoroughness and accuracy of the assessment.")
    error_message: Optional[str] = Field(None, description="Error message if status is not SUCCESS.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging or deeper analysis.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")

class ProactiveRiskAssessorAgent_v1(BaseAgent[ProactiveRiskAssessorInput, ProactiveRiskAssessorOutput]):
    AGENT_ID: ClassVar[str] = "ProactiveRiskAssessorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Proactive Risk Assessor Agent v1"
    DESCRIPTION: ClassVar[str] = "Analyzes LOPRDs, Blueprints, or Plans for potential risks, issues, and optimization opportunities."
    VERSION: ClassVar[str] = "0.2.0" # Bumped version
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.RISK_ASSESSMENT
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC

    _llm_provider: LLMProvider # Now required
    _prompt_manager: PromptManager # Now required
    _project_chroma_manager: ProjectChromaManagerAgent_v1 # Now required
    _logger: logging.Logger

    def __init__(
        self, 
        llm_provider: LLMProvider, 
        prompt_manager: PromptManager, 
        project_chroma_manager: ProjectChromaManagerAgent_v1,
        system_context: Optional[Dict[str, Any]] = None,
        **kwargs # To catch other potential BaseAgent args like config, agent_id
    ):
        super().__init__(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            project_chroma_manager=project_chroma_manager,
            system_context=system_context,
            **kwargs
        )
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        self._project_chroma_manager = project_chroma_manager
        
        if system_context and "logger" in system_context:
            self._logger = system_context["logger"]
        else:
            self._logger = logging.getLogger(f"{__name__}.{self.AGENT_ID}")

        if not self._llm_provider:
            self._logger.error("LLMProvider not provided during initialization.")
            raise ValueError("LLMProvider is required for ProactiveRiskAssessorAgent_v1.")
        if not self._prompt_manager:
            self._logger.error("PromptManager not provided during initialization.")
            raise ValueError("PromptManager is required for ProactiveRiskAssessorAgent_v1.")
        if not self._project_chroma_manager:
            self._logger.error("ProjectChromaManagerAgent_v1 not provided during initialization.")
            raise ValueError("ProjectChromaManagerAgent_v1 is required for ProactiveRiskAssessorAgent_v1.")
        
        self._logger.info(f"{self.AGENT_ID} (v{self.VERSION}) initialized.")

    async def invoke_async(
        self,
        task_input: ProactiveRiskAssessorInput,
        full_context: Optional[Dict[str, Any]] = None,
    ) -> ProactiveRiskAssessorOutput:
        # Resolve dependencies, preferring those from full_context if available and different
        llm_provider = self._llm_provider
        prompt_manager = self._prompt_manager
        pcma_instance = self._project_chroma_manager
        logger_instance = self._logger

        if full_context:
            if "llm_provider" in full_context and full_context["llm_provider"] != self._llm_provider:
                llm_provider = full_context["llm_provider"]
                logger_instance.info("Using LLMProvider from full_context.")
            if "prompt_manager" in full_context and full_context["prompt_manager"] != self._prompt_manager:
                prompt_manager = full_context["prompt_manager"]
                logger_instance.info("Using PromptManager from full_context.")
            if "project_chroma_manager_agent_instance" in full_context and \
               full_context["project_chroma_manager_agent_instance"] != self._project_chroma_manager:
                pcma_instance = full_context["project_chroma_manager_agent_instance"]
                logger_instance.info("Using ProjectChromaManagerAgent from full_context.")
            if "system_context" in full_context and "logger" in full_context["system_context"] and \
               full_context["system_context"]["logger"] != self._logger:
                logger_instance = full_context["system_context"]["logger"]
        
        # Dependencies are now class members and required, so this initial check is less critical here
        # but good for defensive programming if invoke_async could be called bypassing __init__ (unlikely with BaseAgent pattern)
        if not llm_provider or not prompt_manager or not pcma_instance:
            err_msg = "LLMProvider, PromptManager, or ProjectChromaManager not available."
            logger_instance.error(f"{err_msg} for {self.AGENT_ID}")
            return ProactiveRiskAssessorOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_CONFIGURATION", message=err_msg, error_message=err_msg)

        logger_instance.info(f"{self.AGENT_ID} invoked for task {task_input.task_id}, artifact {task_input.artifact_id} ({task_input.artifact_type}) in project {task_input.project_id}")

        # --- Artifact Content Retrieval via PCMA ---
        loprd_content_for_prompt: Optional[str] = None
        blueprint_content_for_prompt: Optional[str] = None
        primary_artifact_content_str: Optional[str] = None
        
        # Conceptual PCMA method: pcma_instance.get_document_content_by_id(project_id, doc_id, collection_name)
        # Or a more specific one like: pcma_instance.get_planning_artifact_content(project_id, artifact_id)
        
        # Determine the source collection based on artifact_type
        source_collection_name: Optional[str] = None
        if task_input.artifact_type == "LOPRD":
            source_collection_name = LOPRD_ARTIFACTS_COLLECTION
        elif task_input.artifact_type == "Blueprint":
            source_collection_name = BLUEPRINT_ARTIFACTS_COLLECTION
        elif task_input.artifact_type == "MasterExecutionPlan":
            source_collection_name = EXECUTION_PLANS_COLLECTION
        else:
            logger_instance.error(f"Unsupported artifact_type for PRAA: {task_input.artifact_type}")
            # Defaulting to a general planning collection if specific type is unknown or unhandled properly elsewhere
            # This path should ideally not be hit if inputs are validated upstream.
            source_collection_name = LOPRD_ARTIFACTS_COLLECTION # Fallback, consider erroring out

        try:
            # Retrieve primary artifact
            primary_artifact_doc: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                base_collection_name=source_collection_name,
                document_id=task_input.artifact_id
            )
            if not (primary_artifact_doc and primary_artifact_doc.status == "SUCCESS" and primary_artifact_doc.content):
                raise ValueError(f"Primary artifact {task_input.artifact_id} from collection {source_collection_name} not found, content empty, or retrieval failed. Status: {primary_artifact_doc.status if primary_artifact_doc else 'N/A'}")
            primary_artifact_content_str = str(primary_artifact_doc.content) # Ensure content is string

            logger_instance.debug(f"Retrieved primary artifact content for {task_input.artifact_id} from {source_collection_name}")

            if task_input.artifact_type == "LOPRD":
                loprd_content_for_prompt = primary_artifact_content_str
            elif task_input.artifact_type == "Blueprint":
                blueprint_content_for_prompt = primary_artifact_content_str
                if task_input.loprd_document_id_for_blueprint_context:
                    loprd_context_doc: RetrieveArtifactOutput = await pcma_instance.retrieve_artifact(
                        base_collection_name=LOPRD_ARTIFACTS_COLLECTION, # LOPRDs are in their own collection
                        document_id=task_input.loprd_document_id_for_blueprint_context
                    )
                    if loprd_context_doc and loprd_context_doc.status == "SUCCESS" and loprd_context_doc.content:
                        loprd_content_for_prompt = str(loprd_context_doc.content) # Ensure content is string
                        logger_instance.debug(f"Retrieved LOPRD context for Blueprint: {task_input.loprd_document_id_for_blueprint_context}")
                    else:
                        logger_instance.warning(f"Could not retrieve LOPRD context {task_input.loprd_document_id_for_blueprint_context} for Blueprint analysis or content was empty. Status: {loprd_context_doc.status if loprd_context_doc else 'N/A'}")
            elif task_input.artifact_type == "MasterExecutionPlan":
                 # Assuming the prompt can handle plan content directly or it implies needing LOPRD/Blueprint for full context
                 # For now, let's treat it like blueprint; it might need its own prompt logic or data prep.
                blueprint_content_for_prompt = primary_artifact_content_str # Or a specific field for plan content
                logger_instance.warning("MasterExecutionPlan assessment might need LOPRD/Blueprint context not explicitly handled yet by this mock retrieval.")


        except Exception as e_pcma:
            msg = f"Failed to retrieve content for artifact_id {task_input.artifact_id} or its context via PCMA: {e_pcma}"
            logger_instance.error(msg, exc_info=True)
            return ProactiveRiskAssessorOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_ARTIFACT_RETRIEVAL", message=msg, error_message=str(e_pcma))
        
        if not primary_artifact_content_str : # Check if the main artifact was loaded
             msg = f"Primary artifact content for {task_input.artifact_id} ({task_input.artifact_type}) could not be loaded."
             logger_instance.error(msg)
             return ProactiveRiskAssessorOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_ARTIFACT_RETRIEVAL", message=msg, error_message=msg)

        # --- Prompt Rendering Data Preparation ---
        prompt_render_data = {
            "analysis_focus": task_input.artifact_type, # Matches prompt's 'analysis_focus'
            "loprd_json_content": loprd_content_for_prompt if loprd_content_for_prompt else "N/A - LOPRD not provided or not applicable for this focus.",
            "project_blueprint_md_content": blueprint_content_for_prompt if blueprint_content_for_prompt else "N/A - Blueprint not provided or not applicable for this focus.",
            # The prompt has 'artifact_content_str' as a top-level variable, but the detailed section uses specific ones.
            # Let's ensure the prompt uses 'loprd_json_content' and 'project_blueprint_md_content' directly.
            "focus_areas_list": task_input.focus_areas or [], # This was in old agent, prompt doesn't explicitly list it but could be useful general instruction.
            "current_date_iso": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        # --- LLM Interaction ---
        llm_response_json_str: Optional[str] = None
        llm_usage_metadata: Optional[Dict[str, Any]] = None
        try:
            logger_instance.info(f"Sending request to LLM for {task_input.artifact_type} assessment using prompt {PRAA_PROMPT_NAME}.")
            
            # Using generate_text_async_with_prompt_manager
            llm_response_json_str = await llm_provider.generate_text_async_with_prompt_manager(
                prompt_manager=prompt_manager,
                prompt_name=PRAA_PROMPT_NAME,
                prompt_data=prompt_render_data,
                sub_dir="autonomous_engine", # Ensure this matches where the prompt is located
                temperature=0.4 # Example, could be configured
                # model_id="gpt-4-turbo-preview" # Or from config
            )
            # Assuming generate_text_async_with_prompt_manager returns usage if available, or we get it separately
            # llm_usage_metadata = ... 

            if not llm_response_json_str or not llm_response_json_str.strip():
                raise ValueError("LLM returned empty or whitespace-only response.")
            
            parsed_llm_output = json.loads(llm_response_json_str)
            risk_report_md = parsed_llm_output.get("risk_assessment_report_md")
            opt_report_md = parsed_llm_output.get("optimization_opportunities_report_md") # Prompt uses "optimization_opportunities_report_md"
            assessment_confidence_data = parsed_llm_output.get("assessment_confidence")

            if not risk_report_md or not opt_report_md or not assessment_confidence_data:
                missing_fields = []
                if not risk_report_md: missing_fields.append("risk_assessment_report_md")
                if not opt_report_md: missing_fields.append("optimization_opportunities_report_md")
                if not assessment_confidence_data: missing_fields.append("assessment_confidence")
                raise ValueError(f"LLM JSON output missing required fields: {', '.join(missing_fields)}. Got: {llm_response_json_str[:500]}")
            
            agent_confidence_score = ConfidenceScore(**assessment_confidence_data)
            logger_instance.info("Successfully received and parsed assessment reports from LLM.")

        except PromptRenderError as e_prompt:
            msg = f"Prompt rendering failed for PRAA: {e_prompt}"
            logger_instance.error(msg, exc_info=True)
            return ProactiveRiskAssessorOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_PROMPT_RENDERING", message=msg, error_message=str(e_prompt))
        except json.JSONDecodeError as e_json:
            msg = f"Failed to parse LLM JSON response: {e_json}. Response was: {llm_response_json_str[:500] if llm_response_json_str else 'None'}"
            logger_instance.error(msg, exc_info=True)
            return ProactiveRiskAssessorOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_LLM_OUTPUT_PARSING", message=msg, error_message=str(e_json), llm_full_response=llm_response_json_str)
        except ValueError as e_val: # Catch missing fields or other validation issues from Pydantic
            msg = f"LLM output validation error or other value error: {e_val}"
            logger_instance.error(msg, exc_info=True)
            return ProactiveRiskAssessorOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_LLM_OUTPUT_VALIDATION", message=msg, error_message=str(e_val), llm_full_response=llm_response_json_str)
        except Exception as e_llm: # Catch-all for other LLM or processing issues
            msg = f"LLM interaction or output processing failed unexpectedly: {e_llm}"
            logger_instance.error(msg, exc_info=True)
            return ProactiveRiskAssessorOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_LLM", message=msg, error_message=str(e_llm), llm_full_response=llm_response_json_str)

        # --- Store Artifacts via PCMA ---
        risk_report_doc_id: Optional[str] = None
        opt_report_doc_id: Optional[str] = None
        current_utc_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Store Risk Assessment Report
        try:
            risk_metadata = {
                "artifact_type": ARTIFACT_TYPE_RISK_ASSESSMENT_REPORT_MD,
                "project_id": task_input.project_id,
                "assessed_artifact_id": task_input.artifact_id,
                "assessed_artifact_type": task_input.artifact_type,
                "confidence_score": assessment_confidence_data, # Store the dict form
                "source_agent_id": self.AGENT_ID,
                "source_task_id": task_input.task_id,
                "timestamp_utc": current_utc_time_iso
            }
            if hasattr(task_input, 'cycle_id') and task_input.cycle_id: # Check if cycle_id exists
                risk_metadata["cycle_id"] = task_input.cycle_id

            store_risk_report_input = StoreArtifactInput(
                base_collection_name=RISK_ASSESSMENT_REPORTS_COLLECTION,
                artifact_content=risk_report_md,
                metadata=risk_metadata,
                source_agent_id=self.AGENT_ID,
                source_task_id=task_input.task_id,
                cycle_id=getattr(task_input, 'cycle_id', None) # Get cycle_id if present
            )
            risk_store_output = await pcma_instance.store_artifact(store_risk_report_input)
            if risk_store_output.status == "SUCCESS":
                risk_report_doc_id = risk_store_output.document_id
                logger_instance.info(f"Stored Risk Assessment Report with ID: {risk_report_doc_id}")
            else:
                logger_instance.error(f"Failed to store Risk Assessment Report: {risk_store_output.message or risk_store_output.error_message}")
        except Exception as e_store_risk:
            logger_instance.error(f"Exception storing Risk Assessment Report: {e_store_risk}", exc_info=True)

        # Store Optimization Suggestions Report
        try:
            opt_metadata = {
                "artifact_type": ARTIFACT_TYPE_OPTIMIZATION_SUGGESTION_REPORT_MD,
                "project_id": task_input.project_id,
                "assessed_artifact_id": task_input.artifact_id,
                "assessed_artifact_type": task_input.artifact_type,
                "confidence_score": assessment_confidence_data, # Store the dict form
                "source_agent_id": self.AGENT_ID,
                "source_task_id": task_input.task_id,
                "timestamp_utc": current_utc_time_iso
            }
            if hasattr(task_input, 'cycle_id') and task_input.cycle_id: # Check if cycle_id exists
                opt_metadata["cycle_id"] = task_input.cycle_id

            store_opt_report_input = StoreArtifactInput(
                base_collection_name=OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION,
                artifact_content=opt_report_md,
                metadata=opt_metadata,
                source_agent_id=self.AGENT_ID,
                source_task_id=task_input.task_id,
                cycle_id=getattr(task_input, 'cycle_id', None) # Get cycle_id if present
            )
            opt_store_output = await pcma_instance.store_artifact(store_opt_report_input)
            if opt_store_output.status == "SUCCESS":
                opt_report_doc_id = opt_store_output.document_id
                logger_instance.info(f"Stored Optimization Suggestions Report with ID: {opt_report_doc_id}")
            else:
                logger_instance.error(f"Failed to store Optimization Suggestions Report: {opt_store_output.message or opt_store_output.error_message}")
        except Exception as e_store_opt:
            logger_instance.error(f"Exception storing Optimization Suggestions Report: {e_store_opt}", exc_info=True)

        # Store GenericAgentReflection
        try:
            input_artifacts_used = [task_input.artifact_id]
            if task_input.loprd_document_id_for_blueprint_context:
                input_artifacts_used.append(task_input.loprd_document_id_for_blueprint_context)
            
            output_artifacts_generated = []
            if risk_report_doc_id:
                output_artifacts_generated.append(risk_report_doc_id)
            if opt_report_doc_id:
                output_artifacts_generated.append(opt_report_doc_id)

            # Simplified LLMCallDetails; full details might require more from llm_provider
            llm_call = LLMCallDetails(
                model_name=llm_provider.model_id, # Assuming model_id is accessible
                prompt_template_id=PRAA_PROMPT_NAME,
                # temperature=0.4 (if accessible or passed through usage_metadata)
                # tokens_total=llm_usage_metadata.get("total_tokens") if llm_usage_metadata else None
            )

            reflection = GenericAgentReflection(
                project_id=task_input.project_id,
                cycle_id=getattr(task_input, 'cycle_id', None),
                agent_id=self.AGENT_ID,
                agent_version=self.VERSION,
                source_task_id=task_input.task_id,
                summary_of_activity=f"Performed risk and optimization assessment on {task_input.artifact_type} ID: {task_input.artifact_id}",
                input_artifact_ids_used=input_artifacts_used,
                output_artifact_ids_generated=output_artifacts_generated,
                decision_rationale="Assessment based on LLM analysis of provided artifact content and contextual LOPRD if applicable.",
                process_confidence_score=agent_confidence_score.value if agent_confidence_score else None,
                llm_calls=[llm_call],
                tool_calls=[], # PRAA currently doesn't use external tools via MCP
                contextual_adherence_explanation="Adherence to prompt for structured JSON output containing risk, optimization reports, and confidence.",
            )
            
            reflection_metadata = {
                "artifact_type": ARTIFACT_TYPE_AGENT_REFLECTION_JSON,
                "project_id": task_input.project_id,
                "reflected_agent_id": self.AGENT_ID,
                "reflected_task_id": task_input.task_id,
                "timestamp_utc": current_utc_time_iso
            }
            if hasattr(task_input, 'cycle_id') and task_input.cycle_id:
                reflection_metadata["cycle_id"] = task_input.cycle_id

            store_reflection_input = StoreArtifactInput(
                base_collection_name=AGENT_REFLECTIONS_AND_LOGS_COLLECTION, # Defined in PCMA
                artifact_content=reflection.model_dump(mode='json'), # Store as dict/json
                metadata=reflection_metadata,
                source_agent_id=self.AGENT_ID, # Self-reflection
                source_task_id=task_input.task_id,
                cycle_id=getattr(task_input, 'cycle_id', None)
            )
            reflection_store_output = await pcma_instance.store_artifact(store_reflection_input)
            if reflection_store_output.status == "SUCCESS":
                logger_instance.info(f"Stored GenericAgentReflection for PRAA task {task_input.task_id} with ID: {reflection_store_output.document_id}")
            else:
                logger_instance.error(f"Failed to store GenericAgentReflection for PRAA: {reflection_store_output.message or reflection_store_output.error_message}")

        except Exception as e_store_reflection:
            logger_instance.error(f"Exception storing GenericAgentReflection for PRAA: {e_store_reflection}", exc_info=True)


        # --- Prepare Final Output ---
        return ProactiveRiskAssessorOutput(
            task_id=task_input.task_id,
            project_id=task_input.project_id,
            risk_assessment_report_doc_id=risk_report_doc_id,
            optimization_suggestions_report_doc_id=opt_report_doc_id,
            status="SUCCESS",
            message="Successfully generated risk and optimization reports.",
            confidence_score=agent_confidence_score,
            llm_full_response=llm_response_json_str, # For debugging
            usage_metadata=llm_usage_metadata
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = ProactiveRiskAssessorInput.model_json_schema()
        output_schema = ProactiveRiskAssessorOutput.model_json_schema()
        module_path = ProactiveRiskAssessorAgent_v1.__module__
        class_name = ProactiveRiskAssessorAgent_v1.__name__

        return AgentCard(
            agent_id=ProactiveRiskAssessorAgent_v1.AGENT_ID,
            name=ProactiveRiskAssessorAgent_v1.AGENT_NAME,
            description=ProactiveRiskAssessorAgent_v1.DESCRIPTION,
            version=ProactiveRiskAssessorAgent_v1.VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[cat.value for cat in [ProactiveRiskAssessorAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=ProactiveRiskAssessorAgent_v1.VISIBILITY.value,
            pcma_collections_used=[
                LOPRD_ARTIFACTS_COLLECTION, 
                BLUEPRINT_ARTIFACTS_COLLECTION,
                EXECUTION_PLANS_COLLECTION,
                RISK_ASSESSMENT_REPORTS_COLLECTION,
                OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION 
            ],
            capability_profile={
                "analyzes_artifacts": ["LOPRD", "Blueprint", "MasterExecutionPlan"],
                "generates_reports": ["RiskAssessmentReport_Markdown", "OptimizationSuggestionsReport_Markdown"],
                "primary_function": "Artifact Quality Assurance and Risk Identification"
            },
            metadata={
                "callable_fn_path": f"{module_path}.{class_name}"
            }
        ) 