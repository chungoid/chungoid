from __future__ import annotations

import logging
import datetime # For potential timestamping
import uuid
import json
from typing import Any, Dict, Optional, Literal, ClassVar

from pydantic import BaseModel, Field

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1, 
    RetrieveArtifactOutput,
    StoreArtifactInput,
    TRACEABILITY_REPORTS_COLLECTION, 
    LOPRD_ARTIFACTS_COLLECTION, 
    BLUEPRINT_ARTIFACTS_COLLECTION,
    EXECUTION_PLANS_COLLECTION
)
from chungoid.utils.agent_registry import AgentCard # For AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard
from chungoid.runtime.agents.agent_base import AgentInputError # Removed AgentOutput, AgentOutputStatus

logger = logging.getLogger(__name__)

RTA_PROMPT_NAME = "requirements_tracer_agent_v1_prompt" # Changed from .yaml

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

class RequirementsTracerAgent_v1(BaseAgent[RequirementsTracerInput, RequirementsTracerOutput]):
    AGENT_ID: ClassVar[str] = "RequirementsTracerAgent_v1"
    AGENT_NAME: ClassVar[str] = "Requirements Tracer Agent v1"
    DESCRIPTION: ClassVar[str] = "Generates a traceability report (Markdown) between two development artifacts (e.g., LOPRD to Blueprint)."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = RTA_PROMPT_NAME # Added ClassVar for prompt name
    VERSION: ClassVar[str] = "0.2.0" # Bumped version
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS
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
            project_chroma_manager=project_chroma_manager, # Pass to super if BaseAgent handles it
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

        if not self._llm_provider: # Should be caught by super if BaseAgent checks
            self._logger.error("LLMProvider not provided during initialization.")
            raise ValueError("LLMProvider is required for RequirementsTracerAgent_v1.")
        if not self._prompt_manager:
            self._logger.error("PromptManager not provided during initialization.")
            raise ValueError("PromptManager is required for RequirementsTracerAgent_v1.")
        if not self._project_chroma_manager:
            self._logger.error("ProjectChromaManagerAgent_v1 not provided during initialization.")
            raise ValueError("ProjectChromaManagerAgent_v1 is required for RequirementsTracerAgent_v1.")
        
        self._logger.info(f"{self.AGENT_ID} (v{self.VERSION}) initialized.")

    async def invoke_async(
        self,
        task_input: RequirementsTracerInput,
        full_context: Optional[Dict[str, Any]] = None,
    ) -> RequirementsTracerOutput:
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
        
        if not llm_provider or not prompt_manager or not pcma_instance:
            err_msg = "LLMProvider, PromptManager, or ProjectChromaManager not available."
            logger_instance.error(f"{err_msg} for {self.AGENT_ID}")
            return RequirementsTracerOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_CONFIGURATION", message=err_msg, error_message=err_msg)

        logger_instance.info(f"{self.AGENT_ID} invoked for task {task_input.task_id} in project {task_input.project_id}. Tracing {task_input.source_artifact_type} ({task_input.source_artifact_doc_id}) to {task_input.target_artifact_type} ({task_input.target_artifact_doc_id}).")

        # --- Artifact Content Retrieval via PCMA ---
        source_artifact_content: Optional[str] = None
        target_artifact_content: Optional[str] = None

        def get_collection_for_artifact_type(artifact_type: Literal["LOPRD", "Blueprint", "UserStories", "MasterExecutionPlan", "CodeModules"]) -> str:
            if artifact_type == "LOPRD" or artifact_type == "UserStories": # Assuming UserStories are part of LOPRD context
                return LOPRD_ARTIFACTS_COLLECTION
            elif artifact_type == "Blueprint":
                return BLUEPRINT_ARTIFACTS_COLLECTION
            elif artifact_type == "MasterExecutionPlan":
                return EXECUTION_PLANS_COLLECTION
            # elif artifact_type == "CodeModules": # If RTA needs to trace to code
            #     return LIVE_CODEBASE_COLLECTION # Or GENERATED_CODE_ARTIFACTS_COLLECTION
            else:
                logger_instance.warning(f"Unknown artifact type '{artifact_type}' for collection mapping. Defaulting to LOPRD_ARTIFACTS_COLLECTION.")
                # This default is a fallback and ideally inputs should be validated for supported types.
                return LOPRD_ARTIFACTS_COLLECTION 

        try:
            source_collection = get_collection_for_artifact_type(task_input.source_artifact_type)
            source_artifact_doc = await pcma_instance.retrieve_artifact(
                base_collection_name=source_collection,
                document_id=task_input.source_artifact_doc_id
            )
            if not source_artifact_doc or not source_artifact_doc.content:
                raise ValueError(f"Source artifact {task_input.source_artifact_doc_id} ({task_input.source_artifact_type}) from {source_collection} not found or content empty.")
            source_artifact_content = source_artifact_doc.content
            logger_instance.debug(f"Retrieved source artifact content for {task_input.source_artifact_doc_id} from {source_collection}")

            target_collection = get_collection_for_artifact_type(task_input.target_artifact_type)
            target_artifact_doc = await pcma_instance.retrieve_artifact(
                base_collection_name=target_collection,
                document_id=task_input.target_artifact_doc_id
            )
            if not target_artifact_doc or not target_artifact_doc.content:
                raise ValueError(f"Target artifact {task_input.target_artifact_doc_id} ({task_input.target_artifact_type}) from {target_collection} not found or content empty.")
            target_artifact_content = target_artifact_doc.content
            logger_instance.debug(f"Retrieved target artifact content for {task_input.target_artifact_doc_id} from {target_collection}")

        except Exception as e_pcma:
            msg = f"Failed to retrieve content for source/target artifacts via PCMA: {e_pcma}"
            logger_instance.error(msg, exc_info=True)
            return RequirementsTracerOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_ARTIFACT_RETRIEVAL", message=msg, error_message=str(e_pcma))

        # --- Prompt Rendering Data Preparation ---
        prompt_render_data = {
            "source_artifact_type": task_input.source_artifact_type,
            "source_artifact_content": source_artifact_content,
            "target_artifact_type": task_input.target_artifact_type,
            "target_artifact_content": target_artifact_content,
            "project_name": task_input.project_id, # Using project_id as project_name for the prompt
            "current_date_iso": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        # --- LLM Interaction ---
        llm_response_json_str: Optional[str] = None # Renamed for clarity
        llm_usage_metadata: Optional[Dict[str, Any]] = None
        try:
            logger_instance.info(f"Sending request to LLM for traceability report generation using prompt {RTA_PROMPT_NAME}.")
            llm_response_json_str = await llm_provider.generate_text_async_with_prompt_manager(
                prompt_name=RTA_PROMPT_NAME,
                prompt_version=self.VERSION,
                prompt_render_data=prompt_render_data,
                prompt_sub_path="autonomous_engine",
                temperature=0.3,
                model_id=None,
                expected_json_schema=None
            )
            # llm_usage_metadata = ... # If returned by the call

            if not llm_response_json_str or not isinstance(llm_response_json_str, str) or not llm_response_json_str.strip():
                raise ValueError("LLM returned empty or non-string response where a JSON string was expected.")
            
            # Parse the JSON response expecting traceability_report_md and assessment_confidence
            parsed_llm_output = json.loads(llm_response_json_str)
            generated_trace_report_md = parsed_llm_output.get("traceability_report_md")
            assessment_confidence_data = parsed_llm_output.get("assessment_confidence")

            if not generated_trace_report_md or not assessment_confidence_data:
                missing_fields = []
                if not generated_trace_report_md: missing_fields.append("traceability_report_md")
                if not assessment_confidence_data: missing_fields.append("assessment_confidence")
                raise ValueError(f"LLM JSON output missing required fields: {', '.join(missing_fields)}. Got: {llm_response_json_str[:500]}")

            # Validate and structure the confidence score from LLM
            llm_confidence = ConfidenceScore(**assessment_confidence_data)

            logger_instance.info("Successfully received and parsed traceability report and confidence from LLM.")
        except PromptRenderError as e_prompt:
            logger_instance.error(f"Prompt rendering failed for {RTA_PROMPT_NAME}: {e_prompt}", exc_info=True)
            return RequirementsTracerOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_PROMPT_RENDERING", message=str(e_prompt), error_message=str(e_prompt))
        except json.JSONDecodeError as e_json:
            logger_instance.error(f"Failed to decode LLM JSON response: {e_json}. Response: {llm_response_json_str[:500]}...", exc_info=True)
            return RequirementsTracerOutput(
                task_id=task_input.task_id, 
                project_id=task_input.project_id, 
                status="FAILURE_LLM_OUTPUT_PARSING", 
                message=f"LLM response not valid JSON: {e_json}", 
                error_message=str(e_json), 
                llm_full_response=llm_response_json_str
            )
        except Exception as e_llm: # Catches ValueErrors from missing fields or other LLM/parsing issues
            logger_instance.error(f"LLM interaction or output processing failed: {e_llm}", exc_info=True)
            return RequirementsTracerOutput(
                task_id=task_input.task_id, 
                project_id=task_input.project_id, 
                status="FAILURE_LLM", 
                message=str(e_llm), 
                error_message=str(e_llm), 
                llm_full_response=llm_response_json_str
            )

        # --- Store Traceability Report in ChromaDB (via PCMA) ---
        report_doc_id: Optional[str] = None
        try:
            report_metadata = {
                "report_type": "traceability_matrix",
                "source_artifact_id": task_input.source_artifact_doc_id,
                "source_artifact_type": task_input.source_artifact_type,
                "target_artifact_id": task_input.target_artifact_doc_id,
                "target_artifact_type": task_input.target_artifact_type,
                "generated_by_agent": self.AGENT_ID,
                "task_id": task_input.task_id,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "confidence_value": llm_confidence.value,
                "confidence_level": llm_confidence.level,
                "confidence_explanation": llm_confidence.explanation,
                "confidence_method": llm_confidence.method,
                "project_id": task_input.project_id,
            }
            store_input_args = StoreArtifactInput(
                base_collection_name=TRACEABILITY_REPORTS_COLLECTION,
                artifact_content=generated_trace_report_md,
                metadata=report_metadata,
                # document_id can be omitted for PCMA to generate one, or set explicitly if needed.
                # Assuming RTA doesn't pre-determine the ID for the report itself.
                source_agent_id=self.AGENT_ID, # Added for lineage
                source_task_id=task_input.task_id # Added for lineage
            )
            store_operation_result = await pcma_instance.store_artifact(args=store_input_args)
            if store_operation_result.status == "SUCCESS" and store_operation_result.document_id:
                report_doc_id = store_operation_result.document_id
                logger_instance.info(f"Stored traceability report with doc_id: {report_doc_id}")
            else:
                error_msg_store = store_operation_result.error_message or "PCMA store_artifact failed without specific error."
                logger_instance.error(f"Failed to store traceability report via PCMA: {error_msg_store}")
                # Update output for storage failure, but keep generated report data
                return RequirementsTracerOutput(
                    task_id=task_input.task_id,
                    project_id=task_input.project_id,
                    traceability_report_doc_id=None,
                    status="PARTIAL_SUCCESS_STORAGE_FAILED",
                    message=f"Traceability report generated by LLM but failed to store: {error_msg_store}",
                    agent_confidence_score=llm_confidence,
                    llm_full_response=generated_trace_report_md,
                    usage_metadata=llm_usage_metadata,
                    error_message=error_msg_store
                )
            
        except Exception as e_store: # This broad except might catch the specific store failure if not handled above
            logger_instance.error(f"Failed to store traceability report via PCMA: {e_store}", exc_info=True)
            # Non-fatal for now, but update status and don't return a doc_id if storage failed
            return RequirementsTracerOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                traceability_report_doc_id=None,
                status="PARTIAL_SUCCESS_STORAGE_FAILED",
                message=f"Traceability report generated by LLM but failed to store: {e_store}",
                agent_confidence_score=llm_confidence, # Use the parsed LLM confidence
                llm_full_response=generated_trace_report_md, # Store only MD part for this field
                usage_metadata=llm_usage_metadata,
                error_message=str(e_store)
            )

        return RequirementsTracerOutput(
            task_id=task_input.task_id,
            project_id=task_input.project_id,
            traceability_report_doc_id=report_doc_id,
            status="SUCCESS" if report_doc_id else "PARTIAL_SUCCESS_STORAGE_FAILED",
            message=f"Traceability report generated. Stored as doc_id: {report_doc_id}",
            agent_confidence_score=llm_confidence, # Use parsed LLM confidence
            llm_full_response=generated_trace_report_md, # Store only MD part for this field
            usage_metadata=llm_usage_metadata
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = RequirementsTracerInput.model_json_schema()
        output_schema = RequirementsTracerOutput.model_json_schema()
        module_path = RequirementsTracerAgent_v1.__module__
        class_name = RequirementsTracerAgent_v1.__name__

        # Schema for the LLM's direct output (JSON object)
        llm_direct_output_schema = {
            "type": "object",
            "properties": {
                "traceability_report_md": {"type": "string"},
                "assessment_confidence": ConfidenceScore.model_json_schema()
            },
            "required": ["traceability_report_md", "assessment_confidence"]
        }

        return AgentCard(
            agent_id=RequirementsTracerAgent_v1.AGENT_ID,
            name=RequirementsTracerAgent_v1.AGENT_NAME,
            description=RequirementsTracerAgent_v1.DESCRIPTION,
            version=RequirementsTracerAgent_v1.VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            llm_direct_output_schema=llm_direct_output_schema, # Add schema for LLM's JSON output
            categories=[cat.value for cat in [RequirementsTracerAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=RequirementsTracerAgent_v1.VISIBILITY.value,
            pcma_collections_used=[
                LOPRD_ARTIFACTS_COLLECTION, 
                BLUEPRINT_ARTIFACTS_COLLECTION, 
                EXECUTION_PLANS_COLLECTION,
                # Could also add LIVE_CODEBASE_COLLECTION if tracing to code becomes a feature
                TRACEABILITY_REPORTS_COLLECTION 
            ],
            capability_profile={
                "analyzes_artifacts": ["LOPRD", "Blueprint", "MasterExecutionPlan"],
                "generates_reports": ["TraceabilityReport_Markdown"],
                "primary_function": "Requirements Traceability Verification"
            },
            metadata={
                "callable_fn_path": f"{RequirementsTracerAgent_v1.__module__}.{RequirementsTracerAgent_v1.__name__}"
            }
        ) 