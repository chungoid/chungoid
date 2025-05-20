from __future__ import annotations

import logging
import datetime
import uuid
from typing import Any, Dict, Optional, List, ClassVar

from pydantic import BaseModel, Field

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1,
    StoreArtifactInput,
    PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION,
    ARTIFACT_TYPE_PROJECT_README_MD,
    ARTIFACT_TYPE_DOCS_DIRECTORY_MANIFEST_JSON,
    ARTIFACT_TYPE_DEPENDENCY_AUDIT_MD,
    ARTIFACT_TYPE_RELEASE_NOTES_MD,
    RetrieveArtifactOutput
)

logger = logging.getLogger(__name__)

# PROJECT_DOCUMENTATION_AGENT_PROMPT_NAME = "ProjectDocumentationAgent.yaml" # In server_prompts/autonomous_project_engine/
# Consistent prompt naming
PROMPT_NAME = "project_documentation_agent_v1_prompt.yaml"
PROMPT_SUB_DIR = "autonomous_engine"

# --- Input and Output Schemas based on the prompt file --- #

class ProjectDocumentationAgentInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this documentation task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    
    refined_user_goal_doc_id: str = Field(..., description="Document ID of the refined user goal specification.")
    project_blueprint_doc_id: str = Field(..., description="Document ID of the project blueprint.")
    master_execution_plan_doc_id: str = Field(..., description="Document ID of the master execution plan.")
    
    # Path to the root of the generated codebase. Agent will conceptually scan this.
    generated_code_root_path: str = Field(..., description="Path to the root directory of the generated codebase.")
    
    test_summary_doc_id: Optional[str] = Field(None, description="Optional document ID of the test summary report.")
    
    # For ARCA feedback loop if any (not used in initial MVP generation, but schema-ready)
    # arca_feedback_doc_id: Optional[str] = Field(None, description="Document ID for feedback from ARCA for refinement.")
    cycle_id: Optional[str] = Field(None, description="The ID of the current refinement cycle, passed by ARCA for lineage tracking.")
    
    # Optional: Specific documents or sections to focus on or regenerate
    # focus_sections: Optional[List[str]] = Field(None, description="Specific document sections to focus on or regenerate.")

class ProjectDocumentationAgentOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    
    readme_doc_id: Optional[str] = Field(None, description="ChromaDB document ID for the generated README.md.")
    docs_directory_doc_id: Optional[str] = Field(None, description="ChromaDB document ID for a manifest or bundle representing the generated 'docs/' directory content.")
    codebase_dependency_audit_doc_id: Optional[str] = Field(None, description="ChromaDB document ID for the generated codebase_dependency_audit.md.")
    release_notes_doc_id: Optional[str] = Field(None, description="Optional ChromaDB document ID for generated RELEASE_NOTES.md.")
    
    status: str = Field(..., description="Status of the documentation generation (e.g., SUCCESS, FAILURE_LLM).")
    message: str = Field(..., description="A message detailing the outcome.")
    agent_confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the generated documentation.")
    error_message: Optional[str] = Field(None, description="Error message if status is not SUCCESS.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging.")
    # usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")


class ProjectDocumentationAgent_v1(BaseAgent[ProjectDocumentationAgentInput, ProjectDocumentationAgentOutput]):
    AGENT_ID: ClassVar[str] = "ProjectDocumentationAgent_v1"
    AGENT_NAME: ClassVar[str] = "Project Documentation Agent v1"
    DESCRIPTION: ClassVar[str] = "Generates project documentation (README, API docs, dependency audit) from project artifacts and codebase context."
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.DOCUMENTATION_GENERATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC

    _llm_provider: LLMProvider
    _prompt_manager: PromptManager
    _pcma_agent: ProjectChromaManagerAgent_v1
    _logger: logging.Logger

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        pcma_agent: ProjectChromaManagerAgent_v1,
        system_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        if not llm_provider:
            raise ValueError("LLMProvider is required for ProjectDocumentationAgent_v1")
        if not prompt_manager:
            raise ValueError("PromptManager is required for ProjectDocumentationAgent_v1")
        if not pcma_agent:
            raise ValueError("ProjectChromaManagerAgent_v1 is required for ProjectDocumentationAgent_v1")
            
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        self._pcma_agent = pcma_agent
        
        # Ensure logger is properly initialized
        if system_context and "logger" in system_context and isinstance(system_context["logger"], logging.Logger):
            self._logger = system_context["logger"]
        else:
            self._logger = logging.getLogger(self.AGENT_ID)

    async def invoke_async(
        self,
        task_input: ProjectDocumentationAgentInput,
        full_context: Optional[Dict[str, Any]] = None, 
    ) -> ProjectDocumentationAgentOutput:
        logger_instance = self._logger
        logger_instance.info(f"{self.AGENT_ID} invoked for task {task_input.task_id} in project {task_input.project_id}")

        # --- 1. Retrieve artifact contents using PCMA ---
        refined_goal_content: Optional[str] = None
        blueprint_content: Optional[str] = None
        plan_content: Optional[str] = None
        test_summary_content: Optional[str] = None

        try:
            # Refined user goal
            doc_output: RetrieveArtifactOutput = await self._pcma_agent.retrieve_artifact(
                base_collection_name=PROJECT_GOALS_COLLECTION, # Assuming goals are in a general project_goals collection
                document_id=task_input.refined_user_goal_doc_id
            )
            if not (doc_output and doc_output.status == "SUCCESS" and doc_output.content):
                raise ValueError(f"Refined user goal document {task_input.refined_user_goal_doc_id} not found, content empty, or retrieval failed. Status: {doc_output.status if doc_output else 'N/A'}")
            refined_goal_content = str(doc_output.content)
            logger_instance.debug(f"Retrieved refined_user_goal_doc_id: {task_input.refined_user_goal_doc_id}")

            # Project blueprint
            doc_output: RetrieveArtifactOutput = await self._pcma_agent.retrieve_artifact(
                base_collection_name=BLUEPRINT_ARTIFACTS_COLLECTION, # Assuming blueprints are in BLUEPRINT_ARTIFACTS_COLLECTION
                document_id=task_input.project_blueprint_doc_id
            )
            if not (doc_output and doc_output.status == "SUCCESS" and doc_output.content):
                raise ValueError(f"Project blueprint document {task_input.project_blueprint_doc_id} not found, content empty, or retrieval failed. Status: {doc_output.status if doc_output else 'N/A'}")
            blueprint_content = str(doc_output.content)
            logger_instance.debug(f"Retrieved project_blueprint_doc_id: {task_input.project_blueprint_doc_id}")

            # Master execution plan
            doc_output: RetrieveArtifactOutput = await self._pcma_agent.retrieve_artifact(
                base_collection_name=EXECUTION_PLANS_COLLECTION, # Assuming plans are in EXECUTION_PLANS_COLLECTION
                document_id=task_input.master_execution_plan_doc_id
            )
            if not (doc_output and doc_output.status == "SUCCESS" and doc_output.content):
                raise ValueError(f"Master execution plan document {task_input.master_execution_plan_doc_id} not found, content empty, or retrieval failed. Status: {doc_output.status if doc_output else 'N/A'}")
            plan_content = str(doc_output.content)
            logger_instance.debug(f"Retrieved master_execution_plan_doc_id: {task_input.master_execution_plan_doc_id}")
            
            if task_input.test_summary_doc_id:
                doc_output: RetrieveArtifactOutput = await self._pcma_agent.retrieve_artifact(
                    base_collection_name=TEST_REPORTS_COLLECTION, # Assuming test summaries are in TEST_REPORTS_COLLECTION
                    document_id=task_input.test_summary_doc_id
                )
                if doc_output and doc_output.status == "SUCCESS" and doc_output.content: # It's optional
                    test_summary_content = str(doc_output.content)
                    logger_instance.debug(f"Retrieved test_summary_doc_id: {task_input.test_summary_doc_id}")
                else:
                    logger_instance.warning(f"Optional test_summary_doc_id {task_input.test_summary_doc_id} not found or content empty. Status: {doc_output.status if doc_output else 'N/A'}")
            
        except Exception as e:
            logger_instance.error(f"Failed to retrieve input documents via PCMA: {e}", exc_info=True)
            return ProjectDocumentationAgentOutput(
                task_id=task_input.task_id, 
                project_id=task_input.project_id, 
                status="FAILURE_INPUT_RETRIEVAL", 
                message=f"Error retrieving input documents: {e}",
                error_message=str(e)
            )

        # --- 2. Prepare Prompt Data & Render Prompt ---
        prompt_render_data = {
            "task_id": task_input.task_id,
            "project_id": task_input.project_id,
            "refined_user_goal_doc_id": task_input.refined_user_goal_doc_id, # Pass IDs for reference in prompt if needed
            "project_blueprint_doc_id": task_input.project_blueprint_doc_id,
            "master_execution_plan_doc_id": task_input.master_execution_plan_doc_id,
            "generated_code_root_path": task_input.generated_code_root_path,
            "test_summary_doc_id": task_input.test_summary_doc_id,
            # Actual content for the LLM to process:
            "_refined_user_goal_content": refined_goal_content,
            "_project_blueprint_content": blueprint_content,
            "_master_execution_plan_content": plan_content,
            "_test_summary_content": test_summary_content,
        }

        # --- 3. LLM Interaction --- 
        llm_full_response_str: Optional[str] = None
        try:
            logger_instance.debug(f"Attempting to generate documentation via LLM for task {task_input.task_id}.")
            # The prompt YAML defines user_prompt_template, system_prompt, input_schema, output_schema
            # generate_text_async_with_prompt_manager will use these.
            llm_full_response_str = await self._llm_provider.generate_text_async_with_prompt_manager(
                prompt_name=PROMPT_NAME,
                prompt_sub_dir=PROMPT_SUB_DIR,
                prompt_render_data=prompt_render_data,
                # output_pydantic_model=ProjectDocumentationAgentOutput, # This would be if output_schema in YAML matches Pydantic perfectly
                # For now, we expect JSON string based on prompt's output_schema and parse manually or with a simpler Pydantic model
                # that mirrors the prompt's output_schema.
                expected_response_type="json_string", # or a Pydantic model that matches the prompt's output_schema
                json_indent=4 # If LLM is asked to produce JSON directly
            )

            if not llm_full_response_str:
                raise ValueError("LLM returned an empty response.")
            
            logger_instance.info(f"Successfully received LLM response for task {task_input.task_id}.")
            # For now, assume llm_full_response_str is a JSON string matching the prompt's output_schema
            # This might require a temporary Pydantic model that mirrors the prompt's output_schema if it differs from ProjectDocumentationAgentOutput.
            # Or, parse it directly to extract fields.
            
            import json # Local import
            # This parsing assumes the LLM directly returns a JSON string that matches the fields
            # described in the prompt's `output_schema` section.
            llm_parsed_output = json.loads(llm_full_response_str)

            # Extract data based on prompt's output_schema keys
            readme_content = llm_parsed_output.get("readme_content") # Assuming LLM gives content, not just path
            docs_directory_content_map = llm_parsed_output.get("docs_directory_content_map") # e.g., {"api/module1.md": "content"}
            codebase_dependency_audit_content = llm_parsed_output.get("codebase_dependency_audit_content")
            release_notes_content = llm_parsed_output.get("release_notes_content")
            
            # Fields for enhanced reporting from prompt's output_schema
            contextual_adherence_summary = llm_parsed_output.get("contextual_adherence_summary")
            key_decision_rationale_summary = llm_parsed_output.get("key_decision_rationale_summary")
            confidence_score_value = llm_parsed_output.get("confidence_score_value")
            confidence_score_level = llm_parsed_output.get("confidence_score_level")
            confidence_score_reasoning = llm_parsed_output.get("confidence_score_reasoning")
            ambiguities_found_summary = llm_parsed_output.get("ambiguities_found_summary")

            if not readme_content: # Basic check
                 raise ValueError("LLM output missing critical readme_content.")

        except Exception as e:
            logger_instance.error(f"LLM interaction or output parsing failed for task {task_input.task_id}: {e}", exc_info=True)
            return ProjectDocumentationAgentOutput(
                task_id=task_input.task_id, 
                project_id=task_input.project_id, 
                status="FAILURE_LLM", 
                message=f"LLM interaction failed: {str(e)}", 
                error_message=str(e), 
                llm_full_response=llm_full_response_str
            )

        # --- 4. Store generated documents using PCMA ---
        # In a real implementation, each content string would be stored, and their IDs returned.
        # The prompt output_schema now returns doc_ids, so LLM is expected to provide content, 
        # and PCMA will store this content and generate the IDs. These mock IDs simulate that.
        try:
            readme_doc_id: Optional[str] = None
            docs_dir_id: Optional[str] = None
            dep_audit_id: Optional[str] = None
            release_notes_id: Optional[str] = None
            storage_success = True
            storage_error_message = ""

            if readme_content:
                try:
                    readme_metadata = {
                        "artifact_type": ARTIFACT_TYPE_PROJECT_README_MD,
                        "source_task_id": task_input.task_id,
                        "generated_by_agent": self.AGENT_ID,
                        "project_id": task_input.project_id,
                        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }
                    if confidence_score_value is not None and confidence_score_level is not None:
                        readme_metadata["confidence_value"] = float(confidence_score_value)
                        readme_metadata["confidence_level"] = str(confidence_score_level)
                    
                    store_input_readme = StoreArtifactInput(
                        base_collection_name=PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION,
                        artifact_content=readme_content,
                        metadata=readme_metadata,
                        source_agent_id=self.AGENT_ID,
                        source_task_id=task_input.task_id,
                        cycle_id=full_context.get("cycle_id") if full_context else None
                    )
                    store_result = await self._pcma_agent.store_artifact(args=store_input_readme)
                    if store_result.status == "SUCCESS" and store_result.document_id:
                        readme_doc_id = store_result.document_id
                        logger_instance.info(f"Stored README.md with doc_id: {readme_doc_id}")
                    else:
                        storage_success = False
                        err = store_result.error_message or f"Failed to store {ARTIFACT_TYPE_PROJECT_README_MD}"
                        storage_error_message += f" {err};"
                        logger_instance.error(f"Failed to store README.md: {err}")
                except Exception as e_store_readme:
                    storage_success = False
                    storage_error_message += f" Error storing README: {str(e_store_readme)};"
                    logger_instance.error(f"Exception storing README.md: {e_store_readme}", exc_info=True)


            if docs_directory_content_map and isinstance(docs_directory_content_map, dict):
                try:
                    docs_manifest_content = json.dumps(docs_directory_content_map, indent=2)
                    manifest_metadata = {
                        "artifact_type": ARTIFACT_TYPE_DOCS_DIRECTORY_MANIFEST_JSON,
                        "source_task_id": task_input.task_id,
                        "generated_by_agent": self.AGENT_ID,
                        "project_id": task_input.project_id,
                        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }
                    if confidence_score_value is not None and confidence_score_level is not None:
                        manifest_metadata["confidence_value"] = float(confidence_score_value)
                        manifest_metadata["confidence_level"] = str(confidence_score_level)

                    store_input_manifest = StoreArtifactInput(
                        base_collection_name=PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION,
                        artifact_content=docs_manifest_content,
                        metadata=manifest_metadata,
                        source_agent_id=self.AGENT_ID,
                        source_task_id=task_input.task_id,
                        cycle_id=full_context.get("cycle_id") if full_context else None
                    )
                    store_result = await self._pcma_agent.store_artifact(args=store_input_manifest)
                    if store_result.status == "SUCCESS" and store_result.document_id:
                        docs_dir_id = store_result.document_id
                        logger_instance.info(f"Stored docs directory manifest with doc_id: {docs_dir_id}")
                    else:
                        storage_success = False
                        err = store_result.error_message or f"Failed to store {ARTIFACT_TYPE_DOCS_DIRECTORY_MANIFEST_JSON}"
                        storage_error_message += f" {err};"
                        logger_instance.error(f"Failed to store docs directory manifest: {err}")
                except Exception as e_store_manifest:
                    storage_success = False
                    storage_error_message += f" Error storing docs manifest: {str(e_store_manifest)};"
                    logger_instance.error(f"Exception storing docs directory manifest: {e_store_manifest}", exc_info=True)
            
            if codebase_dependency_audit_content:
                try:
                    audit_metadata = {
                        "artifact_type": ARTIFACT_TYPE_DEPENDENCY_AUDIT_MD,
                        "source_task_id": task_input.task_id,
                        "generated_by_agent": self.AGENT_ID,
                        "project_id": task_input.project_id,
                        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }
                    if confidence_score_value is not None and confidence_score_level is not None:
                        audit_metadata["confidence_value"] = float(confidence_score_value)
                        audit_metadata["confidence_level"] = str(confidence_score_level)

                    store_input_audit = StoreArtifactInput(
                        base_collection_name=PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION,
                        artifact_content=codebase_dependency_audit_content,
                        metadata=audit_metadata,
                        source_agent_id=self.AGENT_ID,
                        source_task_id=task_input.task_id,
                        cycle_id=full_context.get("cycle_id") if full_context else None
                    )
                    store_result = await self._pcma_agent.store_artifact(args=store_input_audit)
                    if store_result.status == "SUCCESS" and store_result.document_id:
                        dep_audit_id = store_result.document_id
                        logger_instance.info(f"Stored codebase dependency audit with doc_id: {dep_audit_id}")
                    else:
                        storage_success = False
                        err = store_result.error_message or f"Failed to store {ARTIFACT_TYPE_DEPENDENCY_AUDIT_MD}"
                        storage_error_message += f" {err};"
                        logger_instance.error(f"Failed to store codebase dependency audit: {err}")
                except Exception as e_store_audit:
                    storage_success = False
                    storage_error_message += f" Error storing dependency audit: {str(e_store_audit)};"
                    logger_instance.error(f"Exception storing codebase dependency audit: {e_store_audit}", exc_info=True)

            if release_notes_content:
                try:
                    notes_metadata = {
                        "artifact_type": ARTIFACT_TYPE_RELEASE_NOTES_MD,
                        "source_task_id": task_input.task_id,
                        "generated_by_agent": self.AGENT_ID,
                        "project_id": task_input.project_id,
                        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }
                    if confidence_score_value is not None and confidence_score_level is not None:
                        notes_metadata["confidence_value"] = float(confidence_score_value)
                        notes_metadata["confidence_level"] = str(confidence_score_level)
                    
                    store_input_notes = StoreArtifactInput(
                        base_collection_name=PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION,
                        artifact_content=release_notes_content,
                        metadata=notes_metadata,
                        source_agent_id=self.AGENT_ID,
                        source_task_id=task_input.task_id,
                        cycle_id=full_context.get("cycle_id") if full_context else None
                    )
                    store_result = await self._pcma_agent.store_artifact(args=store_input_notes)
                    if store_result.status == "SUCCESS" and store_result.document_id:
                        release_notes_id = store_result.document_id
                        logger_instance.info(f"Stored release notes with doc_id: {release_notes_id}")
                    else:
                        storage_success = False
                        err = store_result.error_message or f"Failed to store {ARTIFACT_TYPE_RELEASE_NOTES_MD}"
                        storage_error_message += f" {err};"
                        logger_instance.error(f"Failed to store release notes: {err}")
                except Exception as e_store_notes:
                    storage_success = False
                    storage_error_message += f" Error storing release notes: {str(e_store_notes)};"
                    logger_instance.error(f"Exception storing release notes: {e_store_notes}", exc_info=True)
        
        except Exception as e_overall_store: # Should ideally be caught by individual try-excepts
            logger_instance.error(f"Overall error during document storage via PCMA: {e_overall_store}", exc_info=True)
            storage_success = False
            storage_error_message = str(e_overall_store)
            # Output what was successfully stored, but reflect the error in status

        final_status = "SUCCESS"
        final_message = "Documentation generated and all artifacts stored successfully."
        if not storage_success:
            final_status = "PARTIAL_SUCCESS_STORAGE_FAILED"
            final_message = f"Documentation generated, but one or more artifacts failed to store: {storage_error_message.strip()}"
            # If all critical docs failed, might consider it a FAILURE_OUTPUT_STORAGE
            if not readme_doc_id: # Example: README is critical
                final_status = "FAILURE_OUTPUT_STORAGE"

        # --- 5. Construct Final Output --- 
        agent_confidence = None
        if confidence_score_value is not None and confidence_score_level is not None:
            agent_confidence = ConfidenceScore(
                value=float(confidence_score_value),
                level=str(confidence_score_level),
                method="LLMGeneration_ProjectDocumentationAgent_v1",
                reasoning=str(confidence_score_reasoning) if confidence_score_reasoning else "Confidence provided by LLM."
            )
        
        return ProjectDocumentationAgentOutput(
            task_id=task_input.task_id, 
            project_id=task_input.project_id, 
            readme_doc_id=readme_doc_id,
            docs_directory_doc_id=docs_dir_id,
            codebase_dependency_audit_doc_id=dep_audit_id,
            release_notes_doc_id=release_notes_id,
            status=final_status, 
            message=final_message,
            agent_confidence_score=agent_confidence,
            error_message=storage_error_message.strip() if not storage_success else None,
            llm_full_response=llm_full_response_str
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=ProjectDocumentationAgent_v1.AGENT_ID,
            name=ProjectDocumentationAgent_v1.AGENT_NAME,
            description=ProjectDocumentationAgent_v1.DESCRIPTION,
            version="0.2.0",
            input_schema=ProjectDocumentationAgentInput.model_json_schema(),
            output_schema=ProjectDocumentationAgentOutput.model_json_schema(),
            categories=[cat.value for cat in [ProjectDocumentationAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=ProjectDocumentationAgent_v1.VISIBILITY.value,
            capability_profile={
                "generates_documentation": [
                    "README.md (content)", 
                    "API_Docs_Markdown (content map)", 
                    "DependencyAudit_Markdown (content)", 
                    "ReleaseNotes_Markdown (content, optional)"
                ],
                "consumes_artifacts_via_pcma_doc_ids": [
                    "RefinedUserGoal", 
                    "ProjectBlueprint", 
                    "MasterExecutionPlan", 
                    "TestSummary (optional)"
                ],
                "consumes_direct_inputs": ["generated_code_root_path"],
                "primary_function": "Automated Project Documentation Generation from comprehensive project context.",
                "pcma_collections_used": [
                    "project_goals",
                    "project_planning_artifacts",
                    "test_reports_collection",
                    "documentation_artifacts"
                ]
            },
            metadata={
                "prompt_name": PROMPT_NAME,
                "prompt_sub_dir": PROMPT_SUB_DIR,
                "callable_fn_path": f"{ProjectDocumentationAgent_v1.__module__}.{ProjectDocumentationAgent_v1.__name__}"
            }
        ) 