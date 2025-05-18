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

logger = logging.getLogger(__name__)

PROJECT_DOCUMENTATION_AGENT_PROMPT_NAME = "ProjectDocumentationAgent.yaml" # In server_prompts/autonomous_project_engine/

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


class ProjectDocumentationAgent_v1(BaseAgent):
    AGENT_ID: ClassVar[str] = "ProjectDocumentationAgent_v1"
    AGENT_NAME: ClassVar[str] = "Project Documentation Agent v1"
    DESCRIPTION: ClassVar[str] = "Generates project documentation (README, API docs, dependency audit) from project artifacts and codebase context."
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.DOCUMENTATION_GENERATION
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
    ) -> ProjectDocumentationAgentOutput:
        llm_provider = self._llm_provider
        prompt_manager = self._prompt_manager
        logger_instance = self._logger

        if full_context:
            if not llm_provider and "llm_provider" in full_context:
                llm_provider = full_context["llm_provider"]
            if not prompt_manager and "prompt_manager" in full_context:
                prompt_manager = full_context["prompt_manager"]
            if "system_context" in full_context and "logger" in full_context["system_context"]:
                if full_context["system_context"]["logger"] != self._logger:
                    logger_instance = full_context["system_context"]["logger"]
        
        if not llm_provider or not prompt_manager:
            err_msg = "LLMProvider or PromptManager not available."
            logger_instance.error(f"{err_msg} for {self.AGENT_ID}")
            task_id_fb = inputs.get("task_id", "unknown_task_dep_fail")
            proj_id_fb = inputs.get("project_id", "unknown_proj_dep_fail")
            return ProjectDocumentationAgentOutput(task_id=task_id_fb, project_id=proj_id_fb, status="FAILURE_CONFIGURATION", message=err_msg, error_message=err_msg)

        try:
            parsed_inputs = ProjectDocumentationAgentInput(**inputs)
        except Exception as e:
            logger_instance.error(f"Input parsing failed: {e}", exc_info=True)
            return ProjectDocumentationAgentOutput(task_id=inputs.get("task_id", "parse_err"), project_id=inputs.get("project_id", "parse_err"), status="FAILURE_INPUT_VALIDATION", message=str(e), error_message=str(e))

        logger_instance.info(f"{self.AGENT_ID} invoked for task {parsed_inputs.task_id} in project {parsed_inputs.project_id}")

        # --- MOCK: Retrieve artifact contents (would use PCMA) ---
        mock_goal_content = f"Mock content for refined_user_goal_doc_id: {parsed_inputs.refined_user_goal_doc_id}"
        mock_blueprint_content = f"Mock content for project_blueprint_doc_id: {parsed_inputs.project_blueprint_doc_id}"
        mock_plan_content = f"Mock content for master_execution_plan_doc_id: {parsed_inputs.master_execution_plan_doc_id}"
        mock_test_summary_content = f"Mock content for test_summary_doc_id: {parsed_inputs.test_summary_doc_id}" if parsed_inputs.test_summary_doc_id else None
        
        # Conceptual codebase scan path is available in parsed_inputs.generated_code_root_path

        # --- Prompt Rendering ---
        prompt_render_data = {
            "refined_user_goal_content": mock_goal_content,
            "project_blueprint_content": mock_blueprint_content,
            "master_execution_plan_content": mock_plan_content,
            "generated_code_root_path": parsed_inputs.generated_code_root_path,
            "test_summary_content": mock_test_summary_content,
            # Any other fields expected by the prompt
        }
        try:
            # Assuming the prompt is in autonomous_project_engine subdir based on design
            rendered_prompts = prompt_manager.render_prompt_template(
                PROJECT_DOCUMENTATION_AGENT_PROMPT_NAME, 
                prompt_render_data, 
                sub_dir="autonomous_project_engine" 
            )
            system_prompt = rendered_prompts.get("system_prompt")
            main_prompt = rendered_prompts.get("prompt_details") # Or user_prompt, depends on YAML structure
            if not system_prompt or not main_prompt:
                raise PromptRenderError("Missing system or main prompt content after rendering for ProjectDocumentationAgent.")
        except PromptRenderError as e:
            logger_instance.error(f"Prompt rendering failed: {e}", exc_info=True)
            return ProjectDocumentationAgentOutput(task_id=parsed_inputs.task_id, project_id=parsed_inputs.project_id, status="FAILURE_PROMPT_RENDERING", message=str(e), error_message=str(e))

        # --- LLM Interaction (Mocked) ---
        mock_llm_data = {
            "readme_content": f"# Project {parsed_inputs.project_id}\nThis is a mock README.",
            "api_docs_content": {
                "docs/api/main_module.md": "## Main Module API\nDetails...",
                "docs/api/utils_module.md": "## Utils Module API\nDetails..."
            },
            "dependency_audit_content": "### Dependency Audit\n- library_a: 1.0\n- library_b: 2.1",
            "release_notes_content": "### Release Notes v0.1\n- Initial release generated by ProjectDocumentationAgent_v1."
        }
        llm_full_response = json.dumps(mock_llm_data, indent=4) # Convert dict to JSON string for consistency
        
        try:
            # In a real scenario, LLM call would be here:
            # llm_full_response = await llm_provider.generate(prompt=main_prompt, system_prompt=system_prompt, temperature=0.3)
            # For MVP, we use the mock string and parse it.
            # The prompt instructs the LLM to provide structured output (e.g., JSON with markdown strings)
            # For MVP, we'll assume the mock_llm_output_str is this JSON string.
            
            import json # Local import for parsing
            parsed_llm_docs = json.loads(llm_full_response)
            
            readme_content = parsed_llm_docs.get("readme_content")
            # api_docs_content = parsed_llm_docs.get("api_docs_content") # This might be a dict of file_path: content
            # dependency_audit_content = parsed_llm_docs.get("dependency_audit_content")
            # release_notes_content = parsed_llm_docs.get("release_notes_content")

            if not readme_content: # Basic check
                 raise ValueError("LLM output missing critical readme_content.")
            logger_instance.info("Successfully (mock) received and parsed documentation content from LLM.")

        except Exception as e:
            logger_instance.error(f"LLM interaction or output parsing failed: {e}", exc_info=True)
            return ProjectDocumentationAgentOutput(
                task_id=parsed_inputs.task_id, 
                project_id=parsed_inputs.project_id, 
                status="FAILURE_LLM", 
                message=str(e), 
                error_message=str(e), 
                llm_full_response=llm_full_response
            )

        # --- MOCK: Store generated documents (would use PCMA) ---
        # In a real implementation, each document would be stored, and their IDs returned.
        # For docs_directory_doc_id, it might be an ID for a manifest file listing all doc files,
        # or a tarball ID, or simply a convention that sub-docs are stored relative to a base PCMA entry.
        # For MVP, we just generate mock IDs.
        mock_readme_doc_id = f"mock_readme_{parsed_inputs.project_id}_{uuid.uuid4()}"
        mock_docs_dir_id = f"mock_docs_dir_{parsed_inputs.project_id}_{uuid.uuid4()}" # Represents the collection of docs
        mock_dep_audit_id = f"mock_dep_audit_{parsed_inputs.project_id}_{uuid.uuid4()}"
        mock_release_notes_id = f"mock_release_notes_{parsed_inputs.project_id}_{uuid.uuid4()}" if parsed_llm_docs.get("release_notes_content") else None
        
        agent_confidence = ConfidenceScore(
            value=0.75, 
            level="Medium", 
            method="LLMGeneration_MVPHeuristic", 
            reasoning="Documentation generated by LLM based on provided artifact IDs and conceptual codebase scan. Manual review recommended."
        )

        return ProjectDocumentationAgentOutput(
            task_id=parsed_inputs.task_id,
            project_id=parsed_inputs.project_id,
            readme_doc_id=mock_readme_doc_id,
            docs_directory_doc_id=mock_docs_dir_id,
            codebase_dependency_audit_doc_id=mock_dep_audit_id,
            release_notes_doc_id=mock_release_notes_id,
            status="SUCCESS",
            message="Project documentation (mock) generated successfully.",
            agent_confidence_score=agent_confidence,
            llm_full_response=llm_full_response
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=ProjectDocumentationAgent_v1.AGENT_ID,
            name=ProjectDocumentationAgent_v1.AGENT_NAME,
            description=ProjectDocumentationAgent_v1.DESCRIPTION,
            version=ProjectDocumentationAgent_v1.VERSION,
            input_schema=ProjectDocumentationAgentInput.model_json_schema(),
            output_schema=ProjectDocumentationAgentOutput.model_json_schema(),
            categories=[cat.value for cat in [ProjectDocumentationAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=ProjectDocumentationAgent_v1.VISIBILITY.value,
            capability_profile={
                "generates_documentation": ["README.md", "API_Docs_Markdown", "DependencyAudit_Markdown", "ReleaseNotes_Markdown"],
                "consumes_artifacts": ["RefinedUserGoal", "ProjectBlueprint", "MasterExecutionPlan", "TestSummary", "CodebaseStructure (conceptual)"],
                "primary_function": "Automated Project Documentation Generation"
            },
            metadata={
                "prompt_name": PROJECT_DOCUMENTATION_AGENT_PROMPT_NAME,
                "prompt_sub_dir": "autonomous_project_engine",
                "callable_fn_path": f"{ProjectDocumentationAgent_v1.__module__}.{ProjectDocumentationAgent_v1.__name__}"
            }
        ) 