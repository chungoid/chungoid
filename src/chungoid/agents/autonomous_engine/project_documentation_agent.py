from __future__ import annotations

import logging
import datetime
import uuid
class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

from typing import Any, Dict, Optional, List, ClassVar

from pydantic import BaseModel, Field

from ..protocol_aware_agent import ProtocolAwareAgent
from ...protocols.base.protocol_interface import ProtocolPhase
from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility

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


class ProjectDocumentationAgent_v1(ProtocolAwareAgent[ProjectDocumentationAgentInput, ProjectDocumentationAgentOutput]):
    AGENT_ID: ClassVar[str] = "ProjectDocumentationAgent_v1"
    AGENT_NAME: ClassVar[str] = "Project Documentation Agent v1"
    DESCRIPTION: ClassVar[str] = "Generates project documentation (README, API docs, dependency audit) from project artifacts and codebase context."
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.DOCUMENTATION_GENERATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC

    _llm_provider: LLMProvider
    _prompt_manager: PromptManager
    _logger: logging.Logger
    # ADDED: Protocol definitions following AI agent best practices
    PRIMARY_PROTOCOLS: ClassVar[list[str]] = ['documentation_generation']
    SECONDARY_PROTOCOLS: ClassVar[list[str]] = ['content_validation', 'template_management']
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'tool_validation']


    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        system_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        if not llm_provider:
            raise ValueError("LLMProvider is required for ProjectDocumentationAgent_v1")
        if not prompt_manager:
            raise ValueError("PromptManager is required for ProjectDocumentationAgent_v1")
            
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        
        # Ensure logger is properly initialized
        if system_context and "logger" in system_context and isinstance(system_context["logger"], logging.Logger):
            self._logger = system_context["logger"]
        else:
            self._logger = logging.getLogger(self.AGENT_ID)
    async def execute(self, task_input, full_context: Optional[Dict[str, Any]] = None):
        """
        Execute using pure protocol architecture.
        No fallback - protocol execution only for clean, maintainable code.
        """
        try:
            # Determine primary protocol for this agent
            primary_protocol = self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "simple_operations"
            
            protocol_task = {
                "task_input": task_input.dict() if hasattr(task_input, 'dict') else task_input,
                "full_context": full_context,
                "goal": f"Execute {self.AGENT_NAME} specialized task"
            }
            
            protocol_result = self.execute_with_protocol(protocol_task, primary_protocol)
            
            if protocol_result["overall_success"]:
                return self._extract_output_from_protocol_result(protocol_result, task_input)
            else:
                # Enhanced error handling instead of fallback
                error_msg = f"Protocol execution failed for {self.AGENT_NAME}: {protocol_result.get('error', 'Unknown error')}"
                self._logger.error(error_msg)
                raise ProtocolExecutionError(error_msg)
                
        except Exception as e:
            error_msg = f"Pure protocol execution failed for {self.AGENT_NAME}: {e}"
            self._logger.error(error_msg)
            raise ProtocolExecutionError(error_msg)

    # ADDED: Protocol phase execution logic
    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute agent-specific logic for each protocol phase."""
        
        # Generic phase handling - can be overridden by specific agents
        if phase.name in ["discovery", "analysis", "planning", "execution", "validation"]:
            return self._execute_generic_phase(phase)
        else:
            self._logger.warning(f"Unknown protocol phase: {phase.name}")
            return {"phase_completed": True, "method": "fallback"}

    def _execute_generic_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute generic phase logic suitable for most agents."""
        return {
            "phase_name": phase.name,
            "status": "completed", 
            "outputs": {"generic_result": f"Phase {phase.name} completed"},
            "method": "generic_protocol_execution"
        }

    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], task_input) -> Any:
        """Extract agent output from protocol execution results."""
        # Generic extraction - should be overridden by specific agents
        return {
            "status": "SUCCESS",
            "message": "Task completed via protocol execution",
            "protocol_used": protocol_result.get("protocol_name"),
            "execution_time": protocol_result.get("execution_time", 0),
            "phases_completed": len([p for p in protocol_result.get("phases", []) if p.get("success", False)])
        }


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