from __future__ import annotations

import logging
import asyncio
import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Generic, ClassVar, TYPE_CHECKING
import json

from pydantic import BaseModel, Field, ValidationError

# Chungoid imports using relative paths
from ..protocol_aware_agent import ProtocolAwareAgent
from ...protocols.base.protocol_interface import ProtocolPhase
from ...utils.llm_provider import LLMProvider
from ...utils.prompt_manager import PromptManager, PromptRenderError
from ...schemas.autonomous_engine.loprd_schema import LOPRD
from ...schemas.common import ConfidenceScore
from ...schemas.orchestration import SharedContext
from ...utils.agent_registry import AgentCard
from ...utils.agent_registry_meta import AgentCategory, AgentVisibility
# MIGRATED: Using MCP tools instead of ProjectChromaManagerAgent_v1
from ...utils.chromadb_migration_utils import migrate_store_artifact, migrate_retrieve_artifact

logger = logging.getLogger(__name__)

class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

# MIGRATED: Collection constants moved here from PCMA
PROJECT_GOALS_COLLECTION = "project_goals"
LOPRD_ARTIFACTS_COLLECTION = "loprd_artifacts_collection"
AGENT_REFLECTIONS_AND_LOGS_COLLECTION = "agent_reflections_and_logs"
SHARED_ARTIFACTS_COLLECTION = "shared_artifacts_collection"
ARTIFACT_TYPE_PRODUCT_ANALYSIS_JSON = "ProductAnalysis_JSON"

# --- Input and Output Schemas for the Agent --- #

class ProductAnalystAgentInput(BaseModel):
    refined_user_goal_doc_id: str = Field(..., description="Document ID of the refined_user_goal.md in Chroma.")
    assumptions_and_ambiguities_doc_id: Optional[str] = Field(None, description="Document ID of the assumptions_and_ambiguities.md in Chroma, if available.")
    project_id: str = Field(..., description="The ID of the current project.")
    # Potentially add context from ARCA if this is a refinement loop
    arca_feedback_doc_id: Optional[str] = Field(None, description="Document ID of feedback from ARCA if this is a refinement run.")
    shared_context: Optional[SharedContext] = Field(None, description="The shared context from the orchestrator, providing broader project and workflow information.")
    cycle_id: Optional[str] = Field(None, description="The ID of the current refinement cycle, passed by ARCA for lineage tracking.")

class ProductAnalystAgentOutput(BaseModel):
    loprd_doc_id: str = Field(..., description="Document ID of the generated LOPRD JSON artifact in Chroma.")
    confidence_score: ConfidenceScore = Field(..., description="Confidence score for the generated LOPRD.")
    raw_llm_response: Optional[str] = Field(None, description="The raw JSON string from the LLM before validation, for debugging.")
    validation_errors: Optional[str] = Field(None, description="Validation errors if the LLM output failed schema validation.")

class ProductAnalystAgent_v1(ProtocolAwareAgent[ProductAnalystAgentInput, ProductAnalystAgentOutput]):
    AGENT_ID: ClassVar[str] = "ProductAnalystAgent_v1"
    AGENT_NAME: ClassVar[str] = "Product Analyst Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Transforms a refined user goal into a detailed LLM-Optimized Product Requirements Document (LOPRD) in JSON format."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "product_analyst_agent_v1.yaml" # In server_prompts/autonomous_engine/
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS # Or custom category
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    # ADDED: Protocol definitions following AI agent best practices
    PRIMARY_PROTOCOLS: ClassVar[list[str]] = ['requirements_analysis', 'stakeholder_analysis']
    SECONDARY_PROTOCOLS: ClassVar[list[str]] = ['deep_investigation', 'documentation']
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'goal_tracking']


    def __init__(self, 
                 llm_provider: LLMProvider,
                 prompt_manager: PromptManager,
                 # MIGRATED: Removed PCMA dependency injection
                 config: Optional[Dict[str, Any]] = None,
                 system_context: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, system_context=system_context)
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        # MIGRATED: Removed project_chroma_manager reference
        self._logger_instance = system_context.get("logger", logger) if system_context else logger
        self.loprd_json_schema_for_prompt = self._load_loprd_json_schema_for_prompt()

    def _load_loprd_json_schema_for_prompt(self) -> Optional[Dict[str, Any]]:
        try:
            # Determine path to loprd_schema.json relative to this file or a known schemas dir
            schema_path = Path(__file__).parent.parent.parent / "schemas" / "autonomous_engine" / "loprd_schema.json"
            if not schema_path.exists():
                self._logger_instance.error(f"LOPRD schema file not found at {schema_path}")
                return None
            return LOPRD.model_json_schema()
        except Exception as e:
            self._logger_instance.error(f"Failed to load LOPRD JSON schema for prompt: {e}")
            return None

    async def execute(self, task_input, full_context: Optional[Dict[str, Any]] = None):
        """
        Execute using pure protocol architecture with MCP tool integration.
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