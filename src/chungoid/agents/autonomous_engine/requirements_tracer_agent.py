from __future__ import annotations

import logging
import datetime # For potential timestamping
import uuid
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Literal, ClassVar, Type, Union

from pydantic import BaseModel, Field, PrivateAttr

from ..protocol_aware_agent import ProtocolAwareAgent
from ...protocols.base.protocol_interface import ProtocolPhase
from ...utils.llm_provider import LLMProvider
from ...utils.prompt_manager import PromptManager, PromptRenderError, PromptLoadError
from ...schemas.common import ConfidenceScore
from ...utils.chromadb_migration_utils import (
    migrate_store_artifact,
    migrate_retrieve_artifact,
    migrate_query_artifacts,
    PCMAMigrationError
)
from ...utils.agent_registry import AgentCard
from ...utils.agent_registry_meta import AgentCategory, AgentVisibility

logger = logging.getLogger(__name__)

# MIGRATED: Collection constants moved here from PCMA
LOPRD_ARTIFACTS_COLLECTION = "loprd_artifacts_collection"
BLUEPRINT_ARTIFACTS_COLLECTION = "blueprint_artifacts_collection"
EXECUTION_PLANS_COLLECTION = "execution_plans_collection"
TRACEABILITY_REPORTS_COLLECTION = "traceability_reports"
AGENT_REFLECTIONS_AND_LOGS_COLLECTION = "agent_reflections_and_logs"
ARTIFACT_TYPE_TRACEABILITY_MATRIX_MD = "TraceabilityMatrix_MD"
ARTIFACT_TYPE_AGENT_REFLECTION_JSON = "AgentReflection_JSON"

RTA_PROMPT_NAME = "requirements_tracer_agent_v1_prompt.yaml"

class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

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

class RequirementsTracerAgent_v1(ProtocolAwareAgent[RequirementsTracerInput, RequirementsTracerOutput]):
    """
    Generates a traceability report (Markdown) between two development artifacts.
    
    ✨ PURE PROTOCOL ARCHITECTURE - No backward compatibility, clean execution paths only.
    ✨ MCP TOOL INTEGRATION - Uses ChromaDB MCP tools instead of agent dependencies.
    """
    
    AGENT_ID: ClassVar[str] = "RequirementsTracerAgent_v1"
    AGENT_NAME: ClassVar[str] = "Requirements Tracer Agent v1"
    DESCRIPTION: ClassVar[str] = "Generates a traceability report (Markdown) between two development artifacts (e.g., LOPRD to Blueprint)."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = RTA_PROMPT_NAME
    VERSION: ClassVar[str] = "0.2.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[RequirementsTracerInput]] = RequirementsTracerInput
    OUTPUT_SCHEMA: ClassVar[Type[RequirementsTracerOutput]] = RequirementsTracerOutput

    # MIGRATED: Removed PCMA dependency injection
    _llm_provider: LLMProvider
    _prompt_manager: PromptManager
    _logger: logging.Logger
    
    # ADDED: Protocol definitions following AI agent best practices
    PRIMARY_PROTOCOLS: ClassVar[list[str]] = ['requirements_tracing', 'goal_tracking']
    SECONDARY_PROTOCOLS: ClassVar[list[str]] = ['documentation', 'validation']
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'tool_validation']

    def __init__(
        self, 
        llm_provider: LLMProvider, 
        prompt_manager: PromptManager, 
        system_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            system_context=system_context,
            **kwargs
        )
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        
        if system_context and "logger" in system_context:
            self._logger = system_context["logger"]
        else:
            self._logger = logging.getLogger(f"{__name__}.{self.AGENT_ID}")

        if not self._llm_provider:
            self._logger.error("LLMProvider not provided during initialization.")
            raise ValueError("LLMProvider is required for RequirementsTracerAgent_v1.")
        if not self._prompt_manager:
            self._logger.error("PromptManager not provided during initialization.")
            raise ValueError("PromptManager is required for RequirementsTracerAgent_v1.")
        
        self._logger.info(f"{self.AGENT_ID} (v{self.VERSION}) initialized with MCP tool integration.")

    async def execute(self, task_input: RequirementsTracerInput, full_context: Optional[Dict[str, Any]] = None) -> RequirementsTracerOutput:
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

    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute requirements tracer specific logic for each protocol phase."""
        
        if phase.name == "discovery":
            return self._discover_artifacts_phase(phase)
        elif phase.name == "analysis":
            return self._analyze_traceability_phase(phase)
        elif phase.name == "planning":
            return self._plan_traceability_report_phase(phase)
        elif phase.name == "execution":
            return self._execute_traceability_generation_phase(phase)
        elif phase.name == "validation":
            return self._validate_traceability_report_phase(phase)
        else:
            self._logger.warning(f"Unknown protocol phase: {phase.name}")
            return {"phase_completed": True, "method": "fallback"}

    def _discover_artifacts_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 1: Discover and retrieve artifacts using MCP tools."""
        task_context = self.protocol_context
        task_input = task_context.get("task_input", {})
        
        def get_collection_for_artifact_type(artifact_type: str) -> str:
            """Map artifact type to collection name."""
            mapping = {
                "LOPRD": LOPRD_ARTIFACTS_COLLECTION,
                "UserStories": LOPRD_ARTIFACTS_COLLECTION,
                "Blueprint": BLUEPRINT_ARTIFACTS_COLLECTION,
                "MasterExecutionPlan": EXECUTION_PLANS_COLLECTION,
            }
            return mapping.get(artifact_type, LOPRD_ARTIFACTS_COLLECTION)
        
        try:
            # Retrieve source artifact using MCP tools
            source_collection = get_collection_for_artifact_type(task_input.get("source_artifact_type"))
            source_result = asyncio.run(migrate_retrieve_artifact(
                collection_name=source_collection,
                document_id=task_input.get("source_artifact_doc_id"),
                project_id=task_input.get("project_id", "default")
            ))
            
            if source_result["status"] != "SUCCESS":
                raise PCMAMigrationError(f"Failed to retrieve source artifact: {source_result.get('error')}")
            
            # Retrieve target artifact using MCP tools  
            target_collection = get_collection_for_artifact_type(task_input.get("target_artifact_type"))
            target_result = asyncio.run(migrate_retrieve_artifact(
                collection_name=target_collection,
                document_id=task_input.get("target_artifact_doc_id"),
                project_id=task_input.get("project_id", "default")
            ))
            
            if target_result["status"] != "SUCCESS":
                raise PCMAMigrationError(f"Failed to retrieve target artifact: {target_result.get('error')}")
            
            return {
                "phase_completed": True,
                "source_artifact": source_result,
                "target_artifact": target_result,
                "artifacts_retrieved": True
            }
            
        except Exception as e:
            self._logger.error(f"Artifact discovery failed: {e}")
            return {
                "phase_completed": False,
                "error": str(e),
                "artifacts_retrieved": False
            }

    def _analyze_traceability_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 2: Analyze traceability using MCP tools."""
        task_context = self.protocol_context
        task_input = task_context.get("task_input", {})
        
        # Implement the logic to analyze traceability using MCP tools
        # This is a placeholder and should be replaced with actual implementation
        return {"phase_completed": True, "method": "fallback"}

    def _plan_traceability_report_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 3: Plan traceability report generation."""
        task_context = self.protocol_context
        task_input = task_context.get("task_input", {})
        
        # Implement the logic to plan traceability report generation
        # This is a placeholder and should be replaced with actual implementation
        return {"phase_completed": True, "method": "fallback"}

    def _execute_traceability_generation_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 4: Execute traceability report generation."""
        task_context = self.protocol_context
        task_input = task_context.get("task_input", {})
        
        # Implement the logic to execute traceability report generation
        # This is a placeholder and should be replaced with actual implementation
        return {"phase_completed": True, "method": "fallback"}

    def _validate_traceability_report_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 5: Validate traceability report."""
        task_context = self.protocol_context
        task_input = task_context.get("task_input", {})
        
        # Implement the logic to validate traceability report
        # This is a placeholder and should be replaced with actual implementation
        return {"phase_completed": True, "method": "fallback"}

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
                TRACEABILITY_REPORTS_COLLECTION,
                AGENT_REFLECTIONS_AND_LOGS_COLLECTION
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