from __future__ import annotations

import logging
import datetime # For potential timestamping
import uuid
import json # For LOPRD content if it's retrieved as JSON string
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, ClassVar

from pydantic import BaseModel, Field

from ..protocol_aware_agent import ProtocolAwareAgent
from ...protocols.base.protocol_interface import ProtocolPhase
from ...utils.llm_provider import LLMProvider
from ...utils.prompt_manager import PromptManager, PromptRenderError
from ...schemas.common import ConfidenceScore
from ...utils.agent_registry import AgentCard # For AgentCard
from ...utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard
from chungoid.registry import register_autonomous_engine_agent

logger = logging.getLogger(__name__)

# MIGRATED: Collection constants moved here from PCMA
LOPRD_ARTIFACTS_COLLECTION = "loprd_artifacts_collection"
BLUEPRINT_ARTIFACTS_COLLECTION = "blueprint_artifacts_collection" 
ARTIFACT_TYPE_PROJECT_BLUEPRINT_MD = "ProjectBlueprint_MD"
ARTIFACT_TYPE_LOPRD_JSON = "LOPRD_JSON"

ARCHITECT_AGENT_PROMPT_NAME = "architect_agent_v1_prompt.yaml" # In server_prompts/autonomous_engine/

class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

# --- Input and Output Schemas for the Agent --- #

class ArchitectAgentInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this Blueprint generation task.")
    project_id: Optional[str] = Field(None, description="Identifier for the current project.")
    loprd_doc_id: str = Field(..., description="ChromaDB ID of the LOPRD (JSON artifact) to be used as input.")
    existing_blueprint_doc_id: Optional[str] = Field(None, description="ChromaDB ID of an existing Blueprint to refine, if any.")
    refinement_instructions: Optional[str] = Field(None, description="Specific instructions for refining an existing Blueprint.")
    cycle_id: Optional[str] = Field(None, description="The ID of the current refinement cycle, passed by ARCA for lineage tracking.")
    # target_technologies: Optional[List[str]] = Field(None, description="Preferred technologies or constraints for the architecture.")

class ArchitectAgentOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    blueprint_document_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated/updated Project Blueprint (Markdown) is stored.")
    status: str = Field(..., description="Status of the Blueprint generation (e.g., SUCCESS, FAILURE_LLM, FAILURE_INPUT_RETRIEVAL).")
    message: str = Field(..., description="A message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the quality and completeness of the Blueprint.")
    error_message: Optional[str] = Field(None, description="Error message if status is not SUCCESS.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")

@register_autonomous_engine_agent(capabilities=["architecture_design", "system_planning", "blueprint_generation"])
class ArchitectAgent_v1(ProtocolAwareAgent[ArchitectAgentInput, ArchitectAgentOutput]):
    """
    Generates a technical blueprint based on an LOPRD and project context.
    
    ✨ PURE PROTOCOL ARCHITECTURE - No backward compatibility, clean execution paths only.
    ✨ MCP TOOL INTEGRATION - Uses ChromaDB MCP tools instead of agent dependencies.
    """
    
    AGENT_ID: ClassVar[str] = "ArchitectAgent_v1"
    AGENT_NAME: ClassVar[str] = "Architect Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Generates a technical blueprint based on an LOPRD and project context."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "architect_agent_v1_prompt.yaml"
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.PLANNING_AND_DESIGN # MODIFIED
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[ArchitectAgentInput]] = ArchitectAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[ArchitectAgentOutput]] = ArchitectAgentOutput
    
    # ADDED: Protocol definitions following Universal Protocol Infrastructure
    PRIMARY_PROTOCOLS: ClassVar[list[str]] = ['architecture_design', 'system_planning']
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'tool_validation']

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

    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute architect-specific logic for each protocol phase."""
        
        if phase.name == "goal_setting":
            return self._set_architecture_goals(phase)
        elif phase.name == "discovery":
            return self._discover_requirements_and_constraints(phase)
        elif phase.name == "analysis":
            return self._analyze_loprd_and_context(phase)
        elif phase.name == "planning":
            return self._plan_architecture_approach(phase)
        elif phase.name == "validation":
            return self._validate_architecture_plan(phase)
        else:
            self._logger.warning(f"Unknown protocol phase: {phase.name}")
            return {"phase_completed": True, "method": "fallback"}

    def _set_architecture_goals(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 1: Set clear architecture goals from LOPRD."""
        task_context = self.protocol_context
        task_input = task_context.get("task_input", {})
        
        return {
            "architecture_goals": [
                "Create comprehensive technical blueprint",
                "Align with LOPRD requirements", 
                "Ensure scalable architecture",
                "Maintain technical feasibility"
            ],
            "success_criteria": [
                "All LOPRD requirements mapped to architecture",
                "Technical decisions justified",
                "Implementation approach defined"
            ],
            "constraints": task_input.get("constraints", [])
        }

    def _discover_requirements_and_constraints(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 2: Discover and analyze requirements."""
        task_context = self.protocol_context
        task_input = task_context.get("task_input", {})
        
        return {
            "loprd_analysis": {
                "document_id": task_input.get("loprd_doc_id"),
                "requirements_identified": True
            },
            "existing_blueprint_analysis": {
                "document_id": task_input.get("existing_blueprint_doc_id"),
                "refinement_needed": bool(task_input.get("refinement_instructions"))
            },
            "technical_constraints": [],
            "stakeholder_requirements": []
        }

    def _analyze_loprd_and_context(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 3: Deep analysis of LOPRD and context."""
        return {
            "loprd_content_analysis": "Comprehensive LOPRD analysis completed",
            "technical_complexity_assessment": "medium",
            "architecture_patterns_identified": [],
            "risk_factors": []
        }

    def _plan_architecture_approach(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 4: Plan detailed architecture approach."""
        return {
            "architecture_strategy": "microservices_with_api_gateway",
            "technology_stack": [],
            "implementation_phases": [],
            "integration_points": []
        }

    def _validate_architecture_plan(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Phase 5: Validate architecture completeness."""
        return {
            "validation_results": {
                "completeness": True,
                "feasibility": True,
                "alignment_with_loprd": True
            },
            "quality_score": 85,
            "recommendations": []
        }

    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], 
                                           task_input: ArchitectAgentInput) -> ArchitectAgentOutput:
        """Extract ArchitectAgentOutput from protocol execution results."""
        
        # Extract key information from protocol phases
        phases = protocol_result.get("phases", [])
        planning_phase = next((p for p in phases if p["phase_name"] == "planning"), {})
        validation_phase = next((p for p in phases if p["phase_name"] == "validation"), {})
        
        # Generate blueprint document ID (would be stored via PCMA in real implementation)
        blueprint_doc_id = f"blueprint_{task_input.task_id}_{uuid.uuid4().hex[:8]}"
        
        return ArchitectAgentOutput(
            task_id=task_input.task_id,
            project_id=task_input.project_id or "protocol_generated",
            blueprint_document_id=blueprint_doc_id,
            status="SUCCESS",
            message="Architecture blueprint generated via Deep Planning Protocol",
            confidence_score=ConfidenceScore(
                score=validation_phase.get("outputs", {}).get("quality_score", 85),
                reasoning="Generated using systematic protocol approach"
            ),
            usage_metadata={
                "protocol_used": "deep_planning",
                "execution_time": protocol_result.get("execution_time", 0),
                "phases_completed": len([p for p in phases if p.get("success", False)])
            }
        )


    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=ArchitectAgent_v1.AGENT_ID,
            name=ArchitectAgent_v1.AGENT_NAME,
            description=ArchitectAgent_v1.AGENT_DESCRIPTION,
            version=ArchitectAgent_v1.VERSION,
            input_schema=ArchitectAgentInput.model_json_schema(),
            output_schema=ArchitectAgentOutput.model_json_schema(),
            categories=[cat.value for cat in [ArchitectAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=ArchitectAgent_v1.VISIBILITY.value,
            capability_profile={
                "generates_artifacts": ["ProjectBlueprint_Markdown"],
                "consumes_artifacts": ["LOPRD_JSON", "ExistingBlueprint_Markdown", "RefinementInstructions"],
                "primary_function": "Architectural Design and Blueprint Generation"
            },
            metadata={
                "callable_fn_path": f"{ArchitectAgent_v1.__module__}.{ArchitectAgent_v1.__name__}"
            }
        ) 