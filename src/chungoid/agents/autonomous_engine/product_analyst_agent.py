from __future__ import annotations

import logging
import asyncio
import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Generic, ClassVar, TYPE_CHECKING, List
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

# Registry-first architecture import
from chungoid.registry import register_autonomous_engine_agent

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
    refined_user_goal_md: str = Field(..., description="The refined user goal in Markdown format.")
    assumptions_and_ambiguities_md: Optional[str] = Field(None, description="Assumptions and ambiguities related to the goal.")
    arca_feedback_md: Optional[str] = Field(None, description="Feedback from ARCA on previous LOPRD generation attempts.")
    loprd_json_schema_str: str = Field(..., description="The JSON schema string that the LOPRD output must conform to.")

class ProductAnalystAgentOutput(BaseModel):
    loprd_doc_id: str = Field(..., description="Document ID of the generated LOPRD JSON artifact in Chroma.")
    confidence_score: ConfidenceScore = Field(..., description="Confidence score for the generated LOPRD.")
    raw_llm_response: Optional[str] = Field(None, description="The raw JSON string from the LLM before validation, for debugging.")
    validation_errors: Optional[str] = Field(None, description="Validation errors if the LLM output failed schema validation.")

@register_autonomous_engine_agent(capabilities=["requirements_analysis", "stakeholder_analysis", "documentation"])
class ProductAnalystAgent_v1(ProtocolAwareAgent):
    AGENT_ID: ClassVar[str] = "ProductAnalystAgent_v1"
    AGENT_VERSION: ClassVar[str] = "0.1.0"
    AGENT_NAME: ClassVar[str] = "Product Analyst Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Transforms a refined user goal into a detailed LLM-Optimized Product Requirements Document (LOPRD) in JSON format."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "product_analyst_agent_v1.yaml"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["requirements_analysis", "stakeholder_analysis"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["agent_communication", "tool_validation"]
    CAPABILITIES: ClassVar[List[str]] = ['requirements_analysis', 'stakeholder_analysis', 'documentation']

    def __init__(self, 
                 llm_provider: LLMProvider,
                 prompt_manager: PromptManager,
                 system_context: Optional[Dict[str, Any]] = None,
                 agent_id: Optional[str] = None,
                 **kwargs):
        super().__init__(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            system_context=system_context,
            agent_id=agent_id or self.AGENT_ID,
            **kwargs
        )
        
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        
        self.logger.info(f"{self.AGENT_ID} (v{self.AGENT_VERSION}) initialized as autonomous protocol-aware agent.")

    async def _execute_phase_logic(self, phase: ProtocolPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute protocol phase logic for product analysis."""
        if phase.name == "discovery":
            # Discover and analyze user goals
            user_goal = context.get("refined_user_goal_md", "")
            analysis = await self._analyze_user_goal(user_goal)
            return {"goal_analysis": analysis, "phase": "discovery"}
        elif phase.name == "analysis":
            # Analyze requirements and create LOPRD structure
            goal_analysis = context.get("goal_analysis", {})
            loprd_structure = await self._create_loprd_structure(goal_analysis)
            return {"loprd_structure": loprd_structure, "phase": "analysis"}
        elif phase.name == "validation":
            # Validate LOPRD against schema
            loprd_structure = context.get("loprd_structure", {})
            validation_result = self._validate_loprd(loprd_structure)
            return {"validation": validation_result, "phase": "validation"}
        elif phase.name == "documentation":
            # Generate final LOPRD document
            loprd_structure = context.get("loprd_structure", {})
            validation = context.get("validation", {})
            final_loprd = self._generate_final_loprd(loprd_structure, validation)
            return {"final_loprd": final_loprd, "phase": "documentation"}
        else:
            # Default phase handling
            return {"phase": phase.name, "status": "completed"}

    async def _analyze_user_goal(self, user_goal: str) -> Dict[str, Any]:
        """Analyze user goal to extract key insights."""
        try:
            if self._llm_provider:
                prompt = f"""
                Analyze the following user goal for product requirements:
                
                Goal: {user_goal}
                
                Please identify:
                1. Core objectives
                2. Key stakeholders
                3. Success criteria
                4. Potential challenges
                
                Format as JSON.
                """
                
                response = await self._llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You are a product analyst. Provide structured analysis.",
                    response_format="json_object"
                )
                
                if response and response.content:
                    try:
                        return json.loads(response.content)
                    except json.JSONDecodeError:
                        pass
            
            # Fallback analysis
            return self._generate_fallback_analysis(user_goal)
            
        except Exception as e:
            self.logger.error(f"Error in user goal analysis: {e}")
            return self._generate_fallback_analysis(user_goal)

    async def _create_loprd_structure(self, goal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create LOPRD structure based on goal analysis."""
        return {
            "project_overview": goal_analysis.get("core_objectives", "Project overview"),
            "user_stories": self._generate_user_stories(goal_analysis),
            "functional_requirements": self._generate_functional_requirements(goal_analysis),
            "non_functional_requirements": self._generate_non_functional_requirements(goal_analysis),
            "acceptance_criteria": self._generate_acceptance_criteria(goal_analysis)
        }

    def _validate_loprd(self, loprd_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LOPRD structure."""
        validation = {
            "is_valid": True,
            "issues": [],
            "completeness_score": 0.8
        }
        
        required_sections = ["project_overview", "user_stories", "functional_requirements"]
        for section in required_sections:
            if section not in loprd_structure or not loprd_structure[section]:
                validation["is_valid"] = False
                validation["issues"].append(f"Missing {section}")
        
        return validation

    def _generate_final_loprd(self, loprd_structure: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final LOPRD document."""
        final_loprd = loprd_structure.copy()
        final_loprd["validation_status"] = validation
        final_loprd["generated_at"] = "2025-01-25T00:00:00Z"  # Placeholder
        return final_loprd

    def _generate_fallback_analysis(self, user_goal: str) -> Dict[str, Any]:
        """Generate fallback analysis when LLM fails."""
        return {
            "core_objectives": [f"Implement solution for: {user_goal}"],
            "key_stakeholders": ["End users", "Development team"],
            "success_criteria": ["Solution meets user needs", "System is reliable"],
            "potential_challenges": ["Technical complexity", "User adoption"]
        }

    def _generate_user_stories(self, goal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate user stories from goal analysis."""
        return [
            {
                "id": "US001",
                "title": "Basic functionality",
                "description": "As a user, I want basic functionality so that I can achieve my goals",
                "acceptance_criteria": ["Feature works as expected", "User interface is intuitive"]
            }
        ]

    def _generate_functional_requirements(self, goal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate functional requirements."""
        return [
            {
                "id": "FR001",
                "description": "System shall provide core functionality",
                "priority": "Must Have"
            }
        ]

    def _generate_non_functional_requirements(self, goal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate non-functional requirements."""
        return [
            {
                "id": "NFR001",
                "category": "Performance",
                "description": "System shall respond within 2 seconds"
            }
        ]

    def _generate_acceptance_criteria(self, goal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate acceptance criteria."""
        return [
            {
                "id": "AC001",
                "description": "All functional requirements are implemented",
                "testable": True
            }
        ]

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = ProductAnalystAgentInput.model_json_schema()
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
            version=ProductAnalystAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            produced_artifacts_schemas={
                "loprd.json (stored_in_chroma)": loprd_artifact_schema_for_docs
            },
            llm_direct_output_schema=llm_expected_output_schema_for_docs,
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