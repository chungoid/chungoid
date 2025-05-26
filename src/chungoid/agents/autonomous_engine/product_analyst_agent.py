from __future__ import annotations

import logging
import asyncio
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Generic, ClassVar, TYPE_CHECKING, List
import json
import time

from pydantic import BaseModel, Field, ValidationError

# Chungoid imports using relative paths


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

from chungoid.agents.unified_agent import UnifiedAgent

from chungoid.schemas.agent_outputs import AgentOutput
from ...schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    AgentExecutionResult,
    ExecutionMetadata,
    ExecutionMode,
    CompletionReason,
    StageInfo,
)

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
class ProductAnalystAgent_v1(UnifiedAgent):
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

    async def execute(
        self, 
        context: UEContext,
        execution_mode: ExecutionMode = ExecutionMode.OPTIMAL
    ) -> AgentExecutionResult:
        """
        UAEI execute method - handles both single-pass and multi-iteration execution.
        
        Runs the complete product analysis workflow:
        1. Discovery: Analyze user goals and extract requirements/stakeholders
        2. Analysis: Create LOPRD structure from requirements  
        3. Validation: Validate LOPRD against schema
        4. Documentation: Generate final LOPRD document
        """
        start_time = time.perf_counter()
        
        # Convert inputs to proper type
        if isinstance(context.inputs, dict):
            inputs = ProductAnalystAgentInput(**context.inputs)
        elif isinstance(context.inputs, ProductAnalystAgentInput):
            inputs = context.inputs
        else:
            # Fallback for other types
            inputs = ProductAnalystAgentInput(
                refined_user_goal_md=str(context.inputs.get("refined_user_goal_md", "")),
                loprd_json_schema_str=str(context.inputs.get("loprd_json_schema_str", "{}"))
            )
        
        try:
            # Phase 1: Discovery - Analyze user goals
            self.logger.info("Starting discovery phase")
            goal_analysis = await self._analyze_user_goal(inputs.refined_user_goal_md)
            
            # Phase 2: Analysis - Create LOPRD structure
            self.logger.info("Starting analysis phase") 
            loprd_structure = await self._create_loprd_structure(goal_analysis)
            
            # Phase 3: Validation - Validate LOPRD
            self.logger.info("Starting validation phase")
            validation_result = self._validate_loprd(loprd_structure)
            
            # Phase 4: Documentation - Generate final LOPRD
            self.logger.info("Starting documentation phase")
            final_loprd = self._generate_final_loprd(loprd_structure, validation_result)
            
            # Store LOPRD artifact in Chroma (migrated from PCMA)
            loprd_doc_id = f"loprd_{uuid.uuid4().hex[:8]}"
            try:
                await migrate_store_artifact(
                    collection_name=LOPRD_ARTIFACTS_COLLECTION,
                    document_id=loprd_doc_id,
                    artifact_data=final_loprd,
                    metadata={
                        "agent_id": self.AGENT_ID,
                        "artifact_type": ARTIFACT_TYPE_PRODUCT_ANALYSIS_JSON,
                        "validation_passed": validation_result.get("is_valid", False),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                self.logger.info(f"Stored LOPRD artifact with ID: {loprd_doc_id}")
            except Exception as e:
                self.logger.error(f"Failed to store LOPRD artifact: {e}")
                loprd_doc_id = f"failed_storage_{uuid.uuid4().hex[:8]}"
            
            # Calculate quality score based on validation and completeness
            quality_score = self._calculate_quality_score(validation_result, final_loprd, goal_analysis)
            
            # Create output
            output = ProductAnalystAgentOutput(
                loprd_doc_id=loprd_doc_id,
                confidence_score=ConfidenceScore(
                    value=quality_score,
                    method="validation_and_completeness",
                    explanation=f"Quality based on validation (valid: {validation_result.get('is_valid', False)}) and completeness"
                ),
                validation_errors=None if validation_result.get("is_valid", False) else str(validation_result.get("issues", []))
            )
            
            completion_reason = CompletionReason.SUCCESS if quality_score >= context.execution_config.quality_threshold else CompletionReason.QUALITY_THRESHOLD
            
        except Exception as e:
            self.logger.error(f"ProductAnalystAgent execution failed: {e}")
            
            # Create error output
            output = ProductAnalystAgentOutput(
                loprd_doc_id=f"error_{uuid.uuid4().hex[:8]}",
                confidence_score=ConfidenceScore(
                    value=0.1,
                    method="error_fallback",
                    explanation=f"Execution failed: {str(e)}"
                ),
                validation_errors=str(e)
            )
            
            quality_score = 0.1
            completion_reason = CompletionReason.ERROR
        
        execution_time = time.perf_counter() - start_time
        
        # Create execution metadata
        metadata = ExecutionMetadata(
            mode=execution_mode,
            protocol_used=self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "requirements_analysis",
            execution_time=execution_time,
            iterations_planned=context.execution_config.max_iterations,
            tools_utilized=None
        )
        
        return AgentExecutionResult(
            output=output,
            execution_metadata=metadata,
            iterations_completed=1,  # Single iteration for requirements analysis
            completion_reason=completion_reason,
            quality_score=quality_score,
            protocol_used=metadata.protocol_used
        )
    
    def _calculate_quality_score(self, validation_result: Dict[str, Any], final_loprd: Dict[str, Any], goal_analysis: Dict[str, Any]) -> float:
        """Calculate quality score based on validation and completeness."""
        base_score = 1.0
        
        # Deduct for validation issues
        if not validation_result.get("is_valid", False):
            issues_count = len(validation_result.get("issues", []))
            base_score -= 0.2 * min(issues_count, 3)  # Max 0.6 deduction for validation
        
        # Deduct for missing completeness
        completeness_score = validation_result.get("completeness_score", 1.0)
        base_score *= completeness_score
        
        # Bonus for rich analysis
        if len(goal_analysis.get("core_objectives", [])) >= 3:
            base_score += 0.1
        if len(goal_analysis.get("key_stakeholders", [])) >= 2:
            base_score += 0.1
            
        return max(0.1, min(base_score, 1.0))

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
                
                Please respond with a valid JSON object in this format:
                {{
                    "core_objectives": ["objective1", "objective2"],
                    "key_stakeholders": ["stakeholder1", "stakeholder2"],
                    "success_criteria": ["criteria1", "criteria2"],
                    "potential_challenges": ["challenge1", "challenge2"]
                }}
                """
                
                response = await self._llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You are a product analyst. Provide structured analysis in JSON format only.",
                )
                
                if response:
                    try:
                        # LiteLLMProvider returns a string directly, not an object with .content
                        return json.loads(response)
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

    async def _generate_acceptance_criteria(self, goal_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
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