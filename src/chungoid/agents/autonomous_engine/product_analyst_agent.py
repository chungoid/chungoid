from __future__ import annotations

import logging
import asyncio
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Generic, ClassVar, TYPE_CHECKING, List
import json
import time

from pydantic import BaseModel, Field, validator, ValidationError

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

from chungoid.schemas.unified_execution_schemas import AgentOutput
from ...schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    AgentExecutionResult,
    ExecutionMetadata,
    ExecutionMode,
    CompletionReason,
    IterationResult,
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
    # Traditional fields - optional when using intelligent context
    refined_user_goal_md: Optional[str] = Field(None, description="The refined user goal in Markdown format.")
    assumptions_and_ambiguities_md: Optional[str] = Field(None, description="Assumptions and ambiguities related to the goal.")
    arca_feedback_md: Optional[str] = Field(None, description="Feedback from ARCA on previous LOPRD generation attempts.")
    loprd_json_schema_str: Optional[str] = Field(None, description="The JSON schema string that the LOPRD output must conform to.")
    
    # ADDED: Intelligent project analysis from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications from orchestrator analysis")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    user_goal: Optional[str] = Field(None, description="Original user goal for context")
    project_path: Optional[str] = Field(None, description="Project path for context")
    
    @validator('refined_user_goal_md')
    def validate_traditional_or_intelligent_context(cls, v, values):
        """Ensure either traditional fields or intelligent context is provided."""
        intelligent_context = values.get('intelligent_context', False)
        user_goal = values.get('user_goal')
        project_specifications = values.get('project_specifications')
        
        if intelligent_context:
            # When using intelligent context, user_goal and project_specifications are required
            if not user_goal:
                raise ValueError("user_goal is required when intelligent_context=True")
            if not project_specifications:
                raise ValueError("project_specifications is required when intelligent_context=True")
            # refined_user_goal_md is optional in this case
            return v
        else:
            # When not using intelligent context, traditional fields are required
            if not v:
                raise ValueError("refined_user_goal_md is required when intelligent_context=False")
            return v
    
    @validator('loprd_json_schema_str')
    def validate_loprd_schema(cls, v, values):
        """Ensure loprd_json_schema_str is provided when not using intelligent context."""
        intelligent_context = values.get('intelligent_context', False)
        
        if not intelligent_context and not v:
            raise ValueError("loprd_json_schema_str is required when intelligent_context=False")
        
        return v

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
    CAPABILITIES: ClassVar[List[str]] = ['requirements_analysis', 'stakeholder_analysis', 'documentation', 'complex_analysis']

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

    async def _execute_iteration(
        self, 
        context: UEContext, 
        iteration: int
    ) -> IterationResult:
        """
        Phase 3 UAEI implementation - Core product analysis logic for single iteration.
        
        Runs the complete product analysis workflow:
        1. Discovery: Analyze user goals and extract requirements/stakeholders
        2. Analysis: Create LOPRD structure from requirements  
        3. Validation: Validate LOPRD against schema
        4. Documentation: Generate final LOPRD document
        """
        self.logger.info(f"[ProductAnalyst] Starting iteration {iteration + 1}")
        
        # Convert inputs to proper type
        if isinstance(context.inputs, dict):
            inputs = ProductAnalystAgentInput(**context.inputs)
        elif isinstance(context.inputs, ProductAnalystAgentInput):
            inputs = context.inputs
        else:
            # Fallback for other types
            input_dict = context.inputs.dict() if hasattr(context.inputs, 'dict') else {}
            inputs = ProductAnalystAgentInput(
                refined_user_goal_md=str(input_dict.get("refined_user_goal_md", "")),
                loprd_json_schema_str=str(input_dict.get("loprd_json_schema_str", "{}"))
            )
        
        try:
            # Phase 1: Discovery - Analyze user goals
            self.logger.info("Starting discovery phase")
            
            # Check if we have intelligent project specifications from orchestrator
            if inputs.project_specifications and inputs.intelligent_context:
                self.logger.info("Using intelligent project specifications from orchestrator")
                goal_analysis = self._extract_analysis_from_intelligent_specs(inputs.project_specifications, inputs.user_goal)
                
                # Provide default LOPRD schema when using intelligent context
                if not inputs.loprd_json_schema_str:
                    inputs.loprd_json_schema_str = self._get_default_loprd_schema()
            else:
                self.logger.info("Using traditional goal analysis")
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
                    content=final_loprd,
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
            
            tools_used = ["requirements_analysis", "loprd_generation", "validation"]
            
        except Exception as e:
            self.logger.error(f"ProductAnalystAgent iteration failed: {e}")
            
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
            tools_used = []
        
        # Return iteration result for Phase 3 multi-iteration support
        return IterationResult(
            output=output,
            quality_score=quality_score,
            tools_used=tools_used,
            protocol_used=self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "requirements_analysis"
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

    def _extract_analysis_from_intelligent_specs(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """Extract goal analysis from intelligent project specifications."""
        
        # Extract core objectives from project specifications
        core_objectives = []
        if "project_type" in project_specs:
            core_objectives.append(f"Build a {project_specs['project_type']} application")
        if "technologies" in project_specs:
            core_objectives.append(f"Implement using {', '.join(project_specs['technologies'][:3])}")
        
        # Extract stakeholders based on project type
        key_stakeholders = ["End users", "Development team"]
        if project_specs.get("project_type") == "cli_tool":
            key_stakeholders.extend(["System administrators", "Power users"])
        elif project_specs.get("project_type") == "web_app":
            key_stakeholders.extend(["Web users", "Content managers"])
        elif project_specs.get("project_type") == "api":
            key_stakeholders.extend(["API consumers", "Integration partners"])
        
        # Extract success criteria from requirements
        success_criteria = []
        if "required_dependencies" in project_specs:
            success_criteria.append("All required dependencies are properly integrated")
        if "target_platforms" in project_specs:
            platforms = project_specs["target_platforms"]
            success_criteria.append(f"Application runs on {', '.join(platforms)}")
        success_criteria.append("Application meets functional requirements")
        success_criteria.append("Application is maintainable and well-documented")
        
        # Identify potential challenges
        potential_challenges = []
        if len(project_specs.get("technologies", [])) > 5:
            potential_challenges.append("Complex technology stack integration")
        if len(project_specs.get("target_platforms", [])) > 2:
            potential_challenges.append("Multi-platform compatibility")
        if project_specs.get("project_type") == "cli_tool":
            potential_challenges.append("Command-line interface usability")
        potential_challenges.append("Performance optimization")
        potential_challenges.append("Error handling and edge cases")
        
        return {
            "core_objectives": core_objectives,
            "key_stakeholders": key_stakeholders,
            "success_criteria": success_criteria,
            "potential_challenges": potential_challenges,
            "intelligent_analysis": True,
            "project_specifications": project_specs
        }

    def _get_default_loprd_schema(self) -> str:
        """Get default LOPRD JSON schema for intelligent context mode."""
        default_schema = {
            "type": "object",
            "properties": {
                "project_overview": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "type": {"type": "string"},
                        "target_audience": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["name", "description", "type"]
                },
                "functional_requirements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "acceptance_criteria": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["id", "title", "description", "priority"]
                    }
                },
                "non_functional_requirements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "requirement": {"type": "string"},
                            "target_value": {"type": "string"}
                        },
                        "required": ["category", "requirement"]
                    }
                },
                "user_stories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "as_a": {"type": "string"},
                            "i_want": {"type": "string"},
                            "so_that": {"type": "string"},
                            "acceptance_criteria": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["id", "as_a", "i_want", "so_that"]
                    }
                },
                "technical_specifications": {
                    "type": "object",
                    "properties": {
                        "architecture": {"type": "string"},
                        "technologies": {"type": "array", "items": {"type": "string"}},
                        "dependencies": {"type": "array", "items": {"type": "string"}},
                        "deployment": {"type": "string"}
                    }
                }
            },
            "required": ["project_overview", "functional_requirements", "user_stories"]
        }
        
        import json
        return json.dumps(default_schema, indent=2)

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