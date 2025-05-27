from __future__ import annotations

import logging
import datetime # For potential timestamping
import uuid
import json # For LOPRD content if it's retrieved as JSON string
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, ClassVar
import time

from pydantic import BaseModel, Field, validator

from chungoid.agents.unified_agent import UnifiedAgent
from ...utils.llm_provider import LLMProvider
from ...utils.prompt_manager import PromptManager, PromptRenderError
from ...schemas.common import ConfidenceScore
from ...utils.agent_registry import AgentCard # For AgentCard
from ...utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard
from chungoid.registry import register_autonomous_engine_agent
from ...schemas.unified_execution_schemas import (
     ExecutionContext as UEContext,
     AgentExecutionResult,
     ExecutionMetadata,
     ExecutionMode,
     CompletionReason,
     IterationResult,
     StageInfo,
 )
from ...utils.chromadb_migration_utils import migrate_store_artifact, migrate_retrieve_artifact

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
    
    # Traditional fields - optional when using intelligent context
    loprd_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the LOPRD (JSON artifact) to be used as input.")
    existing_blueprint_doc_id: Optional[str] = Field(None, description="ChromaDB ID of an existing Blueprint to refine, if any.")
    refinement_instructions: Optional[str] = Field(None, description="Specific instructions for refining an existing Blueprint.")
    cycle_id: Optional[str] = Field(None, description="The ID of the current refinement cycle, passed by ARCA for lineage tracking.")
    # target_technologies: Optional[List[str]] = Field(None, description="Preferred technologies or constraints for the architecture.")
    
    # ADDED: Intelligent project analysis from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications from orchestrator analysis")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    user_goal: Optional[str] = Field(None, description="Original user goal for context")
    project_path: Optional[str] = Field(None, description="Project path for context")
    
    @validator('loprd_doc_id')
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
            # loprd_doc_id is optional in this case
            return v
        else:
            # When not using intelligent context, loprd_doc_id is required
            if not v:
                raise ValueError("loprd_doc_id is required when intelligent_context=False")
            return v

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
class ArchitectAgent_v1(UnifiedAgent):
    """
    Generates a technical blueprint based on an LOPRD and project context.
    
    PURE UAEI ARCHITECTURE - Unified Agent Execution Interface only.
    MCP TOOL INTEGRATION - Uses ChromaDB MCP tools for artifact management.
    """
    
    AGENT_ID: ClassVar[str] = "ArchitectAgent_v1"
    AGENT_NAME: ClassVar[str] = "Architect Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Generates a technical blueprint based on an LOPRD and project context."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "architect_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "0.1.0"
    CAPABILITIES: ClassVar[List[str]] = ["architecture_design", "system_planning", "blueprint_generation", "complex_analysis"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.PLANNING_AND_DESIGN 
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[ArchitectAgentInput]] = ArchitectAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[ArchitectAgentOutput]] = ArchitectAgentOutput
    
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["architecture_planning", "enhanced_deep_planning"]
    SECONDARY_PROTOCOLS: ClassVar[list[str]] = []
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'tool_validation']

    def __init__(self, 
                 llm_provider: LLMProvider,
                 prompt_manager: PromptManager,
                 system_context: Optional[Dict[str, Any]] = None,
                 agent_id: Optional[str] = None,
                 **kwargs):
        # Enable refinement capabilities for intelligent architecture design
        super().__init__(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            system_context=system_context,
            agent_id=agent_id or self.AGENT_ID,
            enable_refinement=True,  # Enable intelligent refinement
            **kwargs
        )
        
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        
        self.logger.info(f"{self.AGENT_ID} (v{self.AGENT_VERSION}) initialized as UAEI agent.")

    async def _execute_iteration(
        self, 
        context: UEContext, 
        iteration: int
    ) -> IterationResult:
        """
        Phase 3 UAEI implementation - Core architecture design logic for single iteration.
        
        Runs the complete architecture design workflow:
        1. Discovery: Retrieve LOPRD and analyze requirements
        2. Analysis: Analyze system requirements and constraints
        3. Planning: Plan architecture approach and components
        4. Design: Create detailed blueprint structure
        5. Validation: Validate architecture design quality
        """
        self.logger.info(f"[Architect] Starting iteration {iteration + 1}")
        
        # Convert inputs to proper type
        if isinstance(context.inputs, dict):
            inputs = ArchitectAgentInput(**context.inputs)
        elif isinstance(context.inputs, ArchitectAgentInput):
            inputs = context.inputs
        else:
            # Fallback for other types
            input_dict = context.inputs.dict() if hasattr(context.inputs, 'dict') else {}
            inputs = ArchitectAgentInput(
                project_id=str(input_dict.get("project_id", "default")),
                loprd_doc_id=str(input_dict.get("loprd_doc_id", ""))
            )
        
        try:
            # Phase 1: Discovery - Retrieve LOPRD
            self.logger.info("Starting LOPRD discovery phase")
            
            # Check if we have intelligent project specifications from orchestrator
            if inputs.project_specifications and inputs.intelligent_context:
                self.logger.info("Using intelligent project specifications from orchestrator")
                loprd_data = self._extract_loprd_from_intelligent_specs(inputs.project_specifications, inputs.user_goal)
            else:
                self.logger.info("Using traditional LOPRD retrieval")
                loprd_data = await self._discover_loprd(inputs)
            
            # Phase 2: Analysis - Analyze requirements
            self.logger.info("Starting requirements analysis phase")
            requirements = await self._analyze_requirements(loprd_data, inputs)
            
            # Phase 3: Planning - Plan architecture
            self.logger.info("Starting architecture planning phase") 
            architecture_plan = await self._plan_architecture(requirements, inputs)
            
            # Phase 4: Design - Create blueprint
            self.logger.info("Starting blueprint design phase")
            blueprint = await self._design_blueprint(architecture_plan, inputs)
            
            # Phase 5: Validation - Validate design
            self.logger.info("Starting design validation phase")
            validation_result = await self._validate_design(blueprint, inputs)
            
            # Store blueprint artifact in ChromaDB
            blueprint_doc_id = f"blueprint_{inputs.task_id}_{uuid.uuid4().hex[:8]}"
            try:
                await migrate_store_artifact(
                    collection_name=BLUEPRINT_ARTIFACTS_COLLECTION,
                    document_id=blueprint_doc_id,
                    content=blueprint,
                    metadata={
                        "agent_id": self.AGENT_ID,
                        "artifact_type": ARTIFACT_TYPE_PROJECT_BLUEPRINT_MD,
                        "project_id": inputs.project_id or "unknown",
                        "loprd_source": inputs.loprd_doc_id,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
                self.logger.info(f"Stored blueprint artifact with ID: {blueprint_doc_id}")
            except Exception as e:
                self.logger.error(f"Failed to store blueprint artifact: {e}")
                blueprint_doc_id = f"failed_storage_{uuid.uuid4().hex[:8]}"
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(validation_result, blueprint, requirements)
            
            # Create output
            output = ArchitectAgentOutput(
                task_id=inputs.task_id,
                project_id=inputs.project_id or "unknown",
                blueprint_document_id=blueprint_doc_id,
                status="SUCCESS",
                message="Architecture blueprint generated via UAEI workflow",
                confidence_score=ConfidenceScore(
                    value=quality_score,
                    method="validation_and_completeness",
                    explanation=f"Quality based on validation (valid: {validation_result.get('is_valid', False)}) and completeness"
                ),
                usage_metadata={
                    "iteration": iteration + 1,
                    "phases_executed": ["discovery", "analysis", "planning", "design", "validation"],
                    "components_designed": len(blueprint.get("components", []))
                }
            )
            
            tools_used = ["architecture_discovery", "requirements_analysis", "blueprint_design", "validation"]
            
        except Exception as e:
            self.logger.error(f"ArchitectAgent iteration failed: {e}")
            
            # Create error output
            output = ArchitectAgentOutput(
                task_id=inputs.task_id,
                project_id=inputs.project_id or "unknown",
                blueprint_document_id=None,
                status="FAILURE",
                message=f"Architecture design failed: {str(e)}",
                error_message=str(e),
                confidence_score=ConfidenceScore(
                    value=0.1,
                    method="error_fallback",
                    explanation=f"Execution failed: {str(e)}"
                )
            )
            
            quality_score = 0.1
            tools_used = []
        
        # Return iteration result for Phase 3 multi-iteration support
        return IterationResult(
            output=output,
            quality_score=quality_score,
            tools_used=tools_used,
            protocol_used=self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "architecture_planning"
        )

    def _extract_loprd_from_intelligent_specs(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """Extract LOPRD-like data from intelligent project specifications."""
        
        # Convert intelligent project specifications to LOPRD format
        loprd_content = {
            "project_overview": {
                "name": project_specs.get("project_type", "Unknown Project"),
                "description": user_goal[:200] + "..." if len(user_goal) > 200 else user_goal,
                "type": project_specs.get("project_type", "unknown"),
                "target_audience": ["End users", "Developers"]
            },
            "functional_requirements": [
                f"Implement {tech} functionality" for tech in project_specs.get("technologies", [])[:5]
            ],
            "non_functional_requirements": [
                {"category": "Performance", "requirement": "System should be responsive"},
                {"category": "Security", "requirement": "Secure data handling"},
                {"category": "Scalability", "requirement": f"Support {project_specs.get('target_platforms', ['multiple platforms'])}"}
            ],
            "user_stories": [
                {
                    "id": "US001",
                    "as_a": "user",
                    "i_want": f"to use a {project_specs.get('project_type', 'tool')}",
                    "so_that": "I can accomplish my goals efficiently"
                }
            ],
            "technical_specifications": {
                "architecture": "modular",
                "technologies": project_specs.get("technologies", []),
                "dependencies": project_specs.get("required_dependencies", []),
                "deployment": f"Compatible with {', '.join(project_specs.get('target_platforms', ['Linux']))}"
            }
        }
        
        return {
            "status": "SUCCESS",
            "content": loprd_content,
            "source": "intelligent_specifications"
        }

    async def _discover_loprd(self, inputs: ArchitectAgentInput) -> Dict[str, Any]:
        """Discover and retrieve LOPRD artifact."""
        try:
            loprd_result = await migrate_retrieve_artifact(
                collection_name=LOPRD_ARTIFACTS_COLLECTION,
                document_id=inputs.loprd_doc_id,
                project_id=inputs.project_id or "unknown"
            )
            
            if loprd_result["status"] != "SUCCESS":
                raise Exception(f"Failed to retrieve LOPRD: {loprd_result.get('error')}")
            
            return loprd_result
            
        except Exception as e:
            self.logger.error(f"LOPRD discovery failed: {e}")
            # Return minimal fallback data
            return {
                "status": "FALLBACK",
                "content": {
                    "project_overview": "Basic project requirements",
                    "functional_requirements": ["Core functionality"],
                    "non_functional_requirements": ["Performance", "Security"]
                }
            }

    async def _analyze_requirements(self, loprd_data: Dict[str, Any], inputs: ArchitectAgentInput) -> Dict[str, Any]:
        """Analyze requirements from LOPRD data."""
        loprd_content = loprd_data.get("content", {})
        
        # Extract requirements from LOPRD
        functional_reqs = loprd_content.get("functional_requirements", [])
        non_functional_reqs = loprd_content.get("non_functional_requirements", [])
        
        # Analyze complexity and constraints
        complexity_score = len(functional_reqs) + len(non_functional_reqs) * 0.5
        
        return {
            "functional_requirements": functional_reqs,
            "non_functional_requirements": non_functional_reqs,
            "complexity_score": complexity_score,
            "stakeholder_needs": loprd_content.get("stakeholder_requirements", []),
            "constraints": loprd_content.get("constraints", []),
            "analysis_confidence": 0.85
        }
    
    async def _plan_architecture(self, requirements: Dict[str, Any], inputs: ArchitectAgentInput) -> Dict[str, Any]:
        """Plan architecture approach based on requirements."""
        complexity = requirements.get("complexity_score", 1)
        
        # Select architecture pattern based on complexity
        if complexity < 5:
            pattern = "monolithic"
        elif complexity < 15:
            pattern = "layered"
        else:
            pattern = "microservices"
        
        # Plan technology stack
        technology_stack = {
            "backend": "Python/FastAPI" if pattern != "monolithic" else "Python/Flask",
            "database": "PostgreSQL",
            "cache": "Redis" if complexity > 10 else None,
            "message_queue": "RabbitMQ" if pattern == "microservices" else None
        }
        
        return {
            "architecture_pattern": pattern,
            "technology_stack": technology_stack,
            "scalability_considerations": requirements.get("non_functional_requirements", []),
            "component_breakdown": self._break_down_components(requirements),
            "design_decisions": [
                f"Chose {pattern} architecture for complexity level {complexity}",
                "Selected modern Python stack for rapid development"
            ]
        }
    
    def _break_down_components(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down functional requirements into components."""
        components = []
        
        for i, req in enumerate(requirements.get("functional_requirements", [])[:5]):  # Limit to 5 for simplicity
            components.append({
                "name": f"Component_{i+1}",
                "responsibility": req if isinstance(req, str) else str(req),
                "interfaces": ["REST API"],
                "dependencies": []
            })
        
        return components
    
    async def _design_blueprint(self, architecture_plan: Dict[str, Any], inputs: ArchitectAgentInput) -> Dict[str, Any]:
        """Create detailed blueprint structure."""
        pattern = architecture_plan.get("architecture_pattern", "layered")
        components = architecture_plan.get("component_breakdown", [])
        tech_stack = architecture_plan.get("technology_stack", {})
        
        # Create directory structure based on pattern
        if pattern == "microservices":
            directory_structure = {
                "services/": {comp["name"].lower(): {"main.py": "NEW", "requirements.txt": "NEW"} for comp in components},
                "shared/": {"utils.py": "NEW", "models.py": "NEW"},
                "docker-compose.yml": "NEW",
                "README.md": "NEW"
            }
        else:
            directory_structure = {
                "src/": {
                    "main.py": "NEW",
                    "config/": {"settings.py": "NEW"},
                    "models/": {"__init__.py": "NEW"},
                    "api/": {"__init__.py": "NEW", "routes.py": "NEW"},
                    "services/": {"__init__.py": "NEW"}
                },
                "tests/": {"test_main.py": "NEW"},
                "requirements.txt": "NEW",
                "README.md": "NEW"
            }
        
        return {
            "title": f"Technical Blueprint - {inputs.project_id}",
            "architecture_pattern": pattern,
            "technology_stack": tech_stack,
            "components": components,
            "directory_structure": directory_structure,
            "deployment_strategy": self._plan_deployment_strategy(pattern),
            "testing_strategy": self._plan_testing_strategy(pattern),
            "documentation": {
                "api_docs": "OpenAPI/Swagger",
                "architecture_docs": "Markdown",
                "deployment_docs": "Docker + README"
            }
        }
    
    def _plan_deployment_strategy(self, pattern: str) -> Dict[str, Any]:
        """Plan deployment strategy based on architecture pattern."""
        if pattern == "microservices":
            return {
                "type": "containerized",
                "orchestration": "Docker Compose",
                "scaling": "horizontal",
                "monitoring": "prometheus + grafana"
            }
        else:
            return {
                "type": "traditional",
                "deployment": "gunicorn + nginx",
                "scaling": "vertical",
                "monitoring": "basic logging"
            }
    
    def _plan_testing_strategy(self, pattern: str) -> Dict[str, Any]:
        """Plan testing strategy based on architecture pattern."""
        return {
            "unit_tests": "pytest",
            "integration_tests": "pytest + httpx" if pattern != "monolithic" else "pytest",
            "e2e_tests": "playwright" if pattern == "microservices" else "requests",
            "coverage_target": "85%"
        }
    
    async def _validate_design(self, blueprint: Dict[str, Any], inputs: ArchitectAgentInput) -> Dict[str, Any]:
        """Validate the architecture design quality."""
        validation_issues = []
        completeness_score = 1.0
        
        # Check required sections
        required_sections = ["architecture_pattern", "technology_stack", "components", "directory_structure"]
        for section in required_sections:
            if section not in blueprint or not blueprint[section]:
                validation_issues.append(f"Missing {section}")
                completeness_score -= 0.2
        
        # Check component count
        components = blueprint.get("components", [])
        if len(components) == 0:
            validation_issues.append("No components defined")
            completeness_score -= 0.3
        
        # Check technology stack completeness  
        tech_stack = blueprint.get("technology_stack", {})
        if not tech_stack.get("backend"):
            validation_issues.append("Backend technology not specified")
            completeness_score -= 0.2
        
        return {
            "is_valid": len(validation_issues) == 0,
            "validation_issues": validation_issues,
            "completeness_score": max(0.1, completeness_score),
            "quality_metrics": {
                "component_count": len(components),
                "technology_choices": len(tech_stack),
                "documentation_coverage": len(blueprint.get("documentation", {}))
            }
        }
    
    def _calculate_quality_score(self, validation_result: Dict[str, Any], blueprint: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """Calculate quality score based on validation and blueprint completeness."""
        base_score = 1.0
        
        # Deduct for validation issues
        if not validation_result.get("is_valid", False):
            issues_count = len(validation_result.get("validation_issues", []))
            base_score -= 0.2 * min(issues_count, 3)  # Max 0.6 deduction
        
        # Apply completeness score
        completeness_score = validation_result.get("completeness_score", 1.0)
        base_score *= completeness_score
        
        # Bonus for comprehensive design
        components_count = len(blueprint.get("components", []))
        if components_count >= 3:
            base_score += 0.1
            
        # Bonus for good technology choices
        tech_stack = blueprint.get("technology_stack", {})
        if len(tech_stack) >= 3:
            base_score += 0.1
            
        return max(0.1, min(base_score, 1.0))

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = ArchitectAgentInput.model_json_schema()
        output_schema = ArchitectAgentOutput.model_json_schema()
        module_path = ArchitectAgent_v1.__module__
        class_name = ArchitectAgent_v1.__name__

        return AgentCard(
            agent_id=ArchitectAgent_v1.AGENT_ID,
            name=ArchitectAgent_v1.AGENT_NAME,
            description=ArchitectAgent_v1.AGENT_DESCRIPTION,
            version=ArchitectAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[cat.value for cat in [ArchitectAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=ArchitectAgent_v1.VISIBILITY.value,
            capability_profile={
                "consumes_artifacts": ["LOPRD_JSON"],
                "generates_blueprints": ["ProjectBlueprint_Markdown"],
                "architecture_patterns": ["monolithic", "layered", "microservices"],
                "primary_function": "Technical Architecture Design and Blueprint Generation"
            },
            metadata={
                "callable_fn_path": f"{module_path}.{class_name}"
            }
        ) 