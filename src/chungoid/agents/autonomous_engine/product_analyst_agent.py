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
        # Enable refinement capabilities for intelligent product analysis
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
                goal_analysis = await self._extract_analysis_from_intelligent_specs(inputs.project_specifications, inputs.user_goal)
                
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
                        # Extract JSON from markdown code blocks if present
                        json_content = self._extract_json_from_response(response)
                        parsed_result = json.loads(json_content)
                        
                        # Validate that we got a dictionary as expected
                        if not isinstance(parsed_result, dict):
                            self.logger.warning(f"Expected dict from LLM analysis, got {type(parsed_result)}. Using fallback.")
                            return self._generate_fallback_analysis(user_goal)
                        
                        return parsed_result
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

    async def _extract_analysis_from_intelligent_specs(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """
        Enhanced with universal MCP tool access and intelligent selection.
        
        ENHANCED: Uses ALL 53+ MCP tools with intelligent selection for comprehensive product analysis.
        """
        
        try:
            # First try enhanced discovery with universal MCP tools
            if hasattr(self, '_enhanced_discovery_with_universal_tools'):
                self.logger.info("[MCP] Using enhanced discovery with universal MCP tool access")
                enhanced_analysis = await self._enhanced_discovery_with_universal_tools(
                    {"project_specifications": project_specs, "user_goal": user_goal}, 
                    {"project_specs": project_specs, "user_goal": user_goal}
                )
                
                if enhanced_analysis.get("universal_tool_access"):
                    # Convert MCP analysis to product analysis format
                    return await self._convert_mcp_analysis_to_product_analysis(
                        enhanced_analysis.get("project_analysis", {}),
                        enhanced_analysis.get("content_analysis", {}), 
                        enhanced_analysis.get("intelligence_analysis", {}),
                        enhanced_analysis.get("historical_context", {}),
                        enhanced_analysis.get("environment_info", {}),
                        project_specs,
                        user_goal
                    )
            
            # Fallback to LLM-based analysis if MCP tools unavailable
            if self._llm_provider:
                # Use LLM to intelligently analyze the project specifications and user goal
                prompt = f"""
                You are a product analyst. Analyze the following project specifications and user goal to create a comprehensive product analysis.
                
                User Goal: {user_goal}
                
                Project Specifications:
                - Project Type: {project_specs.get('project_type', 'unknown')}
                - Primary Language: {project_specs.get('primary_language', 'unknown')}
                - Target Languages: {project_specs.get('target_languages', [])}
                - Target Platforms: {project_specs.get('target_platforms', [])}
                - Technologies: {project_specs.get('technologies', [])}
                - Required Dependencies: {project_specs.get('required_dependencies', [])}
                - Optional Dependencies: {project_specs.get('optional_dependencies', [])}
                
                Based on this information, provide a detailed analysis in JSON format:
                {{
                    "core_objectives": ["specific objective 1", "specific objective 2", "specific objective 3"],
                    "key_stakeholders": ["stakeholder 1", "stakeholder 2", "stakeholder 3"],
                    "success_criteria": ["measurable criteria 1", "measurable criteria 2", "measurable criteria 3"],
                    "potential_challenges": ["technical challenge 1", "business challenge 2", "implementation challenge 3"],
                    "functional_requirements": ["requirement 1", "requirement 2", "requirement 3"],
                    "non_functional_requirements": ["performance requirement", "security requirement", "usability requirement"],
                    "user_personas": ["primary user type", "secondary user type"],
                    "business_value": "clear statement of business value",
                    "technical_complexity": "low|medium|high",
                    "estimated_effort": "small|medium|large"
                }}
                
                Make the analysis specific to the project type and technologies mentioned. Be detailed and actionable.
                """
                
                response = await self._llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You are an expert product analyst. Provide comprehensive, specific analysis in valid JSON format only.",
                    max_tokens=1000,
                    temperature=0.3
                )
                
                if response:
                    try:
                        # Extract JSON from markdown code blocks if present
                        json_content = self._extract_json_from_response(response)
                        parsed_result = json.loads(json_content)
                        
                        # Validate that we got a dictionary as expected
                        if not isinstance(parsed_result, dict):
                            self.logger.warning(f"Expected dict from intelligent analysis, got {type(parsed_result)}. Using fallback.")
                            return self._generate_fallback_intelligent_analysis(project_specs, user_goal)
                        
                        # Add metadata about the intelligent analysis
                        parsed_result["intelligent_analysis"] = True
                        parsed_result["project_specifications"] = project_specs
                        parsed_result["analysis_method"] = "llm_intelligent_processing"
                        return parsed_result
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
            
            # Fallback to basic extraction if LLM fails
            self.logger.info("Using fallback analysis due to LLM unavailability")
            return self._generate_fallback_intelligent_analysis(project_specs, user_goal)
            
        except Exception as e:
            self.logger.error(f"Error in intelligent specs analysis: {e}")
            return self._generate_fallback_intelligent_analysis(project_specs, user_goal)

    async def _convert_mcp_analysis_to_product_analysis(
        self, 
        project_analysis: Dict[str, Any], 
        content_analysis: Dict[str, Any], 
        intelligence_analysis: Dict[str, Any],
        historical_context: Dict[str, Any],
        environment_info: Dict[str, Any],
        project_specs: Dict[str, Any],
        user_goal: str
    ) -> Dict[str, Any]:
        """Convert MCP tool analysis results to product analysis format."""
        
        try:
            # Extract core objectives from project analysis
            core_objectives = []
            if project_analysis.get("success") and project_analysis.get("result"):
                project_data = project_analysis["result"]
                if isinstance(project_data, dict):
                    # Extract from project structure
                    if "project_type" in project_data:
                        core_objectives.append(f"Build a {project_data['project_type']} application")
                    if "technologies" in project_data:
                        core_objectives.append(f"Implement using {', '.join(project_data['technologies'][:3])}")
            
            # Add objectives from user goal
            if user_goal:
                core_objectives.append(f"Achieve user goal: {user_goal}")
            
            # Extract stakeholders from content analysis
            key_stakeholders = ["End users", "Development team"]
            if content_analysis.get("success"):
                content_data = content_analysis.get("result", {})
                if isinstance(content_data, dict):
                    # Infer stakeholders from content structure
                    if "documentation" in str(content_data).lower():
                        key_stakeholders.append("Documentation users")
                    if "api" in str(content_data).lower():
                        key_stakeholders.append("API consumers")
                    if "cli" in str(content_data).lower():
                        key_stakeholders.append("Command-line users")
            
            # Extract success criteria from intelligence analysis
            success_criteria = ["Application meets functional requirements"]
            if intelligence_analysis.get("success"):
                intel_data = intelligence_analysis.get("result", {})
                if isinstance(intel_data, dict):
                    # Extract performance criteria from intelligence
                    if "performance" in intel_data:
                        success_criteria.append("Performance targets are met")
                    if "quality" in intel_data:
                        success_criteria.append("Quality standards are achieved")
            
            # Add criteria from project specifications
            if project_specs.get("target_platforms"):
                platforms = project_specs["target_platforms"]
                success_criteria.append(f"Application runs on {', '.join(platforms)}")
            
            # Extract challenges from historical context
            potential_challenges = ["Performance optimization", "Error handling and edge cases"]
            if historical_context.get("success"):
                hist_data = historical_context.get("result", {})
                if isinstance(hist_data, dict) and "documents" in hist_data:
                    # Analyze historical patterns for challenges
                    hist_text = str(hist_data).lower()
                    if "error" in hist_text or "fail" in hist_text:
                        potential_challenges.append("Error handling based on historical patterns")
                    if "performance" in hist_text:
                        potential_challenges.append("Performance optimization based on history")
            
            # Add challenges from project complexity
            if len(project_specs.get("technologies", [])) > 5:
                potential_challenges.append("Complex technology stack integration")
            if len(project_specs.get("target_platforms", [])) > 2:
                potential_challenges.append("Multi-platform compatibility")
            
            # Extract functional requirements from environment analysis
            functional_requirements = []
            if environment_info.get("success"):
                env_data = environment_info.get("result", {})
                if isinstance(env_data, dict):
                    # Extract requirements from environment
                    if "python" in str(env_data).lower():
                        functional_requirements.append("Python runtime compatibility")
                    if "node" in str(env_data).lower():
                        functional_requirements.append("Node.js runtime compatibility")
            
            # Add requirements from project specifications
            if project_specs.get("required_dependencies"):
                functional_requirements.append("Integration with required dependencies")
            if project_specs.get("project_type"):
                functional_requirements.append(f"{project_specs['project_type']} specific functionality")
            
            # Generate non-functional requirements
            non_functional_requirements = [
                "Performance: Response time under acceptable limits",
                "Security: Secure handling of user data",
                "Usability: Intuitive user interface"
            ]
            
            # Add environment-specific non-functional requirements
            if environment_info.get("success"):
                non_functional_requirements.append("Compatibility: Works in detected environment")
            
            # Generate user personas based on project type
            user_personas = ["Primary users"]
            project_type = project_specs.get("project_type", "")
            if "cli" in project_type.lower():
                user_personas.extend(["Command-line users", "System administrators"])
            elif "web" in project_type.lower():
                user_personas.extend(["Web users", "Browser users"])
            elif "api" in project_type.lower():
                user_personas.extend(["API consumers", "Integration developers"])
            else:
                user_personas.append("Application users")
            
            # Determine business value
            business_value = f"Delivers {user_goal} through {project_specs.get('project_type', 'application')} implementation"
            
            # Assess technical complexity
            complexity_factors = 0
            if len(project_specs.get("technologies", [])) > 3:
                complexity_factors += 1
            if len(project_specs.get("target_platforms", [])) > 1:
                complexity_factors += 1
            if len(project_specs.get("required_dependencies", [])) > 5:
                complexity_factors += 1
            
            technical_complexity = "low" if complexity_factors == 0 else "medium" if complexity_factors <= 2 else "high"
            
            # Estimate effort based on complexity and scope
            effort_factors = complexity_factors
            if len(functional_requirements) > 5:
                effort_factors += 1
            if len(potential_challenges) > 3:
                effort_factors += 1
            
            estimated_effort = "small" if effort_factors <= 1 else "medium" if effort_factors <= 3 else "large"
            
            return {
                "core_objectives": core_objectives,
                "key_stakeholders": key_stakeholders,
                "success_criteria": success_criteria,
                "potential_challenges": potential_challenges,
                "functional_requirements": functional_requirements,
                "non_functional_requirements": non_functional_requirements,
                "user_personas": user_personas,
                "business_value": business_value,
                "technical_complexity": technical_complexity,
                "estimated_effort": estimated_effort,
                "intelligent_analysis": True,
                "project_specifications": project_specs,
                "analysis_method": "mcp_enhanced_analysis",
                "mcp_tool_analysis": {
                    "project_analysis": project_analysis,
                    "content_analysis": content_analysis,
                    "intelligence_analysis": intelligence_analysis,
                    "historical_context": historical_context,
                    "environment_info": environment_info
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error converting MCP analysis to product analysis: {e}")
            # Fallback to basic analysis
            return self._generate_fallback_intelligent_analysis(project_specs, user_goal)

    async def _enhanced_discovery_with_universal_tools(self, inputs: Dict[str, Any], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Universal tool access pattern for ProductAnalystAgent_v1"""
        
        # 1. Get ALL available tools (no filtering)
        tool_discovery = await self._get_all_available_mcp_tools()
        
        if not tool_discovery["discovery_successful"]:
            self.logger.error("[MCP] Tool discovery failed - falling back to limited functionality")
            return {"error": "Tool discovery failed", "limited_functionality": True}
        
        all_tools = tool_discovery["tools"]
        
        # 2. Intelligent tool selection based on context
        selected_tools = self._intelligently_select_tools(all_tools, inputs, shared_context)
        
        # 3. Use filesystem tools for project analysis
        project_analysis = {}
        if "filesystem_project_scan" in selected_tools:
            project_path = shared_context.get("project_path", inputs.get("project_path", "."))
            project_analysis = await self._call_mcp_tool(
                "filesystem_project_scan", 
                {
                    "scan_path": project_path,
                    "project_path": project_path,
                    "detect_project_type": True,
                    "analyze_structure": True,
                    "include_stats": True
                }
            )
        
        # 4. Use intelligence tools for analysis
        intelligence_analysis = {}
        if "adaptive_learning_analyze" in selected_tools:
            intelligence_analysis = await self._call_mcp_tool(
                "adaptive_learning_analyze",
                {"context": project_analysis, "domain": self.AGENT_ID}
            )
        
        # 5. Use content tools for deeper analysis
        content_analysis = {}
        if "web_content_extract" in selected_tools and project_analysis.get("success"):
            content_analysis = await self._call_mcp_tool(
                "web_content_extract",
                {
                    "content": str(project_analysis.get("structure", {})),
                    "extraction_type": "text"
                }
            )
        
        # 6. Use ChromaDB tools for historical context
        historical_context = {}
        if "chromadb_query_documents" in selected_tools:
            historical_context = await self._call_mcp_tool(
                "chromadb_query_documents",
                {"query": f"agent:{self.AGENT_ID} product_analysis", "limit": 10}
            )
        
        # 7. Use terminal tools for environment validation
        environment_info = {}
        if "terminal_get_environment" in selected_tools:
            environment_info = await self._call_mcp_tool(
                "terminal_get_environment",
                {}
            )
        
        # 8. Use tool discovery for dynamic capabilities
        tool_recommendations = {}
        if "get_tool_composition_recommendations" in selected_tools:
            tool_recommendations = await self._call_mcp_tool(
                "get_tool_composition_recommendations",
                {"context": {"agent_id": self.AGENT_ID, "task_type": "product_analysis"}}
            )
        
        # 9. Combine all analyses
        return {
            "universal_tool_access": True,
            "tools_available": len(all_tools),
            "tools_selected": len(selected_tools),
            "tool_categories": tool_discovery["categories"],
            "project_analysis": project_analysis,
            "intelligence_analysis": intelligence_analysis,
            "content_analysis": content_analysis,
            "historical_context": historical_context,
            "environment_info": environment_info,
            "tool_recommendations": tool_recommendations,
            "agent_domain": self.AGENT_ID,
            "analysis_timestamp": time.time()
        }

    def _intelligently_select_tools(self, all_tools: Dict[str, Any], inputs: Any, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent tool selection for ProductAnalystAgent_v1 - product analysis focused"""
        
        # Start with core tools every agent should consider
        core_tools = [
            "filesystem_project_scan",
            "chromadb_query_documents", 
            "terminal_get_environment"
        ]
        
        # Add product analysis specific tools
        product_analysis_tools = [
            "web_content_extract",
            "content_generate_dynamic",
            "chromadb_store_document",
            "get_tool_composition_recommendations"
        ]
        core_tools.extend(product_analysis_tools)
        
        # Add requirements analysis tools
        requirements_tools = [
            "filesystem_read_file",
            "content_generate_dynamic",
            "chromadb_store_document",
            "generate_performance_recommendations"
        ]
        core_tools.extend(requirements_tools)
        
        # Add intelligence tools for all agents
        intelligence_tools = [
            "adaptive_learning_analyze",
            "get_real_time_performance_analysis",
            "generate_performance_recommendations"
        ]
        core_tools.extend(intelligence_tools)
        
        # Select available tools
        selected = {}
        for tool_name in core_tools:
            if tool_name in all_tools:
                selected[tool_name] = all_tools[tool_name]
        
        self.logger.info(f"[MCP] Selected {len(selected)} tools for {getattr(self, 'AGENT_ID', 'unknown_agent')}")
        return selected

    def _generate_fallback_intelligent_analysis(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """Generate fallback analysis when LLM is unavailable."""
        
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
            "project_specifications": project_specs,
            "analysis_method": "fallback_extraction"
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

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from LLM response, handling markdown code blocks."""
        response = response.strip()
        
        # Check if response is wrapped in markdown code blocks
        if response.startswith('```json'):
            # Find the end of the code block
            lines = response.split('\n')
            json_lines = []
            in_json_block = False
            
            for line in lines:
                if line.strip() == '```json':
                    in_json_block = True
                    continue
                elif line.strip() == '```' and in_json_block:
                    break
                elif in_json_block:
                    json_lines.append(line)
            
            return '\n'.join(json_lines)
        elif response.startswith('```'):
            # Handle generic code blocks
            lines = response.split('\n')
            json_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```') and not in_code_block:
                    in_code_block = True
                    continue
                elif line.strip() == '```' and in_code_block:
                    break
                elif in_code_block:
                    json_lines.append(line)
            
            return '\n'.join(json_lines)
        else:
            # Response is already clean JSON
            return response

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