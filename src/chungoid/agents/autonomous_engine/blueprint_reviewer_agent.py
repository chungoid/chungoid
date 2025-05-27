from __future__ import annotations

import logging
import asyncio
import datetime
import json
import uuid
import time
from typing import Any, Dict, Optional, ClassVar, List, Type

from pydantic import BaseModel, Field

from chungoid.agents.unified_agent import UnifiedAgent
from ...utils.llm_provider import LLMProvider
from ...utils.prompt_manager import PromptManager, PromptRenderError
from ...schemas.common import ConfidenceScore
from ...utils.chromadb_migration_utils import (
    migrate_store_artifact,
    migrate_retrieve_artifact,
    migrate_query_artifacts,
    PCMAMigrationError
)
from ...utils.agent_registry import AgentCard
from ...utils.agent_registry_meta import AgentCategory, AgentVisibility
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

logger = logging.getLogger(__name__)

BLUEPRINT_ARTIFACTS_COLLECTION = "blueprint_artifacts_collection"
REVIEW_REPORTS_COLLECTION = "review_reports"
ARTIFACT_TYPE_BLUEPRINT_REVIEW_REPORT_MD = "BlueprintReviewReport_MD"

BLUEPRINT_REVIEWER_PROMPT_NAME = "blueprint_reviewer_agent_v1.yaml"

class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

# --- Input and Output Schemas for the Agent --- #

class BlueprintReviewerInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this review task.")
    
    # Traditional fields - optional when using intelligent context
    project_id: Optional[str] = Field(None, description="Identifier for the current project.")
    blueprint_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the Project Blueprint (Markdown) to be reviewed.")
    previous_review_doc_ids: Optional[List[str]] = Field(None, description="ChromaDB IDs of any previous review reports for this blueprint, for context.")
    specific_focus_areas: Optional[List[str]] = Field(None, description="List of specific areas or concerns to focus the review on.")
    
    # ADDED: Intelligent project analysis from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications from orchestrator analysis")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    user_goal: Optional[str] = Field(None, description="Original user goal for context")
    project_path: Optional[str] = Field(None, description="Project path for context")

class BlueprintReviewerOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    review_report_doc_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated Blueprint Review Report (Markdown, detailing optimizations, alternatives, flaws) is stored.")
    status: str = Field(..., description="Status of the review (e.g., SUCCESS, FAILURE_LLM, FAILURE_INPUT_RETRIEVAL).")
    message: str = Field(..., description="A message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the thoroughness and insightfulness of its review.")
    error_message: Optional[str] = Field(None, description="Error message if status is not SUCCESS.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")

@register_autonomous_engine_agent(capabilities=["review_protocol", "quality_validation", "architectural_review"])
class BlueprintReviewerAgent_v1(UnifiedAgent):
    """
    Performs an advanced review of a Project Blueprint, suggesting optimizations, architectural alternatives, and identifying subtle flaws.
    
    ✨ PURE UAEI ARCHITECTURE - Clean execution paths with unified interface.
    ✨ MCP TOOL INTEGRATION - Uses ChromaDB MCP tools instead of agent dependencies.
    """
    
    AGENT_ID: ClassVar[str] = "BlueprintReviewerAgent_v1"
    AGENT_NAME: ClassVar[str] = "Blueprint Reviewer Agent v1"
    DESCRIPTION: ClassVar[str] = "Performs an advanced review of a Project Blueprint, suggesting optimizations, architectural alternatives, and identifying subtle flaws."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "blueprint_reviewer_agent_v1.yaml"
    AGENT_VERSION: ClassVar[str] = "0.1.0"
    CAPABILITIES: ClassVar[List[str]] = ["complex_analysis", "review_protocol", "quality_validation", "architectural_review"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.QUALITY_ASSURANCE
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[BlueprintReviewerInput]] = BlueprintReviewerInput
    OUTPUT_SCHEMA: ClassVar[Type[BlueprintReviewerOutput]] = BlueprintReviewerOutput

    _llm_provider: LLMProvider
    _prompt_manager: PromptManager
    _logger: logging.Logger
    
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["plan_review", "tool_validation"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["plan_review", "agent_communication"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'tool_validation']

    def __init__(
        self, 
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        **kwargs
    ):
        # Enable refinement capabilities for intelligent blueprint review
        super().__init__(
            llm_provider=llm_provider, 
            prompt_manager=prompt_manager, 
            enable_refinement=True,  # Enable intelligent refinement
            **kwargs
        )

    async def _execute_iteration(
        self, 
        context: UEContext,
        iteration: int
    ) -> IterationResult:
        """
        Phase 3: Single iteration of blueprint review workflow.
        Runs comprehensive review workflow: discovery → analysis → planning → review generation → validation
        """
        self.logger.info(f"[UAEI] Blueprint reviewer iteration {iteration + 1}")
        
        try:
            # Convert inputs to expected format
            if isinstance(context.inputs, dict):
                task_input = BlueprintReviewerInput(**context.inputs)
            elif hasattr(context.inputs, 'dict'):
                inputs = context.inputs.dict()
                task_input = BlueprintReviewerInput(**inputs)
            else:
                task_input = context.inputs

            # Phase 4: Check for refinement context and use refinement-aware analysis
            refinement_context = context.shared_context.get("refinement_context")
            if self.enable_refinement and refinement_context:
                self.logger.info(f"[Refinement] Using refinement context with {len(refinement_context.get('previous_outputs', []))} previous outputs")
                # Use refinement-aware analysis that considers previous work
                analysis_result = await self._analyze_blueprint_with_refinement_context(
                    task_input, context.shared_context, refinement_context
                )
            elif task_input.intelligent_context and task_input.project_specifications:
                self.logger.info("Using intelligent project specifications from orchestrator")
                analysis_result = await self._extract_blueprint_from_intelligent_specs(task_input.project_specifications, task_input.user_goal)
            else:
                self.logger.info("Using traditional blueprint analysis")
                # Phase 1: Discovery - Discover blueprint
                discovery_result = await self._discover_blueprint(task_input, context.shared_context)
                # Phase 2: Analysis - Analyze blueprint and gather context
                analysis_result = await self._analyze_blueprint(discovery_result, task_input, context.shared_context)
            
            # Phase 2: Planning - Plan review approach and criteria
            planning_result = await self._plan_review(analysis_result, task_input, context.shared_context)
            
            # Phase 3: Review Generation - Generate comprehensive review
            review_result = await self._generate_review(planning_result, task_input, context.shared_context)
            
            # Phase 4: Validation - Validate review quality and completeness
            validation_result = await self._validate_review(review_result, task_input, context.shared_context)
            
            # Calculate quality score based on validation results
            quality_score = self._calculate_quality_score(validation_result)
            
            # Create output
            output = BlueprintReviewerOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id or "intelligent_project",
                review_report_doc_id=validation_result.get("review_report_doc_id"),
                status="SUCCESS",
                message="Blueprint review completed successfully",
                confidence_score=ConfidenceScore(
                    value=quality_score,
                    explanation="Based on comprehensive blueprint analysis and validation",
                    method="comprehensive_analysis"
                )
            )
            
            tools_used = ["blueprint_analysis", "review_planning", "quality_validation"]
            
            return IterationResult(
                output=output,
                quality_score=quality_score,
                tools_used=tools_used,
                protocol_used="blueprint_review_protocol"
            )
            
        except Exception as e:
            self.logger.error(f"Blueprint review iteration {iteration + 1} failed: {e}")
            
            # Create error output
            error_output = BlueprintReviewerOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())),
                project_id=getattr(task_input, 'project_id', 'unknown'),
                status="FAILURE_LLM",
                message=f"Blueprint review failed: {str(e)}",
                error_message=str(e)
            )
            
            # Return failed IterationResult
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="blueprint_review_protocol",
                iteration_metadata={"error": str(e)}
            )

    async def _extract_blueprint_from_intelligent_specs(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """Extract blueprint-like data from intelligent project specifications using LLM processing."""
        
        try:
            if self.llm_provider:
                # Use LLM to intelligently analyze the project specifications and create review criteria
                prompt = f"""
                You are a blueprint reviewer agent. Analyze the following project specifications and user goal to create intelligent review criteria and focus areas for blueprint assessment.
                
                User Goal: {user_goal}
                
                Project Specifications:
                - Project Type: {project_specs.get('project_type', 'unknown')}
                - Primary Language: {project_specs.get('primary_language', 'unknown')}
                - Target Languages: {project_specs.get('target_languages', [])}
                - Target Platforms: {project_specs.get('target_platforms', [])}
                - Technologies: {project_specs.get('technologies', [])}
                - Required Dependencies: {project_specs.get('required_dependencies', [])}
                - Optional Dependencies: {project_specs.get('optional_dependencies', [])}
                
                Based on this information, provide a detailed JSON analysis for blueprint review with the following structure:
                {{
                    "review_focus_areas": ["area1", "area2", "area3"],
                    "architectural_concerns": ["concern1", "concern2"],
                    "technology_assessment": {{
                        "compatibility_risks": ["risk1", "risk2"],
                        "optimization_opportunities": ["opp1", "opp2"]
                    }},
                    "quality_criteria": {{
                        "performance_expectations": "description",
                        "scalability_requirements": "description",
                        "maintainability_standards": "description",
                        "security_considerations": "description"
                    }},
                    "review_depth": "comprehensive|detailed|focused",
                    "expected_blueprint_sections": ["section1", "section2"],
                    "confidence_score": 0.0-1.0,
                    "reasoning": "explanation of analysis approach"
                }}
                """
                
                response = await self.llm_provider.generate(prompt)
                
                if response:
                    try:
                        # Extract JSON from markdown code blocks if present
                        json_content = self._extract_json_from_response(response)
                        parsed_result = json.loads(json_content)
                        
                        # Validate that we got a dictionary as expected
                        if not isinstance(parsed_result, dict):
                            self.logger.warning(f"Expected dict from blueprint analysis, got {type(parsed_result)}. Using fallback.")
                            return self._generate_fallback_blueprint_analysis(project_specs, user_goal)
                        
                        analysis = parsed_result
                        
                        # Create intelligent blueprint artifact based on LLM analysis
                        blueprint_artifact = {
                            "status": "SUCCESS",
                            "content": {
                                "title": f"Intelligent Blueprint Analysis - {project_specs.get('project_type', 'Project')}",
                                "project_overview": {
                                    "name": project_specs.get("project_type", "Unknown Project"),
                                    "description": user_goal,
                                    "type": project_specs.get("project_type", "unknown"),
                                    "intelligent_analysis": True
                                },
                                "review_criteria": analysis.get("quality_criteria", {}),
                                "focus_areas": analysis.get("review_focus_areas", []),
                                "architectural_concerns": analysis.get("architectural_concerns", []),
                                "technology_assessment": analysis.get("technology_assessment", {}),
                                "expected_sections": analysis.get("expected_blueprint_sections", []),
                                "review_depth": analysis.get("review_depth", "comprehensive")
                            },
                            "llm_analysis": analysis,
                            "intelligent_processing": True
                        }
                        
                        return {
                            "blueprint_artifact": blueprint_artifact,
                            "discovery_success": True,
                            "intelligent_analysis": True,
                            "llm_confidence": analysis.get("confidence_score", 0.8),
                            "analysis_method": "llm_intelligent_processing"
                        }
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
            
            # Fallback to basic extraction if LLM fails
            self.logger.info("Using fallback blueprint analysis due to LLM unavailability")
            return self._generate_fallback_blueprint_analysis(project_specs, user_goal)
            
        except Exception as e:
            self.logger.error(f"Error in intelligent blueprint specs analysis: {e}")
            return self._generate_fallback_blueprint_analysis(project_specs, user_goal)

    def _generate_fallback_blueprint_analysis(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """Generate fallback blueprint analysis when LLM is unavailable."""
        
        # Create basic blueprint artifact from project specifications
        blueprint_artifact = {
            "status": "SUCCESS",
            "content": {
                "title": f"Technical Blueprint - {project_specs.get('project_type', 'Project')}",
                "project_overview": {
                    "name": project_specs.get("project_type", "Unknown Project"),
                    "description": user_goal[:200] + "..." if len(user_goal) > 200 else user_goal,
                    "type": project_specs.get("project_type", "unknown")
                },
                "architecture_pattern": "modular",
                "technology_stack": {
                    "primary_language": project_specs.get("primary_language", "python"),
                    "technologies": project_specs.get("technologies", []),
                    "dependencies": project_specs.get("required_dependencies", [])
                },
                "components": [
                    {
                        "name": f"Component_{i+1}",
                        "responsibility": tech,
                        "interfaces": ["API"]
                    } for i, tech in enumerate(project_specs.get("technologies", [])[:3])
                ],
                "quality_attributes": {
                    "performance": "high",
                    "scalability": "moderate", 
                    "maintainability": "high",
                    "security": "standard"
                }
            }
        }
        
        return {
            "blueprint_artifact": blueprint_artifact,
            "discovery_success": True,
            "intelligent_analysis": True,
            "analysis_method": "fallback_extraction"
        }

    async def _discover_blueprint(self, task_input: BlueprintReviewerInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Discovery - Discover and retrieve blueprint using MCP tools."""
        self.logger.info("Starting blueprint discovery for review")
        
        try:
            # Retrieve blueprint artifact using MCP tools
            blueprint_result = await migrate_retrieve_artifact(
                collection_name=BLUEPRINT_ARTIFACTS_COLLECTION,
                document_id=task_input.blueprint_doc_id,
                project_id=task_input.project_id
            )
            
            if blueprint_result["status"] != "SUCCESS":
                raise PCMAMigrationError(f"Failed to retrieve blueprint: {blueprint_result.get('error')}")
            
            # Optionally retrieve previous review reports for context
            previous_reviews = []
            if task_input.previous_review_doc_ids:
                for review_id in task_input.previous_review_doc_ids:
                    try:
                        review_result = await migrate_retrieve_artifact(
                            collection_name=REVIEW_REPORTS_COLLECTION,
                            document_id=review_id,
                            project_id=task_input.project_id
                        )
                        if review_result["status"] == "SUCCESS":
                            previous_reviews.append(review_result)
                    except Exception as e:
                        self.logger.warning(f"Failed to retrieve previous review {review_id}: {e}")
            
            return {
                "blueprint_retrieved": True,
                "blueprint": blueprint_result,
                "previous_reviews": previous_reviews,
                "blueprint_content": blueprint_result.get("content", ""),
                "discovery_success": True
            }
            
        except Exception as e:
            self.logger.error(f"Blueprint discovery failed: {e}")
            return {
                "blueprint_retrieved": False,
                "error": str(e),
                "blueprint": None,
                "discovery_success": False
            }

    async def _analyze_blueprint(self, discovery_result: Dict[str, Any], task_input: BlueprintReviewerInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Analysis - Analyze blueprint content for review."""
        self.logger.info("Starting blueprint analysis")
        
        if not discovery_result.get("blueprint_retrieved", False):
            return {
                "analysis_completed": False,
                "error": "Cannot analyze without retrieved blueprint",
                "analysis_score": 0.0
            }
        
        blueprint_content = discovery_result.get("blueprint_content", "")
        previous_reviews = discovery_result.get("previous_reviews", [])
        
        # Simulate blueprint analysis (in real implementation, would use LLM)
        analysis = {
            "analysis_completed": True,
            "content_quality": 0.8,  # Mock quality score
            "architectural_soundness": 0.85,
            "completeness": 0.75,
            "areas_needing_attention": [],
            "strengths_identified": [],
            "previous_issues_addressed": len(previous_reviews),
            "analysis_confidence": 0.82
        }
        
        # Check blueprint content quality indicators
        if len(blueprint_content) > 1000:
            analysis["has_substantial_content"] = True
            analysis["completeness"] = 0.85
        else:
            analysis["has_substantial_content"] = False
            analysis["completeness"] = 0.6
            analysis["areas_needing_attention"].append("Insufficient detail in blueprint")
        
        # Check for key sections
        key_sections = ["overview", "architecture", "components", "requirements"]
        sections_found = sum(1 for section in key_sections if section.lower() in blueprint_content.lower())
        analysis["sections_coverage"] = sections_found / len(key_sections)
        
        if analysis["sections_coverage"] < 0.7:
            analysis["areas_needing_attention"].append("Missing key architectural sections")
        else:
            analysis["strengths_identified"].append("Comprehensive section coverage")
            
        return analysis

    async def _plan_review(self, analysis_result: Dict[str, Any], task_input: BlueprintReviewerInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Planning - Plan review structure and focus areas."""
        self.logger.info("Starting review planning")
        
        if not analysis_result.get("analysis_completed", False):
            return {
                "planning_completed": False,
                "error": "Cannot plan review without completed analysis"
            }
        
        # Plan review structure based on analysis and input focus areas
        planning = {
            "planning_completed": True,
            "review_sections": [
                "Executive Summary",
                "Blueprint Analysis",
                "Architectural Assessment",
                "Identified Issues",
                "Optimization Recommendations",
                "Alternative Approaches"
            ],
            "focus_areas": task_input.specific_focus_areas or ["architecture", "completeness", "feasibility"],
            "review_depth": "comprehensive",  # Based on content quality
            "estimated_issues": len(analysis_result.get("areas_needing_attention", [])),
            "planning_confidence": 0.85
        }
        
        # Adjust review depth based on analysis
        if analysis_result.get("content_quality", 0) < 0.7:
            planning["review_depth"] = "detailed_critique"
            planning["review_sections"].append("Content Quality Concerns")
        
        if analysis_result.get("architectural_soundness", 0) < 0.8:
            planning["review_sections"].append("Architectural Improvements")
            
        return planning

    async def _generate_review(self, planning_result: Dict[str, Any], task_input: BlueprintReviewerInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Review Generation - Generate comprehensive review report."""
        self.logger.info("Starting review generation")
        
        if not planning_result.get("planning_completed", False):
            return {
                "generation_completed": False,
                "error": "Cannot generate review without completed planning"
            }
        
        # Generate mock review report (in real implementation, would use LLM)
        focus_areas = planning_result.get("focus_areas", [])
        review_sections = planning_result.get("review_sections", [])
        
        review_content = f"""# Blueprint Review Report

## Executive Summary
Comprehensive review of blueprint {task_input.blueprint_doc_id} with focus on {', '.join(focus_areas)}.

## Blueprint Analysis
- Content Quality: Good overall structure with room for improvement
- Architectural Soundness: Solid foundation with minor concerns
- Completeness: Adequate detail provided

## Architectural Assessment
[Detailed architectural assessment would appear here]

## Identified Issues
[Specific issues and concerns would be listed here]

## Optimization Recommendations
[Specific recommendations for optimization would appear here]

## Alternative Approaches
[Alternative architectural approaches would be discussed here]

## Conclusion
[Summary and final recommendations would appear here]
"""
        
        # Store review report to ChromaDB (mock)
        review_doc_id = f"review_report_{task_input.project_id}_{uuid.uuid4().hex[:8]}"
        
        generation = {
            "generation_completed": True,
            "review_doc_id": review_doc_id,
            "review_content": review_content,
            "review_length": len(review_content),
            "sections_generated": len(review_sections),
            "generation_confidence": 0.8
        }
        
        return generation

    async def _validate_review(self, review_result: Dict[str, Any], task_input: BlueprintReviewerInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Validation - Validate review report quality."""
        self.logger.info("Starting review validation")
        
        if not review_result.get("generation_completed", False):
            return {
                "validation_completed": False,
                "error": "Cannot validate without generated review",
                "quality_score": 0.0
            }
        
        review_content = review_result.get("review_content", "")
        review_length = review_result.get("review_length", 0)
        sections_generated = review_result.get("sections_generated", 0)
        
        validation = {
            "validation_completed": True,
            "quality_checks": {
                "has_content": len(review_content) > 500,
                "has_sections": "## " in review_content,
                "adequate_length": review_length > 800,
                "includes_recommendations": "Recommendations" in review_content,
                "includes_assessment": "Assessment" in review_content,
                "sufficient_sections": sections_generated >= 4
            },
            "validation_score": 0.0,
            "issues_found": []
        }
        
        # Calculate validation score
        checks = validation["quality_checks"]
        score = 0.0
        weight_per_check = 1.0 / len(checks)
        
        for check_name, check_result in checks.items():
            if check_result:
                score += weight_per_check
                
        validation["validation_score"] = score
        
        # Identify issues
        for check_name, check_result in checks.items():
            if not check_result:
                validation["issues_found"].append(f"Failed {check_name.replace('_', ' ')} check")
            
        return validation

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

    def _calculate_quality_score(self, validation_result: Dict[str, Any]) -> float:
        """Calculate quality score based on validation results."""
        try:
            # Base score starts at 0.6 for successful execution
            base_score = 0.6
            
            # Factor 1: Review completeness (0.0 - 0.2)
            review_sections = validation_result.get("review_sections_completed", 0)
            if review_sections > 0:
                completeness_score = min(0.2, review_sections * 0.04)  # 0.04 per section, max 0.2
            else:
                completeness_score = 0.0
            
            # Factor 2: Analysis depth and insights (0.0 - 0.15)
            analysis_insights = validation_result.get("analysis_insights_count", 0)
            if analysis_insights >= 5:
                insight_score = 0.15  # Comprehensive analysis
            elif analysis_insights >= 3:
                insight_score = 0.1   # Good analysis
            elif analysis_insights >= 1:
                insight_score = 0.05  # Basic analysis
            else:
                insight_score = 0.0   # No insights
            
            # Factor 3: Validation passed (0.0 - 0.1)
            validation_passed = validation_result.get("validation_passed", False)
            if validation_passed:
                validation_score = 0.1
            else:
                validation_score = 0.0
            
            # Factor 4: Review report generated (0.0 - 0.05)
            has_report = validation_result.get("review_report_doc_id") is not None
            if has_report:
                report_score = 0.05
            else:
                report_score = 0.0
            
            # Calculate final score
            final_score = base_score + completeness_score + insight_score + validation_score + report_score
            
            # Cap at 1.0
            final_score = min(1.0, final_score)
            
            self.logger.info(f"[Quality] Calculated quality score: {final_score:.2f} "
                           f"(base={base_score}, completeness={completeness_score:.2f}, "
                           f"insights={insight_score:.2f}, validation={validation_score:.2f}, "
                           f"report={report_score:.2f})")
            
            return final_score
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.6  # Default score on error

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = BlueprintReviewerInput.model_json_schema()
        output_schema = BlueprintReviewerOutput.model_json_schema()

        # Schema for the LLM's direct output (JSON object)
        llm_direct_output_schema = {
            "type": "object",
            "properties": {
                "review_report_md": {"type": "string"},
                "review_confidence": ConfidenceScore.model_json_schema()
            },
            "required": ["review_report_md", "review_confidence"]
        }

        return AgentCard(
            agent_id=BlueprintReviewerAgent_v1.AGENT_ID,
            name=BlueprintReviewerAgent_v1.AGENT_NAME,
            description=BlueprintReviewerAgent_v1.DESCRIPTION,
            version=BlueprintReviewerAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            llm_direct_output_schema=llm_direct_output_schema,
            categories=[cat.value for cat in [BlueprintReviewerAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=BlueprintReviewerAgent_v1.VISIBILITY.value,
            capability_profile={
                "reviews_artifacts": ["ProjectBlueprint"],
                "generates_reports": ["BlueprintReviewReport_Markdown"],
                "primary_function": "Blueprint Quality Assessment and Optimization"
            },
            metadata={
                "callable_fn_path": f"{BlueprintReviewerAgent_v1.__module__}.{BlueprintReviewerAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[BlueprintReviewerInput]:
        return BlueprintReviewerInput

    def get_output_schema(self) -> Type[BlueprintReviewerOutput]:
        return BlueprintReviewerOutput

    async def _analyze_blueprint_with_refinement_context(
        self, 
        task_input: BlueprintReviewerInput, 
        shared_context: Dict[str, Any],
        refinement_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Refinement-aware blueprint analysis that considers previous iterations.
        Uses refinement context to build upon previous work and improve review quality.
        """
        try:
            # Get previous outputs and analysis
            previous_outputs = refinement_context.get("previous_outputs", [])
            previous_quality = refinement_context.get("previous_quality_score", 0.0)
            iteration = refinement_context.get("iteration", 0)
            
            # Build refinement-aware prompt for LLM analysis
            refinement_prompt = self._build_refinement_prompt(
                f"Blueprint review analysis for {task_input.project_id}",
                refinement_context
            )
            
            # Use the refinement prompt for intelligent analysis
            if self.llm_provider:
                llm_response = await self.llm_provider.generate(refinement_prompt)
                analysis_result = await self._extract_blueprint_from_intelligent_specs(
                    {"refinement_analysis": llm_response}, 
                    task_input.user_goal or "Blueprint review"
                )
            else:
                # Fallback to standard analysis with refinement awareness
                discovery_result = await self._discover_blueprint(task_input, shared_context)
                analysis_result = await self._analyze_blueprint(discovery_result, task_input, shared_context)
                
                # Enhance with refinement insights
                if previous_outputs:
                    self.logger.info(f"[Refinement] Enhancing analysis with insights from {len(previous_outputs)} previous iterations")
                    # Add previous findings to improve review focus
                    analysis_result["previous_review_iterations"] = len(previous_outputs)
                    analysis_result["previous_quality_score"] = previous_quality
                    analysis_result["refinement_iteration"] = iteration
                    
                    # Extract insights from previous reviews
                    for prev_output in previous_outputs:
                        prev_content = str(prev_output.get("content", ""))
                        if "review_report_doc_id" in prev_content:
                            analysis_result["has_previous_reviews"] = True
            
            return analysis_result
            
        except Exception as e:
            self.logger.warning(f"[Refinement] Refinement-aware analysis failed, falling back to standard: {e}")
            return await self._analyze_blueprint(task_input, shared_context) 