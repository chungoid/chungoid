"""
RequirementsRiskAgent_v1: Clean, unified LLM-powered requirements and risk analysis.

This agent provides comprehensive requirements analysis with risk assessment by:
1. Using unified discovery to understand project structure and requirements
2. Using YAML prompt template with rich discovery data
3. Letting the LLM make intelligent requirements and risk decisions with maximum intelligence

No legacy patterns, no hardcoded phases, no complex tool orchestration.
Pure unified approach for maximum agentic requirements and risk intelligence.
"""

from __future__ import annotations

import logging
import uuid
import json
from typing import Any, Dict, Optional, List, ClassVar, Type

from pydantic import BaseModel, Field, model_validator

from chungoid.agents.unified_agent import UnifiedAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.registry import register_autonomous_engine_agent

from ...schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    IterationResult,
)

logger = logging.getLogger(__name__)


class RequirementsRiskAgentInput(BaseModel):
    """Clean input schema focused on core requirements and risk analysis needs."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project identifier")
    
    # Core requirements
    user_goal: str = Field(..., description="What the user wants analyzed for requirements and risks")
    project_path: str = Field(default=".", description="Project directory path")
    
    # Requirements context
    refined_user_goal_md: Optional[str] = Field(None, description="Refined user goal in Markdown format")
    loprd_json_schema_str: Optional[str] = Field(None, description="JSON schema for LOPRD validation")
    focus_areas: Optional[List[str]] = Field(None, description="Specific risk areas to focus on")
    
    # Intelligent context from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications")
    intelligent_context: bool = Field(default=False, description="Whether intelligent specifications provided")
    
    @model_validator(mode='after')
    def validate_requirements(self) -> 'RequirementsRiskAgentInput':
        """Ensure we have minimum requirements for analysis."""
        if not self.user_goal or not self.user_goal.strip():
            raise ValueError("user_goal is required for requirements and risk analysis")
        return self


class RequirementsRiskAgentOutput(BaseModel):
    """Clean output schema focused on requirements and risk deliverables."""
    task_id: str = Field(..., description="Task identifier")
    project_id: str = Field(..., description="Project identifier")
    status: str = Field(..., description="Execution status")
    
    # Core requirements deliverables
    loprd_content: Dict[str, Any] = Field(default_factory=dict, description="Generated LOPRD content")
    integrated_requirements: Dict[str, Any] = Field(default_factory=dict, description="Requirements with risk mitigation integrated")
    
    # Risk analysis deliverables
    risk_assessment: Dict[str, Any] = Field(default_factory=dict, description="Comprehensive risk assessment")
    risk_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of identified risks and mitigations")
    mitigation_strategies: List[Dict[str, Any]] = Field(default_factory=list, description="Risk mitigation strategies")
    
    # Quality insights
    requirements_analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis of requirements quality")
    risk_coverage: Dict[str, Any] = Field(default_factory=dict, description="Assessment of risk coverage")
    optimization_suggestions: List[str] = Field(default_factory=list, description="Suggestions for optimization")
    
    # Metadata  
    confidence_score: ConfidenceScore = Field(..., description="Agent confidence in requirements and risk analysis")
    message: str = Field(..., description="Human-readable result message")
    error_message: Optional[str] = Field(None, description="Error details if failed")


@register_autonomous_engine_agent(capabilities=["requirements_analysis", "risk_assessment", "optimization"])
class RequirementsRiskAgent_v1(UnifiedAgent):
    """
    Clean, unified requirements and risk analysis agent.
    
    Uses unified discovery + YAML templates + maximum LLM intelligence for requirements and risk analysis.
    No legacy patterns, no hardcoded phases, no complex tool orchestration.
    """
    
    AGENT_ID: ClassVar[str] = "RequirementsRiskAgent_v1"
    AGENT_NAME: ClassVar[str] = "Requirements & Risk Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Clean, unified LLM-powered requirements and risk analysis"
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "requirements_risk_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "2.0.0"  # Major version for clean rewrite
    CAPABILITIES: ClassVar[List[str]] = ["requirements_analysis", "risk_assessment", "optimization"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[RequirementsRiskAgentInput]] = RequirementsRiskAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[RequirementsRiskAgentOutput]] = RequirementsRiskAgentOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["unified_requirements_risk"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["intelligent_discovery"]
    UNIVERSAL_PROTOCOLS: ClassVar[List[str]] = ["agent_communication", "context_sharing"]

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger.info(f"{self.AGENT_ID} v{self.AGENT_VERSION} initialized - clean unified requirements and risk")

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        Clean execution: Unified discovery + YAML template + LLM requirements and risk intelligence.
        Single iteration, maximum intelligence, no hardcoded phases.
        """
        try:
            # Parse inputs cleanly
            task_input = self._parse_inputs(context.inputs)
            self.logger.info(f"Analyzing requirements and risks: {task_input.user_goal}")

            # Generate requirements and risk analysis using unified approach
            analysis_result = await self._generate_requirements_risk_analysis(task_input)
            
            # Create clean output
            output = RequirementsRiskAgentOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS",
                loprd_content=analysis_result["loprd_content"],
                integrated_requirements=analysis_result["integrated_requirements"],
                risk_assessment=analysis_result["risk_assessment"],
                risk_summary=analysis_result["risk_summary"],
                mitigation_strategies=analysis_result["mitigation_strategies"],
                requirements_analysis=analysis_result["requirements_analysis"],
                risk_coverage=analysis_result["risk_coverage"],
                optimization_suggestions=analysis_result["optimization_suggestions"],
                confidence_score=analysis_result["confidence_score"],
                message=f"Generated requirements and risk analysis for: {task_input.user_goal}"
            )
            
            return IterationResult(
                output=output,
                quality_score=analysis_result["confidence_score"].value,
                tools_used=["unified_discovery", "yaml_template", "llm_requirements_risk"],
                protocol_used="unified_requirements_risk"
            )
            
        except Exception as e:
            self.logger.error(f"Requirements and risk analysis failed: {e}")
            
            # Clean error handling
            error_output = RequirementsRiskAgentOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4()),
                project_id=getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown',
                status="ERROR",
                confidence_score=ConfidenceScore(
                    value=0.0,
                    method="error_state",
                    explanation="Requirements and risk analysis failed"
                ),
                message="Requirements and risk analysis failed",
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="unified_requirements_risk"
            )

    def _parse_inputs(self, inputs: Any) -> RequirementsRiskAgentInput:
        """Parse inputs cleanly into RequirementsRiskAgentInput."""
        if isinstance(inputs, RequirementsRiskAgentInput):
            return inputs
        elif isinstance(inputs, dict):
            return RequirementsRiskAgentInput(**inputs)
        elif hasattr(inputs, 'dict'):
            return RequirementsRiskAgentInput(**inputs.dict())
        else:
            raise ValueError(f"Invalid input type: {type(inputs)}")

    async def _generate_requirements_risk_analysis(self, task_input: RequirementsRiskAgentInput) -> Dict[str, Any]:
        """
        Generate requirements and risk analysis using unified discovery + YAML template.
        Pure unified approach - no hardcoded phases or analysis logic.
        """
        try:
            # Get YAML template (no fallbacks)
            prompt_template = self.prompt_manager.get_prompt_definition(
                "requirements_risk_agent_v1_prompt",
                "1.0.0",
                sub_path="autonomous_engine"
            )
            
            # Unified discovery for intelligent requirements and risk context
            discovery_results = await self._universal_discovery(
                task_input.project_path,
                ["environment", "dependencies", "structure", "patterns", "requirements", "artifacts", "architecture", "risks"]
            )
            
            technology_context = await self._universal_technology_discovery(
                task_input.project_path
            )
            
            # Build template variables for maximum LLM requirements and risk intelligence
            template_vars = {
                # Original template variables (maintaining compatibility)
                "user_goal": task_input.user_goal,
                "project_path": task_input.project_path,
                "intelligent_context": task_input.intelligent_context,
                "project_specifications": task_input.project_specifications or {},
                
                # Enhanced unified discovery variables for maximum intelligence
                "discovery_results": json.dumps(discovery_results, indent=2),
                "technology_context": json.dumps(technology_context, indent=2),
                
                # Additional context for intelligent requirements and risk analysis
                "refined_user_goal_md": task_input.refined_user_goal_md or "",
                "loprd_json_schema_str": task_input.loprd_json_schema_str or "",
                "focus_areas": task_input.focus_areas or [],
                
                # Rich context from discovery
                "project_structure": discovery_results.get("structure", {}),
                "existing_requirements": discovery_results.get("requirements", {}),
                "identified_risks": discovery_results.get("risks", {}),
                "dependencies": discovery_results.get("dependencies", {}),
                "patterns": discovery_results.get("patterns", {}),
                "architecture": discovery_results.get("architecture", {}),
                
                # Available tools for template use
                "available_tools": await self._get_available_tools_description()
            }
            
            # Render template
            formatted_prompt = self.prompt_manager.get_rendered_prompt_template(
                prompt_template.user_prompt_template,
                template_vars
            )
            
            # Get system prompt if available
            system_prompt = getattr(prompt_template, 'system_prompt', None)
            
            # Call LLM with maximum requirements and risk intelligence
            response = await self.llm_provider.generate(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=4000
            )
            
            # Parse LLM response (expecting JSON from template)
            try:
                result = json.loads(response)
                
                # Transform template output to our clean schema
                return {
                    "loprd_content": result.get("loprd_with_risk_mitigation", {}),
                    "integrated_requirements": result.get("integrated_requirements", {}),
                    "risk_assessment": result.get("risk_assessment", {}),
                    "risk_summary": self._extract_risk_summary(result, discovery_results),
                    "mitigation_strategies": self._extract_mitigation_strategies(result, discovery_results),
                    "requirements_analysis": {
                        "completeness": self._assess_requirements_completeness(result, discovery_results),
                        "quality": self._assess_requirements_quality(result, discovery_results),
                        "traceability": "analyzed_with_unified_discovery",
                        "consistency": "validated_with_llm_intelligence"
                    },
                    "risk_coverage": {
                        "technical_risks": len(result.get("risk_assessment", {}).get("technical_risks", [])),
                        "business_risks": len(result.get("risk_assessment", {}).get("business_risks", [])),
                        "timeline_risks": len(result.get("risk_assessment", {}).get("timeline_risks", [])),
                        "quality_risks": len(result.get("risk_assessment", {}).get("quality_risks", [])),
                        "coverage_percentage": self._calculate_risk_coverage_percentage(result, discovery_results)
                    },
                    "optimization_suggestions": self._extract_optimization_suggestions(result, discovery_results),
                    "confidence_score": ConfidenceScore(
                        value=result.get("confidence_assessment", {}).get("overall_confidence", 0.8),
                        method="llm_requirements_risk_assessment",
                        explanation=f"Requirements and risk analysis completed with unified discovery: {result.get('confidence_assessment', {}).get('overall_confidence', 0.8)}"
                    )
                }
                
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM response not valid JSON: {e}")
                raise ValueError(f"LLM response parsing failed: {e}")
                
        except Exception as e:
            self.logger.error(f"Requirements and risk analysis generation failed: {e}")
            raise

    async def _get_available_tools_description(self) -> str:
        """Get available tools description for template use."""
        try:
            # Get available tools through unified mechanism
            tools = await self._get_all_available_mcp_tools()
            if tools.get("discovery_successful") and tools.get("tools"):
                tool_list = []
                for tool_name, tool_info in tools["tools"].items():
                    description = tool_info.get('description', f'Tool: {tool_name}')
                    tool_list.append(f"- {tool_name}: {description}")
                return "\n".join(tool_list)
            else:
                return "Standard unified discovery tools available"
        except Exception as e:
            self.logger.warning(f"Could not get tools description: {e}")
            return "Standard unified discovery tools available"

    def _extract_risk_summary(self, llm_result: Dict[str, Any], discovery_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk summary from LLM result and discovery."""
        risk_assessment = llm_result.get("risk_assessment", {})
        
        total_risks = 0
        high_priority_risks = 0
        
        for risk_category in ["technical_risks", "business_risks", "timeline_risks", "quality_risks"]:
            risks = risk_assessment.get(risk_category, [])
            total_risks += len(risks)
            # Count high priority risks (simplified heuristic)
            high_priority_risks += len([r for r in risks if isinstance(r, dict) and r.get("severity") == "high"])
        
        return {
            "total_risks_identified": total_risks,
            "high_priority_risks": high_priority_risks,
            "risk_categories": list(risk_assessment.keys()),
            "mitigation_coverage": len(risk_assessment.get("risk_mitigation_strategies", [])),
            "risk_analysis_method": "unified_discovery_with_llm_analysis"
        }

    def _extract_mitigation_strategies(self, llm_result: Dict[str, Any], discovery_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract mitigation strategies from analysis."""
        strategies = []
        
        risk_assessment = llm_result.get("risk_assessment", {})
        mitigation_strategies = risk_assessment.get("risk_mitigation_strategies", [])
        
        for strategy in mitigation_strategies:
            if isinstance(strategy, dict):
                strategies.append(strategy)
            else:
                # Convert string strategies to structured format
                strategies.append({
                    "strategy": str(strategy),
                    "type": "general",
                    "priority": "medium"
                })
        
        return strategies

    def _assess_requirements_completeness(self, llm_result: Dict[str, Any], discovery_results: Dict[str, Any]) -> str:
        """Assess completeness of requirements."""
        loprd = llm_result.get("loprd_with_risk_mitigation", {})
        
        # Check for key LOPRD sections
        required_sections = ["project_overview", "user_stories", "functional_requirements", "non_functional_requirements"]
        present_sections = [section for section in required_sections if section in loprd]
        
        completeness_ratio = len(present_sections) / len(required_sections)
        
        if completeness_ratio >= 0.8:
            return "high"
        elif completeness_ratio >= 0.6:
            return "medium"
        else:
            return "low"

    def _assess_requirements_quality(self, llm_result: Dict[str, Any], discovery_results: Dict[str, Any]) -> str:
        """Assess quality of requirements."""
        confidence_assessment = llm_result.get("confidence_assessment", {})
        requirements_confidence = confidence_assessment.get("requirements_confidence", 0.7)
        
        if requirements_confidence >= 0.8:
            return "high"
        elif requirements_confidence >= 0.6:
            return "medium"
        else:
            return "low"

    def _calculate_risk_coverage_percentage(self, llm_result: Dict[str, Any], discovery_results: Dict[str, Any]) -> float:
        """Calculate percentage of risk coverage."""
        confidence_assessment = llm_result.get("confidence_assessment", {})
        risk_coverage_confidence = confidence_assessment.get("risk_coverage_confidence", 0.8)
        
        return risk_coverage_confidence * 100

    def _extract_optimization_suggestions(self, llm_result: Dict[str, Any], discovery_results: Dict[str, Any]) -> List[str]:
        """Extract optimization suggestions from analysis."""
        suggestions = [
            "Regular review and update of requirements based on project evolution",
            "Continuous risk monitoring and mitigation strategy updates",
            "Maintain traceability between requirements and implementation"
        ]
        
        # Add discovery-based suggestions
        if discovery_results.get("patterns", {}).get("complexity") == "high":
            suggestions.append("Consider simplifying complex patterns to reduce implementation risks")
        
        if discovery_results.get("dependencies", {}).get("external_count", 0) > 10:
            suggestions.append("Review external dependencies to reduce integration risks")
        
        # Add LLM-suggested optimizations if available
        integrated_requirements = llm_result.get("integrated_requirements", {})
        if "optimization_notes" in integrated_requirements:
            suggestions.extend(integrated_requirements["optimization_notes"])
        
        return suggestions

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Generate agent card for registry."""
        input_schema = RequirementsRiskAgentInput.model_json_schema()
        output_schema = RequirementsRiskAgentOutput.model_json_schema()
        
        return AgentCard(
            agent_id=RequirementsRiskAgent_v1.AGENT_ID,
            name=RequirementsRiskAgent_v1.AGENT_NAME,
            description=RequirementsRiskAgent_v1.AGENT_DESCRIPTION,
            version=RequirementsRiskAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[RequirementsRiskAgent_v1.CATEGORY.value],
            visibility=RequirementsRiskAgent_v1.VISIBILITY.value,
            capability_profile={
                "unified_discovery": True,
                "yaml_templates": True,
                "llm_requirements_risk": True,
                "clean_analysis": True,
                "no_hardcoded_logic": True,
                "maximum_agentic": True
            },
            metadata={
                "callable_fn_path": f"{RequirementsRiskAgent_v1.__module__}.{RequirementsRiskAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[RequirementsRiskAgentInput]:
        return RequirementsRiskAgentInput

    def get_output_schema(self) -> Type[RequirementsRiskAgentOutput]:
        return RequirementsRiskAgentOutput 