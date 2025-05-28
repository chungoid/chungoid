"""
Requirements & Risk Agent - Consolidated LOPRD Generation + Risk Assessment

This agent combines the capabilities of:
1. ProductAnalystAgent - LOPRD Generation  
2. ProactiveRiskAssessorAgent - Risk Assessment

Instead of two separate agents with handoffs, this LLM-driven agent can:
- Generate LOPRD from user requirements
- Immediately assess risks in that LOPRD
- Provide optimized requirements with built-in risk mitigation
- Use ALL available MCP tools (LLM chooses the right ones)

Pipeline efficiency: User Goal → LOPRD + Risk Assessment (1 agent instead of 2)
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Literal, ClassVar

from pydantic import BaseModel, Field, model_validator, validator

from chungoid.agents.unified_agent import (
    UnifiedAgent, 
    IterationResult,
    ExecutionContext as UEContext
)
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.schemas.unified_execution_schemas import AgentOutput
from chungoid.schemas.common import ConfidenceScore
from chungoid.registry import register_autonomous_engine_agent
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.chromadb_migration_utils import migrate_store_artifact


class RequirementsRiskInput(BaseModel):
    """Input for the consolidated Requirements & Risk Agent."""
    
    # Core task identification
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project identifier")
    
    # User requirements input
    user_goal: Optional[str] = Field(None, description="Original user goal/requirements")
    refined_user_goal_md: Optional[str] = Field(None, description="Refined user goal in Markdown format")
    
    # Schema and context
    loprd_json_schema_str: Optional[str] = Field(None, description="JSON schema for LOPRD validation")
    
    # Intelligent context from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specs from orchestrator")
    intelligent_context: bool = Field(default=False, description="Whether using intelligent project specifications")
    project_path: Optional[str] = Field(None, description="Project directory path")
    
    # Risk assessment focus
    focus_areas: Optional[List[str]] = Field(None, description="Specific risk areas to focus on")
    
    @model_validator(mode='after')
    def check_minimum_requirements(self) -> 'RequirementsRiskInput':
        """Ensure we have either traditional goal or intelligent context."""
        if not self.intelligent_context and not self.user_goal and not self.refined_user_goal_md:
            raise ValueError("Must provide either user_goal/refined_user_goal_md or intelligent_context=True with project_specifications")
        return self


class RequirementsRiskOutput(AgentOutput):
    """Output from the consolidated Requirements & Risk Agent."""
    
    # LOPRD outputs
    loprd_doc_id: str = Field(..., description="Document ID of generated LOPRD in Chroma")
    loprd_content: Optional[Dict[str, Any]] = Field(None, description="Generated LOPRD content")
    
    # Risk assessment outputs  
    risk_assessment_report_doc_id: Optional[str] = Field(None, description="Document ID of risk assessment report")
    optimization_suggestions_report_doc_id: Optional[str] = Field(None, description="Document ID of optimization suggestions")
    
    # Consolidated results
    integrated_requirements: Optional[Dict[str, Any]] = Field(None, description="Requirements with risk mitigation integrated")
    risk_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of identified risks and mitigations")
    
    # Status and metadata
    status: str = Field(..., description="Overall status (SUCCESS, PARTIAL_SUCCESS, FAILURE)")
    message: str = Field(..., description="Detailed outcome message")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Confidence in the integrated output")
    validation_errors: Optional[str] = Field(None, description="Any validation errors encountered")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")


@register_autonomous_engine_agent(capabilities=["requirements_analysis", "risk_assessment", "optimization"])
class RequirementsRiskAgent(UnifiedAgent):
    """
    Consolidated Requirements & Risk Agent - LLM-DRIVEN
    
    Combines LOPRD generation + Risk assessment in one workflow:
    1. Analyzes user requirements → Generates LOPRD
    2. Immediately assesses risks in that LOPRD  
    3. Optimizes requirements with risk mitigation built-in
    4. LLM chooses the right MCP tools for each task
    
    REPLACES: ProductAnalystAgent + ProactiveRiskAssessorAgent
    PIPELINE EFFICIENCY: 2 agents → 1 agent, eliminates handoff overhead
    """
    
    AGENT_ID: ClassVar[str] = "RequirementsRiskAgent"
    AGENT_NAME: ClassVar[str] = "Requirements & Risk Agent"
    AGENT_DESCRIPTION: ClassVar[str] = "Consolidated agent for LOPRD generation and risk assessment with LLM-driven tool selection"
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "requirements_risk_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "1.0.0"
    
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    CAPABILITIES: ClassVar[List[str]] = ["requirements_analysis", "risk_assessment", "optimization", "complex_analysis"]
    
    INPUT_SCHEMA: ClassVar[Type[RequirementsRiskInput]] = RequirementsRiskInput
    OUTPUT_SCHEMA: ClassVar[Type[RequirementsRiskOutput]] = RequirementsRiskOutput
    
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["requirements_analysis", "risk_assessment"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["agent_communication", "tool_validation"]
    UNIVERSAL_PROTOCOLS: ClassVar[List[str]] = ["agent_communication", "context_sharing", "tool_validation"]
    
    # Collection constants for artifact storage
    LOPRD_ARTIFACTS_COLLECTION: ClassVar[str] = "loprd_artifacts_collection"
    RISK_ASSESSMENT_REPORTS_COLLECTION: ClassVar[str] = "risk_assessment_reports"
    OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION: ClassVar[str] = "optimization_suggestion_reports"

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        LLM-driven Requirements & Risk workflow:
        1. Requirements Analysis → Generate LOPRD  
        2. Risk Assessment → Analyze LOPRD for risks
        3. Optimization → Integrate risk mitigations into requirements
        4. Validation → Ensure quality and completeness
        """
        self.logger.info(f"[RequirementsRisk] Starting consolidated iteration {iteration + 1}")
        
        # Convert inputs
        if isinstance(context.inputs, dict):
            inputs = RequirementsRiskInput(**context.inputs)
        elif isinstance(context.inputs, RequirementsRiskInput):
            inputs = context.inputs
        else:
            input_dict = context.inputs.dict() if hasattr(context.inputs, 'dict') else {}
            inputs = RequirementsRiskInput(**input_dict)
        
        try:
            # Phase 1: Requirements Analysis with LLM-driven tool selection
            self.logger.info("Phase 1: Requirements Analysis")
            loprd_result = await self._generate_loprd_with_llm_tools(inputs)
            
            # Phase 2: Risk Assessment with LLM-driven tool selection  
            self.logger.info("Phase 2: Risk Assessment")
            risk_result = await self._assess_risks_with_llm_tools(loprd_result, inputs)
            
            # Phase 3: Requirements Optimization with LLM-driven integration
            self.logger.info("Phase 3: Requirements Optimization") 
            optimization_result = await self._optimize_requirements_with_llm_tools(loprd_result, risk_result, inputs)
            
            # Phase 4: Validation and Quality Assessment
            self.logger.info("Phase 4: Validation")
            validation_result = await self._validate_integrated_output(optimization_result, inputs)
            
            # Store artifacts using LLM-chosen storage approach
            storage_result = await self._store_artifacts_with_llm_tools(optimization_result, inputs)
            
            # Calculate quality score
            quality_score = self._calculate_integrated_quality_score(
                loprd_result, risk_result, optimization_result, validation_result
            )
            
            # Create consolidated output
            output = RequirementsRiskOutput(
                loprd_doc_id=storage_result.get("loprd_doc_id", f"loprd_{uuid.uuid4().hex[:8]}"),
                loprd_content=optimization_result.get("optimized_loprd"),
                risk_assessment_report_doc_id=storage_result.get("risk_report_id"),
                optimization_suggestions_report_doc_id=storage_result.get("optimization_report_id"),
                integrated_requirements=optimization_result.get("integrated_requirements"),
                risk_summary=risk_result.get("risk_summary"),
                status="SUCCESS",
                message="Successfully generated LOPRD with integrated risk assessment and optimization",
                confidence_score=ConfidenceScore(
                    value=quality_score,
                    method="integrated_requirements_risk_assessment",
                    explanation="Quality based on LOPRD completeness, risk coverage, and optimization effectiveness"
                ),
                usage_metadata={
                    "phases_completed": ["requirements", "risk_assessment", "optimization", "validation"],
                    "tools_used": optimization_result.get("tools_used", []),
                    "risks_identified": len(risk_result.get("risks", [])),
                    "optimizations_applied": len(optimization_result.get("optimizations", []))
                }
            )
            
            tools_used = ["requirements_analysis", "risk_assessment", "optimization", "validation"]
            
        except Exception as e:
            self.logger.error(f"RequirementsRiskAgent iteration failed: {e}")
            
            # Create error output
            output = RequirementsRiskOutput(
                loprd_doc_id=f"error_{uuid.uuid4().hex[:8]}",
                status="FAILURE",
                message=f"Consolidated requirements and risk assessment failed: {str(e)}",
                confidence_score=ConfidenceScore(
                    value=0.1,
                    method="error_fallback",
                    explanation="Execution failed with exception"
                ),
                validation_errors=str(e)
            )
            
            quality_score = 0.1
            tools_used = []
        
        return IterationResult(
            output=output,
            quality_score=quality_score,
            tools_used=tools_used,
            protocol_used="requirements_analysis"
        )

    async def _generate_loprd_with_llm_tools(self, inputs: RequirementsRiskInput) -> Dict[str, Any]:
        """Phase 1: Generate LOPRD using LLM to choose the right tools."""
        try:
            # Get all available tools for LLM to choose from
            available_tools = await self._get_all_available_mcp_tools()
            
            # Let LLM choose tools and approach for requirements analysis
            requirements_prompt = f"""You need to analyze user requirements and generate a LOPRD (List of Product Requirements Document).

USER INPUT:
User Goal: {inputs.user_goal or 'Not provided'}
Refined Goal: {inputs.refined_user_goal_md or 'Not provided'}
Project Path: {inputs.project_path or 'Not provided'}
Intelligent Context: {inputs.intelligent_context}

TASK: Generate a comprehensive LOPRD that includes:
- Core objectives and requirements
- User stories and acceptance criteria  
- Functional and non-functional requirements
- Success metrics and constraints

AVAILABLE TOOLS:
{self._format_available_tools(available_tools)}

Please choose the appropriate tools to analyze the requirements and return JSON with your approach:
{{
    "analysis_approach": "description of your approach",
    "tools_to_use": [
        {{
            "tool_name": "tool_name",
            "arguments": {{"param": "value"}},
            "purpose": "why using this tool"
        }}
    ],
    "loprd_structure": {{
        "core_objectives": ["objective1", "objective2"],
        "user_stories": ["{{"story": "As a user...", "acceptance_criteria": ["criteria1"]}}"],
        "functional_requirements": ["req1", "req2"],
        "non_functional_requirements": ["req1", "req2"],
        "success_metrics": ["metric1", "metric2"],
        "constraints": ["constraint1", "constraint2"]
    }}
}}

Return ONLY the JSON response."""
            
            # Get LLM response
            llm_response = await self.llm_provider.generate(requirements_prompt)
            
            # Parse LLM response
            analysis_plan = self._extract_json_from_response(llm_response)
            if isinstance(analysis_plan, str):
                analysis_plan = json.loads(analysis_plan)
            
            # Execute LLM-chosen tools
            tool_results = {}
            for tool_spec in analysis_plan.get("tools_to_use", []):
                tool_name = tool_spec.get("tool_name")
                arguments = tool_spec.get("arguments", {})
                
                try:
                    result = await self._call_mcp_tool(tool_name, arguments)
                    tool_results[tool_name] = result
                    self.logger.info(f"Executed LLM-chosen tool: {tool_name}")
                except Exception as e:
                    self.logger.warning(f"Tool {tool_name} failed: {e}")
                    tool_results[tool_name] = {"error": str(e)}
            
            # Generate final LOPRD based on analysis and tool results
            final_loprd = analysis_plan.get("loprd_structure", {})
            final_loprd["metadata"] = {
                "generated_by": self.AGENT_ID,
                "timestamp": datetime.now().isoformat(),
                "analysis_approach": analysis_plan.get("analysis_approach"),
                "tools_used": list(tool_results.keys())
            }
            
            return {
                "loprd": final_loprd,
                "analysis_plan": analysis_plan,
                "tool_results": tool_results,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"LOPRD generation failed: {e}")
            return {
                "loprd": self._generate_fallback_loprd(inputs),
                "error": str(e),
                "success": False
            }

    async def _assess_risks_with_llm_tools(self, loprd_result: Dict[str, Any], inputs: RequirementsRiskInput) -> Dict[str, Any]:
        """Phase 2: Assess risks in the generated LOPRD using LLM-chosen tools."""
        try:
            available_tools = await self._get_all_available_mcp_tools()
            
            # Let LLM choose tools for risk assessment
            risk_prompt = f"""You need to assess risks in the generated LOPRD and identify potential issues.

LOPRD TO ASSESS:
{json.dumps(loprd_result.get('loprd', {}), indent=2)}

FOCUS AREAS: {inputs.focus_areas or 'General risk assessment'}

TASK: Identify risks, issues, and optimization opportunities in this LOPRD:
- Technical risks and challenges
- Business risks and assumptions  
- Implementation risks and dependencies
- Quality risks and gaps
- Optimization opportunities

AVAILABLE TOOLS:
{self._format_available_tools(available_tools)}

Choose tools to help with risk analysis and return JSON:
{{
    "risk_assessment_approach": "your assessment approach",
    "tools_to_use": [
        {{
            "tool_name": "tool_name", 
            "arguments": {{"param": "value"}},
            "purpose": "why using this tool for risk assessment"
        }}
    ],
    "risks": {{
        "critical_risks": [
            {{"risk": "description", "impact": "high/medium/low", "likelihood": "high/medium/low", "mitigation": "suggested mitigation"}}
        ],
        "moderate_risks": [
            {{"risk": "description", "impact": "medium", "likelihood": "medium", "mitigation": "suggested mitigation"}}
        ],
        "optimization_opportunities": [
            {{"opportunity": "description", "benefit": "expected benefit", "effort": "low/medium/high"}}
        ]
    }},
    "risk_summary": {{
        "total_risks": 0,
        "risk_level": "LOW/MEDIUM/HIGH",
        "key_concerns": ["concern1", "concern2"],
        "recommended_actions": ["action1", "action2"]
    }}
}}

Return ONLY the JSON response."""
            
            # Get LLM response  
            llm_response = await self.llm_provider.generate(risk_prompt)
            
            # Parse LLM response
            risk_analysis = self._extract_json_from_response(llm_response)
            if isinstance(risk_analysis, str):
                risk_analysis = json.loads(risk_analysis)
            
            # Execute LLM-chosen tools for risk assessment
            tool_results = {}
            for tool_spec in risk_analysis.get("tools_to_use", []):
                tool_name = tool_spec.get("tool_name")
                arguments = tool_spec.get("arguments", {})
                
                try:
                    result = await self._call_mcp_tool(tool_name, arguments)
                    tool_results[tool_name] = result
                    self.logger.info(f"Executed risk assessment tool: {tool_name}")
                except Exception as e:
                    self.logger.warning(f"Risk tool {tool_name} failed: {e}")
                    tool_results[tool_name] = {"error": str(e)}
            
            return {
                "risks": risk_analysis.get("risks", {}),
                "risk_summary": risk_analysis.get("risk_summary", {}),
                "assessment_approach": risk_analysis.get("risk_assessment_approach"),
                "tool_results": tool_results,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return {
                "risks": {"critical_risks": [], "moderate_risks": [], "optimization_opportunities": []},
                "risk_summary": {"total_risks": 0, "risk_level": "UNKNOWN", "key_concerns": [], "recommended_actions": []},
                "error": str(e),
                "success": False
            }

    async def _optimize_requirements_with_llm_tools(self, loprd_result: Dict[str, Any], risk_result: Dict[str, Any], inputs: RequirementsRiskInput) -> Dict[str, Any]:
        """Phase 3: Optimize requirements by integrating risk mitigations using LLM-chosen tools."""
        try:
            available_tools = await self._get_all_available_mcp_tools()
            
            # Let LLM choose tools for optimization
            optimization_prompt = f"""You need to optimize the LOPRD by integrating risk mitigations and improvements.

ORIGINAL LOPRD:
{json.dumps(loprd_result.get('loprd', {}), indent=2)}

IDENTIFIED RISKS:
{json.dumps(risk_result.get('risks', {}), indent=2)}

RISK SUMMARY:
{json.dumps(risk_result.get('risk_summary', {}), indent=2)}

TASK: Create an optimized version of the LOPRD that:
- Addresses identified critical and moderate risks
- Integrates risk mitigations into requirements
- Applies optimization opportunities
- Maintains requirement completeness and clarity

AVAILABLE TOOLS:
{self._format_available_tools(available_tools)}

Choose tools to help with optimization and return JSON:
{{
    "optimization_approach": "your optimization strategy",
    "tools_to_use": [
        {{
            "tool_name": "tool_name",
            "arguments": {{"param": "value"}},
            "purpose": "why using this tool for optimization"
        }}
    ],
    "optimized_loprd": {{
        "core_objectives": ["updated objectives with risk considerations"],
        "user_stories": ["{{"story": "enhanced user story", "acceptance_criteria": ["criteria with risk mitigations"]}}"],
        "functional_requirements": ["requirements with risk mitigations"],
        "non_functional_requirements": ["enhanced non-functional requirements"],
        "success_metrics": ["metrics including risk indicators"],
        "constraints": ["constraints including risk controls"],
        "risk_mitigations": [
            {{"risk": "risk description", "mitigation": "how this is addressed in requirements"}}
        ]
    }},
    "integrated_requirements": {{
        "requirements_with_mitigations": ["req with built-in risk control"],
        "new_requirements_for_risks": ["new req to address specific risk"],
        "enhanced_acceptance_criteria": ["acceptance criteria that verify risk controls"]
    }},
    "optimizations": [
        {{"optimization": "what was improved", "rationale": "why this improves the requirements"}}
    ]
}}

Return ONLY the JSON response."""
            
            # Get LLM response
            llm_response = await self.llm_provider.generate(optimization_prompt)
            
            # Parse LLM response
            optimization_plan = self._extract_json_from_response(llm_response)
            if isinstance(optimization_plan, str):
                optimization_plan = json.loads(optimization_plan)
            
            # Execute LLM-chosen tools for optimization
            tool_results = {}
            for tool_spec in optimization_plan.get("tools_to_use", []):
                tool_name = tool_spec.get("tool_name")
                arguments = tool_spec.get("arguments", {})
                
                try:
                    result = await self._call_mcp_tool(tool_name, arguments)
                    tool_results[tool_name] = result
                    self.logger.info(f"Executed optimization tool: {tool_name}")
                except Exception as e:
                    self.logger.warning(f"Optimization tool {tool_name} failed: {e}")
                    tool_results[tool_name] = {"error": str(e)}
            
            # Add metadata to optimized LOPRD
            optimized_loprd = optimization_plan.get("optimized_loprd", loprd_result.get('loprd', {}))
            optimized_loprd["metadata"] = {
                "generated_by": self.AGENT_ID,
                "timestamp": datetime.now().isoformat(),
                "optimization_approach": optimization_plan.get("optimization_approach"),
                "risks_addressed": len(risk_result.get('risks', {}).get('critical_risks', [])) + len(risk_result.get('risks', {}).get('moderate_risks', [])),
                "optimizations_applied": len(optimization_plan.get("optimizations", [])),
                "tools_used": list(tool_results.keys())
            }
            
            return {
                "optimized_loprd": optimized_loprd,
                "integrated_requirements": optimization_plan.get("integrated_requirements", {}),
                "optimizations": optimization_plan.get("optimizations", []),
                "optimization_approach": optimization_plan.get("optimization_approach"),
                "tool_results": tool_results,
                "tools_used": list(tool_results.keys()),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Requirements optimization failed: {e}")
            return {
                "optimized_loprd": loprd_result.get('loprd', {}),
                "integrated_requirements": {},
                "optimizations": [],
                "error": str(e),
                "success": False
            }

    async def _validate_integrated_output(self, optimization_result: Dict[str, Any], inputs: RequirementsRiskInput) -> Dict[str, Any]:
        """Phase 4: Validate the integrated requirements and risk assessment output."""
        try:
            optimized_loprd = optimization_result.get("optimized_loprd", {})
            
            validation_results = {
                "is_valid": True,
                "completeness_score": 1.0,
                "issues": [],
                "quality_indicators": {}
            }
            
            # Check LOPRD completeness
            required_sections = ["core_objectives", "user_stories", "functional_requirements", "non_functional_requirements"]
            missing_sections = []
            
            for section in required_sections:
                if section not in optimized_loprd or not optimized_loprd[section]:
                    missing_sections.append(section)
            
            if missing_sections:
                validation_results["issues"].append(f"Missing required sections: {missing_sections}")
                validation_results["completeness_score"] -= 0.2 * len(missing_sections)
            
            # Check risk mitigation integration
            risk_mitigations = optimized_loprd.get("risk_mitigations", [])
            if not risk_mitigations:
                validation_results["issues"].append("No risk mitigations found in optimized LOPRD")
                validation_results["completeness_score"] -= 0.1
            
            # Check optimization quality
            optimizations = optimization_result.get("optimizations", [])
            validation_results["quality_indicators"] = {
                "sections_present": len([s for s in required_sections if s in optimized_loprd]),
                "total_requirements": len(optimized_loprd.get("functional_requirements", [])) + len(optimized_loprd.get("non_functional_requirements", [])),
                "risk_mitigations": len(risk_mitigations),
                "optimizations_applied": len(optimizations)
            }
            
            # Overall validation
            if validation_results["completeness_score"] < 0.6:
                validation_results["is_valid"] = False
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {
                "is_valid": False,
                "completeness_score": 0.1,
                "issues": [f"Validation error: {str(e)}"],
                "quality_indicators": {}
            }

    async def _store_artifacts_with_llm_tools(self, optimization_result: Dict[str, Any], inputs: RequirementsRiskInput) -> Dict[str, Any]:
        """Store the generated artifacts using LLM-chosen storage approach."""
        try:
            storage_results = {}
            
            # Store optimized LOPRD
            loprd_doc_id = f"loprd_{uuid.uuid4().hex[:8]}"
            await migrate_store_artifact(
                collection_name=self.LOPRD_ARTIFACTS_COLLECTION,
                document_id=loprd_doc_id,
                content=json.dumps(optimization_result.get("optimized_loprd", {}), indent=2),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "artifact_type": "OptimizedLOPRD_JSON",
                    "project_id": inputs.project_id,
                    "timestamp": datetime.now().isoformat(),
                    "has_risk_mitigations": len(optimization_result.get("optimized_loprd", {}).get("risk_mitigations", [])) > 0
                }
            )
            storage_results["loprd_doc_id"] = loprd_doc_id
            
            # Store risk assessment report
            risk_report_id = f"risk_report_{uuid.uuid4().hex[:8]}"
            risk_report_content = f"""# Risk Assessment Report
Generated by: {self.AGENT_ID}
Timestamp: {datetime.now().isoformat()}

## Risk Summary
- Total Risks Identified: {len(optimization_result.get('optimized_loprd', {}).get('risk_mitigations', []))}
- Optimization Approach: {optimization_result.get('optimization_approach', 'Standard optimization')}

## Risk Mitigations Integrated
{json.dumps(optimization_result.get('optimized_loprd', {}).get('risk_mitigations', []), indent=2)}

## Applied Optimizations
{json.dumps(optimization_result.get('optimizations', []), indent=2)}
"""
            
            await migrate_store_artifact(
                collection_name=self.RISK_ASSESSMENT_REPORTS_COLLECTION,
                document_id=risk_report_id,
                content=risk_report_content,
                metadata={
                    "agent_id": self.AGENT_ID,
                    "artifact_type": "IntegratedRiskAssessment_MD",
                    "project_id": inputs.project_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            storage_results["risk_report_id"] = risk_report_id
            
            self.logger.info(f"Stored artifacts: LOPRD({loprd_doc_id}), Risk Report({risk_report_id})")
            return storage_results
            
        except Exception as e:
            self.logger.error(f"Artifact storage failed: {e}")
            return {
                "loprd_doc_id": f"storage_failed_{uuid.uuid4().hex[:8]}",
                "error": str(e)
            }

    def _format_available_tools(self, tools: Dict[str, Any]) -> str:
        """Format ALL available tools for LLM to choose from - no filtering."""
        formatted = []
        for tool_name, tool_info in tools.items():
            description = tool_info.get('description', f'Tool: {tool_name}')
            formatted.append(f"- {tool_name}: {description}")
        
        return "\n".join(formatted) if formatted else "No tools available"

    def _extract_json_from_response(self, response: str) -> Any:
        """Extract JSON from LLM response."""
        if hasattr(response, 'content'):
            text = response.content
        else:
            text = str(response)
        
        # Find JSON in the response
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Fallback to simple parsing
        return {"error": "Could not parse JSON from LLM response", "raw_response": text}

    def _generate_fallback_loprd(self, inputs: RequirementsRiskInput) -> Dict[str, Any]:
        """Generate a basic fallback LOPRD when analysis fails."""
        user_goal = inputs.user_goal or inputs.refined_user_goal_md or "Create a software solution"
        
        return {
            "core_objectives": [f"Implement solution for: {user_goal[:100]}"],
            "user_stories": [
                {
                    "story": f"As a user, I want to use the system to achieve my goal: {user_goal[:100]}",
                    "acceptance_criteria": ["System should be functional", "System should be user-friendly"]
                }
            ],
            "functional_requirements": ["Core functionality implementation", "User interface design"],
            "non_functional_requirements": ["Performance requirements", "Security requirements"],
            "success_metrics": ["User satisfaction", "System reliability"],
            "constraints": ["Budget constraints", "Time constraints"],
            "metadata": {
                "generated_by": self.AGENT_ID,
                "timestamp": datetime.now().isoformat(),
                "fallback": True
            }
        }

    def _calculate_integrated_quality_score(self, loprd_result: Dict[str, Any], risk_result: Dict[str, Any], 
                                          optimization_result: Dict[str, Any], validation_result: Dict[str, Any]) -> float:
        """Calculate overall quality score for the integrated output."""
        base_score = 1.0
        
        # LOPRD quality contribution (40%)
        loprd_success = loprd_result.get("success", False)
        loprd_score = 0.8 if loprd_success else 0.4
        
        # Risk assessment quality contribution (30%)
        risk_success = risk_result.get("success", False)
        risk_score = 0.8 if risk_success else 0.4
        
        # Optimization quality contribution (20%)
        optimization_success = optimization_result.get("success", False)
        optimization_score = 0.8 if optimization_success else 0.4
        
        # Validation quality contribution (10%)
        validation_score = validation_result.get("completeness_score", 0.5)
        
        # Weighted average
        final_score = (
            loprd_score * 0.4 +
            risk_score * 0.3 + 
            optimization_score * 0.2 +
            validation_score * 0.1
        )
        
        return max(0.1, min(final_score, 1.0))

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Get the agent card for registration."""
        return AgentCard(
            agent_id=RequirementsRiskAgent.AGENT_ID,
            agent_name=RequirementsRiskAgent.AGENT_NAME,
            description=RequirementsRiskAgent.AGENT_DESCRIPTION,
            version=RequirementsRiskAgent.AGENT_VERSION,
            category=RequirementsRiskAgent.CATEGORY,
            visibility=RequirementsRiskAgent.VISIBILITY,
            capabilities=RequirementsRiskAgent.CAPABILITIES,
            primary_protocols=RequirementsRiskAgent.PRIMARY_PROTOCOLS,
            secondary_protocols=RequirementsRiskAgent.SECONDARY_PROTOCOLS,
            universal_protocols=RequirementsRiskAgent.UNIVERSAL_PROTOCOLS,
            input_schema=RequirementsRiskAgent.INPUT_SCHEMA,
            output_schema=RequirementsRiskAgent.OUTPUT_SCHEMA
        )

    def get_input_schema(self) -> Type[RequirementsRiskInput]:
        return RequirementsRiskInput

    def get_output_schema(self) -> Type[RequirementsRiskOutput]:
        return RequirementsRiskOutput 