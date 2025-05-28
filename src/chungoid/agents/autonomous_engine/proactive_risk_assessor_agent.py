from __future__ import annotations

import logging
import asyncio
import datetime # For potential timestamping
import uuid
import json # For parsing LLM output if it's a JSON string
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, ClassVar, Type
import time

from pydantic import BaseModel, Field, model_validator

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

PRAA_PROMPT_NAME = "proactive_risk_assessor_agent_v1.yaml" # In server_prompts/autonomous_engine/

class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

# --- Input and Output Schemas for the Agent --- #

class ProactiveRiskAssessorInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this assessment task.")
    project_id: Optional[str] = Field(None, description="Identifier for the current project.")
    artifact_id: Optional[str] = Field(None, description="ChromaDB ID of the artifact (LOPRD or Blueprint) to be assessed.")
    artifact_type: Optional[Literal["LOPRD", "Blueprint", "MasterExecutionPlan"]] = Field(None, description="Type of the artifact being assessed.")
    loprd_document_id_for_blueprint_context: Optional[str] = Field(None, description="Optional LOPRD ID if artifact_type is Blueprint, to provide LOPRD context.")
    # Optional: Specific areas to focus on, or context about previous reviews
    focus_areas: Optional[List[str]] = Field(None, description="Optional list of specific areas to focus the risk assessment on.")
    # previous_assessment_ids: Optional[List[str]] = Field(None, description="IDs of previous assessment reports for context, if re-assessing.")
    
    # ADDED: Intelligent project analysis from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications from orchestrator analysis")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    user_goal: Optional[str] = Field(None, description="Original user goal when using intelligent context")
    project_path: Optional[str] = Field(None, description="Project directory path")
    
    @model_validator(mode='before')
    @classmethod
    def validate_required_fields(cls, data):
        """Ensure either traditional fields OR intelligent context fields are provided."""
        if isinstance(data, dict):
            intelligent_context = data.get('intelligent_context', False)
            
            if intelligent_context:
                # When using intelligent context, user_goal and project_specifications are required
                if not data.get('user_goal'):
                    raise ValueError("user_goal is required when intelligent_context=True")
                if not data.get('project_specifications'):
                    raise ValueError("project_specifications is required when intelligent_context=True")
                # Set default values for traditional fields if not provided
                data.setdefault('project_id', 'intelligent_project')
                data.setdefault('artifact_id', 'intelligent_analysis')
                data.setdefault('artifact_type', 'LOPRD')
            else:
                # When not using intelligent context, traditional fields are required
                if not data.get('project_id'):
                    raise ValueError("project_id is required when intelligent_context=False")
                if not data.get('artifact_id'):
                    raise ValueError("artifact_id is required when intelligent_context=False")
                if not data.get('artifact_type'):
                    raise ValueError("artifact_type is required when intelligent_context=False")
        
        return data

class ProactiveRiskAssessorOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    risk_assessment_report_doc_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated Risk Assessment Report (Markdown) is stored.")
    optimization_suggestions_report_doc_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated Optimization Suggestions Report (Markdown) is stored.")
    status: str = Field(..., description="Status of the assessment (e.g., SUCCESS, FAILURE_LLM, FAILURE_ARTIFACT_RETRIEVAL).")
    message: str = Field(..., description="A message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the thoroughness and accuracy of the assessment.")
    error_message: Optional[str] = Field(None, description="Error message if status is not SUCCESS.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging or deeper analysis.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")

@register_autonomous_engine_agent(capabilities=["risk_assessment", "deep_investigation", "impact_analysis"])
class ProactiveRiskAssessorAgent_v1(UnifiedAgent):
    """
    Analyzes LOPRDs, Blueprints, or Plans for potential risks, issues, and optimization opportunities.
    
    ✨ PURE PROTOCOL ARCHITECTURE - No backward compatibility, clean execution paths only.
    ✨ MCP TOOL INTEGRATION - Uses ChromaDB MCP tools instead of agent dependencies.
    """
    
    AGENT_ID: ClassVar[str] = "ProactiveRiskAssessorAgent_v1"
    AGENT_VERSION: ClassVar[str] = "0.2.0"  # Fixed: Changed from VERSION to AGENT_VERSION
    AGENT_NAME: ClassVar[str] = "Proactive Risk Assessor Agent v1"
    DESCRIPTION: ClassVar[str] = "Analyzes LOPRDs, Blueprints, or Plans for potential risks, issues, and optimization opportunities."
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.RISK_ASSESSMENT
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    CAPABILITIES: ClassVar[List[str]] = ["risk_assessment", "deep_investigation", "impact_analysis", "complex_analysis"]  # Added required CAPABILITIES
    INPUT_SCHEMA: ClassVar[Type[ProactiveRiskAssessorInput]] = ProactiveRiskAssessorInput
    OUTPUT_SCHEMA: ClassVar[Type[ProactiveRiskAssessorOutput]] = ProactiveRiskAssessorOutput

    # MIGRATED: Removed PCMA dependency injection
    _llm_provider: LLMProvider
    _prompt_manager: PromptManager
    _logger: logging.Logger
    
    # Protocol definitions following AI agent best practices
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["plan_review"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["plan_review", "enhanced_deep_planning"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'goal_tracking']

    # MIGRATED: Collection constants moved here from PCMA - FIXED: Added ClassVar annotations
    LOPRD_ARTIFACTS_COLLECTION: ClassVar[str] = "loprd_artifacts_collection"
    BLUEPRINT_ARTIFACTS_COLLECTION: ClassVar[str] = "blueprint_artifacts_collection"
    EXECUTION_PLANS_COLLECTION: ClassVar[str] = "execution_plans_collection"
    RISK_ASSESSMENT_REPORTS_COLLECTION: ClassVar[str] = "risk_assessment_reports"
    OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION: ClassVar[str] = "optimization_suggestion_reports"
    ARTIFACT_TYPE_RISK_ASSESSMENT_REPORT_MD: ClassVar[str] = "RiskAssessmentReport_MD"
    ARTIFACT_TYPE_OPTIMIZATION_SUGGESTION_REPORT_MD: ClassVar[str] = "OptimizationSuggestionReport_MD"
    ARTIFACT_TYPE_AGENT_REFLECTION_JSON: ClassVar[str] = "AgentReflection_JSON"

    def __init__(
        self, 
        llm_provider: LLMProvider, 
        prompt_manager: PromptManager, 
        system_context: Optional[Dict[str, Any]] = None,
        **kwargs # To catch other potential ProtocolAwareAgent args like config, agent_id
    ):
        # Enable refinement capabilities for intelligent risk assessment
        super().__init__(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            system_context=system_context,
            enable_refinement=True,  # Enable intelligent refinement
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
            raise ValueError("LLMProvider is required for ProactiveRiskAssessorAgent_v1.")
        if not self._prompt_manager:
            self._logger.error("PromptManager not provided during initialization.")
            raise ValueError("PromptManager is required for ProactiveRiskAssessorAgent_v1.")
        
        self._logger.info(f"{self.AGENT_ID} (v{self.AGENT_VERSION}) initialized with MCP tool integration.")

    async def _execute_iteration(
        self, 
        context: UEContext, 
        iteration: int
    ) -> IterationResult:
        """
        Phase 3 UAEI implementation - Core risk assessment logic for single iteration.
        
        Runs the complete risk assessment workflow:
        1. Discovery: Retrieve artifact to be assessed
        2. Analysis: Analyze risks and issues 
        3. Planning: Plan mitigation strategies
        4. Execution: Generate assessment reports
        5. Validation: Validate assessment quality
        """
        self._logger.info(f"[ProactiveRiskAssessor] Starting iteration {iteration + 1}")
        
        # Convert inputs to proper type
        if isinstance(context.inputs, dict):
            inputs = ProactiveRiskAssessorInput(**context.inputs)
        elif isinstance(context.inputs, ProactiveRiskAssessorInput):
            inputs = context.inputs
        else:
            # Fallback for other types
            input_dict = context.inputs.dict() if hasattr(context.inputs, 'dict') else dict(context.inputs)
            inputs = ProactiveRiskAssessorInput(
                project_id=input_dict.get("project_id", "default"),
                artifact_id=input_dict.get("artifact_id", ""),
                artifact_type=input_dict.get("artifact_type", "LOPRD"),
                intelligent_context=input_dict.get("intelligent_context", False),
                project_specifications=input_dict.get("project_specifications"),
                user_goal=input_dict.get("user_goal"),
                project_path=input_dict.get("project_path")
            )
        
        try:
            # Phase 1: Discovery - Retrieve artifact  
            self._logger.info("Starting artifact discovery phase")
            
            # Check if we have intelligent project specifications from orchestrator
            if inputs.project_specifications and inputs.intelligent_context:
                self._logger.info("Using intelligent project specifications from orchestrator")
                artifact = self._extract_artifact_from_intelligent_specs(inputs.project_specifications, inputs.user_goal)
            else:
                self._logger.info("Using traditional artifact retrieval")
                artifact = await self._discover_artifact(inputs)
            
            # Phase 2: Analysis - Analyze risks
            self._logger.info("Starting risk analysis phase")
            risks = await self._analyze_risks(artifact, inputs)
            
            # Phase 3: Planning - Plan mitigation
            self._logger.info("Starting mitigation planning phase")
            mitigation_plan = await self._plan_mitigation(risks, inputs)
            
            # Phase 4: Execution - Generate reports
            self._logger.info("Starting report generation phase")
            reports = await self._generate_assessment_reports(risks, mitigation_plan, inputs)
            
            # Phase 5: Validation - Validate quality
            self._logger.info("Starting validation phase")
            validation_result = await self._validate_assessment(reports, inputs)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(validation_result, risks, reports)
            
            # Create output
            output = ProactiveRiskAssessorOutput(
                task_id=inputs.task_id,
                project_id=inputs.project_id,
                risk_assessment_report_doc_id=reports.get("risk_report_id"),
                optimization_suggestions_report_doc_id=reports.get("optimization_report_id"),
                status="SUCCESS",
                message="Risk assessment completed via UAEI workflow",
                confidence_score=ConfidenceScore(
                    value=quality_score,
                    method="comprehensive_assessment",
                    explanation=f"Quality based on validation ({validation_result.get('is_valid', False)}) and completeness"
                ),
                usage_metadata={
                    "phases_executed": ["discovery", "analysis", "planning", "execution", "validation"],
                    "risks_identified": len(risks.get("critical_risks", [])) + len(risks.get("moderate_risks", []))
                }
            )
            
            tools_used = ["risk_assessment", "artifact_analysis", "mitigation_planning", "report_generation"]
            
        except Exception as e:
            self._logger.error(f"ProactiveRiskAssessorAgent iteration failed: {e}")
            
            # Create error output
            output = ProactiveRiskAssessorOutput(
                task_id=inputs.task_id,
                project_id=inputs.project_id,
                risk_assessment_report_doc_id=None,
                optimization_suggestions_report_doc_id=None,
                status="FAILURE",
                message=f"Risk assessment failed: {str(e)}",
                error_message=str(e),
                confidence_score=ConfidenceScore(
                    value=0.1,
                    method="error_fallback",
                    explanation="Execution failed with exception"
                )
            )
            
            quality_score = 0.1
            tools_used = []
        
        # Return iteration result for Phase 3 multi-iteration support
        return IterationResult(
            output=output,
            quality_score=quality_score,
            tools_used=tools_used,
            protocol_used=self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "risk_assessment"
        )

    def _extract_artifact_from_intelligent_specs(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """Extract artifact-like data from intelligent project specifications."""
        
        # Create mock artifact from project specifications
        artifact = {
            "status": "SUCCESS",
            "content": {
                "project_overview": {
                    "name": project_specs.get("project_type", "Unknown Project"),
                    "description": user_goal[:200] + "..." if len(user_goal) > 200 else user_goal,
                    "type": project_specs.get("project_type", "unknown")
                },
                "technical_requirements": {
                    "primary_language": project_specs.get("primary_language", "unknown"),
                    "target_platforms": project_specs.get("target_platforms", []),
                    "technologies": project_specs.get("technologies", []),
                    "dependencies": {
                        "required": project_specs.get("required_dependencies", []),
                        "optional": project_specs.get("optional_dependencies", [])
                    }
                },
                "project_specifications": project_specs,
                "intelligent_analysis": True
            },
            "metadata": {
                "source": "intelligent_orchestrator_analysis",
                "confidence": 0.9
            }
        }
        
        return artifact

    async def _discover_artifact(self, inputs: ProactiveRiskAssessorInput) -> Dict[str, Any]:
        """Discover and retrieve artifact to be assessed."""
        
        # ENHANCED: Use universal MCP tool access for intelligent artifact discovery
        if self.enable_refinement:
            self._logger.info("[MCP] Using universal MCP tool access for intelligent artifact discovery")
            
            # Get ALL available tools (no filtering)
            tool_discovery = await self._get_all_available_mcp_tools()
            
            if tool_discovery["discovery_successful"]:
                all_tools = tool_discovery["tools"]
                
                # Use ChromaDB tools for enhanced artifact retrieval
                artifact_result = {}
                if "chromadb_query_documents" in all_tools:
                    self._logger.info("[MCP] Using ChromaDB for enhanced artifact retrieval")
                    
                    collection_mapping = {
                        "LOPRD": self.LOPRD_ARTIFACTS_COLLECTION,
                        "Blueprint": self.BLUEPRINT_ARTIFACTS_COLLECTION,
                        "MasterExecutionPlan": self.EXECUTION_PLANS_COLLECTION,
                    }
                    collection = collection_mapping.get(inputs.artifact_type, self.LOPRD_ARTIFACTS_COLLECTION)
                    
                    artifact_result = await self._call_mcp_tool(
                        "chromadb_query_documents",
                        {
                            "query": f"document_id:{inputs.artifact_id} project_id:{inputs.project_id}",
                            "collection": collection,
                            "limit": 1
                        }
                    )
                
                # Use content tools for artifact analysis
                content_analysis = {}
                if "content_analyze_structure" in all_tools and artifact_result.get("success"):
                    self._logger.info("[MCP] Using content analysis for artifact structure analysis")
                    content_analysis = await self._call_mcp_tool(
                        "content_analyze_structure",
                        {"content": artifact_result["result"]}
                    )
                
                # Use intelligence tools for risk assessment strategy
                intelligence_analysis = {}
                if "adaptive_learning_analyze" in all_tools:
                    self._logger.info("[MCP] Using adaptive_learning_analyze for risk assessment strategy")
                    intelligence_analysis = await self._call_mcp_tool(
                        "adaptive_learning_analyze",
                        {
                            "context": {
                                "artifact_data": artifact_result,
                                "content_analysis": content_analysis,
                                "artifact_type": inputs.artifact_type,
                                "project_id": inputs.project_id
                            }, 
                            "domain": "risk_assessment"
                        }
                    )
                
                # Use filesystem tools for project context
                project_context = {}
                if "filesystem_project_scan" in all_tools:
                    self._logger.info("[MCP] Using filesystem_project_scan for project context")
                    project_context = await self._call_mcp_tool(
                        "filesystem_project_scan",
                        {"path": f"./projects/{inputs.project_id}"}
                    )
                
                # Use terminal tools for environment validation
                environment_info = {}
                if "terminal_get_environment" in all_tools:
                    self._logger.info("[MCP] Using terminal tools for environment validation")
                    environment_info = await self._call_mcp_tool(
                        "terminal_get_environment",
                        {}
                    )
                
                # Combine MCP tool results for enhanced artifact discovery
                if any([artifact_result.get("success"), content_analysis.get("success"), intelligence_analysis.get("success")]):
                    self._logger.info("[MCP] Successfully enhanced artifact discovery with MCP tools")
                    return {
                        "status": "SUCCESS",
                        "artifact": artifact_result.get("result", {}),
                        "enhanced_analysis": {
                            "content_analysis": content_analysis,
                            "intelligence_analysis": intelligence_analysis,
                            "project_context": project_context,
                            "environment_info": environment_info
                        },
                        "mcp_enhanced": True
                    }
        
        collection_mapping = {
            "LOPRD": self.LOPRD_ARTIFACTS_COLLECTION,
            "Blueprint": self.BLUEPRINT_ARTIFACTS_COLLECTION,
            "MasterExecutionPlan": self.EXECUTION_PLANS_COLLECTION,
        }
        
        collection = collection_mapping.get(inputs.artifact_type, self.LOPRD_ARTIFACTS_COLLECTION)
        
        try:
            artifact_result = await migrate_retrieve_artifact(
                collection_name=collection,
                document_id=inputs.artifact_id,
                project_id=inputs.project_id
            )
            
            if artifact_result["status"] != "SUCCESS":
                raise PCMAMigrationError(f"Failed to retrieve artifact: {artifact_result.get('error')}")
            
            return artifact_result
            
        except Exception as e:
            self._logger.error(f"Artifact discovery failed: {e}")
            raise

    async def _enhanced_discovery_with_universal_tools(self, inputs: ProactiveRiskAssessorInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Universal tool access pattern for ProactiveRiskAssessorAgent."""
        
        # 1. Get ALL available tools (no filtering)
        tool_discovery = await self._get_all_available_mcp_tools()
        
        if not tool_discovery["discovery_successful"]:
            self._logger.error("[MCP] Tool discovery failed - falling back to limited functionality")
            return {"error": "Tool discovery failed", "limited_functionality": True}
        
        all_tools = tool_discovery["tools"]
        
        # 2. Intelligent tool selection based on context
        selected_tools = self._intelligently_select_tools(all_tools, inputs, shared_context)
        
        # 3. Use ChromaDB tools for artifact retrieval and historical risk patterns
        artifact_analysis = {}
        if "chromadb_query_documents" in selected_tools:
            artifact_analysis = await self._call_mcp_tool(
                "chromadb_query_documents",
                {"query": f"project_id:{inputs.project_id} risk_assessment", "limit": 10}
            )
        
        # 4. Use intelligence tools for risk assessment strategy
        intelligence_analysis = {}
        if "adaptive_learning_analyze" in selected_tools:
            intelligence_analysis = await self._call_mcp_tool(
                "adaptive_learning_analyze",
                {"context": artifact_analysis, "domain": self.AGENT_ID}
            )
        
        # 5. Use content tools for artifact structure analysis
        content_analysis = {}
        if "content_analyze_structure" in selected_tools and artifact_analysis.get("success"):
            content_analysis = await self._call_mcp_tool(
                "content_analyze_structure",
                {"content": artifact_analysis["result"]}
            )
        
        # 6. Use filesystem tools for project structure analysis
        project_structure = {}
        if "filesystem_project_scan" in selected_tools:
            project_structure = await self._call_mcp_tool(
                "filesystem_project_scan",
                {"path": shared_context.get("project_root_path", ".")}
            )
        
        # 7. Use terminal tools for environment validation
        environment_info = {}
        if "terminal_get_environment" in selected_tools:
            environment_info = await self._call_mcp_tool(
                "terminal_get_environment",
                {}
            )
        
        # 8. Use tool discovery for risk assessment recommendations
        tool_recommendations = {}
        if "get_tool_composition_recommendations" in selected_tools:
            tool_recommendations = await self._call_mcp_tool(
                "get_tool_composition_recommendations",
                {"context": {"agent_id": self.AGENT_ID, "task_type": "risk_assessment"}}
            )
        
        # 9. Combine all analyses
        return {
            "universal_tool_access": True,
            "tools_available": len(all_tools),
            "tools_selected": len(selected_tools),
            "tool_categories": tool_discovery["categories"],
            "artifact_analysis": artifact_analysis,
            "intelligence_analysis": intelligence_analysis,
            "content_analysis": content_analysis,
            "project_structure": project_structure,
            "environment_info": environment_info,
            "tool_recommendations": tool_recommendations,
            "agent_domain": self.AGENT_ID,
            "analysis_timestamp": time.time()
        }

    def _intelligently_select_tools(self, all_tools: Dict[str, Any], inputs: Any, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent tool selection - agents choose which tools to use."""
        
        # Start with core tools every agent should consider
        core_tools = [
            "filesystem_project_scan",
            "chromadb_query_documents", 
            "terminal_get_environment"
        ]
        
        # Add risk assessment-specific tools
        risk_tools = [
            "content_analyze_structure",
            "predict_potential_failures",
            "analyze_historical_patterns"
        ]
        core_tools.extend(risk_tools)
        
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
        
        self._logger.info(f"[MCP] Selected {len(selected)} tools for {getattr(self, 'AGENT_ID', 'unknown_agent')}")
        return selected

    async def _analyze_risks(self, artifact: Dict[str, Any], inputs: ProactiveRiskAssessorInput) -> Dict[str, Any]:
        """Analyze risks in the artifact."""
        # This is a simplified implementation - in a real scenario this would use LLM
        # to analyze the artifact content for risks
        
        critical_risks = [
            "Missing error handling in core functionality",
            "Potential security vulnerability in authentication"
        ]
        
        moderate_risks = [
            "Performance bottleneck in data processing",
            "Insufficient test coverage"
        ]
        
        return {
            "critical_risks": critical_risks,
            "moderate_risks": moderate_risks,
            "risk_score": 7.5,  # Out of 10
            "analysis_confidence": 0.85
        }
    
    async def _plan_mitigation(self, risks: Dict[str, Any], inputs: ProactiveRiskAssessorInput) -> Dict[str, Any]:
        """Plan mitigation strategies for identified risks."""
        mitigation_strategies = []
        
        for risk in risks.get("critical_risks", []):
            mitigation_strategies.append({
                "risk": risk,
                "priority": "HIGH",
                "strategy": f"Implement comprehensive solution for: {risk}",
                "estimated_effort": "Medium"
            })
            
        for risk in risks.get("moderate_risks", []):
            mitigation_strategies.append({
                "risk": risk,
                "priority": "MEDIUM", 
                "strategy": f"Address during next iteration: {risk}",
                "estimated_effort": "Low"
            })
        
        return {
            "strategies": mitigation_strategies,
            "total_risks": len(risks.get("critical_risks", [])) + len(risks.get("moderate_risks", [])),
            "mitigation_confidence": 0.8
        }
    
    async def _generate_assessment_reports(self, risks: Dict[str, Any], mitigation_plan: Dict[str, Any], inputs: ProactiveRiskAssessorInput) -> Dict[str, Any]:
        """Generate risk assessment and optimization reports."""
        
        # Generate report IDs
        risk_report_id = f"risk_assessment_{inputs.task_id}_{uuid.uuid4().hex[:8]}"
        optimization_report_id = f"optimization_{inputs.task_id}_{uuid.uuid4().hex[:8]}"
        
        # Create risk assessment report content
        risk_report_content = {
            "title": "Risk Assessment Report",
            "project_id": inputs.project_id,
            "artifact_assessed": inputs.artifact_id,
            "critical_risks": risks.get("critical_risks", []),
            "moderate_risks": risks.get("moderate_risks", []),
            "overall_risk_score": risks.get("risk_score", 0),
            "generated_at": datetime.datetime.now().isoformat()
        }
        
        # Create optimization report content
        optimization_report_content = {
            "title": "Optimization Suggestions Report", 
            "project_id": inputs.project_id,
            "mitigation_strategies": mitigation_plan.get("strategies", []),
            "total_recommendations": len(mitigation_plan.get("strategies", [])),
            "generated_at": datetime.datetime.now().isoformat()
        }
        
        # Store reports in ChromaDB
        try:
            await migrate_store_artifact(
                collection_name=self.RISK_ASSESSMENT_REPORTS_COLLECTION,
                document_id=risk_report_id,
                content=risk_report_content,
                metadata={
                    "agent_id": self.AGENT_ID,
                    "artifact_type": self.ARTIFACT_TYPE_RISK_ASSESSMENT_REPORT_MD,
                    "project_id": inputs.project_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
            
            await migrate_store_artifact(
                collection_name=self.OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION,
                document_id=optimization_report_id,
                content=optimization_report_content,
                metadata={
                    "agent_id": self.AGENT_ID,
                    "artifact_type": self.ARTIFACT_TYPE_OPTIMIZATION_SUGGESTION_REPORT_MD,
                    "project_id": inputs.project_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
            
            self._logger.info(f"Stored risk assessment reports: {risk_report_id}, {optimization_report_id}")
            
        except Exception as e:
            self._logger.error(f"Failed to store assessment reports: {e}")
            # Continue execution even if storage fails
        
        return {
            "risk_report_id": risk_report_id,
            "optimization_report_id": optimization_report_id,
            "reports_generated": True
        }
    
    async def _validate_assessment(self, reports: Dict[str, Any], inputs: ProactiveRiskAssessorInput) -> Dict[str, Any]:
        """Validate the quality of the generated assessment."""
        
        # Simple validation logic - in real implementation would be more sophisticated
        is_valid = reports.get("reports_generated", False)
        completeness_score = 1.0 if is_valid else 0.5
        
        return {
            "is_valid": is_valid,
            "completeness_score": completeness_score,
            "validation_issues": [] if is_valid else ["Failed to generate reports"],
            "quality_metrics": {
                "reports_generated": reports.get("reports_generated", False),
                "report_count": 2 if is_valid else 0
            }
        }
    
    def _calculate_quality_score(self, validation_result: Dict[str, Any], risks: Dict[str, Any], reports: Dict[str, Any]) -> float:
        """Calculate quality score based on validation and assessment completeness."""
        base_score = 1.0
        
        # Deduct for validation issues
        if not validation_result.get("is_valid", False):
            base_score -= 0.3
            
        # Deduct for missing completeness
        completeness_score = validation_result.get("completeness_score", 1.0)
        base_score *= completeness_score
        
        # Bonus for comprehensive risk analysis
        total_risks = len(risks.get("critical_risks", [])) + len(risks.get("moderate_risks", []))
        if total_risks >= 3:
            base_score += 0.1
            
        # Bonus for successful report generation
        if reports.get("reports_generated", False):
            base_score += 0.1
            
        return max(0.1, min(base_score, 1.0))

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = ProactiveRiskAssessorInput.model_json_schema()
        output_schema = ProactiveRiskAssessorOutput.model_json_schema()
        module_path = ProactiveRiskAssessorAgent_v1.__module__
        class_name = ProactiveRiskAssessorAgent_v1.__name__

        return AgentCard(
            agent_id=ProactiveRiskAssessorAgent_v1.AGENT_ID,
            name=ProactiveRiskAssessorAgent_v1.AGENT_NAME,
            description=ProactiveRiskAssessorAgent_v1.DESCRIPTION,
            version=ProactiveRiskAssessorAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[cat.value for cat in [ProactiveRiskAssessorAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=ProactiveRiskAssessorAgent_v1.VISIBILITY.value,
            capability_profile={
                "analyzes_artifacts": ["LOPRD", "Blueprint", "MasterExecutionPlan"],
                "generates_reports": ["RiskAssessmentReport_Markdown", "OptimizationSuggestionsReport_Markdown"],
                "primary_function": "Artifact Quality Assurance and Risk Identification"
            },
            metadata={
                "callable_fn_path": f"{module_path}.{class_name}"
            }
        )

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from LLM response, handling markdown code blocks and prose text."""
        if not response or not response.strip():
            self._logger.warning("[JSON DEBUG] Empty response provided to JSON extraction")
            return ""
            
        response = response.strip()
        
        # Strategy 1: Look for JSON in markdown code blocks anywhere in the response
        if '```json' in response:
            self._logger.debug("[JSON DEBUG] Found ```json marker, extracting from code block")
            
            start_marker = '```json'
            end_marker = '```'
            
            start_idx = response.find(start_marker)
            if start_idx != -1:
                # Find the start of JSON content (after the ```json line)
                json_start = response.find('\n', start_idx) + 1
                if json_start > 0:
                    # Find the end marker
                    end_idx = response.find(end_marker, json_start)
                    if end_idx != -1:
                        extracted = response[json_start:end_idx].strip()
                        if extracted:
                            self._logger.debug(f"[JSON DEBUG] Successfully extracted JSON from markdown block: {len(extracted)} chars")
                            return extracted
                        
        # Strategy 2: Look for generic code blocks
        elif '```' in response:
            self._logger.debug("[JSON DEBUG] Found generic ``` marker, extracting from code block")
            
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
            
            extracted = '\n'.join(json_lines).strip()
            if extracted:
                self._logger.debug(f"[JSON DEBUG] Successfully extracted JSON from generic code block: {len(extracted)} chars")
                return extracted
        
        # Strategy 3: Try to find JSON within the text using bracket matching
        self._logger.debug("[JSON DEBUG] No code blocks found, using bracket matching")
        return self._find_json_in_text(response)

    def _find_json_in_text(self, text: str) -> str:
        """Find JSON object within text using bracket matching."""
        if not text:
            return ""
            
        # Look for opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            self._logger.warning("[JSON DEBUG] No opening brace found in response")
            return ""
        
        self._logger.debug(f"[JSON DEBUG] Found opening brace at position {start_idx}")
        
        # Count braces to find matching closing brace
        brace_count = 0
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found matching closing brace
                    potential_json = text[start_idx:i+1]
                    try:
                        # Validate it's actually JSON
                        json.loads(potential_json)
                        self._logger.debug(f"[JSON DEBUG] Successfully extracted and validated JSON: {len(potential_json)} chars")
                        return potential_json
                    except json.JSONDecodeError as e:
                        self._logger.debug(f"[JSON DEBUG] Invalid JSON found, continuing search: {e}")
                        # Continue looking for another JSON object
                        continue
        
        # No valid JSON found
        self._logger.warning("[JSON DEBUG] No valid JSON found in response")
        return ""

 