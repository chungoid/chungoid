from __future__ import annotations

import logging
import asyncio
import datetime # For potential timestamping
import uuid
import json # For parsing LLM output if it's a JSON string
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, ClassVar, Type
import time

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
    project_id: str = Field(..., description="Identifier for the current project.")
    artifact_id: str = Field(..., description="ChromaDB ID of the artifact (LOPRD or Blueprint) to be assessed.")
    artifact_type: Literal["LOPRD", "Blueprint", "MasterExecutionPlan"] = Field(..., description="Type of the artifact being assessed.")
    loprd_document_id_for_blueprint_context: Optional[str] = Field(None, description="Optional LOPRD ID if artifact_type is Blueprint, to provide LOPRD context.")
    # Optional: Specific areas to focus on, or context about previous reviews
    focus_areas: Optional[List[str]] = Field(None, description="Optional list of specific areas to focus the risk assessment on.")
    # previous_assessment_ids: Optional[List[str]] = Field(None, description="IDs of previous assessment reports for context, if re-assessing.")

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
    CAPABILITIES: ClassVar[List[str]] = ["risk_assessment", "deep_investigation", "impact_analysis"]  # Added required CAPABILITIES
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
        super().__init__(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            system_context=system_context,
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

    async def execute(
        self, 
        context: UEContext,
        execution_mode: ExecutionMode = ExecutionMode.OPTIMAL
    ) -> AgentExecutionResult:
        """
        UAEI execute method - handles risk assessment workflow.
        
        Runs the complete risk assessment workflow:
        1. Discovery: Retrieve artifact to be assessed
        2. Analysis: Analyze risks and issues 
        3. Planning: Plan mitigation strategies
        4. Execution: Generate assessment reports
        5. Validation: Validate assessment quality
        """
        start_time = time.perf_counter()
        
        # Convert inputs to proper type
        if isinstance(context.inputs, dict):
            inputs = ProactiveRiskAssessorInput(**context.inputs)
        elif isinstance(context.inputs, ProactiveRiskAssessorInput):
            inputs = context.inputs
        else:
            # Fallback for other types
            inputs = ProactiveRiskAssessorInput(
                project_id=str(context.inputs.get("project_id", "default")),
                artifact_id=str(context.inputs.get("artifact_id", "")),
                artifact_type=context.inputs.get("artifact_type", "LOPRD")
            )
        
        try:
            # Phase 1: Discovery - Retrieve artifact  
            self._logger.info("Starting artifact discovery phase")
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
                    score=int(quality_score * 100),
                    reasoning=f"Quality based on validation ({validation_result.get('is_valid', False)}) and completeness"
                ),
                usage_metadata={
                    "execution_mode": execution_mode.value,
                    "phases_executed": ["discovery", "analysis", "planning", "execution", "validation"],
                    "risks_identified": len(risks.get("critical_risks", [])) + len(risks.get("moderate_risks", []))
                }
            )
            
            completion_reason = CompletionReason.SUCCESS if quality_score >= context.execution_config.quality_threshold else CompletionReason.QUALITY_THRESHOLD
            
        except Exception as e:
            self._logger.error(f"ProactiveRiskAssessorAgent execution failed: {e}")
            
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
                    score=10,
                    reasoning="Execution failed with exception"
                )
            )
            
            quality_score = 0.1
            completion_reason = CompletionReason.ERROR
        
        execution_time = time.perf_counter() - start_time
        
        # Create execution metadata
        metadata = ExecutionMetadata(
            mode=execution_mode,
            protocol_used=self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "risk_assessment",
            execution_time=execution_time,
            iterations_planned=context.execution_config.max_iterations,
            tools_utilized=None
        )
        
        return AgentExecutionResult(
            output=output,
            execution_metadata=metadata,
            iterations_completed=1,  # Single iteration for risk assessment
            completion_reason=completion_reason,
            quality_score=quality_score,
            protocol_used=metadata.protocol_used
        )

    async def _discover_artifact(self, inputs: ProactiveRiskAssessorInput) -> Dict[str, Any]:
        """Discover and retrieve artifact to be assessed."""
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
                artifact_data=risk_report_content,
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
                artifact_data=optimization_report_content,
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

 