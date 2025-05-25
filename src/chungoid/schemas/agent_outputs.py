"""
Standardized Agent Output Schemas

Type-safe output schemas for all autonomous agents.
"""

from typing import Dict, Any, Optional, List, Literal, Union
from datetime import datetime
from pydantic import BaseModel, Field


class AgentOutput(BaseModel):
    """Standardized output for all agents."""
    success: bool = Field(..., description="Whether the agent execution was successful")
    data: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific output data")
    next_stage: Optional[str] = Field(None, description="Recommended next stage in the workflow")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about execution")
    
    # Execution tracking
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the output was generated")
    agent_id: Optional[str] = Field(None, description="ID of the agent that generated this output")
    
    # Protocol information
    protocol_used: Optional[str] = Field(None, description="Protocol used for execution")
    phases_completed: List[str] = Field(default_factory=list, description="Protocol phases completed")


class MasterPlannerReviewerOutput(AgentOutput):
    """Specific output for reviewer agents."""
    recommendation: Literal["RETRY", "SKIP", "MODIFY", "ESCALATE"] = Field(
        ..., description="Reviewer recommendation for the plan"
    )
    modifications: Optional[Dict[str, Any]] = Field(
        None, description="Specific modifications to apply if recommendation is MODIFY"
    )
    reasoning: str = Field(..., description="Detailed reasoning for the recommendation")
    confidence_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in the recommendation"
    )
    review_criteria_met: Dict[str, bool] = Field(
        default_factory=dict, description="Which review criteria were met"
    )


class RequirementsGatheringOutput(AgentOutput):
    """Output for requirements gathering agents."""
    requirements_document_id: Optional[str] = Field(
        None, description="ID of the document artifact containing the refined requirements"
    )
    requirements_summary: str = Field(..., description="Textual summary of gathered requirements")
    functional_requirements: List[str] = Field(
        default_factory=list, description="List of functional requirements"
    )
    technical_requirements: List[str] = Field(
        default_factory=list, description="List of technical requirements and constraints"
    )
    acceptance_criteria: List[str] = Field(
        default_factory=list, description="List of acceptance criteria for the project"
    )
    stakeholder_needs: Dict[str, List[str]] = Field(
        default_factory=dict, description="Requirements organized by stakeholder"
    )


class CodeGenerationOutput(AgentOutput):
    """Output for code generation agents."""
    generated_files: List[Dict[str, str]] = Field(
        default_factory=list, description="List of generated files with path and content"
    )
    test_files: List[Dict[str, str]] = Field(
        default_factory=list, description="List of generated test files"
    )
    documentation: Optional[str] = Field(None, description="Generated documentation")
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Code quality metrics"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Required dependencies"
    )
    build_instructions: Optional[str] = Field(
        None, description="Instructions for building/running the code"
    )


class FileManagementOutput(AgentOutput):
    """Output for file management agents."""
    files_created: List[str] = Field(default_factory=list, description="List of created files")
    files_modified: List[str] = Field(default_factory=list, description="List of modified files")
    files_deleted: List[str] = Field(default_factory=list, description="List of deleted files")
    backup_location: Optional[str] = Field(None, description="Location of backup files")
    operation_log: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed log of file operations"
    )
    rollback_available: bool = Field(default=False, description="Whether rollback is available")


class ArchitectureDesignOutput(AgentOutput):
    """Output for architecture design agents."""
    architecture_document: Optional[str] = Field(
        None, description="Generated architecture documentation"
    )
    component_specifications: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed component specifications"
    )
    integration_points: List[Dict[str, Any]] = Field(
        default_factory=list, description="System integration points"
    )
    technology_stack: List[str] = Field(
        default_factory=list, description="Recommended technology stack"
    )
    deployment_strategy: Optional[Dict[str, Any]] = Field(
        None, description="Deployment strategy and requirements"
    )
    scalability_considerations: List[str] = Field(
        default_factory=list, description="Scalability considerations and recommendations"
    )


class TestingOutput(AgentOutput):
    """Output for testing agents."""
    test_results: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed test execution results"
    )
    test_coverage: float = Field(default=0.0, ge=0.0, le=1.0, description="Test coverage percentage")
    passed_tests: int = Field(default=0, description="Number of passed tests")
    failed_tests: int = Field(default=0, description="Number of failed tests")
    test_report: Optional[str] = Field(None, description="Detailed test report")
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance test metrics"
    )


class SecurityAnalysisOutput(AgentOutput):
    """Output for security analysis agents."""
    vulnerabilities_found: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of identified vulnerabilities"
    )
    security_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall security score")
    recommendations: List[str] = Field(
        default_factory=list, description="Security improvement recommendations"
    )
    compliance_status: Dict[str, bool] = Field(
        default_factory=dict, description="Compliance with security standards"
    )
    risk_assessment: Dict[str, str] = Field(
        default_factory=dict, description="Risk assessment for identified issues"
    )


class DeploymentOutput(AgentOutput):
    """Output for deployment agents."""
    deployment_status: Literal["success", "failed", "partial"] = Field(
        ..., description="Overall deployment status"
    )
    deployed_components: List[str] = Field(
        default_factory=list, description="Successfully deployed components"
    )
    failed_components: List[str] = Field(
        default_factory=list, description="Components that failed to deploy"
    )
    deployment_url: Optional[str] = Field(None, description="URL of deployed application")
    rollback_instructions: Optional[str] = Field(
        None, description="Instructions for rolling back deployment"
    )
    monitoring_endpoints: List[str] = Field(
        default_factory=list, description="Endpoints for monitoring the deployment"
    )


class DocumentationOutput(AgentOutput):
    """Output for documentation agents."""
    documentation_files: List[Dict[str, str]] = Field(
        default_factory=list, description="Generated documentation files"
    )
    api_documentation: Optional[str] = Field(None, description="API documentation")
    user_guide: Optional[str] = Field(None, description="User guide documentation")
    developer_guide: Optional[str] = Field(None, description="Developer guide documentation")
    changelog: Optional[str] = Field(None, description="Generated changelog")
    documentation_coverage: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Documentation coverage percentage"
    )


# Union type for all possible agent outputs
AnyAgentOutput = Union[
    AgentOutput,
    MasterPlannerReviewerOutput,
    RequirementsGatheringOutput,
    CodeGenerationOutput,
    FileManagementOutput,
    ArchitectureDesignOutput,
    TestingOutput,
    SecurityAnalysisOutput,
    DeploymentOutput,
    DocumentationOutput
] 